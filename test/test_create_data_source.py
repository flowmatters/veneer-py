"""Unit tests for Veneer.create_data_source: the name a new data source gets.

A fake HTTP session stands in for Veneer and the Source behind it, so the real
client code runs end to end: the POST to /dataSources, the server-side scripts
(executed in CPython against a fake data manager, with the .NET modules they import
stood in for) and the response handling.

The fake names data sources the way Veneer and Source do. Veneer names a new data
source as asked - after the filename, for one loaded from a file - and replaces one
of exactly that name, moving its usages across. With its interface open (gui=True),
Source then renames each data source it shows with DataManager.RestrictName.
"""
import json
import logging
import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from veneer import Veneer

PATH = 'C:\\models\\inputs\\fire_tss.csv'


def restrict_name(name):
    """Source's DataManager.RestrictName."""
    for c in '.@>,\\':
        name = name.replace(c, '_')
    return name[:200].strip(' ')


class Groups(list):
    def FirstOrDefault(self, predicate):
        return next((g for g in self if predicate(g)), None)

    def LastOrDefault(self, predicate):
        return next((g for g in reversed(self) if predicate(g)), None)


class Usages(list):
    """A .NET collection of a column's usages (its assignments)."""
    def Add(self, item):
        self.append(item)

    def Clear(self):
        self.clear()


class Detail:
    def __init__(self, name, usages=()):
        self.Name = name
        self.Usages = Usages(usages)
        self.TimeSeries = SimpleNamespace(units=None)


class Group:
    def __init__(self, name, columns=(), filename=None, reload_on_run=False):
        self.Name = name
        self.DataDetails = [Detail(c) for c in columns]
        self.filename = filename
        self.reload_on_run = reload_on_run
        # What set_data_source_units walks: one input set item holding the series.
        data = [SimpleNamespace(Data=SimpleNamespace(TimeSeries=d.TimeSeries))
                for d in self.DataDetails]
        self.InputSetItems = [SimpleNamespace(DataSource=SimpleNamespace(Data=data))]

    def units(self):
        return [d.TimeSeries.units for d in self.DataDetails]

    def usages(self):
        return {d.Name: d.Usages for d in self.DataDetails if d.Usages}


class DataManager:
    def __init__(self):
        self.DataGroups = Groups()
        self.refreshed = 0

    def RemoveGroup(self, group):
        self.DataGroups.remove(group)

    def Refresh(self):
        self.refreshed += 1


class Response:
    def __init__(self, status_code, body=None, url=''):
        self.status_code = status_code
        self.text = '' if body is None else json.dumps(body)
        self.url = url
        self.reason = ''
        self.headers = {}
        self.request = None


def veneer_value(value):
    """A script result, as Veneer returns it."""
    if value is None:
        return None
    if isinstance(value, str):
        return {'__type': 'StringResponse', 'Value': value}
    if isinstance(value, (list, tuple)):
        return {'__type': 'ListResponse', 'Value': [veneer_value(v) for v in value]}
    return {'__type': 'IntegerResponse', 'Value': value}


class FakeSource:
    """Veneer's HTTP interface, over a fake data manager."""

    def __init__(self, gui=True, scripts=True, files=None, rule=restrict_name):
        self.gui = gui
        self.scripts = scripts
        self.files = files or {PATH: ['A', 'B']}
        self.rule = rule
        self.dm = DataManager()
        self.posted = []

    @property
    def names(self):
        return [g.Name for g in self.dm.DataGroups]

    def group(self, name):
        [g] = [g for g in self.dm.DataGroups if g.Name == name]
        return g

    # requests.Session's interface, as Veneer uses it
    def request(self, method, url, data=None, headers=None):
        path = url.split(':9876', 1)[1]
        body = json.loads(data) if data else None
        if method == 'POST' and path == '/dataSources':
            return self._create(body)
        if method == 'POST' and path == '/ironpython':
            return self._run(body['Script'])
        raise AssertionError('unexpected request %s %s' % (method, path))

    def get(self, url):
        path = url.split(':9876', 1)[1]
        assert path == '/dataSources', path
        return Response(200, [{'Name': g.Name} for g in self.dm.DataGroups])

    def _create(self, body):
        self.posted.append(body)
        name = body['Name']
        [item] = body['Items']
        if item.get('DetailsAsCSV') is None:
            columns, filename = self.files[name], name
        else:
            columns, filename = item['DetailsAsCSV'].splitlines()[0].split(',')[1:], None
        new = Group(name, columns, filename, item['ReloadOnRun'])
        existing = [g for g in self.dm.DataGroups if g.Name == name]
        if existing:
            # SourceService.CreateDataSource -> ReplaceInScenario
            [old] = existing
            for detail in old.DataDetails:
                [dest] = [d for d in new.DataDetails if d.Name == detail.Name]
                dest.Usages.extend(detail.Usages)
            self.dm.DataGroups.append(new)
            self.dm.RemoveGroup(old)
        else:
            if self.gui:
                new.Name = self.rule(name)
            self.dm.DataGroups.append(new)
        return Response(200)

    def _run(self, script):
        if not self.scripts:
            return Response(403)
        scope = {'scenario': SimpleNamespace(Network=SimpleNamespace(DataManager=self.dm))}
        try:
            exec(script, scope)
        except Exception as e:
            return Response(200, {'Exception': str(e), 'Response': None, 'StandardOut': ''})
        return Response(200, {'Exception': None, 'Response': veneer_value(scope.get('result')),
                              'StandardOut': ''})


@pytest.fixture
def dotnet(monkeypatch):
    """Stand-ins for the .NET modules veneer-py's scripts import. Returns the fake
    RiverSystem DataManager module, so a test can swap its RestrictName."""
    modules = {}

    def module(name, **attrs):
        m = ModuleType(name)
        m.__dict__.update(attrs)
        modules[name] = m
        return m

    module('clr', AddReference=lambda name: None, ImportExtensions=lambda ns: None)
    module('System', Linq=object())
    for name in ['FlowMatters', 'FlowMatters.Source', 'FlowMatters.Source.Veneer',
                 'FlowMatters.Source.Veneer.RemoteScripting',
                 'FlowMatters.Source.Veneer.RemoteScripting.ScriptHelpers',
                 'TIME', 'RiverSystem', 'RiverSystem.DataManagement']:
        module(name)
    module('TIME.Core', Unit=SimpleNamespace(parse=lambda s: 'unit:' + s))
    data_manager = module('RiverSystem.DataManagement.DataManager',
                          DataManager=SimpleNamespace(RestrictName=restrict_name))
    for name, m in modules.items():
        monkeypatch.setitem(sys.modules, name, m)
    return data_manager


def connect(source):
    v = Veneer()
    v._session = source
    return v


def frame(*columns):
    return pd.DataFrame({c: [1.0, 2.0] for c in columns},
                        index=pd.date_range('2000-01-01', periods=2))


# A data source loaded from a file

@pytest.mark.parametrize('gui', [True, False])
def test_a_file_data_source_is_named_after_the_file_stem(dotnet, gui):
    source = FakeSource(gui=gui)
    assert connect(source).create_data_source(PATH) == 'fire_tss'
    assert source.names == ['fire_tss']


def test_it_still_loads_from_the_file_and_reloads_it(dotnet):
    source = FakeSource()
    connect(source).create_data_source(PATH, reload_on_run=True)
    [posted] = source.posted
    assert posted['Name'] == PATH
    group = source.group('fire_tss')
    assert group.filename == PATH and group.reload_on_run


@pytest.mark.parametrize('gui', [True, False])
def test_a_file_data_source_can_be_given_a_name(dotnet, gui):
    source = FakeSource(gui=gui)
    v = connect(source)
    assert v.create_data_source(PATH, data_source_name='Fire TSS') == 'Fire TSS'
    assert source.names == ['Fire TSS']


@pytest.mark.parametrize('gui', [True, False])
def test_the_name_is_one_source_keeps(dotnet, gui, caplog):
    """Otherwise the interface renames it, and only when it is open."""
    source = FakeSource(gui=gui, files={'C:\\runs\\run.v2.csv': ['A'], PATH: ['A']})
    v = connect(source)
    assert v.create_data_source('C:\\runs\\run.v2.csv') == 'run_v2'
    assert v.create_data_source(PATH, data_source_name='fire.tss') == 'fire_tss'
    assert sorted(source.names) == ['fire_tss', 'run_v2']
    assert "'fire.tss'" in caplog.text and "'fire_tss'" in caplog.text


def test_the_name_follows_the_rule_of_the_source_it_is_sent_to(dotnet):
    def other_rule(name):
        return restrict_name(name).replace('-', '_')
    dotnet.DataManager = SimpleNamespace(RestrictName=other_rule)
    source = FakeSource(rule=other_rule)
    v = connect(source)
    assert v.create_data_source(PATH, data_source_name='fire-tss') == 'fire_tss'
    assert source.names == ['fire_tss']


@pytest.mark.parametrize('gui', [True, False])
def test_units_are_applied_to_the_named_data_source(dotnet, gui):
    source = FakeSource(gui=gui)
    connect(source).create_data_source(PATH, units='kg/d')
    assert source.group('fire_tss').units() == ['unit:kg/d', 'unit:kg/d']


@pytest.mark.parametrize('gui', [True, False])
def test_one_of_the_same_name_is_replaced_keeping_its_assignments(dotnet, gui):
    """As Veneer replaces a data source created under an existing name."""
    source = FakeSource(gui=gui)
    old = Group('fire_tss', ['A', 'B'])
    old.DataDetails[0].Usages.append('usage of A')
    source.dm.DataGroups.extend([old, Group('other', ['X'])])

    connect(source).create_data_source(PATH)

    assert sorted(source.names) == ['fire_tss', 'other']
    new = source.group('fire_tss')
    assert new is not old and new.filename == PATH
    assert new.usages() == {'A': ['usage of A']}
    assert source.dm.refreshed


def test_assignments_with_no_column_to_move_to_are_reported(dotnet, caplog):
    source = FakeSource()
    old = Group('fire_tss', ['A', 'Gone'])
    old.DataDetails[1].Usages.append('usage of Gone')
    source.dm.DataGroups.append(old)

    connect(source).create_data_source(PATH)

    assert source.names == ['fire_tss']
    assert 'Gone' in caplog.text


def test_one_named_after_the_path_by_an_older_version_is_left_alone(dotnet):
    source = FakeSource()
    earlier = Group(restrict_name(PATH), ['A', 'B'])
    source.dm.DataGroups.append(earlier)

    connect(source).create_data_source(PATH)

    assert sorted(source.names) == sorted(['fire_tss', restrict_name(PATH)])
    assert source.group(restrict_name(PATH)) is earlier


@pytest.mark.parametrize('gui', [True, False])
def test_without_scripts_it_keeps_the_name_source_gave_it(dotnet, gui, caplog):
    """Renaming needs Veneer's Allow Scripts. Without it, the data source keeps
    whatever name Source gave it, and says so."""
    source = FakeSource(gui=gui, scripts=False)
    given = restrict_name(PATH) if gui else PATH
    assert connect(source).create_data_source(PATH) == given
    assert source.names == [given]
    assert 'Allow Scripts' in caplog.text


def test_without_scripts_units_still_cannot_be_applied(dotnet):
    source = FakeSource(scripts=False)
    with pytest.raises(Exception):
        connect(source).create_data_source(PATH, units='kg/d')


# A data source created from a DataFrame

@pytest.mark.parametrize('gui', [True, False])
def test_a_dataframe_data_source_keeps_its_name(dotnet, gui):
    source = FakeSource(gui=gui)
    assert connect(source).create_data_source('flow', frame('A')) == 'flow'
    assert source.names == ['flow']


@pytest.mark.parametrize('gui', [True, False])
def test_a_dataframe_data_source_gets_a_name_source_keeps(dotnet, gui):
    source = FakeSource(gui=gui)
    assert connect(source).create_data_source('fire.tss', frame('A')) == 'fire_tss'
    assert source.names == ['fire_tss']
    [posted] = source.posted
    assert posted['Name'] == 'fire_tss'


def test_a_dataframe_data_source_can_be_given_a_name(dotnet):
    source = FakeSource()
    v = connect(source)
    assert v.create_data_source('flow', frame('A'), data_source_name='Flow 2') == 'Flow 2'
    assert source.names == ['Flow 2']


def test_a_dataframe_data_source_needs_no_scripts(dotnet):
    source = FakeSource(scripts=False)
    assert connect(source).create_data_source('fire.tss', frame('A')) == 'fire_tss'
    assert source.names == ['fire_tss']


def test_recreating_a_dataframe_data_source_replaces_it(dotnet):
    source = FakeSource()
    v = connect(source)
    v.create_data_source('fire.tss', frame('A'))
    v.create_data_source('fire.tss', frame('A'))
    assert source.names == ['fire_tss']

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
        # Source's interface, when its data source tree insists on the UI thread
        self.ui = None

    def _check_thread(self):
        """As Source 6.10's interface fails a data group change off its UI thread."""
        if self.ui is not None and not self.ui.on_ui_thread:
            raise Exception('The calling thread must be STA, because many UI '
                            'components require this.')

    def add(self, group):
        self._check_thread()
        self.DataGroups.append(group)

    def RemoveGroup(self, group):
        self._check_thread()
        self.DataGroups.remove(group)

    def Refresh(self):
        self._check_thread()
        self.refreshed += 1


class FakeForm:
    """Source's main window: RiverSystem.Forms.MainForm.Instance."""

    def __init__(self, responds=True):
        self.InvokeRequired = True
        self.on_ui_thread = False
        self.invoked = 0
        self.responds = responds

    def BeginInvoke(self, action):
        self.invoked += 1
        if self.responds:
            self.on_ui_thread = True
            try:
                action()
            finally:
                self.on_ui_thread = False
        return SimpleNamespace(
            AsyncWaitHandle=SimpleNamespace(WaitOne=lambda timeout: self.responds))

    def EndInvoke(self, handle):
        pass


class SimpleDataItem:
    """Veneer's FlowMatters.Source.Veneer.ExchangeObjects.DataSources.SimpleDataItem."""
    Name = None
    InputSets = None
    DetailsAsCSV = None
    UnitsForNewTS = None
    ReloadOnRun = False


class SimpleDataGroupItem:
    """Veneer's SimpleDataGroupItem, as its AddToScenario and ReplaceInScenario
    behave: the group named as given - after the file, for one loaded from a file."""
    Name = None
    Items = None

    def AddToScenario(self, scenario):
        source = scenario.fake_source
        [item] = self.Items
        if item.DetailsAsCSV is None:
            if self.Name not in source.files:
                raise Exception('Could not find file ' + self.Name)
            columns, filename = source.files[self.Name], self.Name
        else:
            columns, filename = item.DetailsAsCSV.splitlines()[0].split(',')[1:], None
        group = Group(self.Name, columns, filename, item.ReloadOnRun)
        if source.gui:
            group.Name = source.rule(group.Name)
        scenario.Network.DataManager.add(group)
        return group

    def ReplaceInScenario(self, scenario, existing):
        dm = scenario.Network.DataManager
        name = self.Name
        self.Name = 'TMP___' + name
        new = self.AddToScenario(scenario)
        for detail in existing.DataDetails:
            [dest] = [d for d in new.DataDetails if d.Name == detail.Name]
            dest.Usages.extend(detail.Usages)
            detail.Usages.Clear()
        self.Name = name
        new.Name = name
        dm.RemoveGroup(existing)
        dm.Refresh()


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
    """Veneer's HTTP interface, over a fake data manager.

    ui: Source's main window, when Source has its interface open. Veneer older than
        DATA_SOURCE_CONTRACT changes data groups on a worker thread, which Source
        6.10's interface refuses: a POST to /dataSources then fails with 400, leaving
        an empty group behind under the name the interface gives it."""

    def __init__(self, gui=True, scripts=True, files=None, rule=restrict_name, ui=None,
                 version=20261019):
        self.gui = gui
        self.scripts = scripts
        self.files = {PATH: ['A', 'B']} if files is None else files
        self.rule = rule
        self.dm = DataManager()
        self.dm.ui = ui
        self.ui = ui
        self.version = version
        self.posted = []
        self.scripts_run = []

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
        if path == '/':
            return Response(200, {'Version': self.version})
        assert path == '/dataSources', path
        return Response(200, [{'Name': g.Name} for g in self.dm.DataGroups])

    def _create(self, body):
        self.posted.append(body)
        if self.version >= 20261027:
            return self._create_by_contract(body)
        name = body['Name']
        [item] = body['Items']
        if self.ui is not None and self.version < 20261027:
            # The group is added before Source's interface throws, so it stays,
            # empty and renamed - under TMP___<name> when replacing one.
            existing = any(g.Name == name for g in self.dm.DataGroups)
            left = Group(self.rule(('TMP___' if existing else '') + name))
            self.dm.DataGroups.append(left)
            return Response(400, '<html>Request Error</html>')
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

    def _create_by_contract(self, body):
        """POST /dataSources from Veneer protocol 20261027: a file item loads from its
        Filename, the server applies RestrictName itself and replaces a data source of
        that name, and the response is the new data source's summary."""
        name = body.get('Name') or ''
        if not name.strip():
            return Response(400, {'Message': 'A data source needs a Name.',
                                  'StackTrace': 'at CreateDataSource'})
        [item] = body['Items']
        kept = self.rule(name)
        if item.get('DetailsAsCSV') is None:
            filename = item.get('Filename') or name
            if filename not in self.files:
                return Response(500, {'Message': 'Could not find file ' + filename,
                                      'StackTrace': 'at LoadFromFile'})
            columns = self.files[filename]
        else:
            columns, filename = item['DetailsAsCSV'].splitlines()[0].split(',')[1:], None
        new = Group(kept, columns, filename, item['ReloadOnRun'])
        for old in [g for g in self.dm.DataGroups if g.Name in (kept, name)]:
            for detail in old.DataDetails:
                dest = [d for d in new.DataDetails if d.Name == detail.Name]
                if dest:
                    dest[0].Usages.extend(detail.Usages)
            self.dm.DataGroups.remove(old)
        self.dm.DataGroups.append(new)
        return Response(200, {
            'Name': kept, 'id': '/dataSources/' + kept, 'FullName': '/' + kept,
            'Path': '/', 'ReloadMatchesOnName': False,
            'Items': [{'Name': item['Name'], 'InputSets': item['InputSets'],
                       'Filename': filename, 'FilenameIsRelative': False,
                       'ReloadOnRun': item['ReloadOnRun'], 'UseName': False,
                       'Details': [{'Name': c, 'TimeSeries': {}} for c in columns]}]})

    def _run(self, script):
        self.scripts_run.append(script)
        if not self.scripts:
            return Response(403)
        sys.modules['RiverSystem.Forms'].MainForm.Instance = self.ui
        scope = {'scenario': SimpleNamespace(Network=SimpleNamespace(DataManager=self.dm),
                                             fake_source=self)}
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

    class Array:
        """System.Array: Array[T](items)."""
        def __class_getitem__(cls, item_type):
            return list

    module('clr', AddReference=lambda name: None, ImportExtensions=lambda ns: None)
    module('System', Linq=object(), Action=lambda f: f, Array=Array, String=str,
           TimeSpan=SimpleNamespace(FromSeconds=lambda s: s))
    for name in ['FlowMatters', 'FlowMatters.Source', 'FlowMatters.Source.Veneer',
                 'FlowMatters.Source.Veneer.RemoteScripting',
                 'FlowMatters.Source.Veneer.RemoteScripting.ScriptHelpers',
                 'FlowMatters.Source.Veneer.ExchangeObjects',
                 'TIME', 'RiverSystem', 'RiverSystem.DataManagement']:
        module(name)
    module('FlowMatters.Source.Veneer.ExchangeObjects.DataSources',
           SimpleDataGroupItem=SimpleDataGroupItem, SimpleDataItem=SimpleDataItem)
    module('RiverSystem.Forms', MainForm=SimpleNamespace(Instance=None))
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


# Source's interface with a Veneer older than DATA_SOURCE_CONTRACT: Veneer adds the
# group on a worker thread, the interface refuses, and the POST fails with 400.
# veneer-py then does the same work in a script on the interface's own thread.

def unfixed_gui(**kwargs):
    return FakeSource(gui=True, ui=FakeForm(), **kwargs)


def test_an_unfixed_interface_gets_its_file_data_source_on_its_ui_thread(dotnet):
    source = unfixed_gui()
    assert connect(source).create_data_source(PATH, reload_on_run=True) == 'fire_tss'

    assert source.names == ['fire_tss']
    group = source.group('fire_tss')
    assert [d.Name for d in group.DataDetails] == ['A', 'B']
    assert group.filename == PATH and group.reload_on_run
    assert source.ui.invoked


def test_an_unfixed_interface_gets_its_dataframe_data_source(dotnet):
    source = unfixed_gui()
    assert connect(source).create_data_source('flow', frame('A')) == 'flow'
    assert source.names == ['flow']
    assert [d.Name for d in source.group('flow').DataDetails] == ['A']


def test_an_unfixed_interface_replaces_a_dataframe_data_source_keeping_assignments(dotnet):
    source = unfixed_gui()
    old = Group('flow', ['A'])
    old.DataDetails[0].Usages.append('usage of A')
    source.dm.DataGroups.append(old)

    assert connect(source).create_data_source('flow', frame('A')) == 'flow'

    assert source.names == ['flow']
    new = source.group('flow')
    assert new is not old and new.usages() == {'A': ['usage of A']}


def test_an_unfixed_interface_replaces_a_named_file_data_source(dotnet):
    """The rename and replace that follow creating a file data source change data
    groups too, so they also run on the interface's thread."""
    source = unfixed_gui()
    old = Group('fire_tss', ['A', 'B'])
    old.DataDetails[0].Usages.append('usage of A')
    source.dm.DataGroups.append(old)

    assert connect(source).create_data_source(PATH) == 'fire_tss'

    assert source.names == ['fire_tss']
    assert source.group('fire_tss').usages() == {'A': ['usage of A']}


def test_only_the_empty_group_the_failed_post_left_is_removed(dotnet):
    """A data source already of that name, with columns, is not the failed POST's."""
    source = unfixed_gui(files={PATH: ['A'], 'C:\other\fire_tss.csv': ['A']})
    earlier = Group(restrict_name(PATH), ['A'])
    source.dm.DataGroups.append(earlier)

    connect(source).create_data_source(PATH, data_source_name='new_tss')

    assert source.group(restrict_name(PATH)) is earlier
    assert sorted(source.names) == sorted(['new_tss', restrict_name(PATH)])


def test_a_file_the_interface_cannot_load_is_reported(dotnet):
    source = unfixed_gui(files={})
    with pytest.raises(Exception, match='Could not find file'):
        connect(source).create_data_source(PATH)


def test_an_interface_that_does_not_respond_is_reported(dotnet):
    source = FakeSource(gui=True, ui=FakeForm(responds=False))
    with pytest.raises(Exception, match='did not respond'):
        connect(source).create_data_source('flow', frame('A'))


def test_a_fixed_veneer_is_not_worked_around(dotnet):
    """From DATA_SOURCE_CONTRACT, Veneer makes the change on the UI thread itself, so
    a 400 from it means something else, and is raised as it is."""
    from veneer.general import VeneerHTTPError
    source = FakeSource(gui=True, ui=FakeForm(), version=20261027)
    source._create = lambda body: Response(400, '<html>Request Error</html>')
    with pytest.raises(VeneerHTTPError) as raised:
        connect(source).create_data_source('flow', frame('A'))
    assert raised.value.status_code == 400
    assert source.scripts_run == []


def test_a_command_line_400_is_raised_as_it_is(dotnet):
    """With no interface there is no UI thread to blame: the POST's error stands, and
    nothing is removed."""
    from veneer.general import VeneerHTTPError
    source = FakeSource(gui=False)
    left = Group(PATH)
    def failing(body):
        source.dm.DataGroups.append(left)
        return Response(400, '<html>Request Error</html>')
    source._create = failing
    with pytest.raises(VeneerHTTPError):
        connect(source).create_data_source(PATH)
    assert source.names == [PATH]


def test_without_scripts_the_400_is_raised_as_it_is(dotnet):
    from veneer.general import VeneerHTTPError
    source = unfixed_gui(scripts=False)
    with pytest.raises(VeneerHTTPError) as raised:
        connect(source).create_data_source('flow', frame('A'))
    assert raised.value.status_code == 400


def test_other_failures_are_not_worked_around(dotnet):
    from veneer.general import VeneerHTTPError
    source = unfixed_gui()
    source._create = lambda body: Response(500, 'boom')
    with pytest.raises(VeneerHTTPError) as raised:
        connect(source).create_data_source('flow', frame('A'))
    assert raised.value.status_code == 500
    assert source.scripts_run == []


# A Veneer with the data source contract (protocol 20261027 and later): the file
# loads from Items[0].Filename, so the data source is named in the POST itself, by a
# server that applies Source's naming rule and returns the name it kept. No script.

def contract(**kwargs):
    kwargs.setdefault('version', 20261027)
    return FakeSource(**kwargs)


@pytest.mark.parametrize('gui', [True, False])
def test_a_file_data_source_is_named_in_the_post_with_no_script(dotnet, gui):
    source = contract(gui=gui, scripts=False)
    assert connect(source).create_data_source(PATH, reload_on_run=True) == 'fire_tss'

    [posted] = source.posted
    assert posted['Name'] == 'fire_tss'
    assert posted['Items'][0]['Filename'] == PATH
    assert posted['Items'][0]['ReloadOnRun'] is True
    assert source.names == ['fire_tss']
    assert source.group('fire_tss').filename == PATH
    assert source.scripts_run == []


def test_the_name_given_is_the_one_the_server_kept(dotnet, caplog):
    def other_rule(name):
        return restrict_name(name).replace('-', '_')
    source = contract(rule=other_rule)
    v = connect(source)
    assert v.create_data_source(PATH, data_source_name='fire-tss') == 'fire_tss'
    assert source.names == ['fire_tss']
    assert "'fire_tss'" in caplog.text


def test_a_name_with_reserved_characters_is_kept_as_source_keeps_it(dotnet, caplog):
    source = contract()
    assert connect(source).create_data_source(PATH, data_source_name='fire.tss') == 'fire_tss'
    assert "'fire.tss'" in caplog.text and "'fire_tss'" in caplog.text


def test_units_for_a_file_are_applied_to_the_name_the_server_kept(dotnet):
    """The server still ignores units for a file item."""
    source = contract()
    assert connect(source).create_data_source(PATH, units='kg/d') == 'fire_tss'
    assert source.group('fire_tss').units() == ['unit:kg/d', 'unit:kg/d']
    [posted] = source.posted
    assert posted['Items'][0]['UnitsForNewTS'] == ''


def test_a_dataframe_data_source_is_posted_as_before(dotnet):
    source = contract(scripts=False)
    assert connect(source).create_data_source('fire.tss', frame('A'), units='kg/d') \
        == 'fire_tss'
    [posted] = source.posted
    item = posted['Items'][0]
    assert item['UnitsForNewTS'] == 'kg/d'
    assert item['DetailsAsCSV'].startswith(',A')
    assert 'Filename' not in item
    assert source.names == ['fire_tss']


def test_the_server_replaces_one_of_the_same_name_keeping_assignments(dotnet):
    source = contract(scripts=False)
    old = Group('fire_tss', ['A', 'B'])
    old.DataDetails[0].Usages.append('usage of A')
    source.dm.DataGroups.extend([old, Group('other', ['X'])])

    assert connect(source).create_data_source(PATH) == 'fire_tss'

    assert sorted(source.names) == ['fire_tss', 'other']
    assert source.group('fire_tss').usages() == {'A': ['usage of A']}


def test_a_file_the_server_cannot_load_is_reported_with_its_message(dotnet):
    from veneer.general import VeneerHTTPError
    source = contract(files={})
    with pytest.raises(VeneerHTTPError) as raised:
        connect(source).create_data_source(PATH)
    assert raised.value.status_code == 500
    assert 'Could not find file' in str(raised.value)
    assert 'at LoadFromFile' in str(raised.value)
    assert source.names == []


def test_a_contract_server_is_not_worked_around_in_its_interface(dotnet):
    """It makes its changes on Source's UI thread itself."""
    from veneer.general import VeneerHTTPError
    source = contract(ui=FakeForm())
    source._create_by_contract = lambda body: Response(400, {'Message': 'nope'})
    with pytest.raises(VeneerHTTPError):
        connect(source).create_data_source('flow', frame('A'))
    assert source.scripts_run == []


def test_an_older_server_is_never_sent_a_filename(dotnet):
    """It would ignore Filename and load from Name: so Name is the path."""
    source = FakeSource()
    connect(source).create_data_source(PATH, data_source_name='other_name')
    [posted] = source.posted
    assert posted['Name'] == PATH
    assert 'Filename' not in posted['Items'][0]


def test_a_veneer_that_reports_no_version_is_treated_as_older(dotnet):
    source = FakeSource()
    source.get = lambda url: (Response(200, {}) if url.endswith(':9876/')
                              else FakeSource.get(source, url))
    assert connect(source).create_data_source(PATH) == 'fire_tss'
    assert source.posted[0]['Name'] == PATH

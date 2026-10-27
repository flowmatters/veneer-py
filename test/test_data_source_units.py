"""Unit tests for VeneerIronPython.set_data_source_units, and so for the units
v.create_data_source applies to a file-based data source.

The generated script is run in CPython against a fake data manager, with the .NET
pieces it imports (clr, TIME.Core.Unit and RiverSystem's DataManager) stood in for.
"""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from veneer.server_side import VeneerIronPython

PATH = 'C:\\models\\inputs\\flow.csv'


def restrict_name(name):
    """Source's DataManager.RestrictName, as the Source interface applies it to
    each data source it shows."""
    for c in '.@>,\\':
        name = name.replace(c, '_')
    return name[:200].strip(' ')


class Groups(list):
    def FirstOrDefault(self, predicate):
        return next((g for g in self if predicate(g)), None)

    def LastOrDefault(self, predicate):
        return next((g for g in reversed(self) if predicate(g)), None)


def group(name, columns=1):
    data = [SimpleNamespace(Data=SimpleNamespace(TimeSeries=SimpleNamespace(units=None)))
            for _ in range(columns)]
    item = SimpleNamespace(DataSource=SimpleNamespace(Data=data))
    return SimpleNamespace(Name=name, InputSetItems=[item])


def units_of(grp):
    return [d.Data.TimeSeries.units for isi in grp.InputSetItems
            for d in isi.DataSource.Data]


class FakeModel(VeneerIronPython):
    """Runs the scripts it generates against `groups`, in place of Source."""

    def __init__(self, groups):
        self.groups = Groups(groups)

    def _init_script(self, namespace=None):
        return 'import clr\n'

    def run_script(self, script, run_async=False, init=False):
        scope = {'scenario': SimpleNamespace(
            Network=SimpleNamespace(DataManager=SimpleNamespace(DataGroups=self.groups)))}
        try:
            exec(script, scope)
        except Exception as e:
            return {'Exception': str(e), 'Response': None}
        return {'Exception': None, 'Response': scope.get('result')}

    def simplify_response(self, response):
        return response


@pytest.fixture
def dotnet(monkeypatch):
    """Stand-ins for the .NET modules the script imports. Returns the fake
    RiverSystem DataManager module, so a test can remove RestrictName."""
    clr = ModuleType('clr')
    clr.AddReference = lambda name: None
    time_core = ModuleType('TIME.Core')
    time_core.Unit = SimpleNamespace(parse=lambda s: 'unit:' + s)
    data_manager = ModuleType('RiverSystem.DataManagement.DataManager')
    data_manager.DataManager = SimpleNamespace(RestrictName=restrict_name)
    for name, module in [('clr', clr),
                         ('TIME', ModuleType('TIME')),
                         ('TIME.Core', time_core),
                         ('RiverSystem', ModuleType('RiverSystem')),
                         ('RiverSystem.DataManagement',
                          ModuleType('RiverSystem.DataManagement')),
                         ('RiverSystem.DataManagement.DataManager', data_manager)]:
        monkeypatch.setitem(sys.modules, name, module)
    return data_manager


def test_units_are_set_on_the_named_data_source(dotnet):
    flow, other = group('flow', 2), group('other')
    model = FakeModel([other, flow])
    assert model.set_data_source_units('flow', 'ML/d') == 2
    assert units_of(flow) == ['unit:ML/d', 'unit:ML/d']
    assert units_of(other) == [None]


def test_units_cycle_over_the_columns(dotnet):
    flow = group('flow', 3)
    FakeModel([flow]).set_data_source_units('flow', 'ML/d, mm/d')
    assert units_of(flow) == ['unit:ML/d', 'unit:mm/d', 'unit:ML/d']


def test_a_data_source_the_interface_renamed_is_found(dotnet):
    """With its interface open, Source renames a data source created from a file
    path, replacing the path's backslashes and full stop with underscores."""
    renamed = group(restrict_name(PATH))
    model = FakeModel([renamed])
    assert model.set_data_source_units(PATH, 'ML/d') == 1
    assert units_of(renamed) == ['unit:ML/d']


def test_the_most_recent_renamed_data_source_is_the_one_set(dotnet):
    earlier, latest = group(restrict_name(PATH)), group(restrict_name(PATH))
    FakeModel([earlier, latest]).set_data_source_units(PATH, 'ML/d')
    assert units_of(latest) == ['unit:ML/d']
    assert units_of(earlier) == [None]


def test_an_exact_name_is_preferred_to_a_renamed_one(dotnet):
    exact, renamed = group(PATH), group(restrict_name(PATH))
    FakeModel([exact, renamed]).set_data_source_units(PATH, 'ML/d')
    assert units_of(exact) == ['unit:ML/d']
    assert units_of(renamed) == [None]


def test_without_restrict_name_only_the_exact_name_is_found(dotnet):
    """Source versions without DataManager.RestrictName do not rename."""
    del dotnet.DataManager
    model = FakeModel([group(restrict_name(PATH))])
    with pytest.raises(Exception, match='Unknown data source'):
        model.set_data_source_units(PATH, 'ML/d')


def test_an_unknown_data_source_is_an_error(dotnet):
    with pytest.raises(Exception, match='Unknown data source: missing'):
        FakeModel([group('flow')]).set_data_source_units('missing', 'ML/d')

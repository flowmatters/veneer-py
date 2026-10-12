"""v.model.for_scenario: IronPython helpers that act on a NAMED scenario
without changing Veneer's active one. Veneer cannot switch scenario inside
the Source GUI (SourceService.SetScenario throws when RunningInGUI), but
every script reaches the model through ONE scope variable, `scenario`, set
fresh per call (ScriptRunner.ExecuteScript). Rebinding it at the top of the
script is therefore per-script and cannot leak."""
import pathlib
import re

import numpy as np
import pandas as pd
import pytest

from veneer.navigate import Queryable
from veneer.server_side import VeneerIronPython, scenario_rebind


class FakeVeneer:
    def __init__(self):
        self.scripts = []
        self.selected = []

    def run_server_side_script(self, script, run_async=False):
        self.scripts.append(script)
        return {'Exception': None, 'Response': None,
                'StandardOut': '', 'StandardError': ''}

    def select_scenario(self, scenario):
        self.selected.append(scenario)


class FakeScenarioWrapper:
    '''Mimics a Source RSScenario wrapper: the two properties the generated
    rebind reads -- .ScenarioName and .riverSystemScenario.'''
    def __init__(self, name):
        self.ScenarioName = name
        self.riverSystemScenario = object()  # a unique sentinel per wrapper


class FakeRiverSystemProject:
    def __init__(self, wrappers):
        self._wrappers = wrappers

    def GetRSScenarios(self):
        return self._wrappers


class FakeScenario:
    '''Mimics the IronPython `scenario` variable just enough for
    scenario_rebind's generated script to run under exec() on CPython 3 --
    the same engine Source 6.x embeds (IronPython 3.4, not 2.7).'''
    def __init__(self, wrappers):
        self.RiverSystemProject = FakeRiverSystemProject(wrappers)


def _exec_rebind(rebind_text, wrappers):
    ns = {'scenario': FakeScenario(wrappers)}
    exec(rebind_text, ns)
    return ns


def test_an_unscoped_script_is_sent_unchanged():
    v = FakeVeneer()
    VeneerIronPython(v).run_script('result = 1\n')
    assert v.scripts == ['result = 1\n']


def test_a_scoped_script_starts_with_the_rebind_by_name():
    v = FakeVeneer()
    VeneerIronPython(v).for_scenario('Tributaries').run_script('result = 1\n')
    assert v.scripts[0].startswith(scenario_rebind('Tributaries'))
    assert v.scripts[0].endswith('result = 1\n')


def test_the_name_lookup_mirrors_SetScenario():
    text = scenario_rebind('Tributaries')
    assert 'scenario.RiverSystemProject.GetRSScenarios()' in text
    assert '__s.ScenarioName == u"Tributaries"' in text
    assert '.riverSystemScenario' in text
    assert 'No scenario named "' in text
    assert '" in this project' in text


def test_the_not_found_message_quotes_the_name():
    text = scenario_rebind('Tributaries')
    assert '''raise Exception('No scenario named "' + u"Tributaries" + '" in this project')''' in text


def test_a_scoped_script_by_index():
    text = scenario_rebind(1)
    assert 'list(scenario.RiverSystemProject.GetRSScenarios())' in text
    assert '[1].riverSystemScenario' in text
    assert 'out of range' in text


def test_the_rebind_cleans_up_its_own_working_name():
    # __scenarios must not survive the rebind, on either engine.
    name_text = scenario_rebind('Tributaries')
    assert 'del __scenarios' in name_text
    index_text = scenario_rebind(1)
    assert 'del __scenarios' in index_text


# The rebind must run on both IronPython 2.7 (Source 5.x) and 3.4 (Source
# 6.x). Under IronPython/CPython 2.7 a list-comprehension variable LEAKS
# into the enclosing scope; under 3.x it does NOT (its own frame). A `del
# __s` therefore raises NameError on 3.x -- exactly what Source 6.x's
# embedded IronPython 3.4 runs. These exec the generated script for real,
# under this (CPython 3) interpreter, against fakes standing in for the
# `scenario` the real script would see.

def test_exec_rebind_by_name_found():
    wrappers = [FakeScenarioWrapper('Historic'), FakeScenarioWrapper('Tributaries')]
    ns = _exec_rebind(scenario_rebind('Tributaries'), wrappers)
    assert ns['scenario'] is wrappers[1].riverSystemScenario
    assert '__scenarios' not in ns


def test_exec_rebind_by_index_found():
    wrappers = [FakeScenarioWrapper('Historic'), FakeScenarioWrapper('Tributaries')]
    ns = _exec_rebind(scenario_rebind(1), wrappers)
    assert ns['scenario'] is wrappers[1].riverSystemScenario
    assert '__scenarios' not in ns


def test_exec_rebind_by_name_missing_raises():
    wrappers = [FakeScenarioWrapper('Historic')]
    with pytest.raises(Exception) as exc_info:
        _exec_rebind(scenario_rebind('Nope'), wrappers)
    assert 'No scenario named "Nope"' in str(exc_info.value)


def test_exec_rebind_by_index_out_of_range_raises():
    wrappers = [FakeScenarioWrapper('Historic')]
    with pytest.raises(Exception) as exc_info:
        _exec_rebind(scenario_rebind(5), wrappers)
    assert 'out of range' in str(exc_info.value)


def test_exec_rebind_non_ascii_name_found():
    wrappers = [FakeScenarioWrapper('Café')]
    ns = _exec_rebind(scenario_rebind('Café'), wrappers)
    assert ns['scenario'] is wrappers[0].riverSystemScenario


def test_generated_rebind_is_valid_python():
    # The generated IronPython must stay syntactically valid: compile()
    # parses it without executing, so undefined names like `scenario` are
    # fine, but a quoting mistake would show up as a SyntaxError here.
    compile(scenario_rebind('Tributaries'), '<scenario_rebind:name>', 'exec')
    compile(scenario_rebind('Café'), '<scenario_rebind:unicode-name>', 'exec')
    compile(scenario_rebind(1), '<scenario_rebind:index>', 'exec')


@pytest.mark.parametrize('bad', [True, 1.5, None, ['x']])
def test_a_bad_scenario_type_is_refused_before_anything_is_sent(bad):
    with pytest.raises(TypeError):
        scenario_rebind(bad)


@pytest.mark.parametrize('bad', [-1, '', '   '])
def test_a_bad_scenario_value_is_refused_before_anything_is_sent(bad):
    with pytest.raises(ValueError):
        scenario_rebind(bad)


def test_a_non_ascii_name_is_a_unicode_literal():
    # The rebind must run on both IronPython 2.7 (Source 5.x) and 3.4
    # (Source 6.x): under 2.7 a plain "é" literal is NOT an escape, so a
    # u"..." literal is used -- it is also valid, and portable, on 3.4.
    assert 'u"Caf\\u00e9"' in scenario_rebind('Café')


def test_an_integral_index_other_than_int_is_accepted_and_normalised():
    # eg numpy.int64, which pandas/numpy code hands back routinely. Not a
    # bool (bool is also "integral" but is refused above).
    assert scenario_rebind(np.int64(1)) == scenario_rebind(1)


def test_scoping_never_selects_a_scenario():
    v = FakeVeneer()
    scoped = VeneerIronPython(v).for_scenario('Tributaries')
    scoped.run_script('result = 1\n')
    scoped.get('scenario.Name')
    assert v.selected == []


def test_helpers_reached_through_a_scoped_model_are_scoped():
    # Every helper object (operations, simulation, node, link, ...) is built
    # with the instance itself, so the scope reaches all of them.
    v = FakeVeneer()
    scoped = VeneerIronPython(v).for_scenario('Tributaries')
    scoped.get('scenario.Name')
    scoped.simulation.select_input_set('Default Input Set')
    scoped.operations.clear_overrides('Forecast')
    assert len(v.scripts) == 3
    for script in v.scripts:
        assert script.startswith(scenario_rebind('Tributaries'))


def test_an_indented_script_is_cleaned_before_the_rebind_is_added():
    # veneer-py's own convention is a triple-quoted script opening with a
    # newline (see find_parameters). clean_script dedents by the leading
    # whitespace; a rebind added FIRST would leave it nothing to dedent.
    # The test compares with clean_script's own output rather than a literal,
    # because clean_script keeps one column (l[indent - 1:]) by design.
    script = '\n    result = 1\n    x = 2\n'
    v = FakeVeneer()
    ironpy = VeneerIronPython(v)
    ironpy.for_scenario(0).run_script(script)
    assert v.scripts[0] == scenario_rebind(0) + ironpy.clean_script(script)


def test_scoped_scenario_is_exposed():
    v = FakeVeneer()
    assert VeneerIronPython(v).scoped_scenario is None
    scoped = VeneerIronPython(v).for_scenario('Tributaries')
    assert scoped.scoped_scenario == 'Tributaries'


def test_a_scoped_apply_overrides_is_scoped():
    v = FakeVeneer()
    df = pd.DataFrame({('StorageA', 'Release', '-'): [1.0]},
                      index=pd.DatetimeIndex(['2026-10-13']))
    VeneerIronPython(v).for_scenario('Tributaries').operations.apply_overrides(
        df, 'Forecast')
    assert v.scripts[0].startswith(scenario_rebind('Tributaries'))


def test_an_unscoped_apply_overrides_script_is_unprefixed():
    v = FakeVeneer()
    df = pd.DataFrame({('StorageA', 'Release', '-'): [1.0]},
                      index=pd.DatetimeIndex(['2026-10-13']))
    VeneerIronPython(v).operations.apply_overrides(df, 'Forecast')
    assert v.scripts[0].startswith('# Generated Script')
    assert '__scenarios' not in v.scripts[0]


def test_nothing_bypasses_run_script():
    # The ONE allowed call to run_server_side_script is run_script's own.
    root = pathlib.Path(__file__).resolve().parent.parent / 'veneer'
    hits = []
    for path in sorted(root.rglob('*.py')):
        for line in path.read_text(encoding='utf-8').splitlines():
            if re.search(r'\brun_server_side_script\(', line) \
                    and 'def run_server_side_script' not in line:
                hits.append((path.name, line.strip()))
    assert hits, 'the sweep found nothing — it is not looking where it should'
    assert hits == [('server_side.py',
                     'return self._veneer.run_server_side_script(script, run_async)')]


def test_veneer_model_access_outside_general_py_is_a_known_short_list():
    # `_veneer.model` always resolves to the UNSCOPED original model
    # (general.py's own `self.model = VeneerIronPython(self)`), never a
    # scoped instance's own `self` -- so any other hit is a potential scope
    # bypass, same shape as the run_server_side_script sweep above. This is
    # NOT forced to zero: bulk.py's spatial-raster helpers (checking/saving
    # rasters) are a pre-existing, unrelated use with no scenario-scoped
    # concept, not a new bypass risk introduced by this feature. The sweep
    # instead pins the current, reported list, so a NEW occurrence anywhere
    # else still fails loudly.
    root = pathlib.Path(__file__).resolve().parent.parent / 'veneer'
    hits = []
    for path in sorted(root.rglob('*.py')):
        if path.name == 'general.py':
            continue
        for line in path.read_text(encoding='utf-8').splitlines():
            if re.search(r'_veneer\.model\b', line):
                hits.append((path.name, line.strip()))
    assert hits == [
        ('bulk.py', 'if not self._veneer.model.spatial.has_raster(r):'),
        ('bulk.py', "self._veneer.model.spatial.save_raster(r,self.make_dest(f'spatial/{r}','asc'),categories)"),
    ]


def test_ironpy_veneer_rest_mixing_is_a_known_pinned_list():
    # `_ironpy._veneer` reaches past the IronPython layer to the raw Veneer
    # client -- legitimate for a REST call, which has no scope concept at
    # all, but worth a deliberate look each time one appears, same shape as
    # the two sweeps above. Pinned to what's actually here today:
    # component.py's parse_veneer_date (REST, no scope to honour),
    # nav_first's own Queryable constructor call (passes the SCOPED
    # instance's veneer through correctly, alongside ironpy=self._ironpy),
    # and the now-guarded update_variable_piecewise. A new helper mixing
    # IronPython and REST fails this loudly instead of silently bypassing
    # scope.
    root = pathlib.Path(__file__).resolve().parent.parent / 'veneer'
    hits = []
    for path in sorted(root.rglob('*.py')):
        for line in path.read_text(encoding='utf-8').splitlines():
            if re.search(r'_ironpy\._veneer\b', line):
                hits.append((path.name, line.strip()))
    assert hits == [
        ('component.py',
         "index = [self._ironpy._veneer.parse_veneer_date(e['Date']) for e in list(res.values())[0]['Events']]"),
        ('server_side.py',
         'return Queryable(self._ironpy._veneer, path=accessor, namespace=self._ns,'),
        ('server_side.py',
         'self._ironpy._veneer.update_variable_piecewise(variable_name,table)'),
    ]


def test_queryable_without_ironpy_arg_evaluates_through_v_model():
    # The documented top-level usage (`Queryable(v)`, no scoped instance in
    # hand) must keep working: ironpy defaults to v.model.
    class FakeModel:
        def __init__(self):
            self.calls = []

        def get(self, path, namespace=None):
            self.calls.append(path)
            return 'the value'

    class FakeVWithModel:
        def __init__(self):
            self.model = FakeModel()

    v = FakeVWithModel()
    q = Queryable(v)
    assert q._eval_() == 'the value'
    assert v.model.calls == ['scenario']


def test_nav_is_scoped():
    # nav()'s Queryable used to always evaluate through the ORIGINAL,
    # unscoped model (general.py's self.model), regardless of which
    # instance built it -- a silent scope bypass.
    v = FakeVeneer()
    scoped = VeneerIronPython(v).for_scenario('Tributaries')
    scoped.nav()._eval_()
    assert v.scripts
    assert v.scripts[-1].startswith(scenario_rebind('Tributaries'))


def test_nav_first_is_scoped():
    v = FakeVeneer()
    scoped = VeneerIronPython(v).for_scenario('Tributaries')
    scoped.node.nav_first(nodes='StorageA')._eval_()
    assert v.scripts
    assert v.scripts[-1].startswith(scenario_rebind('Tributaries'))


def test_create_piecewise_variable_with_table_refuses_on_a_scoped_instance():
    # update_variable_piecewise is a REST call: it always fills the table on
    # Veneer's ACTIVE scenario, so it cannot honour a scope. Must refuse
    # before doing anything -- no script sent at all.
    v = FakeVeneer()
    scoped = VeneerIronPython(v).for_scenario('Tributaries')
    table = pd.DataFrame({'Lookup': [1.0], 'Result': [2.0]})
    with pytest.raises(NotImplementedError):
        scoped.functions.create_piecewise_variable('x', table=table)
    assert v.scripts == []


def test_create_piecewise_variable_without_table_is_fine_on_a_scoped_instance():
    v = FakeVeneer()
    scoped = VeneerIronPython(v).for_scenario('Tributaries')
    scoped.functions.create_piecewise_variable('x')
    assert v.scripts
    assert v.scripts[0].startswith(scenario_rebind('Tributaries'))


def test_each_deferred_fragment_carries_its_own_rebind():
    # flush sends the joined fragments through _safe_run -> run_script, which
    # adds the rebind once more: two fragments give THREE rebinds, all naming
    # the same scenario, which is harmless (spec, veneer-py section).
    v = FakeVeneer()
    scoped = VeneerIronPython(v).for_scenario('Tributaries')
    scoped.defer()
    scoped.run_script('a = 1\n')
    scoped.run_script('b = 2\n')
    scoped.flush()
    assert len(v.scripts) == 1
    assert v.scripts[0].startswith(scenario_rebind('Tributaries'))
    assert v.scripts[0].count(scenario_rebind('Tributaries')) == 3

"""v.model.for_scenario: IronPython helpers that act on a NAMED scenario
without changing Veneer's active one. Veneer cannot switch scenario inside
the Source GUI (SourceService.SetScenario throws when RunningInGUI), but
every script reaches the model through ONE scope variable, `scenario`, set
fresh per call (ScriptRunner.ExecuteScript). Rebinding it at the top of the
script is therefore per-script and cannot leak."""
import pytest

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
    assert 's.ScenarioName == u"Tributaries"' in text
    assert '.riverSystemScenario' in text
    assert "raise Exception('No scenario named '" in text


def test_a_scoped_script_by_index():
    text = scenario_rebind(1)
    assert 'list(scenario.RiverSystemProject.GetRSScenarios())' in text
    assert '[1].riverSystemScenario' in text
    assert 'out of range' in text


@pytest.mark.parametrize('bad', [-1, True, 1.5, '', '   ', None, ['x']])
def test_a_bad_scenario_is_refused_before_anything_is_sent(bad):
    with pytest.raises((TypeError, ValueError)):
        scenario_rebind(bad)


def test_a_non_ascii_name_is_a_unicode_literal():
    # IronPython 2.7: a plain "é" literal is NOT an escape; u"..." is.
    assert 'u"Caf\\u00e9"' in scenario_rebind('Café')


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


import pathlib
import re

import pandas as pd


def test_a_scoped_apply_overrides_is_scoped():
    v = FakeVeneer()
    df = pd.DataFrame({('StorageA', 'Release', '-'): [1.0]},
                      index=pd.DatetimeIndex(['2026-10-13']))
    VeneerIronPython(v).for_scenario('Tributaries').operations.apply_overrides(
        df, 'Forecast')
    assert v.scripts[0].startswith(scenario_rebind('Tributaries'))


def test_an_unscoped_apply_overrides_sends_the_same_script_as_before():
    v = FakeVeneer()
    df = pd.DataFrame({('StorageA', 'Release', '-'): [1.0]},
                      index=pd.DatetimeIndex(['2026-10-13']))
    VeneerIronPython(v).operations.apply_overrides(df, 'Forecast')
    assert v.scripts[0].startswith('# Generated Script')


def test_nothing_bypasses_run_script():
    # The ONE allowed call to run_server_side_script is run_script's own.
    root = pathlib.Path(__file__).resolve().parent.parent / 'veneer'
    hits = []
    for path in sorted(root.rglob('*.py')):
        for n, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if re.search(r'\brun_server_side_script\(', line) \
                    and 'def run_server_side_script' not in line:
                hits.append((path.name, line.strip()))
    assert hits, 'the sweep found nothing — it is not looking where it should'
    assert hits == [('server_side.py',
                     'return self._veneer.run_server_side_script(script, run_async)')]

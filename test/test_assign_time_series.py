"""Unit tests for the script VeneerIronPython.assign_time_series generates.

The data source name and the column names are pasted into the script as string
literals. Source 5.x embeds IronPython 2.7, which reads a backslash-u escape in a
plain "..." literal as those six characters, and any other backslash sequence (eg
"\\U..." in a Windows path) as an escape or a syntax error. A u"..." literal from
json.dumps reads correctly on IronPython 2.7 and 3.4 (see scenario_rebind).

These run under CPython 3, which reads either kind of literal, so they check the
literals' form as well as what they read as.
"""
import ast
import io
import tokenize

import pytest

from veneer import Veneer

THING = 'scenario.Network.Nodes.*NodeModel.Flow'


def script_for(values, data_group, **kwargs):
    return Veneer().model.assign_time_series(THING, values, data_group=data_group,
                                             dry_run=True, **kwargs)


def literals_holding(script, value):
    """The source text of each string literal in `script` whose value is `value`."""
    return [tok.string for tok in tokenize.generate_tokens(io.StringIO(script).readline)
            if tok.type == tokenize.STRING and ast.literal_eval(tok.string) == value]


def assert_unicode_literal(script, value):
    literals = literals_holding(script, value)
    assert literals, 'no literal reads as %r' % value
    assert all(lit.startswith('u') for lit in literals), literals


@pytest.mark.parametrize('data_group', [
    'C:\\Users\\someone\\inputs\\flow.csv',   # \U: a syntax error in a plain literal
    "flows 'north'",                          # a quote would end a '...' literal
    'Caf\u00e9 inflows',                      # non-ASCII
    'flows 100% %s',                          # the script template is %-formatted
])
def test_the_data_source_name_reaches_source_intact(data_group):
    script = script_for('A', data_group)
    compile(script, '<assign_time_series>', 'exec')
    assert_unicode_literal(script, data_group)


@pytest.mark.parametrize('column', ["SC #1 'north'", 'Caf\u00e9'])
def test_a_column_name_reaches_source_intact(column):
    script = script_for(column, 'flows')
    compile(script, '<assign_time_series>', 'exec')
    assert_unicode_literal(script, column)


def test_columns_from_a_list_reach_source_intact():
    columns = ["SC #1 'north'", 'Caf\u00e9', 'plain']
    script = script_for(columns, 'flows', from_list=True)
    compile(script, '<assign_time_series>', 'exec')
    for column in columns:
        assert_unicode_literal(script, column)


def test_a_plain_assignment_still_reads_the_same():
    script = script_for('A', 'flows')
    compile(script, '<assign_time_series>', 'exec')
    assert literals_holding(script, 'flows') and literals_holding(script, 'A')


def test_a_filename_value_still_uses_forward_slashes():
    """AssignTimeSeries loads the value as a file when no column has that name."""
    script = script_for('C:\\inputs\\flow.csv', 'flows')
    assert literals_holding(script, 'C:/inputs/flow.csv')


def test_a_single_column_from_a_list_is_still_a_list():
    script = script_for(['A'], 'flows', from_list=True)
    compile(script, '<assign_time_series>', 'exec')
    assert 'origNewVal = [u"A"]' in script

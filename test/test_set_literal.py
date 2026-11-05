"""Unit tests for the string literals that model scripts paste text into.

v.model.set(..., literal=True) and a few siblings put caller-supplied text into
the IronPython script as a string literal. A plain '...' literal breaks on a
quote or a newline, reads a backslash as an escape, and lets a quote in the text
end the literal early, so that the rest of the text runs as code. A u"..." literal
from json.dumps (_unicode_literal) is ASCII-only and reads as the original text on
IronPython 2.7 and 3.4 alike (see test_assign_time_series and scenario_rebind).

These run under CPython 3, which reads either kind of literal, so they check the
literals' form as well as what they read as. ast.literal_eval reads only literals:
text it returns intact was not evaluated as code.
"""
import ast
import io
import re
import tokenize

import pandas as pd
import pytest

from veneer import Veneer
from veneer.server_side import VeneerIronPython

THING = 'scenario.Network.Nodes.*Name'

TEXTS = [
    "it's a test",            # a quote would end a '...' literal
    'a\nb',                   # a newline would split the literal across lines
    'C:\\temp\\new',          # \t and \n would read as a tab and a newline
    'caf\u00e9',              # non-ASCII
    "x' + str(1+1) + '",      # would run as code and read as "x2"
]

MIXED = ["it's", 'plain', 3, 2.5]


class FakeVeneer:
    '''Records each script instead of sending it to Source.'''
    def __init__(self):
        self.scripts = []

    def run_server_side_script(self, script, run_async=False):
        self.scripts.append(script)
        return {'Exception': None, 'Response': {'Value': []},
                'StandardOut': '', 'StandardError': ''}


def literals_holding(script, value):
    """The source text of each string literal in `script` whose value is `value`."""
    return [tok.string for tok in tokenize.generate_tokens(io.StringIO(script).readline)
            if tok.type == tokenize.STRING and ast.literal_eval(tok.string) == value]


def assert_unicode_literal(script, value):
    compile(script, '<generated>', 'exec')
    literals = literals_holding(script, value)
    assert literals, 'no literal reads as %r' % value
    for lit in literals:
        assert lit.startswith('u'), lit
        assert lit.isascii(), lit


def assigned_value(script):
    '''The source text assigned to origNewVal, and what it reads as.'''
    compile(script, '<set>', 'exec')
    match = re.search(r'^origNewVal = (.*)$', script, re.MULTILINE)
    assert match, script
    text = match.group(1)
    return text, ast.literal_eval(text)


def set_script(value, **kwargs):
    return Veneer().model.set(THING, value, literal=True, dry_run=True, **kwargs)


@pytest.mark.parametrize('text', TEXTS)
def test_a_literal_string_arrives_verbatim(text):
    source, value = assigned_value(set_script(text))
    assert value == text
    assert source.startswith('u') and source.isascii(), source


@pytest.mark.parametrize('text', TEXTS)
def test_a_literal_string_from_a_list_arrives_verbatim(text):
    source, value = assigned_value(set_script([text, 'plain'], fromList=True))
    assert value == [text, 'plain']
    assert source.isascii(), source


def test_non_strings_in_a_literal_list_are_untouched():
    source, value = assigned_value(set_script(MIXED, fromList=True))
    assert value == MIXED
    assert source == '[u"it\'s",u"plain",3,2.5]'


def test_quoted_text_is_not_run_as_code():
    text = "x' + str(1+1) + '"
    source, value = assigned_value(set_script(text))
    assert value == text and value != 'x2'
    tokens = [tok for tok in tokenize.generate_tokens(io.StringIO(source).readline)
              if tok.type not in (tokenize.NEWLINE, tokenize.ENDMARKER)]
    assert len(tokens) == 1 and tokens[0].type == tokenize.STRING, tokens


def test_a_non_literal_value_is_still_pasted_as_code():
    source, _ = assigned_value(Veneer().model.set(THING, 2.5, dry_run=True))
    assert source == '2.5'
    script = Veneer().model.set(THING, 'NodeA.Name', dry_run=True)
    assert re.search(r'^origNewVal = NodeA\.Name$', script, re.MULTILINE)


# Sibling sites that quote caller-supplied text.

INPUT_SETS = TEXTS + ['100% %s']  # the list query's loop body is %-formatted


@pytest.mark.parametrize('input_set', INPUT_SETS)
@pytest.mark.parametrize('thing', [THING, 'scenario.Network.Nodes[0].Name'])
def test_an_input_set_name_arrives_verbatim(thing, input_set):
    fake = FakeVeneer()
    VeneerIronPython(fake).get_data_sources(thing, input_set=input_set)
    assert_unicode_literal(fake.scripts[-1], input_set)


def test_no_input_set_is_still_none():
    fake = FakeVeneer()
    VeneerIronPython(fake).get_data_sources('scenario.Network.Nodes[0].Name')
    assert 'H.FindDataSource(scenario.Network.Nodes[0],Name,None)' in fake.scripts[-1]


@pytest.mark.parametrize('unit', TEXTS + ['m^3/s'])
def test_an_si_unit_arrives_verbatim(unit):
    fake = FakeVeneer()
    VeneerIronPython(fake).functions.set_options('SIUnit', unit, functions='$FunctionA')
    assert_unicode_literal(fake.scripts[-1], unit)
    assert 'Unit.parse(u' in fake.scripts[-1]


@pytest.mark.parametrize('text', TEXTS)
def test_an_assurance_rule_arrives_verbatim(text):
    fake = FakeVeneer()
    VeneerIronPython(fake).simulation.configure_assurance_rule(
        level=text, rule=text, category=text)
    script = fake.scripts[-1]
    assert_unicode_literal(script, text)
    assert len(literals_holding(script, text)) == 3


def test_an_assurance_rule_left_out_is_still_none():
    fake = FakeVeneer()
    VeneerIronPython(fake).simulation.configure_assurance_rule(level='Warning')
    assert 'H.ConfigureAssuranceRule(scenario,u"Warning",None,None)' in fake.scripts[-1]


def overrides_for(location, variable, secondary):
    columns = pd.MultiIndex.from_tuples([(location, variable, secondary)])
    return pd.DataFrame([[1.5]], index=pd.to_datetime(['2020-01-01']), columns=columns)


@pytest.mark.parametrize('text', TEXTS)
def test_override_names_arrive_verbatim(text):
    fake = FakeVeneer()
    VeneerIronPython(fake).operations.apply_overrides(
        overrides_for(text, text, text), 'Forecast')
    script = fake.scripts[-1]
    assert_unicode_literal(script, text)
    assert len(literals_holding(script, text)) == 3


def test_an_override_with_no_secondary_still_passes_none():
    fake = FakeVeneer()
    VeneerIronPython(fake).operations.apply_overrides(
        overrides_for('StorageA', 'Release', '-'), 'Forecast')
    assert 'get_override_time_series(u"StorageA",u"Release",None)' in fake.scripts[-1]

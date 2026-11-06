"""Unit tests for more places that paste caller text into a script as a string literal.

The same defect as test_set_literal, at four further sites: select_input_set's
input set name, save()'s path, a Queryable attribute assignment, and the
arguments of a deferred call (batch runs and PEST templates). A plain quoted
literal breaks on its own quote character or a newline, reads a backslash as an
escape, and lets a quote in the text end the literal early, so that the rest of
the text runs as code. Each site now writes a u"..." literal from json.dumps
(_unicode_literal), which is ASCII-only and reads as the original text on
IronPython 2.7 and 3.4 alike.

ast.literal_eval reads only literals: text it returns intact was not evaluated
as code.
"""
import ast
import io
import os
import re
import tokenize

import pytest

from veneer.navigate import Queryable
from veneer.pest import Case
from veneer.server_side import VeneerIronPython
from veneer.utils import DeferredActionCollection

TEXTS = [
    "Input set (it's a test)",   # a quote would end a '...' literal
    'a\nb',                      # a newline would split the literal across lines
    '100% sure',                 # a % must survive any %-formatting of the script
    'caf\u00e9',                 # non-ASCII
    'Set \\ A',                  # a backslash
    'C:\\temp\\new',             # \t and \n would read as a tab and a newline
    'x" + str(1+1) + "',         # would run as code and read as "x2"
    "x' + str(1+1) + '",         # the same, for a '...' literal
]


class FakeVeneer:
    '''Records each script instead of sending it to Source.'''
    def __init__(self):
        self.scripts = []

    def run_server_side_script(self, script, run_async=False):
        self.scripts.append(script)
        return {'Exception': None, 'Response': {'Value': True},
                'StandardOut': '', 'StandardError': ''}


def string_tokens(source):
    return [tok.string for tok in tokenize.generate_tokens(io.StringIO(source).readline)
            if tok.type == tokenize.STRING]


def literals_holding(script, value):
    """The source text of each string literal in `script` whose value is `value`."""
    return [s for s in string_tokens(script) if ast.literal_eval(s) == value]


def assert_unicode_literal(script, value):
    compile(script, '<generated>', 'exec')
    literals = literals_holding(script, value)
    assert literals, 'no literal reads as %r' % value
    for lit in literals:
        assert lit.startswith('u'), lit
        assert lit.isascii(), lit


def assert_single_literal(source, value):
    '''`source` is exactly one string literal, reading as `value`: nothing in it runs.'''
    tokens = [tok for tok in tokenize.generate_tokens(io.StringIO(source).readline)
              if tok.type not in (tokenize.NEWLINE, tokenize.ENDMARKER)]
    assert len(tokens) == 1 and tokens[0].type == tokenize.STRING, tokens
    assert ast.literal_eval(source) == value


# select_input_set: an input set name is whatever the modeller called it.

def select_input_set_script(name):
    fake = FakeVeneer()
    VeneerIronPython(fake).simulation.select_input_set(name)
    return fake.scripts[-1]


def compared_name(script):
    match = re.search(r'i\.Name==(.*)\)$', script, re.MULTILINE)
    assert match, script
    return match.group(1)


@pytest.mark.parametrize('name', TEXTS)
def test_an_input_set_name_to_select_arrives_verbatim(name):
    script = select_input_set_script(name)
    assert_unicode_literal(script, name)
    assert_single_literal(compared_name(script), name)


def test_an_ordinary_input_set_name_is_compared_as_before():
    script = select_input_set_script('Default Input Set')
    assert ('scenario.Network.InputSets.First(lambda i: i.Name==u"Default Input Set")'
            in script)
    assert 'scenario.CurrentConfiguration.SelectedInputSet = input_set' in script


# save(fn): a project path can hold any character a folder name can.

PATHS = [
    "C:\\temp\\it's here\\ExampleProject.rsproj",
    'C:\\temp\\new\\x.rsproj',
    'C:\\temp\\100% sure\\x.rsproj',
    "C:\\temp\\x' + str(1+1) + '.rsproj",
]


def save_script(fn):
    fake = FakeVeneer()
    VeneerIronPython(fake).save(fn)
    return fake.scripts[-1]


def saved_file_name(script):
    compile(script, '<save>', 'exec')
    match = re.search(r'^\s*cb\.OutputFileName=(.*)$', script, re.MULTILINE)
    assert match, script
    return match.group(1)


@pytest.mark.parametrize('fn', PATHS)
def test_a_save_path_arrives_verbatim(fn):
    source = saved_file_name(save_script(fn))
    assert_single_literal(source, os.path.abspath(fn))
    assert source.startswith('u') and source.isascii(), source


@pytest.mark.skipif(os.name != 'nt', reason='a Windows path is already absolute only on Windows')
@pytest.mark.parametrize('fn', PATHS)
def test_a_windows_save_path_round_trips_exactly(fn):
    assert ast.literal_eval(saved_file_name(save_script(fn))) == fn


def test_save_without_a_path_still_uses_the_current_file():
    script = save_script(None)
    assert re.search(r'^\s*cb\.OutputFileName=ph\.ProjectMetaStructure\.OutputFile$',
                     script, re.MULTILINE)


# Queryable: scenario.Name = '...' and the like, with any text.

def assigned_through_queryable(attribute, value):
    fake = FakeVeneer()
    q = Queryable(fake, ironpy=VeneerIronPython(fake))
    setattr(q.Network.Nodes[0], attribute, value)
    script = fake.scripts[-1]
    compile(script, '<set>', 'exec')
    match = re.search(r'^origNewVal = (.*)$', script, re.MULTILINE)
    assert match, script
    return script, match.group(1)


@pytest.mark.parametrize('text', TEXTS)
def test_text_assigned_through_a_queryable_arrives_verbatim(text):
    script, source = assigned_through_queryable('Name', text)
    assert_unicode_literal(script, text)
    assert_single_literal(source, text)


def test_an_ordinary_name_assigned_through_a_queryable_is_text_as_before():
    script, source = assigned_through_queryable('Name', 'StorageA')
    assert source == 'u"StorageA"'
    assert 'scenario.Network.Nodes[0].Name = newVal' in script


@pytest.mark.parametrize('value', [2.5, 3, True])
def test_a_non_string_assigned_through_a_queryable_is_untouched(value):
    _, source = assigned_through_queryable('Value', value)
    assert source == str(value)


# DeferredCall: arguments recorded for a batch run or a PEST template.

class Recorder:
    '''Stands in for `v` when a deferred script runs, recording each call.'''
    def __init__(self):
        self.calls = []

    def set_value(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def deferred_script(*args, **kwargs):
    actions = DeferredActionCollection('$', 'v.')
    actions.set_value(*args, **kwargs)
    return actions


@pytest.mark.parametrize('text', TEXTS)
def test_a_deferred_string_argument_arrives_verbatim(text):
    actions = deferred_script('NodeA', text, label=text)
    script = actions.script()
    assert_unicode_literal(script, text)
    assert len(literals_holding(script, text)) == 2

    v = Recorder()
    actions.eval_script({}, {'v': v})
    assert v.calls == [(('NodeA', text), {'label': text})]


def test_an_ordinary_deferred_call_reads_as_before():
    script = deferred_script('NodeA', 'Release').script()
    assert script == 'v.set_value(u"NodeA",u"Release")'


def test_a_calibration_parameter_is_still_left_for_substitution():
    actions = deferred_script('NodeA', '$x1$', scale='$x2$')
    assert actions.script() == 'v.set_value(u"NodeA",$x1$,scale=$x2$)'
    v = Recorder()
    actions.eval_script({'x1': 1.5, 'x2': 2}, {'v': v})
    assert v.calls == [(('NodeA', 1.5), {'scale': 2})]


@pytest.mark.parametrize('value', [2.5, 3, None])
def test_a_non_string_deferred_keyword_is_untouched(value):
    actions = deferred_script('NodeA', value=value)
    assert actions.script() == 'v.set_value(u"NodeA",value=%s)' % value


@pytest.mark.parametrize('text', TEXTS)
def test_a_deferred_string_argument_survives_the_pest_template(text):
    # The PEST template's text is %-formatted after the instructions are added.
    case = Case('ExampleProject')
    case.parameters.model.set_value('NodeA', text)
    match = re.search(r'^v\.model\.set_value\((.*)\)$', case.ptf_text(), re.MULTILINE)
    assert match
    assert ast.literal_eval('(%s,)' % match.group(1)) == ('NodeA', text)

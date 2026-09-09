"""Unit tests for VeneerIronPython.save() preserving project metadata.

Exercises script generation WITHOUT a live Source instance by stubbing run_script.
"""
import ast

import pytest

from veneer.server_side import VeneerIronPython


class FakeVeneer:
    """Stands in for the Veneer client; save() never touches it directly."""
    pass


@pytest.fixture
def ironpy():
    ip = VeneerIronPython(FakeVeneer())
    ip.scripts = []

    def fake_run_script(script):
        ip.scripts.append(script)
        return {'Exception': None, 'Response': None}

    ip.run_script = fake_run_script
    return ip


def test_save_captures_and_restores_output_file(ironpy):
    ironpy.save('C:/tmp/scratch.rsproj')
    script = ironpy.scripts[-1]
    assert 'saved_output_file = ph.ProjectMetaStructure.OutputFile' in script
    assert 'ph.ProjectMetaStructure.OutputFile = saved_output_file' in script


def test_save_captures_and_restores_project(ironpy):
    ironpy.save('C:/tmp/scratch.rsproj')
    script = ironpy.scripts[-1]
    assert 'saved_project = ph.ProjectMetaStructure.Project' in script
    assert 'ph.ProjectMetaStructure.Project = saved_project' in script


def test_save_can_opt_out_of_restoring(ironpy):
    ironpy.save('C:/tmp/scratch.rsproj', preserve_current=False)
    script = ironpy.scripts[-1]
    assert 'ph.ProjectMetaStructure.OutputFile = saved_output_file' not in script


def test_save_with_no_filename_still_reads_current_output_file(ironpy):
    """fn=None must resolve the filename BEFORE OutputFile is blanked."""
    ironpy.save()
    script = ironpy.scripts[-1]
    assert 'cb.OutputFileName=ph.ProjectMetaStructure.OutputFile' in script


@pytest.mark.parametrize('preserve', [True, False])
def test_generated_script_survives_clean_script(ironpy, preserve):
    """clean_script strips a fixed indent from every line; a mis-indented
    interpolation would produce syntactically invalid Python."""
    ironpy.save('C:/tmp/scratch.rsproj', preserve_current=preserve)
    cleaned = ironpy.clean_script(ironpy.scripts[-1])
    compile(cleaned, '<generated>', 'exec')


def test_restore_statements_are_inside_the_finally_block(ironpy):
    """A regression that relocated these out of `finally` could still compile
    and still pass a substring check - so assert on structure, not text."""
    ironpy.save('C:/tmp/scratch.rsproj')
    tree = ast.parse(ironpy.clean_script(ironpy.scripts[-1]))
    tries = [n for n in ast.walk(tree) if isinstance(n, ast.Try)]
    assert len(tries) == 1
    finalbody = ast.dump(ast.Module(body=tries[0].finalbody, type_ignores=[]))
    assert 'saved_output_file' in finalbody
    assert 'saved_project' in finalbody

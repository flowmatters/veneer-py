import os
import pytest
import veneer.manage as manage


def test_copies_project_and_extras(tmp_path):
    src = tmp_path / 'src'
    src.mkdir()
    proj = src / 'model.rsproj'
    proj.write_text('PROJECT')
    (src / 'climate.csv').write_text('DATA')
    inputs = src / 'inputs'
    inputs.mkdir()
    (inputs / 'a.txt').write_text('A')

    dest = tmp_path / 'dest'
    dest.mkdir()

    result = manage.copy_project_files(str(proj), ['climate.csv', 'inputs'], str(dest))

    assert result == os.path.join(str(dest), 'model.rsproj')
    assert (dest / 'model.rsproj').read_text() == 'PROJECT'
    assert (dest / 'climate.csv').read_text() == 'DATA'
    assert (dest / 'inputs' / 'a.txt').read_text() == 'A'


def test_missing_extra_raises(tmp_path):
    src = tmp_path / 'src'
    src.mkdir()
    proj = src / 'model.rsproj'
    proj.write_text('P')
    dest = tmp_path / 'dest'
    dest.mkdir()
    with pytest.raises(AssertionError):
        manage.copy_project_files(str(proj), ['nope.csv'], str(dest))

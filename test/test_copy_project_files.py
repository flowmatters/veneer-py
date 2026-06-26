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


def test_cluster_copy_project_parity(tmp_path):
    """cluster.copy_project still produces the expected on-disk layout after
    delegating to copy_project_files."""
    import shutil
    from veneer.cluster import copy_project

    src = tmp_path / 'src'
    src.mkdir()
    proj = src / 'm.rsproj'
    proj.write_text('P')
    (src / 'data.csv').write_text('D')

    # copy_project is dask.delayed; .compute() runs it with the default scheduler.
    tmp = copy_project('veneer-parity-test-', str(proj), ['data.csv']).compute()
    try:
        assert os.path.exists(os.path.join(tmp, 'm.rsproj'))
        assert os.path.exists(os.path.join(tmp, 'data.csv'))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

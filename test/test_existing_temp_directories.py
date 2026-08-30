import os
import shutil
import tempfile

import pytest

import veneer.cluster as cluster

PREFIX = 'source-cluster-test-'


def make_dirs(root, *names):
    made = []
    for n in names:
        d = os.path.join(str(root), n)
        os.makedirs(d)
        with open(os.path.join(d, 'model.rsproj'), 'w') as f:
            f.write('PROJECT')
        made.append(d)
    return made


@pytest.fixture
def fake_tempdir(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, 'gettempdir', lambda: str(tmp_path))
    return tmp_path


def test_raise_on_existing(fake_tempdir):
    make_dirs(fake_tempdir, PREFIX + 'a')
    with pytest.raises(Exception, match='existing temporary directories'):
        cluster.check_existing_cluster_temp_directory(PREFIX, 'raise')


def test_remove_deletes_directories(fake_tempdir):
    make_dirs(fake_tempdir, PREFIX + 'a', PREFIX + 'b', 'unrelated-')
    cluster.check_existing_cluster_temp_directory(PREFIX, 'remove')
    assert sorted(os.listdir(str(fake_tempdir))) == ['unrelated-']


def lock_one(monkeypatch, locked_name):
    """Make rmtree fail for the directory whose basename is locked_name."""
    real_rmtree = shutil.rmtree

    def fake_rmtree(path, *args, **kwargs):
        if os.path.basename(os.path.normpath(path)) == locked_name:
            if kwargs.get('ignore_errors'):
                return
            raise PermissionError(13, 'The process cannot access the file because it is being used by another process', path)
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, 'rmtree', fake_rmtree)


def test_remove_propagates_lock_failure(fake_tempdir, monkeypatch):
    make_dirs(fake_tempdir, PREFIX + 'locked')
    lock_one(monkeypatch, PREFIX + 'locked')
    with pytest.raises(PermissionError):
        cluster.check_existing_cluster_temp_directory(PREFIX, 'remove')


def test_try_remove_tolerates_lock_and_continues(fake_tempdir, monkeypatch, caplog):
    make_dirs(fake_tempdir, PREFIX + 'locked', PREFIX + 'free')
    lock_one(monkeypatch, PREFIX + 'locked')

    with caplog.at_level('WARNING'):
        cluster.check_existing_cluster_temp_directory(PREFIX, 'try_remove')

    remaining = os.listdir(str(fake_tempdir))
    assert remaining == [PREFIX + 'locked']
    assert any(PREFIX + 'locked' in r.getMessage() for r in caplog.records)


def test_try_remove_with_nothing_to_remove(fake_tempdir):
    cluster.check_existing_cluster_temp_directory(PREFIX, 'try_remove')


def test_tryremove_alias(fake_tempdir):
    make_dirs(fake_tempdir, PREFIX + 'a')
    cluster.check_existing_cluster_temp_directory(PREFIX, 'tryremove')
    assert os.listdir(str(fake_tempdir)) == []


def test_unknown_behaviour_lists_valid_options(fake_tempdir):
    with pytest.raises(Exception, match='try_remove'):
        cluster.check_existing_cluster_temp_directory(PREFIX, 'nonsense')

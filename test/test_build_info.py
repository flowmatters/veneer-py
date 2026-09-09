"""Unit tests for the build identifier."""
import subprocess

import pytest

import veneer


def test_version_attribute_exists():
    assert isinstance(veneer.__version__, str)
    assert veneer.__version__


def test_build_info_reports_version_and_provenance():
    info = veneer.build_info()
    assert info['version'] == veneer.__version__
    assert info['source'] in ('git', 'distribution', 'unknown')


def test_build_info_includes_revision_key_even_when_unknown():
    """Consumers must be able to distinguish 'differs' from 'cannot tell',
    so the key is always present."""
    assert 'revision' in veneer.build_info()


@pytest.mark.parametrize('exc', [
    FileNotFoundError('git not found'),
    subprocess.CalledProcessError(128, ['git', 'rev-parse', 'HEAD']),
], ids=['no-git-binary', 'not-a-git-repo'])
def test_build_info_falls_through_when_git_is_unavailable(monkeypatch, exc):
    """Running the tests from inside a source checkout means git rev-parse
    always succeeds, so the distribution/unknown fallback branch of
    build_info() can never be exercised incidentally - it must be forced by
    making subprocess.check_output fail the way it would with no git binary
    on PATH (FileNotFoundError) or, the commoner real case, with git present
    but cwd outside any repo (CalledProcessError). Without this test,
    someone narrowing the except in the git branch would silently break the
    fallback."""
    def _raise(*args, **kwargs):
        raise exc

    monkeypatch.setattr(subprocess, 'check_output', _raise)

    info = veneer.build_info()
    assert info['revision'] is None
    assert info['source'] in ('distribution', 'unknown')


def test_build_info_reports_unknown_when_neither_git_nor_distribution_resolve(monkeypatch):
    """The 'unknown' branch is one level deeper than the git fallback: it
    additionally requires importlib.metadata to fail to find an installed
    'veneer-py' distribution. From inside a source checkout with the package
    pip-installed (editable or not), that never happens on its own - force
    both failures to prove the branch actually works, rather than trusting
    that it does because it reads plausibly."""
    from importlib.metadata import PackageNotFoundError

    def _raise_check_output(*args, **kwargs):
        raise FileNotFoundError('git not found')

    def _raise_dist_version(*args, **kwargs):
        raise PackageNotFoundError('veneer-py')

    monkeypatch.setattr(subprocess, 'check_output', _raise_check_output)
    monkeypatch.setattr('importlib.metadata.version', _raise_dist_version)

    info = veneer.build_info()
    assert info['source'] == 'unknown'
    assert info['revision'] is None
    assert info['version'] == veneer.__version__

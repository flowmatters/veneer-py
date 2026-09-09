__version__ = '0.2.0'


def build_info():
    '''
    Identify which build of veneer-py is in use.

    Returns a dict with:
      version  - the declared version (always __version__, regardless of source)
      revision - git SHA when running from a source checkout, else None
      source   - 'git', 'distribution', or 'unknown'; says HOW the revision was
                 obtained, so callers can tell "differs" from "cannot tell"

    Caveat: git rev-parse walks up from this file's directory to find any
    enclosing .git, so a veneer-py copy vendored or installed inside an
    unrelated repository (one without its own .git) will report that
    repository's SHA as its own revision.
    '''
    import os
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=here,
            stderr=subprocess.DEVNULL).decode().strip()
        return {'version': __version__, 'revision': sha, 'source': 'git'}
    except Exception:
        pass

    try:
        from importlib.metadata import version as _dist_version
        _dist_version('veneer-py')  # raises if no matching distribution is installed
        return {'version': __version__, 'revision': None, 'source': 'distribution'}
    except Exception:
        return {'version': __version__, 'revision': None, 'source': 'unknown'}


from .general import *


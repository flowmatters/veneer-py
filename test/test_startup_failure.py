"""start() must fail fast when VeneerCmd reports that Veneer could not start.

When SourceRESTfulService.Start() throws, Veneer logs "COULD NOT START VENEER"
(with a timestamp prefix) and the exception, but VeneerCmd keeps running and
never answers HTTP. Without recognising that line, start() waited out the full
startup_timeout and then raised a generic VeneerStartupTimeout.

These tests run a fake VeneerCmd (a Python script) that prints such output and
then stays alive, as the real one does.
"""
import subprocess
import sys
import time

import pytest

import veneer.manage as manage

pytestmark = pytest.mark.skipif(
    sys.platform != 'win32',
    reason='start() launches veneer_exe as a single command string, which needs Windows Popen',
)

FAKE_VENEER = '''
import sys, time
print("Project Loaded", flush=True)
print("[10:51:30 AM] {failure}", flush=True)
print("[10:51:30 AM] Some exception message from the web host", flush=True)
print("[10:51:30 AM]    at Some.Stack.Frame()", flush=True)
print("Server started. Ctrl-C to exit, or POST /shutdown command", flush=True)
time.sleep(120)
'''


@pytest.mark.parametrize('failure', [
    'COULD NOT START VENEER',
    'COULD NOT START VENEER ON PORT 9990',
])
def test_could_not_start_veneer_fails_fast_with_the_reason(tmp_path, failure):
    script = tmp_path / 'fake_veneer_cmd.py'
    script.write_text(FAKE_VENEER.format(failure=failure))
    veneer_exe = subprocess.list2cmdline([sys.executable, str(script)])

    began = time.monotonic()
    with pytest.raises(Exception) as excinfo:
        manage.start(veneer_exe=veneer_exe, ports=9990, quiet=True,
                     startup_timeout=20.0, return_log_paths=False)
    elapsed = time.monotonic() - began

    assert not isinstance(excinfo.value, manage.VeneerStartupTimeout)
    assert elapsed < 15.0
    message = str(excinfo.value)
    assert failure in message
    assert 'Some exception message from the web host' in message

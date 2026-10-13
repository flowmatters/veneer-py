"""start()'s GC decision, logged exactly once per call via logger.info (never print, so
the embedding application's own logging config decides where it ends up) -- requested so
an app log records which DOTNET_GC* variables (if any) were applied and why.

Tests below cover the pure formatter (_format_gc_decision_line) directly for every branch
shape, plus a handful of real start() calls (Popen/readiness mocked, like
test_gc_settings.py) with caplog, proving the line is actually wired in and reaches the
logger under the real call path. _logical_cpu_count is monkeypatched in the real-start()
tests so the CPU check (and the heap/CPU numbers in the log line) are deterministic
regardless of the real test host's core count.
"""
import logging
import queue
import psutil
import pytest
import veneer.manage as manage
from veneer.manage import _format_gc_decision_line


class _EmptyQueue:
    def get_nowait(self):
        raise queue.Empty


class _FakeVirtualMemory:
    def __init__(self, total):
        self.total = total


# --- Pure formatter: _format_gc_decision_line ---------------------------------------

def test_default_workstation_matches_the_requested_example_format():
    # The coordinator's own worked example: 4 instances x 112 heaps (one heap per logical
    # CPU) oversubscribes a 112-CPU host on its own, before RAM even enters it.
    line = _format_gc_decision_line(
        gc=None, n_instances=4, gc_vars={'DOTNET_gcServer': '0'},
        physical_ram_bytes=31.7 * 1024 ** 3, existing_gc_env={}, logical_cpus=112,
    )
    assert line == (
        "GC policy (gc=None, default): 4 instances × 112 heaps > 112 CPUs, "
        "31.7 GB RAM → 6.3 GB/instance < 16 GB → DOTNET_gcServer=0"
    )


def test_default_server_single_instance_ample_ram():
    # A 64 GB host, one instance: 64 GB / (1 + 1) = 32 GB/instance, at or above the
    # threshold -- server GC. The CPU check always holds for a single instance, whatever
    # the heap count is.
    line = _format_gc_decision_line(
        gc=None, n_instances=1, gc_vars={'DOTNET_gcServer': '1'},
        physical_ram_bytes=64 * 1024 ** 3, existing_gc_env={}, logical_cpus=16,
    )
    assert "(gc=None, default)" in line
    assert "1 instance ×" in line  # singular, not "1 instances"
    assert "16 heaps ≤ 16 CPUs" in line
    assert "≥ 16 GB" in line
    assert line.endswith("DOTNET_gcServer=1")


def test_default_workstation_single_instance_tight_ram():
    line = _format_gc_decision_line(
        gc=None, n_instances=1, gc_vars={'DOTNET_gcServer': '0'},
        physical_ram_bytes=1, existing_gc_env={}, logical_cpus=16,
    )
    assert "(gc=None, default)" in line
    assert "1 instance ×" in line
    assert "16 heaps ≤ 16 CPUs" in line
    assert "< 16 GB" in line
    assert line.endswith("DOTNET_gcServer=0")


def test_default_single_core_host_heap_word_is_singular():
    line = _format_gc_decision_line(
        gc=None, n_instances=1, gc_vars={'DOTNET_gcServer': '1'},
        physical_ram_bytes=64 * 1024 ** 3, existing_gc_env={}, logical_cpus=1,
    )
    assert "1 heap ≤ 1 CPUs" in line  # "heap", not "heaps" -- logical_cpus == 1


def test_default_workstation_multiple_instances_despite_ample_ram_due_to_cpu():
    # Realistic post-finding scenario: n_instances=4 against 112 heaps/CPUs already fails
    # the CPU check on its own, so even plentiful RAM (which alone would have said server)
    # doesn't flip this to server GC -- both reasons are shown, and the CPU one wins.
    line = _format_gc_decision_line(
        gc=None, n_instances=4, gc_vars={'DOTNET_gcServer': '0'},
        physical_ram_bytes=1000 * 1024 ** 3, existing_gc_env={}, logical_cpus=112,
    )
    assert "(gc=None, default)" in line
    assert "4 instances × 112 heaps > 112 CPUs" in line
    assert "≥ 16 GB" in line  # RAM alone would have said server...
    assert line.endswith("DOTNET_gcServer=0")  # ...but the CPU check still wins


def test_explicit_workstation():
    line = _format_gc_decision_line(
        gc='workstation', n_instances=1, gc_vars={'DOTNET_gcServer': '0'},
        physical_ram_bytes=64 * 1024 ** 3, existing_gc_env={}, logical_cpus=16,
    )
    assert "(gc='workstation', explicit)" in line
    assert line.endswith("DOTNET_gcServer=0")


def test_explicit_dynamic_sets_both_server_and_adaptation():
    line = _format_gc_decision_line(
        gc='dynamic', n_instances=1,
        gc_vars={'DOTNET_gcServer': '1', 'DOTNET_GCDynamicAdaptationMode': '1'},
        physical_ram_bytes=64 * 1024 ** 3, existing_gc_env={}, logical_cpus=16,
    )
    assert "(gc='dynamic', explicit)" in line
    assert line.endswith('DOTNET_gcServer=1, DOTNET_GCDynamicAdaptationMode=1')


def test_explicit_int_sets_server_and_heap_count():
    line = _format_gc_decision_line(
        gc=4, n_instances=1, gc_vars={'DOTNET_gcServer': '1', 'DOTNET_GCHeapCount': '4'},
        physical_ram_bytes=64 * 1024 ** 3, existing_gc_env={}, logical_cpus=16,
    )
    assert "(gc=4, explicit)" in line
    assert line.endswith('DOTNET_gcServer=1, DOTNET_GCHeapCount=4')


def test_explicit_false_is_bare_none():
    line = _format_gc_decision_line(
        gc=False, n_instances=1, gc_vars={},
        physical_ram_bytes=64 * 1024 ** 3, existing_gc_env={}, logical_cpus=16,
    )
    assert "(gc=False, explicit)" in line
    assert line.endswith("none")


def test_existing_env_vars_are_appended_as_a_replaces_suffix():
    line = _format_gc_decision_line(
        gc=None, n_instances=4, gc_vars={'DOTNET_gcServer': '0'},
        physical_ram_bytes=31.7 * 1024 ** 3, existing_gc_env={'DOTNET_gcServer': '1'},
        logical_cpus=112,
    )
    assert line.endswith("; replaces existing DOTNET_gcServer=1 from the environment")


def test_no_existing_env_vars_means_no_suffix_at_all():
    line = _format_gc_decision_line(
        gc=None, n_instances=4, gc_vars={'DOTNET_gcServer': '0'},
        physical_ram_bytes=31.7 * 1024 ** 3, existing_gc_env={}, logical_cpus=112,
    )
    assert 'replaces existing' not in line


def test_multiple_existing_env_vars_are_all_listed():
    line = _format_gc_decision_line(
        gc=None, n_instances=4, gc_vars={'DOTNET_gcServer': '0'},
        physical_ram_bytes=31.7 * 1024 ** 3,
        existing_gc_env={'DOTNET_gcServer': '1', 'DOTNET_GCHeapCount': '8'},
        logical_cpus=112,
    )
    assert 'DOTNET_gcServer=1' in line
    assert 'DOTNET_GCHeapCount=8' in line
    assert line.endswith('from the environment')


# --- _existing_gc_env_vars: case-insensitive DOTNET_GC*/DOTNET_gcServer scan ---------

def test_existing_gc_env_vars_matches_every_allowed_name(monkeypatch):
    monkeypatch.setenv('DOTNET_gcServer', '1')
    monkeypatch.setenv('DOTNET_GCHeapCount', '4')
    monkeypatch.setenv('DOTNET_GCDynamicAdaptationMode', '1')
    monkeypatch.setenv('DOTNET_GCHeapHardLimit', '0x10000000')
    monkeypatch.setenv('DOTNET_gcConcurrent', '0')
    monkeypatch.setenv('DOTNET_NOLOGO', '1')  # a DOTNET_ var that isn't GC-related at all
    found = manage._existing_gc_env_vars()
    assert found == {
        'DOTNET_gcServer': '1',
        'DOTNET_GCHeapCount': '4',
        'DOTNET_GCDynamicAdaptationMode': '1',
        'DOTNET_GCHeapHardLimit': '0x10000000',
        'DOTNET_gcConcurrent': '0',
    }


# --- Real start() calls (Popen/readiness mocked), with caplog -----------------------

def _patch_popen_and_readiness(monkeypatch):
    """Shared setup, same pattern as test_gc_settings.py's: fakes Popen (never spawns
    anything real) and makes the readiness check report "ready" on its first poll."""
    captured_calls = []

    class _FakeProc:
        stdout = None
        stderr = None
        stdin = None
        pid = 99999

        def poll(self):
            return None

        def kill(self):
            pass

        def wait(self, timeout=None):
            pass

    def fake_popen(cmd, **kwargs):
        captured_calls.append(kwargs)
        return _FakeProc()

    def fake_non_blocking_io(processes, *a, **k):
        n = len(processes)
        return [_EmptyQueue() for _ in range(n)], [None] * n

    monkeypatch.setattr(manage, 'Popen', fake_popen)
    monkeypatch.setattr(manage, 'configure_non_blocking_io', fake_non_blocking_io)
    monkeypatch.setattr(manage, '_probe_ready', lambda *a, **k: True)
    return captured_calls


def _start(monkeypatch, tmp_path, **kwargs):
    _patch_popen_and_readiness(monkeypatch)
    project = tmp_path / "model.rsproj"
    project.write_text("not a real project")
    manage.start(project_fn=str(project), veneer_exe="VeneerCmd.exe",
                 exit_on_stdin_eof=False, return_log_paths=True, **kwargs)


def test_start_logs_default_workstation_despite_ample_ram_multiple_instances(
        monkeypatch, tmp_path, caplog):
    # Real start() path: n_instances=4 against the host's own logical-CPU count already
    # fails the CPU check (oversubscription, the Fedora livelock finding) regardless of
    # how much RAM is available -- the case that would have picked server GC under the
    # old RAM-only rule.
    monkeypatch.setattr(manage, '_logical_cpu_count', lambda: 112)
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: _FakeVirtualMemory(1000 * 1024 ** 3))
    with caplog.at_level(logging.INFO, logger='veneer.manage'):
        _start(monkeypatch, tmp_path, n_instances=4)
    [line] = [r.message for r in caplog.records if r.message.startswith('GC policy')]
    assert '(gc=None, default)' in line
    assert '4 instances × 112 heaps > 112 CPUs' in line
    assert line.endswith('DOTNET_gcServer=0')


def test_start_logs_default_workstation_case(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(manage, '_logical_cpu_count', lambda: 1000)
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: _FakeVirtualMemory(32 * 1024 ** 3))
    with caplog.at_level(logging.INFO, logger='veneer.manage'):
        _start(monkeypatch, tmp_path, n_instances=4)
    [line] = [r.message for r in caplog.records if r.message.startswith('GC policy')]
    assert '(gc=None, default)' in line
    assert '4 instances' in line
    assert line.endswith('DOTNET_gcServer=0')


def test_start_logs_default_single_instance_workstation_case(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(manage, '_logical_cpu_count', lambda: 16)
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: _FakeVirtualMemory(1))
    with caplog.at_level(logging.INFO, logger='veneer.manage'):
        _start(monkeypatch, tmp_path, n_instances=1)
    [line] = [r.message for r in caplog.records if r.message.startswith('GC policy')]
    assert '(gc=None, default)' in line
    assert '1 instance ×' in line
    assert line.endswith('DOTNET_gcServer=0')


def test_start_logs_default_single_instance_server_case(monkeypatch, tmp_path, caplog):
    # On a 32 GB host, one instance: 32 GB / (1 + 1) = 16 GB/instance -- server GC. The
    # CPU check always holds for a single instance.
    monkeypatch.setattr(manage, '_logical_cpu_count', lambda: 16)
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: _FakeVirtualMemory(32 * 1024 ** 3))
    with caplog.at_level(logging.INFO, logger='veneer.manage'):
        _start(monkeypatch, tmp_path, n_instances=1)
    [line] = [r.message for r in caplog.records if r.message.startswith('GC policy')]
    assert '(gc=None, default)' in line
    assert '1 instance ×' in line
    assert line.endswith('DOTNET_gcServer=1')


def test_start_logs_explicit_gc_case(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(manage, '_logical_cpu_count', lambda: 16)
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: _FakeVirtualMemory(32 * 1024 ** 3))
    with caplog.at_level(logging.INFO, logger='veneer.manage'):
        _start(monkeypatch, tmp_path, n_instances=1, gc=4)
    [line] = [r.message for r in caplog.records if r.message.startswith('GC policy')]
    assert '(gc=4, explicit)' in line
    assert line.endswith('DOTNET_gcServer=1, DOTNET_GCHeapCount=4')


def test_start_logs_explicit_dynamic_case(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(manage, '_logical_cpu_count', lambda: 16)
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: _FakeVirtualMemory(1))
    with caplog.at_level(logging.INFO, logger='veneer.manage'):
        _start(monkeypatch, tmp_path, n_instances=1, gc='dynamic')
    [line] = [r.message for r in caplog.records if r.message.startswith('GC policy')]
    assert "(gc='dynamic', explicit)" in line
    # Forced unconditionally -- RAM is tight enough here that the default policy would
    # have picked workstation, but gc='dynamic' is an explicit override.
    assert line.endswith('DOTNET_gcServer=1, DOTNET_GCDynamicAdaptationMode=1')


def test_start_logs_pre_existing_env_vars_case(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv('DOTNET_gcServer', '1')
    monkeypatch.setattr(manage, '_logical_cpu_count', lambda: 1000)
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: _FakeVirtualMemory(32 * 1024 ** 3))
    with caplog.at_level(logging.INFO, logger='veneer.manage'):
        _start(monkeypatch, tmp_path, n_instances=4)
    [line] = [r.message for r in caplog.records if r.message.startswith('GC policy')]
    assert line.endswith('; replaces existing DOTNET_gcServer=1 from the environment')


def test_start_logs_exactly_one_gc_policy_line(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(manage, '_logical_cpu_count', lambda: 1000)
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: _FakeVirtualMemory(32 * 1024 ** 3))
    with caplog.at_level(logging.INFO, logger='veneer.manage'):
        _start(monkeypatch, tmp_path, n_instances=3)
    gc_lines = [r.message for r in caplog.records if r.message.startswith('GC policy')]
    assert len(gc_lines) == 1


# --- The coordinator's actual question: can the literal arrow/>=/CPU glyphs ever raise? --

def test_gc_decision_log_line_survives_a_cp1252_console_handler():
    """Direct evidence, not an argument from reading the stdlib source: a console handler
    that cannot encode the literal U+2192 arrow, U+2265 (>=) or U+2264 (<=) (e.g. a real
    Windows cp1252 console -- cp1252 has no mapping for either comparator, unlike U+00D7
    (x), which cp1252 maps at byte 0xD7 and so was never actually at risk) must not crash
    the caller. logging.Handler.emit() wraps the actual stream write in its own
    try/except and routes any failure to handleError(), which never re-raises regardless
    of logging.raiseExceptions -- demonstrated here against a real StreamHandler writing
    through a cp1252-encoded TextIOWrapper, with a message that contains both
    comparators (the ample-RAM/CPU-fits branch uses >= and <=; every branch uses ->)."""
    import io

    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding='cp1252', errors='strict', write_through=True)
    handler = logging.StreamHandler(stream)
    test_logger = logging.getLogger('veneer-py-test-cp1252-safety')
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.INFO)
    old_raise = logging.raiseExceptions
    # Quiets handleError's diagnostic stderr print for a clean test run; does not change
    # whether it re-raises -- it never does, in either branch.
    logging.raiseExceptions = False
    try:
        line = _format_gc_decision_line(
            None, 1, {'DOTNET_gcServer': '1'}, 64 * 1024 ** 3, {}, 16
        )
        assert '≥' in line  # RAM comparator
        assert '≤' in line  # CPU comparator
        assert '×' in line  # instances-x-heaps
        test_logger.info(line)
    finally:
        logging.raiseExceptions = old_raise
        test_logger.removeHandler(handler)
    # Reaching this line at all is the assertion: an unhandled UnicodeEncodeError from
    # stream.write() inside emit() would have propagated out of test_logger.info() above.

"""Under `--output json`, stdout must parse as JSON -- whatever delivers it.

`test_machine_readable_output_stream.py` next door pins the mechanism this
branch uses: that `route_logs_to_stderr()` moves the handler and the default
stream, that the move survives `reload_logger()`, that the callback rides on
the shared decorator. Those are worth keeping -- the first two are regressions
that actually happened while the fix was being written -- but they are
assertions about one implementation. Run them against a different fix for the
same bug and they fail with `AttributeError`, which says "this is not my
implementation" rather than "your fix leaves a hole".

This file asserts the contract instead:

    under `--output json`, everything a command writes to stdout parses as
    JSON.

That is what a programmatic consumer actually depends on, and it is checkable
against any implementation. Porting it to one means editing a single fixture
body -- `activate_json_mode` below, the only implementation-specific line here
-- and nothing else. It can be pointed at skypilot-org/skypilot#10763, or at
any other candidate, as is.

Four mechanisms can put text on the client's stdout ahead of the payload, and
they are not equally covered:

    emitter                                 #10763   this branch
    relayed server line (sdk.py:206)          yes        yes
    client-side logger.info                    no        yes
    sky_logging.print                          no        yes
    WSL SSH-config notice (bare print)         no         no

The client-side logger is the discriminating case. `sky status` happens to log
nothing client-side, which is why the original report only exposed the relay
path and why a relay-only fix looks sufficient when that is the only command
tested. `sky queue` logs 'Fetching job queue for: ...' at
`sky/client/cli/command.py:2648` under the same `--output` option (`:2632`),
and there the two diverge.

The WSL notice is a bare `print` that no fix on the table reaches; it is marked
`xfail(strict=True)` rather than fixed here, so that closing it forces an edit
to this file instead of passing silently.
"""
# Both disables are the point of the file rather than incidental to it. The
# contract is reached through SkyPilot's own private entry points -- the
# --output option callback, the _logs_to_stderr flag it sets, and the WSL
# classmethod that owns the bare print -- and a public route to any of them
# would not be the thing a CLI invocation actually takes. pytest fixtures
# shadow their own names by construction.
# pylint: disable=protected-access,redefined-outer-name
import builtins
import json
import logging

import pytest

from sky import sky_logging
from sky.client import sdk
from sky.client.cli import flags
from sky.utils import cluster_utils


@pytest.fixture
def activate_json_mode():
    """Turn on machine-readable output.

    The reload first: the default handler is built at import time and
    `_setup_logger()` will not rebuild an existing one, so without this the
    test inherits whatever stream the handler was constructed with in some
    earlier test.

    The activation itself is THE ONLY IMPLEMENTATION-SPECIFIC LINE IN THIS
    FILE. Swap it for whatever your fix uses as an entry point; everything
    below asserts the contract, not the mechanism.
    """
    sky_logging.reload_logger()
    flags._keep_stdout_for_the_payload(None, None, flags.OUTPUT_FORMAT_JSON)
    yield
    sky_logging._logs_to_stderr = False
    sky_logging.print = builtins.print
    sky_logging.reload_logger()


class _OneLineResponse:
    """The minimum `decode_rich_status()` needs of a `requests.Response`.

    Only the HTTP response is stubbed; the print that puts the line on the
    client's stdout is the real one in `sdk.stream_response()`.
    """

    def __init__(self, line: str):
        self._line = line.encode('utf-8')

    def iter_content(self, chunk_size=None):
        del chunk_size  # unused; decode_rich_status passes chunk_size=None
        yield self._line


def _emit_relayed_server_line(tmp_path, monkeypatch):
    """A line the API server logged, relayed and printed client-side.

    This is the `Cluster(s) not found` case from the original report: produced
    by `logger.info` in the server, so redirecting the client's own logger does
    nothing for it, and printed at `sky/client/sdk.py:206`.
    """
    del tmp_path, monkeypatch  # unused; the signature is shared by all four
    sdk.stream_response(
        request_id=None,
        response=_OneLineResponse('Cluster(s) not found: \x1b[1mc\x1b[0m.\n'),
        get_result=False,
    )


def _emit_client_logger_info(tmp_path, monkeypatch):
    """The client's own logger, which never touches the wire.

    The real instance is `sky/client/cli/command.py:2648`, in `sky queue`,
    which carries `@flags.output_format_option()` at `:2632`. The real logger
    name is used so the record travels the real handler hierarchy ('sky' is the
    root, propagate=False).
    """
    del tmp_path, monkeypatch  # unused
    logging.getLogger('sky.client.cli.command').info(
        'Fetching job queue for: c')


def _emit_sky_logging_print(tmp_path, monkeypatch):
    """Library code that prints rather than logs."""
    del tmp_path, monkeypatch  # unused
    # Attribute lookup at call time, because route_logs_to_stderr() rebinds the
    # module global; `from sky.sky_logging import print` would capture the
    # pre-redirect builtin.
    sky_logging.print('prose')


def _emit_wsl_notice(tmp_path, monkeypatch):
    """The WSL SSH-config notice: a builtin `print` with no `file=`.

    `sky/utils/cluster_utils.py:386`, reached through the real classmethod
    rather than reproduced. The gate is `get_wsl_windows_home()`, which returns
    None unless `common_utils.is_wsl()` and a Windows home is resolvable;
    patching it is what puts a non-WSL machine on the WSL branch. The notice
    fires only once `codegen` is non-empty and the Windows config has been
    written, so the whole branch has to run.
    """
    windows_home = tmp_path / 'windows-home'
    (windows_home / '.ssh').mkdir(parents=True)
    key_path = tmp_path / 'sky-key.pem'
    key_path.write_text('not-a-real-key\n', encoding='utf-8')

    monkeypatch.setattr(cluster_utils, 'get_wsl_windows_home',
                        lambda: str(windows_home))
    # A class attribute that latches after the first notice, so a second test
    # in the same process would emit nothing and pass for the wrong reason.
    monkeypatch.setattr(cluster_utils.SSHConfigHelper,
                        '_windows_ssh_setup_warned', False)

    cluster_utils.SSHConfigHelper._add_cluster_to_windows_ssh_config(
        cluster_name='c',
        cluster_name_on_cloud='c-abcdef',
        ips=['1.2.3.4'],
        username='ubuntu',
        key_path=str(key_path),
        ports=[22],
        proxy_command=None,
        uses_docker=False,
    )

    # The enclosing method swallows OSError/PermissionError, so a branch that
    # never reached the print would leave stdout clean and this case would
    # xpass -- reporting a fixed bug that is merely untested. The flag is set
    # immediately before the print.
    assert cluster_utils.SSHConfigHelper._windows_ssh_setup_warned, (
        'the WSL branch was not reached, so the notice was never emitted and '
        'this test would pass for the wrong reason')


@pytest.mark.parametrize('emit', [
    _emit_relayed_server_line,
    _emit_client_logger_info,
    _emit_sky_logging_print,
    pytest.param(_emit_wsl_notice,
                 marks=pytest.mark.xfail(
                     strict=True,
                     reason='cluster_utils.py:386 is a builtin print with no '
                     'file=; no redirect in sky_logging or sdk reaches it. '
                     'Fixing it needs either deletion or contextlib.'
                     'redirect_stdout around the command body.')),
])
def test_stdout_carries_only_the_payload(activate_json_mode, emit, capfd,
                                         tmp_path, monkeypatch):
    """Whatever the emitter put on a stream, stdout still parses as JSON.

    `capfd`, not `capsys`, and the difference is load-bearing. A logging
    handler stores a stream OBJECT, captured when the handler is built, which
    is not necessarily the `sys.stdout` in place when the test body runs;
    `capsys` reads the object and so cannot see what a handler holding an older
    one writes. Measured: with `capsys` and `setStream()` commented out of
    `route_logs_to_stderr()`, prose landed on fd 1 and this test still passed
    -- a silent false negative in the one case that discriminates between the
    candidate fixes. `capfd` captures the descriptor, so it sees every write to
    stdout no matter which object made it, which is also what a program
    consuming the output actually reads.
    """
    del activate_json_mode  # fixture, requested for its side effect
    emit(tmp_path, monkeypatch)
    print(json.dumps([]))  # the payload the command would emit
    stdout = capfd.readouterr().out
    try:
        json.loads(stdout)
    except json.JSONDecodeError as e:
        pytest.fail(f'stdout under --output json did not parse as JSON '
                    f'({e}); it carried: {stdout!r}')

"""Under `--output json`, stdout carries only the payload.

SkyPilot's default log handler writes to stdout, and the client relays server-side request
logs there too. So `sky status -o json <unknown>` emitted

    Cluster(s) not found: \x1b[1m<unknown>\x1b[0m.
    []

on stdout -- ANSI escapes included -- and json.loads() on it failed, while stderr stayed
empty. Exit codes already separate the cases a consumer cares about (0 with an empty array
for "no such cluster", non-zero for "the query failed"), so a clean stdout was the only
thing missing; without it, callers fall back to scraping the human table, where those two
outcomes are indistinguishable.

Two mechanisms have to agree, and each broke separately while this was being written:

* the log handler, which reload_logger() rebuilds -- a redirect stored on the handler
  instance is silently undone the next time that happens, so the stream is process state;
* the client's relay of server-side lines, which is NOT the client's logger at all. The
  text is produced in the API server and printed by sdk.stream_response(), so redirecting
  the client's handler does nothing for it.
"""
import io
import sys

import click

from sky import sky_logging
from sky.client.cli import flags


def _restore():
    sky_logging._logs_to_stderr = False
    sky_logging.print = print
    sky_logging.reload_logger()


def test_route_logs_to_stderr_moves_the_handler_and_the_default_stream():
    try:
        sky_logging.route_logs_to_stderr()
        assert sky_logging.default_stream() is sys.stderr
        assert sky_logging._default_handler.stream is sys.stderr
    finally:
        _restore()


def test_the_redirect_survives_reload_logger():
    """reload_logger() drops the handler and _setup_logger() builds a new one.

    The first version of this fix set the stream on the handler instance, so the next
    reload -- and several code paths call it -- put logging back on stdout.
    """
    try:
        sky_logging.route_logs_to_stderr()
        sky_logging.reload_logger()
        assert sky_logging._default_handler.stream is sys.stderr, (
            "reload_logger() put logging back on the payload stream")
    finally:
        _restore()


def test_sky_logging_print_follows():
    """Library code prints through sky_logging.print, which must move with the rest."""
    try:
        sky_logging.route_logs_to_stderr()
        captured = io.StringIO()
        stderr, sys.stderr = sys.stderr, captured
        try:
            sky_logging.print('prose')
        finally:
            sys.stderr = stderr
        assert captured.getvalue() == 'prose\n'
    finally:
        _restore()


def test_the_option_callback_fires_only_for_a_machine_readable_format():
    """The redirect hangs off the shared --output option, so every command carrying it is
    covered rather than whichever one last needed it. A click option callback runs during
    parameter parsing, before any command body can log."""
    try:
        assert sky_logging.default_stream() is sys.stdout
        flags._keep_stdout_for_the_payload(None, None,
                                           flags.OUTPUT_FORMAT_TABLE)
        assert sky_logging.default_stream(
        ) is sys.stdout, 'table output was redirected'
        flags._keep_stdout_for_the_payload(None, None, flags.OUTPUT_FORMAT_JSON)
        assert sky_logging.default_stream() is sys.stderr
    finally:
        _restore()


def test_every_output_format_option_carries_the_callback():
    """Guards the wiring: the fix is in the decorator so it cannot be added per-command and
    forgotten for the next one."""

    @flags.output_format_option()
    def _cmd(output_format):
        return output_format

    params = [p for p in _cmd.__click_params__ if isinstance(p, click.Option)]
    assert params, 'the decorator stopped attaching an option'
    assert params[0].callback is flags._keep_stdout_for_the_payload

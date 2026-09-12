"""The command surface a user meets first."""

from __future__ import annotations

from importlib.metadata import version

from typer.testing import CliRunner

from averta.cli import app

runner = CliRunner()


class TestVersion:
    def test_reports_the_installed_version(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert version("averta") in result.stdout

    def test_adding_the_callback_did_not_swallow_subcommands(self):
        # A Typer callback with a required-looking parameter can shadow every
        # command below it, so this asserts the group still dispatches.
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for command in ("ingest", "train", "site", "score", "validate"):
            assert command in result.stdout


class TestHelp:
    def test_every_command_has_a_help_line(self):
        listing = runner.invoke(app, ["--help"]).stdout
        for command in ("ingest", "features", "report", "train", "diagnose"):
            assert command in listing
            detail = runner.invoke(app, [command, "--help"])
            assert detail.exit_code == 0, f"{command} --help failed"

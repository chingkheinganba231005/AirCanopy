"""The package installs, imports and runs its command."""

import pytest

import aircanopy
from aircanopy import cli


def test_version_is_set():
    assert aircanopy.__version__ == "0.1.0"


def test_command_prints_help_without_arguments(capsys):
    assert cli.main([]) == 0
    assert "aircanopy" in capsys.readouterr().out


def test_command_reports_its_version(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["--version"])
    assert stop.value.code == 0
    assert "aircanopy 0.1.0" in capsys.readouterr().out

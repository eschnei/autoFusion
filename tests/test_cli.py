"""CLI surface: help works, errors are friendly not tracebacks (MAR-21)."""

import pytest

from autofusion.cli import main

SUBCOMMANDS = ["init", "config-check", "smoke", "fuse", "eval", "budget", "serve"]


@pytest.mark.parametrize("cmd", SUBCOMMANDS)
def test_each_subcommand_help_exits_zero(cmd):
    with pytest.raises(SystemExit) as e:
        main([cmd, "--help"])
    assert e.value.code == 0


def test_unknown_model_is_friendly_error_not_traceback(capsys):
    # smoke resolves the model before any network call; unknown -> friendly error.
    rc = main(["-c", "configs/local-plus-frontier.toml", "smoke", "-m", "does-not-exist"])
    err = capsys.readouterr().err
    assert rc == 2
    assert err.startswith("error:")
    assert "does-not-exist" in err and "Configured" in err  # lists valid options


def test_missing_config_is_friendly_error(capsys):
    rc = main(["-c", "/no/such/autofusion.toml", "config-check"])
    err = capsys.readouterr().err
    assert rc == 2 and err.startswith("error:")

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


def test_spend_command_handles_budget_cap_gracefully(monkeypatch, capsys):
    # A budget cap hit mid-run must stop cleanly (exit 2 + message), not traceback.
    import autofusion.eval.runner as runner
    from autofusion.budget import BudgetExceeded

    async def boom(*a, **k):
        raise BudgetExceeded("projected $1.00 would push spend past the $25.00 cap")

    monkeypatch.setattr(runner, "run_baseline", boom)
    rc = main(["eval", "-m", "llama3.2", "-n", "1"])
    err = capsys.readouterr().err
    assert rc == 2 and "budget cap reached" in err

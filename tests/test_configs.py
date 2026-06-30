"""The shipped local+frontier profile parses and degrades gracefully (MAR-20)."""

from pathlib import Path

from autofusion.cli import _cmd_config_check
from autofusion.config import load_config
from autofusion.strategies import Fusion, resolve_strategy

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
PROFILE = str(CONFIGS / "local-plus-frontier.toml")
FRONTIER = str(CONFIGS / "frontier-bench.toml")


def test_profile_builds_local_proposers_hosted_aggregator():
    cfg = load_config(PROFILE)
    strat = resolve_strategy(cfg, "fusion")
    assert isinstance(strat, Fusion)
    assert all(p.is_local for p in strat.proposers)        # local drafts
    assert not strat.aggregator.is_local                    # one hosted aggregator
    assert strat.aggregator.name == "gpt-4o-mini"


def test_frontier_bench_profile_parses_and_resolves():
    cfg = load_config(FRONTIER)
    for m in ("gpt-4o", "claude-sonnet", "gemini"):
        assert m in cfg.models and not cfg.models[m].is_local   # frontier baselines
    strat = resolve_strategy(cfg, "fusion")
    assert isinstance(strat, Fusion)
    assert all(p.is_local for p in strat.proposers)             # free local drafts
    assert strat.aggregator.name == "gpt-4o"                     # frontier aggregator


def test_config_check_reports_missing_key_without_crashing(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import argparse
    rc = _cmd_config_check(argparse.Namespace(config=PROFILE))
    out = capsys.readouterr().out
    assert rc == 0                                          # no crash
    assert "MISSING OPENAI_API_KEY" in out

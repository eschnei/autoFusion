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
    for m in ("opus", "sonnet", "haiku", "gpt-4o", "deepseek", "qwen-72b"):
        assert m in cfg.models and not cfg.models[m].is_local   # hosted basket
    strat = resolve_strategy(cfg, "fusion")
    assert isinstance(strat, Fusion)
    # diverse strong-open proposers, synthesized by the strongest model
    assert [p.name for p in strat.proposers] == ["deepseek", "qwen-72b", "llama-70b"]
    assert strat.aggregator.name == "opus"
    assert cfg.budget.total_usd == 25.0                          # hard cap wired in


def test_config_check_reports_missing_key_without_crashing(monkeypatch, capsys):
    # Neutralize .env loading so the test is deterministic regardless of local keys.
    monkeypatch.setattr("autofusion.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import argparse
    rc = _cmd_config_check(argparse.Namespace(config=PROFILE))
    out = capsys.readouterr().out
    assert rc == 0                                          # no crash
    assert "MISSING OPENAI_API_KEY" in out

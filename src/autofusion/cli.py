"""autoFusion CLI (Phase 0/1 surface).

Commands:
  config-check          show configured models + which provider keys are present
  smoke   --model M     call one model end-to-end (Phase 0 gate)
  fuse    "prompt"      run fusion (MoA) on one prompt (Phase 2)
  route   "prompt"      run the router (picks one model) on one prompt (Phase 6)
  cascade "prompt"      cheap->critic->escalate cost cascade on one prompt (Phase 7)
  eval    --models a,b  run a benchmark baseline / fusion / route / cascade -> leaderboard
  budget  status        show the configured budget caps
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from .budget import BudgetExceeded, BudgetTracker
from .config import load_config
from .providers import complete
from .strategies import resolve_strategy


_KEY_ENV = {
    "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY",
}


def _print_model_table(cfg) -> None:
    """Render the model registry + per-model key/endpoint status."""
    print(f"{'model':<16}{'litellm id':<26}{'local':>7}{'key/endpoint':>20}")
    print("-" * 69)
    for spec in cfg.models.values():
        if spec.is_local:
            status = spec.api_base or "ollama"
        else:
            provider = spec.model.split("/")[0] if "/" in spec.model else "openai"
            env = _KEY_ENV.get(provider, f"{provider.upper()}_API_KEY")
            status = "OK" if os.environ.get(env) else f"MISSING {env}"
        print(f"{spec.name:<16}{spec.model:<26}{'yes' if spec.is_local else 'no':>7}{status:>20}")


def _cmd_config_check(args) -> int:
    cfg = load_config(args.config)
    print(f"config: {cfg.path}\n")
    _print_model_table(cfg)
    return 0


def _cmd_init(args) -> int:
    from pathlib import Path

    from .config import DEFAULT_CONFIG_NAME, STARTER_CONFIG, load_config

    target = Path(args.config) if args.config else Path(DEFAULT_CONFIG_NAME)
    if target.exists() and not args.force:
        print(f"{target} already exists — not overwriting (use --force to replace).")
        return 0
    target.write_text(STARTER_CONFIG)
    print(f"wrote {target}\n")
    _print_model_table(load_config(target))
    print(
        "\nNext steps:\n"
        "  • local $0 path: `ollama serve` then `ollama pull llama3.2` (and qwen2.5:3b)\n"
        "  • hosted models: copy .env.example to .env and add a provider key\n"
        "  • then: `autofusion smoke -m llama3.2`  or  `autofusion fuse \"...\"`"
    )
    return 0


def _cmd_smoke(args) -> int:
    cfg = load_config(args.config)
    spec = cfg.model(args.model)
    print(f"-> {spec.name} ({spec.model})\n")
    result = complete(spec, [{"role": "user", "content": args.prompt}])
    if not result.ok:
        print(f"ERROR: {result.error}", file=sys.stderr)
        return 1
    print(result.text)
    print(
        f"\n[{result.latency_s:.2f}s | "
        f"{result.prompt_tokens}+{result.completion_tokens} tok | "
        f"${result.cost_usd:.6f}]"
    )
    return 0


def _cmd_budget(args) -> int:
    cfg = load_config(args.config)
    print(BudgetTracker.from_config(cfg.budget).status_line())
    return 0


def _cmd_fuse(args) -> int:
    cfg = load_config(args.config)
    strategy = resolve_strategy(cfg, "fusion")
    budget = BudgetTracker.from_config(cfg.budget)
    props = ", ".join(p.name for p in strategy.proposers)
    print(f"-> fusion | proposers: {props} | aggregator: {strategy.aggregator.name} "
          f"| layers: {strategy.layers}\n")
    try:
        result = asyncio.run(
            strategy.run([{"role": "user", "content": args.prompt}], budget=budget)
        )
    except BudgetExceeded as exc:
        print(f"budget cap hit: {exc}", file=sys.stderr)
        return 2
    if not result.ok:
        print(f"ERROR: {result.error}", file=sys.stderr)
        return 1
    print(result.text)
    print(f"\n[{result.latency_s:.2f}s | {result.n_calls} calls | ${result.cost_usd:.6f}]")
    return 0


def _cmd_route(args) -> int:
    cfg = load_config(args.config)
    strategy = resolve_strategy(cfg, "route")
    budget = BudgetTracker.from_config(cfg.budget)
    try:
        result = asyncio.run(
            strategy.run([{"role": "user", "content": args.prompt}], budget=budget)
        )
    except BudgetExceeded as exc:
        print(f"budget cap hit: {exc}", file=sys.stderr)
        return 2
    if not result.ok:
        print(f"ERROR: {result.error}", file=sys.stderr)
        return 1
    print(f"-> route selected: {result.model}\n")
    print(result.text)
    print(f"\n[{result.latency_s:.2f}s | ${result.cost_usd:.6f}]")
    return 0


def _cmd_cascade(args) -> int:
    cfg = load_config(args.config)
    strategy = resolve_strategy(cfg, "cascade")
    budget = BudgetTracker.from_config(cfg.budget)
    tiers = ", ".join(t.name for t in strategy.tiers)
    print(f"-> cascade | tiers: {tiers} | critic: {strategy.critic.name} "
          f"| threshold: {strategy.threshold}\n")
    try:
        result = asyncio.run(
            strategy.run([{"role": "user", "content": args.prompt}], budget=budget)
        )
    except BudgetExceeded as exc:
        print(f"budget cap hit: {exc}", file=sys.stderr)
        return 2
    if not result.ok:
        print(f"ERROR: {result.error}", file=sys.stderr)
        return 1
    print(result.text)
    print(f"\n[{result.latency_s:.2f}s | {result.n_calls} calls | ${result.cost_usd:.6f}]")
    return 0


def _cmd_serve(args) -> int:
    import uvicorn

    from .server import create_app

    cfg = load_config(args.config)
    print(f"serving autoFusion on http://{args.host}:{args.port}  (POST /v1/chat/completions)")
    uvicorn.run(create_app(cfg), host=args.host, port=args.port, log_level="info")
    return 0


def _cmd_eval(args) -> int:
    from .eval.results import render_leaderboard, save_run
    from .eval.runner import run_baseline

    cfg = load_config(args.config)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"benchmark: {args.benchmark} | models: {', '.join(models)} | limit: {args.limit}\n")
    results = asyncio.run(
        run_baseline(cfg, models, args.benchmark, limit=args.limit, concurrency=args.concurrency)
    )
    print(render_leaderboard(results))
    path = save_run(results)
    print(f"\nsaved: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autofusion", description=__doc__)
    parser.add_argument("-c", "--config", help="path to autofusion.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="scaffold autofusion.toml + check keys")
    p_init.add_argument("-f", "--force", action="store_true", help="overwrite an existing config")

    sub.add_parser("config-check", help="show configured models + key status")

    p_smoke = sub.add_parser("smoke", help="call one model end-to-end")
    p_smoke.add_argument("-m", "--model", required=True, help="model name from config")
    p_smoke.add_argument("-p", "--prompt", default="In one sentence, what is a mixture-of-agents?")

    p_fuse = sub.add_parser("fuse", help="run fusion (MoA) on one prompt")
    p_fuse.add_argument("prompt", help="the prompt to fuse")

    p_route = sub.add_parser("route", help="route one prompt to a single model")
    p_route.add_argument("prompt", help="the prompt to route")

    p_cascade = sub.add_parser("cascade", help="cheap->critic->escalate cost cascade")
    p_cascade.add_argument("prompt", help="the prompt to run through the cascade")

    p_budget = sub.add_parser("budget", help="budget caps")
    p_budget.add_argument("action", choices=["status"], help="what to show")

    p_serve = sub.add_parser("serve", help="OpenAI-compatible HTTP endpoint")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    p_eval = sub.add_parser("eval", help="run a benchmark baseline or fusion")
    p_eval.add_argument(
        "-m", "--models", required=True,
        help="comma-separated strategy names (configured models and/or 'fusion')",
    )
    p_eval.add_argument("-b", "--benchmark", default="humaneval")
    p_eval.add_argument("-n", "--limit", type=int, default=None, help="limit number of tasks")
    p_eval.add_argument("--concurrency", type=int, default=4)

    args = parser.parse_args(argv)
    handlers = {
        "init": _cmd_init, "config-check": _cmd_config_check, "smoke": _cmd_smoke,
        "fuse": _cmd_fuse, "route": _cmd_route, "cascade": _cmd_cascade,
        "eval": _cmd_eval, "budget": _cmd_budget, "serve": _cmd_serve,
    }
    try:
        return handlers[args.command](args)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        # Friendly one-liner instead of a traceback (unknown model/strategy lists
        # valid options; missing/!parseable config; bad fusion config).
        msg = exc.args[0] if exc.args else str(exc)
        print(f"error: {msg}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

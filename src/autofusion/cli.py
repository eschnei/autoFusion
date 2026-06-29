"""autoFusion CLI (Phase 0/1 surface).

Commands:
  config-check          show configured models + which provider keys are present
  smoke   --model M     call one model end-to-end (Phase 0 gate)
  eval    --models a,b  run a benchmark baseline -> leaderboard (Phase 1 gate)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from .config import load_config
from .providers import complete


def _cmd_config_check(args) -> int:
    cfg = load_config(args.config)
    print(f"config: {cfg.path}\n")
    print(f"{'model':<16}{'litellm id':<26}{'local':>7}{'key/endpoint':>20}")
    print("-" * 69)
    key_env = {
        "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY",
    }
    for spec in cfg.models.values():
        if spec.is_local:
            status = spec.api_base or "ollama"
        else:
            provider = spec.model.split("/")[0] if "/" in spec.model else "openai"
            env = key_env.get(provider, f"{provider.upper()}_API_KEY")
            status = "OK" if os.environ.get(env) else f"MISSING {env}"
        print(f"{spec.name:<16}{spec.model:<26}{'yes' if spec.is_local else 'no':>7}{status:>20}")
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

    sub.add_parser("config-check", help="show configured models + key status")

    p_smoke = sub.add_parser("smoke", help="call one model end-to-end")
    p_smoke.add_argument("-m", "--model", required=True, help="model name from config")
    p_smoke.add_argument("-p", "--prompt", default="In one sentence, what is a mixture-of-agents?")

    p_eval = sub.add_parser("eval", help="run a benchmark baseline")
    p_eval.add_argument("-m", "--models", required=True, help="comma-separated model names")
    p_eval.add_argument("-b", "--benchmark", default="humaneval")
    p_eval.add_argument("-n", "--limit", type=int, default=None, help="limit number of tasks")
    p_eval.add_argument("--concurrency", type=int, default=4)

    args = parser.parse_args(argv)
    handlers = {"config-check": _cmd_config_check, "smoke": _cmd_smoke, "eval": _cmd_eval}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

"""autoFusion CLI (Phase 0/1 surface).

Commands:
  config-check          show configured models + which provider keys are present
  registry              show each model's capabilities + cost ($/Mtok) + context
  smoke   --model M     call one model end-to-end (Phase 0 gate)
  fuse    "prompt"      run fusion (MoA) on one prompt (Phase 2)
  route   "prompt"      run the router (picks one model) on one prompt (Phase 6)
  cascade "prompt"      cheap->critic->escalate cost cascade on one prompt (Phase 7)
  bestofn "prompt"      sample N candidates, a verifier/critic picks the best (Phase 9)
  code    "task"        write a solution file, verified by your test command (Phase 11)
  agent   "task"        run a coding agent (read/edit/run) in a repo (Phase A)
  eval    --models a,b  run a benchmark baseline / fusion / route / cascade -> leaderboard
  optimize --benchmark  sweep recipes over available models -> quality×cost frontier
  report  --benchmarks  recipes vs all available models across tasks -> scoreboard
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


def _cmd_registry(args) -> int:
    from .registry import build_registry

    cfg = load_config(args.config)
    reg = build_registry(cfg)
    print(f"{'model':<16}{'local':>6}{'$/Mtok in':>11}{'$/Mtok out':>12}"
          f"{'context':>10}  capabilities")
    print("-" * 78)
    for p in reg.values():
        ctx = f"{p.context_window:,}" if p.context_window else "?"
        caps = ", ".join(p.capabilities) or "—"
        print(f"{p.name:<16}{'yes' if p.is_local else 'no':>6}"
              f"{p.input_cost_per_mtok:>11.2f}{p.output_cost_per_mtok:>12.2f}{ctx:>10}  {caps}")
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


def _cmd_bestofn(args) -> int:
    cfg = load_config(args.config)
    strategy = resolve_strategy(cfg, "bestofn")
    budget = BudgetTracker.from_config(cfg.budget)
    basket = ", ".join(m.name for m in strategy.models)
    critic = strategy.critic.name if strategy.critic else "none"
    print(f"-> best-of-{strategy.n} | basket: {basket} | critic: {critic} "
          f"(ad-hoc: no test verifier, critic picks)\n")
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


def _cmd_agent(args) -> int:
    from .agent import Workspace, best_of_n_agents, run_agent

    cfg = load_config(args.config)
    budget = BudgetTracker.from_config(cfg.budget)

    if args.best_of > 1:  # Phase B — best-of-N trajectories, verified by tests
        if not args.tests:
            print("error: --best-of needs --tests (the verifier that picks the winner)",
                  file=sys.stderr)
            return 2
        names = cfg.code.models or cfg.bestofn.models
        if not names:
            print("error: no coding basket — set [code].models or [bestofn].models", file=sys.stderr)
            return 2
        ws = Workspace(args.repo)
        print(f"-> agent best-of-{args.best_of} | basket: {', '.join(names)} | repo: {ws.root} "
              f"| verify: {args.tests}\n  (N isolated trajectories; the repo's tests pick the winner)\n")
        res = asyncio.run(best_of_n_agents(
            [cfg.model(m) for m in names], args.task, ws.root, args.tests, args.best_of,
            budget=budget, max_steps=args.max_steps))
        for t in res.trajectories:
            mark = "PASS ✓" if t.passed else "fail  "
            print(f"  {t.model:<16}{mark}  ({t.result.steps} steps, ${t.result.cost_usd:.4f})")
        if res.winner:
            print(f"\napplied winner: {res.winner.model}  |  tests PASS ✓  |  total ${res.total_cost:.4f}")
        else:
            print(f"\nno trajectory passed — repo unchanged  |  total ${res.total_cost:.4f}")
        return 0 if res.winner else 1

    # Phase A — single agent
    model_name = args.model or cfg.categories.default
    if not model_name:
        print("error: pass --model (or set [categories].default)", file=sys.stderr)
        return 2
    spec = cfg.model(model_name)
    ws = Workspace(args.repo)
    print(f"-> agent | model: {spec.name} | repo: {ws.root} | max-steps: {args.max_steps}")
    print("  (the agent reads/edits files and runs commands in the repo)\n")
    result = asyncio.run(run_agent(spec, args.task, ws, budget=budget, max_steps=args.max_steps))
    if result.error:
        print(f"agent stopped: {result.error}", file=sys.stderr)
    print(f"\n{result.summary}\n")
    verdict = ""
    if args.tests:
        verdict = " | tests: " + ("PASS ✓" if ws.run(args.tests).startswith("exit 0") else "FAIL ✗")
    print(f"[{result.steps} steps | {result.n_calls} calls | ${result.cost_usd:.4f}"
          f" | {'finished' if result.finished else 'incomplete'}{verdict}]")
    return 0 if result.finished and not result.error else 1


def _cmd_auto(args) -> int:
    cfg = load_config(args.config)
    strategy = resolve_strategy(cfg, "auto")
    budget = BudgetTracker.from_config(cfg.budget)
    msgs = [{"role": "user", "content": args.prompt}]
    print(f"-> auto | classified to: {strategy.select(msgs).name}\n")
    try:
        result = asyncio.run(strategy.run(msgs, budget=budget))
    except BudgetExceeded as exc:
        print(f"budget cap hit: {exc}", file=sys.stderr)
        return 2
    if not result.ok:
        print(f"ERROR: {result.error}", file=sys.stderr)
        return 1
    print(result.text)
    print(f"\n[{result.latency_s:.2f}s | {result.n_calls} calls | ${result.cost_usd:.6f}]")
    return 0


def _cmd_code(args) -> int:
    from pathlib import Path

    from .coding import build_task_messages, build_verifier
    from .eval.benchmarks import extract_code
    from .strategies import VerifiedBestOfN

    cfg = load_config(args.config)
    names = cfg.code.models or cfg.bestofn.models
    if not names:
        print("error: no coding basket — set [code].models or [bestofn].models", file=sys.stderr)
        return 2
    critic_name = cfg.code.critic or cfg.bestofn.critic
    strategy = VerifiedBestOfN(
        models=[cfg.model(m) for m in names],
        n=args.n or cfg.code.n,
        critic=cfg.model(critic_name) if critic_name else None,
        temperature=cfg.code.temperature,
    )
    budget = BudgetTracker.from_config(cfg.budget)
    verify = build_verifier(args.file, args.tests, args.timeout)
    print(f"-> code | basket: {', '.join(names)} | n={strategy.n} | target: {args.file}"
          + (f" | tests: {args.tests}" if args.tests else " | (no tests — critic picks)")
          + "\n  (writes model code to the target file and runs your test command)\n")
    messages = build_task_messages(args.task, args.file, args.context)
    try:
        result = asyncio.run(strategy.run(messages, budget=budget, verify=verify))
    except BudgetExceeded as exc:
        print(f"budget cap hit: {exc}", file=sys.stderr)
        return 2
    if not result.ok:
        print(f"ERROR: {result.error}", file=sys.stderr)
        return 1
    Path(args.file).write_text(extract_code(result.text))
    passed = verify(result.text) if verify else None
    verdict = "TESTS PASS ✓" if passed else ("TESTS FAIL ✗ (best candidate written)" if verify else "written (unverified)")
    print(f"wrote {args.file}  |  {result.n_calls} candidates  |  ${result.cost_usd:.4f}  |  {verdict}")
    return 0 if (passed or not verify) else 1


def _cmd_optimize(args) -> int:
    from .eval.runner import run_baseline
    from .optimizer import (
        RecipeOutcome, available_model_names, candidate_recipes, mark_pareto, render,
    )

    cfg = load_config(args.config)
    available = available_model_names(cfg)
    recipes = candidate_recipes(cfg, available)
    if not recipes:
        print("no available models — add a provider key or run Ollama.", file=sys.stderr)
        return 1
    print(f"optimizing on {args.benchmark} (limit={args.limit}) over "
          f"{len(recipes)} recipes: {', '.join(recipes)}\n")
    results = asyncio.run(
        run_baseline(cfg, recipes, args.benchmark, limit=args.limit, concurrency=args.concurrency)
    )
    outcomes = mark_pareto([RecipeOutcome.from_run(r) for r in results])
    print(render(outcomes, available))
    return 0


def _cmd_report(args) -> int:
    from .report import render_report, run_report

    cfg = load_config(args.config)
    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    print(f"building report across {len(benchmarks)} tasks ({', '.join(benchmarks)}), "
          f"limit={args.limit}...\n")
    rows, available = asyncio.run(
        run_report(cfg, benchmarks, limit=args.limit, concurrency=args.concurrency)
    )
    print(render_report(rows, benchmarks, available))
    return 0


def _cmd_learn(args) -> int:
    from .eval.runner import run_baseline
    from .optimizer import RecipeOutcome, available_model_names, candidate_recipes
    from .recipe_cache import load_recipes, save_recipes

    cfg = load_config(args.config)
    pairs = [p.split("=", 1) for p in args.benchmarks.split(",") if "=" in p]
    recipes = candidate_recipes(cfg, available_model_names(cfg))
    cache = load_recipes(cfg)
    print(f"learning best recipe per category over {len(recipes)} candidates...\n")
    for category, bench in pairs:
        results = asyncio.run(
            run_baseline(cfg, recipes, bench, limit=args.limit, concurrency=args.concurrency)
        )
        outs = [RecipeOutcome.from_run(r) for r in results]
        if not outs:
            continue
        best = max(outs, key=lambda o: (o.pass_at_1, -o.avg_cost_usd))  # quality, then cheap
        cache[category] = {"recipe": best.recipe, "score": round(best.pass_at_1, 4),
                           "cost": round(best.avg_cost_usd, 6), "benchmark": bench,
                           "n": args.limit or 0}
        print(f"  {category:<12} -> {best.recipe}  ({best.pass_at_1:.0%}, "
              f"${best.avg_cost_usd:.5f}/task on {bench})")
    path = save_recipes(cfg, cache)
    print(f"\nlearned recipes written to {path}  (auto now uses them)")
    return 0


def _cmd_recipes(args) -> int:
    from .recipe_cache import cache_path, load_recipes

    cfg = load_config(args.config)
    learned = load_recipes(cfg)
    if not learned:
        print("no learned recipes yet — run `autofusion learn`.")
        return 0
    print(f"{'category':<14}{'recipe':<14}{'score':>7}{'$/task':>10}  benchmark")
    print("-" * 58)
    for cat, d in sorted(learned.items()):
        print(f"{cat:<14}{d['recipe']:<14}{d['score']:>6.0%}{d['cost']:>10.5f}  {d.get('benchmark', '')}")
    print(f"\n({cache_path(cfg)})")
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
    sub.add_parser("registry", help="show capabilities + cost per model")

    p_smoke = sub.add_parser("smoke", help="call one model end-to-end")
    p_smoke.add_argument("-m", "--model", required=True, help="model name from config")
    p_smoke.add_argument("-p", "--prompt", default="In one sentence, what is a mixture-of-agents?")

    p_fuse = sub.add_parser("fuse", help="run fusion (MoA) on one prompt")
    p_fuse.add_argument("prompt", help="the prompt to fuse")

    p_route = sub.add_parser("route", help="route one prompt to a single model")
    p_route.add_argument("prompt", help="the prompt to route")

    p_cascade = sub.add_parser("cascade", help="cheap->critic->escalate cost cascade")
    p_cascade.add_argument("prompt", help="the prompt to run through the cascade")

    p_bestofn = sub.add_parser("bestofn", help="sample N candidates, pick the best")
    p_bestofn.add_argument("prompt", help="the prompt to sample")

    p_auto = sub.add_parser("auto", help="classify the task, route to its best recipe")
    p_auto.add_argument("prompt", help="the prompt to route by category")

    p_agent = sub.add_parser("agent", help="run a coding agent in a repo")
    p_agent.add_argument("task", help="what the agent should do")
    p_agent.add_argument("-r", "--repo", default=".", help="repo/workspace directory")
    p_agent.add_argument("-m", "--model", default="", help="agent model (default: [categories].default)")
    p_agent.add_argument("-t", "--tests", default="", help="test command to verify at the end")
    p_agent.add_argument("--best-of", type=int, default=1, help="N trajectories; tests pick the winner")
    p_agent.add_argument("--max-steps", type=int, default=20)

    p_code = sub.add_parser("code", help="write a verified solution file")
    p_code.add_argument("task", help="what to build")
    p_code.add_argument("-f", "--file", default="solution.py", help="target file to write")
    p_code.add_argument("-t", "--tests", default="", help="test command (returncode 0 = pass)")
    p_code.add_argument("--context", nargs="*", default=[], help="context files to include")
    p_code.add_argument("-n", type=int, default=0, help="candidates (0 = config default)")
    p_code.add_argument("--timeout", type=float, default=60.0, help="per-test-run timeout (s)")

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

    p_opt = sub.add_parser("optimize", help="sweep recipes -> quality×cost frontier")
    p_opt.add_argument("-b", "--benchmark", default="humaneval")
    p_opt.add_argument("-n", "--limit", type=int, default=None, help="limit number of tasks")
    p_opt.add_argument("--concurrency", type=int, default=4)

    p_report = sub.add_parser("report", help="recipes vs all models across tasks")
    p_report.add_argument("--benchmarks", default="humaneval,gsm8k", help="comma-separated")
    p_report.add_argument("-n", "--limit", type=int, default=None, help="limit tasks per benchmark")
    p_report.add_argument("--concurrency", type=int, default=4)

    p_learn = sub.add_parser("learn", help="measure + cache the best recipe per category")
    p_learn.add_argument("--benchmarks", default="code=livecodebench,math=gsm8k,reasoning=mmlu-pro",
                         help="comma-separated category=benchmark pairs")
    p_learn.add_argument("-n", "--limit", type=int, default=None, help="tasks per benchmark")
    p_learn.add_argument("--concurrency", type=int, default=4)

    sub.add_parser("recipes", help="show the learned per-category recipes")

    args = parser.parse_args(argv)
    handlers = {
        "init": _cmd_init, "config-check": _cmd_config_check, "registry": _cmd_registry,
        "smoke": _cmd_smoke,
        "fuse": _cmd_fuse, "route": _cmd_route, "cascade": _cmd_cascade,
        "bestofn": _cmd_bestofn, "auto": _cmd_auto, "code": _cmd_code, "agent": _cmd_agent,
        "eval": _cmd_eval,
        "optimize": _cmd_optimize, "report": _cmd_report,
        "learn": _cmd_learn, "recipes": _cmd_recipes,
        "budget": _cmd_budget, "serve": _cmd_serve,
    }
    try:
        return handlers[args.command](args)
    except BudgetExceeded as exc:
        # Money tool: a cap hit during eval/optimize/report stops cleanly, not
        # with a stack trace. Partial results are dropped — lower -n or raise the cap.
        print(f"budget cap reached: {exc}\n"
              f"(lower -n, or raise [budget].total_usd in your config)", file=sys.stderr)
        return 2
    except (KeyError, ValueError, FileNotFoundError) as exc:
        # Friendly one-liner instead of a traceback (unknown model/strategy lists
        # valid options; missing/!parseable config; bad fusion config).
        msg = exc.args[0] if exc.args else str(exc)
        print(f"error: {msg}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Budget guardrails (Phase 4).

Fusion is N+1 calls per request and can silently multiply a bill, so cost caps
are enforced **before** any model call fires — not reconciled after. A request
whose projected cost would breach a cap raises `BudgetExceeded` and no call is
made. Local/$0 models are never blocked. Caps of `None` mean unlimited.

Spend tracking is in-process (per run/session) for v1.
"""

from __future__ import annotations

from dataclasses import dataclass

import litellm

from .config import BudgetConfig, ModelSpec

# Conservative default for the not-yet-known completion length when projecting
# a request's cost ahead of the call.
DEFAULT_MAX_OUTPUT_TOKENS = 1024


class BudgetExceeded(Exception):
    """Raised before a call fires when it would breach a configured cap."""


def _count_input_tokens(spec: ModelSpec, messages: list[dict]) -> int:
    try:
        return litellm.token_counter(model=spec.model, messages=messages)
    except Exception:
        # Offline / unknown encoding fallback: ~4 chars per token.
        chars = sum(len(m.get("content", "")) for m in messages)
        return max(1, chars // 4)


def _per_token_costs(spec: ModelSpec) -> tuple[float, float]:
    """(input, output) USD per token. Config overrides win; else LiteLLM's map."""
    in_cost, out_cost = spec.input_cost_per_token, spec.output_cost_per_token
    if in_cost is None or out_cost is None:
        info = litellm.model_cost.get(spec.model) or {}
        in_cost = info.get("input_cost_per_token", 0.0) if in_cost is None else in_cost
        out_cost = info.get("output_cost_per_token", 0.0) if out_cost is None else out_cost
    return float(in_cost or 0.0), float(out_cost or 0.0)


def estimate_request_cost(
    spec: ModelSpec, messages: list[dict], max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
) -> float:
    """Project the USD cost of one completion before it runs. 0.0 for $0 models."""
    in_cost, out_cost = _per_token_costs(spec)
    if in_cost == 0.0 and out_cost == 0.0:
        return 0.0
    return _count_input_tokens(spec, messages) * in_cost + max_output_tokens * out_cost


@dataclass
class BudgetTracker:
    """Enforces per-request + cumulative caps and accumulates spend."""

    per_request_usd: float | None = None
    total_usd: float | None = None
    spent_usd: float = 0.0

    @classmethod
    def from_config(cls, cfg: BudgetConfig) -> "BudgetTracker":
        return cls(per_request_usd=cfg.per_request_usd, total_usd=cfg.total_usd)

    def check(self, projected_cost: float) -> None:
        """Raise BudgetExceeded if this projected cost would breach a cap.
        Free ($0) requests are always allowed."""
        if projected_cost <= 0.0:
            return
        if self.per_request_usd is not None and projected_cost > self.per_request_usd:
            raise BudgetExceeded(
                f"projected ${projected_cost:.4f} exceeds per-request cap "
                f"${self.per_request_usd:.4f}"
            )
        if self.total_usd is not None and self.spent_usd + projected_cost > self.total_usd:
            raise BudgetExceeded(
                f"projected ${projected_cost:.4f} would push session spend "
                f"${self.spent_usd:.4f} past the ${self.total_usd:.4f} cap"
            )

    def record(self, actual_cost: float) -> None:
        self.spent_usd += max(0.0, actual_cost)

    def status_line(self) -> str:
        def cap(v: float | None) -> str:
            return "unlimited" if v is None else f"${v:.4f}"
        return (
            f"session spend: ${self.spent_usd:.4f}  |  "
            f"per-request cap: {cap(self.per_request_usd)}  |  "
            f"total cap: {cap(self.total_usd)}"
        )

"""Alpha hypothesis generator.

Samples strategy candidates from explicit, finite parameter grids. Every candidate the
generator emits is counted: the critic uses the running total ``trials`` to raise its
significance hurdle (the more hypotheses tried, the higher the bar — the antidote to
data snooping). Sampling is seeded so a research run is reproducible.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.strategy_brain import STRATEGY_REGISTRY, Strategy, build_strategy

ParamGrid = Mapping[str, Sequence[Any]]

DEFAULT_SEARCH_SPACE: dict[str, dict[str, list[Any]]] = {
    "trend_breakout": {
        "channel": [20, 30, 55],
        "stop_atr": [2.0, 3.0],
        "target_r": [2.0, 3.0],
    },
    "gold_pullback": {
        "fast": [10, 20],
        "slow": [50, 100],
        "stop_atr": [1.5, 2.0],
        "target_r": [2.0, 3.0],
    },
    # Pre-registered 2026-09-26 before any result was seen; do not edit to chase results.
    "ker_trend_vol": {
        "er_min": [0.35, 0.50],
        "trend_ema": [100, 200],
        "vol_expansion": [1.0, 1.2],
        "stop_atr": [2.0, 3.0],
    },
}

HYPOTHESES: dict[str, str] = {
    "trend_breakout": ("Price that closes beyond its {channel}-bar extreme in an efficient (ER>=0.30) "
                       "regime continues far enough to pay {target_r}R on a {stop_atr}-ATR stop."),
    "gold_pullback": ("In an established EMA{fast}/EMA{slow} trend, a retest of the fast EMA followed by a "
                      "resumption close offers {target_r}R on a {stop_atr}-ATR stop."),
    "ker_trend_vol": ("A channel breakout that is simultaneously efficient (ER>={er_min}), with the "
                      "EMA{trend_ema} trend and during volatility expansion (ATR14/ATR50>={vol_expansion}) "
                      "pays 2.5R on a {stop_atr}-ATR stop."),
}


@dataclass(frozen=True, slots=True)
class StrategyCandidate:
    """A concrete, testable strategy hypothesis."""

    candidate_id: str
    strategy_name: str
    params: Mapping[str, Any]
    hypothesis: str
    trial_number: int

    def build(self) -> Strategy:
        """Instantiate the strategy object."""
        return build_strategy(self.strategy_name, self.params)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation."""
        return {"candidate_id": self.candidate_id, "strategy": self.strategy_name,
                "params": dict(self.params), "hypothesis": self.hypothesis, "trial": self.trial_number}


def candidate_id(strategy_name: str, params: Mapping[str, Any]) -> str:
    """Stable content hash of a (strategy, params) pair."""
    blob = json.dumps({"s": strategy_name, "p": dict(params)}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


@dataclass
class HypothesisGenerator:
    """Seeded sampler over :data:`DEFAULT_SEARCH_SPACE` (or a custom space).

    Attributes:
        search_space: ``{strategy_name: {param: [values...]}}``.
        seed: RNG seed.
        trials: Total distinct candidates emitted so far (for snooping correction).
    """

    search_space: Mapping[str, ParamGrid] = field(default_factory=lambda: DEFAULT_SEARCH_SPACE)
    seed: int = 42
    trials: int = 0
    _seen: set[str] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        unknown = set(self.search_space) - set(STRATEGY_REGISTRY)
        if unknown:
            raise ValueError(f"search space references unknown strategies: {sorted(unknown)}")
        self._rng = random.Random(self.seed)

    def grid_size(self) -> int:
        """Total number of distinct candidates in the search space."""
        total = 0
        for grid in self.search_space.values():
            n = 1
            for values in grid.values():
                n *= len(values)
            total += n
        return total

    def _all_params(self) -> list[tuple[str, dict[str, Any]]]:
        out: list[tuple[str, dict[str, Any]]] = []
        for name, grid in self.search_space.items():
            keys = list(grid)
            for combo in itertools.product(*(grid[k] for k in keys)):
                params = dict(zip(keys, combo, strict=True))
                if name == "gold_pullback" and params["fast"] >= params["slow"]:
                    continue  # structurally invalid hypothesis, never counts as a trial
                out.append((name, params))
        return out

    def generate(self, max_candidates: int | None = None) -> list[StrategyCandidate]:
        """Emit up to ``max_candidates`` not-yet-seen candidates in random order.

        Each emitted candidate increments :attr:`trials` — including those later
        rejected — because the critic's hurdle must account for every test performed.
        """
        pool = [(n, p) for n, p in self._all_params() if candidate_id(n, p) not in self._seen]
        self._rng.shuffle(pool)
        if max_candidates is not None:
            pool = pool[:max_candidates]
        out: list[StrategyCandidate] = []
        for name, params in pool:
            cid = candidate_id(name, params)
            self._seen.add(cid)
            self.trials += 1
            text = HYPOTHESES.get(name, "{}").format(**params)
            out.append(StrategyCandidate(cid, name, params, text, self.trials))
        return out

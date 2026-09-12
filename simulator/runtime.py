"""Bounded search evaluation with persistent, provenance-aware caching."""
from dataclasses import dataclass
import math
import time
from threading import RLock
from pathlib import Path

from .cache import EvaluationCache
from .ngspice import NgSpiceConfig
from .receiver import evaluate_receiver


@dataclass(frozen=True)
class SearchLimits:
    wall_time_s: float = 120.0
    max_evaluations: int = 4
    per_stage_timeout_s: float = 30.0

    def __post_init__(self):
        if any(not math.isfinite(x) or x <= 0 for x in (self.wall_time_s, self.per_stage_timeout_s)):
            raise ValueError("timeouts must be finite and positive")
        if not isinstance(self.max_evaluations, int) or self.max_evaluations < 1:
            raise ValueError("max_evaluations must be a positive integer")


class SearchBudgetExpired(RuntimeError):
    pass


class BudgetedEvaluator:
    def __init__(self, limits=SearchLimits(), *, cache_root=Path(".nebula-cache"), evaluator=evaluate_receiver, charge_cache_hits=True):
        self.lock = RLock()
        self.charge_cache_hits = charge_cache_hits
        self.limits = limits
        self.started = time.monotonic()
        self.deadline = self.started + limits.wall_time_s
        self.evaluator = evaluator
        self.cache = EvaluationCache(cache_root)
        self.calls = self.cache_hits = self.requests = 0
        self.stop_reason = None

    def available(self):
        if self.calls >= self.limits.max_evaluations:
            self.stop_reason = "evaluation budget reached"
        elif time.monotonic() >= self.deadline:
            self.stop_reason = "search time budget reached"
        return self.stop_reason is None

    def __call__(self, parameters, conditions, fidelity, **kwargs):
        with self.lock:
            if not self.available():
                raise SearchBudgetExpired(self.stop_reason)
            self.calls += 1
            self.requests += 1
        result = self.evaluator(parameters, conditions, fidelity,
            cache=self.cache, deadline=self.deadline,
            ngspice=NgSpiceConfig(timeout_s=min(self.limits.per_stage_timeout_s,
                                               max(.001, self.deadline-time.monotonic()))), **kwargs)
        with self.lock:
            self.cache_hits += int(result.cache_hit)
            if result.cache_hit and not self.charge_cache_hits:
                self.calls -= 1
        return result

    def summary(self):
        return dict(elapsed_s=time.monotonic()-self.started, evaluation_calls=self.calls,
                    evaluation_requests=self.requests, cache_hits=self.cache_hits, stop_reason=self.stop_reason,
                    wall_time_budget_s=self.limits.wall_time_s,
                    max_evaluations=self.limits.max_evaluations)

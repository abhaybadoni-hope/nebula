"""Fair PPO vs Random Search vs CEM comparison.

Reads existing results/*.jsonl files only (plus, once run, one new
uniform-initialized CEM result file this milestone adds) -- no simulation
happens in this module.

THE CENTRAL FAIRNESS ISSUE THIS MODULE EXISTS TO FIX
------------------------------------------------------------------------
Random Search (experiments/receiver_search.py) and CEM
(experiments/train_cem.py) both rank/report candidates using
simulator.rl_adapter.reward_v1 -- a continuous score
(100*height + 10*width + 50*margin - 1000*power - |peaking-6|, clipped to
[-99, 100], or -100.0 on any simulator-stage failure). Its "success" is
simply `evaluation.success` -- whether every configured SPICE stage
(DC/AC/channel/noise/HD3/transient) passed its own gates. At TRAINING
fidelity (what all three methods use), simulator.receiver._transient_violations
hard-gates only `dfe_min_margin_v > 0`, power in (0, 15mW), and the
rail/error checks -- it does NOT hard-gate `dfe_locked_phase_eye_height_v`
or `dfe_eye_width_ui` at TRAINING fidelity (those become hard gates only at
CANDIDATE fidelity or above). So `evaluation.success=True` under reward_v1
does not guarantee height>=0.1 or width>=0.4.

PPO (experiments/train_autockt.py) instead uses rl.autockt_reward.autockt_reward
against a rl.target_spec.TargetSpec -- reward reaches the terminal bonus
(10.0) only when ALL FOUR of height/width/margin/power clear the target's
thresholds simultaneously. This is a strictly different, and for the
"trivial" target's specific threshold values, STRICTER, criterion than
reward_v1's `evaluation.success`.

Comparing "success rate" using each method's own native definition would
therefore not be comparing the same thing. This module recomputes ONE
shared criterion -- autockt_reward against
TargetSpec.from_existing_thresholds() (the "trivial" target, whose four
numbers ARE the same constants reward_v1/constraints_from_evaluation
already reference, EXISTING_THRESHOLDS) -- for every candidate from every
method that has raw per-metric data logged, and reports both the native and
the uniform criterion side by side rather than picking one silently.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Optional

from analysis.baseline_comparison import EvaluationRecord, load_run
from analysis.policy_inspection import TrajectoryStep, load_training_step_rows
from rl.autockt_reward import FAILURE_REWARD, TERMINAL_BONUS, autockt_reward
from rl.target_spec import EXISTING_THRESHOLDS, TargetSpec

# The one target ALL THREE methods are scored against in this comparison.
# Chosen because it IS EXISTING_THRESHOLDS -- the same four numbers
# reward_v1/constraints_from_evaluation already reference natively, so it
# is the least-arbitrary common ground, not an invented number.
COMPARISON_TARGET = TargetSpec.from_existing_thresholds()


def uniform_score(metrics: Optional[dict[str, float]], success: bool) -> tuple[bool, float]:
    """Recomputes (spec_satisfied, reward) under the ONE shared criterion,
    regardless of which native reward function originally scored this
    candidate. Returns (False, FAILURE_REWARD) if `metrics` is unavailable
    (never fabricates a value) -- callers should treat that as "cannot be
    uniformly scored", not as "failed".
    """

    if not success:
        return False, FAILURE_REWARD
    if metrics is None:
        return False, FAILURE_REWARD
    reward = autockt_reward(metrics, COMPARISON_TARGET, success=True)
    return reward >= TERMINAL_BONUS, reward


@dataclass(frozen=True)
class UniformEvaluation:
    """One candidate, both under its method's native criterion and the
    shared uniform one."""

    method: str
    index: int
    native_success: bool
    native_reward: float
    uniform_scoreable: bool  # False if metrics were unavailable (older CEM rows)
    uniform_success: bool
    uniform_reward: float
    wall_clock_s: Optional[float]


def _from_evaluation_record(record: EvaluationRecord) -> UniformEvaluation:
    scoreable = record.metrics is not None
    uniform_success, uniform_reward = uniform_score(record.metrics, record.success)
    return UniformEvaluation(
        method=record.method, index=record.index,
        native_success=record.success, native_reward=record.reward,
        uniform_scoreable=scoreable, uniform_success=uniform_success, uniform_reward=uniform_reward,
        wall_clock_s=record.wall_clock_s,
    )


def _from_trajectory_step(step: TrajectoryStep) -> UniformEvaluation:
    # PPO rows already use autockt_reward against whatever target that
    # episode's TargetSpec was -- "native" here means "as originally
    # scored against ITS OWN target", which may not be COMPARISON_TARGET.
    # "uniform" recomputes strictly against COMPARISON_TARGET so the two
    # only coincide when the row's own target already was the trivial one.
    scoreable = step.metrics is not None
    if scoreable:
        uniform_success, uniform_reward = uniform_score(step.metrics, step.success)
    else:
        uniform_success, uniform_reward = (step.spec_satisfied, step.reward)
    return UniformEvaluation(
        method="ppo", index=step.step,
        native_success=step.spec_satisfied, native_reward=step.reward,
        uniform_scoreable=scoreable, uniform_success=uniform_success, uniform_reward=uniform_reward,
        wall_clock_s=None,
    )


@dataclass(frozen=True)
class MethodSummary:
    method: str
    source_file: str
    n_evaluations: int
    native_success_rate: Optional[float]
    uniform_success_rate: Optional[float]
    uniform_scoreable_fraction: float
    native_evaluations_to_first_success: Optional[int]
    uniform_evaluations_to_first_success: Optional[int]
    best_native_reward: Optional[float]
    best_uniform_reward: Optional[float]
    running_best_uniform_reward: tuple[float, ...]
    wall_clock_total_s: Optional[float]
    wall_clock_available: bool
    caveats: tuple[str, ...]


def summarize_uniform(
    evaluations: list[UniformEvaluation], *, method: str, source_file: str,
    caveats: tuple[str, ...] = (),
) -> MethodSummary:
    n = len(evaluations)
    if n == 0:
        return MethodSummary(
            method=method, source_file=source_file, n_evaluations=0,
            native_success_rate=None, uniform_success_rate=None, uniform_scoreable_fraction=0.0,
            native_evaluations_to_first_success=None, uniform_evaluations_to_first_success=None,
            best_native_reward=None, best_uniform_reward=None, running_best_uniform_reward=(),
            wall_clock_total_s=None, wall_clock_available=False,
            caveats=caveats + ("no evaluations",),
        )

    scoreable = [e for e in evaluations if e.uniform_scoreable]
    native_successes = sum(1 for e in evaluations if e.native_success)
    uniform_successes = sum(1 for e in scoreable if e.uniform_success)

    native_first: Optional[int] = None
    uniform_first: Optional[int] = None
    running_best: list[float] = []
    best_so_far = -math.inf
    for position, e in enumerate(evaluations):
        if e.native_success and native_first is None:
            native_first = position + 1
        if e.uniform_scoreable and e.uniform_success and uniform_first is None:
            uniform_first = position + 1
        best_so_far = max(best_so_far, e.uniform_reward if e.uniform_scoreable else -math.inf)
        running_best.append(best_so_far)

    wall_clocks = [e.wall_clock_s for e in evaluations if e.wall_clock_s is not None]
    wall_clock_available = len(wall_clocks) == n

    return MethodSummary(
        method=method, source_file=source_file, n_evaluations=n,
        native_success_rate=native_successes / n,
        uniform_success_rate=(uniform_successes / len(scoreable)) if scoreable else None,
        uniform_scoreable_fraction=len(scoreable) / n,
        native_evaluations_to_first_success=native_first,
        uniform_evaluations_to_first_success=uniform_first,
        best_native_reward=max(e.native_reward for e in evaluations),
        best_uniform_reward=max((e.uniform_reward for e in scoreable), default=None),
        running_best_uniform_reward=tuple(running_best),
        wall_clock_total_s=sum(wall_clocks) if wall_clock_available else None,
        wall_clock_available=wall_clock_available,
        caveats=caveats,
    )


def summarize_random_search(path: str | Path) -> MethodSummary:
    loaded = load_run(path)
    evaluations = [_from_evaluation_record(r) for r in loaded.records]
    return summarize_uniform(
        evaluations, method="random_search", source_file=str(path),
        caveats=(
            "uniform sampling over the full normalized [-1,1]^5 action space "
            "(no bias toward any known-good region)",
        ),
    )


def summarize_cem(path: str | Path, *, warm_started: bool, complete: bool, planned: int) -> MethodSummary:
    loaded = load_run(path)
    evaluations = [_from_evaluation_record(r) for r in loaded.records]
    caveats: list[str] = []
    if warm_started:
        caveats.append(
            "WARM-STARTED: initial Gaussian mean is biased toward a known-good "
            "region, not centered/uniform -- not comparable to random_search's "
            "uniform coverage; excluded from the fair headline comparison"
        )
    if not complete:
        caveats.append(
            f"INCOMPLETE: {len(loaded.records)}/{planned} planned evaluations -- "
            "excluded from the fair headline comparison"
        )
    return summarize_uniform(evaluations, method="cem", source_file=str(path), caveats=tuple(caveats))


def summarize_ppo_subset(
    path: str | Path, *, target: TargetSpec, label: str,
) -> MethodSummary:
    """PPO's contribution: the subset of an existing training log's steps
    whose episode target equals `target`. NOT a fresh, independent-candidate
    experiment -- see module docstring and the report's confounds section
    for why this is structurally different from random_search/CEM's i.i.d.
    per-candidate evaluations (PPO's steps within an episode are
    sequentially dependent, and -- when drawn from a multi-target run like
    results/autockt_mixed_target_confirmation.jsonl -- interleaved in real
    evaluation order with steps against OTHER targets, which this filtered
    subsequence does not account for in its own evaluation count).
    """

    steps = load_training_step_rows(path)
    matching = [s for s in steps if s.target == target.as_dict()]
    evaluations = [_from_trajectory_step(s) for s in matching]
    return summarize_uniform(
        evaluations, method="ppo", source_file=f"{path} ({label}, target-filtered subset)",
        caveats=(
            f"{len(matching)}/{len(steps)} rows in this file matched target={label}; "
            "steps are sequentially dependent within an episode, not i.i.d. candidates",
            "evaluation order is real (interleaved with other-target steps in the "
            "source run when applicable), so 'evaluations to first success' here "
            "undercounts the true cumulative SPICE budget spent by that point",
        ),
    )

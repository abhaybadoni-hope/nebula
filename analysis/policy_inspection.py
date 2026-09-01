"""Post-hoc inspection of a trained AutoCkt-style PPO policy.

Goal this module serves: determine whether the policy learned *meaningful,
parameter-specific* action preferences (for rload_ohm, rdeg_ohm, cdeg_f,
itail_a, dfe_tap_v) rather than merely landing on one passing circuit by
chance. It does this in three read-only/no-simulation ways, plus one
proposal-only fourth:

  1. `head_action_probabilities` -- extract the policy's per-parameter
     action-probability distribution for a single fixed state (a forward
     pass through the already-trained rl.ppo_agent.PPOAgent's policy
     network; no gradient step).
  2. `compare_action_distributions` / `compare_checkpoints_at_state` --
     compare an "initial" (untrained/pre-update) and "final"
     (post-training) policy's distributions at the SAME state, the same
     matched-checkpoint methodology experiments/train_autockt.py already
     uses for reward comparisons (see `_evaluate_checkpoint`), applied here
     to the action distribution instead of the episode outcome.
  3. `load_training_step_rows` / `group_by_episode` /
     `parameter_changes_along_episode` -- reconstruct, from an ALREADY
     WRITTEN experiments/train_autockt.py --output JSONL file, the sequence
     of parameter values the policy actually chose during training, and
     which reward/success/spec_satisfied each change led to. No new SPICE.
  4. `propose_one_factor_at_a_time_sweep` -- DESIGNS (but does not run) a
     one-factor-at-a-time counterfactual sweep around a converged design:
     for rload_ohm / rdeg_ohm / cdeg_f in turn, step that parameter's grid
     index while holding the other four fixed. Returns the candidate
     parameter points as data; nothing here calls
     simulator.receiver.evaluate_receiver or any other evaluator.

KNOWN GAP (documented, not silently worked around): rl/trainer.py's
per-step training row does NOT currently log the raw SPICE `metrics` dict
(eye height/width/margin/power) -- only `reward`, `success`, `failure_stage`,
and `spec_satisfied` are logged, even though
rl/autockt_env.py::AutoCktReceiverEnv.step already computes and returns
`info["metrics"]` (see AutoCktStep). `TrajectoryStep.metrics` below is
`None` for any row from a log written before that gap is closed (which is
every log on disk as of this session). `load_training_step_rows` reports
this explicitly rather than fabricating metric values; it does not attempt
to recover them (that would require re-running SPICE at each historical
step, which is out of scope for a read-only inspection module).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from rl.parameter_grid import ACTION_DELTAS, PARAMETER_NAMES, ParameterGrid
from rl.ppo_agent import PPOAgent

# ---------------------------------------------------------------------------
# 1/2: per-parameter action probabilities for a fixed state
# ---------------------------------------------------------------------------


def head_action_probabilities(agent: PPOAgent, state: Sequence[float]) -> dict[str, list[float]]:
    """One forward pass through `agent.policy` (torch.no_grad(), no update).

    Returns, per physical parameter name (rl.parameter_grid.PARAMETER_NAMES
    order -- positionally identical to the policy's action heads, see
    rl/ppo_agent.py::PolicyNetwork.forward and rl/autockt_action.py), the
    softmax probability of each of the three discrete actions
    ACTION_DELTAS = (-1, 0, +2): "how likely is the policy, in exactly this
    state, to decrease / hold / increase this parameter's grid index."
    """

    with torch.no_grad():
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        logits_list = agent.policy(state_t)
    if len(logits_list) != len(PARAMETER_NAMES):
        raise ValueError(
            f"policy has {len(logits_list)} action heads but PARAMETER_NAMES has "
            f"{len(PARAMETER_NAMES)} entries -- head-to-parameter mapping is positional "
            "and would silently mismatch if these ever diverged"
        )
    probabilities: dict[str, list[float]] = {}
    for name, logits in zip(PARAMETER_NAMES, logits_list):
        probs = torch.softmax(logits.squeeze(0), dim=-1)
        probabilities[name] = [float(p) for p in probs]
    return probabilities


def most_likely_action(probabilities: Mapping[str, Sequence[float]]) -> dict[str, int]:
    """Per parameter, the ACTION_DELTAS value with highest probability --
    matches what `agent.act(state, deterministic=True)` would choose for
    that head (`Categorical(logits=...).probs.argmax()`, see rl/ppo_agent.py).
    """

    return {
        name: ACTION_DELTAS[max(range(len(probs)), key=lambda i: probs[i])]
        for name, probs in probabilities.items()
    }


# ---------------------------------------------------------------------------
# 3: trained vs untrained action-distribution comparison
# ---------------------------------------------------------------------------


def _kl_divergence(p: Sequence[float], q: Sequence[float], epsilon: float = 1e-12) -> float:
    """KL(p || q) in nats. Both are already-normalized softmax outputs."""

    return sum(
        pi * math.log((pi + epsilon) / (qi + epsilon))
        for pi, qi in zip(p, q)
        if pi > 0.0
    )


def _entropy(p: Sequence[float], epsilon: float = 1e-12) -> float:
    return -sum(pi * math.log(pi + epsilon) for pi in p if pi > 0.0)


@dataclass(frozen=True)
class ParameterDistributionShift:
    parameter: str
    initial_probabilities: tuple[float, float, float]
    final_probabilities: tuple[float, float, float]
    initial_argmax_delta: int
    final_argmax_delta: int
    argmax_changed: bool
    l1_distance: float
    kl_final_given_initial: float
    initial_entropy: float
    final_entropy: float


def compare_action_distributions(
    initial_probabilities: Mapping[str, Sequence[float]],
    final_probabilities: Mapping[str, Sequence[float]],
) -> dict[str, ParameterDistributionShift]:
    """Per-parameter comparison of two already-computed probability dicts
    (e.g. from `head_action_probabilities` on the SAME state, once per
    checkpoint). Pure arithmetic; runs no forward pass itself.
    """

    if set(initial_probabilities) != set(final_probabilities):
        raise ValueError("initial/final probability dicts must cover the same parameters")
    initial_choices = most_likely_action(initial_probabilities)
    final_choices = most_likely_action(final_probabilities)
    shifts: dict[str, ParameterDistributionShift] = {}
    for name in initial_probabilities:
        p_init = list(initial_probabilities[name])
        p_final = list(final_probabilities[name])
        l1 = sum(abs(a - b) for a, b in zip(p_init, p_final))
        shifts[name] = ParameterDistributionShift(
            parameter=name,
            initial_probabilities=tuple(p_init),
            final_probabilities=tuple(p_final),
            initial_argmax_delta=initial_choices[name],
            final_argmax_delta=final_choices[name],
            argmax_changed=initial_choices[name] != final_choices[name],
            l1_distance=l1,
            kl_final_given_initial=_kl_divergence(p_final, p_init),
            initial_entropy=_entropy(p_init),
            final_entropy=_entropy(p_final),
        )
    return shifts


def compare_checkpoints_at_state(
    *,
    agent: PPOAgent,
    initial_policy_state: Mapping[str, Any],
    final_policy_state: Mapping[str, Any],
    state: Sequence[float],
) -> dict[str, ParameterDistributionShift]:
    """Loads `initial_policy_state` into `agent.policy`, captures action
    probabilities at `state`, loads `final_policy_state`, captures again,
    compares. Leaves `agent.policy` holding `final_policy_state` afterward
    (matching experiments/train_autockt.py::main's own convention of always
    ending with the trained policy loaded) regardless of call order.
    """

    agent.policy.load_state_dict(initial_policy_state)
    initial_probabilities = head_action_probabilities(agent, state)
    agent.policy.load_state_dict(final_policy_state)
    final_probabilities = head_action_probabilities(agent, state)
    return compare_action_distributions(initial_probabilities, final_probabilities)


# ---------------------------------------------------------------------------
# 4/5: parameter trajectories from existing training logs
# ---------------------------------------------------------------------------

_TRAINING_STEP_KEYS = {"step", "parameters", "reward", "episode", "indices"}
_CHECKPOINT_SUMMARY_KEYS = {"checkpoint", "episode_reward", "steps"}


def _row_kind(row: Mapping[str, Any]) -> str:
    keys = set(row)
    if _TRAINING_STEP_KEYS <= keys:
        return "training_step"
    if _CHECKPOINT_SUMMARY_KEYS <= keys:
        return "checkpoint_summary"
    return "unknown"


@dataclass(frozen=True)
class TrajectoryStep:
    episode: int
    step: int
    parameters: dict[str, float]
    indices: tuple[int, ...]
    reward: float
    success: bool
    spec_satisfied: bool
    failure_stage: str | None
    target: dict[str, float]
    metrics: dict[str, float] | None  # None unless the source log has per-step metrics logging


def load_training_step_rows(path: str | Path) -> list[TrajectoryStep]:
    """Reads an experiments/train_autockt.py --output JSONL file and returns
    only its per-STEP training rows (checkpoint-evaluation summary rows,
    written by the same file when --checkpoint-eval-episodes > 0, are a
    different row shape and are skipped here -- see
    `load_checkpoint_summary_rows`). Read-only; parses JSON already on disk,
    runs no simulation.
    """

    steps: list[TrajectoryStep] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if _row_kind(row) != "training_step":
            continue
        try:
            steps.append(TrajectoryStep(
                episode=int(row["episode"]), step=int(row["step"]),
                parameters=dict(row["parameters"]), indices=tuple(row["indices"]),
                reward=float(row["reward"]), success=bool(row["success"]),
                spec_satisfied=bool(row["spec_satisfied"]),
                failure_stage=row.get("failure_stage"), target=dict(row["target"]),
                metrics=dict(row["metrics"]) if "metrics" in row else None,
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed training-step row at {path}:{line_number}") from exc
    return steps


def load_checkpoint_summary_rows(path: str | Path) -> list[dict[str, Any]]:
    """Reads the checkpoint-evaluation summary rows (label, per-episode
    reward/spec_satisfied/steps/target) from the same file, if any --
    written by experiments/train_autockt.py's `_evaluate_checkpoint` when
    --checkpoint-eval-episodes > 0. Read-only.
    """

    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if _row_kind(row) == "checkpoint_summary":
            rows.append(row)
    return rows


def group_by_episode(steps: Sequence[TrajectoryStep]) -> dict[int, list[TrajectoryStep]]:
    """Groups training-step rows by episode index, each episode's steps
    sorted by step number.
    """

    episodes: dict[int, list[TrajectoryStep]] = {}
    for step in steps:
        episodes.setdefault(step.episode, []).append(step)
    for episode_steps in episodes.values():
        episode_steps.sort(key=lambda s: s.step)
    return episodes


@dataclass(frozen=True)
class ParameterChange:
    episode: int
    from_step: int
    to_step: int
    parameter: str
    from_value: float
    to_value: float
    from_index: int
    to_index: int
    index_delta: int
    reward_after: float
    success_after: bool
    spec_satisfied_after: bool


def parameter_changes_along_episode(episode_steps: Sequence[TrajectoryStep]) -> list[ParameterChange]:
    """For one episode's ordered steps, one ParameterChange per
    (consecutive-step-pair, parameter): which of the 5 parameters' grid
    index moved between step i-1 and step i (index_delta may be 0 -- the
    policy chose "hold" for that head), and the reward/success/
    spec_satisfied that resulted from landing on step i. Associates each
    parameter move with `reward_after` (always available) but NOT with raw
    per-metric SPICE values (see module docstring "KNOWN GAP") unless the
    source log's rows carry `metrics` (only true for logs written after
    that gap is closed).

    The first logged step's own preceding state (the episode's true initial
    design) is not reconstructed here -- callers needing that should pass
    the known initial parameters/indices separately (e.g.
    rl.parameter_grid.VERIFIED_INITIAL_PARAMETERS / verified_initial_indices
    for an unrandomized-initial-state run).
    """

    changes: list[ParameterChange] = []
    for previous, current in zip(episode_steps, episode_steps[1:]):
        for name, previous_index, current_index in zip(PARAMETER_NAMES, previous.indices, current.indices):
            changes.append(ParameterChange(
                episode=current.episode, from_step=previous.step, to_step=current.step,
                parameter=name,
                from_value=previous.parameters[name], to_value=current.parameters[name],
                from_index=int(previous_index), to_index=int(current_index),
                index_delta=int(current_index) - int(previous_index),
                reward_after=current.reward, success_after=current.success,
                spec_satisfied_after=current.spec_satisfied,
            ))
    return changes


# ---------------------------------------------------------------------------
# 6: one-factor-at-a-time counterfactual sweep DESIGN (proposal only)
# ---------------------------------------------------------------------------

DEFAULT_COUNTERFACTUAL_PARAMETERS: tuple[str, ...] = ("rload_ohm", "rdeg_ohm", "cdeg_f")
DEFAULT_COUNTERFACTUAL_INDEX_OFFSETS: tuple[int, ...] = (-4, -2, -1, 1, 2, 4)


@dataclass(frozen=True)
class CounterfactualCandidate:
    parameter: str
    index_offset: int
    baseline_index: int
    candidate_index: int
    clipped: bool  # True if the requested offset landed outside the grid and was clamped
    baseline_value: float
    candidate_value: float
    parameters: dict[str, float]  # full 5-parameter point; only `parameter` differs from baseline


def propose_one_factor_at_a_time_sweep(
    baseline_parameters: Mapping[str, float],
    grids: Mapping[str, ParameterGrid],
    *,
    parameter_names: Sequence[str] = DEFAULT_COUNTERFACTUAL_PARAMETERS,
    index_offsets: Sequence[int] = DEFAULT_COUNTERFACTUAL_INDEX_OFFSETS,
) -> list[CounterfactualCandidate]:
    """DESIGNS (does not run) a one-factor-at-a-time counterfactual sweep
    around `baseline_parameters` (intended to be a converged/trained
    design's final parameters): for each of `parameter_names` in turn, hold
    the other four parameters fixed at their baseline value and step the
    target parameter's grid index by each of `index_offsets`, clipped to
    that grid's own bounds via the existing, unmodified
    rl.parameter_grid.ParameterGrid.clip_index -- the identical clipping the
    RL action mechanics already use, reused here read-only.

    An offset that clips to an index already produced by an earlier offset
    (or to the baseline index itself) is dropped, so no candidate point is
    proposed twice.

    This function calls no evaluator and performs no simulation -- its
    output is the experiment DESIGN (a list of concrete parameter dicts),
    meant to be reviewed/approved before any candidate is passed to
    simulator.receiver.evaluate_receiver.
    """

    if not parameter_names:
        raise ValueError("parameter_names must be non-empty")
    missing_grids = [name for name in parameter_names if name not in grids]
    if missing_grids:
        raise ValueError(f"no grid supplied for: {missing_grids}")
    missing_baseline = [name for name in PARAMETER_NAMES if name not in baseline_parameters]
    if missing_baseline:
        raise ValueError(f"baseline_parameters missing: {missing_baseline}")

    candidates: list[CounterfactualCandidate] = []
    for name in parameter_names:
        grid = grids[name]
        baseline_index = grid.nearest_index(baseline_parameters[name])
        seen_indices = {baseline_index}
        for offset in index_offsets:
            if offset == 0:
                continue
            requested_index = baseline_index + offset
            candidate_index = grid.clip_index(requested_index)
            if candidate_index in seen_indices:
                continue
            seen_indices.add(candidate_index)
            candidate_parameters = dict(baseline_parameters)
            candidate_parameters[name] = grid.value_at(candidate_index)
            candidates.append(CounterfactualCandidate(
                parameter=name, index_offset=offset,
                baseline_index=baseline_index, candidate_index=candidate_index,
                clipped=candidate_index != requested_index,
                baseline_value=baseline_parameters[name], candidate_value=candidate_parameters[name],
                parameters=candidate_parameters,
            ))
    return candidates

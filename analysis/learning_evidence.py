"""Task 4 (overnight chunk, sec 22): consolidates the existing real-SPICE
PPO evidence into the requested A-D structure. Reuses
analysis.policy_inspection's loaders throughout -- computes from the
actual, already-written training logs wherever the data supports it,
rather than re-asserting prior narrative numbers. No new SPICE; no PPO
retraining.

A. Initial vs final policy -- matched-checkpoint summaries are pulled
   directly from the checkpoint-summary rows already in each training log
   (written by experiments/train_autockt.py::_evaluate_checkpoint).
B. Learning progression -- per-update mean reward is RECOMPUTED from the
   log's own per-step rows (grouped by episode, `episodes_per_update`
   episodes per update), not copied from a prior run's printed console
   summary -- this file is a genuine second, independent computation of
   the same underlying data. No smoothing, no curve-fitting: if the
   result is noisy, it is reported as noisy.
C. Parameter behavior -- delegates to the already-existing, already-tested
   analysis.policy_inspection module (trained-vs-untrained action
   distributions, parameter trajectories) and the R/C counterfactual
   sweep results already on disk.
D. Training efficiency -- evaluation counts and wall-clock are read
   directly from the log rows (wall-clock only where instrumented; PPO's
   training-phase logs predate per-step wall_clock_s -- see sec 16 KNOWN
   GAP -- so training wall-clock is reported as unavailable, not guessed,
   for those runs).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from analysis.policy_inspection import (
    TrajectoryStep,
    group_by_episode,
    load_checkpoint_summary_rows,
    load_training_step_rows,
)


@dataclass(frozen=True)
class MatchedCheckpointSummary:
    source_file: str
    initial_satisfaction_rate: Optional[float]
    final_satisfaction_rate: Optional[float]
    initial_mean_reward: Optional[float]
    final_mean_reward: Optional[float]
    wins: Optional[int]
    ties: Optional[int]
    losses: Optional[int]
    n_matched: Optional[int]
    note: str = ""


def summarize_matched_checkpoint(path: str | Path) -> MatchedCheckpointSummary:
    """(A) Reads the checkpoint-summary rows already written by
    experiments/train_autockt.py's --checkpoint-eval-episodes (aggregate
    per-checkpoint numbers only -- per-episode win/tie/loss requires the
    per-episode rows, which this aggregate-only row shape does not carry
    for the two general training-log files; see
    controlled_unseen_target_generalization.jsonl below for a log that DOES
    carry per-case win/tie/loss, from
    experiments/controlled_unseen_target_eval.py's richer row shape).
    """

    if not Path(path).is_file():
        return MatchedCheckpointSummary(str(path), None, None, None, None, None, None, None, None,
                                         note="source file does not exist")
    rows = load_checkpoint_summary_rows(path)
    if not rows:
        return MatchedCheckpointSummary(str(path), None, None, None, None, None, None, None, None,
                                         note="no checkpoint-summary rows found")
    initial = [r for r in rows if r.get("checkpoint") == "initial"]
    final = [r for r in rows if r.get("checkpoint") == "final"]
    if not initial or not final:
        return MatchedCheckpointSummary(str(path), None, None, None, None, None, None, None, None,
                                         note="missing initial or final checkpoint rows")

    def rate(group):
        return sum(1 for r in group if r.get("spec_satisfied")) / len(group)

    def mean_reward(group):
        return sum(r["episode_reward"] for r in group) / len(group)

    return MatchedCheckpointSummary(
        source_file=str(path),
        initial_satisfaction_rate=rate(initial), final_satisfaction_rate=rate(final),
        initial_mean_reward=mean_reward(initial), final_mean_reward=mean_reward(final),
        wins=None, ties=None, losses=None,  # per-episode win/tie/loss needs matched episode indices; see note
        n_matched=min(len(initial), len(final)),
        note="aggregate-only row shape (episode/episode_reward/spec_satisfied/steps/target); "
        "per-episode win/tie/loss requires pairing by episode index, done separately "
        "for controlled_unseen_target_generalization.jsonl below, which logs richer per-case rows",
    )


def summarize_controlled_generalization_check(path: str | Path) -> dict:
    """(A) For experiments/controlled_unseen_target_eval.py's output, which
    DOES carry per-case initial/final reward and an explicit win/tie/loss
    outcome per row -- the richest of the three checkpoint-comparison logs.
    """

    resolved = Path(path)
    if not resolved.is_file():
        return {"available": False, "source_file": str(resolved)}
    rows = [json.loads(line) for line in resolved.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = [r for r in rows if r.get("row_type") == "case"]
    summary_rows = [r for r in rows if r.get("row_type") == "summary"]
    if not cases:
        return {"available": False, "source_file": str(resolved)}
    outcomes = {"win": 0, "tie": 0, "loss": 0}
    for c in cases:
        outcomes[c["outcome"]] += 1
    summary = summary_rows[0] if summary_rows else {}
    return {
        "available": True, "source_file": str(resolved), "n_matched": len(cases),
        "initial_satisfaction_rate": summary.get("initial_satisfaction_rate"),
        "final_satisfaction_rate": summary.get("final_satisfaction_rate"),
        "initial_mean_reward": summary.get("initial_mean_reward"),
        "final_mean_reward": summary.get("final_mean_reward"),
        "wins": outcomes["win"], "ties": outcomes["tie"], "losses": outcomes["loss"],
    }


def compute_per_update_progression(path: str | Path, episodes_per_update: int) -> list[dict]:
    """(B) Recomputes per-update mean episode reward from the log's own
    per-step rows -- grouped into consecutive blocks of `episodes_per_update`
    episodes (matching how experiments/train_autockt.py's training loop
    itself batches episodes into updates). This is an independent
    recomputation from the raw per-step data, not a copy of any prior
    printed summary.
    """

    steps = load_training_step_rows(path)
    episodes = group_by_episode(steps)
    episode_indices = sorted(episodes)
    progression = []
    for update_index, start in enumerate(range(0, len(episode_indices), episodes_per_update)):
        block = episode_indices[start:start + episodes_per_update]
        if not block:
            continue
        episode_rewards = [sum(s.reward for s in episodes[i]) for i in block]
        satisfied = [any(s.spec_satisfied for s in episodes[i]) for i in block]
        progression.append({
            "update": update_index, "episodes": len(block),
            "mean_episode_reward": sum(episode_rewards) / len(episode_rewards),
            "any_spec_satisfied": any(satisfied),
            "satisfaction_rate": sum(satisfied) / len(satisfied),
        })
    return progression


def training_efficiency(path: str | Path) -> dict:
    """(D) Evaluation count and, where instrumented, wall-clock -- read
    directly from the log, never estimated.
    """

    if not Path(path).is_file():
        return {"source_file": str(path), "n_evaluations": 0}
    steps = load_training_step_rows(path)
    if not steps:
        return {"source_file": str(path), "n_evaluations": 0}
    first_success = next((i + 1 for i, s in enumerate(steps) if s.spec_satisfied), None)
    return {
        "source_file": str(path),
        "n_evaluations": len(steps),
        "n_episodes": len(group_by_episode(steps)),
        "evaluations_to_first_success": first_success,
        "wall_clock_available": False,  # per-step training rows predate wall_clock_s logging (sec 16 KNOWN GAP)
    }


def build_learning_evidence_report() -> dict:
    """Assembles A, B, D from real, already-existing files. C is
    deliberately left as a pointer to analysis.policy_inspection /
    docs/autockt-mapping.md sec 17's R/C counterfactual sweep results
    rather than duplicated here -- those already ARE the parameter-behavior
    evidence, computed and tested elsewhere.
    """

    return {
        "A_matched_checkpoint": {
            "single_target_hard_confirmation": summarize_matched_checkpoint(
                "results/autockt_normfix_confirmation.jsonl"
            ).__dict__,
            "mixed_target_confirmation": summarize_matched_checkpoint(
                "results/autockt_mixed_target_confirmation.jsonl"
            ).__dict__,
            "controlled_unseen_target_generalization": summarize_controlled_generalization_check(
                "results/controlled_unseen_target_generalization.jsonl"
            ),
        },
        "B_learning_progression": {
            "mixed_target_confirmation_per_update": compute_per_update_progression(
                "results/autockt_mixed_target_confirmation.jsonl", episodes_per_update=6
            ),
        },
        "C_parameter_behavior": {
            "note": "see analysis/policy_inspection.py (trained-vs-untrained action "
            "distributions, parameter trajectories) and "
            "results/rc_counterfactual_sweep_mixed_target.jsonl / docs/autockt-mapping.md "
            "sec 17 (R/C perturbations vs. feasibility) -- not duplicated here.",
        },
        "D_training_efficiency": {
            "single_target_hard_confirmation": training_efficiency(
                "results/autockt_normfix_confirmation.jsonl"
            ),
            "mixed_target_confirmation": training_efficiency(
                "results/autockt_mixed_target_confirmation.jsonl"
            ),
        },
    }


def _main() -> int:
    report = build_learning_evidence_report()
    output_path = Path("results/learning_evidence_report.json")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing {output_path}")
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

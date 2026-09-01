"""One-factor-at-a-time R/C/Cdeg counterfactual sweep around a converged
PPO design, via DIRECT real-SPICE simulator.receiver.evaluate_receiver
calls -- no ReceiverRLAdapter, no PPOAgent, no policy, no retraining.

Uses analysis.policy_inspection.propose_one_factor_at_a_time_sweep to
generate the candidate list (see that module for the design rationale), and
evaluates each candidate under the same conditions
experiments/train_autockt.py's real backend implicitly uses
(SimulationConditions() defaults, EvaluationFidelity.TRAINING, the default
synthetic channel, no cache -- every call is a fresh simulation).

First used to analyze results/autockt_mixed_target_confirmation.jsonl's own
best_parameters design; see docs/autockt-mapping.md section 17 for that
run's results and interpretation. Kept general (source/output/parameter/
offset selection are all CLI flags) so a later milestone can point it at a
different baseline run without copy-pasting this script again.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from simulator.config import SimulationConditions
from simulator.receiver import EvaluationFidelity, ReceiverParameters, evaluate_receiver
from simulator.provenance import git_identity

from analysis.policy_inspection import (
    DEFAULT_COUNTERFACTUAL_INDEX_OFFSETS,
    DEFAULT_COUNTERFACTUAL_PARAMETERS,
    load_training_step_rows,
    propose_one_factor_at_a_time_sweep,
)
from rl.autockt_reward import autockt_reward
from rl.parameter_grid import build_parameter_grids
from rl.target_spec import TargetSpec

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a one-factor-at-a-time R/C/Cdeg counterfactual sweep "
        "around a converged PPO design's best_parameters, via direct real-SPICE "
        "evaluate_receiver calls (no policy, no retraining)."
    )
    parser.add_argument(
        "--source", type=Path, required=True,
        help="experiments/train_autockt.py --output JSONL to pull the baseline design from "
        "(the row with the highest reward, via analysis.policy_inspection.load_training_step_rows)",
    )
    parser.add_argument("--output", type=Path, required=True, help="new JSONL path; refuses to overwrite")
    parser.add_argument(
        "--parameters", nargs="+", default=list(DEFAULT_COUNTERFACTUAL_PARAMETERS),
        help=f"which physical parameters to sweep (default: {DEFAULT_COUNTERFACTUAL_PARAMETERS})",
    )
    parser.add_argument(
        "--offsets", type=int, nargs="+", default=list(DEFAULT_COUNTERFACTUAL_INDEX_OFFSETS),
        help=f"grid-index offsets to test per parameter (default: {DEFAULT_COUNTERFACTUAL_INDEX_OFFSETS})",
    )
    parser.add_argument("--grid-points", type=int, default=21)
    parser.add_argument(
        "--grid-spacing", choices=("linear", "log"), default="log",
        help="must match the --grid-spacing the source run used, or candidate grid "
        "indices will not correspond to the same physical grid the policy chose from",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing {args.output}")

    steps = load_training_step_rows(args.source)
    if not steps:
        raise ValueError(f"no training-step rows found in {args.source}")
    best = max(steps, key=lambda s: s.reward)
    baseline_target = TargetSpec(**best.target)

    print(json.dumps({
        "baseline_source": str(args.source),
        "baseline_episode": best.episode, "baseline_step": best.step,
        "baseline_reward": best.reward, "baseline_success": best.success,
        "baseline_parameters": best.parameters, "baseline_target": best.target,
    }, indent=2))

    grids = build_parameter_grids(args.grid_points, spacing=args.grid_spacing)
    candidates = propose_one_factor_at_a_time_sweep(
        best.parameters, grids, parameter_names=tuple(args.parameters), index_offsets=tuple(args.offsets),
    )
    print(f"proposed {len(candidates)} candidates")

    conditions = SimulationConditions()
    fidelity = EvaluationFidelity.TRAINING

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for index, candidate in enumerate(candidates):
            parameters = ReceiverParameters(**candidate.parameters)
            start = time.perf_counter()
            evaluation = evaluate_receiver(parameters, conditions, fidelity=fidelity)
            wall_clock_s = time.perf_counter() - start
            # Matches rl/autockt_env.py::AutoCktReceiverEnv.step's own
            # `success = rl_step.info["failure_stage"] is None` exactly,
            # for parity with what the training loop actually used.
            success = evaluation.failed_stage is None
            reward = autockt_reward(evaluation.metrics, baseline_target, success=success)
            row = {
                "candidate_index": index,
                "parameter": candidate.parameter,
                "index_offset": candidate.index_offset,
                "baseline_index": candidate.baseline_index,
                "candidate_index_on_grid": candidate.candidate_index,
                "clipped": candidate.clipped,
                "baseline_value": candidate.baseline_value,
                "candidate_value": candidate.candidate_value,
                "parameters": candidate.parameters,
                "target": baseline_target.as_dict(),
                "success": success,
                "failed_stage": evaluation.failed_stage,
                "metrics": evaluation.metrics,
                "autockt_reward": reward,
                "spec_satisfied": reward >= 10.0,
                "evaluation_id": evaluation.evaluation_id,
                "wall_clock_s": wall_clock_s,
                "baseline_parameters": best.parameters,
                "baseline_reward": best.reward,
                "source_run": str(args.source),
                **git_identity(REPOSITORY_ROOT),
            }
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(json.dumps({
                "progress": f"{index + 1}/{len(candidates)}",
                "parameter": candidate.parameter, "offset": candidate.index_offset,
                "success": success, "failed_stage": evaluation.failed_stage,
                "autockt_reward": reward, "wall_clock_s": round(wall_clock_s, 2),
            }))

    print(json.dumps({"done": True, "output": str(args.output), "count": len(candidates)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

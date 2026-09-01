"""Task 1 (overnight chunk, sec 22): the clean end-to-end automated
pipeline --

    TargetSpec -> validate -> generate candidates (existing PPO policy,
    deterministic rollout, real SPICE per step) -> nominal feasibility
    filter -> PVT-aware selection (optional) -> final schematic export ->
    final specification report

Every stage below is a thin orchestration wrapper around ALREADY-EXISTING,
unmodified components -- nothing here reimplements PPO, the environment,
the reward, the simulator, or PVT evaluation. See each function's docstring
for exactly which existing module it delegates to. No new PPO training
happens anywhere in this file: candidate generation always uses
`agent.act(..., deterministic=True)` (a read-only forward pass) against
either an explicitly-loaded, already-trained checkpoint, or (for dry-run /
synthetic-backend testing only) a freshly-constructed, untrained policy --
`agent.update()` is never called.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from simulator.rl_adapter import ReceiverRLAdapter, RLBudget

from rl.autockt_env import AutoCktReceiverEnv
from rl.autockt_state import STATE_DIM
from rl.parameter_grid import PARAMETER_NAMES, build_parameter_grids
from rl.ppo_agent import PPOAgent
from rl.synthetic_benchmark import synthetic_evaluate_receiver
from rl.target_spec import SPEC_NAMES, TargetSpec

from experiments.export_final_schematic import render_final_schematic
from experiments.train_autockt import _resolve_initial_indices

from analysis.design_catalog import FeasibleDesign, rank_by_measured_trade_offs
from analysis.final_specification import build_final_specification_report, format_report
from analysis.pvt_selection import (
    PVTRobustnessResult,
    run_pvt_evaluation,
    select_with_trade_off_preference,
)


# ---------------------------------------------------------------------------
# Stage 1: target validation
# ---------------------------------------------------------------------------

def validate_target(target: TargetSpec) -> list[str]:
    """Lightweight, simulator-free validation reusing rl.target_spec.SPEC_NAMES.
    Returns a list of problems; empty means valid.
    """

    problems: list[str] = []
    for name in SPEC_NAMES:
        value = getattr(target, name)
        if not math.isfinite(value):
            problems.append(f"{name} is not finite: {value}")
    if not (0 < target.dfe_eye_width_ui <= 1):
        problems.append(f"dfe_eye_width_ui out of plausible (0,1] UI range: {target.dfe_eye_width_ui}")
    if target.ctle_power_w <= 0:
        problems.append(f"ctle_power_w must be positive: {target.ctle_power_w}")
    return problems


# ---------------------------------------------------------------------------
# Stage 2/3: candidate generation via the existing RL framework + SPICE
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineCandidate:
    episode: int
    parameters: dict[str, float]
    metrics: dict[str, float]
    reward: float
    spec_satisfied: bool
    steps: int


def generate_candidates(
    *,
    target: TargetSpec,
    checkpoint_path: Optional[Path],
    agent_seed: int = 42,
    eval_seed: int = 42,
    episodes: int = 3,
    horizon: int = 4,
    backend: str = "real",
    grid_points: int = 21,
    grid_spacing: str = "log",
    initial_indices_source: str = "verified",
    randomize_initial_state: bool = True,
    max_evaluations: int = 1000,
) -> list[PipelineCandidate]:
    """Deterministic rollout of an existing policy against `target`, via the
    real, UNMODIFIED AutoCktReceiverEnv + PPOAgent + ReceiverRLAdapter.
    `checkpoint_path=None` uses a freshly-constructed, untrained policy
    (only sensible for backend='synthetic' dry-run testing -- an untrained
    policy against real SPICE has no learned behavior to demonstrate, see
    docs/autockt-mapping.md sec 20). `backend='synthetic'` uses
    rl.synthetic_benchmark.synthetic_evaluate_receiver (no SPICE, no PDK
    needed); `backend='real'` calls real ngspice via the unmodified
    evaluate_receiver, once per environment step.
    """

    grids = build_parameter_grids(grid_points, spacing=grid_spacing)
    initial_indices = _resolve_initial_indices(grids, initial_indices_source)

    adapter_kwargs: dict[str, Any] = {}
    if backend == "synthetic":
        adapter_kwargs["evaluator"] = synthetic_evaluate_receiver
    adapter = ReceiverRLAdapter(budget=RLBudget(max_evaluations), seed=eval_seed, **adapter_kwargs)

    env = AutoCktReceiverEnv(
        target_pool=(target,), initial_indices=initial_indices, horizon=horizon,
        adapter=adapter, grids=grids, seed=eval_seed, randomize_initial_state=randomize_initial_state,
    )
    agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=agent_seed)
    if checkpoint_path is not None:
        agent.policy.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    candidates: list[PipelineCandidate] = []
    for episode in range(episodes):
        state, _ = env.reset()
        done = truncated = False
        episode_reward = 0.0
        steps = 0
        last_parameters: Optional[dict[str, float]] = None
        last_metrics: Optional[dict[str, float]] = None
        while not done and not truncated:
            choices, _log_prob, _value = agent.act(state, deterministic=True)
            step_out = env.step(choices)
            episode_reward += step_out.reward
            state = step_out.state
            done, truncated = step_out.done, step_out.truncated
            steps += 1
            last_parameters = step_out.info["parameters"]
            last_metrics = step_out.info["metrics"]
        candidates.append(PipelineCandidate(
            episode=episode, parameters=last_parameters or {}, metrics=last_metrics or {},
            reward=episode_reward, spec_satisfied=done, steps=steps,
        ))
    return candidates


# ---------------------------------------------------------------------------
# Stage 4: nominal feasibility filter
# ---------------------------------------------------------------------------

def filter_nominal_feasible(candidates: list[PipelineCandidate]) -> list[PipelineCandidate]:
    """`spec_satisfied` was already computed inside AutoCktReceiverEnv.step()
    via the existing, unmodified autockt_reward -- this filters on that
    result directly rather than recomputing the check.
    """

    return [c for c in candidates if c.spec_satisfied]


# ---------------------------------------------------------------------------
# Stage 5: PVT-aware selection (optional -- spends real SPICE only if a
# condition set is explicitly given)
# ---------------------------------------------------------------------------

def _candidate_to_design(candidate: PipelineCandidate) -> FeasibleDesign:
    return FeasibleDesign(
        design_id=f"pipeline_ep{candidate.episode}", source_file="pipeline (in-memory, this run)",
        source_description=f"generate_candidates episode {candidate.episode}, "
        f"{candidate.steps} step(s), reward={candidate.reward:.3f}",
        parameters=candidate.parameters, metrics=candidate.metrics,
        native_reward=candidate.reward, native_reward_scale="autockt_reward",
    )


def select_final_design(
    feasible_candidates: list[PipelineCandidate],
    *,
    pvt_conditions: Optional[tuple] = None,
    pvt_fidelity=None,
    minimum_pass_rate: float = 1.0,
    trade_off_preference: str = "most_robust",
) -> dict[str, Any]:
    """Priority, deterministic and documented (not an arbitrary weighted
    score): (1) nominal feasibility -- already guaranteed by only receiving
    `feasible_candidates`; (2) PVT pass rate, if `pvt_conditions` given
    (highest first); (3) robustness tie-break (n_conditions descending);
    (4) among any remaining PVT ties, `trade_off_preference` (one of
    `analysis.pvt_selection.TRADE_OFF_PREFERENCES`) via
    `analysis.pvt_selection.select_with_trade_off_preference` -- never
    overriding a strictly higher PVT pass rate. When `pvt_conditions` is
    None, falls back to nominal-only trade-off ranking (unchanged). PVT is
    evaluated (spending real SPICE) ONLY for candidates reaching this
    stage, and ONLY if the caller explicitly supplies `pvt_conditions` --
    never automatic.
    """

    designs = [_candidate_to_design(c) for c in feasible_candidates]
    if not designs:
        return {"selected": None, "reason": "no nominally feasible candidates", "pvt": None}

    if pvt_conditions is None:
        ranked = rank_by_measured_trade_offs(designs)
        best = ranked[0].design
        return {
            "selected": {"design_id": best.design_id, "parameters": best.parameters, "metrics": best.metrics},
            "trade_off_labels": list(ranked[0].trade_off_labels),
            "pvt": None,
            "selection_basis": "nominal-only (no PVT conditions supplied)",
        }

    from simulator.receiver import EvaluationFidelity
    fidelity = pvt_fidelity or EvaluationFidelity.FINAL
    pvt_results: list[PVTRobustnessResult] = [
        run_pvt_evaluation(d, pvt_conditions, fidelity=fidelity) for d in designs
    ]
    top = select_with_trade_off_preference(
        pvt_results, designs, preference=trade_off_preference, minimum_pass_rate=minimum_pass_rate,
    )
    matching_design = next(d for d in designs if d.design_id == top.design_id)
    return {
        "selected": {
            "design_id": matching_design.design_id, "parameters": matching_design.parameters,
            "metrics": matching_design.metrics,
        },
        "pvt": {
            "n_conditions": top.n_conditions, "n_passing": top.n_passing, "pass_rate": top.pass_rate,
            "met_minimum_pass_rate": top.pass_rate >= minimum_pass_rate,
            "worst_case_conditions": [
                {"corner": p.process_corner, "vdd": p.supply_v, "temp_c": p.temperature_c}
                for p in top.worst_case_conditions
            ],
        },
        "selection_basis": (
            f"PVT-ranked across {len(pvt_conditions)} conditions, "
            f"trade_off_preference={trade_off_preference!r}"
        ),
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    *,
    target: TargetSpec,
    checkpoint_path: Optional[Path],
    agent_seed: int = 42,
    eval_seed: int = 42,
    episodes: int = 3,
    horizon: int = 4,
    backend: str = "real",
    randomize_initial_state: bool = True,
    initial_indices_source: str = "verified",
    pvt_conditions: Optional[tuple] = None,
    trade_off_preference: str = "most_robust",
    export_schematic_to: Optional[Path] = None,
) -> dict[str, Any]:
    problems = validate_target(target)
    if problems:
        raise ValueError(f"invalid target specification: {problems}")

    candidates = generate_candidates(
        target=target, checkpoint_path=checkpoint_path, agent_seed=agent_seed, eval_seed=eval_seed,
        episodes=episodes, horizon=horizon, backend=backend, randomize_initial_state=randomize_initial_state,
        initial_indices_source=initial_indices_source,
    )
    feasible = filter_nominal_feasible(candidates)
    selection = select_final_design(
        feasible, pvt_conditions=pvt_conditions, trade_off_preference=trade_off_preference,
    )

    result: dict[str, Any] = {
        "target": target.as_dict(),
        "backend": backend,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "n_candidates_generated": len(candidates),
        "n_nominally_feasible": len(feasible),
        "selection": selection,
    }

    if selection["selected"] is not None and export_schematic_to is not None:
        from simulator.receiver import ReceiverParameters
        parameters = ReceiverParameters(**selection["selected"]["parameters"])
        schematic_text = render_final_schematic(
            parameters, achieved_metrics=selection["selected"]["metrics"],
            target_description=str(target.as_dict()),
            source_description=f"experiments.run_autockt_pipeline, backend={backend}",
        )
        export_schematic_to.parent.mkdir(parents=True, exist_ok=True)
        export_schematic_to.write_text(schematic_text, encoding="utf-8")
        result["schematic_path"] = str(export_schematic_to)

    if selection["selected"] is not None:
        report = build_final_specification_report(
            design_id=selection["selected"]["design_id"],
            parameters=selection["selected"]["parameters"],
            nominal_metrics=selection["selected"]["metrics"],
            nominal_source=f"pipeline run, backend={backend}, checkpoint={checkpoint_path}",
            pvt_result=None,  # this run's own PVT (if any) is summarized in `selection["pvt"]` instead
        )
        result["final_specification"] = report

    return result


def _main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end NEBULA pipeline: target -> PPO -> SPICE -> final design.")
    parser.add_argument("--target-mode", choices=("trivial", "hard"), default="trivial")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--backend", choices=("real", "synthetic"), default="synthetic")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--agent-seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--randomize-initial-state", action="store_true", default=True)
    parser.add_argument("--initial-indices-source", choices=("verified", "grid-center"), default="verified")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-schematic", type=Path, default=None)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing {args.output}")

    target = TargetSpec.from_hard_target() if args.target_mode == "hard" else TargetSpec.from_existing_thresholds()

    result = run_pipeline(
        target=target, checkpoint_path=args.checkpoint, agent_seed=args.agent_seed, eval_seed=args.eval_seed,
        episodes=args.episodes, horizon=args.horizon, backend=args.backend,
        randomize_initial_state=args.randomize_initial_state,
        initial_indices_source=args.initial_indices_source, export_schematic_to=args.export_schematic,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "final_specification"}, indent=2))
    if "final_specification" in result:
        print(format_report(result["final_specification"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

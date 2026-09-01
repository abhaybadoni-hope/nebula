"""Runs simulator.receiver.evaluate_pvt_grid (existing, unmodified,
previously never exercised by any test/experiment -- see
docs/autockt-mapping.md sec 21) on a given design across a PVT condition
set, and persists the results -- something no prior experiment in this
repository has done even once, despite the full 60-point PVT_GRID and
evaluate_pvt_grid having existed since early in the project.

Does not modify simulator/receiver.py or simulator/rl_adapter.py. Reuses
evaluate_pvt_grid exactly as already implemented.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from simulator.config import PVT_GRID, ProcessCorner, SimulationConditions
from simulator.receiver import EvaluationFidelity, ReceiverParameters, evaluate_pvt_grid

# [NEBULA ADAPTATION] a reduced, minimum-viable subset of the full 60-point
# PVT_GRID, matching the official brief's literal corner list (TT/SS/FF --
# not the full TT/SS/FF/SF/FS PVT_GRID includes) x VDD +/-5% (1.71/1.8/1.89V,
# already PVT_GRID's own values) x a 3-point temperature set spanning the
# brief's stated 0-125C range plus nominal (not the full 4-point 0/27/75/125
# PVT_GRID uses). 27 points, not 60 -- a deliberate minimum-viable subset
# for the ~10-day remaining timeline, not a claim of full PVT_GRID coverage.
MINIMAL_27_CONDITIONS = tuple(
    SimulationConditions(ProcessCorner(corner), temperature, supply)
    for corner in ("tt", "ss", "ff")
    for supply in (1.71, 1.8, 1.89)
    for temperature in (0.0, 27.0, 125.0)
)

CONDITION_SETS = {
    "minimal27": MINIMAL_27_CONDITIONS,
    "full60": PVT_GRID,
}


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Run evaluate_pvt_grid on a design and persist the results."
    )
    parser.add_argument(
        "--parameters-json", type=Path, required=True,
        help="JSON file with rload_ohm/rdeg_ohm/cdeg_f/itail_a/dfe_tap_v",
    )
    parser.add_argument("--condition-set", choices=tuple(CONDITION_SETS) + ("custom",), default="minimal27")
    parser.add_argument(
        "--custom-conditions", nargs="+", default=None,
        help="only used with --condition-set custom: one or more CORNER:VDD:TEMP "
        "triples, e.g. ff:1.80:27.0 -- for reproducing/re-checking specific points "
        "via the exact same evaluate_pvt_grid code path as a full sweep.",
    )
    parser.add_argument(
        "--fidelity", choices=("training", "candidate", "final"), default="final",
        help="evaluate_pvt_grid's own default is FINAL; training/candidate are "
        "cheaper options for a quicker, less-strict pass if needed.",
    )
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.condition_set == "custom":
        if not args.custom_conditions:
            raise ValueError("--condition-set custom requires --custom-conditions")
        parsed_conditions = []
        for spec in args.custom_conditions:
            corner_str, vdd_str, temp_str = spec.split(":")
            parsed_conditions.append(
                SimulationConditions(ProcessCorner(corner_str), float(temp_str), float(vdd_str))
            )
        CONDITION_SETS_RUNTIME = {**CONDITION_SETS, "custom": tuple(parsed_conditions)}
    else:
        CONDITION_SETS_RUNTIME = CONDITION_SETS

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing {args.output}")

    parameters = ReceiverParameters(**json.loads(args.parameters_json.read_text(encoding="utf-8")))
    conditions = CONDITION_SETS_RUNTIME[args.condition_set]
    fidelity = EvaluationFidelity[args.fidelity.upper()]

    print(json.dumps({
        "parameters": json.loads(args.parameters_json.read_text(encoding="utf-8")),
        "condition_set": args.condition_set, "n_conditions": len(conditions),
        "fidelity": args.fidelity,
    }, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with args.output.open("w", encoding="utf-8") as stream:
        for index, condition in enumerate(conditions):
            start = time.perf_counter()
            evaluation = evaluate_pvt_grid(
                parameters, conditions=(condition,), fidelity=fidelity,
                stop_on_failure=False,
            )[0]
            wall_clock_s = time.perf_counter() - start
            row = {
                "index": index,
                "process_corner": condition.process_corner.value,
                "supply_v": condition.supply_v,
                "temperature_c": condition.temperature_c,
                "success": evaluation.success,
                "failed_stage": evaluation.failed_stage,
                "metrics": evaluation.metrics,
                "evaluation_id": evaluation.evaluation_id,
                "wall_clock_s": wall_clock_s,
            }
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            results.append(row)
            print(json.dumps({
                "progress": f"{index + 1}/{len(conditions)}",
                "corner": condition.process_corner.value, "vdd": condition.supply_v,
                "temp_c": condition.temperature_c, "success": evaluation.success,
                "failed_stage": evaluation.failed_stage, "wall_clock_s": round(wall_clock_s, 2),
            }))
            if args.stop_on_failure and not evaluation.success:
                print(json.dumps({"stopped_early_at": index}))
                break

    successes = sum(1 for r in results if r["success"])
    summary = {
        "done": True, "output": str(args.output),
        "n_evaluated": len(results), "n_planned": len(conditions),
        "n_successes": successes, "success_rate": successes / len(results) if results else None,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

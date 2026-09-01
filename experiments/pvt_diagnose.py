"""Small, targeted diagnostic tool: re-evaluates a design at explicitly
given PVT conditions and prints FULL per-stage detail (violations, errors,
failure_code, runtime) -- more than results/*_pvt_*.jsonl rows carry
(those only log the top-level evaluation.metrics, not per-stage
violations/errors). Built to diagnose exactly why
results/design_a_pvt_minimal27.jsonl's 4 FF-corner failures happened,
before designing any local search around them (per the task's explicit
"perform READ-ONLY diagnosis first" requirement).

Read-only with respect to existing results: never writes to any
results/*.jsonl file used elsewhere; only prints. A separate, explicit
--output flag persists a diagnostic JSONL if requested, to a NEW filename.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from simulator.config import ProcessCorner, SimulationConditions
from simulator.receiver import EvaluationFidelity, ReceiverParameters, evaluate_receiver


def diagnose(
    parameters: ReceiverParameters,
    conditions: SimulationConditions,
    *,
    fidelity: EvaluationFidelity = EvaluationFidelity.FINAL,
) -> dict:
    evaluation = evaluate_receiver(parameters, conditions, fidelity=fidelity)
    return {
        "process_corner": conditions.process_corner.value,
        "supply_v": conditions.supply_v,
        "temperature_c": conditions.temperature_c,
        "success": evaluation.success,
        "failed_stage": evaluation.failed_stage,
        "metrics": evaluation.metrics,
        "stages": [
            {
                "name": stage.name, "success": stage.success, "runtime_s": stage.runtime_s,
                "violations": list(stage.violations), "failure_code": stage.failure_code,
                "errors": list(stage.errors), "warnings": list(stage.warnings),
                "metrics_keys": sorted(stage.metrics.keys()),
            }
            for stage in evaluation.stages
        ],
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose a design at specific PVT conditions.")
    parser.add_argument("--parameters-json", type=Path, required=True)
    parser.add_argument(
        "--conditions", nargs="+", required=True,
        help="one or more CORNER:VDD:TEMP triples, e.g. ff:1.80:27.0",
    )
    parser.add_argument("--fidelity", choices=("training", "candidate", "final"), default="final")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.output is not None and args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing {args.output}")

    parameters = ReceiverParameters(**json.loads(args.parameters_json.read_text(encoding="utf-8")))
    fidelity = EvaluationFidelity[args.fidelity.upper()]

    results = []
    for spec in args.conditions:
        corner_str, vdd_str, temp_str = spec.split(":")
        conditions = SimulationConditions(ProcessCorner(corner_str), float(temp_str), float(vdd_str))
        result = diagnose(parameters, conditions, fidelity=fidelity)
        results.append(result)
        print(json.dumps(result, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            for row in results:
                stream.write(json.dumps(row) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

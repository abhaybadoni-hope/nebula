"""SPICE-free tests for experiments/pvt_diagnose.py."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from simulator.config import SimulationConditions
from simulator.receiver import EvaluationFidelity, ReceiverEvaluation, ReceiverParameters, StageResult

from experiments.pvt_diagnose import _main, diagnose

DESIGN_A = ReceiverParameters(
    rload_ohm=2342.472156058411, rdeg_ohm=822.3558626926603,
    cdeg_f=9.999375862792168e-13, itail_a=0.0006028331705063624,
    dfe_tap_v=-0.011090823885148815,
)


def _fake_evaluation() -> ReceiverEvaluation:
    conditions = SimulationConditions()
    stages = (
        StageResult("dc", True, 1.0, metrics={"output_common_mode_v": 1.7}),
        StageResult(
            "transient", False, 5.0, metrics={"ctle_power_w": 0.001},
            violations=("post-warm-up DFE minimum decision margin is not positive",),
            failure_code="simulation_error",
        ),
    )
    return ReceiverEvaluation(
        False, DESIGN_A, conditions, EvaluationFidelity.FINAL, stages,
        {"ctle_power_w": 0.001}, "transient", 6.0, "fake-id", {},
    )


class DiagnoseFunctionTests(unittest.TestCase):
    def test_captures_per_stage_violations_and_failure_code(self):
        with patch("experiments.pvt_diagnose.evaluate_receiver", return_value=_fake_evaluation()):
            result = diagnose(DESIGN_A, SimulationConditions())
        self.assertFalse(result["success"])
        self.assertEqual(result["failed_stage"], "transient")
        transient_stage = next(s for s in result["stages"] if s["name"] == "transient")
        self.assertIn("positive", transient_stage["violations"][0])
        self.assertEqual(transient_stage["failure_code"], "simulation_error")

    def test_reports_which_metrics_each_stage_actually_computed(self):
        with patch("experiments.pvt_diagnose.evaluate_receiver", return_value=_fake_evaluation()):
            result = diagnose(DESIGN_A, SimulationConditions())
        dc_stage = next(s for s in result["stages"] if s["name"] == "dc")
        self.assertEqual(dc_stage["metrics_keys"], ["output_common_mode_v"])


class CLITests(unittest.TestCase):
    def test_refuses_to_overwrite_existing_output(self):
        with TemporaryDirectory() as tmp:
            params_path = Path(tmp) / "params.json"
            params_path.write_text(json.dumps({
                "rload_ohm": 1000.0, "rdeg_ohm": 1000.0, "cdeg_f": 5e-13,
                "itail_a": 1e-4, "dfe_tap_v": 0.0,
            }))
            output = Path(tmp) / "out.jsonl"
            output.write_text("already here\n")
            argv = [
                "pvt_diagnose.py", "--parameters-json", str(params_path),
                "--conditions", "tt:1.80:27.0", "--output", str(output),
            ]
            with patch("sys.argv", argv):
                with self.assertRaises(FileExistsError):
                    _main()

    def test_parses_corner_vdd_temp_triples_correctly(self):
        with TemporaryDirectory() as tmp:
            params_path = Path(tmp) / "params.json"
            params_path.write_text(json.dumps({
                "rload_ohm": 1000.0, "rdeg_ohm": 1000.0, "cdeg_f": 5e-13,
                "itail_a": 1e-4, "dfe_tap_v": 0.0,
            }))
            argv = [
                "pvt_diagnose.py", "--parameters-json", str(params_path),
                "--conditions", "ff:1.71:125.0",
            ]
            seen = []

            def fake_evaluate_receiver(parameters, conditions, fidelity, **kwargs):
                seen.append(conditions)
                return _fake_evaluation()

            with patch("sys.argv", argv), \
                 patch("experiments.pvt_diagnose.evaluate_receiver", side_effect=fake_evaluate_receiver):
                _main()

            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0].process_corner.value, "ff")
            self.assertEqual(seen[0].supply_v, 1.71)
            self.assertEqual(seen[0].temperature_c, 125.0)


if __name__ == "__main__":
    unittest.main()

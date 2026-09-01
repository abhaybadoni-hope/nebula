"""SPICE-free tests for experiments/pvt_sweep.py's condition sets and CLI
safety guard.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from experiments.pvt_sweep import CONDITION_SETS, MINIMAL_27_CONDITIONS, _main
from simulator.config import PVT_GRID


class ConditionSetTests(unittest.TestCase):
    def test_minimal27_has_27_unique_points(self):
        identities = {(c.process_corner, c.supply_v, c.temperature_c) for c in MINIMAL_27_CONDITIONS}
        self.assertEqual(len(identities), 27)

    def test_minimal27_covers_exactly_tt_ss_ff_no_sf_fs(self):
        corners = {c.process_corner.value for c in MINIMAL_27_CONDITIONS}
        self.assertEqual(corners, {"tt", "ss", "ff"})

    def test_minimal27_vdd_matches_the_plus_minus_5_percent_brief(self):
        vdds = {c.supply_v for c in MINIMAL_27_CONDITIONS}
        self.assertEqual(vdds, {1.71, 1.8, 1.89})

    def test_full60_option_is_the_real_pvt_grid_unmodified(self):
        self.assertIs(CONDITION_SETS["full60"], PVT_GRID)
        self.assertEqual(len(CONDITION_SETS["full60"]), 60)


class CustomConditionsTests(unittest.TestCase):
    def test_custom_conditions_are_parsed_and_evaluated(self):
        with TemporaryDirectory() as tmp:
            params_path = Path(tmp) / "params.json"
            params_path.write_text(json.dumps({
                "rload_ohm": 1000.0, "rdeg_ohm": 1000.0, "cdeg_f": 5e-13,
                "itail_a": 1e-4, "dfe_tap_v": 0.0,
            }))
            output = Path(tmp) / "out.jsonl"
            argv = [
                "pvt_sweep.py", "--parameters-json", str(params_path),
                "--condition-set", "custom", "--custom-conditions", "ff:1.80:27.0", "tt:1.71:0.0",
                "--output", str(output),
            ]
            seen = []

            def fake_evaluate_pvt_grid(parameters, *, conditions, fidelity, stop_on_failure):
                seen.extend(conditions)
                from simulator.receiver import EvaluationFidelity, ReceiverEvaluation
                return tuple(
                    ReceiverEvaluation(True, parameters, c, EvaluationFidelity.FINAL, (), {}, None, 0.0, "id", {})
                    for c in conditions
                )

            with patch("sys.argv", argv), \
                 patch("experiments.pvt_sweep.evaluate_pvt_grid", side_effect=fake_evaluate_pvt_grid):
                _main()

            self.assertEqual(len(seen), 2)
            self.assertEqual((seen[0].process_corner.value, seen[0].supply_v, seen[0].temperature_c), ("ff", 1.80, 27.0))
            self.assertEqual((seen[1].process_corner.value, seen[1].supply_v, seen[1].temperature_c), ("tt", 1.71, 0.0))

    def test_custom_without_conditions_raises(self):
        with TemporaryDirectory() as tmp:
            params_path = Path(tmp) / "params.json"
            params_path.write_text(json.dumps({
                "rload_ohm": 1000.0, "rdeg_ohm": 1000.0, "cdeg_f": 5e-13,
                "itail_a": 1e-4, "dfe_tap_v": 0.0,
            }))
            output = Path(tmp) / "out.jsonl"
            argv = [
                "pvt_sweep.py", "--parameters-json", str(params_path),
                "--condition-set", "custom", "--output", str(output),
            ]
            with patch("sys.argv", argv):
                with self.assertRaises(ValueError):
                    _main()


class CLIOverwriteGuardTests(unittest.TestCase):
    def test_refuses_to_overwrite_before_touching_spice(self):
        with TemporaryDirectory() as tmp:
            params_path = Path(tmp) / "params.json"
            params_path.write_text(json.dumps({
                "rload_ohm": 1000.0, "rdeg_ohm": 1000.0, "cdeg_f": 5e-13,
                "itail_a": 1e-4, "dfe_tap_v": 0.0,
            }))
            output = Path(tmp) / "out.jsonl"
            output.write_text("already here\n")

            argv = [
                "pvt_sweep.py", "--parameters-json", str(params_path), "--output", str(output),
            ]
            with patch("sys.argv", argv):
                with self.assertRaises(FileExistsError):
                    _main()


if __name__ == "__main__":
    unittest.main()

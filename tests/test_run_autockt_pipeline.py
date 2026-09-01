"""SPICE-free orchestration tests for experiments/run_autockt_pipeline.py
-- Task 1 of the overnight chunk (sec 22). All tests use
backend='synthetic' (rl.synthetic_benchmark.synthetic_evaluate_receiver,
no ngspice, no PDK) -- this IS the pipeline's dry-run mode, exercised
through the real orchestration code, not a separate mock of it.
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from rl.target_spec import TargetSpec

from experiments.run_autockt_pipeline import (
    PipelineCandidate,
    _main,
    filter_nominal_feasible,
    generate_candidates,
    select_final_design,
    validate_target,
)


class ValidateTargetTests(unittest.TestCase):
    def test_valid_target_has_no_problems(self):
        self.assertEqual(validate_target(TargetSpec.from_existing_thresholds()), [])
        self.assertEqual(validate_target(TargetSpec.from_hard_target()), [])

    def test_non_finite_value_is_a_problem(self):
        target = TargetSpec(
            dfe_locked_phase_eye_height_v=math.inf, dfe_eye_width_ui=0.4,
            dfe_min_margin_v=0.0, ctle_power_w=0.015,
        )
        problems = validate_target(target)
        self.assertTrue(any("not finite" in p for p in problems))

    def test_width_out_of_ui_range_is_a_problem(self):
        target = TargetSpec(
            dfe_locked_phase_eye_height_v=0.1, dfe_eye_width_ui=1.5,
            dfe_min_margin_v=0.0, ctle_power_w=0.015,
        )
        problems = validate_target(target)
        self.assertTrue(any("dfe_eye_width_ui" in p for p in problems))

    def test_nonpositive_power_is_a_problem(self):
        target = TargetSpec(
            dfe_locked_phase_eye_height_v=0.1, dfe_eye_width_ui=0.4,
            dfe_min_margin_v=0.0, ctle_power_w=0.0,
        )
        problems = validate_target(target)
        self.assertTrue(any("ctle_power_w" in p for p in problems))


class GenerateCandidatesTests(unittest.TestCase):
    def test_synthetic_backend_produces_the_requested_episode_count(self):
        candidates = generate_candidates(
            target=TargetSpec.from_existing_thresholds(), checkpoint_path=None,
            agent_seed=1, eval_seed=1, episodes=5, horizon=1, backend="synthetic",
            initial_indices_source="grid-center",
        )
        self.assertEqual(len(candidates), 5)
        for c in candidates:
            self.assertIsInstance(c, PipelineCandidate)
            self.assertEqual(set(c.parameters), {"rload_ohm", "rdeg_ohm", "cdeg_f", "itail_a", "dfe_tap_v"})
            self.assertGreaterEqual(c.steps, 1)

    def test_deterministic_given_the_same_seeds(self):
        kwargs = dict(
            target=TargetSpec.from_existing_thresholds(), checkpoint_path=None,
            agent_seed=7, eval_seed=7, episodes=4, horizon=1, backend="synthetic",
            initial_indices_source="grid-center",
        )
        a = generate_candidates(**kwargs)
        b = generate_candidates(**kwargs)
        self.assertEqual([c.parameters for c in a], [c.parameters for c in b])
        self.assertEqual([c.spec_satisfied for c in a], [c.spec_satisfied for c in b])

    def test_grid_center_init_reliably_finds_feasible_candidates_on_synthetic(self):
        # The synthetic benchmark's own "good region" is centered at
        # normalized 0.5 in every dimension -- the same point grid-center
        # initialization starts from -- so this combination should produce
        # a healthy feasible fraction, unlike the default 'verified'
        # (Random-Search-derived) starting point, which is unrelated to the
        # synthetic landscape's geometry.
        candidates = generate_candidates(
            target=TargetSpec.from_existing_thresholds(), checkpoint_path=None,
            agent_seed=1, eval_seed=1, episodes=10, horizon=1, backend="synthetic",
            initial_indices_source="grid-center",
        )
        feasible = filter_nominal_feasible(candidates)
        self.assertGreater(len(feasible), 0)


class FilterNominalFeasibleTests(unittest.TestCase):
    def test_keeps_only_spec_satisfied_candidates(self):
        candidates = [
            PipelineCandidate(0, {}, {}, 10.0, True, 1),
            PipelineCandidate(1, {}, {}, -1.0, False, 4),
        ]
        feasible = filter_nominal_feasible(candidates)
        self.assertEqual(len(feasible), 1)
        self.assertEqual(feasible[0].episode, 0)


class SelectFinalDesignTests(unittest.TestCase):
    def test_no_feasible_candidates_returns_none_selected_not_a_crash(self):
        selection = select_final_design([])
        self.assertIsNone(selection["selected"])
        self.assertIn("reason", selection)

    def test_nominal_only_selection_picks_a_design_and_labels_it(self):
        candidates = [
            PipelineCandidate(0, {"rload_ohm": 1000.0, "rdeg_ohm": 1000.0, "cdeg_f": 5e-13,
                                  "itail_a": 1e-4, "dfe_tap_v": 0.0},
                               {"ctle_power_w": 0.001, "dfe_locked_phase_eye_height_v": 1.0,
                                "dfe_eye_width_ui": 0.5, "dfe_min_margin_v": 0.3},
                               10.0, True, 1),
        ]
        selection = select_final_design(candidates)
        self.assertIsNotNone(selection["selected"])
        self.assertEqual(selection["selection_basis"], "nominal-only (no PVT conditions supplied)")
        self.assertIsNone(selection["pvt"])

    def test_pvt_path_calls_run_pvt_evaluation_and_ranks_by_pass_rate(self):
        from simulator.config import ProcessCorner, SimulationConditions
        from simulator.receiver import EvaluationFidelity, ReceiverEvaluation

        candidates = [
            PipelineCandidate(0, {"rload_ohm": 1000.0, "rdeg_ohm": 1000.0, "cdeg_f": 5e-13,
                                  "itail_a": 1e-4, "dfe_tap_v": 0.0}, {"ctle_power_w": 0.001}, 10.0, True, 1),
            PipelineCandidate(1, {"rload_ohm": 2000.0, "rdeg_ohm": 1000.0, "cdeg_f": 5e-13,
                                  "itail_a": 1e-4, "dfe_tap_v": 0.0}, {"ctle_power_w": 0.001}, 10.0, True, 1),
        ]
        conditions = (SimulationConditions(ProcessCorner.TT, 27.0, 1.8),)

        def fake_evaluate_pvt_grid(parameters, *, conditions, fidelity):
            # episode-0 design (rload=1000) always passes; episode-1 (rload=2000) always fails
            success = parameters.rload_ohm == 1000.0
            return tuple(
                ReceiverEvaluation(success, parameters, c, fidelity, (), {},
                                    None if success else "transient", 0.0, "id", {})
                for c in conditions
            )

        with patch("analysis.pvt_selection.evaluate_pvt_grid", side_effect=fake_evaluate_pvt_grid):
            selection = select_final_design(candidates, pvt_conditions=conditions)

        self.assertEqual(selection["selected"]["design_id"], "pipeline_ep0")
        self.assertEqual(selection["pvt"]["pass_rate"], 1.0)
        self.assertTrue(selection["pvt"]["met_minimum_pass_rate"])


class CLIIntegrationTests(unittest.TestCase):
    def test_full_synthetic_dry_run_produces_a_structured_result_with_schematic_and_spec(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            schematic = Path(tmp) / "schematic.spice"
            argv = [
                "run_autockt_pipeline.py", "--target-mode", "trivial", "--backend", "synthetic",
                "--episodes", "8", "--horizon", "1", "--initial-indices-source", "grid-center",
                "--agent-seed", "1", "--eval-seed", "1",
                "--output", str(output), "--export-schematic", str(schematic),
            ]
            with patch("sys.argv", argv):
                _main()

            self.assertTrue(output.is_file())
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["backend"], "synthetic")
            self.assertEqual(result["n_candidates_generated"], 8)
            self.assertIn("selection", result)
            if result["selection"]["selected"] is not None:
                self.assertTrue(schematic.is_file())
                self.assertIn("final_specification", result)
                self.assertIn("rows", result["final_specification"])

    def test_refuses_to_overwrite_existing_output(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            output.write_text("already here")
            argv = [
                "run_autockt_pipeline.py", "--backend", "synthetic", "--episodes", "1",
                "--output", str(output),
            ]
            with patch("sys.argv", argv):
                with self.assertRaises(FileExistsError):
                    _main()


if __name__ == "__main__":
    unittest.main()

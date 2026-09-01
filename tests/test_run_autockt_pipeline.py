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
    _pvt_result_from_selection,
    filter_nominal_feasible,
    generate_candidates,
    run_pipeline,
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

    def test_pvt_tie_is_broken_by_trade_off_preference(self):
        # Task 3 (NEXT IMPLEMENTATION CHUNK): select_final_design must
        # actually use analysis.pvt_selection.select_with_trade_off_
        # preference, not just rank_by_robustness, so a genuine PVT tie is
        # broken by the caller's stated preference rather than by
        # incidental list order.
        from simulator.config import ProcessCorner, SimulationConditions
        from simulator.receiver import EvaluationFidelity, ReceiverEvaluation

        candidates = [
            PipelineCandidate(0, {"rload_ohm": 1000.0, "rdeg_ohm": 1000.0, "cdeg_f": 5e-13,
                                  "itail_a": 1e-4, "dfe_tap_v": 0.0},
                               {"ctle_power_w": 0.01, "dfe_locked_phase_eye_height_v": 1.0,
                                "dfe_eye_width_ui": 0.5, "dfe_min_margin_v": 0.3}, 10.0, True, 1),
            PipelineCandidate(1, {"rload_ohm": 2000.0, "rdeg_ohm": 1000.0, "cdeg_f": 5e-13,
                                  "itail_a": 1e-4, "dfe_tap_v": 0.0},
                               {"ctle_power_w": 0.001, "dfe_locked_phase_eye_height_v": 1.0,
                                "dfe_eye_width_ui": 0.5, "dfe_min_margin_v": 0.3}, 10.0, True, 1),
        ]
        conditions = (SimulationConditions(ProcessCorner.TT, 27.0, 1.8),)

        def fake_evaluate_pvt_grid(parameters, *, conditions, fidelity):
            # both designs pass at every condition -- a genuine tie.
            return tuple(
                ReceiverEvaluation(True, parameters, c, fidelity, (), {}, None, 0.0, "id", {})
                for c in conditions
            )

        with patch("analysis.pvt_selection.evaluate_pvt_grid", side_effect=fake_evaluate_pvt_grid):
            lowest_power = select_final_design(
                candidates, pvt_conditions=conditions, trade_off_preference="lowest_power",
            )
            most_robust = select_final_design(
                candidates, pvt_conditions=conditions, trade_off_preference="most_robust",
            )

        # pipeline_ep1 has the lower ctle_power_w (0.001 vs 0.01).
        self.assertEqual(lowest_power["selected"]["design_id"], "pipeline_ep1")
        self.assertIn("lowest_power", lowest_power["selection_basis"])
        # 'most_robust' ignores trade-offs and returns the first by
        # robustness order among the tie -- pipeline_ep0.
        self.assertEqual(most_robust["selected"]["design_id"], "pipeline_ep0")


class PvtResultFlowsIntoFinalSpecificationTests(unittest.TestCase):
    """Integration requirement (NEXT IMPLEMENTATION CHUNK): the pipeline's
    own real PVT result (if any) must reach the final specification
    report's PVT row, not be silently dropped in favor of a fresh
    'NOT CLAIMED' -- and must not require re-running PVT a second time to
    get there.
    """

    def test_pvt_result_from_selection_reconstructs_the_summary(self):
        selection = {
            "selected": {"design_id": "x", "parameters": {}, "metrics": {}},
            "pvt": {
                "n_conditions": 4, "n_passing": 3, "pass_rate": 0.75,
                "met_minimum_pass_rate": False,
                "worst_case_conditions": [
                    {"corner": "ff", "vdd": 1.71, "temp_c": 125.0, "failed_stage": "transient"},
                ],
            },
        }
        result = _pvt_result_from_selection(selection)
        self.assertEqual(result.design_id, "x")
        self.assertEqual((result.n_conditions, result.n_passing, result.pass_rate), (4, 3, 0.75))
        self.assertEqual(len(result.worst_case_conditions), 1)
        self.assertEqual(result.worst_case_conditions[0].failed_stage, "transient")

    def test_no_pvt_selection_reconstructs_to_none(self):
        self.assertIsNone(_pvt_result_from_selection({"selected": {"design_id": "x"}, "pvt": None}))

    def test_run_pipeline_final_specification_reflects_real_pvt_result(self):
        from simulator.config import ProcessCorner, SimulationConditions
        from simulator.receiver import ReceiverEvaluation

        conditions = (SimulationConditions(ProcessCorner.TT, 27.0, 1.8),
                      SimulationConditions(ProcessCorner.FF, 125.0, 1.71))

        def fake_evaluate_pvt_grid(parameters, *, conditions, fidelity):
            return tuple(
                ReceiverEvaluation(True, parameters, c, fidelity, (), {}, None, 0.0, "id", {})
                for c in conditions
            )

        with patch("analysis.pvt_selection.evaluate_pvt_grid", side_effect=fake_evaluate_pvt_grid):
            result = run_pipeline(
                target=TargetSpec.from_existing_thresholds(), checkpoint_path=None,
                agent_seed=1, eval_seed=1, episodes=8, horizon=1, backend="synthetic",
                initial_indices_source="grid-center", pvt_conditions=conditions,
            )

        self.assertIsNotNone(result["selection"]["selected"])
        pvt_row = next(r for r in result["final_specification"]["rows"] if r["metric"] == "PVT (pass/total)")
        # must reflect the real 2-condition sweep, not "NOT CLAIMED".
        self.assertEqual(pvt_row["measured"], "2/2")
        self.assertEqual(pvt_row["verdict"], "PASS")


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

    def test_pvt_condition_set_flag_reaches_the_pvt_branch_via_the_cli(self):
        # Confirms the CLI-level gap (analysis.pvt_selection was previously
        # unreachable from _main()) is closed: --pvt-condition-set smoke
        # must cause select_final_design's PVT branch to run, and its
        # result must reach the final specification report's PVT row.
        from simulator.receiver import ReceiverEvaluation

        def fake_evaluate_pvt_grid(parameters, *, conditions, fidelity):
            return tuple(
                ReceiverEvaluation(True, parameters, c, fidelity, (), {}, None, 0.0, "id", {})
                for c in conditions
            )

        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            argv = [
                "run_autockt_pipeline.py", "--target-mode", "trivial", "--backend", "synthetic",
                "--episodes", "8", "--horizon", "1", "--initial-indices-source", "grid-center",
                "--agent-seed", "1", "--eval-seed", "1", "--pvt-condition-set", "smoke",
                "--output", str(output),
            ]
            with patch("analysis.pvt_selection.evaluate_pvt_grid", side_effect=fake_evaluate_pvt_grid):
                with patch("sys.argv", argv):
                    _main()

            result = json.loads(output.read_text(encoding="utf-8"))
            selection = result["selection"]
            if selection["selected"] is not None:
                self.assertIsNotNone(selection["pvt"])
                self.assertEqual(selection["pvt"]["n_conditions"], 2)
                self.assertIn("PVT-ranked", selection["selection_basis"])
                pvt_row = next(r for r in result["final_specification"]["rows"]
                                if r["metric"] == "PVT (pass/total)")
                self.assertEqual(pvt_row["measured"], "2/2")

    def test_target_json_flag_builds_an_arbitrary_target(self):
        import json as _json
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            custom_target = {
                "dfe_locked_phase_eye_height_v": 0.2, "dfe_eye_width_ui": 0.5,
                "dfe_min_margin_v": 0.05, "ctle_power_w": 0.012,
            }
            argv = [
                "run_autockt_pipeline.py", "--backend", "synthetic", "--episodes", "1",
                "--horizon", "1", "--initial-indices-source", "grid-center",
                "--target-json", _json.dumps(custom_target),
                "--output", str(output),
            ]
            with patch("sys.argv", argv):
                _main()
            result = _json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["target"], custom_target)

    def test_target_json_missing_field_raises_clearly(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            argv = [
                "run_autockt_pipeline.py", "--backend", "synthetic", "--episodes", "1",
                "--target-json", '{"dfe_locked_phase_eye_height_v": 0.2}',
                "--output", str(output),
            ]
            with patch("sys.argv", argv):
                with self.assertRaises(ValueError):
                    _main()

    def test_default_pvt_condition_set_is_none_unchanged_behavior(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            argv = [
                "run_autockt_pipeline.py", "--backend", "synthetic", "--episodes", "1",
                "--horizon", "1", "--initial-indices-source", "grid-center",
                "--output", str(output),
            ]
            with patch("sys.argv", argv):
                _main()
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertIsNone(result["selection"]["pvt"])


if __name__ == "__main__":
    unittest.main()

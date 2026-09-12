"""Regression coverage for strict acceptance and final candidate selection."""
import math
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from analysis.acceptance import acceptance_violations, gate
from analysis.final_specification import build_final_specification_report
from analysis.pvt_selection import PVTPointResult, summarize_pvt_results
from experiments.run_autockt_pipeline import PipelineCandidate, filter_nominal_feasible, run_pipeline
from rl.autockt_reward import autockt_reward
from rl.target_spec import TargetSpec
from simulator.config import PVT_GRID

TARGET = TargetSpec.from_existing_thresholds()
METRICS = dict(dfe_locked_phase_eye_height_v=0.5, dfe_eye_width_ui=0.8,
               dfe_min_margin_v=0.2, ctle_power_w=0.001, peaking_db=6.0,
               hd3_db=-40.0, input_referred_noise_vrms=0.0005)
PARAMS = dict(rload_ohm=1000., rdeg_ohm=1000., cdeg_f=5e-13, itail_a=1e-4, dfe_tap_v=0.)

class StrictAcceptanceTests(unittest.TestCase):
    def test_reward_bonus_does_not_accept_99mv(self):
        metrics = {**METRICS, "dfe_locked_phase_eye_height_v": 0.099}
        self.assertEqual(autockt_reward(metrics, TARGET, success=True), 10.)
        self.assertEqual(filter_nominal_feasible([PipelineCandidate(0, PARAMS, metrics, 10., True, 1)]), [])

    def test_missing_and_nonfinite_values_never_pass(self):
        for name in METRICS:
            for value in (None, math.nan, math.inf, -math.inf):
                with self.subTest(name=name, value=value):
                    self.assertIn(name, acceptance_violations({**METRICS, name: value}, TARGET, final=True))
        self.assertNotEqual(gate(math.nan, maximum=1), "PASS")

    def test_strict_boundaries_and_inclusive_peaking(self):
        for name, value in [("dfe_locked_phase_eye_height_v", .1), ("dfe_eye_width_ui", .4),
                            ("dfe_min_margin_v", 0), ("ctle_power_w", .015),
                            ("hd3_db", -30), ("input_referred_noise_vrms", .0015)]:
            self.assertIn(name, acceptance_violations({**METRICS, name: value}, TARGET, final=True))
        for value in (3., 12.):
            self.assertEqual(acceptance_violations({**METRICS, "peaking_db": value}, TARGET, final=True), [])

    def test_custom_target_and_simulation_failure_are_enforced(self):
        c = PipelineCandidate(0, PARAMS, METRICS, 10., True, 1)
        self.assertEqual(filter_nominal_feasible([c], TargetSpec.from_hard_target()), [])
        self.assertEqual(filter_nominal_feasible([replace(c, simulation_success=False)]), [])
        self.assertEqual(filter_nominal_feasible([replace(c, spec_satisfied=False)]), [replace(c, spec_satisfied=False)])

    def test_report_boundaries_and_nan_fail(self):
        values = {**METRICS, "dfe_locked_phase_eye_height_v": .1, "ctle_power_w": math.nan}
        report = build_final_specification_report(design_id="x", parameters=PARAMS, nominal_metrics=values, nominal_source="test")
        rows = {r["metric"]: r for r in report["rows"]}
        self.assertEqual(rows["Eye height (V)"]["verdict"], "FAIL")
        self.assertEqual(rows["Power (W)"]["verdict"], "FAIL")

    def test_full_grid_is_available_in_pipeline_and_ui(self):
        from experiments.run_autockt_pipeline import PVT_CONDITION_SETS
        from experiments.web_ui import INDEX_HTML
        self.assertEqual(PVT_CONDITION_SETS["full60"], PVT_GRID)
        self.assertIn('value="full60"', INDEX_HTML)
        self.assertIn("The 27-point sweep omits SF and FS", INDEX_HTML)

    def test_only_complete_pvt_grid_can_pass(self):
        points = [PVTPointResult(c.process_corner.value, c.supply_v, c.temperature_c, True, None) for c in PVT_GRID]
        for subset, verdict in [(points, "PASS"), (points[:2], "NOT CLAIMED"),
                                ([p for p in points if p.process_corner not in ("sf", "fs")], "NOT CLAIMED"),
                                ([points[0]] * 60, "NOT CLAIMED"), ([], "NOT CLAIMED"),
                                ([replace(points[0], success=False)] + points[1:], "FAIL")]:
            report = build_final_specification_report(design_id="x", parameters=PARAMS, nominal_metrics=METRICS,
                nominal_source="test", pvt_result=summarize_pvt_results("x", subset))
            self.assertEqual(report["rows"][-1]["verdict"], verdict)

class FinalValidationTests(unittest.TestCase):
    def candidates(self):
        return [PipelineCandidate(i, {**PARAMS, "rload_ohm": 1000. + i}, METRICS, 10., True, 1) for i in range(2)]

    def test_failed_candidate_replaced_and_only_passing_design_exported(self):
        def evaluate(params):
            ok = params["rload_ohm"] == 1001.
            return dict(success=ok, metrics=METRICS, failed_stage=None if ok else "hd3")
        with TemporaryDirectory() as directory, patch("experiments.run_autockt_pipeline.generate_candidates", return_value=self.candidates()), patch("experiments.run_autockt_pipeline.measure_hd3_and_noise", side_effect=evaluate) as check:
            result = run_pipeline(target=TARGET, checkpoint_path=None, measure_hd3_noise_flag=True,
                                  export_schematic_to=Path(directory) / "final.spice")
            self.assertEqual(check.call_count, 2)
            self.assertEqual(result["selection"]["selected"]["parameters"]["rload_ohm"], 1001.)
            self.assertTrue(Path(result["schematic_path"]).is_file())
            self.assertFalse(result["hd3_noise_refinement"]["attempts"][0]["success"])

    def test_all_fail_no_export_or_report_even_if_simulator_claims_success(self):
        with TemporaryDirectory() as directory, patch("experiments.run_autockt_pipeline.generate_candidates", return_value=self.candidates()), patch("experiments.run_autockt_pipeline.measure_hd3_and_noise", return_value=dict(success=True, metrics={**METRICS, "hd3_db": -20.}, failed_stage=None)):
            output = Path(directory) / "final.spice"
            result = run_pipeline(target=TARGET, checkpoint_path=None, measure_hd3_noise_flag=True, export_schematic_to=output)
            self.assertIsNone(result["selection"]["selected"])
            self.assertNotIn("final_specification", result)
            self.assertFalse(output.exists())
            self.assertEqual(len(result["hd3_noise_refinement"]["attempts"]), 2)

    def test_pvt_is_reused_when_final_validation_rejects_a_design(self):
        from analysis.pvt_selection import summarize_pvt_results
        def pvt(design, conditions, fidelity):
            return summarize_pvt_results(design.design_id, [PVTPointResult("tt", 1.71, 0., True, None)])
        outcomes = [dict(success=False, metrics=METRICS, failed_stage="transient"), dict(success=True, metrics=METRICS, failed_stage=None)]
        with patch("experiments.run_autockt_pipeline.generate_candidates", return_value=self.candidates()), patch("experiments.run_autockt_pipeline.measure_hd3_and_noise", side_effect=outcomes), patch("experiments.run_autockt_pipeline.run_pvt_evaluation", side_effect=pvt) as check:
            result = run_pipeline(target=TARGET, checkpoint_path=None, measure_hd3_noise_flag=True, pvt_conditions=PVT_GRID[:1])
            self.assertEqual(check.call_count, 2)
            self.assertIsNotNone(result["selection"]["selected"])
            self.assertEqual(len(result["selection"]["pvt"]["points"]), 1)

    def test_no_candidate_passing_requested_pvt_returns_none(self):
        def pvt(design, conditions, fidelity):
            return summarize_pvt_results(design.design_id, [PVTPointResult("tt", 1.71, 0., False, "hd3")])
        with patch("experiments.run_autockt_pipeline.generate_candidates", return_value=self.candidates()), patch("experiments.run_autockt_pipeline.run_pvt_evaluation", side_effect=pvt):
            result = run_pipeline(target=TARGET, checkpoint_path=None, pvt_conditions=PVT_GRID[:1])
            self.assertIsNone(result["selection"]["selected"])

if __name__ == "__main__":
    unittest.main()

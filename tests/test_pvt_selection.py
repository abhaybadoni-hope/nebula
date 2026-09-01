"""Tests for analysis/pvt_selection.py -- Task 4 (PVT-aware candidate
selection). SPICE-free: run_pvt_evaluation is tested by mocking
evaluate_pvt_grid; the rest is pure aggregation/ranking over already-loaded
data (including the REAL results/design_a_pvt_minimal27.jsonl).
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from simulator.config import ProcessCorner, SimulationConditions
from simulator.receiver import EvaluationFidelity, ReceiverEvaluation, ReceiverParameters

from analysis.design_catalog import FeasibleDesign
from analysis.pvt_selection import (
    PVTPointResult,
    load_pvt_results_from_jsonl,
    rank_by_robustness,
    run_pvt_evaluation,
    select_final_designs,
    summarize_pvt_results,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class SummarizePvtResultsTests(unittest.TestCase):
    def test_pass_rate_and_worst_case_conditions(self):
        points = [
            PVTPointResult("tt", 1.8, 27.0, True, None),
            PVTPointResult("ff", 1.71, 125.0, False, "transient"),
            PVTPointResult("ss", 1.89, 0.0, True, None),
        ]
        result = summarize_pvt_results("x", points)
        self.assertEqual(result.n_conditions, 3)
        self.assertEqual(result.n_passing, 2)
        self.assertAlmostEqual(result.pass_rate, 2 / 3)
        self.assertEqual(len(result.worst_case_conditions), 1)
        self.assertEqual(result.worst_case_conditions[0].process_corner, "ff")

    def test_empty_points_gives_zero_pass_rate_not_a_crash(self):
        result = summarize_pvt_results("x", [])
        self.assertEqual(result.pass_rate, 0.0)
        self.assertEqual(result.n_conditions, 0)


class LoadFromRealDataTests(unittest.TestCase):
    def test_matches_the_known_23_of_27_result(self):
        path = REPOSITORY_ROOT / "results" / "design_a_pvt_minimal27.jsonl"
        if not path.is_file():
            self.skipTest("fixture not present")
        result = load_pvt_results_from_jsonl("design_a", path)
        self.assertEqual(result.n_conditions, 27)
        self.assertEqual(result.n_passing, 23)
        self.assertAlmostEqual(result.pass_rate, 23 / 27)
        self.assertEqual(len(result.worst_case_conditions), 4)
        for point in result.worst_case_conditions:
            self.assertEqual(point.process_corner, "ff")
            self.assertEqual(point.failed_stage, "transient")


class RunPvtEvaluationTests(unittest.TestCase):
    def test_calls_evaluate_pvt_grid_with_the_given_conditions_and_summarizes(self):
        design = FeasibleDesign(
            design_id="test_design", source_file="f", source_description="d",
            parameters={"rload_ohm": 1000.0, "rdeg_ohm": 1000.0, "cdeg_f": 5e-13,
                        "itail_a": 1e-4, "dfe_tap_v": 0.0},
            metrics={}, reward=10.0,
        )
        conditions = (
            SimulationConditions(ProcessCorner.TT, 27.0, 1.8),
            SimulationConditions(ProcessCorner.FF, 125.0, 1.71),
        )

        def fake_evaluate_pvt_grid(parameters, *, conditions, fidelity):
            results = []
            for i, c in enumerate(conditions):
                success = i == 0  # first passes, second fails
                results.append(ReceiverEvaluation(
                    success, parameters, c, fidelity, (), {},
                    None if success else "transient", 0.0, f"id-{i}", {},
                ))
            return tuple(results)

        with patch("analysis.pvt_selection.evaluate_pvt_grid", side_effect=fake_evaluate_pvt_grid):
            result = run_pvt_evaluation(design, conditions)

        self.assertEqual(result.design_id, "test_design")
        self.assertEqual(result.n_conditions, 2)
        self.assertEqual(result.n_passing, 1)
        self.assertEqual(len(result.worst_case_conditions), 1)
        self.assertEqual(result.worst_case_conditions[0].process_corner, "ff")


class RankAndSelectTests(unittest.TestCase):
    def test_rank_by_robustness_orders_by_pass_rate_descending(self):
        low = summarize_pvt_results("low", [PVTPointResult("tt", 1.8, 27.0, False, "ac")])
        high = summarize_pvt_results("high", [PVTPointResult("tt", 1.8, 27.0, True, None)])
        ranked = rank_by_robustness([low, high])
        self.assertEqual([r.design_id for r in ranked], ["high", "low"])

    def test_select_final_designs_respects_minimum_pass_rate(self):
        partial = summarize_pvt_results("partial", [
            PVTPointResult("tt", 1.8, 27.0, True, None),
            PVTPointResult("ff", 1.71, 125.0, False, "transient"),
        ])
        full = summarize_pvt_results("full", [
            PVTPointResult("tt", 1.8, 27.0, True, None),
            PVTPointResult("ff", 1.71, 125.0, True, None),
        ])
        selected = select_final_designs([partial, full], minimum_pass_rate=1.0, top_n=1)
        self.assertEqual(selected[0].design_id, "full")

    def test_select_final_designs_does_not_hide_when_no_candidate_meets_the_bar(self):
        partial = summarize_pvt_results("partial", [
            PVTPointResult("tt", 1.8, 27.0, True, None),
            PVTPointResult("ff", 1.71, 125.0, False, "transient"),
        ])
        selected = select_final_designs([partial], minimum_pass_rate=1.0, top_n=1)
        self.assertEqual(len(selected), 1)  # still returns the best available, not empty
        self.assertLess(selected[0].pass_rate, 1.0)


if __name__ == "__main__":
    unittest.main()

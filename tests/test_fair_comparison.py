"""Tests for analysis/fair_comparison.py -- the uniform (autockt_reward vs.
the trivial target) recomputation used to make PPO/Random Search/CEM
success rates actually comparable. SPICE-free: parses existing results
files and small in-memory fixtures only.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from analysis.fair_comparison import (
    COMPARISON_TARGET,
    summarize_cem,
    summarize_ppo_subset,
    summarize_random_search,
    summarize_uniform,
    uniform_score,
)
from rl.autockt_reward import FAILURE_REWARD, TERMINAL_BONUS
from rl.target_spec import EXISTING_THRESHOLDS, TargetSpec

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# Design A, results/receiver_random_search_20_seed123.jsonl candidate_index 8
# -- the repo's one independently-confirmed, real-SPICE-verified successful
# design, already documented (docs/autockt-mapping.md sec 10) to clear the
# trivial target by 2x-15x on every dimension.
KNOWN_GOOD_METRICS = {
    "dfe_locked_phase_eye_height_v": 1.5215334399802445,
    "dfe_eye_width_ui": 0.8699999999999999,
    "dfe_min_margin_v": 0.5174474651666459,
    "ctle_power_w": 0.0010850994,
}


class UniformScoreTests(unittest.TestCase):
    def test_comparison_target_is_the_trivial_existing_thresholds(self):
        self.assertEqual(COMPARISON_TARGET, TargetSpec(**EXISTING_THRESHOLDS))

    def test_known_good_design_scores_the_terminal_bonus(self):
        satisfied, reward = uniform_score(KNOWN_GOOD_METRICS, success=True)
        self.assertTrue(satisfied)
        self.assertEqual(reward, TERMINAL_BONUS)

    def test_failed_evaluation_never_scoreable_as_success(self):
        satisfied, reward = uniform_score(KNOWN_GOOD_METRICS, success=False)
        self.assertFalse(satisfied)
        self.assertEqual(reward, FAILURE_REWARD)

    def test_missing_metrics_is_not_fabricated_as_success(self):
        satisfied, reward = uniform_score(None, success=True)
        self.assertFalse(satisfied)
        self.assertEqual(reward, FAILURE_REWARD)

    def test_success_with_metrics_missing_all_four_specs_fails_uniform_criterion(self):
        # A design that only clears margin/power but not height/width --
        # evaluation.success=True (reward_v1 would score it), but should
        # NOT count as uniformly successful.
        weak_metrics = {
            "dfe_locked_phase_eye_height_v": 0.02,  # below the 0.1 trivial threshold
            "dfe_eye_width_ui": 0.1,  # below the 0.4 trivial threshold
            "dfe_min_margin_v": 0.01,
            "ctle_power_w": 0.001,
        }
        satisfied, reward = uniform_score(weak_metrics, success=True)
        self.assertFalse(satisfied)
        self.assertLess(reward, TERMINAL_BONUS)


class RandomSearchGroundTruthTests(unittest.TestCase):
    """Anchors the uniform recomputation against the actual, already
    fully-characterized seed=123 random search run.
    """

    def test_seed123_uniform_success_matches_the_one_known_good_design(self):
        path = REPOSITORY_ROOT / "results" / "receiver_random_search_20_seed123.jsonl"
        if not path.is_file():
            self.skipTest("fixture result file not present in this checkout")
        summary = summarize_random_search(path)
        self.assertEqual(summary.n_evaluations, 20)
        # This run's raw rows all carry metrics, so every candidate should
        # be uniformly scoreable.
        self.assertEqual(summary.uniform_scoreable_fraction, 1.0)
        # Independently known (docs/autockt-mapping.md sec 10): exactly one
        # candidate (index 8) is a real success that clears the trivial
        # target comfortably; the uniform criterion should agree.
        self.assertEqual(summary.uniform_success_rate, 1 / 20)
        self.assertEqual(summary.best_uniform_reward, TERMINAL_BONUS)


class SummarizeUniformTests(unittest.TestCase):
    def test_running_best_uniform_reward_is_nondecreasing(self):
        from analysis.fair_comparison import UniformEvaluation

        evaluations = [
            UniformEvaluation("x", 0, False, -100.0, True, False, -1.0, None),
            UniformEvaluation("x", 1, True, 50.0, True, False, 3.0, None),
            UniformEvaluation("x", 2, True, 100.0, True, True, 10.0, None),
            UniformEvaluation("x", 3, False, -100.0, True, False, -1.0, None),
        ]
        summary = summarize_uniform(evaluations, method="x", source_file="fixture")
        running = summary.running_best_uniform_reward
        for earlier, later in zip(running, running[1:]):
            self.assertLessEqual(earlier, later)
        self.assertEqual(running[-1], 10.0)
        self.assertEqual(summary.uniform_evaluations_to_first_success, 3)

    def test_empty_evaluations_gives_none_metrics(self):
        summary = summarize_uniform([], method="x", source_file="fixture")
        self.assertEqual(summary.n_evaluations, 0)
        self.assertIsNone(summary.uniform_success_rate)
        self.assertIn("no evaluations", summary.caveats)


class CemSummaryCaveatTests(unittest.TestCase):
    def _write(self, directory, rows):
        path = Path(directory) / "cem.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        return path

    def test_warm_started_incomplete_run_is_flagged_on_both_axes(self):
        rows = [
            {"iteration": 1, "evaluation": 1, "reward": 100.0, "action": [0.0] * 5,
             "success": True, "failure_stage": None},
        ]
        with TemporaryDirectory() as tmp:
            path = self._write(tmp, rows)
            summary = summarize_cem(path, warm_started=True, complete=False, planned=30)
        self.assertTrue(any("WARM-STARTED" in c for c in summary.caveats))
        self.assertTrue(any("INCOMPLETE" in c for c in summary.caveats))

    def test_complete_unbiased_run_has_no_such_caveats(self):
        rows = [
            {"iteration": 1, "evaluation": 1, "reward": 100.0, "action": [0.0] * 5,
             "success": True, "failure_stage": None},
        ]
        with TemporaryDirectory() as tmp:
            path = self._write(tmp, rows)
            summary = summarize_cem(path, warm_started=False, complete=True, planned=1)
        self.assertFalse(any("WARM-STARTED" in c for c in summary.caveats))
        self.assertFalse(any("INCOMPLETE" in c for c in summary.caveats))


class PPOSubsetTests(unittest.TestCase):
    def test_filters_to_only_matching_target_rows(self):
        trivial = TargetSpec.from_existing_thresholds()
        hard = TargetSpec.from_hard_target()
        rows = [
            {"episode": 0, "step": 1, "parameters": {}, "indices": [0] * 5, "reward": 10.0,
             "success": True, "spec_satisfied": True, "failure_stage": None, "target": trivial.as_dict()},
            {"episode": 1, "step": 1, "parameters": {}, "indices": [0] * 5, "reward": -1.0,
             "success": False, "spec_satisfied": False, "failure_stage": "ac", "target": hard.as_dict()},
            {"episode": 2, "step": 1, "parameters": {}, "indices": [0] * 5, "reward": 10.0,
             "success": True, "spec_satisfied": True, "failure_stage": None, "target": trivial.as_dict()},
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            with path.open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
            summary = summarize_ppo_subset(path, target=trivial, label="trivial")
        self.assertEqual(summary.n_evaluations, 2)
        self.assertEqual(summary.native_success_rate, 1.0)


if __name__ == "__main__":
    unittest.main()

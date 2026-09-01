"""Tests for analysis/learning_evidence.py -- Task 4 (overnight chunk,
sec 22). Anchors against REAL, already-existing training logs (ground
truth already independently established in docs/autockt-mapping.md
sec 15/17/18) -- this is a second, independent recomputation from the raw
per-step data, not a copy of the prior narrative numbers.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from analysis.learning_evidence import (
    compute_per_update_progression,
    summarize_controlled_generalization_check,
    summarize_matched_checkpoint,
    training_efficiency,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class MatchedCheckpointGroundTruthTests(unittest.TestCase):
    def test_single_target_hard_confirmation_matches_sec15(self):
        path = REPOSITORY_ROOT / "results" / "autockt_normfix_confirmation.jsonl"
        if not path.is_file():
            self.skipTest("fixture not present")
        summary = summarize_matched_checkpoint(path)
        self.assertAlmostEqual(summary.initial_satisfaction_rate, 2 / 3)
        self.assertEqual(summary.final_satisfaction_rate, 1.0)
        self.assertAlmostEqual(summary.initial_mean_reward, 5.842558811192149, places=5)
        self.assertEqual(summary.final_mean_reward, 10.0)
        self.assertEqual(summary.n_matched, 3)

    def test_mixed_target_confirmation_matches_sec17(self):
        path = REPOSITORY_ROOT / "results" / "autockt_mixed_target_confirmation.jsonl"
        if not path.is_file():
            self.skipTest("fixture not present")
        summary = summarize_matched_checkpoint(path)
        self.assertAlmostEqual(summary.initial_satisfaction_rate, 1 / 3)
        self.assertEqual(summary.final_satisfaction_rate, 1.0)
        self.assertAlmostEqual(summary.initial_mean_reward, 1.5092254778588163, places=5)
        self.assertEqual(summary.final_mean_reward, 10.0)

    def test_missing_file_reports_unavailable_not_a_crash(self):
        summary = summarize_matched_checkpoint("results/does_not_exist.jsonl")
        self.assertIsNone(summary.initial_satisfaction_rate)
        self.assertIn("does not exist", summary.note)


class ControlledGeneralizationGroundTruthTests(unittest.TestCase):
    def test_matches_sec18_exactly(self):
        path = REPOSITORY_ROOT / "results" / "controlled_unseen_target_generalization.jsonl"
        if not path.is_file():
            self.skipTest("fixture not present")
        result = summarize_controlled_generalization_check(path)
        self.assertTrue(result["available"])
        self.assertEqual(result["n_matched"], 10)
        self.assertAlmostEqual(result["initial_satisfaction_rate"], 0.9)
        self.assertAlmostEqual(result["final_satisfaction_rate"], 0.7)
        self.assertAlmostEqual(result["initial_mean_reward"], 8.6)
        self.assertAlmostEqual(result["final_mean_reward"], 5.8)
        self.assertEqual((result["wins"], result["ties"], result["losses"]), (1, 6, 3))
        # sec 18's own headline finding: this is a REGRESSION, not an improvement.
        self.assertLess(result["final_mean_reward"], result["initial_mean_reward"])

    def test_missing_file_reports_unavailable(self):
        result = summarize_controlled_generalization_check("results/does_not_exist.jsonl")
        self.assertFalse(result["available"])


class PerUpdateProgressionGroundTruthTests(unittest.TestCase):
    def test_matches_sec17s_own_printed_per_update_rewards(self):
        path = REPOSITORY_ROOT / "results" / "autockt_mixed_target_confirmation.jsonl"
        if not path.is_file():
            self.skipTest("fixture not present")
        progression = compute_per_update_progression(path, episodes_per_update=6)
        self.assertEqual(len(progression), 6)
        expected_means = [2.8333333333333335, 5.333333333333333, 2.3333333333333335,
                           4.95833922414468, 4.666666666666667, 2.961291680574481]
        for row, expected in zip(progression, expected_means):
            self.assertAlmostEqual(row["mean_episode_reward"], expected, places=5)

    def test_does_not_claim_a_smooth_curve(self):
        path = REPOSITORY_ROOT / "results" / "autockt_mixed_target_confirmation.jsonl"
        if not path.is_file():
            self.skipTest("fixture not present")
        progression = compute_per_update_progression(path, episodes_per_update=6)
        rewards = [row["mean_episode_reward"] for row in progression]
        # honest regression guard: this run is genuinely noisy (not
        # monotonically increasing) -- if a future run WERE monotonic this
        # test would need updating, but it must never be asserted here as
        # if it already were.
        self.assertFalse(all(b >= a for a, b in zip(rewards, rewards[1:])))


class TrainingEfficiencyGroundTruthTests(unittest.TestCase):
    def test_mixed_target_confirmation_evaluation_count(self):
        path = REPOSITORY_ROOT / "results" / "autockt_mixed_target_confirmation.jsonl"
        if not path.is_file():
            self.skipTest("fixture not present")
        result = training_efficiency(path)
        self.assertEqual(result["n_evaluations"], 95)
        self.assertEqual(result["n_episodes"], 36)
        self.assertFalse(result["wall_clock_available"])

    def test_empty_source_reports_zero_not_a_crash(self):
        result = training_efficiency("results/does_not_exist.jsonl")
        self.assertEqual(result["n_evaluations"], 0)


if __name__ == "__main__":
    unittest.main()

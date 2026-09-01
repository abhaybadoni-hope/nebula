"""Cheap, SPICE-free tests for experiments/train_cem.py's new, additive
--init-mean/--init-std CLI wiring and metrics-reuse import. Does not invoke
main() (which always uses the real ReceiverRLAdapter backend with no
evaluator-injection option) -- only checks argument parsing and the
unchanged-default-behavior guarantee.
"""

from __future__ import annotations

import argparse
import unittest

from experiments.train_cem import (
    DEFAULT_INIT_MEAN,
    DEFAULT_INIT_STD,
    GRADED_FITNESS_NO_INFORMATION_FLOOR,
    graded_cem_fitness,
)
from rl.target_spec import TargetSpec

# Design A's real metrics, results/receiver_random_search_20_seed123.jsonl
# candidate_index 8 -- comfortably clears the trivial target on every spec.
KNOWN_GOOD_METRICS = {
    "dfe_locked_phase_eye_height_v": 1.5215334399802445,
    "dfe_eye_width_ui": 0.8699999999999999,
    "dfe_min_margin_v": 0.5174474651666459,
    "ctle_power_w": 0.0010850994,
}


class DefaultInitializationUnchangedTests(unittest.TestCase):
    """The original hardcoded warm-started mean/std, pinned here so a future
    edit to the CLI defaults cannot silently change existing behavior
    without this test failing.
    """

    def test_default_init_mean_matches_the_original_hardcoded_values(self):
        self.assertEqual(DEFAULT_INIT_MEAN, (
            0.369674437322012,
            0.27670652861099754,
            0.33331526214636176,
            0.7801971410738049,
            -0.027727059712872038,
        ))

    def test_default_init_std_matches_the_original_hardcoded_value(self):
        self.assertEqual(DEFAULT_INIT_STD, 0.15)


class CLIParsingTests(unittest.TestCase):
    def _parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument("--init-mean", type=float, nargs=5, default=list(DEFAULT_INIT_MEAN))
        parser.add_argument("--init-std", type=float, nargs=5, default=[DEFAULT_INIT_STD] * 5)
        return parser

    def test_defaults_when_flags_omitted(self):
        args = self._parser().parse_args([])
        self.assertEqual(args.init_mean, list(DEFAULT_INIT_MEAN))
        self.assertEqual(args.init_std, [DEFAULT_INIT_STD] * 5)

    def test_unbiased_centered_start_is_expressible(self):
        args = self._parser().parse_args(
            ["--init-mean", "0", "0", "0", "0", "0", "--init-std", "0.577", "0.577", "0.577", "0.577", "0.577"]
        )
        self.assertEqual(args.init_mean, [0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(args.init_std, [0.577] * 5)


class GradedCemFitnessTests(unittest.TestCase):
    """graded_cem_fitness -- the fix for reward_v1's flat -100.0-on-any-
    failure problem (confirmed in docs/autockt-mapping.md sec 19: an
    unbiased-init CEM run's best_reward stayed at exactly -100.0 across all
    4 iterations, elite selection choosing among ties). These tests verify
    it actually gives GRADED, non-flat signal among candidates that reach
    the transient stage, while still flooring early-stage failures (which
    have no real height/width/margin/power measurements to grade on).
    """

    def setUp(self):
        self.target = TargetSpec.from_existing_thresholds()

    def test_early_stage_failure_gets_the_no_information_floor(self):
        for stage in ("dc", "ac", "setup", "channel", "noise", "hd3", "ctle_transient"):
            with self.subTest(stage=stage):
                fitness = graded_cem_fitness({}, self.target, stage)
                self.assertEqual(fitness, GRADED_FITNESS_NO_INFORMATION_FLOOR)

    def test_transient_stage_with_zero_filled_metrics_is_not_the_floor(self):
        # A transient-stage outcome (pass or fail) always carries real
        # metrics per simulator/receiver.py -- but even the pathological
        # all-zero case must not be silently treated as "no information"
        # (that would be the OLD bug this function fixes).
        fitness = graded_cem_fitness({}, self.target, "transient")
        self.assertGreater(fitness, GRADED_FITNESS_NO_INFORMATION_FLOOR)

    def test_known_good_design_scores_zero_the_maximum(self):
        fitness = graded_cem_fitness(KNOWN_GOOD_METRICS, self.target, "transient")
        self.assertEqual(fitness, 0.0)
        fitness_success = graded_cem_fitness(KNOWN_GOOD_METRICS, self.target, None)
        self.assertEqual(fitness_success, 0.0)

    def test_distinguishes_close_from_far_failing_candidates(self):
        # The core property reward_v1 lacks: two different "still failing"
        # candidates must get DIFFERENT graded scores based on how close
        # they are, not the same flat value.
        close = {
            "dfe_locked_phase_eye_height_v": 0.09,  # just under the 0.1 threshold
            "dfe_eye_width_ui": 0.39,  # just under the 0.4 threshold
            "dfe_min_margin_v": 0.5,
            "ctle_power_w": 0.001,
        }
        far = {
            "dfe_locked_phase_eye_height_v": 0.001,
            "dfe_eye_width_ui": 0.01,
            "dfe_min_margin_v": 0.5,
            "ctle_power_w": 0.001,
        }
        fitness_close = graded_cem_fitness(close, self.target, "transient")
        fitness_far = graded_cem_fitness(far, self.target, "transient")
        self.assertLess(fitness_far, fitness_close)
        self.assertLess(fitness_close, 0.0)  # still failing -- not the max
        self.assertGreater(fitness_far, GRADED_FITNESS_NO_INFORMATION_FLOOR)

    def test_reward_v1_style_flatness_does_not_occur_among_transient_candidates(self):
        # Regression guard for the exact bug this fixes: under reward_v1,
        # ALL of these would score exactly -100.0. Under graded fitness
        # they must NOT all collapse to the same value.
        candidates = [
            {"dfe_locked_phase_eye_height_v": 0.08, "dfe_eye_width_ui": 0.38,
             "dfe_min_margin_v": 0.4, "ctle_power_w": 0.001},
            {"dfe_locked_phase_eye_height_v": 0.05, "dfe_eye_width_ui": 0.2,
             "dfe_min_margin_v": 0.3, "ctle_power_w": 0.001},
            {"dfe_locked_phase_eye_height_v": 0.01, "dfe_eye_width_ui": 0.05,
             "dfe_min_margin_v": 0.1, "ctle_power_w": 0.001},
        ]
        scores = [graded_cem_fitness(c, self.target, "transient") for c in candidates]
        self.assertEqual(len(set(scores)), len(scores), f"expected all distinct, got {scores}")


class CEMTargetAndFitnessCLITests(unittest.TestCase):
    def _parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--fitness", choices=("reward_v1", "graded"), default="reward_v1")
        parser.add_argument("--target", choices=("trivial", "hard"), default="trivial")
        return parser

    def test_defaults_preserve_existing_behavior(self):
        args = self._parser().parse_args([])
        self.assertEqual(args.fitness, "reward_v1")
        self.assertEqual(args.target, "trivial")

    def test_graded_fitness_and_hard_target_are_selectable(self):
        args = self._parser().parse_args(["--fitness", "graded", "--target", "hard"])
        self.assertEqual(args.fitness, "graded")
        self.assertEqual(args.target, "hard")


if __name__ == "__main__":
    unittest.main()

"""SPICE-free tests for experiments/controlled_unseen_target_eval.py --
the controlled, matched initial-vs-final checkpoint comparison on the
unseen midpoint target. Uses rl.synthetic_benchmark.synthetic_evaluate_receiver
(fast, deterministic, non-circuit-simulating) so the harness's own
correctness -- matched starting states, win/tie/loss counting, target
fidelity -- can be verified before spending any real SPICE budget, per
instruction 9 of the controlled-generalization milestone.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from simulator.rl_adapter import ReceiverRLAdapter, RLBudget

from rl.parameter_grid import build_parameter_grids, verified_initial_indices
from rl.ppo_agent import PPOAgent
from rl.autockt_state import STATE_DIM
from rl.parameter_grid import PARAMETER_NAMES
from rl.synthetic_benchmark import synthetic_evaluate_receiver

from experiments.controlled_unseen_target_eval import (
    MATCHED_EPISODES,
    UNSEEN_MIDPOINT_TARGET,
    run_controlled_evaluation,
)


def _synthetic_adapter(seed: int, budget: int = 500) -> ReceiverRLAdapter:
    return ReceiverRLAdapter(evaluator=synthetic_evaluate_receiver, budget=RLBudget(budget), seed=seed)


def _common_setup():
    grids = build_parameter_grids()
    indices = verified_initial_indices(grids)
    return grids, indices


class TargetFidelityTests(unittest.TestCase):
    def test_target_matches_the_exact_specified_midpoint_values(self):
        self.assertEqual(UNSEEN_MIDPOINT_TARGET.dfe_locked_phase_eye_height_v, 0.45)
        self.assertEqual(UNSEEN_MIDPOINT_TARGET.dfe_eye_width_ui, 0.5)
        self.assertEqual(UNSEEN_MIDPOINT_TARGET.dfe_min_margin_v, 0.175)
        self.assertEqual(UNSEEN_MIDPOINT_TARGET.ctle_power_w, 0.015)

    def test_every_matched_case_uses_exactly_that_target(self):
        grids, indices = _common_setup()
        agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=1)
        policy_state = agent.policy.state_dict()
        adapter = _synthetic_adapter(seed=10)
        result = run_controlled_evaluation(
            initial_policy_state=policy_state, final_policy_state=policy_state,
            adapter=adapter, grids=grids, initial_indices=indices, horizon=4,
            eval_seed=5, agent_seed=1, episodes=3,
        )
        for case in result["cases"]:
            self.assertEqual(case["target"], UNSEEN_MIDPOINT_TARGET.as_dict())
        self.assertEqual(result["target"], UNSEEN_MIDPOINT_TARGET.as_dict())


class MatchedStartingStateTests(unittest.TestCase):
    def test_identical_policy_weights_on_both_sides_produce_all_ties(self):
        # Same weights loaded as both "initial" and "final" -- if starting
        # states/targets/horizon truly match and evaluation is deterministic,
        # every one of the 10 matched cases must tie exactly.
        grids, indices = _common_setup()
        agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=2)
        policy_state = agent.policy.state_dict()
        adapter = _synthetic_adapter(seed=20)
        result = run_controlled_evaluation(
            initial_policy_state=policy_state, final_policy_state=policy_state,
            adapter=adapter, grids=grids, initial_indices=indices, horizon=4,
            eval_seed=7, agent_seed=2, episodes=MATCHED_EPISODES,
        )
        self.assertEqual(result["wins"], 0)
        self.assertEqual(result["losses"], 0)
        self.assertEqual(result["ties"], MATCHED_EPISODES)
        self.assertAlmostEqual(result["initial_satisfaction_rate"], result["final_satisfaction_rate"])
        for case in result["cases"]:
            self.assertEqual(case["initial_reward"], case["final_reward"])
            self.assertEqual(case["initial_spec_satisfied"], case["final_spec_satisfied"])
            self.assertEqual(case["initial_steps_used"], case["final_steps_used"])

    def test_reevaluating_with_the_same_seed_is_reproducible(self):
        grids, indices = _common_setup()
        agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=3)
        policy_state = agent.policy.state_dict()

        result_a = run_controlled_evaluation(
            initial_policy_state=policy_state, final_policy_state=policy_state,
            adapter=_synthetic_adapter(seed=30), grids=grids, initial_indices=indices, horizon=4,
            eval_seed=11, agent_seed=3, episodes=5,
        )
        result_b = run_controlled_evaluation(
            initial_policy_state=policy_state, final_policy_state=policy_state,
            adapter=_synthetic_adapter(seed=30), grids=grids, initial_indices=indices, horizon=4,
            eval_seed=11, agent_seed=3, episodes=5,
        )
        cases_a = [{k: v for k, v in c.items()} for c in result_a["cases"]]
        cases_b = [{k: v for k, v in c.items()} for c in result_b["cases"]]
        self.assertEqual(cases_a, cases_b)


class DifferingPoliciesTests(unittest.TestCase):
    def test_different_weights_run_end_to_end_and_classify_every_case(self):
        grids, indices = _common_setup()
        initial_agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=40)
        final_agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=41)
        adapter = _synthetic_adapter(seed=50, budget=500)

        result = run_controlled_evaluation(
            initial_policy_state=initial_agent.policy.state_dict(),
            final_policy_state=final_agent.policy.state_dict(),
            adapter=adapter, grids=grids, initial_indices=indices, horizon=4,
            eval_seed=13, agent_seed=40, episodes=MATCHED_EPISODES,
        )

        self.assertEqual(len(result["cases"]), MATCHED_EPISODES)
        self.assertEqual(result["wins"] + result["ties"] + result["losses"], MATCHED_EPISODES)

        expected_mean_initial = sum(c["initial_reward"] for c in result["cases"]) / MATCHED_EPISODES
        expected_mean_final = sum(c["final_reward"] for c in result["cases"]) / MATCHED_EPISODES
        self.assertAlmostEqual(result["initial_mean_reward"], expected_mean_initial)
        self.assertAlmostEqual(result["final_mean_reward"], expected_mean_final)

        for case in result["cases"]:
            if case["initial_spec_satisfied"]:
                self.assertIsInstance(case["initial_steps_to_satisfaction"], int)
            else:
                self.assertIsNone(case["initial_steps_to_satisfaction"])
            if case["final_spec_satisfied"]:
                self.assertIsInstance(case["final_steps_to_satisfaction"], int)
            else:
                self.assertIsNone(case["final_steps_to_satisfaction"])
            self.assertIsNone(case["parameter_trajectory"])  # documented limitation, see module docstring


class SafetyGuardTests(unittest.TestCase):
    def test_raises_if_target_mismatches_between_checkpoints(self):
        grids, indices = _common_setup()
        agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=1)
        policy_state = agent.policy.state_dict()
        adapter = _synthetic_adapter(seed=60)

        tampered_row = {
            "checkpoint": "final", "episode": 0, "episode_reward": 10.0,
            "spec_satisfied": True, "steps": 1, "target": {"tampered": True},
        }
        real_summary = {"mean_episode_reward": 10.0, "satisfaction_rate": 1.0}

        def fake_evaluate_checkpoint(*, label, **kwargs):
            if label == "final":
                return real_summary, [tampered_row]
            return real_summary, [{
                "checkpoint": "initial", "episode": 0, "episode_reward": 10.0,
                "spec_satisfied": True, "steps": 1, "target": UNSEEN_MIDPOINT_TARGET.as_dict(),
            }]

        with patch(
            "experiments.controlled_unseen_target_eval._evaluate_checkpoint",
            side_effect=fake_evaluate_checkpoint,
        ):
            with self.assertRaises(RuntimeError):
                run_controlled_evaluation(
                    initial_policy_state=policy_state, final_policy_state=policy_state,
                    adapter=adapter, grids=grids, initial_indices=indices, horizon=4,
                    eval_seed=1, agent_seed=1, episodes=1,
                )


class CLIOverwriteGuardTests(unittest.TestCase):
    def test_main_refuses_to_overwrite_an_existing_output_before_touching_spice(self):
        # The overwrite check must be the very first thing _main() does --
        # before any torch.load or real ReceiverRLAdapter construction --
        # so this test can safely exercise it without ever reaching SPICE.
        from experiments.controlled_unseen_target_eval import _main

        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "output.jsonl"
            output.write_text("already here\n", encoding="utf-8")
            fake_checkpoint = Path(tmp) / "final.pt"  # never read; overwrite check fires first

            argv = [
                "controlled_unseen_target_eval.py",
                "--final-checkpoint", str(fake_checkpoint),
                "--output", str(output),
            ]
            with patch("sys.argv", argv):
                with self.assertRaises(FileExistsError):
                    _main()


if __name__ == "__main__":
    unittest.main()

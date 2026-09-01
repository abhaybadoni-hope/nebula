"""Tests for analysis/policy_inspection.py.

All SPICE-free. Distribution-comparison tests only run forward passes
through a freshly constructed rl.ppo_agent.PPOAgent (no environment
interaction, no gradient step). Trajectory tests parse small in-memory
fixtures plus one already-completed real-SPICE log already on disk
(results/autockt_normfix_confirmation.jsonl) as a ground-truth anchor.
Counterfactual-sweep tests call no evaluator at all.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rl.autockt_state import STATE_DIM
from rl.parameter_grid import (
    ACTION_DELTAS, PARAMETER_NAMES, VERIFIED_INITIAL_PARAMETERS, build_parameter_grids,
)
from rl.ppo_agent import PPOAgent

from analysis.policy_inspection import (
    ParameterChange,
    compare_action_distributions,
    compare_checkpoints_at_state,
    group_by_episode,
    head_action_probabilities,
    load_checkpoint_summary_rows,
    load_training_step_rows,
    most_likely_action,
    parameter_changes_along_episode,
    propose_one_factor_at_a_time_sweep,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class HeadActionProbabilitiesTests(unittest.TestCase):
    def test_returns_one_valid_distribution_per_parameter(self):
        agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=0)
        state = tuple(0.0 for _ in range(STATE_DIM))
        probabilities = head_action_probabilities(agent, state)
        self.assertEqual(set(probabilities), set(PARAMETER_NAMES))
        for name in PARAMETER_NAMES:
            probs = probabilities[name]
            self.assertEqual(len(probs), 3)
            self.assertAlmostEqual(sum(probs), 1.0, places=5)
            for p in probs:
                self.assertGreaterEqual(p, 0.0)

    def test_matches_deterministic_act_choice(self):
        # Cross-checks the new module against the already-tested,
        # already-used PPOAgent.act(..., deterministic=True) argmax choice
        # on the identical state/weights.
        agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=5)
        state = tuple((i - STATE_DIM / 2) * 0.01 for i in range(STATE_DIM))
        # agent.act() returns raw Categorical indices in {0, 1, 2}; the
        # ACTION_DELTAS mapping to physical {-1, 0, +2} happens downstream
        # in rl/autockt_action.py::apply_action, not inside PPOAgent.act.
        # most_likely_action() returns the already-mapped delta, so compare
        # against ACTION_DELTAS[choice], not the raw index.
        choices, _log_prob, _value = agent.act(state, deterministic=True)
        probabilities = head_action_probabilities(agent, state)
        derived = most_likely_action(probabilities)
        for name, choice in zip(PARAMETER_NAMES, choices):
            self.assertEqual(derived[name], ACTION_DELTAS[choice])

    def test_rejects_head_count_mismatch(self):
        agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES) - 1, seed=0)
        state = tuple(0.0 for _ in range(STATE_DIM))
        with self.assertRaises(ValueError):
            head_action_probabilities(agent, state)


class CompareActionDistributionsTests(unittest.TestCase):
    def test_identical_distributions_have_zero_divergence_and_no_argmax_change(self):
        agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=1)
        state = tuple(0.1 for _ in range(STATE_DIM))
        probabilities = head_action_probabilities(agent, state)
        shifts = compare_action_distributions(probabilities, probabilities)
        for name in PARAMETER_NAMES:
            shift = shifts[name]
            self.assertFalse(shift.argmax_changed)
            self.assertAlmostEqual(shift.l1_distance, 0.0, places=6)
            self.assertAlmostEqual(shift.kl_final_given_initial, 0.0, places=4)

    def test_different_seeds_generically_produce_nonzero_divergence(self):
        agent_a = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=1)
        agent_b = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=2)
        state = tuple(0.1 for _ in range(STATE_DIM))
        probs_a = head_action_probabilities(agent_a, state)
        probs_b = head_action_probabilities(agent_b, state)
        shifts = compare_action_distributions(probs_a, probs_b)
        total_l1 = sum(shift.l1_distance for shift in shifts.values())
        self.assertGreater(total_l1, 0.0)

    def test_mismatched_parameter_sets_raise(self):
        with self.assertRaises(ValueError):
            compare_action_distributions({"rload_ohm": [1.0, 0.0, 0.0]}, {"rdeg_ohm": [1.0, 0.0, 0.0]})

    def test_compare_checkpoints_at_state_leaves_agent_holding_final_state(self):
        agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=3)
        initial_state = {k: v.clone() for k, v in agent.policy.state_dict().items()}
        other_agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=4)
        final_state = other_agent.policy.state_dict()
        state = tuple(0.0 for _ in range(STATE_DIM))

        shifts = compare_checkpoints_at_state(
            agent=agent, initial_policy_state=initial_state, final_policy_state=final_state, state=state,
        )
        self.assertEqual(set(shifts), set(PARAMETER_NAMES))
        loaded = agent.policy.state_dict()
        for key in final_state:
            self.assertTrue(__import__("torch").equal(loaded[key], final_state[key]))


class TrajectoryLoadingTests(unittest.TestCase):
    def _write(self, directory: Path, rows: list[dict]) -> Path:
        path = directory / "run.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        return path

    def test_loads_training_step_rows_and_skips_checkpoint_summaries(self):
        rows = [
            {"episode": 0, "step": 1, "parameters": {n: 1.0 for n in PARAMETER_NAMES},
             "indices": [10, 10, 10, 10, 10], "reward": -1.0, "success": False,
             "spec_satisfied": False, "failure_stage": "dc", "target": {"a": 1.0}},
            {"episode": 0, "step": 2, "parameters": {n: 2.0 for n in PARAMETER_NAMES},
             "indices": [11, 10, 10, 10, 10], "reward": 10.0, "success": True,
             "spec_satisfied": True, "failure_stage": None, "target": {"a": 1.0}},
            {"checkpoint": "initial", "episode": 0, "episode_reward": 5.0,
             "spec_satisfied": True, "steps": 1, "target": {"a": 1.0}},
        ]
        with TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), rows)
            steps = load_training_step_rows(path)
            checkpoints = load_checkpoint_summary_rows(path)
        self.assertEqual(len(steps), 2)
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(steps[0].reward, -1.0)
        self.assertIsNone(steps[0].metrics)
        self.assertEqual(steps[1].indices, (11, 10, 10, 10, 10))

    def test_carries_metrics_when_present_in_the_row(self):
        rows = [
            {"episode": 0, "step": 1, "parameters": {n: 1.0 for n in PARAMETER_NAMES},
             "indices": [10, 10, 10, 10, 10], "reward": 10.0, "success": True,
             "spec_satisfied": True, "failure_stage": None, "target": {"a": 1.0},
             "metrics": {"dfe_min_margin_v": 0.5}},
        ]
        with TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), rows)
            steps = load_training_step_rows(path)
        self.assertEqual(steps[0].metrics, {"dfe_min_margin_v": 0.5})

    def test_malformed_row_raises_with_line_context(self):
        rows = [{"episode": 0, "step": 1, "parameters": {}, "indices": [1], "reward": "not-a-number"}]
        with TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), rows)
            with self.assertRaises(ValueError):
                load_training_step_rows(path)

    def test_group_by_episode_sorts_by_step(self):
        rows = [
            {"episode": 0, "step": 2, "parameters": {n: 0.0 for n in PARAMETER_NAMES},
             "indices": [0] * 5, "reward": 1.0, "success": True, "spec_satisfied": True,
             "failure_stage": None, "target": {}},
            {"episode": 0, "step": 1, "parameters": {n: 0.0 for n in PARAMETER_NAMES},
             "indices": [0] * 5, "reward": 0.0, "success": False, "spec_satisfied": False,
             "failure_stage": "dc", "target": {}},
            {"episode": 1, "step": 1, "parameters": {n: 0.0 for n in PARAMETER_NAMES},
             "indices": [0] * 5, "reward": 0.0, "success": False, "spec_satisfied": False,
             "failure_stage": "dc", "target": {}},
        ]
        with TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), rows)
            steps = load_training_step_rows(path)
        episodes = group_by_episode(steps)
        self.assertEqual(set(episodes), {0, 1})
        self.assertEqual([s.step for s in episodes[0]], [1, 2])


class RealLogGroundTruthTests(unittest.TestCase):
    """Anchors the loader against the actual completed real-SPICE run from
    the prior (single-target, --target-mode hard) milestone, whose row
    counts were independently established during inspection: 100 per-step
    training rows + 6 checkpoint-summary rows (3 initial + 3 final).
    """

    def test_autockt_normfix_confirmation_row_counts(self):
        path = REPOSITORY_ROOT / "results" / "autockt_normfix_confirmation.jsonl"
        if not path.is_file():
            self.skipTest("fixture result file not present in this checkout")
        steps = load_training_step_rows(path)
        checkpoints = load_checkpoint_summary_rows(path)
        self.assertEqual(len(steps), 100)
        self.assertEqual(len(checkpoints), 6)
        self.assertEqual({row["checkpoint"] for row in checkpoints}, {"initial", "final"})
        # every training row currently on disk predates metrics-logging:
        self.assertTrue(all(step.metrics is None for step in steps))

    def test_parameter_changes_reconstruct_from_the_real_log(self):
        path = REPOSITORY_ROOT / "results" / "autockt_normfix_confirmation.jsonl"
        if not path.is_file():
            self.skipTest("fixture result file not present in this checkout")
        steps = load_training_step_rows(path)
        episodes = group_by_episode(steps)
        any_multi_step_episode = [ep for ep in episodes.values() if len(ep) > 1]
        self.assertTrue(any_multi_step_episode, "expected at least one multi-step episode in the real log")
        changes = parameter_changes_along_episode(any_multi_step_episode[0])
        self.assertEqual(len(changes), 5 * (len(any_multi_step_episode[0]) - 1))
        for change in changes:
            self.assertIsInstance(change, ParameterChange)
            self.assertIn(change.parameter, PARAMETER_NAMES)
            self.assertEqual(change.to_index - change.from_index, change.index_delta)


class ParameterChangesTests(unittest.TestCase):
    def test_index_delta_and_reward_association_are_correct(self):
        rows = [
            {"episode": 0, "step": 1, "parameters": dict(VERIFIED_INITIAL_PARAMETERS),
             "indices": [10, 10, 10, 10, 10], "reward": -1.0, "success": False,
             "spec_satisfied": False, "failure_stage": "dc", "target": {}},
            {"episode": 0, "step": 2,
             "parameters": {**VERIFIED_INITIAL_PARAMETERS, "rload_ohm": 5000.0},
             "indices": [12, 10, 10, 10, 10], "reward": 10.0, "success": True,
             "spec_satisfied": True, "failure_stage": None, "target": {}},
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            with path.open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
            steps = load_training_step_rows(path)
        changes = parameter_changes_along_episode(steps)
        rload_change = next(c for c in changes if c.parameter == "rload_ohm")
        self.assertEqual(rload_change.index_delta, 2)
        self.assertEqual(rload_change.to_value, 5000.0)
        self.assertEqual(rload_change.reward_after, 10.0)
        self.assertTrue(rload_change.spec_satisfied_after)

        unmoved = next(c for c in changes if c.parameter == "dfe_tap_v")
        self.assertEqual(unmoved.index_delta, 0)
        self.assertEqual(unmoved.from_value, unmoved.to_value)


class CounterfactualSweepTests(unittest.TestCase):
    def test_proposes_candidates_only_along_the_requested_parameters(self):
        grids = build_parameter_grids()
        candidates = propose_one_factor_at_a_time_sweep(VERIFIED_INITIAL_PARAMETERS, grids)
        self.assertTrue(candidates)
        self.assertEqual({c.parameter for c in candidates}, {"rload_ohm", "rdeg_ohm", "cdeg_f"})

    def test_only_the_swept_parameter_differs_from_baseline(self):
        grids = build_parameter_grids()
        candidates = propose_one_factor_at_a_time_sweep(VERIFIED_INITIAL_PARAMETERS, grids)
        for candidate in candidates:
            for name in PARAMETER_NAMES:
                if name == candidate.parameter:
                    continue
                self.assertEqual(
                    candidate.parameters[name], VERIFIED_INITIAL_PARAMETERS[name],
                    f"{name} should stay at baseline while sweeping {candidate.parameter}",
                )

    def test_no_duplicate_candidate_indices_per_parameter(self):
        grids = build_parameter_grids()
        candidates = propose_one_factor_at_a_time_sweep(VERIFIED_INITIAL_PARAMETERS, grids)
        by_parameter: dict[str, set[int]] = {}
        for candidate in candidates:
            seen = by_parameter.setdefault(candidate.parameter, set())
            self.assertNotIn(candidate.candidate_index, seen)
            seen.add(candidate.candidate_index)

    def test_clipping_is_flagged_at_grid_boundary(self):
        grids = build_parameter_grids(points_per_parameter=5)  # small grid to force clipping
        baseline = dict(VERIFIED_INITIAL_PARAMETERS)
        baseline["rload_ohm"] = grids["rload_ohm"].value_at(0)  # pin to the low boundary
        candidates = propose_one_factor_at_a_time_sweep(
            baseline, grids, parameter_names=("rload_ohm",), index_offsets=(-4, -2, -1, 1, 2, 4),
        )
        negative_offset_candidates = [c for c in candidates if c.index_offset < 0]
        self.assertTrue(all(c.clipped for c in negative_offset_candidates))

    def test_rejects_unknown_parameter_names(self):
        grids = build_parameter_grids()
        with self.assertRaises(ValueError):
            propose_one_factor_at_a_time_sweep(
                VERIFIED_INITIAL_PARAMETERS, grids, parameter_names=("not_a_real_parameter",),
            )

    def test_rejects_incomplete_baseline(self):
        grids = build_parameter_grids()
        incomplete = {k: v for k, v in VERIFIED_INITIAL_PARAMETERS.items() if k != "cdeg_f"}
        with self.assertRaises(ValueError):
            propose_one_factor_at_a_time_sweep(incomplete, grids)

    def test_calls_no_evaluator_module_is_import_clean(self):
        # Structural guard: this module must never IMPORT the real
        # evaluator or ngspice-facing code, so accidentally wiring it to
        # run SPICE would be an explicit, reviewable code change, not a
        # silent addition. Mentions of "evaluate_receiver" in prose
        # docstrings are fine and expected; only actual import statements
        # are checked here.
        import ast
        import analysis.policy_inspection as module
        source = Path(module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names.add(node.module)
                imported_names.update(f"{node.module}.{alias.name}" for alias in node.names)
        forbidden_substrings = ("simulator.receiver", "simulator.ngspice", "evaluate_receiver")
        for name in imported_names:
            for forbidden in forbidden_substrings:
                self.assertNotIn(
                    forbidden, name,
                    f"analysis/policy_inspection.py must not import {forbidden} (found: {name})",
                )


if __name__ == "__main__":
    unittest.main()

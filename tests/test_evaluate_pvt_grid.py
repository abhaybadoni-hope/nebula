"""SPICE-free tests for simulator.receiver.evaluate_pvt_grid -- fully
implemented and exported since early in this project (PVT_GRID, 60
conditions: TT/SS/FF/SF/FS x 3 VDD x 4 temperatures) but never exercised by
any test, experiment, or script until this audit (docs/autockt-mapping.md
sec 21, finding F). Written before spending any real SPICE on a PVT
experiment, to catch a control-flow bug here rather than mid-run.

evaluate_pvt_grid calls simulator.receiver.evaluate_receiver directly (no
injectable evaluator parameter, unlike experiments/receiver_search.py's
run_receiver_search) -- these tests mock it at the module level, the same
technique tests/test_ngspice.py already uses for subprocess.run. No
simulator/receiver.py changes; this only imports and calls its existing,
public evaluate_pvt_grid function.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from simulator.config import ProcessCorner, SimulationConditions
from simulator.receiver import (
    EvaluationFidelity,
    ReceiverEvaluation,
    ReceiverParameters,
    evaluate_pvt_grid,
)

DESIGN_A = ReceiverParameters(
    rload_ohm=2342.472156058411, rdeg_ohm=822.3558626926603,
    cdeg_f=9.999375862792168e-13, itail_a=0.0006028331705063624,
    dfe_tap_v=-0.011090823885148815,
)


def _fake_evaluation(success: bool, identity: str) -> ReceiverEvaluation:
    return ReceiverEvaluation(
        success, DESIGN_A, SimulationConditions(), EvaluationFidelity.FINAL, (),
        {}, None if success else "transient", 0.0, identity, {},
    )


class EvaluatePvtGridControlFlowTests(unittest.TestCase):
    def test_evaluates_every_condition_in_order(self):
        conditions = tuple(
            SimulationConditions(ProcessCorner.TT, temperature, 1.8)
            for temperature in (0.0, 27.0, 125.0)
        )
        seen_conditions = []

        def fake_evaluate_receiver(parameters, conditions_arg, fidelity, **kwargs):
            seen_conditions.append(conditions_arg)
            return _fake_evaluation(True, f"eval-{len(seen_conditions)}")

        with patch("simulator.receiver.evaluate_receiver", side_effect=fake_evaluate_receiver):
            results = evaluate_pvt_grid(DESIGN_A, conditions=conditions)

        self.assertEqual(len(results), 3)
        self.assertEqual(list(seen_conditions), list(conditions))
        self.assertTrue(all(r.success for r in results))

    def test_stop_on_failure_halts_immediately_and_does_not_evaluate_the_rest(self):
        conditions = tuple(
            SimulationConditions(ProcessCorner.TT, temperature, 1.8)
            for temperature in (0.0, 27.0, 75.0, 125.0)
        )
        call_count = 0

        def fake_evaluate_receiver(parameters, conditions_arg, fidelity, **kwargs):
            nonlocal call_count
            call_count += 1
            # fail on the SECOND condition specifically
            return _fake_evaluation(call_count != 2, f"eval-{call_count}")

        with patch("simulator.receiver.evaluate_receiver", side_effect=fake_evaluate_receiver):
            results = evaluate_pvt_grid(DESIGN_A, conditions=conditions, stop_on_failure=True)

        self.assertEqual(call_count, 2)  # never reached conditions 3/4
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].success)
        self.assertFalse(results[1].success)

    def test_stop_on_failure_false_runs_the_full_grid_even_after_a_failure(self):
        conditions = tuple(
            SimulationConditions(ProcessCorner.TT, temperature, 1.8)
            for temperature in (0.0, 27.0, 75.0)
        )
        call_count = 0

        def fake_evaluate_receiver(parameters, conditions_arg, fidelity, **kwargs):
            nonlocal call_count
            call_count += 1
            return _fake_evaluation(call_count != 1, f"eval-{call_count}")

        with patch("simulator.receiver.evaluate_receiver", side_effect=fake_evaluate_receiver):
            results = evaluate_pvt_grid(DESIGN_A, conditions=conditions, stop_on_failure=False)

        self.assertEqual(call_count, 3)
        self.assertEqual(len(results), 3)

    def test_default_fidelity_is_final(self):
        seen_fidelity = []

        def fake_evaluate_receiver(parameters, conditions_arg, fidelity, **kwargs):
            seen_fidelity.append(fidelity)
            return _fake_evaluation(True, "eval")

        with patch("simulator.receiver.evaluate_receiver", side_effect=fake_evaluate_receiver):
            evaluate_pvt_grid(DESIGN_A, conditions=(SimulationConditions(),))

        self.assertEqual(seen_fidelity, [EvaluationFidelity.FINAL])

    def test_empty_conditions_returns_empty_without_calling_the_evaluator(self):
        with patch("simulator.receiver.evaluate_receiver") as mock_eval:
            results = evaluate_pvt_grid(DESIGN_A, conditions=())
        mock_eval.assert_not_called()
        self.assertEqual(results, ())

    def test_default_conditions_argument_is_the_full_60_point_grid(self):
        import inspect
        from simulator.config import PVT_GRID
        signature = inspect.signature(evaluate_pvt_grid)
        self.assertEqual(signature.parameters["conditions"].default, PVT_GRID)


if __name__ == "__main__":
    unittest.main()

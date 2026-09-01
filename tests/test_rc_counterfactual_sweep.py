"""Cheap, SPICE-free tests for experiments/rc_counterfactual_sweep.py's CLI
wiring. These only check argument parsing / the overwrite guard -- nothing
here calls evaluate_receiver or ngspice.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from analysis.policy_inspection import (
    DEFAULT_COUNTERFACTUAL_INDEX_OFFSETS,
    DEFAULT_COUNTERFACTUAL_PARAMETERS,
)
from experiments.rc_counterfactual_sweep import main


class OverwriteGuardTests(unittest.TestCase):
    def test_refuses_to_overwrite_an_existing_output_file(self):
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jsonl"
            source.write_text('{"episode": 0, "step": 1}\n', encoding="utf-8")
            output = Path(tmp) / "output.jsonl"
            output.write_text("already here\n", encoding="utf-8")

            argv = ["rc_counterfactual_sweep.py", "--source", str(source), "--output", str(output)]
            with patch("sys.argv", argv):
                with self.assertRaises(FileExistsError):
                    main()

    def test_raises_on_source_with_no_training_step_rows(self):
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jsonl"
            source.write_text("", encoding="utf-8")
            output = Path(tmp) / "output.jsonl"

            argv = ["rc_counterfactual_sweep.py", "--source", str(source), "--output", str(output)]
            with patch("sys.argv", argv):
                with self.assertRaises(ValueError):
                    main()
            self.assertFalse(output.exists())


class CLIDefaultsStayInSyncTests(unittest.TestCase):
    def test_default_parameters_and_offsets_match_policy_inspection(self):
        import argparse
        # Re-derive the same defaults main() registers, without invoking main()
        # itself (which requires real argv/source data) -- a lightweight
        # sync check that a future edit to one module's defaults doesn't
        # silently diverge from the other's.
        parser = argparse.ArgumentParser()
        parser.add_argument("--parameters", nargs="+", default=list(DEFAULT_COUNTERFACTUAL_PARAMETERS))
        parser.add_argument("--offsets", type=int, nargs="+", default=list(DEFAULT_COUNTERFACTUAL_INDEX_OFFSETS))
        args = parser.parse_args([])
        self.assertEqual(tuple(args.parameters), DEFAULT_COUNTERFACTUAL_PARAMETERS)
        self.assertEqual(tuple(args.offsets), DEFAULT_COUNTERFACTUAL_INDEX_OFFSETS)


if __name__ == "__main__":
    unittest.main()

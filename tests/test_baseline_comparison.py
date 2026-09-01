"""Tests for analysis/baseline_comparison.py.

All cheap and SPICE-free: these tests only parse JSON already written to
disk (or small in-memory fixtures) and do arithmetic. Nothing here imports
simulator.receiver's evaluator or calls ngspice.
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from analysis.baseline_comparison import (
    CEM_SCHEMA,
    CTLE_LEGACY_SCHEMA,
    RAW_EVALUATION_SCHEMA,
    RECEIVER_SEARCH_SCHEMA,
    build_report,
    detect_schema,
    load_run,
    summarize,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class SchemaDetectionTests(unittest.TestCase):
    def test_detects_receiver_search_schema(self):
        row = {
            "candidate_index": 0, "reward": 100.0, "success": True,
            "parameters": {"rload_ohm": 1000.0, "rdeg_ohm": 1.0, "cdeg_f": 1e-13,
                            "itail_a": 1e-4, "dfe_tap_v": 0.0},
        }
        self.assertEqual(detect_schema(row), RECEIVER_SEARCH_SCHEMA)

    def test_detects_cem_schema(self):
        row = {"iteration": 1, "evaluation": 1, "reward": 100.0, "action": [0.0] * 5, "success": True}
        self.assertEqual(detect_schema(row), CEM_SCHEMA)

    def test_detects_ctle_legacy_schema(self):
        row = {"RLOAD": 1000.0, "CDEG": 1e-13, "gain_1mhz_db": 1.0}
        self.assertEqual(detect_schema(row), CTLE_LEGACY_SCHEMA)

    def test_detects_raw_evaluation_dump_schema(self):
        row = {"success": True, "parameters": {"rload_ohm": 1.0}, "stages": []}
        self.assertEqual(detect_schema(row), RAW_EVALUATION_SCHEMA)


class LoadRunTests(unittest.TestCase):
    def _write(self, directory: Path, rows: list[dict]) -> Path:
        path = directory / "run.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        return path

    def test_receiver_search_rows_load_with_logged_parameters(self):
        rows = [
            {"candidate_index": 0, "reward": -100.0, "success": False, "failed_stage": "dc",
             "parameters": {"rload_ohm": 100.0, "rdeg_ohm": 10.0, "cdeg_f": 1e-14,
                             "itail_a": 1e-5, "dfe_tap_v": 0.1},
             "evaluation_id": "abc", "wall_clock_s": 15.0},
            {"candidate_index": 1, "reward": 100.0, "success": True, "failed_stage": None,
             "parameters": {"rload_ohm": 2000.0, "rdeg_ohm": 800.0, "cdeg_f": 1e-12,
                             "itail_a": 6e-4, "dfe_tap_v": -0.01},
             "evaluation_id": "def", "wall_clock_s": 63.0},
        ]
        with TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), rows)
            loaded = load_run(path)
        self.assertEqual(loaded.schema, RECEIVER_SEARCH_SCHEMA)
        self.assertEqual(len(loaded.records), 2)
        self.assertEqual(loaded.skipped_rows, 0)
        self.assertFalse(loaded.records[0].parameters_reconstructed)
        self.assertEqual(loaded.records[1].parameters["rload_ohm"], 2000.0)

    def test_cem_rows_reconstruct_parameters_from_action_when_missing(self):
        rows = [
            {"iteration": 1, "evaluation": 1, "reward": 100.0,
             "action": [0.34806, 0.25077, 0.31662, 0.88549, -0.04687],
             "success": True, "failure_stage": None},
        ]
        with TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), rows)
            loaded = load_run(path)
        self.assertEqual(loaded.schema, CEM_SCHEMA)
        self.assertEqual(len(loaded.records), 1)
        record = loaded.records[0]
        self.assertTrue(record.parameters_reconstructed)
        self.assertIsNotNone(record.parameters)
        self.assertIn("rload_ohm", record.parameters)
        self.assertIsNone(record.wall_clock_s)

    def test_cem_rows_use_logged_parameters_and_wall_clock_when_present(self):
        rows = [
            {"iteration": 1, "evaluation": 1, "reward": 100.0, "action": [0.0] * 5,
             "success": True, "failure_stage": None,
             "parameters": {"rload_ohm": 1234.0}, "wall_clock_s": 12.5,
             "evaluation_id": "xyz"},
        ]
        with TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), rows)
            loaded = load_run(path)
        record = loaded.records[0]
        self.assertFalse(record.parameters_reconstructed)
        self.assertEqual(record.parameters, {"rload_ohm": 1234.0})
        self.assertEqual(record.wall_clock_s, 12.5)
        self.assertEqual(record.evaluation_id, "xyz")

    def test_ctle_legacy_rows_produce_no_records(self):
        rows = [{"RLOAD": 1000.0, "CDEG": 1e-13, "RDEG": 100.0, "ITAIL_VAL": 1e-4,
                  "constraints_passed": True}]
        with TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), rows)
            loaded = load_run(path)
        self.assertEqual(loaded.schema, CTLE_LEGACY_SCHEMA)
        self.assertEqual(loaded.records, ())

    def test_empty_file_loads_with_no_records(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.jsonl"
            path.write_text("", encoding="utf-8")
            loaded = load_run(path)
        self.assertEqual(loaded.records, ())


class SummarizeTests(unittest.TestCase):
    def _loaded(self, rows):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            with path.open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
            return load_run(path)

    def test_running_best_is_monotonically_nondecreasing(self):
        rows = [
            {"candidate_index": i, "reward": reward, "success": reward > 0,
             "failed_stage": None if reward > 0 else "dc",
             "parameters": {"rload_ohm": 1.0, "rdeg_ohm": 1.0, "cdeg_f": 1.0,
                             "itail_a": 1.0, "dfe_tap_v": 0.0}}
            for i, reward in enumerate([-100.0, 50.0, -100.0, 100.0, -100.0])
        ]
        loaded = self._loaded(rows)
        summary = summarize(loaded, "random_search")
        running_best = summary.running_best_reward
        for earlier, later in zip(running_best, running_best[1:]):
            self.assertLessEqual(earlier, later)
        self.assertEqual(running_best[-1], 100.0)

    def test_evaluations_to_first_success_is_one_indexed_and_correct(self):
        rows = [
            {"candidate_index": i, "reward": reward, "success": reward > 0,
             "failed_stage": None if reward > 0 else "dc",
             "parameters": {"rload_ohm": 1.0, "rdeg_ohm": 1.0, "cdeg_f": 1.0,
                             "itail_a": 1.0, "dfe_tap_v": 0.0}}
            for i, reward in enumerate([-100.0, -100.0, 100.0, -100.0])
        ]
        loaded = self._loaded(rows)
        summary = summarize(loaded, "random_search")
        self.assertEqual(summary.evaluations_to_first_success, 3)

    def test_no_successes_gives_none_for_evaluations_to_first_success(self):
        rows = [
            {"candidate_index": i, "reward": -100.0, "success": False, "failed_stage": "dc",
             "parameters": {"rload_ohm": 1.0, "rdeg_ohm": 1.0, "cdeg_f": 1.0,
                             "itail_a": 1.0, "dfe_tap_v": 0.0}}
            for i in range(3)
        ]
        loaded = self._loaded(rows)
        summary = summarize(loaded, "random_search")
        self.assertIsNone(summary.evaluations_to_first_success)
        self.assertEqual(summary.success_rate, 0.0)

    def test_best_reward_matches_max_reward_row(self):
        rows = [
            {"candidate_index": i, "reward": reward, "success": reward > 0,
             "failed_stage": None if reward > 0 else "dc",
             "parameters": {"rload_ohm": float(i), "rdeg_ohm": 1.0, "cdeg_f": 1.0,
                             "itail_a": 1.0, "dfe_tap_v": 0.0}}
            for i, reward in enumerate([-100.0, 30.0, 100.0, 60.0])
        ]
        loaded = self._loaded(rows)
        summary = summarize(loaded, "random_search")
        self.assertEqual(summary.best_reward, 100.0)
        self.assertEqual(summary.best_index, 2)
        self.assertEqual(summary.best_parameters["rload_ohm"], 2.0)

    def test_wall_clock_available_only_when_every_row_has_it(self):
        rows_partial = [
            {"candidate_index": 0, "reward": 100.0, "success": True, "failed_stage": None,
             "parameters": {"rload_ohm": 1.0, "rdeg_ohm": 1.0, "cdeg_f": 1.0,
                             "itail_a": 1.0, "dfe_tap_v": 0.0}, "wall_clock_s": 10.0},
            {"candidate_index": 1, "reward": 100.0, "success": True, "failed_stage": None,
             "parameters": {"rload_ohm": 1.0, "rdeg_ohm": 1.0, "cdeg_f": 1.0,
                             "itail_a": 1.0, "dfe_tap_v": 0.0}},
        ]
        loaded = self._loaded(rows_partial)
        summary = summarize(loaded, "random_search")
        self.assertFalse(summary.wall_clock_available)
        self.assertIsNone(summary.wall_clock_mean_s)
        self.assertTrue(any("wall_clock_s present for only" in c for c in summary.caveats))

        rows_full = [dict(row, wall_clock_s=10.0) for row in rows_partial]
        loaded_full = self._loaded(rows_full)
        summary_full = summarize(loaded_full, "random_search")
        self.assertTrue(summary_full.wall_clock_available)
        self.assertEqual(summary_full.wall_clock_mean_s, 10.0)

    def test_empty_run_summary_has_none_metrics_and_a_caveat(self):
        loaded = self._loaded([])
        summary = summarize(loaded, "random_search")
        self.assertEqual(summary.n_evaluations, 0)
        self.assertIsNone(summary.success_rate)
        self.assertIn("no usable records in this file", summary.caveats)


class RealFileGroundTruthTests(unittest.TestCase):
    """Anchors the loader/summarizer against the actual on-disk baseline
    results and their own recorded summary.json, so a regression in the
    analysis code (not the underlying data) would be caught here.
    """

    def test_random_search_seed123_matches_its_own_summary_json(self):
        path = REPOSITORY_ROOT / "results" / "receiver_random_search_20_seed123.jsonl"
        if not path.is_file():
            self.skipTest("fixture result file not present in this checkout")
        summary_path = path.with_suffix(path.suffix + ".summary.json")
        recorded = json.loads(summary_path.read_text(encoding="utf-8"))

        loaded = load_run(path)
        summary = summarize(loaded, "random_search")

        self.assertEqual(summary.n_evaluations, recorded["completed_candidates"])
        self.assertEqual(summary.n_successes, recorded["successful_candidates"])
        self.assertEqual(summary.best_reward, recorded["reward_max"])

    def test_cem_baseline_3x10_is_detected_as_cem_schema_and_is_incomplete(self):
        path = REPOSITORY_ROOT / "results" / "cem_baseline_3x10.jsonl"
        if not path.is_file():
            self.skipTest("fixture result file not present in this checkout")
        loaded = load_run(path)
        self.assertEqual(loaded.schema, CEM_SCHEMA)
        # Configured for 3 iterations x population 10 = 30; known to be an
        # interrupted run with fewer rows on disk.
        self.assertLess(len(loaded.records), 30)

    def test_ctle_only_legacy_files_are_excluded_not_miscounted_as_receiver_runs(self):
        for name in ("random_search.jsonl", "ctle_search_20.jsonl"):
            path = REPOSITORY_ROOT / "results" / name
            if not path.is_file():
                continue
            loaded = load_run(path)
            self.assertEqual(loaded.schema, CTLE_LEGACY_SCHEMA)
            self.assertEqual(loaded.records, ())


class BuildReportTests(unittest.TestCase):
    def test_build_report_runs_and_has_headline_comparison(self):
        report = build_report(REPOSITORY_ROOT)
        self.assertIn("random_search_runs", report)
        self.assertIn("cem_runs", report)
        self.assertTrue(len(report["random_search_runs"]) >= 1)
        self.assertTrue(len(report["cem_runs"]) >= 1)
        if report["headline_comparison"] is not None:
            headline = report["headline_comparison"]
            self.assertIn("spice_evaluation_count", headline)
            self.assertIn("sample_size_caveat", headline)

    def test_build_report_does_not_fabricate_wall_clock_for_historical_runs(self):
        report = build_report(REPOSITORY_ROOT)
        for summary in report["random_search_runs"] + report["cem_runs"]:
            if not summary["wall_clock_available"]:
                self.assertIsNone(summary["wall_clock_total_s"])
                self.assertIsNone(summary["wall_clock_mean_s"])


if __name__ == "__main__":
    unittest.main()

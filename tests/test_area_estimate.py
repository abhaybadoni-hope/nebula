"""Tests for analysis/area_estimate.py -- Task 2 (official-brief audit,
docs/autockt-mapping.md sec 21 finding D). SPICE-free: only parses text.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from simulator.receiver import BLOCK

from analysis.area_estimate import (
    AreaEstimate,
    estimate_ctle_area,
    format_area_report,
    parse_transistor_devices,
)


class ParseTransistorDevicesTests(unittest.TestCase):
    def test_parses_both_devices_from_a_minimal_fixture(self):
        source = (
            ".subckt CTLE inp inn outp outn vdd vss\n"
            "+ params: RLOAD=1k RDEG=1k CDEG=0.5p ITAIL_VAL=100u\n"
            "XMP outp inp srcp vss sky130_fd_pr__nfet_01v8 W=10 L=0.15\n"
            "XMN outn inn srcn vss sky130_fd_pr__nfet_01v8 W=10 L=0.15\n"
            ".ends CTLE\n"
        )
        devices = parse_transistor_devices(source)
        self.assertEqual(len(devices), 2)
        self.assertEqual({d.name for d in devices}, {"MP", "MN"})
        for device in devices:
            self.assertEqual(device.width_um, 10.0)
            self.assertEqual(device.length_um, 0.15)
            self.assertEqual(device.channel_area_um2, 1.5)

    def test_raises_when_no_devices_present(self):
        with self.assertRaises(ValueError):
            parse_transistor_devices("* empty netlist\n.end\n")

    def test_handles_differing_widths_and_lengths(self):
        source = (
            "X1 a b c d sky130_fd_pr__nfet_01v8 W=5 L=0.3\n"
            "X2 e f g h sky130_fd_pr__nfet_01v8 W=20 L=0.15\n"
        )
        devices = parse_transistor_devices(source)
        areas = sorted(d.channel_area_um2 for d in devices)
        self.assertEqual(areas, [1.5, 3.0])


class EstimateCtleAreaTests(unittest.TestCase):
    def test_reads_the_real_unmodified_block_file_by_default(self):
        estimate = estimate_ctle_area()
        self.assertEqual(estimate.source_file, str(BLOCK))
        self.assertEqual(len(estimate.devices), 2)

    def test_matches_the_known_current_dimensions(self):
        # Regression guard: if circuits/blocks/ctle.spice's W/L ever
        # change, this test documents the change rather than silently
        # passing with a stale assumption.
        estimate = estimate_ctle_area()
        for device in estimate.devices:
            self.assertEqual(device.width_um, 10.0)
            self.assertEqual(device.length_um, 0.15)
        self.assertEqual(estimate.transistor_channel_area_um2, 3.0)
        self.assertAlmostEqual(estimate.transistor_channel_area_mm2, 3e-6)

    def test_total_area_is_never_claimed_computable(self):
        estimate = estimate_ctle_area()
        self.assertFalse(estimate.total_area_computable)
        self.assertGreaterEqual(len(estimate.missing_components), 1)

    def test_works_against_a_custom_fixture_file(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom.spice"
            path.write_text(
                "X1 a b c d sky130_fd_pr__nfet_01v8 W=2 L=0.15\n"
                "X2 e f g h sky130_fd_pr__nfet_01v8 W=2 L=0.15\n"
            )
            estimate = estimate_ctle_area(path)
        self.assertEqual(estimate.transistor_channel_area_um2, 0.6)


class ReportFormattingTests(unittest.TestCase):
    def test_report_never_states_a_pass_fail_verdict_against_the_budget(self):
        report = format_area_report(estimate_ctle_area())
        self.assertIn("NOT ASSESSABLE", report)
        self.assertNotIn("PASS", report.split("Verdict")[0])
        # the 0.05 mm^2 budget is mentioned, but never asserted as met:
        self.assertNotIn("0.05 mm^2: PASS", report)

    def test_report_lists_every_missing_component(self):
        estimate = estimate_ctle_area()
        report = format_area_report(estimate)
        for item in estimate.missing_components:
            self.assertIn(item, report)


if __name__ == "__main__":
    unittest.main()

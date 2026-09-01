"""Tests for experiments/export_final_schematic.py -- the final-schematic
exporter that closes the "parameter list, not a real deliverable" gap
identified in docs/autockt-mapping.md sec 21. SPICE-free: never calls
evaluate_receiver or ngspice; only renders text and does file I/O.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from simulator.config import ProcessCorner
from simulator.ngspice import parameterize_netlist
from simulator.receiver import BLOCK, ReceiverParameters

from experiments.export_final_schematic import _main, render_final_schematic

DESIGN_A = ReceiverParameters(
    rload_ohm=2342.472156058411, rdeg_ohm=822.3558626926603,
    cdeg_f=9.999375862792168e-13, itail_a=0.0006028331705063624,
    dfe_tap_v=-0.011090823885148815,
)


class RenderCorrectnessTests(unittest.TestCase):
    def test_ctle_spice_own_params_syntax_does_not_match_parameterize_netlist(self):
        # Regression guard for the exact bug caught and fixed while
        # building this module: circuits/blocks/ctle.spice declares
        # defaults via `.subckt ... params:`, not standalone `.param`
        # lines -- parameterize_netlist's regex only matches the latter.
        # If ctle.spice is ever rewritten to use standalone .param lines,
        # this test (not the exporter) should be revisited.
        with self.assertRaises(KeyError):
            parameterize_netlist(BLOCK.read_text(encoding="utf-8"), {"RLOAD": 1234.0})

    def test_renders_the_exact_final_component_values(self):
        # simulator.ngspice._spice_value formats floats as ".15g" (15
        # significant figures), not Python's repr() -- match that
        # pre-existing, unmodified convention rather than assuming repr.
        schematic = render_final_schematic(DESIGN_A)
        self.assertIn(f"RLOAD={format(DESIGN_A.rload_ohm, '.15g')}", schematic)
        self.assertIn(f"RDEG={format(DESIGN_A.rdeg_ohm, '.15g')}", schematic)
        self.assertIn(f"CDEG={format(DESIGN_A.cdeg_f, '.15g')}", schematic)
        self.assertIn(f"ITAIL_VAL={format(DESIGN_A.itail_a, '.15g')}", schematic)

    def test_includes_the_real_unmodified_block_file_by_absolute_path(self):
        schematic = render_final_schematic(DESIGN_A)
        self.assertIn(f'.include "{BLOCK.as_posix()}"', schematic)
        self.assertTrue(BLOCK.is_file(), "the included block file must actually exist")

    def test_dfe_tap_is_documented_as_behavioral_not_a_netlist_element(self):
        schematic = render_final_schematic(DESIGN_A)
        self.assertIn("BEHAVIORAL, not a netlist element", schematic)
        self.assertIn(f"{DESIGN_A.dfe_tap_v:.6g}", schematic)
        # and it must NOT appear as an actual .param substitution target:
        self.assertNotIn("DFE_TAP_V=", schematic)

    def test_achieved_metrics_are_documented_when_supplied(self):
        schematic = render_final_schematic(
            DESIGN_A, achieved_metrics={"dfe_min_margin_v": 0.5174474651666459},
        )
        self.assertIn("dfe_min_margin_v = 0.5174474651666459", schematic)

    def test_omitting_achieved_metrics_does_not_fabricate_any(self):
        schematic = render_final_schematic(DESIGN_A, achieved_metrics=None)
        self.assertNotIn("Achieved specifications", schematic)

    def test_transistor_channel_area_is_documented_and_never_claimed_total(self):
        schematic = render_final_schematic(DESIGN_A)
        self.assertIn("Transistor channel area:", schematic)
        self.assertIn("3 um^2", schematic)
        self.assertIn("PARTIAL measurement only", schematic)
        self.assertIn("Do NOT", schematic)
        self.assertIn("0.05 mm^2", schematic)

    def test_process_corner_is_documented(self):
        schematic = render_final_schematic(DESIGN_A, process_corner=ProcessCorner.SS)
        self.assertIn("process corner this file documents: ss", schematic)

    def test_ends_with_a_terminated_netlist(self):
        schematic = render_final_schematic(DESIGN_A)
        self.assertTrue(schematic.rstrip().endswith(".end"))

    def test_param_line_has_concrete_values_and_instantiation_references_them(self):
        # The `.param` line itself must carry the real, concrete values
        # (that's what parameterize_netlist substitutes). The XCTLE
        # instantiation line's `{RLOAD}`-style braces are correct, intended
        # SPICE syntax -- a brace expression ngspice resolves against the
        # `.param` line at parse time -- not an unresolved template marker;
        # they are expected to remain literally as `{RLOAD}` in the output.
        schematic = render_final_schematic(DESIGN_A)
        param_line = next(line for line in schematic.splitlines() if line.startswith(".param"))
        self.assertIn(format(DESIGN_A.rload_ohm, ".15g"), param_line)
        self.assertIn("RLOAD={RLOAD}", schematic)


class CLITests(unittest.TestCase):
    def test_refuses_to_overwrite_existing_output(self):
        with TemporaryDirectory() as tmp:
            params_path = Path(tmp) / "params.json"
            params_path.write_text(json.dumps({
                "rload_ohm": 1000.0, "rdeg_ohm": 1000.0, "cdeg_f": 5e-13,
                "itail_a": 1e-4, "dfe_tap_v": 0.0,
            }))
            output = Path(tmp) / "out.spice"
            output.write_text("already here")

            argv = [
                "export_final_schematic.py",
                "--parameters-json", str(params_path),
                "--output", str(output),
            ]
            with patch("sys.argv", argv):
                with self.assertRaises(FileExistsError):
                    _main()

    def test_writes_a_valid_schematic_file_end_to_end(self):
        with TemporaryDirectory() as tmp:
            params_path = Path(tmp) / "params.json"
            params_path.write_text(json.dumps({
                "rload_ohm": DESIGN_A.rload_ohm, "rdeg_ohm": DESIGN_A.rdeg_ohm,
                "cdeg_f": DESIGN_A.cdeg_f, "itail_a": DESIGN_A.itail_a,
                "dfe_tap_v": DESIGN_A.dfe_tap_v,
            }))
            output = Path(tmp) / "out.spice"

            argv = [
                "export_final_schematic.py",
                "--parameters-json", str(params_path),
                "--output", str(output),
            ]
            with patch("sys.argv", argv):
                _main()

            self.assertTrue(output.is_file())
            content = output.read_text(encoding="utf-8")
            self.assertIn(f"RLOAD={format(DESIGN_A.rload_ohm, '.15g')}", content)


if __name__ == "__main__":
    unittest.main()

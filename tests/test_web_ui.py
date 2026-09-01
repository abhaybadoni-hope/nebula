"""SPICE-free tests for experiments/web_ui.py -- the competition-demo
local UI around experiments/run_autockt_pipeline.py.

Covers request validation, CLI argv construction (must match
run_autockt_pipeline.py's own flags exactly), and the full HTTP flow using
--backend synthetic (no ngspice, no PDK). The crash-handling path
(subprocess terminated by a signal) is tested by mocking subprocess.Popen
rather than by relying on the real, intermittent SIGSEGV documented in
docs/autockt-mapping.md sec 23 -- that failure mode is real but not
reliably reproducible on demand, so it is simulated here deterministically.
"""

from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.client import HTTPResponse
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from experiments import web_ui


class ValidateRequestTests(unittest.TestCase):
    def _base(self, **overrides):
        payload = {
            "target_mode": "trivial", "backend": "synthetic", "checkpoint": None,
            "episodes": 3, "horizon": 4, "pvt_condition_set": "none",
            "trade_off_preference": "most_robust",
        }
        payload.update(overrides)
        return payload

    def test_valid_trivial_synthetic_request_has_no_problems(self):
        self.assertEqual(web_ui._validate_request(self._base()), [])

    def test_bad_target_mode_is_rejected(self):
        problems = web_ui._validate_request(self._base(target_mode="nonsense"))
        self.assertTrue(any("target_mode" in p for p in problems))

    def test_custom_target_requires_all_four_fields(self):
        problems = web_ui._validate_request(self._base(target_mode="custom", target={"dfe_eye_width_ui": 0.5}))
        self.assertTrue(any("dfe_locked_phase_eye_height_v" in p for p in problems))

    def test_custom_target_with_all_fields_is_valid(self):
        target = {"dfe_locked_phase_eye_height_v": 0.1, "dfe_eye_width_ui": 0.4,
                  "dfe_min_margin_v": 0.0, "ctle_power_w": 0.015}
        self.assertEqual(web_ui._validate_request(self._base(target_mode="custom", target=target)), [])

    def test_real_backend_without_checkpoint_is_rejected(self):
        problems = web_ui._validate_request(self._base(backend="real", checkpoint=None))
        self.assertTrue(any("checkpoint" in p for p in problems))

    def test_nonexistent_checkpoint_is_rejected(self):
        problems = web_ui._validate_request(self._base(backend="real", checkpoint="results/does_not_exist.pt"))
        self.assertTrue(any("not found" in p for p in problems))

    def test_checkpoint_path_escaping_the_project_is_rejected(self):
        problems = web_ui._validate_request(self._base(backend="real", checkpoint="../../etc/passwd"))
        self.assertTrue(any("not found" in p for p in problems))

    def test_non_integer_episodes_is_rejected(self):
        problems = web_ui._validate_request(self._base(episodes="not-a-number"))
        self.assertTrue(any("episodes" in p for p in problems))

    def test_zero_episodes_is_rejected(self):
        problems = web_ui._validate_request(self._base(episodes=0))
        self.assertTrue(any("episodes" in p for p in problems))

    def test_bad_pvt_condition_set_is_rejected(self):
        problems = web_ui._validate_request(self._base(pvt_condition_set="full60"))
        self.assertTrue(any("pvt_condition_set" in p for p in problems))


class BuildArgvTests(unittest.TestCase):
    def test_trivial_preset_uses_target_mode_flag(self):
        argv = web_ui._build_argv(
            {"target_mode": "trivial", "backend": "synthetic", "episodes": 3, "horizon": 4,
             "pvt_condition_set": "none", "trade_off_preference": "most_robust"},
            output_path=Path("/tmp/x.json"), schematic_path=Path("/tmp/x.spice"),
        )
        self.assertIn("--target-mode", argv)
        self.assertIn("trivial", argv)
        self.assertNotIn("--target-json", argv)

    def test_custom_target_uses_target_json_flag(self):
        target = {"dfe_locked_phase_eye_height_v": 0.2, "dfe_eye_width_ui": 0.5,
                  "dfe_min_margin_v": 0.05, "ctle_power_w": 0.012}
        argv = web_ui._build_argv(
            {"target_mode": "custom", "target": target, "backend": "synthetic", "episodes": 3, "horizon": 4,
             "pvt_condition_set": "none", "trade_off_preference": "most_robust"},
            output_path=Path("/tmp/x.json"), schematic_path=Path("/tmp/x.spice"),
        )
        self.assertIn("--target-json", argv)
        self.assertNotIn("--target-mode", argv)
        json_arg = argv[argv.index("--target-json") + 1]
        self.assertEqual(json.loads(json_arg), target)

    def test_checkpoint_included_only_when_given(self):
        base = {"target_mode": "trivial", "backend": "synthetic", "episodes": 1, "horizon": 1,
                "pvt_condition_set": "none", "trade_off_preference": "most_robust"}
        argv_without = web_ui._build_argv(base, output_path=Path("/tmp/x.json"), schematic_path=Path("/tmp/x.spice"))
        self.assertNotIn("--checkpoint", argv_without)

        argv_with = web_ui._build_argv(
            {**base, "checkpoint": "results/foo.pt"}, output_path=Path("/tmp/x.json"), schematic_path=Path("/tmp/x.spice"),
        )
        self.assertIn("--checkpoint", argv_with)
        self.assertIn("results/foo.pt", argv_with)

    def test_module_invoked_is_the_existing_unmodified_pipeline_entry_point(self):
        argv = web_ui._build_argv(
            {"target_mode": "trivial", "backend": "synthetic", "episodes": 1, "horizon": 1,
             "pvt_condition_set": "none", "trade_off_preference": "most_robust"},
            output_path=Path("/tmp/x.json"), schematic_path=Path("/tmp/x.spice"),
        )
        self.assertIn("experiments.run_autockt_pipeline", argv)
        self.assertIn("-m", argv)


class ExecuteRunCrashHandlingTests(unittest.TestCase):
    """Simulates the real, intermittent SIGSEGV documented in
    docs/autockt-mapping.md sec 23 deterministically, via a mocked
    subprocess.Popen -- so this test does not depend on the crash actually
    occurring, and runs with zero real SPICE.
    """

    def test_signal_terminated_subprocess_is_reported_clearly_not_as_a_hang(self):
        run_id = "testrun_crash_00000000000000"
        with web_ui._RUNS_LOCK:
            web_ui._RUNS[run_id] = {
                "run_id": run_id, "status": "queued", "command": ["python", "-m", "x"],
                "output_path": "/tmp/nope.json", "schematic_path": "/tmp/nope.spice",
                "started_at": None, "finished_at": None, "returncode": None,
                "error": None, "result": None, "stdout_tail": "", "stderr_tail": "",
            }
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("partial stdout before crash", "")
        mock_proc.returncode = -11  # SIGSEGV, as subprocess.Popen reports on POSIX
        with patch("experiments.web_ui.subprocess.Popen", return_value=mock_proc):
            web_ui._execute_run(run_id, ["python", "-m", "x"], Path("/tmp/nope.json"), Path("/tmp/nope.spice"))

        payload = web_ui._status_payload(run_id)
        self.assertEqual(payload["status"], "failed")
        self.assertIn("signal 11", payload["error"])
        self.assertIn("SIGSEGV", payload["error"])

    def test_nonzero_exit_without_signal_is_reported_as_a_normal_failure(self):
        run_id = "testrun_fail_000000000000000"
        with web_ui._RUNS_LOCK:
            web_ui._RUNS[run_id] = {
                "run_id": run_id, "status": "queued", "command": ["python", "-m", "x"],
                "output_path": "/tmp/nope2.json", "schematic_path": "/tmp/nope2.spice",
                "started_at": None, "finished_at": None, "returncode": None,
                "error": None, "result": None, "stdout_tail": "", "stderr_tail": "",
            }
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("", "ValueError: invalid target specification")
        mock_proc.returncode = 1
        with patch("experiments.web_ui.subprocess.Popen", return_value=mock_proc):
            web_ui._execute_run(run_id, ["python", "-m", "x"], Path("/tmp/nope2.json"), Path("/tmp/nope2.spice"))

        payload = web_ui._status_payload(run_id)
        self.assertEqual(payload["status"], "failed")
        self.assertNotIn("signal", payload["error"])
        self.assertIn("non-zero status (1)", payload["error"])


class HttpIntegrationTests(unittest.TestCase):
    """Full request/response cycle over real HTTP, using --backend synthetic
    (no SPICE) end to end -- exercises the same code path a browser would.
    """

    @classmethod
    def setUpClass(cls):
        cls.server = web_ui.ThreadingHTTPServer(("127.0.0.1", 0), web_ui.Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _get(self, path: str) -> tuple[int, dict]:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}") as resp:  # type: HTTPResponse
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def _post(self, path: str, payload: dict) -> tuple[int, dict]:
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_index_page_serves_html(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/") as resp:
            body = resp.read().decode("utf-8")
        self.assertEqual(resp.status, 200)
        self.assertIn("Run NEBULA", body)

    def test_checkpoints_endpoint_lists_pt_files(self):
        status, data = self._get("/api/checkpoints")
        self.assertEqual(status, 200)
        self.assertIsInstance(data["checkpoints"], list)

    def test_invalid_request_returns_400_with_problems(self):
        status, data = self._post("/api/run", {"target_mode": "bogus"})
        self.assertEqual(status, 400)
        self.assertIn("problems", data)

    def test_unknown_run_id_returns_404(self):
        status, data = self._get("/api/status/" + "0" * 32)
        self.assertEqual(status, 404)

    def test_malformed_run_id_returns_400(self):
        status, data = self._get("/api/status/not-a-valid-id")
        self.assertEqual(status, 400)

    def test_full_synthetic_run_completes_and_reports_a_final_specification(self):
        with TemporaryDirectory():
            status, data = self._post("/api/run", {
                "target_mode": "trivial", "backend": "synthetic", "checkpoint": None,
                "episodes": 8, "horizon": 1, "initial_indices_source": "grid-center",
                "pvt_condition_set": "none", "trade_off_preference": "most_robust",
            })
            self.assertEqual(status, 202)
            run_id = data["run_id"]

            deadline = threading.Event()
            result_payload = None
            for _ in range(60):
                s, payload = self._get(f"/api/status/{run_id}")
                if payload["status"] not in ("queued", "running"):
                    result_payload = payload
                    break
                deadline.wait(0.5)

            self.assertIsNotNone(result_payload, "synthetic run did not finish in time")
            self.assertEqual(result_payload["status"], "completed")
            self.assertIn("final_specification", result_payload["result"])
            self.assertIn("rows", result_payload["result"]["final_specification"])


if __name__ == "__main__":
    unittest.main()

"""Minimal, competition-demo local UI around experiments/run_autockt_pipeline.py.

Runnable from a clean checkout with one command:

    python -m experiments.web_ui

then open http://127.0.0.1:8765 in a browser.

DESIGN: this server never imports or calls PPO/simulator/reward/PVT/
candidate-selection code itself. Every "Run NEBULA" click launches the
existing, unmodified `python -m experiments.run_autockt_pipeline ...` CLI
as a SEPARATE SUBPROCESS (the exact same entry point used from the command
line and in the real-SPICE smoke test) and only reads back its own JSON
`--output` file -- the pipeline stays the single source of truth, nothing
here duplicates optimization/selection logic.

Subprocess isolation is also a deliberate safety choice, not just
convenience: a real-SPICE run can (rarely, per docs/autockt-mapping.md
sec 23's smoke-test investigation) crash the Python process with a native
SIGSEGV. Because the pipeline runs in a child process, that kind of crash
only ends the run being demonstrated -- it cannot take down this UI
server, and is reported to the judge as a clear, specific status rather
than the whole demo dying silently.

Each run writes to a NEW file under results/web_ui_runs/<run_id>.json
(directory created on first use) -- historical results/*.json files are
never touched.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "results" / "web_ui_runs"
HOST = "127.0.0.1"
DEFAULT_PORT = 8765
TAIL_CHARS = 4000  # stdout/stderr tail kept per run, to bound memory/response size

_RUNS: dict[str, dict[str, Any]] = {}
_RUNS_LOCK = threading.Lock()

TARGET_MODES = ("trivial", "hard", "custom")
BACKENDS = ("synthetic", "real")
PVT_CONDITION_SETS = ("none", "smoke")
TRADE_OFF_PREFERENCES = (
    "most_robust", "lowest_power", "strongest_eye_height", "widest_eye", "largest_margin", "balanced",
)
TARGET_SPEC_FIELDS = (
    "dfe_locked_phase_eye_height_v", "dfe_eye_width_ui", "dfe_min_margin_v", "ctle_power_w",
)

_RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")


# ---------------------------------------------------------------------------
# Run orchestration -- builds the exact CLI invocation and executes it as a
# subprocess. No pipeline logic lives here.
# ---------------------------------------------------------------------------

def _validate_request(payload: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if payload.get("target_mode") not in TARGET_MODES:
        problems.append(f"target_mode must be one of {TARGET_MODES}")
    if payload.get("target_mode") == "custom":
        target = payload.get("target") or {}
        for field in TARGET_SPEC_FIELDS:
            value = target.get(field)
            if not isinstance(value, (int, float)):
                problems.append(f"target.{field} must be a number")
    if payload.get("backend") not in BACKENDS:
        problems.append(f"backend must be one of {BACKENDS}")
    try:
        episodes = int(payload.get("episodes", 0))
        if episodes < 1:
            problems.append("episodes must be >= 1")
    except (TypeError, ValueError):
        problems.append("episodes must be an integer")
    try:
        horizon = int(payload.get("horizon", 0))
        if horizon < 1:
            problems.append("horizon must be >= 1")
    except (TypeError, ValueError):
        problems.append("horizon must be an integer")
    if payload.get("backend") == "real" and not payload.get("checkpoint"):
        problems.append("backend='real' requires a checkpoint (an untrained policy against real SPICE "
                         "has no learned behavior to demonstrate -- see docs/autockt-mapping.md sec 20)")
    checkpoint = payload.get("checkpoint")
    if checkpoint:
        checkpoint_path = (PROJECT_ROOT / checkpoint).resolve()
        if PROJECT_ROOT not in checkpoint_path.parents or not checkpoint_path.is_file():
            problems.append(f"checkpoint not found inside the project: {checkpoint}")
    if payload.get("pvt_condition_set") not in PVT_CONDITION_SETS:
        problems.append(f"pvt_condition_set must be one of {PVT_CONDITION_SETS}")
    if payload.get("trade_off_preference") not in TRADE_OFF_PREFERENCES:
        problems.append(f"trade_off_preference must be one of {TRADE_OFF_PREFERENCES}")
    return problems


def _build_argv(payload: dict[str, Any], *, output_path: Path, schematic_path: Path) -> list[str]:
    argv = [
        sys.executable, "-m", "experiments.run_autockt_pipeline",
        "--backend", payload["backend"],
        "--episodes", str(int(payload["episodes"])),
        "--horizon", str(int(payload["horizon"])),
        "--initial-indices-source", payload.get("initial_indices_source", "verified"),
        "--pvt-condition-set", payload["pvt_condition_set"],
        "--trade-off-preference", payload["trade_off_preference"],
        "--output", str(output_path),
        "--export-schematic", str(schematic_path),
    ]
    if payload["target_mode"] == "custom":
        argv += ["--target-json", json.dumps({f: float(payload["target"][f]) for f in TARGET_SPEC_FIELDS})]
    else:
        argv += ["--target-mode", payload["target_mode"]]
    if payload.get("checkpoint"):
        argv += ["--checkpoint", payload["checkpoint"]]
    return argv


def _execute_run(run_id: str, argv: list[str], output_path: Path, schematic_path: Path) -> None:
    with _RUNS_LOCK:
        _RUNS[run_id]["status"] = "running"
        _RUNS[run_id]["started_at"] = time.monotonic()

    try:
        proc = subprocess.Popen(
            argv, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        stdout, stderr = proc.communicate()
        returncode = proc.returncode
        launch_error: Optional[str] = None
    except OSError as exc:
        stdout, stderr, returncode = "", "", None
        launch_error = f"failed to launch the pipeline subprocess: {exc}"

    finished_at = time.monotonic()

    with _RUNS_LOCK:
        entry = _RUNS[run_id]
        entry["finished_at"] = finished_at
        entry["returncode"] = returncode
        entry["stdout_tail"] = (stdout or "")[-TAIL_CHARS:]
        entry["stderr_tail"] = (stderr or "")[-TAIL_CHARS:]

        if launch_error is not None:
            entry["status"] = "failed"
            entry["error"] = launch_error
        elif returncode == 0 and output_path.is_file():
            try:
                entry["result"] = json.loads(output_path.read_text(encoding="utf-8"))
                entry["status"] = "completed"
            except (OSError, json.JSONDecodeError) as exc:
                entry["status"] = "failed"
                entry["error"] = f"pipeline exited cleanly but its output file could not be read: {exc}"
        elif returncode is not None and returncode < 0:
            entry["status"] = "failed"
            entry["error"] = (
                f"the pipeline subprocess was terminated by signal {-returncode} "
                f"(e.g. 11 = SIGSEGV) -- a native-level crash, not a Python error. "
                "This can occur intermittently during real-SPICE runs; see "
                "docs/autockt-mapping.md sec 23 for the investigation. See the "
                "stderr/stdout tails below for whatever was captured before the crash. "
                "Retrying the same run is a reasonable next step."
            )
        else:
            entry["status"] = "failed"
            entry["error"] = f"the pipeline exited with a non-zero status ({returncode}); see stderr below"


def _start_run(payload: dict[str, Any]) -> str:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    output_path = RUNS_DIR / f"{run_id}.json"
    schematic_path = RUNS_DIR / f"{run_id}_schematic.spice"
    argv = _build_argv(payload, output_path=output_path, schematic_path=schematic_path)

    with _RUNS_LOCK:
        _RUNS[run_id] = {
            "run_id": run_id, "status": "queued", "command": argv,
            "output_path": str(output_path), "schematic_path": str(schematic_path),
            "started_at": None, "finished_at": None, "returncode": None,
            "error": None, "result": None, "stdout_tail": "", "stderr_tail": "",
        }

    thread = threading.Thread(target=_execute_run, args=(run_id, argv, output_path, schematic_path), daemon=True)
    thread.start()
    return run_id


def _status_payload(run_id: str) -> Optional[dict[str, Any]]:
    with _RUNS_LOCK:
        entry = _RUNS.get(run_id)
        if entry is None:
            return None
        entry = dict(entry)  # shallow copy for a consistent snapshot

    if entry["started_at"] is None:
        elapsed_s = 0.0
    elif entry["finished_at"] is not None:
        elapsed_s = entry["finished_at"] - entry["started_at"]
    else:
        elapsed_s = time.monotonic() - entry["started_at"]

    payload = {
        "run_id": entry["run_id"], "status": entry["status"], "elapsed_s": round(elapsed_s, 1),
        "command": " ".join(entry["command"]), "returncode": entry["returncode"], "error": entry["error"],
        "stdout_tail": entry["stdout_tail"], "stderr_tail": entry["stderr_tail"], "result": entry["result"],
    }
    if entry["result"] is not None and entry["result"].get("schematic_path"):
        payload["schematic_path"] = entry["result"]["schematic_path"]
    return payload


def _available_checkpoints() -> list[str]:
    results_dir = PROJECT_ROOT / "results"
    if not results_dir.is_dir():
        return []
    return sorted(
        str(p.relative_to(PROJECT_ROOT)) for p in results_dir.glob("*.pt")
    )


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "NebulaUI/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter default logging
        sys.stderr.write(f"[web_ui] {self.address_string()} - {fmt % args}\n")

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, status: int, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (stdlib method name)
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(200, INDEX_HTML)
        elif path == "/api/checkpoints":
            self._send_json(200, {"checkpoints": _available_checkpoints()})
        elif path.startswith("/api/status/"):
            run_id = path[len("/api/status/"):]
            if not _RUN_ID_RE.match(run_id):
                self._send_json(400, {"error": "invalid run_id"})
                return
            payload = _status_payload(run_id)
            if payload is None:
                self._send_json(404, {"error": "unknown run_id"})
                return
            self._send_json(200, payload)
        elif path.startswith("/api/schematic/"):
            run_id = path[len("/api/schematic/"):]
            if not _RUN_ID_RE.match(run_id):
                self._send_json(400, {"error": "invalid run_id"})
                return
            schematic_path = RUNS_DIR / f"{run_id}_schematic.spice"
            if not schematic_path.is_file():
                self._send_json(404, {"error": "no schematic for this run (not yet finished, or PVT/nominal "
                                                 "selection found no feasible design)"})
                return
            self._send_json(200, {"schematic": schematic_path.read_text(encoding="utf-8")})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/run":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "invalid JSON body"})
            return

        problems = _validate_request(payload)
        if problems:
            self._send_json(400, {"error": "invalid request", "problems": problems})
            return

        run_id = _start_run(payload)
        self._send_json(202, {"run_id": run_id})


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NEBULA -- Automated RF/Analog Receiver Sizing</title>
<style>
  :root {
    --bg: #0f1419; --panel: #161b22; --border: #2a323d; --text: #e6edf3; --muted: #8b96a3;
    --accent: #4fd1c5; --accent-dark: #2c9c92; --pass: #3fb950; --fail: #f85149; --notclaimed: #d29922;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text); font-family: -apple-system, "Segoe UI", Roboto, sans-serif; }
  header { padding: 20px 28px; border-bottom: 1px solid var(--border); }
  header h1 { margin: 0; font-size: 20px; letter-spacing: 0.3px; }
  header p { margin: 4px 0 0; color: var(--muted); font-size: 13px; }
  main { display: grid; grid-template-columns: 380px 1fr; gap: 0; min-height: calc(100vh - 74px); }
  .panel { padding: 20px 24px; }
  .left { border-right: 1px solid var(--border); overflow-y: auto; }
  fieldset { border: 1px solid var(--border); border-radius: 8px; margin: 0 0 16px; padding: 14px 16px; }
  legend { padding: 0 6px; color: var(--accent); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
  label { display: block; font-size: 12px; color: var(--muted); margin: 10px 0 4px; }
  label:first-child { margin-top: 0; }
  input, select { width: 100%; background: #0d1117; border: 1px solid var(--border); color: var(--text);
                  border-radius: 6px; padding: 7px 9px; font-size: 13px; }
  input:focus, select:focus { outline: 1px solid var(--accent); }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  #customTargetFields { display: none; }
  button#runBtn { width: 100%; margin-top: 6px; padding: 11px; border: none; border-radius: 8px;
                  background: var(--accent); color: #06201d; font-weight: 700; font-size: 14px; cursor: pointer; }
  button#runBtn:disabled { background: #33403f; color: #7c8b89; cursor: not-allowed; }
  button#runBtn:not(:disabled):hover { background: var(--accent-dark); color: #eafffb; }
  .status-badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
  .status-idle { background: #2a323d; color: var(--muted); }
  .status-running { background: #2c2a12; color: var(--notclaimed); }
  .status-completed { background: #0d2a17; color: var(--pass); }
  .status-failed { background: #2a1414; color: var(--fail); }
  #statusLine { display: flex; align-items: center; gap: 10px; margin: 16px 0; font-size: 13px; color: var(--muted); }
  #resultsEmpty { color: var(--muted); font-size: 14px; padding: 40px 0; text-align: center; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 18px 20px; margin-bottom: 16px; }
  .card h2 { margin: 0 0 12px; font-size: 14px; color: var(--accent); text-transform: uppercase; letter-spacing: 0.5px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; }
  .verdict { padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; }
  .verdict-PASS { background: #0d2a17; color: var(--pass); }
  .verdict-FAIL { background: #2a1414; color: var(--fail); }
  .verdict-NOT-CLAIMED { background: #2a2210; color: var(--notclaimed); }
  .param-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 10px; }
  .param-grid div { background: #0d1117; border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; }
  .param-grid .k { color: var(--muted); font-size: 11px; }
  .param-grid .v { font-family: ui-monospace, monospace; font-size: 13px; margin-top: 2px; }
  pre.log { background: #0d1117; border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px;
            font-size: 11.5px; max-height: 220px; overflow: auto; white-space: pre-wrap; word-break: break-word; }
  .error-box { background: #2a1414; border: 1px solid #6e2323; color: #ffb4ac; border-radius: 8px;
               padding: 12px 14px; font-size: 13px; margin-bottom: 16px; }
  .links a { color: var(--accent); text-decoration: none; font-size: 13px; margin-right: 16px; }
  .links a:hover { text-decoration: underline; }
  .hint { color: var(--muted); font-size: 11.5px; margin-top: 4px; }
  code { color: var(--accent); }
</style>
</head>
<body>
<header>
  <h1>NEBULA</h1>
  <p>Automated RL-driven PCIe Gen-2 receiver (CTLE+DFE) sizing -- real SPICE, end-to-end pipeline demo</p>
</header>
<main>
  <div class="panel left">
    <fieldset>
      <legend>Target specification</legend>
      <label for="targetMode">Preset</label>
      <select id="targetMode">
        <option value="trivial">Trivial (existing thresholds)</option>
        <option value="hard">Hard target</option>
        <option value="custom">Custom (enter values)</option>
      </select>
      <div id="customTargetFields">
        <label for="tEyeHeight">Min eye height (V)</label>
        <input id="tEyeHeight" type="number" step="any" value="0.1">
        <label for="tEyeWidth">Min eye width (UI, 0-1)</label>
        <input id="tEyeWidth" type="number" step="any" value="0.4">
        <label for="tMargin">Min DFE margin (V)</label>
        <input id="tMargin" type="number" step="any" value="0.0">
        <label for="tPower">Max CTLE power (W)</label>
        <input id="tPower" type="number" step="any" value="0.015">
      </div>
    </fieldset>

    <fieldset>
      <legend>Candidate generation</legend>
      <label for="backend">Backend</label>
      <select id="backend">
        <option value="synthetic">Synthetic (no SPICE -- fast dry run)</option>
        <option value="real">Real ngspice (slow, actual SPICE)</option>
      </select>
      <label for="checkpoint">PPO checkpoint</label>
      <select id="checkpoint"><option value="">(untrained policy -- synthetic backend only)</option></select>
      <div class="row">
        <div>
          <label for="episodes">Episodes</label>
          <input id="episodes" type="number" min="1" value="3">
        </div>
        <div>
          <label for="horizon">Horizon</label>
          <input id="horizon" type="number" min="1" value="4">
        </div>
      </div>
      <label for="initSource">Initial state</label>
      <select id="initSource">
        <option value="verified">Verified (warm start)</option>
        <option value="grid-center">Grid center (unbiased)</option>
      </select>
    </fieldset>

    <fieldset>
      <legend>PVT-aware selection</legend>
      <label for="pvtSet">Condition set</label>
      <select id="pvtSet">
        <option value="none">None (nominal-only selection)</option>
        <option value="smoke">Smoke (2 conditions: nominal + stress corner)</option>
      </select>
      <label for="tradeOff">Trade-off preference (used on PVT ties)</label>
      <select id="tradeOff">
        <option value="most_robust">Most robust</option>
        <option value="lowest_power">Lowest power</option>
        <option value="strongest_eye_height">Strongest eye height</option>
        <option value="widest_eye">Widest eye</option>
        <option value="largest_margin">Largest margin</option>
        <option value="balanced">Balanced</option>
      </select>
      <p class="hint">"smoke" spends real SPICE only on nominally-feasible candidates -- not the full 27-point robustness sweep.</p>
    </fieldset>

    <button id="runBtn">Run NEBULA</button>
    <div id="statusLine"><span id="statusBadge" class="status-badge status-idle">idle</span><span id="elapsed"></span></div>
  </div>

  <div class="panel">
    <div id="resultsEmpty">Configure a target and click <strong>Run NEBULA</strong> to see results here.</div>
    <div id="errorBox" class="error-box" style="display:none"></div>
    <div id="results" style="display:none">
      <div class="card">
        <h2>Final circuit parameters</h2>
        <div id="paramGrid" class="param-grid"></div>
      </div>
      <div class="card">
        <h2>Measured specification</h2>
        <table><thead><tr><th>Metric</th><th>Measured</th><th>Requirement</th><th>Verdict</th></tr></thead>
        <tbody id="specRows"></tbody></table>
      </div>
      <div class="card" id="pvtCard" style="display:none">
        <h2>PVT robustness</h2>
        <div id="pvtSummary"></div>
      </div>
      <div class="card">
        <h2>Run summary</h2>
        <div id="runSummary" class="param-grid"></div>
        <div class="links" style="margin-top:12px">
          <a href="#" id="schematicLink">View schematic</a>
          <a href="#" id="jsonLink">View raw pipeline output (JSON)</a>
        </div>
        <pre class="log" id="schematicView" style="display:none; margin-top:10px"></pre>
      </div>
      <div class="card">
        <h2>Process log</h2>
        <label>stdout</label>
        <pre class="log" id="stdoutLog"></pre>
        <label>stderr</label>
        <pre class="log" id="stderrLog"></pre>
      </div>
    </div>
  </div>
</main>
<script>
const $ = (id) => document.getElementById(id);
let pollTimer = null;

$('targetMode').addEventListener('change', () => {
  $('customTargetFields').style.display = $('targetMode').value === 'custom' ? 'block' : 'none';
});

fetch('/api/checkpoints').then(r => r.json()).then(data => {
  const sel = $('checkpoint');
  for (const path of data.checkpoints) {
    const opt = document.createElement('option');
    opt.value = path; opt.textContent = path;
    sel.appendChild(opt);
  }
  if (data.checkpoints.length) sel.value = data.checkpoints[0];
});

function setBadge(status) {
  const badge = $('statusBadge');
  badge.className = 'status-badge status-' + status;
  badge.textContent = status;
}

function buildPayload() {
  const targetMode = $('targetMode').value;
  const payload = {
    target_mode: targetMode,
    backend: $('backend').value,
    checkpoint: $('checkpoint').value || null,
    episodes: parseInt($('episodes').value, 10),
    horizon: parseInt($('horizon').value, 10),
    initial_indices_source: $('initSource').value,
    pvt_condition_set: $('pvtSet').value,
    trade_off_preference: $('tradeOff').value,
  };
  if (targetMode === 'custom') {
    payload.target = {
      dfe_locked_phase_eye_height_v: parseFloat($('tEyeHeight').value),
      dfe_eye_width_ui: parseFloat($('tEyeWidth').value),
      dfe_min_margin_v: parseFloat($('tMargin').value),
      ctle_power_w: parseFloat($('tPower').value),
    };
  }
  return payload;
}

function showError(msg) {
  $('errorBox').style.display = 'block';
  $('errorBox').textContent = msg;
  $('results').style.display = 'none';
  $('resultsEmpty').style.display = 'none';
}

function verdictClass(v) { return 'verdict verdict-' + v.replace(/ /g, '-'); }

function renderResult(payload) {
  $('errorBox').style.display = 'none';
  $('resultsEmpty').style.display = 'none';
  $('results').style.display = 'block';

  const result = payload.result;
  const selected = result.selection && result.selection.selected;

  const paramGrid = $('paramGrid');
  paramGrid.innerHTML = '';
  if (selected) {
    for (const [k, v] of Object.entries(selected.parameters)) {
      const div = document.createElement('div');
      div.innerHTML = `<div class="k">${k}</div><div class="v">${v}</div>`;
      paramGrid.appendChild(div);
    }
  } else {
    paramGrid.innerHTML = '<div class="k">No feasible design was selected by this run.</div>';
  }

  const specRows = $('specRows');
  specRows.innerHTML = '';
  const rows = (result.final_specification && result.final_specification.rows) || [];
  for (const row of rows) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${row.metric}</td><td>${row.measured ?? 'n/a'}</td><td>${row.requirement}</td>` +
                    `<td><span class="${verdictClass(row.verdict)}">${row.verdict}</span></td>`;
    specRows.appendChild(tr);
  }

  const pvt = selected && result.selection.pvt;
  if (pvt) {
    $('pvtCard').style.display = 'block';
    $('pvtSummary').innerHTML =
      `<div class="param-grid"><div><div class="k">Pass rate</div><div class="v">${pvt.n_passing}/${pvt.n_conditions}` +
      ` (${(pvt.pass_rate*100).toFixed(0)}%)</div></div>` +
      `<div><div class="k">Met minimum</div><div class="v">${pvt.met_minimum_pass_rate}</div></div></div>`;
  } else {
    $('pvtCard').style.display = 'none';
  }

  $('runSummary').innerHTML =
    `<div><div class="k">Candidates generated</div><div class="v">${result.n_candidates_generated}</div></div>` +
    `<div><div class="k">Nominally feasible</div><div class="v">${result.n_nominally_feasible}</div></div>` +
    `<div><div class="k">Backend</div><div class="v">${result.backend}</div></div>` +
    `<div><div class="k">Elapsed</div><div class="v">${payload.elapsed_s}s</div></div>`;

  $('schematicLink').onclick = (e) => {
    e.preventDefault();
    fetch(`/api/schematic/${payload.run_id}`).then(r => r.json()).then(d => {
      const view = $('schematicView');
      view.textContent = d.schematic || d.error;
      view.style.display = 'block';
    });
  };
  $('jsonLink').onclick = (e) => { e.preventDefault(); showJson(result); };

  $('stdoutLog').textContent = payload.stdout_tail || '(empty)';
  $('stderrLog').textContent = payload.stderr_tail || '(empty)';
}

function showJson(result) {
  const view = $('schematicView');
  view.textContent = JSON.stringify(result, null, 2);
  view.style.display = 'block';
}

function poll(runId) {
  fetch(`/api/status/${runId}`).then(r => r.json()).then(payload => {
    setBadge(payload.status);
    $('elapsed').textContent = payload.elapsed_s + 's elapsed';
    if (payload.status === 'running' || payload.status === 'queued') {
      pollTimer = setTimeout(() => poll(runId), 1200);
      return;
    }
    $('runBtn').disabled = false;
    if (payload.status === 'completed') {
      renderResult(payload);
    } else {
      showError((payload.error || 'run failed') + '\n\nstderr tail:\n' + (payload.stderr_tail || '(empty)'));
      $('stdoutLog') && ($('stdoutLog').textContent = payload.stdout_tail || '(empty)');
    }
  }).catch(err => {
    setBadge('failed');
    showError('Lost contact with the NEBULA UI server: ' + err);
    $('runBtn').disabled = false;
  });
}

$('runBtn').addEventListener('click', () => {
  if (pollTimer) clearTimeout(pollTimer);
  $('errorBox').style.display = 'none';
  $('results').style.display = 'none';
  $('resultsEmpty').style.display = 'none';
  $('runBtn').disabled = true;
  setBadge('queued');
  $('elapsed').textContent = '';

  fetch('/api/run', {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(buildPayload()),
  }).then(async (r) => {
    const data = await r.json();
    if (!r.ok) {
      $('runBtn').disabled = false;
      showError('Could not start run:\n' + (data.problems || [data.error]).join('\n'));
      setBadge('idle');
      return;
    }
    poll(data.run_id);
  }).catch(err => {
    $('runBtn').disabled = false;
    setBadge('idle');
    showError('Could not reach the NEBULA UI server: ' + err);
  });
});
</script>
</body>
</html>
"""


def main() -> int:
    server = ThreadingHTTPServer((HOST, DEFAULT_PORT), Handler)
    print(f"NEBULA UI running at http://{HOST}:{DEFAULT_PORT}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

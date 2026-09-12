# Fast design and separate validation

Quick search defaults to 120 seconds and four evaluations. The UI exposes both limits. Search uses staged checks and a persistent, provenance-aware cache; failures stop expensive downstream work. Per-stage simulator timeouts are capped by remaining time. This is a soft wall-clock budget: process setup, parsing, and an in-progress multi-run stage can add overhead. A budget is not a guarantee that a feasible design will be found.

## Setup and checks

From the repository root, using your Python environment:

```powershell
python -m experiments.preflight
python -m experiments.restore_artifacts
python -m tests.run_fast_suite
```

Install requirements first. Real simulation needs ngspice and a SKY130 model library. Set NGSPICE_EXECUTABLE and SKY130_MODEL_LIBRARY as described in the existing setup documentation. An explicitly requested missing model fails rather than silently using another model. Local generated tools/models are ignored by Git and are not bundled with these code changes. Preflight records versions and model fingerprints.

## Existing policy: quick design

```powershell
python -m experiments.run_autockt_pipeline --backend real --checkpoint results/autockt_mixed_target_confirmation_policy.pt --episodes 3 --horizon 4 --search-seconds 120 --max-evaluations 4 --output generated/quick.json
```

Selection uses strict measured constraints independently of reward. Missing metrics or failed simulation cannot qualify. No passing candidate means no selection or misleading design export. Synthetic runs are demonstrations only.

## New sizing policy

The separate nine-variable policy searches load/degeneration resistance, degeneration capacitance, DFE tap, input and tail W/L, and mirror bias resistance. Training is offline. Old five-variable checkpoints are incompatible and rejected. Training writes a checkpoint only after at least one PPO update completes; its JSON records updates and runtime.

```powershell
python -m experiments.train_sized_receiver --output generated/sizing-policy.pt --seconds 600 --max-evaluations 64
python -m experiments.design_sized_receiver --checkpoint generated/sizing-policy.pt --output generated/sizing-run.json --design-output generated/selected.json --seconds 120 --max-evaluations 4
```

A selected design file is emitted only when a candidate passes nominal training-fidelity constraints. Small training budgets validate the workflow, not convergence or design quality. The existing web UI uses the legacy policy; use this CLI for the new sizing policy.

## Finalist validation, reporting, and export

```powershell
python -m experiments.validate_finalist --design generated/selected.json --output generated/validation.json --full-pvt --seconds 600 --max-evaluations 8 --report generated/report.html
python -m experiments.export_bundle --design generated/selected.json --output generated/verification-bundle
```

Full PVT covers TT/SS/FF/SF/FS, 1.71/1.80/1.89 V, and 0/27/75/125 C (60 points per channel). Stress adds PRBS15 and seeded 5 ps RMS jitter, each with 4096 bits. Repeat --channel for additional channel files. Eight evaluations cannot complete the full grid: re-running identical validation reuses provenance-matched cached evaluations without charging them against its simulation count, then progresses to new work. Retryable failures are retried. Check completed/planned and all_requested_pass; a partial grid is never a complete pass.

Optional --tuning runs a separate 120-second sampled RC tuning check; sampled coverage does not prove continuous tunability. Optional --physical-receiver runs the experimental transistor sampler separately (up to 60 seconds). These optional checks have additional budgets. Their results remain distinct from behavioral DFE measurements.

The HTML report uses measured metrics and sparse measured gain samples. Area includes explicit MOS/passive/placement assumptions and remains an estimate, not layout verification. Export contains editable SPICE benches, flattened model dependencies, channel/stimulus, checksums and conditions. It is not a graphical CAD schematic, extracted layout, or a physical-DFE compliance certificate.

## Fair comparisons

```powershell
python -m experiments.benchmark_sized_receiver --checkpoint generated/sizing-policy.pt --output generated/benchmark.json --seconds-per-trial 120 --evaluations 20
```

Compares random search, CEM and PPO in the same nine-variable bounds across three seeds. Supply --targets heldout.json with a list of target dictionaries to evaluate targets excluded from training. Report training cost separately; cache hits and local versus global proposals affect timing comparisons. No PPO superiority is claimed without measured evidence.

## Verified in this change

- 345 fast tests passed, including strict acceptance, sizing, checkpoint compatibility, cache budgets, jitter, export and reporting.
- Real legacy-policy search: four evaluations in 18.23 seconds; no strictly feasible design selected.
- Real sizing training smoke: one PPO update/two evaluations in 5.64 seconds, excluding Python startup. This is not a converged policy.
- Transistor-bias DC example passed headroom/current checks.
- Physical sampler test failed: 49 errors in 96 measured bits. It remains experimental; total receiver functionality, clock-driver power, layout area, full PVT and continuous tuning are not established.

Local verification used Python 3.12, numpy 2.5.2, torch 2.14.0+cpu and ngspice 38 with official SKY130 model sources. Preflight JSON retains the exact environment/model identity. Runtime varies by machine, channel and candidate.

## Larger RL steps

New sizing-policy training defaults to a normalized step of 0.2 (previously 0.1): 10% of the normalized search span per action instead of 5%. Log-scaled parameters therefore move multiplicatively. Values remain clipped to their allowed bounds; hold actions still make no change.

Use `--step-size 0.2` with train_sized_receiver or design_sized_receiver to choose the step. Checkpoints record their training step, and inference/benchmarks use it automatically. Older sizing checkpoints without this metadata retain 0.1 unless explicitly overridden. The legacy five-head UI policy is unchanged. Larger steps may skip feasible regions; they do not reduce individual SPICE simulation time or guarantee faster convergence. Evaluation/time caps remain unchanged.

## Measured parallel-training improvement

New sizing training defaults to `--workers 2`; use `--workers 1` to reduce CPU usage. Independent episodes simulate concurrently, while policy action sampling and PPO updates stay on the main thread. Each episode remains contiguous when computing advantages, including its own horizon bootstrap. The two workers share the same evaluation/time allowance; parallelism does not double that allowance. An incomplete rollout at budget exhaustion is discarded, as in serial training, and only completed updates are saved.

A cold-cache, four-design nominal transient experiment measured 24.69 seconds serial versus 14.77 seconds with two workers (1.67x throughput, about 40% less wall time). All returned scalar metrics were identical. This measures the transient stage, not full training or convergence. The small local sample does not cover failing/borderline designs, all corners or channels. A real two-worker training smoke completed one update/four evaluations in 8.48 seconds; it does not establish policy convergence.

96-bit waveforms took 17.11 seconds serial, but eye height changed by up to 8.98%. They were NOT made the training default. `SimulationConditions(training_bit_count=96)` is available for explicit experiments only; normal training remains 128 bits and final validation remains 1024 or the explicit validation length.

Reproduce the speed experiment with:

```powershell
python -m experiments.benchmark_simulation_speed --design results/design_a_final_specification.json --output generated/simulation-speed.json
```

See [recorded speed comparison](simulation-speed-results.json). Faster throughput allows more experience within a time budget, but improved final model/design quality still requires held-out evaluation. Existing single-trajectory inference and the legacy UI do not become parallel through this change.

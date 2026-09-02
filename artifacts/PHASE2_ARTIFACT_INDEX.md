# PHASE2_ARTIFACT_INDEX.md — NEBULA Phase 2 Experimental Artifacts Index

Package: `NEBULA_phase2_experimental_artifacts.zip` -- built from git commit `4478e326afd2108697e7604c5896b83609ef9e23` on branch `main`. 144 files total.

All files were copied byte-for-byte from the repository (verified via SHA-256 comparison at packaging time -- see `PHASE2_SHA256SUMS.txt`). No historical file was modified, regenerated, or deleted. Missing metadata is marked `NOT RECORDED`.

## Experimental results (`results/`)

| Path | Experiment | Purpose | Status | Seed | Relationship to reported result |
|---|---|---|---|---|---|
| `results/autockt_fair_headtohead_seed123.jsonl` | PPO (fair, no-warm-start head-to-head) | grid-center init, n=20, horizon=1, no warm start | 0/20 (all failed at dc) | 123 | Fair-benchmark headline: 0/20, matches CEM/RS methodology |
| `results/autockt_hard_learning_run1.jsonl` | Early development / smoke test | Preserved as-is; referenced in docs/autockt-mapping.md's early sections | NOT RECORDED | NOT RECORDED | Development history, not a headline reported result |
| `results/autockt_hard_learning_run2.jsonl` | Early development / smoke test | Preserved as-is; referenced in docs/autockt-mapping.md's early sections | NOT RECORDED | NOT RECORDED | Development history, not a headline reported result |
| `results/autockt_hard_smoke_seed43.jsonl` | Early development / smoke test | Preserved as-is; referenced in docs/autockt-mapping.md's early sections | NOT RECORDED | NOT RECORDED | Development history, not a headline reported result |
| `results/autockt_hard_smoke_seed44.jsonl` | Early development / smoke test | Preserved as-is; referenced in docs/autockt-mapping.md's early sections | NOT RECORDED | NOT RECORDED | Development history, not a headline reported result |
| `results/autockt_hard_smoke_test.jsonl` | Early development / smoke test | Preserved as-is; referenced in docs/autockt-mapping.md's early sections | NOT RECORDED | NOT RECORDED | Development history, not a headline reported result |
| `results/autockt_loggrid_smoke.jsonl` | Early development / smoke test | Preserved as-is; referenced in docs/autockt-mapping.md's early sections | NOT RECORDED | NOT RECORDED | Development history, not a headline reported result |
| `results/autockt_mixed_target_confirmation.jsonl` | PPO learning evidence (mixed-target) | Matched initial-vs-final checkpoint comparison, mixed target pool | success (satisfaction 0.333->1.0, reward 1.51->10.0) | 42 | Learning-evidence headline, mixed-target; per-update rewards 2.83/5.33/2.33/4.96/4.67/2.96 |
| `results/autockt_mixed_target_confirmation_policy.pt` | PPO learning evidence (mixed-target) | Saved FINAL policy checkpoint from the above run | the only .pt checkpoint in this repository | 42 | Used by the current pipeline/UI as the demonstrated trained policy. No separate 'initial checkpoint' .pt file exists -- see PHASE2 index note |
| `results/autockt_mixed_target_confirmation_rerun.jsonl` | PPO learning evidence (mixed-target, rerun) | A rerun of the mixed-target confirmation | NOT RECORDED | 42 | Supporting/reproducibility data for the mixed-target result |
| `results/autockt_normfix_confirmation.jsonl` | PPO learning evidence (single-target) | Matched initial-vs-final checkpoint comparison | success (satisfaction 0.667->1.0, reward 5.84->10.0) | 42 | Learning-evidence headline, single-target case |
| `results/autockt_randominit_smoke.jsonl` | Early development / smoke test | Preserved as-is; referenced in docs/autockt-mapping.md's early sections | NOT RECORDED | NOT RECORDED | Development history, not a headline reported result |
| `results/autockt_smoke_test.jsonl` | Early development / smoke test | Preserved as-is; referenced in docs/autockt-mapping.md's early sections | NOT RECORDED | NOT RECORDED | Development history, not a headline reported result |
| `results/autockt_synthetic_ppo_run1.jsonl` | Early development / smoke test | Preserved as-is; referenced in docs/autockt-mapping.md's early sections | NOT RECORDED | NOT RECORDED | Development history, not a headline reported result |
| `results/baseline_comparison_summary.json` | RS/CEM baseline comparison | Early baseline-comparison summary | NOT RECORDED | NOT RECORDED | Supporting comparison data |
| `results/benchmark_report.json` | Consolidated RS/CEM/PPO benchmark report | Recomputes nothing new; consolidates the fair-benchmark + warm-started trials | see file for both trials' results | 123 | docs sec 23 Task 5 deliverable |
| `results/cem_baseline_3x10.jsonl` | CEM (early, warm-started) | Original warm-started CEM run | INCOMPLETE (13/30 planned evaluations), warm-started | NOT RECORDED | Explicitly excluded from fair-benchmark comparisons; kept for reproducibility/history only |
| `results/cem_graded_unbiased_seed123.jsonl` | CEM (graded-fitness fix, unbiased init) | Same config as cem_unbiased_seed123 with the graded-fitness elite-selection fix | 0/20 (fix verified correct in isolation; never exercised here -- no candidate reached a gradient-bearing stage) | 123 | Demonstrates the fix works without changing the fair-benchmark result |
| `results/cem_near_good.jsonl` | CEM (early development) | Early/exploratory CEM run | NOT RECORDED | NOT RECORDED | Development history, not a reported benchmark |
| `results/cem_near_good_v2.jsonl` | CEM (early development) | Early/exploratory CEM run | NOT RECORDED | NOT RECORDED | Development history, not a reported benchmark |
| `results/cem_smoke.jsonl` | CEM (smoke test) | CEM smoke test | NOT RECORDED | NOT RECORDED | Development history, not a reported benchmark |
| `results/cem_unbiased_seed123.jsonl` | CEM (fair benchmark, unbiased init) | 4x5, elite=2, mean=0, std=0.577 | 0/20 (no feasible candidate) | 123 | Fair-benchmark headline: 0/20 |
| `results/controlled_unseen_target_generalization.jsonl` | PPO unseen-target generalization | Controlled matched-checkpoint eval on a held-out target | regression (1 win / 6 ties / 3 losses) | NOT RECORDED | Generalization is NOT claimed as proven -- this is the evidence why |
| `results/ctle_search_20.jsonl` | Early development / smoke test | Preserved as-is; referenced in docs/autockt-mapping.md's early sections | NOT RECORDED | NOT RECORDED | Development history, not a headline reported result |
| `results/design_a_final_specification.json` | Design A — final specification | Authoritative PASS/FAIL/NOT CLAIMED report for Design A | all measurable specs PASS; area NOT CLAIMED | NOT RECORDED | The single most-verified design's full report |
| `results/design_a_pvt_failure_diagnosis.jsonl` | PVT — diagnosis | Full per-stage re-evaluation of the 4 originally-failing conditions | 4/4 PASS on diagnosis | NOT RECORDED | First evidence the 4 failures were non-reproducible, not a design defect |
| `results/design_a_pvt_failure_reproduction_check.jsonl` | PVT — targeted rerun | Same 4 conditions, identical code path as the original sweep | 4/4 PASS | NOT RECORDED | Confirms the diagnosis result via the exact original evaluate_pvt_grid path |
| `results/design_a_pvt_minimal27.jsonl` | PVT — original run | Original 27-point minimal PVT sweep of Design A | 23/27 PASS -- PRESERVED, NOT overwritten | NOT RECORDED | Historical record; motivated the diagnosis/rerun sequence below |
| `results/design_a_pvt_minimal27_rerun.jsonl` | PVT — final full rerun | Complete 27-point rerun, Design A unchanged | 27/27 PASS (138.4 min wall-clock) | NOT RECORDED | Current, most-verified PVT characterization of Design A |
| `results/feasible_design_catalog.jsonl` | Feasible-design catalog | 8 unique feasible designs across RS/CEM/PPO, uniformly rescored | 8 feasible designs | NOT RECORDED | Cross-method catalog, docs sec 22 Task 3 |
| `results/hd3_noise_validation_CRASH_LOG.txt` | Real-SPICE SIGSEGV (crash 3 of 3) | Standalone HD3/noise validation of Design A (no PPO, no PVT) | FAILED -- exit 139 (SIGSEGV); partial faulthandler traceback captured (numpy/linalg/_linalg.py) | NOT RECORDED | The crash that identified the likely root-cause mechanism -- see docs/FINAL_TECHNICAL_AUDIT.md sec 4.E |
| `results/pipeline_real_spice_smoke_test_CRASH_LOG.txt` | Real-SPICE SIGSEGV (crash 1 of 3) | Full pipeline CLI run, candidate-gen + PVT smoke combined | FAILED -- exit 139 (SIGSEGV) | 42 | First crash; exact command recorded in the file itself |
| `results/pipeline_real_spice_smoke_test_CRASH_LOG_2.txt` | Real-SPICE SIGSEGV (crash 2 of 3) | Retry of the identical run configuration | FAILED -- exit 139 (SIGSEGV), 5:35.50 wall-clock before crash | 42 | Second crash, same config; exact command recorded in the file itself |
| `results/random_search.jsonl` | Early development / smoke test | Preserved as-is; referenced in docs/autockt-mapping.md's early sections | NOT RECORDED | NOT RECORDED | Development history, not a headline reported result |
| `results/rc_counterfactual_sweep_mixed_target.jsonl` | R/C counterfactual sweep | 18-point sweep perturbing R/C around a trained policy's choices | NOT RECORDED | NOT RECORDED | Parameter-sensitivity supporting evidence, docs sec 17 |
| `results/receiver_random_search.jsonl` | Early development / smoke test | Preserved as-is; referenced in docs/autockt-mapping.md's early sections | NOT RECORDED | NOT RECORDED | Development history, not a headline reported result |
| `results/receiver_random_search.jsonl.manifest.json` | Early development / smoke test | Preserved as-is; referenced in docs/autockt-mapping.md's early sections | NOT RECORDED | NOT RECORDED | Development history, not a headline reported result |
| `results/receiver_random_search.jsonl.summary.json` | Early development / smoke test | Preserved as-is; referenced in docs/autockt-mapping.md's early sections | NOT RECORDED | NOT RECORDED | Development history, not a headline reported result |
| `results/receiver_random_search_20_seed123.jsonl` | Random Search (fair benchmark) | The 20-candidate uniform random search run | success (1/20 candidates feasible) | 123 | Fair-benchmark headline: 1/20 (5%), 9 evaluations to first success |
| `results/receiver_random_search_20_seed123.jsonl.manifest.json` | Random Search (fair benchmark) | Run manifest: env/version/checksum/config identity | metadata | 123 | Identity record for the above run |
| `results/receiver_random_search_20_seed123.jsonl.summary.json` | Random Search (fair benchmark) | Run summary statistics | metadata | 123 | Summary for the above run |
| `results/receiver_search.jsonl` | Early development / smoke test | Preserved as-is; referenced in docs/autockt-mapping.md's early sections | NOT RECORDED | NOT RECORDED | Development history, not a headline reported result |
| `results/receiver_search_1.jsonl` | Early development / smoke test | Preserved as-is; referenced in docs/autockt-mapping.md's early sections | NOT RECORDED | NOT RECORDED | Development history, not a headline reported result |
| `results/receiver_search_10.jsonl` | Early development / smoke test | Preserved as-is; referenced in docs/autockt-mapping.md's early sections | NOT RECORDED | NOT RECORDED | Development history, not a headline reported result |
| `results/receiver_search_3.jsonl` | Early development / smoke test | Preserved as-is; referenced in docs/autockt-mapping.md's early sections | NOT RECORDED | NOT RECORDED | Development history, not a headline reported result |
| `results/rl_5x5.jsonl` | Early development / smoke test | Preserved as-is; referenced in docs/autockt-mapping.md's early sections | NOT RECORDED | NOT RECORDED | Development history, not a headline reported result |
| `results/rl_reward_directed_smoke.jsonl` | Early development / smoke test | Preserved as-is; referenced in docs/autockt-mapping.md's early sections | NOT RECORDED | NOT RECORDED | Development history, not a headline reported result |
| `results/rl_smoke.jsonl` | Early development / smoke test | Preserved as-is; referenced in docs/autockt-mapping.md's early sections | NOT RECORDED | NOT RECORDED | Development history, not a headline reported result |
| `results/schematics/design_a_final_schematic.spice` | Design A schematic export | Exported real SPICE netlist for Design A | generated from a real successful evaluation | NOT RECORDED | Real, syntactically valid netlist, not a parameter list |
| `results/schematics/design_a_final_schematic_v2_with_area.spice` | Design A schematic export (v2) | Same design, header extended with the area-measurement note | generated from a real successful evaluation | NOT RECORDED | Supersedes v1's header only; same circuit |
| `results/test_search.jsonl` | Early development / smoke test | Preserved as-is; referenced in docs/autockt-mapping.md's early sections | NOT RECORDED | NOT RECORDED | Development history, not a headline reported result |
| `results/web_ui_runs/ed91dd5617fa4c99b9539931c1c25ed6.json` | Successful real-SPICE UI run | The pipeline's own raw output JSON for the reported UI run | success -- 1 nominally feasible candidate, all measured specs PASS | 42 | run_id ed91dd5617fa4c99b9539931c1c25ed6, nominal-only PVT selection |
| `results/web_ui_runs/ed91dd5617fa4c99b9539931c1c25ed6_schematic.spice` | Successful real-SPICE UI run | Exported schematic for the above run's selected design | success | 42 | Companion file to the JSON above |

## SPICE circuit configuration (`circuits/`)

| Path | Origin | Note |
|---|---|---|
| `circuits/benches/ctle_ac.cir` | original file, copied byte-for-byte (verified via SHA-256) | Defines the circuit every experiment in results/ was evaluated against |
| `circuits/benches/ctle_dc.cir` | original file, copied byte-for-byte (verified via SHA-256) | Defines the circuit every experiment in results/ was evaluated against |
| `circuits/benches/ctle_hd3.cir` | original file, copied byte-for-byte (verified via SHA-256) | Defines the circuit every experiment in results/ was evaluated against |
| `circuits/benches/ctle_noise.cir` | original file, copied byte-for-byte (verified via SHA-256) | Defines the circuit every experiment in results/ was evaluated against |
| `circuits/benches/ctle_transient.cir` | original file, copied byte-for-byte (verified via SHA-256) | Defines the circuit every experiment in results/ was evaluated against |
| `circuits/benches/receiver_transient.cir` | original file, copied byte-for-byte (verified via SHA-256) | Defines the circuit every experiment in results/ was evaluated against |
| `circuits/blocks/ctle.spice` | original file, copied byte-for-byte (verified via SHA-256) | Defines the circuit every experiment in results/ was evaluated against |
| `circuits/ctle_1k_op.cir` | original file, copied byte-for-byte (verified via SHA-256) | Defines the circuit every experiment in results/ was evaluated against |
| `circuits/ctle_automation.cir` | original file, copied byte-for-byte (verified via SHA-256) | Defines the circuit every experiment in results/ was evaluated against |
| `circuits/ctle_baseline.cir` | original file, copied byte-for-byte (verified via SHA-256) | Defines the circuit every experiment in results/ was evaluated against |
| `circuits/ctle_op.cir` | original file, copied byte-for-byte (verified via SHA-256) | Defines the circuit every experiment in results/ was evaluated against |
| `circuits/ctle_start.cir` | original file, copied byte-for-byte (verified via SHA-256) | Defines the circuit every experiment in results/ was evaluated against |
| `circuits/sky130_nmos_test.cir` | original file, copied byte-for-byte (verified via SHA-256) | Defines the circuit every experiment in results/ was evaluated against |
| `circuits/test/divider.cir` | original file, copied byte-for-byte (verified via SHA-256) | Defines the circuit every experiment in results/ was evaluated against |
| `circuits/test/wrdata.cir` | original file, copied byte-for-byte (verified via SHA-256) | Defines the circuit every experiment in results/ was evaluated against |

## Channel input data (`channels/`)

| Path | Origin | Note |
|---|---|---|
| `channels/synthetic_regression.json` | original file, copied byte-for-byte (verified via SHA-256) | Referenced by channel_checksum in run manifests |
| `channels/synthetic_regression.s4p` | original file, copied byte-for-byte (verified via SHA-256) | Referenced by channel_checksum in run manifests |

## Environment configuration

| Path | Origin | Note |
|---|---|---|
| `requirements.txt` | original file, copied byte-for-byte (verified via SHA-256) | See PHASE2 index environment section for what it does/doesn't cover |

## Documentation (`docs/`)

| Path | Origin | Note |
|---|---|---|
| `docs/FINAL_TECHNICAL_AUDIT.md` | original file, copied byte-for-byte (verified via SHA-256) | Narrative record of experiments, methodology, and conclusions |
| `docs/autockt-mapping.md` | original file, copied byte-for-byte (verified via SHA-256) | Narrative record of experiments, methodology, and conclusions |
| `docs/baseline-comparison.md` | original file, copied byte-for-byte (verified via SHA-256) | Narrative record of experiments, methodology, and conclusions |
| `docs/stage1-decisions.md` | original file, copied byte-for-byte (verified via SHA-256) | Narrative record of experiments, methodology, and conclusions |

## Complete current system source (pipeline + UI + dependencies)

| Path | Origin | Note |
|---|---|---|
| `analysis/__init__.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `analysis/area_estimate.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `analysis/baseline_comparison.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `analysis/benchmark_report.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `analysis/design_catalog.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `analysis/fair_comparison.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `analysis/final_specification.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `analysis/learning_evidence.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `analysis/policy_inspection.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `analysis/pvt_selection.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `experiments/__init__.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `experiments/export_final_schematic.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `experiments/pvt_diagnose.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `experiments/pvt_sweep.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `experiments/run_autockt_pipeline.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `experiments/train_autockt.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `experiments/web_ui.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `rl/__init__.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `rl/autockt_action.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `rl/autockt_env.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `rl/autockt_reward.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `rl/autockt_state.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `rl/parameter_grid.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `rl/ppo_agent.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `rl/synthetic_benchmark.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `rl/target_spec.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `rl/trainer.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `simulator/__init__.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `simulator/cache.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `simulator/channel.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `simulator/config.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `simulator/ctle.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `simulator/metrics.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `simulator/models.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `simulator/ngspice.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `simulator/parser.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `simulator/provenance.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `simulator/receiver.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `simulator/receiver_metrics.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `simulator/resources/sky130.spiceinit` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `simulator/rl_adapter.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `simulator/stimulus.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |
| `simulator/waveform.py` | original file, copied byte-for-byte (verified via SHA-256) | Complete system source, per Person B's request |

## Historical experiment source scripts

| Path | Origin | Note |
|---|---|---|
| `experiments/controlled_unseen_target_eval.py` | original file, copied byte-for-byte (verified via SHA-256) | Not imported by the current pipeline; preserved for reproducibility |
| `experiments/rc_counterfactual_sweep.py` | original file, copied byte-for-byte (verified via SHA-256) | Not imported by the current pipeline; preserved for reproducibility |
| `experiments/receiver_search.py` | original file, copied byte-for-byte (verified via SHA-256) | Not imported by the current pipeline; preserved for reproducibility |
| `experiments/train_cem.py` | original file, copied byte-for-byte (verified via SHA-256) | Not imported by the current pipeline; preserved for reproducibility |

## Regression tests

| Path | Origin | Note |
|---|---|---|
| `tests/__init__.py` | original file, copied byte-for-byte (verified via SHA-256) | Supports understanding/reproducing the corresponding module |
| `tests/run_fast_suite.py` | original file, copied byte-for-byte (verified via SHA-256) | Supports understanding/reproducing the corresponding module |
| `tests/test_area_estimate.py` | original file, copied byte-for-byte (verified via SHA-256) | Supports understanding/reproducing the corresponding module |
| `tests/test_autockt_rl.py` | original file, copied byte-for-byte (verified via SHA-256) | Supports understanding/reproducing the corresponding module |
| `tests/test_baseline_comparison.py` | original file, copied byte-for-byte (verified via SHA-256) | Supports understanding/reproducing the corresponding module |
| `tests/test_benchmark_report.py` | original file, copied byte-for-byte (verified via SHA-256) | Supports understanding/reproducing the corresponding module |
| `tests/test_controlled_unseen_target_eval.py` | original file, copied byte-for-byte (verified via SHA-256) | Supports understanding/reproducing the corresponding module |
| `tests/test_design_catalog.py` | original file, copied byte-for-byte (verified via SHA-256) | Supports understanding/reproducing the corresponding module |
| `tests/test_evaluate_pvt_grid.py` | original file, copied byte-for-byte (verified via SHA-256) | Supports understanding/reproducing the corresponding module |
| `tests/test_export_final_schematic.py` | original file, copied byte-for-byte (verified via SHA-256) | Supports understanding/reproducing the corresponding module |
| `tests/test_fair_comparison.py` | original file, copied byte-for-byte (verified via SHA-256) | Supports understanding/reproducing the corresponding module |
| `tests/test_final_specification.py` | original file, copied byte-for-byte (verified via SHA-256) | Supports understanding/reproducing the corresponding module |
| `tests/test_learning_evidence.py` | original file, copied byte-for-byte (verified via SHA-256) | Supports understanding/reproducing the corresponding module |
| `tests/test_policy_inspection.py` | original file, copied byte-for-byte (verified via SHA-256) | Supports understanding/reproducing the corresponding module |
| `tests/test_pvt_diagnose.py` | original file, copied byte-for-byte (verified via SHA-256) | Supports understanding/reproducing the corresponding module |
| `tests/test_pvt_selection.py` | original file, copied byte-for-byte (verified via SHA-256) | Supports understanding/reproducing the corresponding module |
| `tests/test_pvt_sweep.py` | original file, copied byte-for-byte (verified via SHA-256) | Supports understanding/reproducing the corresponding module |
| `tests/test_rc_counterfactual_sweep.py` | original file, copied byte-for-byte (verified via SHA-256) | Supports understanding/reproducing the corresponding module |
| `tests/test_run_autockt_pipeline.py` | original file, copied byte-for-byte (verified via SHA-256) | Supports understanding/reproducing the corresponding module |
| `tests/test_synthetic_benchmark.py` | original file, copied byte-for-byte (verified via SHA-256) | Supports understanding/reproducing the corresponding module |
| `tests/test_train_cem.py` | original file, copied byte-for-byte (verified via SHA-256) | Supports understanding/reproducing the corresponding module |
| `tests/test_web_ui.py` | original file, copied byte-for-byte (verified via SHA-256) | Supports understanding/reproducing the corresponding module |

## Notes on metadata marked NOT RECORDED

- **Seeds** are marked `NOT RECORDED` for files where no seed value is present in the file's own rows, its filename, or docs/autockt-mapping.md's description of that run.
- **No separate 'initial PPO checkpoint' file exists** in this repository -- only `results/autockt_mixed_target_confirmation_policy.pt` (the FINAL trained policy) was ever persisted via `--save-final-policy`. Matched initial-vs-final learning evidence was produced by evaluating the policy at two points within one training run (deterministic given `--seed`), not by saving two separate checkpoint files.
- Environment/version/PDK/ngspice identity: see `PHASE2_ENVIRONMENT.md`-equivalent section of the outer artifact index below.

## Environment, package, and identity information

Two distinct sources exist for this, and they are NOT assumed identical:

1. **Per-run manifests already recorded in the repository** (the most authoritative source for any specific experiment): e.g. `results/receiver_random_search_20_seed123.jsonl.manifest.json` records, for that exact run: `ngspice_version` (`ngspice-47`, KLU direct linear solver), `ngspice_executable` (`/opt/homebrew/Cellar/ngspice/47/bin/ngspice`), `python_version` (`3.13.13`, conda-forge, Clang 19.1.7), `numpy_version` (`2.5.2`), `sky130_model_checksum`, `sky130_model_root_checksum`, `sky130_model_dependency_count` (319), `channel_checksum`, `implementation_checksum`, `initialization_checksum`, `git_commit`, `working_tree_dirty`, and `seed`. Not every results/ file has this sidecar manifest -- most PVT/PPO/CEM run files instead carry lighter provenance inline in each row (`evaluation_id`, sometimes `git_commit`), and older files carry none. Check the specific file before assuming any global value applies.
2. **The active environment at artifact-packaging time** (this session, commit `4478e326afd2108697e7604c5896b83609ef9e23`), independently verified, NOT assumed identical to every historical run's own environment: `ngspice-47` (KLU direct linear solver, `/opt/homebrew/bin/ngspice`), Python `3.11.15` (conda-forge, Clang 19.1.7), numpy `2.4.6`, PyTorch `2.13.0`, macOS `26.5.1` (arm64). **`requirements.txt` pins only `numpy>=1.24`** -- PyTorch is a real, separate runtime dependency of `rl/ppo_agent.py` that is not captured in `requirements.txt`; no pinned torch version file exists in this repository, so reproducing the exact torch version used for any specific historical PPO run beyond what a manifest records is **NOT RECORDED**.
3. **SKY130 PDK identity**: `sky130_model_checksum` / `sky130_model_root_checksum` / `sky130_model_dependency_count` (from manifests, e.g. above) are the only repository-recorded PDK identity values. The PDK's own install path/version string (e.g. seen in exported schematic headers, `/Users/dwitisuchak/.ciel/ciel/sky130/versions/<hash>/...`) is recorded where the pipeline itself wrote it (`results/schematics/*.spice` headers) but was not independently re-verified against the PDK cache directly, which is outside this project's own access boundary per project convention.
4. **OS information**: only captured where a manifest or this session's own environment check recorded it (see point 2). No systematic OS capture exists across all historical runs -- **NOT RECORDED** for any run without its own manifest.

## Exact commands

| Experiment | Command | Source |
|---|---|---|
| PPO mixed-target learning evidence | `python -m experiments.train_autockt --updates 6 --episodes-per-update 6 --horizon 4 --target-mode mixed --grid-spacing log --randomize-initial-state --checkpoint-eval-episodes 3 --seed 42 --output results/autockt_mixed_target_confirmation.jsonl` | docs/autockt-mapping.md sec 17 |
| CEM, unbiased init (fair benchmark) | `python -m experiments.train_cem --iterations 4 --population 5 --elite 2 --seed 123 --init-mean 0 0 0 0 0 --init-std 0.5773502691896258 (x5) --output results/cem_unbiased_seed123.jsonl` | docs/autockt-mapping.md sec 19 |
| CEM, graded-fitness fix | `python -m experiments.train_cem --iterations 4 --population 5 --elite 2 --seed 123 --fitness graded --target trivial --init-mean 0 0 0 0 0 --init-std 0.5773502691896258 (x5) --output results/cem_graded_unbiased_seed123.jsonl` | docs/autockt-mapping.md sec 20 |
| PPO, fair no-warm-start head-to-head | `python -m experiments.train_autockt --backend real --target-mode trivial --grid-spacing log --randomize-initial-state --initial-indices-source grid-center --updates 4 --episodes-per-update 5 --horizon 1 --minibatch-size 5 --ppo-epochs 4 --seed 123 --max-evaluations 20 --output results/autockt_fair_headtohead_seed123.jsonl` | docs/autockt-mapping.md sec 20 |
| Real-SPICE pipeline smoke test (both crashes) | `python -m experiments.run_autockt_pipeline --target-mode trivial --backend real --checkpoint results/autockt_mixed_target_confirmation_policy.pt --episodes 3 --horizon 4 --agent-seed 42 --eval-seed 42 --initial-indices-source verified --pvt-condition-set smoke --output results/pipeline_real_spice_smoke_test.json --export-schematic results/schematics/pipeline_real_spice_smoke_test_schematic.spice` | results/pipeline_real_spice_smoke_test_CRASH_LOG.txt / \_2.txt (recorded verbatim in the shell's own crash message) |
| Random Search (fair benchmark, seed 123, n=20) | EXACT COMMAND NOT RECORDED | not present in docs/autockt-mapping.md or any manifest field |
| PVT original 27-point sweep | EXACT COMMAND NOT RECORDED | not present in docs/autockt-mapping.md |
| PVT failure diagnosis (4 conditions) | EXACT COMMAND NOT RECORDED | not present in docs/autockt-mapping.md |
| PVT failure reproduction check (4/4) | EXACT COMMAND NOT RECORDED | not present in docs/autockt-mapping.md |
| PVT final 27-point rerun | EXACT COMMAND NOT RECORDED | not present in docs/autockt-mapping.md |
| Successful real-SPICE UI run (ed91dd56...) | Observed via the UI's own `/api/status` response at run time: `python -m experiments.run_autockt_pipeline --backend real --episodes 3 --horizon 4 --initial-indices-source verified --pvt-condition-set none --trade-off-preference most_robust --output <run_id>.json --export-schematic <run_id>_schematic.spice --target-mode trivial --checkpoint results/autockt_mixed_target_confirmation_policy.pt` | observed in this session's tool output; not persisted to a repository file verbatim |

Representative/reconstructed commands (labeled as such, never claimed exact) for the PVT runs above can be built from `experiments/pvt_sweep.py --condition-set minimal27 --output <path>` and `experiments/pvt_diagnose.py`'s own CLI, per their argparse definitions in this package -- but no specific invocation for those historical runs was ever recorded, so none is presented as exact.

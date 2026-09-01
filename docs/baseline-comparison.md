# Random Search vs. CEM baseline comparison

Generated from existing results/*.jsonl files only. No new SPICE evaluation was run to produce this report.

## Random search runs

| file | n | successes | success rate | best reward | evals to first success | wall-clock |
|---|---:|---:|---:|---:|---:|---|
| results/receiver_random_search_20_seed123.jsonl | 20 | 1 | 5.0% | 100.000 | 9 | not recorded |

## CEM runs

| file | n | successes | success rate | best reward | evals to first success | wall-clock |
|---|---:|---:|---:|---:|---:|---|
| results/cem_baseline_3x10.jsonl | 13 | 8 | 61.5% | 100.000 | 1 | not recorded |
| results/cem_near_good_v2.jsonl | 5 | 3 | 60.0% | 100.000 | 1 | not recorded |
| results/cem_near_good.jsonl | 10 | 0 | 0.0% | -100.000 | n/a | not recorded |
| results/cem_smoke.jsonl | 3 | 0 | 0.0% | -100.000 | n/a | not recorded |

## Caveats per run

- **results/receiver_random_search_20_seed123.jsonl**:
  - documented seed=123, uniform sampling policy, n=20, full-receiver schema with per-candidate metrics; the only random-search run with a manifest.json and summary.json on disk
- **results/cem_baseline_3x10.jsonl**:
  - configured for 3 iterations x population 10 = 30 planned evaluations; file contains 13 rows (iterations 1-2 plus one row of iteration 3) -- this run appears interrupted/incomplete, not a completed 30-eval trial
  - experiments/train_cem.py seeds its initial sampling distribution at a fixed mean=[0.3697, 0.2767, 0.3333, 0.7802, -0.0277] (normalized action space) with std=0.15 per dimension -- a Gaussian localized in a specific, already-decent region, not the uniform(-1, 1)^5 coverage receiver_search.py's 'uniform' sampling_policy uses. Some or all of CEM's higher raw success rate below reflects this warm-start neighborhood advantage, not only CEM's population/elite refinement mechanism -- the two runs are not a controlled comparison of search *algorithm* alone.
- **results/cem_near_good_v2.jsonl**:
  - rows 1-5 are byte-identical to the first 5 rows of results/cem_baseline_3x10.jsonl (same seed/mean/std init) -- an earlier checkpoint of the same run, not an independent trial; kept for completeness but should not be double-counted alongside cem_baseline_3x10.jsonl in an aggregate
- **results/cem_near_good.jsonl**:
  - byte-identical to results/cem_smoke.jsonl -- same run logged twice
- **results/cem_smoke.jsonl**:
  - byte-identical to results/cem_near_good.jsonl -- same run logged twice

## Excluded (out of scope)

- `results/random_search.jsonl` -- CTLE-block-only sweep (uppercase RLOAD/RDEG/CDEG/ITAIL_VAL keys, gain_*_db metrics) -- pre-receiver-integration circuit scope, not comparable to full-receiver random search
- `results/ctle_search_20.jsonl` -- CTLE-block-only sweep, same schema/scope note as random_search.jsonl
- `results/test_search.jsonl` -- CTLE-block-only sweep fixture (identical schema to ctle_search_20.jsonl)
- `results/receiver_search.jsonl` -- 1-row raw-ReceiverEvaluation-dump fixture (no candidate_index/action/reward fields) -- superseded by experiments/receiver_search.py's current row schema
- `results/receiver_search_1.jsonl` -- 1-row raw-ReceiverEvaluation-dump fixture, same note as receiver_search.jsonl
- `results/receiver_search_3.jsonl` -- 3-row raw-ReceiverEvaluation-dump fixture, no documented seed/manifest
- `results/receiver_search_10.jsonl` -- 10-row raw-ReceiverEvaluation-dump fixture, no documented seed/manifest

## Not yet measured

- wall-clock/design time for any run recorded before this session's timing instrumentation (train_cem.py, receiver_search.py) -- no existing row of receiver_random_search_20_seed123.jsonl or cem_baseline_3x10.jsonl carries a timestamp or duration
- a completed (30-evaluation) CEM run at the same n as the seed=123 random search, for a like-for-like success-rate comparison
- PPO/AutoCkt results in this same normalized comparison framework -- deferred until the running real-SPICE PPO confirmation experiment finishes and its result file is finalized
- per-metric (eye height / eye width / margin / power) design-quality comparison for CEM's successful candidates -- CEM's historical rows never logged simulator metrics, only reward_v1's scalar reward
- a CEM run initialized with the same uniform(-1, 1)^5 coverage random search uses (rather than train_cem.py's fixed near-good Gaussian mean) -- needed to isolate the search-algorithm effect from the initialization effect, see initialization_asymmetry_caveat above

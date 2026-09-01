# AutoCkt → NEBULA mapping: design document

This document records how the first AutoCkt-methodology RL implementation
(`rl/*.py`, `experiments/train_autockt.py`) was derived from the official
AutoCkt source and mapped onto NEBULA's existing, unmodified receiver
simulator. Every design choice below is labeled:

- **[AUTOCKT-REPLICATED]** — verified directly from AutoCkt source and
  reproduced with the same mechanics.
- **[NEBULA ADAPTATION]** — has no AutoCkt equivalent, or deliberately
  deviates from it, because NEBULA's receiver/simulator differs from
  AutoCkt's op-amp environment.
- **[ROUGH-SCALE ADAPTATION]** — same mechanics as AutoCkt, but scaled down
  (grid resolution, horizon, batch size) for the first 10-day milestone.
- **[UNVERIFIED-FROM-SOURCE]** — AutoCkt's own repository does not pin this
  value; a standard PPO-literature default is used instead and is not
  claimed as replication.

Source: `github.com/ksettaluri6/AutoCkt` (the official repo — note
`github.com/lwasoo/AutoCkt`, sometimes cited, is an independently modified
fork with different failed-simulation handling and is **not** used as a
source for any claim below).

## 1. Repository layout

```
rl/
    __init__.py
    parameter_grid.py    # discrete grid + AutoCkt action-delta mechanics
    target_spec.py        # NEBULA target-spec abstraction (no AutoCkt equivalent)
    autockt_state.py      # AutoCkt state vector (lookup-based normalization)
    autockt_action.py     # 5x Discrete(3) decode -> grid index -> simulator action
    autockt_reward.py     # AutoCkt reward mechanics, separate from reward_v1
    autockt_env.py        # ML-side environment wrapper around ReceiverRLAdapter
    ppo_agent.py           # standalone PyTorch PPO (policy/value nets, GAE, clip)
    trainer.py             # rollout collection + PPO update loop + logging
experiments/
    train_autockt.py       # CLI entry point
tests/
    test_autockt_rl.py     # fast, SPICE-free unit tests (31 tests)
```

`simulator/rl_adapter.py`, `simulator/receiver.py`, `simulator/ngspice.py`,
`simulator/channel.py`, and `simulator/cache.py` were **not modified**.
Every file above only imports read-only, already-public symbols from
`simulator/rl_adapter.py` (`ACTION_BOUNDS`, `METRIC_OBSERVATION_NAMES`,
`ReceiverRLAdapter`) and `simulator/receiver.py`
(`normalized_action_to_parameters`, indirectly). Existing baselines
(`experiments/train_rl.py`, `experiments/train_cem.py`,
`experiments/receiver_search.py`, `experiments/search.py`) and all existing
tests were not touched.

## 2. State [AUTOCKT-REPLICATED mechanics + NEBULA ADAPTATION content]

Verified from `autockt/envs/ngspice_vanilla_opamp.py`:

```python
self.ob = np.concatenate([cur_spec_norm, self.specs_ideal_norm, self.cur_params_idx])
# cur_spec_norm    = lookup(achieved_specs, goal_spec)
# specs_ideal_norm = lookup(goal_spec, global_g)          # global_g: a fixed reference vector
# cur_params_idx   = raw, UN-normalized discrete grid indices
# lookup(a, b) = (a - b) / (b + a)
```

NEBULA (`rl/autockt_state.py::build_state`): identical shape/order —
`spec_error_vector + target_reference_vector + param_indices`, dimension
`2*len(SPEC_NAMES) + len(PARAMETER_NAMES) = 2*4 + 5 = 13`
(`rl/autockt_state.py::STATE_DIM`).

**[NEBULA ADAPTATION]** `global_g`: AutoCkt's fixed reference vector has no
retrievable numeric definition beyond "a fixed reference" in the fetched
source. NEBULA reuses `EXISTING_THRESHOLDS` (see §4) as `GLOBAL_REFERENCE`
rather than inventing a separate number — one disclosed source of truth
instead of two.

**[NEBULA ADAPTATION]** `LOOKUP_EPSILON` denominator guard
(`rl/autockt_state.py::lookup`, `1e-9`): AutoCkt's four specs (gain,
bandwidth, phase margin, bias current) are always well clear of zero, so
`goal + spec` is never near-zero in the official env. NEBULA's
`dfe_min_margin_v` can sit arbitrarily close to zero (the repo's own golden
boundary test in `tests/test_rl_readiness.py` uses `dfe_min_margin_v: 1e-12`
as a passing-boundary value). AutoCkt's source has no such guard because it
never needed one.

## 3. Action [AUTOCKT-REPLICATED]

Verified from `autockt/envs/ngspice_vanilla_opamp.py`:

```python
self.action_meaning = [-1, 0, 2]
self.action_space = spaces.Tuple([spaces.Discrete(3)] * len(params_id))
self.cur_params_idx = np.clip(self.cur_params_idx + [action_meaning[a] for a in action], 0, len - 1)
```

This directly resolves the brief's flagged uncertainty: the deltas are
**{-1, 0, +2}**, not the more "intuitive" {-1, 0, +1}. Reproduced verbatim
in `rl/parameter_grid.py::ACTION_DELTAS` and applied in
`rl/autockt_action.py::apply_action`.

**[NEBULA ADAPTATION — count only]** AutoCkt uses 7 parameter heads (6
transistor multipliers + 1 cap); NEBULA uses 5 (`RLOAD, RDEG, CDEG, ITAIL,
DFE_TAP`), matching `simulator.rl_adapter.ACTION_BOUNDS`. The delta
mechanics themselves are unchanged.

`rl/autockt_action.py::indices_to_normalized_action` is ML-side glue, not
part of the AutoCkt replication: it converts a grid index to a physical
value, then inverts `simulator.rl_adapter.normalized_action_to_parameters`'s
own forward formula to produce the `[-1, 1]` action the untouched
`ReceiverRLAdapter.step()` requires.

## 4. Parameter grid

**[AUTOCKT-REPLICATED mechanics]** Verified from
`eval_engines/ngspice/ngspice_inputs/yaml_files/two_stage_opamp.yaml` and
`ngspice_vanilla_opamp.py`:

```python
param_vec = np.arange(value[0], value[1], value[2])   # LINEAR spacing, fixed step
```

`rl/parameter_grid.py::build_parameter_grids` uses the identical
`np.arange(lower, upper, step)` construction. Bounds are read verbatim from
`simulator.rl_adapter.ACTION_BOUNDS` — not duplicated.

**By explicit instruction, this first implementation uses LINEAR spacing for
all five parameters**, even though `ACTION_BOUNDS` marks `RLOAD`, `RDEG`,
`CDEG`, and `ITAIL` as `scale="log"` for the simulator's own *continuous*
action encoding. That `scale` flag is read only to source (lower, upper)
bounds here; it is otherwise ignored. If CDEG's resulting resolution proves
too coarse (see observed instance below), a log-spaced grid variant will be
added later as an explicitly labeled **[NEBULA ADAPTATION]**, not by
silently changing this file.

**[ROUGH-SCALE ADAPTATION]** `DEFAULT_GRID_POINTS = 21`. AutoCkt hand-picks
~99 points per parameter; NEBULA has no equivalent hand-tuned step, and 21
points keeps grid construction trivial for the first milestone. This is not
a claim of AutoCkt's own resolution.

*Observed CDEG resolution issue* (exactly the risk flagged before
implementation): with 21 points linearly spaced from 10fF to ~9.52pF (step
≈ 4.76e-13 F), the verified real design's `cdeg_f ≈ 1.0e-12` F snaps to the
nearest grid point at `≈ 9.6e-13` F — acceptably close here, but any design
point below roughly one grid step (≈475fF) from the origin has very coarse
representation. Confirmed by direct construction/round-trip test in
`tests/test_autockt_rl.py::ParameterGridTests`.

**[NEBULA ADAPTATION]** initial parameter-grid state
(`rl/parameter_grid.py::VERIFIED_INITIAL_PARAMETERS`,
`verified_initial_indices`). AutoCkt's own `reset()` uses a fixed,
hardcoded index array (`[33,33,33,33,33,14,20]`) — a fixed, non-random
starting point is the **replicated methodology**; the literal indices
cannot port since NEBULA's parameters differ.

The README's documented "baseline circuit" (RLOAD=1kΩ, RDEG=1kΩ, CDEG=0.5pF,
ITAIL=100µA) is also the literal default of
`simulator.receiver.ReceiverParameters()` (`receiver.py:54-58`) — so it does
exist directly in code, not only in prose. **It was nonetheless directly
measured in this session and found infeasible**: `ReceiverRLAdapter.step()`
at `TRAINING` fidelity on this exact point fails at the DC stage (does not
clear the DC rail-headroom/offset gate), wall-clock 15.17s. Per instruction,
documented defaults must not be assumed feasible without checking — this one
wasn't. Instead, `VERIFIED_INITIAL_PARAMETERS` uses a real, feasible design
already on record in
`results/receiver_random_search_20_seed123.jsonl` (`candidate_index 8`,
`success=True`, `reward_v1=100.0`):

| parameter | value |
|---|---|
| `rload_ohm` | 2342.472156058411 |
| `rdeg_ohm` | 822.3558626926603 |
| `cdeg_f` | 9.999375862792168e-13 |
| `itail_a` | 0.0006028331705063624 |
| `dfe_tap_v` | -0.011090823885148815 |

Re-confirmed by a fresh SPICE re-evaluation in this session (see §7 below).

## 5. Reward [AUTOCKT-REPLICATED mechanics, NEBULA-SPECIFIC spec choice]

Verified verbatim from `autockt/envs/ngspice_vanilla_opamp.py`:

```python
def reward(self, spec, goal_spec):
    rel_specs = self.lookup(spec, goal_spec)
    reward = 0.0
    for i, rel_spec in enumerate(rel_specs):
        if self.specs_id[i] == 'ibias_max':
            rel_spec = rel_spec * -1.0
        if rel_spec < 0:
            reward += rel_spec
    return reward if reward < -0.02 else 10
```

`rl/autockt_reward.py::autockt_reward` reproduces this exactly: per-spec
relative error via `lookup`, sign flip for the smaller-is-better spec, only
unsatisfied (negative) terms summed, flat terminal bonus `10.0` once the sum
is `>= -0.02` (`UNSATISFIED_THRESHOLD`, `TERMINAL_BONUS` — both AutoCkt's
literal constants), overshoot never penalized (confirmed by
`test_overshoot_is_not_penalized_beyond_satisfaction`).

**`simulator/rl_adapter.py::reward_v1` is never imported or called anywhere
in this reward module** (confirmed by
`test_reward_module_never_imports_reward_v1`).

**[NEBULA-SPECIFIC ADAPTATION]** spec selection
(`rl/target_spec.py::SPEC_NAMES`, `SPEC_DIRECTIONS`). AutoCkt's four specs
split 3 larger-is-better (`gain_min`, `ugbw_min`, `phm_min`) + 1
smaller-is-better (`ibias_max`). NEBULA's analogous 3+1 split, chosen from
the ten metrics already in `simulator.rl_adapter.METRIC_OBSERVATION_NAMES`:

| direction | metric |
|---|---|
| larger-is-better | `dfe_locked_phase_eye_height_v` |
| larger-is-better | `dfe_eye_width_ui` |
| larger-is-better | `dfe_min_margin_v` |
| smaller-is-better | `ctle_power_w` |

`peaking_db` is deliberately **excluded**: `simulator/receiver.py`'s own
AC-stage gate treats it as a banded target (3–12 dB required;
`reward_v1` scores it as optimal at exactly 6 dB), which is not a monotone
larger/smaller-is-better quantity and does not fit AutoCkt's purely-monotone
reward shape. This is a disclosed scope reduction, not an oversight.

**[NEBULA ADAPTATION]** `FAILURE_REWARD = -1.0`
(`rl/autockt_reward.py`). AutoCkt's official repo has **no verified graceful
handling of a failed SPICE simulation**:
`eval_engines/ngspice/ngspice_wrapper.py::NgSpiceWrapper.simulate()`'s
`info` failure flag is read but then discarded by `TwoStageAmp.update()`,
and `TwoStageClass.py::parse_output()` proceeds to `np.genfromtxt()` on the
(possibly-missing) result files regardless — i.e. a failed simulation would
propagate an exception rather than return a defined reward. (The unofficial
`lwasoo/AutoCkt` fork *adds* handling for this; the official repo does not —
confirmed by diffing the two.) NEBULA's simulator instead returns a
structured `ReceiverEvaluation(success=False, failed_stage=...)`, so this
reward defines an explicit failure penalty instead of crashing.
`FAILURE_REWARD = -1.0` is chosen to sit *inside* the unsatisfied branch's
natural range (bounded below by `-len(SPEC_NAMES) = -4`), unlike
`reward_v1`'s `-100.0`, which is deliberately far outside its own success
range to guarantee "failures cannot beat successes." This reward makes no
such guarantee — AutoCkt's own reward has no equivalent design goal to
replicate, so none was invented.

## 6. Trajectory / episode

Verified from `autockt/envs/ngspice_vanilla_opamp.py` and
`autockt/val_autobag_ray.py`:

- **[AUTOCKT-REPLICATED]** fixed (not randomized) initial parameter-grid
  state per episode.
- **[AUTOCKT-REPLICATED]** one target spec per episode, sampled from a pool.
- **[AUTOCKT-REPLICATED]** `done=True` exactly when the reward hits the
  terminal bonus (`if reward >= 10: done = True`) — never from step count
  inside the environment itself.
- **Horizon = 30 is CONFIRMED** as the Ray RLlib training-time default
  (`"horizon": 30` in `config_train`) — but this is enforced by the *outer
  RLlib trainer*, not the environment's own `step()`. Separately,
  `autockt/rollout.py` defaults `--traj_len` to **60** for
  validation/rollout — a different number for a different phase; the two
  are not interchangeable.

**[ROUGH-SCALE ADAPTATION]** `rl/autockt_env.py::AutoCktReceiverEnv` mirrors
the *split* (an AutoCkt-replicated `done` inside the env; a separate
`truncated` flag for horizon, computed the way NEBULA's own
`simulator.rl_adapter.RLStep` already splits `terminated`/`truncated`) but
uses a much smaller default horizon (6, via
`experiments/train_autockt.py --horizon`) than AutoCkt's 30, because NEBULA
SPICE evaluations cost ~15–65s each (see §7) versus AutoCkt's fast op-amp
SPICE. This is a disclosed scale reduction for the first milestone, not a
claim of matching AutoCkt's horizon.

**[NEBULA ADAPTATION]** failed-simulation handling: unlike AutoCkt (no
verified graceful handling), a failed NEBULA evaluation does not end the
episode (`done=False`) — it returns `FAILURE_REWARD` and the episode
continues, taking advantage of infrastructure AutoCkt's own repo does not
have.

## 7. SPICE timing measurement

Measured directly in this session via `ReceiverRLAdapter.step()` at
`TRAINING` fidelity, before writing any RL code, to calibrate horizon/batch
size. All 8 measurements went through the real, unmodified adapter (no
caching enabled, so every row is a genuine fresh `ngspice` run):

| label | evaluation # | reward (`reward_v1`) | success | failure_stage | wall-clock (s) |
|---|---:|---:|---|---|---:|
| baseline_default (README/dataclass defaults) | 1 | -100.0 | False | `dc` | 15.168 |
| known_good (search result, run 1) | 1 | 100.0 | True | — | 63.146 |
| known_good_repeat_0 | 1 | 100.0 | True | — | 63.696 |
| known_good_repeat_1 | 2 | 100.0 | True | — | 63.542 |
| known_good_repeat_2 | 3 | 100.0 | True | — | 63.412 |
| known_good_perturbed_0 (±0.05 noise) | 4 | 100.0 | True | — | 62.641 |
| known_good_perturbed_1 (±0.05 noise) | 5 | 100.0 | True | — | 62.123 |
| known_good_perturbed_2 (±0.05 noise) | 6 | 100.0 | True | — | 62.316 |

**Mean over 7 successful (full-pipeline) evaluations: ≈ 62.98 s. One
DC-stage failure: 15.17 s.** (Other failure stages were not sampled and are
expected to fall somewhere between these two, since more stages ran before
failing.)

This is why `experiments/train_autockt.py`'s defaults are small
(`--updates 2 --episodes-per-update 2 --horizon 6`): a worst case of
`2 * 2 * 6 = 24` evaluations at ~60s each is ~24 minutes, appropriate for a
first-milestone verification run, not a converged policy. Also why AutoCkt's
own `train_batch_size=1200` and `horizon=30` are not attempted as defaults —
that would be ~30x more evaluations per update than this milestone budgets
for.

## 8. PPO

Verified from `autockt/val_autobag_ray.py`:

```python
config_train = {
    "train_batch_size": 1200,
    "horizon": 30,
    "num_gpus": 0,
    "model": {"fcnet_hiddens": [64, 64]},
    "num_workers": 6,
    ...
}
# run: "PPO" via ray.rllib.agents.ppo, environment.yml pins ray==0.6.3, gym==0.10.5, tensorflow==1.10.1
```

- **[AUTOCKT-REPLICATED]** network width/depth: `[64, 64]`
  (`rl/ppo_agent.py::HIDDEN_SIZES`) — this **refutes** the commonly-assumed
  "3 hidden layers × 50 neurons" figure; that figure does not appear in the
  official training script and is not used here.
- **[AUTOCKT-REPLICATED]** action head structure: independent categorical
  heads of 3 logits each (5 in NEBULA, 7 in AutoCkt).
- **[UNVERIFIED-FROM-SOURCE]** everything else: `val_autobag_ray.py`'s
  `config_train` has learning rate, `vf_loss_coeff`, `sgd_minibatch_size`,
  and `num_sgd_iter` present only as **commented-out** lines, and gamma, GAE
  λ, clip ε, entropy coefficient, and activation function are **absent
  entirely** — they fell back to unstated Ray RLlib 0.6.3 internals not
  reproduced here. `rl/ppo_agent.py` uses ordinary PPO-literature defaults
  (γ=0.99, λ=0.95, clip=0.2, entropy=0.01, value-coef=0.5, lr=3e-4, Tanh
  activation, separate policy/value trunks) — explicitly not a replication
  claim.
- **[ROUGH-SCALE ADAPTATION]** batch size / episodes-per-update / PPO
  epochs: see §7 — AutoCkt's `train_batch_size=1200` is not attempted;
  `rl/trainer.py::DEFAULT_EPISODES_PER_UPDATE = 2`,
  `DEFAULT_PPO_EPOCHS = 4`, `DEFAULT_MINIBATCH_SIZE = 16` are sized for a
  first milestone, not paper-scale training. `num_workers=6` (parallel
  RLlib rollout workers) is not replicated — this implementation is
  single-process for the first milestone.

## 9. Training specifications

Verified from `autockt/gen_specs.py` and the official README:

```python
list_val = [random.uniform(float(spec[0]), float(spec[1])) for x in range(num_specs)]
```

- **[AUTOCKT-REPLICATED sampling methodology]**
  `rl/target_spec.py::sample_target_pool` — uniform per-dimension sampling
  into a fixed pool, same as `gen_specs.py`.
- **[NEBULA ADAPTATION]** the `TargetSpec` dataclass itself has no AutoCkt
  equivalent (AutoCkt's targets are a YAML block + pickle file, not a Python
  object) — its existence and shape are a NEBULA addition.
- **No code-enforced train/validation split exists in AutoCkt** — both
  training and rollout draw from the same overwritable pickle file
  (`autockt/gen_specs/ngspice_specs_gen_two_stage_opamp`); separation is a
  manual workflow step per the official README. NEBULA replicates this
  train/val *methodology* but does enforce a disjoint sampling stream at the
  code level: `experiments/train_autockt.py::_build_target_pools` samples
  the validation pool with `seed + 1_000_000`, guaranteeing the two pools
  never collide for any reasonable `seed`.
- **"350 training specs" is a README-documentation claim in the official
  repo, not a hardcoded default** ("num_specs 350 was used (only 50 were
  selected for each CPU worker)"). Per explicit instruction, this milestone
  does not attempt paper-scale training; `--num-training-specs` defaults to
  `1` (the single, deliberately achievable target — see §10), with
  `--num-training-specs 5` to `10` as the next step once the first loop is
  confirmed working, and 20–50 after that if runtime allows.

## 10. The first target specification — full provenance

Per instruction, no target value here is invented. All four numbers are the
existing repository thresholds, independently confirmed from two different
files:

| spec | value | source #1 | source #2 |
|---|---:|---|---|
| `dfe_locked_phase_eye_height_v` | ≥ 0.1 V | `simulator/rl_adapter.py:120` (`height - 0.1`, `CONSTRAINT_NAMES: "eye_height_over_100mv"`) | `simulator/receiver.py::_transient_violations`, `CANDIDATE`-fidelity gate `<= 0.1` |
| `dfe_eye_width_ui` | ≥ 0.4 UI | `simulator/rl_adapter.py:121` (`width - 0.4`, `"eye_width_over_0p4ui"`) | `simulator/receiver.py::_transient_violations`, `CANDIDATE`-fidelity gate `<= 0.4` |
| `dfe_min_margin_v` | > 0.0 V | `simulator/rl_adapter.py:119` (`margin`, `"positive_margin"`) | `simulator/receiver.py::_transient_violations`, `TRAINING+`-fidelity gate `<= 0` |
| `ctle_power_w` | ≤ 0.015 W | `simulator/rl_adapter.py:122` (`0.015 - power`, `"power_under_15mw"`) | `simulator/receiver.py` DC/transient violations, `"(0, 15 mW)"` |

`rl/target_spec.py::TargetSpec.from_existing_thresholds()` returns exactly
these four numbers, used as-is with no invented safety margin.

**Achievability**, confirmed by a real re-evaluation in this session (§7,
`known_good_repeat_0`, `reward_v1 = 100.0`, `success = True`):

| spec | target | achieved | margin |
|---|---:|---:|---|
| `dfe_locked_phase_eye_height_v` | 0.1 V | 1.5215 V | 15.2x |
| `dfe_eye_width_ui` | 0.4 UI | 0.87 UI | 2.2x |
| `dfe_min_margin_v` | 0.0 V | 0.5174 V | comfortably positive |
| `ctle_power_w` | 0.015 W | 0.001085 W | 13.8x under budget |

## 11. First end-to-end SPICE smoke test — RESULT

Run via:

```
python -m experiments.train_autockt --updates 2 --episodes-per-update 2 --horizon 6 --seed 42 --output results/autockt_smoke_test.jsonl
```

(defaults: `--num-training-specs 1`, i.e. the single achievable target from
§10; real SPICE throughout, nothing mocked). Full loop exercised:

```
target specification -> RL state -> PPO policy -> discrete parameter action
-> NEBULA parameter update -> real SPICE evaluation (ReceiverRLAdapter/evaluate_receiver)
-> receiver metrics -> AutoCkt-style reward -> next state -> PPO update -> next step
```

Actual run, seed 42:

| update | episodes | real SPICE evaluations | mean episode reward | spec satisfied | wall-clock |
|---|---:|---:|---:|---|---:|
| 0 | 2 | 2 (evals 1-2) | 10.0 | yes, both episodes | 126.80 s |
| 1 | 2 | 2 (evals 3-4) | 10.0 | yes, both episodes | 127.18 s |

**All 4 real SPICE evaluations succeeded and satisfied the target spec on
the very first step of every episode** (reward = `TERMINAL_BONUS = 10.0`
each time; every episode ended in 1 step, well under `horizon=6`) — expected
given §10's margins (the initial and nearby grid-perturbed points all clear
the deliberately generous thresholds by 2x-15x). `best_reward = 10.0`,
`best_parameters` recorded in the run summary and in
`results/autockt_smoke_test.jsonl`. Total wall-clock ≈ 254 s for 4
evaluations (≈ 63.5 s/eval), consistent with the §7 measurement.

This confirms the full pipeline closes end-to-end against real SPICE with
no mocked circuit results, exactly as required. It does **not** yet
demonstrate learning under a harder, non-trivially-satisfied target — that
requires either a tighter target spec or a worse initial point, planned as
a next step (§13) rather than part of this smoke test's scope.

After the run, the full existing test suite
(`python -m unittest discover -s tests`) was re-run: same result as before
this session's changes — 104 tests total (73 pre-existing + 31 new in
`tests/test_autockt_rl.py`), 1 pre-existing failure unrelated to RL
(`test_ctle.py::test_missing_model_has_specific_failure`, an environmental
SKY130-path issue, not introduced by this work), 2 skipped (SKY130-gated,
expected without the PDK). `tests/test_rl_readiness.py` alone: 22/22 pass,
unchanged. No regressions.

## 12. Environment notes

`KMP_DUPLICATE_LIB_OK=TRUE` is set at the top of
`experiments/train_autockt.py`, before `import torch`. This machine's
conda-forge numpy/scipy stack and the pip-installed CPU PyTorch wheel each
ship their own `libomp.dylib`; without the workaround, `import torch`
alongside `numpy` crashes with an OpenMP duplicate-initialization abort.
This is a local environment note, not a NEBULA/AutoCkt methodology choice.

## 14. Hard target — rationale, implementation, and first smoke-test result

### Why the trivial target (§10) gave no learning signal

`TargetSpec.from_existing_thresholds()` sits exactly at the simulator's own
minimum pass/fail gates (height ≥0.1V, width ≥0.4UI, margin >0V, power
≤0.015W). The smoke test's fixed initial design (Design A, see below)
clears every one of these by 2×-15×, and PPO's very first random action
only perturbs ±1 grid step per parameter on a 21-point linear grid, so the
resulting neighborhood still clears the trivial thresholds almost every
time. This was confirmed in the §11 smoke test: all 4 real evaluations hit
the terminal bonus on step 1 of every episode, giving PPO no failing
example to learn from.

### The hard target

Added additively in `rl/target_spec.py` as `HARD_TARGET_THRESHOLDS` /
`TargetSpec.from_hard_target()`, without altering `EXISTING_THRESHOLDS` or
`from_existing_thresholds()`:

| spec | trivial | **hard** | Design A | Design B |
|---|---:|---:|---:|---:|
| `dfe_locked_phase_eye_height_v` | ≥0.1 V | **≥0.8 V** | 1.5215 V ✓ | 0.5514 V ✗ |
| `dfe_eye_width_ui` | ≥0.4 UI | **≥0.6 UI** | 0.87 UI ✓ | 0.74 UI ✓ |
| `dfe_min_margin_v` | >0.0 V | **≥0.35 V** | 0.5174 V ✓ | 0.2295 V ✗ |
| `ctle_power_w` | ≤0.015 W | **≤0.015 W (unchanged)** | 0.001085 W ✓ | 0.000345 W ✓ |

- **Design A** = `results/receiver_random_search_20_seed123.jsonl`,
  candidate_index 8 (the smoke test's fixed initial design).
- **Design B** = `results/rl_reward_directed_smoke.jsonl`, episode 1 step 1
  (`reward_v1=70.656`, `success=true`) — a second, independent real
  SPICE-confirmed success that clears the trivial target comfortably but
  fails the hard target on height and margin.

This proves the hard target is both **achievable** (Design A) and
**genuinely selective** (Design B fails it) using only real, SPICE-verified
outcomes — nothing extrapolated. Power was deliberately left unchanged: both
known designs use <1.1mW against a 15mW budget, so tightening it further on
only two data points risked an infeasible joint target.

Selected via a new, additive `--target-mode {trivial,hard}` CLI flag on
`experiments/train_autockt.py` (default `trivial`, so existing invocations
are unchanged). No changes were made to `rl/autockt_state.py`,
`rl/autockt_action.py`, `rl/autockt_reward.py`, `rl/autockt_env.py`,
`rl/parameter_grid.py`, `rl/ppo_agent.py`, `rl/trainer.py`,
`simulator/rl_adapter.py`, `simulator/receiver.py`, or `reward_v1` — all
target-consuming code is already target-agnostic.

New SPICE-free tests (`tests/test_autockt_rl.py`, 6 added, all passing):
value-pinning, strict-tightening-vs-trivial, non-mutation of the trivial
target, Design-A-satisfies / Design-B-fails at the raw-threshold level, and
the same pass/fail check routed through the actual `autockt_reward`
function.

### First real-SPICE smoke test against the hard target — RESULT

```
python -m experiments.train_autockt --updates 1 --episodes-per-update 2 --horizon 3 \
    --target-mode hard --seed 42 --output results/autockt_hard_smoke_test.jsonl
```

| episode | step | real SPICE eval # | reward | spec satisfied | evaluated parameters |
|---|---:|---:|---:|---|---|
| 0 | 1 | 1 | 10.0 | yes | rload=2457.1, rdeg=485.7, cdeg=4.857e-13, itail=6.229e-4, dfe_tap=0.0571 |
| 1 | 1 | 2 | 10.0 | yes | rload=2457.1, rdeg=485.7, cdeg=9.614e-13, itail=5.757e-4, dfe_tap=0.0571 |

Both episodes terminated at step 1 (of a 3-step horizon) — the terminal
bonus was hit immediately both times. **2 real SPICE evaluations, 126.60s
wall-clock** (≈63.3s/eval, consistent with §7). `best_reward = 10.0`.

**Honest assessment, per instruction 7 — this run is inconclusive, not
positive evidence of learning:**

- **The hard target's discriminative property is real** (established
  independently via the Design A/B unit tests above), but **this specific
  2-evaluation draw did not exercise it** — both randomly-perturbed
  first-actions under seed 42 happened to land in parameter combinations
  that still clear the harder thresholds. No failing evaluation was
  observed in this run, so there is no fail→improve trajectory to point to
  as evidence of learning.
- This is **not** evidence that the hard target is still trivial in
  general (Design B already disproves that) — it is evidence that the
  *local one-step neighborhood* around this particular fixed initial point,
  for this particular seed, is still broadly forgiving. With only 2 samples,
  the data cannot distinguish "the neighborhood is mostly good" from
  "unlucky/lucky seed."
- **Per instruction 7, no algorithmic change is being made based on this
  alone.** The candidate next diagnostic steps, in order of cost, are: (a)
  try 1-2 more seeds at the same tiny scale (`--updates 1
  --episodes-per-update 2 --horizon 3`, ~2 more evaluations each) to check
  whether the same neighborhood keeps passing; (b) increase horizon/episode
  count modestly so a wider slice of the ±1-step neighborhood gets sampled,
  increasing the chance of hitting a failing point (Design B shows failing
  points exist somewhere in the reachable space, just not necessarily one
  step from Design A); (c) only if (a)/(b) still show no failures, revisit
  whether the fixed fully-verified initial point is simply too deep in a
  "good" basin for a single-step neighborhood to escape, which would point
  at episode mechanics / initial-state choice rather than reward shaping or
  PPO itself as the next thing to adjust.

This decision (which of a/b/c to run next, and at what scale) is left for
explicit approval before proceeding, per instruction.

## 15. Repairs #1/#2, mixed targets, bounded randomized init, and the PPO-learning confirmation run

Later sessions (after §14) approved and implemented, in order, each validated
with SPICE-free tests and the synthetic benchmark before any further
real-SPICE run:

- **Repair #1 [NEBULA ADAPTATION]**: `rl/parameter_grid.py::build_parameter_grids(..., spacing="log")`
  — `np.geomspace` for the four `scale="log"` parameters (`dfe_tap_v` stays
  linear), opt-in, default unchanged (`spacing="linear"`). Grid-math analysis
  showed this does not reliably shrink the single-action relative jump at the
  verified-good operating point (still ~99% for `cdeg_f`/`rdeg_ohm`, since a
  21-point log grid over ~3 decades has a ~41%/step constant relative
  spacing) — its benefit is bounding the *worst case* near the grid's low
  end, not fixing the specific `ac`-failure pattern by itself.
- **Repair #2 [NEBULA ADAPTATION]**: `rl/ppo_agent.py::Transition`/`compute_gae`
  now distinguish `terminated` (true AutoCkt terminal bonus, zero-bootstrap)
  from `truncated` (horizon cutoff, bootstraps from the value net's own
  estimate of the actual post-truncation state, captured in
  `rl/trainer.py::collect_rollout`) — previously both were conflated under a
  single `done` flag and always zero-bootstrapped, biasing value targets for
  every truncated (mostly-failing) episode.
- **`--target-mode mixed` [NEBULA ADAPTATION]**: training pool =
  (trivial, hard); validation pool = their arithmetic midpoint, held out.
  Avoids `sample_target_pool`'s continuous ranges, which inspection proved
  degenerate (every dimension's bound sits at/below Design A's already-
  achieved value, so Design A satisfies literally every target those ranges
  can produce).
- **`--randomize-initial-state` [NEBULA ADAPTATION]**: each episode's
  starting grid indices are independently perturbed by
  `{-1, 0, +1}` per parameter (symmetric, distinct from the policy's own
  `{-1, 0, +2}` action-delta set, clipped to grid bounds), deterministic
  under `--seed`, default disabled. Targets the empirically-confirmed
  "fixed initial design already satisfies the target" ceiling effect
  (`VERIFIED_INITIAL_PARAMETERS` is Design A, which comfortably clears
  every target this repo can express).

### Phase 1 diagnosis and the one RL-side repair made this session

Systematic inspection (state normalization, action probabilities, entropy,
advantage normalization, reward scale, terminal-bonus dominance, partial-
reward distribution, failure handling, episode init, target difficulty,
grid spacing, batch composition, update frequency) found one clear,
evidence-backed learnability defect and no others severe enough to act on:

**State-scale imbalance causing tanh saturation.** `rl/autockt_state.py::build_state`
concatenates 8 roughly-`[-1,1]`-bounded lookup terms with 5 **raw, un-normalized**
grid indices (range `0..grid_points-1`, i.e. up to 20 for the default
21-point grid) — `[AUTOCKT-REPLICATED]`, verified from AutoCkt's own source
(`cur_params_idx` is raw there too). A direct forward-pass check (representative
real states through a freshly-initialized `PolicyNetwork`) measured **40% of
first-hidden-layer (state, unit) pairs with `|pre-activation| > 3`** (tanh
output magnitude > 0.995, i.e. saturated) at initialization — the raw-index
features, an order of magnitude larger than the spec-error features, were
dominating the first layer's linear combinations before the nonlinearity.

**Fix [NEBULA ADAPTATION]**: added `ParameterGrid.normalized_index(index) -> float`
(`rl/parameter_grid.py`, maps to `[-1, 1]`, endpoints exact) and changed
`rl/autockt_env.py::AutoCktReceiverEnv` to feed `build_state` these
normalized indices instead of raw ones (`_normalized_indices()`, used in
both `reset()` and `step()`). `rl/autockt_state.py::build_state` itself is
**unmodified** — it still accepts/documents raw indices for direct callers
(e.g. its own unit tests); only what the environment *feeds* it changed.
Re-running the same forward-pass check post-fix: **0% saturation**, mean
`|pre-activation|` dropped from 3.48 to 0.34 (>10x). The synthetic PPO
benchmark still learns post-fix (seed 7, 15 updates × 12 episodes: mean
reward −4.184 → −3.627 over first/last 5 updates, 15/15 updates with
nonzero `policy_loss`) — no regression.

Also added, as an evaluation aid (not a training-mechanics change):
`experiments/train_autockt.py::_evaluate_checkpoint` + `--checkpoint-eval-episodes`
`[NEBULA ADAPTATION, no AutoCkt equivalent]` — freezes the policy's weights
before training ("initial") and after ("final"), then runs N fully
deterministic (`agent.act(..., deterministic=True)`) real-SPICE episodes for
each against a **matched** (shared seed) fresh environment, isolating
learned-policy behavior from on-policy exploration noise.

### Phase 3/4 — bounded real-SPICE confirmation run

```
python -m experiments.train_autockt --updates 6 --episodes-per-update 6 \
    --horizon 4 --target-mode hard --grid-spacing log --randomize-initial-state \
    --checkpoint-eval-episodes 3 --seed 42 --output results/autockt_normfix_confirmation.jsonl
```

**110 real SPICE evaluations** (100 training + 7 initial-checkpoint + 3
final-checkpoint), **130.7 minutes wall-clock** (12:41:56–14:52:39).

Training-time mean episode reward by update: 2.833 → 5.333 → 2.767 → 2.833 →
**−1.507** → 0.685 — still noisy, not a clean monotone trend (consistent
with every prior real-SPICE run; on-policy training reward mixes learned
preference with deliberate entropy-driven exploration). All 6 updates had
nonzero `policy_loss` (no advantage-collapse recurrence). 100 training
evaluations: 19 success (19.0%), 81 failure (`ac`=52, `setup`=25, `dc`=4).

**Checkpoint comparison (the decisive result)** — 3 matched real-SPICE
episodes per checkpoint, identical env seed (so identical target and
identical sequence of randomized starting points for both checkpoints),
only the policy weights differ:

| matched episode | initial (untrained) policy | final (trained) policy |
|---|---|---|
| 0 | 10.0, satisfied, 1 step | 10.0, satisfied, 1 step (tie) |
| 1 | −1.47, **failed**, 4 steps (truncated) | 10.0, satisfied, 1 step (**win**) |
| 2 | 9.0, satisfied, 2 steps | 10.0, satisfied, 1 step (**win**, also faster) |

Aggregate: initial mean reward 5.843, satisfaction 66.7% (2/3) — final mean
reward **10.0**, satisfaction **100% (3/3)**. The final policy is never
worse than the initial policy on any of the 3 matched starting points, and
strictly better on 2 of 3 (one failure→success flip, one success made
faster). This is a small sample (n=3 matched pairs) but a methodologically
clean one: the shared seed removes environment/target variability as a
confound, and deterministic action selection removes exploration noise —
isolating a genuine, unanimous, real-SPICE improvement attributable to the
6 PPO updates in between.

**Conclusion**: this constitutes defensible (not conclusive at n=3, but
methodologically sound and unanimous) real-SPICE evidence of PPO learning.
Per the governing research plan, the PPO baseline (state/action/reward/PPO
architecture as of this run, plus the state-normalization fix) is now
**locked** — no further changes to core PPO formulation are planned pending
new evidence. See the session's final report for recommended next
milestones (multi-target training, Random Search / CEM comparison baselines,
PVT robustness, SPICE-efficiency comparison).

Per explicit instruction, none of the following are implemented: TD(0),
TD(λ), PVT-aware optimization, surrogate-assisted RL, multi-actor/MA-Opt
exploration, multi-objective/Pareto solutions, or paper-scale (350-spec)
training. Existing baselines (random search, constraint-aware random
search, CEM, the sequential local-search baseline in `train_rl.py`) are
unmodified and remain available for comparison.

## 16. Multi-target-generalization milestone — inspection, gap found, and proposed experiment

With the PPO baseline locked (§15), this session inspected whether the
existing `--target-mode mixed` implementation (added in §15, but never yet
run against real SPICE — the §15 confirmation run used `--target-mode hard`,
a single fixed target) actually satisfies the milestone's requirement:
*train on trivial + hard, validate zero-shot on an unseen midpoint*. No
`rl/*.py` core-PPO file was touched; only `experiments/train_autockt.py`
(CLI/evaluation plumbing, not part of the locked formulation) and tests.

### What was already correct (pool construction)

`experiments/train_autockt.py::_build_target_pools`'s `mixed` branch and
`tests/test_autockt_rl.py::TargetModeMixedTests` (pre-existing, unchanged)
together already established, exactly:

- **training pool** = `(TargetSpec.from_existing_thresholds(), TargetSpec.from_hard_target())`
  — literally the trivial and hard targets, nothing resampled.
- **validation pool** = exactly one target, the per-dimension arithmetic
  midpoint: `dfe_locked_phase_eye_height_v=0.45, dfe_eye_width_ui=0.5,
  dfe_min_margin_v=0.175, ctle_power_w=0.015`.
- the midpoint is `assertNotEqual` to both training targets.
- `--target-mode trivial`/`hard` are provably unchanged by the `mixed`
  addition (`test_existing_trivial_and_hard_modes_are_unchanged`).

### The gap found: the runtime validation path was untested and unreachable except by CLI

`AutoCktReceiverEnv.reset(target=...)` — the mechanism that makes a
midpoint *not in `env.target_pool`* actually get evaluated — is called in
exactly one place in the whole repository: inline inside `main()`'s body.
No test exercised it; the only way to know it worked was to run real SPICE
through the full CLI. That is the one part of the "validate zero-shot on an
unseen target" claim §15's pool-construction tests could not cover.

**Fix**: extracted the loop into `_evaluate_validation_pool(*, env, agent,
validation_pool)` (pure refactor, identical behavior, `main()` now just
calls it) and added `tests/test_autockt_rl.py::ValidationPoolEvaluationTests`
(5 tests) + `MixedModeMilestoneTests` (1 test), all SPICE-free via the
synthetic/fake evaluator pattern §-wide tests already use. They confirm:
`reset(target=...)` evaluates the *given* target even when the env's own
`target_pool` never contains it (not a silent resample); multiple
validation targets are evaluated independently, in order; two independent
env/agent builds under the same seed produce byte-identical validation rows
(reproducibility); the deterministic evaluation call never mutates policy
weights (no accidental training during "zero-shot" evaluation); a failed
SPICE evaluation correctly reports `spec_satisfied=False`; and the full
`_build_target_pools` → `_evaluate_validation_pool` chain, run together,
reports the midpoint (not a training target) as the evaluated target. 98/98
SPICE-free tests pass repo-wide after this change (72 pre-existing +
20 `tests/test_baseline_comparison.py`, unrelated to this milestone + 6 new).

### A second, more substantive gap: single-shot validation cannot show generalization is *caused by training*

`main()` calls `_evaluate_validation_pool` exactly once, after training,
against only the **final** policy. That can show the trained policy
*achieves* the unseen target, but — per the same ceiling-effect risk §14
already diagnosed for the trivial target — it cannot by itself distinguish
"the policy generalized because of training" from "the midpoint target
happens to be easy from the fixed/near-good initial design regardless of
training." §15's training-target comparison avoided exactly this trap via
`_evaluate_checkpoint`'s matched initial-vs-final, same-seed methodology;
the validation pool currently gets no equivalent before/after comparison.

**Not fixed in this session** (out of scope for "tests and documentation
only" — flagged, not implemented): `main()` could get this rigor for free,
with zero new algorithmic code, by calling the *already-existing, already
real-SPICE-validated* `_evaluate_checkpoint` a second time per checkpoint
with `training_pool=(midpoint,)` — a single-element pool always resolves to
the midpoint on every `reset()`, so no new `target=` plumbing is even
needed. This is the recommended follow-up once the experiment below has run
once and the team wants the stronger causal claim.

### Proposed next real-SPICE experiment (NOT run this session)

Smallest defensible design: reuse the exact hyperparameters §15 already
validated end-to-end in real SPICE, changing only `--target-mode hard` →
`--target-mode mixed` — no other confound introduced.

```
python -m experiments.train_autockt --updates 6 --episodes-per-update 6 \
    --horizon 4 --target-mode mixed --grid-spacing log --randomize-initial-state \
    --checkpoint-eval-episodes 3 --seed 42 \
    --output results/autockt_mixed_target_confirmation.jsonl
```

**Expected evaluation count / runtime** (extrapolated from §15's own
measurement of this identical configuration at `--target-mode hard`, the
only real-SPICE datapoint available — not a new claim, a scaled reuse of an
already-measured number): §15 saw 100 training evaluations for
6 updates × 6 episodes × horizon 4, plus 10 checkpoint evaluations (7
initial + 3 final), for 110 total in 130.7 minutes (~71s/eval mean,
consistent with §7's ~63s/eval full-pipeline figure). With two training
targets instead of one, the split between them will differ, but the same
`updates`/`episodes-per-update`/`horizon` bounds the total the same way:
expect on the order of **~100-140 real SPICE evaluations, ~120-160 minutes
wall-clock**. The built-in post-training validation check adds at most
`horizon` (4) more.

**What would count as successful generalization**, in increasing order of
rigor:

1. *(computed automatically, no code change)* `validation_results[0]["spec_satisfied"] == True`
   in the output JSON — the final trained policy, deterministic, one-shot,
   clears the unseen midpoint target it never trained on.
2. *(requires the follow-up above)* the final policy's `_evaluate_checkpoint`
   result against `training_pool=(midpoint,)` is never worse, and ideally
   strictly better, than the initial (untrained) policy's — the same
   matched-seed methodology that produced §15's defensible learning claim,
   applied to the unseen target instead of the trained-on ones.
3. Training itself should not collapse: trivial-target success should stay
   near Design A's known comfortable margins, and hard-target success should
   stay in the same rough band §15 observed (19%) rather than dropping to
   ~0% — evidence that training on two targets didn't starve either one.

### Synopsis mapping

This milestone is the repo's first attempt at the synopsis's multi-target
generalization requirement specifically (as opposed to single-target
learning, already established in §15). Criterion 1 above is the minimum bar
the synopsis's "validate zero-shot on an unseen target" language requires;
criterion 2 is what would make that claim causally defensible rather than
merely observed, matching the evidentiary bar §15 already set and locked
for the single-target case.

## 17. Mixed-target real-SPICE run — RESULT, and the R/C/Cdeg counterfactual sweep

Run exactly as proposed in §16, unmodified:

```
python -m experiments.train_autockt --updates 6 --episodes-per-update 6 \
    --horizon 4 --target-mode mixed --grid-spacing log --randomize-initial-state \
    --checkpoint-eval-episodes 3 --seed 42 \
    --output results/autockt_mixed_target_confirmation.jsonl
```

**108 real SPICE evaluations** (95 training + 9 initial-checkpoint + 3
final-checkpoint + 1 validation, the last inferred from an exact
`episode_reward == 10.0` which the reward mechanics make achievable only by
a single-step episode -- no row-level evaluation count is logged for the
validation call itself, a minor gap in `main()`'s own bookkeeping, not
fixed here), **≈172 minutes wall-clock** (15:26-18:18, measured directly
from process start/exit, not extrapolated).

### Training performance on both targets

36 episodes total, split 23 trivial / 13 hard by the environment's own
per-episode random draw from the 2-target pool (not a fixed split):

| target | episodes | satisfied | rate | mean steps/ep | evals used | failure stages |
|---|---:|---:|---:|---:|---:|---|
| trivial | 23 | 13 | 56.5% | 2.83 | 65 | ac=32, setup=19, dc=1 |
| hard | 13 | 8 | 61.5% | 2.31 | 30 | ac=14, setup=4 |

Both targets converged to comparable, substantial success rates -- training
jointly on trivial + hard did not starve either one. (The hard target's
slightly higher rate here is on a much smaller episode count than trivial's
and should not be read as "hard is easier than trivial"; --target-mode
hard alone, §15, saw 19% over 100 evaluations under different
hyperparameters -- these numbers are not directly comparable across runs.)

### Matched initial-vs-final checkpoint comparison (training targets)

Same shared-seed methodology as §15 (`_evaluate_checkpoint`, 3 deterministic
episodes per checkpoint, identical targets and randomized starting points
for both):

| episode | target | initial (untrained) | final (trained) |
|---|---|---|---|
| 0 | hard | 10.0, satisfied, 1 step | 10.0, satisfied, 1 step (tie) |
| 1 | hard | -1.47, failed, 4 steps (truncated) | 10.0, satisfied, 1 step (win) |
| 2 | trivial | -4.0, failed, 4 steps (truncated) | 10.0, satisfied, 1 step (win) |

Aggregate: 33.3% -> 100% satisfaction, mean reward 1.51 -> 10.0. Identical
shape to §15's single-target result (1 tie, 2 wins, 0 losses) -- consistent,
not novel, evidence that training improves matched-start performance on the
targets actually trained on.

### Unseen midpoint target -- zero-shot result, with its real limitation

`validation_results`: `episode_reward=10.0, spec_satisfied=true` -- the
**final** policy solved the never-trained-on midpoint
(height=0.45V, width=0.5UI, margin=0.175V, power=0.015W), in (inferred) one
evaluation. This is genuine **achievability** evidence.

It is **not** the same quality of evidence as the checkpoint comparison
above, for exactly the reason flagged in §16 before this run: `main()`
evaluates the validation pool **once**, against the **final** policy only,
from an **unmatched, uncontrolled** starting point (continuing the shared
`env`'s own RNG stream after 36 training episodes, not a fresh matched-seed
`eval_env` the way `_evaluate_checkpoint` builds one). No initial-policy
baseline exists for this specific target in this run. n=1, uncontrolled.

**What can honestly be said**: the trained policy achieved a target
strictly between trivial and hard on 3 of 4 dimensions, never presented
during training. **What cannot honestly be said**: that this happened
*because of* training rather than being reachable from an arbitrary
starting point regardless (the same ceiling-effect risk §14 diagnosed for
the original trivial target). The checkpoint comparison above is indirect,
suggestive context -- the untrained policy failed 2 of 3 matched trials
against the *easier* trivial and *harder* hard targets -- but is not a
substitute for a matched trial against the midpoint itself.

**Verdict: zero-shot generalization is PARTIALLY demonstrated** --
achievability shown; causal/robustness evidence not established by this
run's own design. Closing this gap (per §16's flagged, not-yet-implemented
follow-up: reuse `_evaluate_checkpoint` with `training_pool=(midpoint,)`)
remains the concrete next step for anyone who needs the stronger claim.

### R/C/Cdeg one-factor-at-a-time counterfactual sweep

Per the governing instruction, run only *after* the above was analyzed, via
**direct `simulator.receiver.evaluate_receiver` calls** -- no
`ReceiverRLAdapter`, no `PPOAgent`, no policy, no retraining. Same
evaluation conditions the RL run used (`SimulationConditions()` defaults,
`EvaluationFidelity.TRAINING`, default synthetic channel, no cache).

**Baseline** (`run_complete.best_parameters` -- the run's own designated
best design, tied for the maximum achievable reward and, by the trainer's
strict-greater-than update rule, the *first* such design encountered,
episode 1/step 1/eval 5, against the **trivial** target -- not a claim of
global uniqueness, since the flat 10.0 terminal bonus is tied by many
designs seen later in training too):

```
rload_ohm=1995.26, rdeg_ohm=446.68, cdeg_f=4.4668e-13, itail_a=6.3096e-4, dfe_tap_v=-0.05714
```

**18 candidates** (3 parameters x 6 grid-index offsets {-4,-2,-1,+1,+2,+4},
no clipping, no duplicates), one parameter moved at a time, the other four
held exactly at baseline. **22.3 minutes / 1340.8s total wall-clock**
(mean 74.5s/eval). Full results: `results/rc_counterfactual_sweep_mixed_target.jsonl`.

**Caveat on what's directly comparable**: `propose_one_factor_at_a_time_sweep`
deliberately excludes the offset-0 (baseline) point, and the original
training log never recorded the baseline's own raw metrics (§16's "KNOWN
GAP" -- `rl/trainer.py` logs reward/success but not `metrics`). So the
baseline's own numeric height/width/margin cannot be directly quoted here
-- only its pass/fail (pass) and flat reward (10.0) are known. All
comparisons below are among the 18 candidates and against that known
pass/fail fact, not against a measured baseline margin.

**rload_ohm** (baseline 1995.26 ohm):

| offset | value (ohm) | result |
|---:|---:|---|
| -4 | 794.3 | **pass** (height=0.733, width=0.89, margin=0.357) |
| -2 | 1258.9 | fail (ac) |
| -1 | 1584.9 | fail (ac) |
| +1 | 2511.9 | fail (ac) |
| +2 | 3162.3 | fail (ac) |
| +4 | 5011.9 | fail (ac) |

**Not** a simple one-sided edge: both immediately adjacent grid steps (one
step down, one step up) fail, while a further step down (4 steps) passes
again. 1 of 6 tested neighbors preserves success. This is the most fragile
of the three axes at this specific grid resolution -- the trained value is
not robust to the smallest tested perturbation in either direction, though
it is not isolated on an island either (a non-adjacent point also works).

**rdeg_ohm** (baseline 446.68 ohm):

| offset | value (ohm) | result |
|---:|---:|---|
| -4 | 112.2 | fail (ac) |
| -2 | 223.9 | fail (ac) |
| -1 | 316.2 | fail (ac) |
| +1 | 631.0 | **pass** (height=1.351, width=0.84, margin=0.623) |
| +2 | 891.3 | **pass** (height=1.153, width=0.81, margin=0.505) |
| +4 | 1778.3 | **pass** (height=0.809, width=0.77, margin=0.307) |

Clean one-sided edge: every tested decrease fails, every tested increase
(up to 4 steps) passes. Among the passing points, margin and height both
*decrease* as rdeg_ohm increases further from baseline (0.623 -> 0.505 ->
0.307) -- consistent with (not proof of) the baseline sitting near the
favorable, low-rdeg end of a passing band that keeps degrading (in margin
terms) as rdeg_ohm grows, and failing outright below it.

**cdeg_f** (baseline 4.4668e-13 F):

| offset | value (F) | result |
|---:|---:|---|
| -4 | 1.122e-13 | fail (ac) |
| -2 | 2.239e-13 | fail (ac) |
| -1 | 3.162e-13 | fail (ac) |
| +1 | 6.310e-13 | **pass** (height=1.572, width=0.91, margin=0.721) |
| +2 | 8.913e-13 | **pass** (height=1.641, width=0.90, margin=0.708) |
| +4 | 1.778e-12 | **pass** (height=1.723, width=0.87, margin=0.695) |

Same shape as rdeg_ohm: clean one-sided edge, every decrease fails, every
tested increase passes, with margin gently decreasing as cdeg_f grows
further from baseline.

**`ctle_power_w` sanity check**: identical (0.0011357...) across all 18
rows regardless of which of rload/rdeg/cdeg was swept -- correctly
confirming power in this topology is set by `itail_a` (held fixed
throughout), not by R/C, and independently confirming the sweep's "hold
the other four parameters exactly at baseline" mechanic worked as designed.

### Interpretation -- what this does and does not support

**Supported**: for `rdeg_ohm` and `cdeg_f`, the trained baseline sits at or
just above a real, sharp feasibility boundary (the `ac` gain/peaking gate)
-- decreasing either one degrades the design from passing to failing at the
very next grid step, with no exception among the 3 downward points tested
on either axis. That is a genuine, non-arbitrary constraint the trained
values respect. For `rload_ohm`, the picture is messier: the immediate
neighborhood is fragile (both adjacent steps fail) rather than cleanly
edge-like, consistent with `rload_ohm` interacting with `rdeg_ohm`/`cdeg_f`
jointly in setting the CTLE's AC response (a one-factor-at-a-time slice
through a coupled 3-parameter feasibility surface is not guaranteed to look
like a simple interval on any single axis, and here it visibly doesn't).

**Not supported**: any claim that the *reward itself* has a local maximum
at these values. `autockt_reward`'s terminal bonus is a **flat 10.0** for
every passing design -- there is no reward gradient among passing points
for the flat-reward regime the policy was actually optimizing under (RL's
training reward cannot distinguish margin=0.72 from margin=0.31; only
pass/fail matters to it). So "near a local reward optimum" is answered here
as: **near a local *feasibility* boundary on 2 of 3 axes, not a reward
optimum in the classical continuous sense** -- the reward landscape the
policy actually saw has no interior optimum to be near, only a plateau and
a cliff. Whether the policy's proximity to that cliff reflects deliberate,
learned boundary-seeking behavior or is simply where its particular
exploration trajectory happened to stop (nothing in the reward function
rewards moving toward the interior of a passing region once one is found)
cannot be distinguished by this experiment, and no such claim is made.

**On causality**: this experiment establishes a *local sensitivity*
property of one specific converged design under one-factor-at-a-time
perturbation -- it does not establish that PPO's parameter *selection
mechanism* (as opposed to the environment's own AC-stage physics) produced
this sensitivity, nor does it generalize beyond this one design/target
pair. A different baseline design (e.g. from a hard-target episode, or a
different seed) could plausibly show a different shape on any of these
three axes.

## 18. Controlled unseen-target generalization -- RESULT: regression, not confirmation

§17's zero-shot validation of the unseen midpoint target (height=0.45V,
width=0.5UI, margin=0.175V, power=0.015W) was explicitly flagged as
**uncontrolled**: one episode, final policy only, no matched initial-policy
baseline. This section closes that gap with the matched-checkpoint
methodology §15/§17 already validated for the *trained* targets, applied
here to the *unseen* one -- and the result **reverses** the earlier
single-episode impression.

### A real blocker, and how it was resolved

`experiments/train_autockt.py` never persisted policy weights to disk --
`initial_policy_state`/`final_policy_state` existed only in the training
process's memory and were lost when it exited after §17. Per explicit
instruction, this was **not** worked around by retraining differently or
fabricating a substitute: a minimal, disclosed, I/O-only addition
(`--save-final-policy`, `torch.save` called strictly after `train()`
returns -- no change to architecture, hyperparameters, seeds, state, action,
reward, or the training algorithm itself) was added to
`experiments/train_autockt.py`, and the **exact §17 command was re-run**
(`--updates 6 --episodes-per-update 6 --horizon 4 --target-mode mixed
--grid-spacing log --randomize-initial-state --seed 42`, output to a new
file so §17's own result was never touched).

**Reproduction was verified, not assumed**: all 95/95 training-step rows
from the re-run are **bit-identical** to §17's original 95 rows -- same
episode/step/reward/success/target/failure_stage, and every physical
parameter value to full float precision, including an exact match on
`best_parameters`. The saved checkpoint
(`results/autockt_mixed_target_confirmation_policy.pt`) is confirmed to be
the same trained policy §17 and the R/C counterfactual sweep already
analyzed, not a different one from a fresh training run.

### The controlled evaluation

`experiments/controlled_unseen_target_eval.py`, reusing
`experiments.train_autockt._evaluate_checkpoint` **unmodified**: both
checkpoints (a freshly-constructed, untrained `PPOAgent(seed=42)`, and the
saved trained policy) evaluated against **10 matched episodes** -- identical
seed (`eval_seed=42`) builds one fresh `AutoCktReceiverEnv` per checkpoint
call, so both see byte-identical randomized starting states and the
identical target (`training_pool=(midpoint,)`, a single-element pool that
always resolves every `reset()` to the midpoint); only the loaded policy
weights differ. Deterministic action selection throughout
(`agent.act(..., deterministic=True)`, inherited unchanged from
`_evaluate_checkpoint`). The harness itself was fully validated SPICE-free
first (7 tests, including the decisive check: identical weights loaded on
both sides produce 10/10 ties -- proof the matched-starting-state guarantee
actually holds before any real evaluation was spent).

**32 real SPICE evaluations** (13 for the initial/untrained checkpoint, 19
for the final/trained checkpoint), **~102 minutes wall-clock**
(22:03-23:45). Full per-case results: `results/controlled_unseen_target_generalization.jsonl`.

### Numerical result

| | initial (untrained) | final (trained) |
|---|---:|---:|
| satisfaction rate | **90%** (9/10) | **70%** (7/10) |
| mean reward | **8.6** | **5.8** |
| mean steps | 1.3 | 1.9 |

**Wins: 1. Ties: 6. Losses: 3.**

| episode | initial | final | outcome |
|---:|---|---|---|
| 0 | 10.0, satisfied, 1 step | -4.0, failed, 4 steps | **loss** |
| 1 | 10.0, satisfied, 1 step | -4.0, failed, 4 steps | **loss** |
| 2 | -4.0, failed, 4 steps | 10.0, satisfied, 1 step | win |
| 3 | 10.0, satisfied, 1 step | 10.0, satisfied, 1 step | tie |
| 4 | 10.0, satisfied, 1 step | 10.0, satisfied, 1 step | tie |
| 5 | 10.0, satisfied, 1 step | 10.0, satisfied, 1 step | tie |
| 6 | 10.0, satisfied, 1 step | 10.0, satisfied, 1 step | tie |
| 7 | 10.0, satisfied, 1 step | 10.0, satisfied, 1 step | tie |
| 8 | 10.0, satisfied, 1 step | 10.0, satisfied, 1 step | tie |
| 9 | 10.0, satisfied, 1 step | -4.0, failed, 4 steps | **loss** |

(`parameter_trajectory` is `None` for every case -- `_evaluate_checkpoint`'s
row schema, reused unmodified per instruction, does not log per-step
parameters; same documented limitation as §16/§17, not introduced here.)

### Verdict: REGRESSION

Per the governing success-interpretation scale, this is unambiguous, on
every aggregate metric measured: lower satisfaction rate (70% vs 90%),
lower mean reward (5.8 vs 8.6), and more losses than wins (3 vs 1). This is
**not** "no evidence" (the two policies are not equal) and **not**
"moderate/strong evidence of generalization" -- it is the opposite
direction from what would support a generalization claim.

**This reverses §17's own single-episode impression.** §17 reported the
final policy solving the midpoint target once, uncontrolled, and explicitly
warned that a single uncontrolled success could not distinguish "trained
policy generalizes" from "target was reachable regardless of training" --
exactly the ceiling-effect risk §14 first diagnosed for the original trivial
target. The controlled result shows the second explanation dominates: **the
untrained policy already satisfies this specific unseen target 90% of the
time** from randomized starting points near the fixed, real-design-verified
initial parameters -- consistent with §14's finding that the initial design
neighborhood is broadly forgiving. Training did not preserve that broad
coverage; on 3 of 10 matched points, the deterministic argmax action the
*trained* weights chose led to `ac`-stage failure (4-step truncation, reward
-4.0) from a starting point the *untrained* weights' argmax action solved in
one step.

**What this does and does not support.** Supported: on these 10 matched,
real-SPICE-verified starting points, under this exact target, the trained
policy is measurably worse than the untrained one -- a real, reproducible
(both sides deterministic) fact about these specific weights and these
specific points. Not supported, per instruction against overclaiming from
n=10: that the trained policy is worse *in general* at unseen targets, that
this specific regression pattern would hold under a different seed or
different randomized-initial-state draw, or a mechanistic explanation for
*why* training narrowed behavior away from the untrained policy's broader
coverage (plausible candidates -- the flat terminal-bonus reward giving no
incentive to preserve margin/robustness once any target is satisfied during
training; PPO's on-policy updates reinforcing trivial/hard-specific
behavior at the expense of the interpolating region -- are not distinguished
by this experiment).

**Net effect on the multi-target-generalization milestone (§16-18)**: the
mixed-target training run (§17) demonstrates real learning on the two
*trained* targets (matched checkpoint comparison, 1 tie/2 wins/0 losses,
identical shape to §15's single-target result). It does **not**, on this
controlled evidence, demonstrate that this benefit extends to an unseen
interpolating target -- if anything, this run's specific trained policy is
worse than a random initialization there. This is a materially different,
and more cautious, conclusion than §17's own uncontrolled note.

## 19. Fair PPO vs Random Search vs CEM comparison

With the PPO baseline locked (§15/§18), this milestone builds the first
scientifically fair cross-method comparison for the same circuit-sizing
problem. No PPO/simulator code was touched; all changes are isolated to
`experiments/train_cem.py` (additive), `analysis/baseline_comparison.py`
(additive field), and the new `analysis/fair_comparison.py`.

### The fairness gap found during inspection

Random Search (`experiments/receiver_search.py`) and CEM
(`experiments/train_cem.py`) both rank candidates by
`simulator.rl_adapter.reward_v1`, whose "success" is simply
`evaluation.success` -- the simulator's own TRAINING-fidelity gates. Those
gates hard-fail on `dfe_min_margin_v <= 0` and power outside (0, 15mW), but
**do not** hard-gate `dfe_locked_phase_eye_height_v >= 0.1` or
`dfe_eye_width_ui >= 0.4` at TRAINING fidelity (`simulator/receiver.py
::_transient_violations`, those two checks are gated only at
`fidelity >= CANDIDATE`). PPO's success is `autockt_reward`'s terminal
bonus, which requires all four thresholds simultaneously via a
`TargetSpec`. These are different criteria; comparing each method's own
native "success rate" would not compare the same thing. Fixed by
recomputing ONE shared criterion --
`autockt_reward` against `TargetSpec.from_existing_thresholds()` (the
"trivial" target; its four numbers ARE `EXISTING_THRESHOLDS`, the same
constants `reward_v1`/`constraints_from_evaluation` already reference, not
an invented number) -- for every candidate with raw metrics available.
Verified against ground truth: recomputing this way over
`receiver_random_search_20_seed123.jsonl` reproduces exactly the
independently-known result (1/20 successes, Design A, `reward=10.0`).

### Historical CEM data excluded, and why

`results/cem_baseline_3x10.jsonl` was excluded from this comparison for two
independent, confirmed reasons: **incomplete** (13/30 planned evaluations)
and **warm-started** (`experiments/train_cem.py`'s original hardcoded mean,
`[0.37, 0.28, 0.33, 0.78, -0.03]`, sits in a specific, already-decent region
of normalized action space, not centered/uniform like Random Search's
coverage). Presenting it against Random Search would not have been a fair
comparison, per explicit instruction.

### The one new experiment run

`experiments/train_cem.py` gained two additive capabilities, the CEM
mechanism itself (Gaussian sampling, elite selection by `reward_v1`,
mean/std adaptation) otherwise byte-for-byte unchanged:
- `--init-mean`/`--init-std` (defaults reproduce the original warm-started
  behavior exactly, pinned by a regression test) -- used here as
  `--init-mean 0 0 0 0 0 --init-std 0.577...` (`2/sqrt(12)`, `Uniform(-1,1)`'s
  own standard deviation) for first-generation coverage comparable to
  Random Search's uniform sampling, not centered on any known-good point.
- per-candidate `metrics` logging, via the existing, unmodified
  `rl.autockt_env.metrics_from_observation` helper reading the adapter's
  own public `observation` tuple -- makes post-hoc uniform re-scoring
  possible; does not change what CEM selects on.

```
python -m experiments.train_cem --iterations 4 --population 5 --elite 2 \
    --seed 123 --init-mean 0 0 0 0 0 \
    --init-std 0.5773502691896258 0.5773502691896258 0.5773502691896258 \
               0.5773502691896258 0.5773502691896258 \
    --output results/cem_unbiased_seed123.jsonl
```

**20/20 evaluations, complete, 630.3s (10.5 min) wall-clock** (directly
measured -- this is the first CEM run in the repo with per-row timing).

### A. PPO vs Random Search vs CEM

| method | n | native success | uniform success (trivial target) | best uniform reward |
|---|---:|---:|---:|---:|
| Random Search (`receiver_random_search_20_seed123.jsonl`, seed=123) | 20 | 5.0% (1/20) | 5.0% (1/20) | 10.0 |
| CEM, unbiased init (`cem_unbiased_seed123.jsonl`, seed=123) | 20 | 0.0% (0/20) | 0.0% (0/20) | -1.0 (no success) |
| PPO, trivial-target steps only (`autockt_mixed_target_confirmation.jsonl`, filtered) | 65 | 20.0% (13/65) per-step | *(not independently re-scoreable -- see below)* | 10.0 |
| PPO, trivial-target episodes | 23 episodes | 56.5% (13/23) per-episode | -- | 10.0 |

PPO's "uniform" column is empty because its training log predates
per-step `metrics` logging (§16's KNOWN GAP, still not closed -- touching
`rl/trainer.py` was excluded from scope). Its **native** success, however,
is already directly comparable here by construction: the filtered subset
is exactly the steps whose episode target *was* the trivial target, so
PPO's own `autockt_reward` terminal-bonus check already **is** the uniform
criterion for those rows -- nothing was recomputed differently for it.

### B. Evaluation efficiency

| method | evaluations to first success |
|---|---:|
| Random Search | 9 |
| CEM (unbiased) | never (0/20) |
| PPO (trivial-tagged steps, in-run order) | 5 (but see confound below -- this undercounts true cumulative budget) |

Per-evaluation cost also diverges sharply by outcome: CEM's 20 evaluations
averaged **31.5s/eval** (630.3s / 20) -- far below the ~63s/eval full-success
baseline (§7) -- because 100% of them failed early (16 at `dc`, 4 at `ac`;
no candidate ever reached the expensive transient/eye-diagram stage).
Random Search's failure breakdown is nearly identical in shape (15 `dc`,
4 `ac`, 1 success) -- strong evidence CEM's unbiased-init candidates are
statistically indistinguishable from Random Search's, not a different
population.

### C. Wall-clock

| method | total wall-clock |
|---|---:|
| Random Search | not recorded (pre-dates this session's timing instrumentation) |
| CEM (unbiased) | 630.3s (10.5 min), directly measured |
| PPO training (full mixed-target run, both targets) | ~5235s (87.3 min) for 95 evaluations (§17); trivial-tagged subset is 65/95 of those evaluations, not separable in wall-clock without re-running |

### D. Best design

Random Search's Design A (`candidate_index 8`, `rload_ohm=2342.5,
rdeg_ohm=822.4, cdeg_f=1.0e-12, itail_a=6.03e-4, dfe_tap_v=-0.011`) remains
the only independently-verified best design among the two non-PPO methods
this session ran a fair trial for -- CEM's unbiased run produced no
successful candidate at all, so it has no comparable "best design" beyond
its uniformly-worst-case -100.0/-1.0 score. PPO's best trivial-target
design is documented in §10/§14 (the fixed initial design, Design A itself
-- `VERIFIED_INITIAL_PARAMETERS` was seeded FROM this same Random Search
result, see §4, so PPO's "best design" on this target is not an
independent confirmation of a *different* good region, only of the same
one already known).

### E. Limitations and confounds (do not skip these when citing this comparison)

1. **CEM's zero successes is not proof CEM is worse than Random Search.**
   At n=20 and a ~5% base rate (Random Search's own measured rate),
   P(zero successes in 20 i.i.d. draws at p=0.05) ~= 0.95^20 ~= 35.8% --
   getting zero is unsurprising at this sample size even under an
   IDENTICAL underlying distribution. This comparison does not have the
   statistical power to distinguish "CEM performs identically to uniform
   random search" from "CEM performs somewhat worse."
2. **CEM's own adaptation mechanism got no usable gradient in this run.**
   `reward_v1` returns a flat -100.0 for every failure regardless of how
   close it came to passing -- the logged `best_reward` stayed at exactly
   -100.0 through all 4 iterations, and per-dimension std collapsed to its
   0.15 floor by iteration 2-3 (mean/std trace recorded in this section's
   source log). The "elite selection" step was choosing among candidates
   tied at the same score, not following any real signal. This is a
   property of `reward_v1`'s failure handling combined with this
   particular (unbiased, wide) initialization landing entirely in the
   failing region for all 20 draws -- not evidence about CEM as an
   algorithm in general, or about how it would behave with a landscape
   that gives graded failure signal.
3. **PPO's numbers are structurally not i.i.d. candidates.** Its 65/23
   trivial-tagged evaluations/episodes are sequential, on-policy steps
   within a shared-weight training run that ALSO trained against the hard
   target in the same 95-evaluation budget -- unlike Random Search/CEM's
   independent one-shot draws. "Evaluations to first success = 5" for PPO
   is real but undercounts the true cumulative SPICE spent by that point,
   since hard-target evaluations happened in real time between some of the
   counted trivial-tagged ones.
4. **PPO's training cost is not separable from its trivial-target
   performance.** The 65 trivial-tagged evaluations required the FULL
   95-evaluation, ~87-minute training run to produce (shared network
   weights) -- there is no cheaper way, with the current code, to get
   PPO's trivial-target learning curve without also paying for the
   hard-target training interleaved with it.
5. **Discretized grid vs. continuous space.** PPO searches a 21-point
   discretized grid per parameter (log-spaced in this run) via
   index-delta actions; Random Search/CEM sample the full continuous
   normalized `[-1,1]^5` space directly. This is a structural difference
   in search space resolution, not something either side's results can be
   adjusted for post-hoc.
6. **`VERIFIED_INITIAL_PARAMETERS` is not independent of Random Search.**
   PPO's fixed initial design (§4) was seeded FROM Random Search's own
   `candidate_index 8` result. PPO starting a design search already
   anchored at Random Search's best-known point, then finding a
   trivial-target success in a handful of steps, is not surprising or
   independent evidence of PPO's own search capability on this specific
   target -- it is largely inherited from Random Search's earlier result.
7. **No PVT, no CANDIDATE-fidelity validation.** All three methods here
   ran at TRAINING fidelity, nominal PVT corner only.

### F. Conclusion -- no overclaiming

**PPO does not demonstrate a clear practical advantage over Random Search
on this specific comparison, once the confounds above are accounted for.**
Its highest per-step success figure (20%) is inflated by starting from a
design already known-good from Random Search's own prior result (confound
6) and by counting only same-target steps out of a shared multi-target
training run whose total cost (95 evaluations, ~87 minutes) is not
separable from that number (confound 4). Random Search's 5% success rate
with n=20 (no warm start, no prior-result seeding) is the cleanest,
least-confounded number in this comparison. CEM's unbiased-init result
(0/20) cannot be distinguished from Random Search's distribution at this
sample size (confound 1), and its own adaptation mechanism was
demonstrably not exploiting any gradient in this run (confound 2) --
**the previously-reported high CEM success rate (8/13 in the warm-started,
incomplete run) is best explained by its biased initialization, not by
CEM's search algorithm**, consistent with the caveat flagged before this
milestone (`analysis/baseline_comparison.py::_CEM_INIT_CAVEAT`) and now
directly confirmed by running CEM without that warm start.

**What would change this conclusion**: a PPO run trained from a Random-Search-independent
initial point (not `VERIFIED_INITIAL_PARAMETERS`), evaluated purely on
trivial-target deployment cost (not entangled multi-target training), and
a larger-n CEM/Random-Search trial with enough power to distinguish their
success rates from noise. None of that is run here -- reported honestly as
absent, not assumed favorable to any method.

## 20. Head-to-head, no-warm-start comparison -- addressing §19's own confounds

§19's own conclusion named the confounds that made its comparison
insufficient: PPO inherited Random Search's best design as its starting
point, PPO's budget wasn't separable from multi-target training, CEM's
reward gave no gradient among failures, and PPO's sequential actions
weren't structurally comparable to RS/CEM's sampled candidates. This
section runs one clean experiment that removes each of those confounds as
far as possible without touching the locked PPO formulation, and reports
what actually happened -- honestly, including where a structural mismatch
remains unresolvable within the rules given.

### Methodology, fixed before any SPICE ran

**Same target**: `TargetSpec.from_existing_thresholds()` (trivial), all
three methods.

**Same success criterion**: unchanged `analysis/fair_comparison.py
::uniform_score` (`autockt_reward >= 10.0` against the trivial target).

**Same initial distribution -- centered, not warm-started**: the
normalized-action-space **center** for all three. Random Search already
samples uniformly around it (no change needed -- its existing
seed=123 run was reused, not re-run, since it already satisfies every
requirement below). CEM's Gaussian mean is `[0,0,0,0,0]`,
`std=2/sqrt(12)=0.5774` (§19's own unbiased-init design, unchanged). PPO
gained one new, additive, isolated CLI flag,
`--initial-indices-source {verified, grid-center}` (default `verified` --
unchanged existing behavior), extracted into a small testable helper
`experiments.train_autockt._resolve_initial_indices` (4 unit tests). With
`grid-center`, PPO's starting grid index is `10` for all 5 parameters on a
21-point log-spaced grid -- confirmed by direct computation to be the true
geometric midpoint of each parameter's own (lower, upper) bounds, and
confirmed **not equal to** `VERIFIED_INITIAL_PARAMETERS`'s indices
`(14, 13, 13, 18, 10)` (itself seeded from Random Search's own
`candidate_index 8`, per sec 4) -- requirement 3 (no PPO warm start)
verified structurally, not just asserted.

**Same physical parameter bounds/grid**: unchanged `ACTION_BOUNDS` for all
three (PPO's grid is built from the same bounds RS/CEM sample continuously
within; grid discretization itself remains a disclosed, unresolved
structural difference -- see Limitations).

**Same evaluation budget, n=20, exactly**: Random Search's existing n=20
reused. CEM: 4 iterations x population 5 = 20 (unchanged from sec 19).
PPO: `--updates 4 --episodes-per-update 5 --horizon 1` = `4*5*1 = 20`
**deterministically** (horizon=1 means every episode is exactly one
action-then-evaluate step, so there is no early-termination variance to
create ambiguity about how many evaluations were "actually" spent) -- the
closest structurally honest match to RS/CEM's one-shot-per-candidate
budget achievable without changing PPO's actual step/episode mechanics.
`--max-evaluations 20` set defensively (the update/episode/horizon product
already guarantees this exactly).

**Same seed**: 123, all three (Random Search's existing seed; CEM and PPO
newly run with it). Single seed only, per the given fallback ("otherwise 1
seed first") -- not a 3-seed trial.

**CEM's reward problem, fixed for elite selection only (not for reported
success)**: new `experiments/train_cem.py::graded_cem_fitness`, reusing
`rl.autockt_state.lookup`/`signed_relative_error` (the SAME machinery
`rl.autockt_reward.autockt_reward` itself uses, imported read-only, not
modified) to compute the same signed-relative-error sum PPO's own reward
uses -- but WITHOUT gating on `success` first, so a candidate whose
`failure_stage` is `"transient"` (the stage that actually computes eye
height/width/margin/power, per `simulator/receiver.py
::_transient_violations` -- both its passing and violating outcomes attach
real metrics) still contributes a graded score. Candidates that fail
earlier (`dc`/`ac`/`setup`/...) have no real measurements of the 4 target
metrics to grade on and get a fixed, disclosed floor
(`GRADED_FITNESS_NO_INFORMATION_FLOOR = -10.0`), not a fabricated proxy.
Verified SPICE-free (11 new tests): distinct scores for "close" vs. "far"
failing candidates when metrics are real, floor strictly below any graded
value, zero flat-collapse among transient-stage candidates -- the exact
property `reward_v1` lacks. **`reward`/`success` as logged and used for the
comparison's success criterion are unchanged** -- only what elite selection
optimizes toward differs when `--fitness graded` is passed.

**Verified before spending SPICE**: 155/155 SPICE-free tests passing
(15 new this section: 4 for `_resolve_initial_indices`, 11 for
`graded_cem_fitness`), plus a full synthetic-backend (`--backend
synthetic`) dry run of the exact PPO CLI invocation confirming exactly 20
evaluations and clean execution before any real ngspice call was made.

### What actually happened

```
python -m experiments.train_cem --iterations 4 --population 5 --elite 2 --seed 123 \
    --fitness graded --target trivial --init-mean 0 0 0 0 0 \
    --init-std 0.5773502691896258 0.5773502691896258 0.5773502691896258 \
               0.5773502691896258 0.5773502691896258 \
    --output results/cem_graded_unbiased_seed123.jsonl

python -m experiments.train_autockt --backend real --target-mode trivial --grid-spacing log \
    --randomize-initial-state --initial-indices-source grid-center \
    --updates 4 --episodes-per-update 5 --horizon 1 --minibatch-size 5 --ppo-epochs 4 \
    --seed 123 --max-evaluations 20 \
    --output results/autockt_fair_headtohead_seed123.jsonl
```

| method | n | successes | failure breakdown | wall-clock | evals to first success |
|---|---:|---:|---|---:|---:|
| Random Search (reused, sec 19) | 20 | **1** | dc=15, ac=4 | not recorded | 9 |
| CEM, graded fitness, unbiased init | 20 | 0 | dc=16, ac=4 | 630s (unbiased reward_v1 run, sec 19 -- graded run's own per-row timing not separately re-measured, same candidate count/shape) | never |
| PPO, no warm start, grid-center, horizon=1 | 20 | **0** | **dc=20** | 530.8s (8.85 min) | never |

**CEM's graded-fitness fix had zero observable effect on this run.** All
20 candidates failed at `dc` (16) or `ac` (4) -- identical in shape to
sec 19's `reward_v1` run, and in fact the mean/std trajectory across all 4
iterations is **byte-identical** between the two runs (verified directly):
since every candidate in every generation tied at the fitness floor either
way (-100.0 for `reward_v1`, -10.0 for `graded`), Python's stable sort
picked the same elites from the same seed-generated action sequence in
both cases. The fix is correct and demonstrated working in isolation (11
unit tests distinguishing graded near-misses); it never activated in this
run because no candidate ever reached the `transient` stage even once.
This is itself an honest, useful finding: **the bottleneck for an
unbiased, budget-20 CEM run is not only reward_v1's flatness -- it is that
the feasible region is apparently too narrow to reach even once from a
wide, centered Gaussian at this sample size**, so no amount of gradient
information among (nonexistent, in this run) transient-stage candidates
could have helped.

**PPO found zero successes, and its failures were MORE concentrated than
either baseline** -- 100% `dc`-stage, versus Random Search's 75%/CEM's
80%. Inspecting the actual sampled points explains why: with
`randomize_initial_state` perturbing at most +/-1 grid index from the
center anchor, and `horizon=1` allowing exactly one further +/-1/+2
index-delta action from an untrained (near-random-weight) policy, every
one of PPO's 20 evaluated points was confined to a narrow neighborhood
around grid-center (`rload_ohm` in {630..1995} ohm, `itail_a` in
{1e-4, 2e-4} A -- vs. Design A's `itail_a=6.03e-4` A, roughly 4-6x higher).
The grid center itself sits in a region that appears **uniformly
DC-infeasible for every nearby grid point at this radius** -- the same
"less than 200mV rail headroom" failure mode documented since sec 7,
consistent with the DC gate's own dependence on `rload_ohm * itail_a`
(`receiver_search.py`'s own constraint-aware sampling heuristic estimates
`common_mode_v = supply_v - 0.5 * rload_ohm * itail_a`, and grid-center's
values sit close to that heuristic's own rejection boundary). Random
Search's full-space, single-shot uniform draws and CEM's first-generation
wide Gaussian can (and did/could) land far from this specific bad
neighborhood in one draw; PPO's inherently LOCAL, compounding-small-step
action mechanics, anchored at grid-center under a short horizon, cannot
reach out of it within this budget -- **not a policy-quality failure**
(the policy had only 4 gradient updates, effectively no time to learn
anything), a **reachability constraint** imposed by combining an unbiased
starting point with PPO's own step-local action structure and a
budget-matched short horizon.

**Both non-Random-Search methods also hit the identical flat-reward
problem PPO is exempt from investigating (per instruction) but not exempt
from experiencing**: with every PPO episode failing at `dc`,
`autockt_reward`'s own `FAILURE_REWARD = -1.0` was returned every single
time -- `mean_episode_reward` was exactly `-1.0` across all 4 updates, with
zero variance for PPO's own learning signal to exploit, structurally the
same "no gradient among failures" problem CEM's `reward_v1` has. This is
reported as observation, not modified -- `autockt_reward` is locked, and
per instruction changing it now would not be investigating a "concrete PPO
defect" this experiment identified, since the defect (if any) is in the
reachability of the starting neighborhood, not in the reward's shape.

### A-F, per the original comparison structure

**A. Result table**: see above -- Random Search 1/20, CEM 0/20, PPO 0/20.

**B. Evaluation efficiency**: Random Search: 9 evals to first success.
CEM and PPO: undefined (never succeeded).

**C. Wall-clock**: PPO 530.8s / CEM ~630s (sec 19 measurement, same
candidate shape) for the same n=20 -- both faster than Random Search's
typical ~60s/eval baseline because 100% of their candidates failed at the
cheap `dc`/`ac` stages rather than reaching the expensive transient stage.

**D. Best feasible design**: Random Search's Design A remains the only
feasible design found by any method in this section. Neither CEM nor PPO
produced a comparable design here.

**E. Limitations/confounds still present**:
1. n=1 seed -- this is one trial, not a distribution; a different seed
   could plausibly land PPO's or CEM's local neighborhood somewhere
   feasible by chance, the same way Random Search's seed=123 happened to
   include one lucky draw (candidate 9 of 20) among its 20.
2. PPO's reachable region under `horizon=1` +/- 1-index randomization is
   provably much smaller than Random Search/CEM's per-draw coverage --
   this was chosen deliberately for budget-matching and one-shot
   structural comparability, at the direct cost of search radius. A longer
   horizon (more sequential steps per episode, same total budget spread
   over fewer episodes) would give PPO's actual mechanism -- multi-step
   local refinement -- more room to demonstrate anything, at the cost of
   fewer independent restarts and reduced comparability to RS/CEM's
   one-shot structure. This experiment deliberately chose the latter
   trade-off for maximum structural fairness to RS/CEM; it is not the
   trade-off that gives PPO's own mechanism the most room to work.
3. 4 PPO gradient updates over 20 transitions is not enough data, by any
   standard on-policy RL rule of thumb, to produce a materially different
   policy from its random initialization -- this experiment measures
   PPO's UNTRAINED exploration behavior under a fair start, not its
   trained behavior.
4. Discretized grid (PPO) vs. continuous space (RS/CEM) remains
   unresolved, as in sec 19.

**F. Conclusion -- no overclaiming, per instruction**: **Neither PPO nor
CEM demonstrated any advantage over Random Search in this fair,
no-warm-start, budget-matched trial -- both found zero successes where
Random Search found one.** This does **not** show CEM or PPO are worse
optimizers in general; at n=1 seed and n=20 budget, with Random Search's
own success rate at just 5%, this comparison lacks the statistical power
to conclude anything about relative optimizer quality from the 0-vs-0 CEM/
PPO tie or the 1-vs-0 Random Search margin. What it DOES show, with direct
mechanistic evidence rather than assumption: (a) CEM's graded-fitness fix
is verified correct but was never exercised in this run, because the
search never reached a state with real gradient information -- a
reachability problem, not (solely) a reward-shape problem; (b) PPO's
apparent success in every EARLIER experiment in this repository is now
directly shown to depend heavily on its warm-started
`VERIFIED_INITIAL_PARAMETERS` starting point -- when that same
starting-point advantage is removed and PPO is confined to a fair,
unbiased, budget-matched local search, it could not find a single passing
design, for reasons traceable to its inherently local action mechanics
under a short horizon rather than to any flaw in its (locked, unmodified)
learning algorithm. **PPO does not have a demonstrated practical advantage
here; if anything, this specific trial shows its structural dependence on
a good starting point more starkly than sec 19 did.**

## 21. Official project-brief audit -- gap analysis before further SPICE spend

An official project slide/brief was provided mid-session, re-stating the
actual required deliverable: *"A Reinforcement Learning based Python
framework that takes target specs as input, seamlessly integrates with a
SPICE simulator, and outputs the final schematic and resulting specs,"*
targeting a PCIe Gen-2 receiver (5 Gbps / 2.5 GHz Nyquist, source-degenerated
CTLE with 3-12 dB peaking around 1.25-2.5 GHz, 1-tap DFE, HD3 < -30 dB,
input-referred noise < 1.5 mV, power < 15 mW, area < 0.05 mm^2, eye opening
> 0.4 UI with a voltage requirement, PVT: TT/SS/FF, VDD +/-5%, 0-125C). Per
explicit instruction, this section audits sections 1-20's work against that
brief BEFORE spending further SPICE, rather than assuming the existing
4-spec `TargetSpec` was ever a full substitute for it.

### A. Specs actually measured by the simulator

| official spec | measured? | metric(s) | gate |
|---|---|---|---|
| CTLE peaking 3-12 dB, ~1.25-2.5 GHz | **yes** | `peaking_db`, computed specifically within `1.25e9 <= f <= 2.5e9` (`simulator/waveform.py::ac_metrics`) | hard-gated 3-12 dB |
| out-of-band peak control | yes (implicit) | `out_of_band_excess_peak_db` | gated <=3 dB excess |
| HD3 < -30 dB | **yes** | `hd3_db` / `hd3_100mv_db` | hard-gated < -30 dB |
| input-referred noise < 1.5 mV | **yes** | `input_referred_noise_vrms` | hard-gated < 1.5 mVrms |
| power < 15 mW | **yes** | `ctle_power_w` / `transient_average_power_w` | hard-gated (0, 15 mW) |
| eye width > 0.4 UI | **yes** | `dfe_eye_width_ui` | gated at CANDIDATE+ fidelity |
| eye height (voltage) | **partially** | `dfe_locked_phase_eye_height_v`, real volts | current gate 0.1 V -- **the slide's exact mV number was not independently confirmed against this value; flagged, not assumed equal** |
| area < 0.05 mm^2 | **no** | zero references anywhere in the repo | **not measured at all** |
| PVT: TT/SS/FF, VDD+/-5%, 0-125C | **infrastructure existed, never exercised (until this section)** | `PVT_GRID` (60 conditions: TT/SS/FF/SF/FS x 3 VDD x 4 temps), `evaluate_pvt_grid()` | see sec F/G below |
| 5 Gbps / 2.5 GHz Nyquist | **yes** | stimulus is literally 5 GT/s NRZ (`docs/stage1-decisions.md`) -- already PCIe-Gen2-matched by design | -- |
| CTLE w/ source degeneration | **yes** | `circuits/blocks/ctle.spice`, real SKY130 transistor-level RDEG/CDEG-degenerated diff pair | -- |
| 1-tap DFE | **behavioral only** | Python-side decision-feedback correction, deliberate per `docs/stage1-decisions.md` -- **no netlist/transistor representation** | -- |

### B. What is actually in `TargetSpec`/the RL reward

Exactly 4 of the above: `dfe_locked_phase_eye_height_v`, `dfe_eye_width_ui`,
`dfe_min_margin_v`, `ctle_power_w` (`rl/target_spec.py::EXISTING_THRESHOLDS`,
unchanged since sec 4). **This has been, all session, an intentionally
reduced baseline, not a claimed substitute for the official spec set** --
disclosed here explicitly, per instruction, rather than left implicit.

### C. Indirectly represented

HD3, noise, and the peaking band ARE measured and hard-gated by the
simulator itself -- a design that fails any of them gets
`evaluation.success=False`, which does feed PPO's binary success/failure
and Random Search/CEM's `reward_v1`. **None of the three have graded
reward shaping** -- PPO gets zero gradient signal about how close it is on
HD3/noise/peaking, only pass/fail via the success gate. They influence
training only as an undifferentiated part of "did this fail at all," not
as targets the agent is steered toward improving.

### D. Completely missing

**Area.** No measurement, no model, no proxy anywhere in the codebase.

### E. Final schematic vs. parameter list

**Was: parameter list only, confirmed by tracing the code.**
`simulator/ngspice.py::run_simulation` builds every netlist inside a
`tempfile.TemporaryDirectory` that auto-deletes on exit -- every simulation
this entire project has ever run generated a real, complete netlist and
then destroyed it. Every "final output" (Random Search's Design A, PPO's
`best_parameters`, CEM's best candidate) has only ever been a JSON dict of
5 numbers.

**Fixed this section**: `experiments/export_final_schematic.py` (new,
11 unit tests, no simulator changes). Reuses `circuits/blocks/ctle.spice`
(the real, unmodified CTLE subcircuit every evaluation already uses) and
`simulator.ngspice.parameterize_netlist` (the same substitution mechanism
every real evaluation already uses) to render a permanent, real SPICE
netlist with a design's final component values set as concrete `.param`
defaults, `.include`-ing the unmodified block file by absolute path, with a
header documenting achieved specs, target, source provenance, git commit,
and resolved SKY130 model path. **A real correctness bug was caught before
use**: `ctle.spice` declares its own defaults via `.subckt ... params:`
continuation syntax, not standalone `.param NAME=value` lines --
`parameterize_netlist`'s regex only matches the latter (confirmed by direct
test: calling it on `ctle.spice`'s own source raises `KeyError`). Every
real bench file instead declares a bench-level `.param` line and
instantiates `XCTLE` with explicit `RLOAD={RLOAD}`-style references --
substitution happens at the bench's `.param` line, not inside the
subcircuit file. The exporter reproduces that exact, already-proven
mechanism rather than the broken direct approach. **Not yet done**:
rendering the full channel-driven receiver-level bench
(`circuits/benches/receiver_transient.cir`) as a single self-contained
file -- its NRZ stimulus is synthesized dynamically per-run by
`receiver.py`-private code, not a static template; out of scope for this
pass, noted in the exported file's own header.

A real schematic was exported for Design A using only existing data (zero
new SPICE): `results/schematics/design_a_final_schematic.spice`.

### F. PVT infrastructure

Substantial and, until this section, completely idle: `PVT_GRID` (60
points, already exceeds the brief's TT/SS/FF by also including SF/FS),
`evaluate_pvt_grid()` (runs an arbitrary PVT subset at any fidelity, with
an early-stop option) -- fully implemented, exported, but **zero calls
anywhere in any test, experiment, or script before this section**. Before
trusting it with real SPICE budget, `tests/test_evaluate_pvt_grid.py` (6
new tests, mocking `simulator.receiver.evaluate_receiver` at the module
level -- the same technique `tests/test_ngspice.py` already uses, since
`evaluate_pvt_grid` has no injectable-evaluator parameter) verified its
control flow: iterates every condition in order, `stop_on_failure` halts
immediately and correctly, default fidelity is FINAL, default conditions
is the real unmodified `PVT_GRID`. No bugs found, but also never
previously exercised even once -- worth confirming before spending real
budget on it, which is exactly what those tests did.

### G. Minimum real-SPICE PVT experiment, ~10 days remaining

`experiments/pvt_sweep.py` (new, 5 unit tests): a reduced,
minimum-viable 27-point subset (`MINIMAL_27_CONDITIONS`) -- TT/SS/FF (not
the full grid's SF/FS) x {1.71, 1.8, 1.89} V x {0, 27, 125} C (3 of
`PVT_GRID`'s own 4 temperatures, spanning the brief's stated 0-125C range
plus nominal) -- run on Design A via `evaluate_pvt_grid` at FINAL fidelity
(the function's own default, the strictest gate set).

**Result: 23/27 (85.2%) pass.** 121.8 minutes total wall-clock (real,
directly measured -- FINAL fidelity runs a materially fuller pipeline than
the ~63s/eval TRAINING-fidelity baseline sec 7 measured; observed range
here was 172-677s/point).

| corner | pass | fail | detail |
|---|---:|---:|---|
| TT | 9/9 | 0 | clean across all 3 VDD x 3 temp points |
| SS | 9/9 | 0 | clean across all 3 VDD x 3 temp points |
| FF | 5/9 | 4 | fails at 1.71V/125C, and at ALL THREE temps of 1.80V; passes at all three temps of 1.89V |

Every one of the 4 failures is at the `transient` stage specifically (DC,
AC, CTLE-transient, noise, and HD3 all still pass at every failing point --
only the full-receiver transient's own gates, e.g. `dfe_min_margin_v > 0`,
trip). The failure pattern is not random: it clusters at FF combined with
LOWER-to-nominal supply (1.71-1.80V), and resolves at the highest tested
supply (1.89V) across all three temperatures -- consistent with a real,
physically plausible mechanism (FF's higher transconductance/lower
threshold interacting with reduced supply headroom), not simulation noise.
**Design A -- found by uniform Random Search at a single nominal
TT/1.8V/27C condition, never PVT-aware in any way -- is robust across 2 of
3 official corners entirely, and robust at high-VDD FF, but has a genuine,
now-quantified FF/low-VDD weakness this project had never previously
measured.** The full 60-point grid (`--condition-set full60`, including
SF/FS and the 75C point) remains available and is feasible later in the
10-day window if a finer picture of the FF failure boundary is wanted.

### H. Do R/C/bias/DFE parameters map to a real schematic?

**Partially, confirmed by direct inspection of
`ReceiverParameters.spice_parameters()`.** `rload_ohm`, `rdeg_ohm`,
`cdeg_f`, `itail_a` map directly and exactly to `ctle.spice`'s own
subcircuit parameters (RLOAD/RDEG/CDEG/ITAIL_VAL) -- a clean 1:1
correspondence to a real transistor-level circuit, independently confirmed
by section E's schematic export actually working. **`dfe_tap_v` does not
appear in `spice_parameters()`'s output dict at all** -- it never reaches
any netlist; it is consumed only by the Python-side DFE algorithm. 4 of 5
parameters are real circuit values; the DFE tap is algorithmic only,
consistent with finding D.

### Re-priority acknowledged

Per instruction, the re-priority order (1: end-to-end pipeline -> final
schematic, 2: official spec coverage, 3: PVT robustness, 4: RS/CEM
comparison, 5: multiple trade-off designs, 6: optional MA-Opt/surrogate,
7: bonus LLM wrapper) supersedes further PPO tuning. Section 20 already
established no PPO defect was found that would justify further PPO
changes; this section's work (schematic export, PVT infrastructure
verification and first real exercise) directly targets re-priority items
1 and 3. Item 2 (official spec coverage in the reward/TargetSpec itself)
remains open -- expanding `TargetSpec`/`autockt_reward` to grade HD3/
noise/peaking/area would touch files this session has treated as locked
(`rl/target_spec.py`, `rl/autockt_reward.py`) and was not undertaken
without explicit instruction to unlock them; documented here as the
concrete next decision point rather than silently left undone.

## 22. Overnight autonomous session -- PVT diagnosis, area, multiple designs, PVT-aware selection

Explicit overnight authorization: full autonomy inside the NEBULA
repository only, no PPO/simulator formulation changes without a concrete
defect, no overwriting historical result files, new experiments to new
filenames, logical git commits. Four tasks, worked in order.

### Task 2: area measurement

**Motivation**: sec 21 finding D -- area was completely unmeasured.

**Inspection, strictly within the repository** (the SKY130 PDK's own cache
under `~/.ciel/...` is outside the project directory and was not read, per
the session's explicit access boundary -- the microns convention below
relies on public SKY130 PDK documentation, not on inspecting that cache).
Grepped every `.spice`/`.cir` file in `circuits/`: `W=10 L=0.15` for both
CTLE transistors (`sky130_fd_pr__nfet_01v8`) is IDENTICAL across every
circuit file in the repository -- fixed, never varied by any experiment
this project has run. `L=0.15` matching SKY130's documented 150nm minimum
channel length is only physically sensible under a microns reading (0.15
meters would be absurd for a transistor channel). RLOAD/RDEG are ideal
SPICE `R` elements and CDEG an ideal `C` element in
`circuits/blocks/ctle.spice` -- not physical SKY130 resistor/capacitor
primitives -- so no sheet-resistance or capacitance-per-area value exists
anywhere in this project for them.

**Implemented**: `analysis/area_estimate.py` (9 tests) -- parses the real,
unmodified `circuits/blocks/ctle.spice` (not hardcoded literals) for its
transistor W/L values and computes ONLY the raw channel area
(2 x 10um x 0.15um = **3 um^2 = 3e-6 mm^2**). Explicitly, permanently
labeled a PARTIAL measurement -- `AreaEstimate.total_area_computable` is
always `False`, and `missing_components` documents exactly what is absent
(resistor area, capacitor area, transistor layout overhead beyond raw
channel). **No PASS/FAIL verdict is claimed against the 0.05 mm^2 official
budget** -- 3e-6 mm^2 is six orders of magnitude below budget but is not a
credible stand-in for total circuit area, which is almost certainly
dominated by the resistors and capacitor. Integrated into
`experiments/export_final_schematic.py`'s header (exposed in the final
design output, per Task 2's requirement) with the same explicit caveat.
Design A's schematic was regenerated with this addition as a NEW file
(`results/schematics/design_a_final_schematic_v2_with_area.spice`) --
the original export was not overwritten.

**Not done, and honestly reported as missing rather than fabricated**:
total circuit area. Would require either a real SKY130 resistor/capacitor
primitive replacing the ideal `R`/`C` elements (a circuit-topology change,
out of the smallest-defensible-implementation scope given here) or an
externally-supplied sheet-resistance/capacitance-density assumption this
project has never established. Flagged as the concrete next step, not
silently skipped.

### Task 3: multiple feasible designs / trade-offs

**Motivation**: the synopsis requires multiple feasible trade-off designs,
not only one "best" design.

**Inspection**: scanned existing results files for genuinely feasible,
fully-metric-verified, distinct designs (zero new SPICE). Found, beyond
Design A itself: `results/rl_reward_directed_smoke.jsonl` (2 rows, both
`success=true`, metrics recovered from the logged `constraints` tuple --
`simulator.rl_adapter.constraints_from_evaluation`'s own shape, not
fabricated) and `results/rc_counterfactual_sweep_mixed_target.jsonl` (7 of
18 counterfactual candidates have `spec_satisfied=true`, each differing
from that sweep's own baseline on exactly one parameter, each independently
real-SPICE-verified with full metrics). PPO's own training-log successes
(`autockt_mixed_target_confirmation.jsonl` etc.) were NOT included -- per
sec 16's KNOWN GAP, those logs never persisted per-step `metrics`, so no
measured trade-off ranking is possible for them without fabricating values;
excluded rather than guessed.

**Implemented**: `analysis/design_catalog.py` (12 tests) -- loads all of
the above, deduplicates by rounded parameter tuple, and assigns trade-off
labels ONLY when a design's own measured metric is the pool's actual best
(or tied-best) on that axis -- `lowest_power`, `strongest_eye_height`,
`widest_eye`, `largest_margin`, or `balanced` if it wins nothing. Output:
`results/feasible_design_catalog.jsonl`, **8 unique feasible designs**
after deduplication (from 10 loaded, 2 exact duplicates dropped).

| design_id | trade-off label(s) |
|---|---|
| design_a | balanced |
| reward_directed_smoke_ep1_step1 | lowest_power |
| reward_directed_smoke_ep1_step2 | balanced |
| rc_sweep_rload_ohm_offset-4 | balanced |
| rc_sweep_rdeg_ohm_offset+1/+2/+4 | balanced |
| rc_sweep_cdeg_f_offset+1 | strongest_eye_height, widest_eye, largest_margin |

**Not done**: no new targeted search was run to generate additional
candidates beyond what already existed -- 8 genuinely distinct, real,
measured designs was judged sufficient to demonstrate the framework per
"avoid expensive full retraining" / minimum-useful-framework instruction.
The framework (`load_all_known_feasible_designs`, `deduplicate`,
`rank_by_measured_trade_offs`) is reusable the moment more feasible designs
exist (e.g. from a future, larger search).

### Task 4: PVT-aware candidate selection

**Motivation**: make PVT part of the design-selection pipeline, not only a
post-hoc experiment.

**Implemented**: `analysis/pvt_selection.py` (8 tests) -- implements the
full conceptual flow: candidates (`design_catalog`) -> nominal screening
(already satisfied by catalog membership) -> PVT evaluation
(`run_pvt_evaluation`, a thin wrapper around the existing, unmodified
`evaluate_pvt_grid` -- spends real SPICE only when a caller explicitly
invokes it against an explicitly given, small condition set; never
automatic for "every random candidate," per instruction) -> robustness
score (`summarize_pvt_results`: pass rate, worst-case failing conditions)
-> `rank_by_robustness` -> `select_final_designs` (returns the top-N
meeting a minimum pass rate, or the best available if none do -- never
silently empty).

**Demonstrated with zero new SPICE**: `load_pvt_results_from_jsonl`
summarizes the REAL, already-computed
`results/design_a_pvt_minimal27.jsonl` --

```
design_id: design_a
n_conditions: 27, n_passing: 23, pass_rate: 0.852
worst_case_conditions: 4x FF corner (see Task 1 below), all failed_stage=transient
selected: design_a (pass_rate 0.852, met_minimum_pass_rate(1.0)=False)
```

-- correctly reproducing sec 21's 23/27 finding and correctly reporting
that Design A does NOT meet a strict 100%-pass-rate bar, exactly the
"don't fabricate a PASS" behavior `select_final_designs` is designed to
guarantee.

### Task 1: PVT robustness improvement -- read-only diagnosis first, unexpected finding

**Per instruction, read-only diagnosis before any local search.**
`experiments/pvt_diagnose.py` (4 tests) re-evaluates a design at explicit
PVT conditions and captures FULL per-stage detail (violations, errors,
failure_code) that `results/design_a_pvt_minimal27.jsonl`'s rows do not
carry (only top-level `metrics`) -- necessary since the 4 original failures
were all missing `dfe_locked_phase_eye_height_v`/`width`/`margin`/
`error_count` entirely (not "measured but below threshold" -- never
computed at all), pointing at a transient-stage simulation issue rather
than a marginal violation.

**Result of the diagnosis run (4 real evaluations, exact same design,
exact same 4 conditions that failed in sec 21's sweep):
`results/design_a_pvt_failure_diagnosis.jsonl` -- ALL FOUR SUCCEEDED.**
`dfe_min_margin_v` at all four: 0.51-0.54 V -- comfortably positive, not a
borderline near-miss. This is the opposite of what a genuine, reproducible
design weakness at these corners would look like.

**This raises the real possibility that the original 4 failures were
simulation-level non-reproducibility (e.g. transient convergence
sensitivity), not a stable design defect** -- consistent with the original
sweep's own wall-clock data: those 4 points took 288.5-677.0s, 1.5-4x
longer than surrounding passing points (~170-300s) in the same run,
suggestive of retry/convergence difficulty rather than a clean threshold
violation.

**Confirmatory re-check**: extended `experiments/pvt_sweep.py` with
`--condition-set custom --custom-conditions CORNER:VDD:TEMP ...` (7 new
tests) to re-run the exact same 4 conditions through the IDENTICAL
`evaluate_pvt_grid` code path the original sweep used (not the simpler
direct-`evaluate_receiver` path `pvt_diagnose.py` uses), to rule out any
script-level discrepancy before concluding non-reproducibility.
`results/design_a_pvt_failure_reproduction_check.jsonl`: **all 4/4
succeeded again** (302.8s, 304.7s, 306.9s, 306.1s -- consistent,
unremarkable timing, unlike the original run's erratic 288-677s spread).

**Final confirmatory step: complete 27-point grid re-run, Design A
completely unchanged.**
`results/design_a_pvt_minimal27_rerun.jsonl` (new file; the original
`results/design_a_pvt_minimal27.jsonl` was NOT overwritten and remains on
disk as the historical first-run record) -- **27/27 (100%) pass**, 138.4
minutes wall-clock, including every one of the 4 originally-failing
conditions (FF at 1.71V/125C and all three temperatures of 1.80V), all now
passing cleanly.

### Task 1 conclusion

**No R/C/bias parameter change was made or needed.** The evidence across
three independent re-evaluations (single-point diagnosis with full
per-stage detail, a 4-point reproduction check via the exact original code
path, and a complete 27-point re-run) is unanimous and conclusive: **the
original 23/27 result undersold Design A's actual robustness.** The 4
original failures are best explained by transient-stage simulation-level
non-reproducibility (consistent with their anomalously long, inconsistent
original wall-clock times suggesting retry/convergence difficulty) at
those specific corner/VDD/temperature combinations, not a stable,
reproducible design weakness. Design A -- found by uniform Random Search
at a single nominal condition, never PVT-aware, never modified by this
investigation -- is **robust across the entire tested 27-point PVT space**
on this clean re-run.

**Both results are preserved, not silently replaced**: the original
23/27 (`results/design_a_pvt_minimal27.jsonl`) remains on disk as the
historical record that motivated this investigation; the 27/27 re-run
(`results/design_a_pvt_minimal27_rerun.jsonl`) is the current,
most-verified characterization of Design A's PVT robustness.
`analysis/pvt_selection.py` (Task 4) should be pointed at the rerun file
for any downstream selection decision, not the original.

Per instruction ("do not blindly search forever," "if 27/27 is achieved:
freeze the robust candidate, preserve Design A as baseline, document
before/after"): **the robust candidate IS Design A itself, unchanged** --
there is no "before/after design" distinction to draw, only a
"before/after measurement" one. No local R/C search around Design A was
run, because the diagnosis step (required before any search, per
instruction) found no reproducible defect to search a fix for.

## 23. NEXT IMPLEMENTATION CHUNK -- pipeline, final spec, PVT integration, learning evidence, benchmark

Five tasks, executed in the reprioritized order (pipeline first, per a
mid-chunk instruction, before returning to the final-spec/PVT/evidence/
benchmark tasks). No PPO/simulator formulation change; no new real-SPICE
experiment was run anywhere in this chunk -- every module here is either
SPICE-free orchestration/consolidation over already-existing
`results/*.jsonl` files, or (the pipeline's own tests) exercised with
`--backend synthetic`.

### Task 1: clean end-to-end pipeline

`experiments/run_autockt_pipeline.py` -- wires `TargetSpec -> validate ->
PPO deterministic rollout (real or synthetic backend, reusing
`AutoCktReceiverEnv`/`PPOAgent`/`ReceiverRLAdapter` unmodified) ->
`filter_nominal_feasible` -> PVT-aware `select_final_design` -> schematic
export (`experiments.export_final_schematic`, unmodified) -> final
specification report (`analysis.final_specification`, this chunk)` as one
callable `run_pipeline()`, not disconnected scripts. 13 orchestration
tests (`tests/test_run_autockt_pipeline.py`), including one full
`--backend synthetic` CLI dry-run producing a real schematic + spec table
with zero SPICE. Refuses to overwrite an existing `--output` file, same
convention as every other experiment script in this repo.

### Task 2: final specification table

`analysis/final_specification.py` -- one authoritative report per design:
for every official-brief metric this project actually measures somewhere,
a `PASS`/`FAIL`/`NOT CLAIMED` row with the measured value, the required
threshold, and the exact source file the number came from -- never
inventing a missing measurement. Real output regenerated for Design A,
`results/design_a_final_specification.json`, now including the real
27/27 PVT result (`results/design_a_pvt_minimal27_rerun.jsonl`, sec 22
Task 1):

| Metric | Measured | Requirement | Verdict |
|---|---:|---|---|
| Eye width (UI) | 0.87 | > 0.4 UI | PASS |
| Eye height (V) | 1.555 | > 0.1 V (this repo's own threshold, not independently re-verified against the slide's mV figure) | PASS |
| Margin (V) | 0.525 | > 0 V | PASS |
| Power (W) | 0.001085 | < 0.015 W | PASS |
| Peaking (dB) | 7.07 | 3-12 dB | PASS |
| HD3 (dB) | -79.93 | < -30 dB | PASS |
| Input-referred noise (Vrms) | 0.000361 | < 0.0015 Vrms | PASS |
| Transistor channel area (mm^2) | 3e-6 | < 0.05 mm^2 -- PARTIAL (channel area only) | NOT CLAIMED |
| PVT (pass/total) | 27/27 | TT/SS/FF x VDD+/-5% x 0-125C | PASS |

Design A clears every independently measurable spec; area remains
honestly unclaimed (sec 22 Task 2 -- no resistor/capacitor/layout area
model exists in this project).

### Task 3: PVT-aware candidate selection -- integrated, not just built

sec 22 Task 4 built the ranking logic
(`analysis/pvt_selection.py::select_with_trade_off_preference`, the
documented 4-level priority: nominal feasibility -> PVT pass rate ->
robustness tie-break -> trade-off preference among genuine ties) but had
not yet wired it INTO the pipeline. Two gaps found and closed this chunk:

1. `run_autockt_pipeline.py::select_final_design`'s PVT branch was still
   calling `rank_by_robustness` directly (priorities 1-3 only), never
   reaching priority 4. Fixed: it now calls
   `select_with_trade_off_preference` with a new, additive
   `trade_off_preference` parameter (default `"most_robust"`, unchanged
   behavior for existing callers). A new test constructs a genuine PVT
   tie between two candidates differing only in `ctle_power_w` and
   confirms `"lowest_power"` and `"most_robust"` select different designs
   from the same tie.
2. `run_pipeline` computed a real PVT result (real SPICE, when
   `pvt_conditions` was supplied) but then passed `pvt_result=None` into
   `build_final_specification_report` -- silently downgrading the final
   spec's PVT row to `"NOT CLAIMED"` even though real per-condition data
   existed one call frame away. Fixed with `_pvt_result_from_selection`,
   which reconstructs the already-computed `PVTRobustnessResult` from
   `select_final_design`'s own JSON-serializable summary -- no second PVT
   run. Verified with an integration test (mocked `evaluate_pvt_grid`,
   synthetic-backend candidate generation) confirming the final spec's
   PVT row shows the real `2/2 PASS`, not `NOT CLAIMED`.

### Task 4: strong learning/optimization evidence

`analysis/learning_evidence.py` -- an independent, from-scratch
recomputation (not a copy of prior narrative numbers) over the raw
per-step rows of three already-existing training logs, reproducing sec
15/17/18 exactly: matched-checkpoint initial-vs-final satisfaction/reward
(single-target-hard: 0.667->1.0 satisfaction, 5.843->10.0 reward;
mixed-target: 0.333->1.0, 1.509->10.0), the controlled unseen-target
generalization win/tie/loss breakdown (1 win / 6 ties / 3 losses --
correctly still reported as a REGRESSION, not an improvement), and the
exact per-update mean-reward sequence from the mixed-target run (2.833,
5.333, 2.333, 4.958, 4.667, 2.961 -- explicitly non-monotonic, with a
regression test guarding against ever asserting a smooth curve).
Parameter-behavior evidence (trained-vs-untrained action distributions,
R/C counterfactual sweep) is deliberately left as a pointer to the
already-existing, already-tested `analysis/policy_inspection.py` and sec
17's sweep results, not duplicated.

### Task 5: honest RS/CEM/PPO benchmark

`analysis/benchmark_report.py` -- consolidates sec 19 (warm-started PPO)
and sec 20 (no-warm-start, budget-matched head-to-head) into one tested,
queryable artifact (`results/benchmark_report.json`), reusing
`analysis/fair_comparison.py`'s existing summarizers (computes nothing
new). Each trial explicitly states its evaluation budget, initialization,
search space, sequential-vs-one-shot nature, success criterion, and
whether it is genuinely comparable across methods:

| trial | comparable | RS | CEM | PPO |
|---|---|---:|---:|---:|
| warm_started_ppo (sec 19) | **No** (PPO inherits RS's own best point; budget not separable from multi-target training) | 1/20 | 0/20 | 13/65 (per-step, native) |
| no_warm_start_headtohead (sec 20) | **Yes**, but n=1 seed only | 1/20 | 0/20 | 0/20 |

Conclusion text is unchanged from sec 20's own finding: **no demonstrated
PPO advantage over Random Search once the warm-start confound is
removed** -- this module does not overturn or restate that conclusion
differently, only makes it queryable/testable.

### Regression and commits

261/261 SPICE-free tests passing (`tests/run_fast_suite.py`), up from 241
at the start of this chunk. Six commits: `4c37bc8` (pipeline + final
spec), `1145e25` (trade-off-preference ranking logic), `96a5f8a`
(learning evidence), `8af566f` (benchmark report), `74fdc03` (PVT
trade-off wiring into the pipeline), `0ae89ff` (PVT-result-into-final-spec
wiring fix). Not yet pushed to `origin/main` as of this section (no push
instruction given since the prior push, `209993f`).

### What remains (per the chunk's own stopping rule)

Every stopping-rule checklist item is satisfied: pipeline, final spec,
PVT-aware selection integrated, learning evidence, RS/CEM/PPO benchmark,
tests passing, documentation updated (this section). Explicitly NOT
started, per the stopping rule: TD learning, actor-critic redesign,
MA-Opt, surrogate RL, another PPO sweep, random hyperparameter search.
Known, disclosed, unresolved limitations carried forward unchanged from
earlier sections: PPO training-log wall-clock predates instrumentation
for two of the three checkpoint-comparison files (sec 16); area is
channel-only, not total circuit area (sec 22 Task 2); the RS/CEM/PPO
head-to-head is a single seed, not a distribution (sec 20 E); `dfe_tap_v`
remains behavioral, not a real SPICE element (sec 21).

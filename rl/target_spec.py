"""Target-specification abstraction. [NEBULA ADAPTATION]

No `TargetSpec`-like object exists anywhere in the NEBULA repo prior to this
file (confirmed by grep across simulator/ and experiments/ during the
inspection phase); AutoCkt's own equivalent is a YAML `target_specs` block
sampled by `autockt/gen_specs.py` (github.com/ksettaluri6/AutoCkt), not a
Python dataclass. This file's *existence and shape* are therefore a NEBULA
addition, not a port of an AutoCkt source file. The *sampling methodology*
built on top of it (uniform per-dimension sampling into a fixed pool, see
`sample_target_pool`) does mirror AutoCkt's gen_specs.py and is labeled
[AUTOCKT-REPLICATED] at the point of use below.

SPEC_NAMES / SPEC_DIRECTIONS: [NEBULA-SPECIFIC ADAPTATION]
AutoCkt's four specs split 3 "larger is better" (gain_min, ugbw_min,
phm_min) + 1 "smaller is better" (ibias_max) -- see
ngspice_vanilla_opamp.py::reward(). NEBULA's analogous 3+1 split is chosen
from the ten metrics already in simulator.rl_adapter.METRIC_OBSERVATION_NAMES:

    larger-is-better: dfe_locked_phase_eye_height_v, dfe_eye_width_ui, dfe_min_margin_v
    smaller-is-better: ctle_power_w

`peaking_db` is deliberately excluded even though it's in the existing
observation vector: simulator/receiver.py's own AC-stage gate treats it as a
banded target (3 dB - 12 dB required; simulator/rl_adapter.py's reward_v1
scores it as optimal at exactly 6 dB, `-abs(peaking - 6.0)`), which is not a
monotone larger/smaller-is-better quantity and does not fit AutoCkt's
purely-monotone reward shape (see rl/autockt_reward.py). This is a real,
disclosed scope reduction, not an oversight.

EXISTING_THRESHOLDS: [values reused verbatim from the existing repo, not invented]
    dfe_locked_phase_eye_height_v >= 0.1    simulator/rl_adapter.py:120 (`height - 0.1`,
                                              CONSTRAINT_NAMES "eye_height_over_100mv")
                                              simulator/receiver.py _transient_violations,
                                              CANDIDATE-fidelity gate `<= 0.1`
    dfe_eye_width_ui              >= 0.4    simulator/rl_adapter.py:121 (`width - 0.4`,
                                              CONSTRAINT_NAMES "eye_width_over_0p4ui")
                                              simulator/receiver.py _transient_violations,
                                              CANDIDATE-fidelity gate `<= 0.4`
    dfe_min_margin_v              >  0.0    simulator/rl_adapter.py:119 (`margin`,
                                              CONSTRAINT_NAMES "positive_margin")
                                              simulator/receiver.py _transient_violations,
                                              TRAINING+-fidelity gate `<= 0`
    ctle_power_w                  <= 0.015  simulator/rl_adapter.py:122 (`0.015 - power`,
                                              CONSTRAINT_NAMES "power_under_15mw")
                                              simulator/receiver.py DC/transient violations,
                                              "(0, 15 mW)"
Both citations for each row were independently confirmed during repository
inspection (rl_adapter.py's RL-facing constraint AND receiver.py's own
simulation-stage hard gate encode the identical number), which is why these
four numbers -- and no others -- were chosen as the reused thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Mapping

SPEC_NAMES: tuple[str, ...] = (
    "dfe_locked_phase_eye_height_v",
    "dfe_eye_width_ui",
    "dfe_min_margin_v",
    "ctle_power_w",
)

# "larger_is_better" == AutoCkt's "*_min" specs; "smaller_is_better" == AutoCkt's "*_max" specs.
SPEC_DIRECTIONS: Mapping[str, str] = {
    "dfe_locked_phase_eye_height_v": "larger_is_better",
    "dfe_eye_width_ui": "larger_is_better",
    "dfe_min_margin_v": "larger_is_better",
    "ctle_power_w": "smaller_is_better",
}

EXISTING_THRESHOLDS: Mapping[str, float] = {
    "dfe_locked_phase_eye_height_v": 0.1,
    "dfe_eye_width_ui": 0.4,
    "dfe_min_margin_v": 0.0,
    "ctle_power_w": 0.015,
}

# [NEBULA ADAPTATION] AutoCkt's `global_g` is a fixed reference vector used
# only to normalize the *target* itself in the state (see rl/autockt_state.py,
# `target_reference_vector`); its numeric contents are not specified anywhere
# retrievable in the official repo beyond "a fixed reference." Reusing
# EXISTING_THRESHOLDS here (rather than inventing a separate reference) keeps
# a single, disclosed source of truth grounded in the repo's own gates.
GLOBAL_REFERENCE: Mapping[str, float] = EXISTING_THRESHOLDS


@dataclass(frozen=True)
class TargetSpec:
    """One episode's goal vector."""

    dfe_locked_phase_eye_height_v: float
    dfe_eye_width_ui: float
    dfe_min_margin_v: float
    ctle_power_w: float

    def as_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in SPEC_NAMES}

    @classmethod
    def from_existing_thresholds(cls) -> "TargetSpec":
        """The single, deliberately-achievable smoke-test target.

        Uses EXISTING_THRESHOLDS directly as the goal -- no invented margin
        above them. Verified achievable by a real successful design already
        on record: results/receiver_random_search_20_seed123.jsonl,
        candidate_index 8 (rload_ohm=2342.47, rdeg_ohm=822.36,
        cdeg_f~1.0e-12, itail_a~6.03e-4, dfe_tap_v~-0.0111), re-confirmed by
        a fresh SPICE re-evaluation in this session (reward_v1=100.0,
        dfe_locked_phase_eye_height_v=1.522 > 0.1,
        dfe_eye_width_ui=0.87 > 0.4, dfe_min_margin_v=0.517 > 0.0,
        ctle_power_w=0.001085 < 0.015). See docs/autockt-mapping.md, "SPICE
        timing measurement", for the full re-evaluation record.
        """

        return cls(**EXISTING_THRESHOLDS)

    @classmethod
    def from_hard_target(cls) -> "TargetSpec":
        """A deliberately harder, but still real-design-verified, target.

        See HARD_TARGET_THRESHOLDS below and docs/autockt-mapping.md,
        section 14, for the full derivation and provenance. Unlike
        from_existing_thresholds(), this target is NOT satisfied by every
        plausible successful design -- it is satisfied by
        results/receiver_random_search_20_seed123.jsonl candidate_index 8
        but NOT by results/rl_reward_directed_smoke.jsonl's design, which is
        the intended discriminating property.
        """

        return cls(**HARD_TARGET_THRESHOLDS)


# [NEBULA ADAPTATION] a second, deliberately harder target, added alongside
# (not replacing) EXISTING_THRESHOLDS/from_existing_thresholds() above. The
# original trivial target sits at the simulator's own minimum pass/fail
# gates, which the fixed initial design clears by 2x-15x on every
# dimension -- giving PPO no discriminating signal (see
# docs/autockt-mapping.md, section 14, for the full analysis). These four
# numbers are not invented: height/margin are tightened to a level that a
# real, SPICE-confirmed successful design clears
# (results/receiver_random_search_20_seed123.jsonl candidate_index 8:
# height=1.5215V, margin=0.5174V) while a second, independent real
# SPICE-confirmed success (results/rl_reward_directed_smoke.jsonl,
# reward_v1=70.656: height=0.5514V, margin=0.2295V) does NOT clear them --
# proving the target is both achievable and genuinely selective, not
# extrapolated beyond demonstrated behavior of the fixed topology. Width is
# tightened more modestly (both known designs clear it); power is left at
# the existing 0.015W ceiling since both known designs already use <1.1mW,
# so tightening it further on sparse evidence would risk an infeasible
# joint target.
HARD_TARGET_THRESHOLDS: Mapping[str, float] = {
    "dfe_locked_phase_eye_height_v": 0.8,
    "dfe_eye_width_ui": 0.6,
    "dfe_min_margin_v": 0.35,
    "ctle_power_w": 0.015,
}


def sample_target_pool(
    count: int,
    *,
    seed: int,
    ranges: Mapping[str, tuple[float, float]],
) -> tuple[TargetSpec, ...]:
    """[AUTOCKT-REPLICATED sampling methodology] uniform per-dimension
    sampling of `count` targets, mirroring autockt/gen_specs.py:

        list_val = [random.uniform(float(spec[0]), float(spec[1])) for x in range(num_specs)]

    `ranges` must be supplied by the caller -- for the first NEBULA
    milestone, callers should anchor ranges at/above EXISTING_THRESHOLDS
    rather than inventing new numbers (see experiments/train_autockt.py).
    Matches AutoCkt's own lack of a code-enforced train/validation split:
    call this twice with different seeds/ranges to build separate pools, as
    autockt/README.md instructs doing manually via gen_specs.py.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    missing = [name for name in SPEC_NAMES if name not in ranges]
    if missing:
        raise ValueError(f"missing ranges for: {missing}")
    rng = random.Random(seed)
    pool = []
    for _ in range(count):
        values = {name: rng.uniform(*ranges[name]) for name in SPEC_NAMES}
        pool.append(TargetSpec(**values))
    return tuple(pool)

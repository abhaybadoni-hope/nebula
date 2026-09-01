"""AutoCkt-style state vector construction.

[AUTOCKT-REPLICATED] matches autockt/envs/ngspice_vanilla_opamp.py
(github.com/ksettaluri6/AutoCkt):

    self.ob = np.concatenate([cur_spec_norm, self.specs_ideal_norm, self.cur_params_idx])

where (verified from source):

    def lookup(self, spec, goal_spec):
        norm_spec = (spec - goal_spec) / (goal_spec + spec)
        return norm_spec

    cur_spec_norm    = lookup(current_achieved_specs, goal_spec)
    specs_ideal_norm = lookup(goal_spec, global_g)   # global_g: a fixed reference vector
    cur_params_idx   = raw, UNnormalized discrete grid indices

State order is therefore: [achieved-vs-goal error, goal-vs-reference error,
raw parameter indices] -- confirmed from source, not assumed.

[NEBULA ADAPTATION] denominator epsilon guard: AutoCkt's four specs (gain,
unity-gain bandwidth, phase margin, bias current) are always comfortably
bounded away from zero, so `goal + spec` is never near-zero in the official
env. NEBULA's `dfe_min_margin_v` can be an arbitrarily small positive number
right at the "just barely passing" boundary (the repo's own golden test in
tests/test_rl_readiness.py uses `dfe_min_margin_v: 1e-12` as a
passing-boundary value), where `(a - b) / (a + b)` is only numerically
well-behaved if `a + b` is not close to zero. AutoCkt's source has no such
guard because it never needed one; this file adds a small denominator floor,
LOOKUP_EPSILON, which AutoCkt's implementation does not have.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from .parameter_grid import PARAMETER_NAMES
from .target_spec import GLOBAL_REFERENCE, SPEC_DIRECTIONS, SPEC_NAMES, TargetSpec

# [NEBULA ADAPTATION] see module docstring.
LOOKUP_EPSILON = 1e-9


def lookup(value: float, reference: float) -> float:
    """[AUTOCKT-REPLICATED] `(value - reference) / (reference + value)`,
    with a [NEBULA ADAPTATION] denominator floor near zero.
    """

    denom = reference + value
    if abs(denom) < LOOKUP_EPSILON:
        denom = LOOKUP_EPSILON if denom >= 0 else -LOOKUP_EPSILON
    return (value - reference) / denom


def signed_relative_error(name: str, relative_error: float) -> float:
    """[AUTOCKT-REPLICATED] sign flip for smaller-is-better specs, mirroring:

        if self.specs_id[i] == 'ibias_max':
            rel_spec = rel_spec * -1.0

    NEBULA's smaller-is-better spec is `ctle_power_w` (see target_spec.py).
    """

    return -relative_error if SPEC_DIRECTIONS[name] == "smaller_is_better" else relative_error


def spec_error_vector(current_metrics: Mapping[str, float], target: TargetSpec) -> tuple[float, ...]:
    """[AUTOCKT-REPLICATED] `cur_spec_norm = lookup(achieved, goal)`, signed."""

    return tuple(
        signed_relative_error(name, lookup(current_metrics.get(name, 0.0), getattr(target, name)))
        for name in SPEC_NAMES
    )


def target_reference_vector(target: TargetSpec) -> tuple[float, ...]:
    """[AUTOCKT-REPLICATED] `specs_ideal_norm = lookup(goal, global_g)`, signed."""

    return tuple(
        signed_relative_error(name, lookup(getattr(target, name), GLOBAL_REFERENCE[name]))
        for name in SPEC_NAMES
    )


def build_state(
    current_metrics: Mapping[str, float],
    target: TargetSpec,
    param_indices: Sequence[int],
) -> tuple[float, ...]:
    """[AUTOCKT-REPLICATED shape] concat(cur_spec_norm, specs_ideal_norm, cur_params_idx).

    `param_indices` are raw, unnormalized grid indices -- matching AutoCkt's
    own choice not to normalize `cur_params_idx` (verified from source).
    """

    if len(param_indices) != len(PARAMETER_NAMES):
        raise ValueError(f"expected {len(PARAMETER_NAMES)} parameter indices, got {len(param_indices)}")
    return (
        spec_error_vector(current_metrics, target)
        + target_reference_vector(target)
        + tuple(float(i) for i in param_indices)
    )


STATE_DIM = 2 * len(SPEC_NAMES) + len(PARAMETER_NAMES)

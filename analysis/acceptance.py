"""Strict design acceptance, independent of the training reward."""

import math
from typing import Mapping

from rl.target_spec import TargetSpec


def gate(value, *, minimum=None, maximum=None, inclusive=False):
    if value is None:
        return "NOT CLAIMED"
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return "FAIL"
    if minimum is not None and (value < minimum if inclusive else value <= minimum):
        return "FAIL"
    if maximum is not None and (value > maximum if inclusive else value >= maximum):
        return "FAIL"
    return "PASS"


def acceptance_violations(metrics: Mapping, target: TargetSpec, *, final=False, require_peaking=True):
    """Require both competition limits and the requested target, without tolerance."""
    checks = {
        "dfe_locked_phase_eye_height_v": dict(minimum=max(0.1, target.dfe_locked_phase_eye_height_v)),
        "dfe_eye_width_ui": dict(minimum=max(0.4, target.dfe_eye_width_ui)),
        "dfe_min_margin_v": dict(minimum=max(0.0, target.dfe_min_margin_v)),
        "ctle_power_w": dict(minimum=0.0, maximum=min(0.015, target.ctle_power_w)),
    }
    if require_peaking:
        checks["peaking_db"] = dict(minimum=3.0, maximum=12.0, inclusive=True)
    if final:
        checks.update({
            "hd3_db": dict(maximum=-30.0),
            "input_referred_noise_vrms": dict(minimum=0.0, maximum=0.0015),
        })
    return [name for name, limits in checks.items() if gate(metrics.get(name), **limits) != "PASS"]

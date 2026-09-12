"""Deterministic PCIe Gen-2 NRZ stimulus generation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Iterable

from .provenance import sha256_text
from .waveform import require_numpy


@dataclass(frozen=True)
class NRZStimulusConfig:
    bit_rate: float = 5e9
    differential_pp_v: float = 0.8
    common_mode_v: float = 1.2
    rise_time_s: float = 20e-12
    fall_time_s: float = 20e-12
    time_step_s: float = 2e-12
    pattern: str = "prbs7"
    seed: int = 7
    bit_count: int = 128
    warmup_bits: int = 16
    tail_bits: int = 16
    jitter_rms_s: float = 0.0

    @property
    def ui_s(self) -> float:
        return 1.0 / self.bit_rate

    def validate(self) -> None:
        if not math.isfinite(self.bit_rate) or self.bit_rate <= 0:
            raise ValueError("bit rate must be finite and positive")
        if not math.isfinite(self.jitter_rms_s) or not 0 <= self.jitter_rms_s <= 0.05 * self.ui_s:
            raise ValueError("jitter RMS must be between zero and 0.05 UI")
        if self.bit_rate <= 0 or self.bit_count < 4:
            raise ValueError("bit rate and bit count must be positive")
        if self.differential_pp_v <= 0:
            raise ValueError("differential amplitude must be positive")
        if not 0 < self.time_step_s <= min(self.rise_time_s, self.fall_time_s):
            raise ValueError("time step must be positive and no larger than edge time")
        if self.warmup_bits + self.tail_bits >= self.bit_count:
            raise ValueError("warmup and tail must leave measurement bits")


@dataclass(frozen=True)
class NRZStimulus:
    config: NRZStimulusConfig
    bits: object
    time_s: object
    differential_v: object
    checksum: str


def _prbs7(count: int, seed: int) -> list[int]:
    state = seed & 0x7F or 0x5D
    bits: list[int] = []
    for _ in range(count):
        output = state & 1
        bits.append(output)
        feedback = ((state >> 6) ^ (state >> 5)) & 1
        state = ((state << 1) & 0x7E) | feedback
    return bits


def generate_bits(pattern: str, count: int, seed: int = 7, custom: Iterable[int] | None = None) -> list[int]:
    if custom is not None:
        values = [int(value) for value in custom]
        if not values or any(value not in (0, 1) for value in values):
            raise ValueError("custom pattern must contain zero and one values")
        return [values[index % len(values)] for index in range(count)]
    name = pattern.lower()
    if name == "prbs7":
        return _prbs7(count, seed)
    if name == "prbs15":
        state = seed & 0x7fff or 1
        values = []
        for _ in range(count):
            values.append(state & 1)
            state = ((state << 1) & 0x7ffe) | (((state >> 14) ^ (state >> 13)) & 1)
        return values
    patterns = {
        "1010": [1, 0],
        "long_runs": [1, 1, 1, 1, 0, 0, 0, 0],
        "regression": [1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
        "isolated_one": [0, 0, 0, 1, 0, 0, 0, 0],
    }
    if name not in patterns:
        raise ValueError(f"unsupported pattern: {pattern}")
    values = patterns[name]
    return [values[index % len(values)] for index in range(count)]


def generate_nrz(config: NRZStimulusConfig, custom_bits: Iterable[int] | None = None) -> NRZStimulus:
    numpy = require_numpy()
    config.validate()
    bits = numpy.asarray(generate_bits(config.pattern, config.bit_count, config.seed, custom_bits), dtype=int)
    sample_count = int(round(config.bit_count * config.ui_s / config.time_step_s)) + 1
    time = numpy.arange(sample_count, dtype=float) * config.time_step_s
    if config.jitter_rms_s:
        rng = numpy.random.default_rng(config.seed)
        jitter = numpy.clip(rng.normal(0., config.jitter_rms_s, config.bit_count - 1), -.2*config.ui_s, .2*config.ui_s)
        edges = numpy.arange(1, config.bit_count)*config.ui_s + jitter
        bit_index = numpy.searchsorted(edges, time, side="right")
    else:
        bit_index = numpy.minimum((time / config.ui_s).astype(int), config.bit_count - 1)
    target = numpy.where(bits[bit_index] > 0, config.differential_pp_v / 2, -config.differential_pp_v / 2)
    waveform = numpy.empty_like(target)
    waveform[0] = target[0]
    for index in range(1, len(waveform)):
        edge = config.rise_time_s if target[index] > waveform[index - 1] else config.fall_time_s
        maximum_change = config.differential_pp_v * config.time_step_s / edge
        delta = target[index] - waveform[index - 1]
        waveform[index] = waveform[index - 1] + max(-maximum_change, min(maximum_change, delta))
    identity = {
        **config.__dict__,
        "bits": bits.tolist(),
    }
    return NRZStimulus(config, bits, time, waveform, sha256_text(json.dumps(identity, sort_keys=True)))


def compress_pwl(time_s, values, tolerance_v):
    """Keep endpoints and bound vertical interpolation error at every input knot."""
    numpy=require_numpy()
    time=numpy.asarray(time_s); signal=numpy.asarray(values)
    if not math.isfinite(tolerance_v) or tolerance_v<0:
        raise ValueError("PWL tolerance must be finite and nonnegative")
    if tolerance_v==0 or len(time)<3: return time,signal
    if len(time)!=len(signal) or numpy.any(numpy.diff(time)<=0) or not numpy.all(numpy.isfinite(signal)) or not numpy.all(numpy.isfinite(time)):
        raise ValueError("PWL requires finite values and strictly increasing matching times")
    keep={0,len(time)-1}; pending=[(0,len(time)-1)]
    while pending:
        left,right=pending.pop()
        if right-left<2: continue
        estimate=signal[left]+(signal[right]-signal[left])*(time[left+1:right]-time[left])/(time[right]-time[left])
        errors=numpy.abs(signal[left+1:right]-estimate)
        offset=int(numpy.argmax(errors))
        if errors[offset]>tolerance_v:
            split=left+1+offset;keep.add(split)
            pending.extend(((left,split),(split,right)))
    indices=sorted(keep)
    return time[indices],signal[indices]


def stimulus_include(time_s, differential_v, common_mode_v: float, division_compensation: float = 2.0, *, tolerance_v: float = 0.0) -> str:
    """Create PWL sources whose terminated differential output equals the input waveform."""
    if division_compensation <= 0:
        raise ValueError("division compensation must be positive")
    time_s, differential_v = compress_pwl(time_s, differential_v, tolerance_v)
    positive = common_mode_v + division_compensation * differential_v / 2
    negative = common_mode_v - division_compensation * differential_v / 2

    def source(name, values):
        pairs = [f"{time:.15g} {value:.15g}" for time, value in zip(time_s, values)]
        lines = [f"{name} {'chp_src' if name == 'VCHP' else 'chn_src'} 0 PWL("]
        for start in range(0, len(pairs), 6):
            lines.append("+ " + " ".join(pairs[start:start + 6]))
        lines.append("+ )")
        return "\n".join(lines)

    return source("VCHP", positive) + "\n" + source("VCHN", negative) + "\n"

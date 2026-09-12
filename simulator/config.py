"""Portable, typed configuration for SKY130 receiver simulations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import os
from pathlib import Path
from .sizing import CircuitSizing


SKY130_MODEL_ENV = "SKY130_MODEL_LIBRARY"


class ProcessCorner(str, Enum):
    TT = "tt"
    SS = "ss"
    FF = "ff"
    SF = "sf"
    FS = "fs"


@dataclass(frozen=True)
class SimulationConditions:
    process_corner: ProcessCorner = ProcessCorner.TT
    temperature_c: float = 27.0
    supply_v: float = 1.8
    input_common_mode_v: float = 1.2
    output_load_f: float = 20e-15
    source_resistance_per_leg_ohm: float = 50.0
    receiver_termination_diff_ohm: float = 100.0
    input_parasitic_f: float = 0.0
    circuit_sizing: CircuitSizing | None = None
    stimulus_pattern: str = "prbs7"
    stimulus_jitter_rms_s: float = 0.0
    stimulus_pwl_tolerance_v: float = 0.0
    training_bit_count: int = 128
    validation_bit_count: int | None = None

    def __post_init__(self):
        if isinstance(self.process_corner, str) and self.process_corner in {c.value for c in ProcessCorner}:
            object.__setattr__(self, "process_corner", ProcessCorner(self.process_corner))
        if isinstance(self.circuit_sizing, dict):
            object.__setattr__(self, "circuit_sizing", CircuitSizing(**self.circuit_sizing))

    def validate(self) -> None:
        if not 0 <= self.stimulus_pwl_tolerance_v <= 1e-5:
            raise ValueError("PWL differential waveform tolerance must be 0..10 microvolts")
        if type(self.training_bit_count) is not int or not 64 <= self.training_bit_count <= 128:
            raise ValueError("training bit count must be an integer between 64 and 128")
        if self.stimulus_pattern not in ("prbs7", "prbs15", "1010", "long_runs", "regression", "isolated_one"):
            raise ValueError("unsupported stimulus pattern")
        if not 0 <= self.stimulus_jitter_rms_s <= 10e-12:
            raise ValueError("jitter must be between zero and 10 ps RMS")
        if self.validation_bit_count is not None and not 128 <= self.validation_bit_count <= 65536:
            raise ValueError("validation bit count must be between 128 and 65536")
        if self.circuit_sizing is not None:
            self.circuit_sizing.validate()
        if not isinstance(self.process_corner, ProcessCorner):
            raise ValueError("process corner must be a ProcessCorner value")
        if not 0 <= self.temperature_c <= 125:
            raise ValueError("temperature must be between 0 C and 125 C")
        if not 1.71 <= self.supply_v <= 1.89:
            raise ValueError("supply voltage must be between 1.71 V and 1.89 V")
        if not 0 < self.input_common_mode_v < self.supply_v:
            raise ValueError("input common mode must lie between ground and VDD")
        if not 10e-15 <= self.output_load_f <= 50e-15:
            raise ValueError("Stage 1 output load must lie between 10 fF and 50 fF")
        if self.source_resistance_per_leg_ohm <= 0:
            raise ValueError("source resistance must be positive")
        if self.receiver_termination_diff_ohm <= 0:
            raise ValueError("receiver termination must be positive")
        if self.input_parasitic_f < 0:
            raise ValueError("input parasitic cannot be negative")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["process_corner"] = (
            self.process_corner.value
            if isinstance(self.process_corner, ProcessCorner)
            else str(self.process_corner)
        )
        return value


@dataclass(frozen=True)
class Sky130Config:
    model_library: str | Path | None = None

    def resolve_model_library(self) -> Path:
        candidates: list[Path] = []
        requested = self.model_library or os.environ.get(SKY130_MODEL_ENV)
        if requested:
            requested_path = Path(requested).expanduser()
            if not requested_path.is_file():
                raise FileNotFoundError(f"SKY130 model library does not exist: {requested_path}")
            return requested_path.resolve()
        local_model = Path(__file__).resolve().parents[1] / "generated/sky130-flat.lib.spice"
        candidates.append(local_model)
        home = Path.home()
        pdk_root = os.environ.get("PDK_ROOT")
        if pdk_root:
            candidates.append(
                Path(pdk_root).expanduser() / "sky130A" / "libs.tech" / "ngspice" / "sky130.lib.spice"
            )
        candidates.extend((
            home / ".ciel" / "sky130A" / "libs.tech" / "ngspice" / "sky130.lib.spice",
            home / ".volare" / "sky130A" / "libs.tech" / "ngspice" / "sky130.lib.spice",
        ))
        for path in candidates:
            if path.is_file():
                return path.resolve()
        if requested:
            raise FileNotFoundError(f"SKY130 model library does not exist: {candidates[0]}")
        raise FileNotFoundError(
            f"Set {SKY130_MODEL_ENV} to sky130A/libs.tech/ngspice/sky130.lib.spice"
        )


PVT_GRID = tuple(
    SimulationConditions(ProcessCorner(corner), temperature, supply)
    for corner in ("tt", "ss", "ff", "sf", "fs")
    for supply in (1.71, 1.8, 1.89)
    for temperature in (0.0, 27.0, 75.0, 125.0)
)

OUTPUT_LOAD_GRID_F = (10e-15, 20e-15, 50e-15)

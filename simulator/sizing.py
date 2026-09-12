"""Grouped transistor dimensions and an optional resistor-fed current mirror."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class CircuitSizing:
    input_width_um: float = 10.0
    input_length_um: float = 0.15
    tail_width_um: float = 20.0
    tail_length_um: float = 0.5
    bias_resistance_ohm: float = 11500.0
    physical_bias: bool = False

    def validate(self):
        limits = {"input_width_um": (.42, 100.), "input_length_um": (.15, 2.),
                  "tail_width_um": (.42, 100.), "tail_length_um": (.15, 2.),
                  "bias_resistance_ohm": (100., 1e6)}
        for name, (low, high) in limits.items():
            value = getattr(self, name)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} outside [{low}, {high}]")
        if not isinstance(self.physical_bias, bool):
            raise ValueError("physical_bias must be boolean")

    def spice_parameters(self):
        self.validate()
        return dict(WIN=self.input_width_um, LIN=self.input_length_um,
                    WTAIL=self.tail_width_um, LTAIL=self.tail_length_um,
                    RBIAS=self.bias_resistance_ohm)

    def channel_area_um2(self):
        return 2*self.input_width_um*self.input_length_um + (
            2*self.tail_width_um*self.tail_length_um if self.physical_bias else 0.)

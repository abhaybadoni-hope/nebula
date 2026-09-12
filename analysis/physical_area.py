"""Geometry-based area estimate with explicit, versioned assumptions.

Capacitor density and resistor sheet values describe SKY130 nominal devices.
Placement factors are engineering assumptions, not foundry rules or DRC area.
"""
from dataclasses import asdict, dataclass
import math
from simulator.sizing import CircuitSizing

@dataclass(frozen=True)
class AreaAssumptions:
    capacitance_ff_per_um2: float = 2.0
    resistor_sheet_ohm: float = 300.0
    resistor_width_um: float = .69
    transistor_layout_factor: float = 10.0
    passive_layout_factor: float = 2.0
    routing_factor: float = 1.3
    source: str = "SKY130 device-details: 300 ohm/sq precision poly; models/r+c/res_typical__cap_typical__lin.spice: camimc=2e-15 F/um2"

def estimate_area(parameters, sizing=None, *, assumptions=AreaAssumptions(), include_sampler=False):
    sizing=sizing or CircuitSizing()
    sizing.validate()
    for name,value in asdict(assumptions).items():
        if name != "source" and (not math.isfinite(value) or value<=0): raise ValueError(f"invalid area assumption {name}")
    resistances=[parameters.rload_ohm]*2+[parameters.rdeg_ohm]*2
    if sizing.physical_bias: resistances.append(sizing.bias_resistance_ohm)
    channel=sizing.channel_area_um2()
    if include_sampler:
        # Expanded sampler_dfe.spice: comparator pair+mirror; 7 inverters; 4 transmission gates.
        channel += 2*5*.15+2*10*.5+7*(2+1)*.15+4*(2+4)*.15
        if parameters.dfe_tap_v:
            resistances += [1.8*1000/abs(parameters.dfe_tap_v)]*2
        resistances += [1000.]*2+[20000.]*2+[10000.]
    resistor=sum(r/assumptions.resistor_sheet_ohm*assumptions.resistor_width_um**2 for r in resistances)
    capacitor=2*parameters.cdeg_f*1e15/assumptions.capacitance_ff_per_um2
    estimate=(channel*assumptions.transistor_layout_factor+(resistor+capacitor)*assumptions.passive_layout_factor)*assumptions.routing_factor
    return {"estimated_area_mm2":estimate*1e-6,"transistor_channel_area_um2":channel,
            "resistor_body_area_um2":resistor,"capacitor_plate_area_um2":capacitor,
            "assumptions":asdict(assumptions),"includes_sampler":include_sampler,
            "verdict":"ESTIMATE ONLY", "layout_verified":False,
            "limitations":["ideal R/C simulation elements have not been replaced by layout-extracted PDK passives", "external clock driver and pads excluded", "feedback resistance uses nominal 1.8 V"]}

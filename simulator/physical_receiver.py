"""Explicit, separate validation of the transistor sampler/feedback prototype.

Never substitutes its measurements for the behavioral training model. The
external clock driver and layout parasitics are outside this model.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import math
import numpy as np
from simulator.config import SimulationConditions, Sky130Config
from simulator.receiver import ReceiverParameters, circuit_block, SYNTHETIC_CHANNEL
from simulator.stimulus import NRZStimulusConfig, generate_nrz, stimulus_include
from simulator.channel import load_s4p, filter_channel
from simulator.models import SimulationRequest
from simulator.ngspice import run_simulation, NgSpiceConfig, spice_path
from simulator.waveform import parse_wrdata

BLOCK = Path(__file__).resolve().parents[1]/"circuits/blocks/sampler_dfe.spice"

def render_bench(parameters, conditions, model, stimulus_path, *, end_s, step_s=2e-12):
    parameters.validate(); conditions.validate()
    sizing=conditions.circuit_sizing
    size_values = sizing.spice_parameters() if sizing else dict(WIN=10,LIN=.15,WTAIL=20,LTAIL=.5,RBIAS=11500)
    params={**parameters.spice_parameters(conditions),**size_values}
    assignments=" ".join(f"{k}={v:.15g}" for k,v in params.items())
    rin=1000.
    rfb=conditions.supply_v*rin/abs(parameters.dfe_tap_v) if parameters.dfe_tap_v else 1e15
    # Swap feedback sign through logical input/output polarity, while retaining data polarity.
    positive,negative="outn","outp"
    fpos,fneg=("qb","q") if parameters.dfe_tap_v>=0 else ("q","qb")
    return f"""* NEBULA physical sampler and one-tap DFE validation prototype
.lib "{spice_path(model)}" {conditions.process_corner.value}
.include "{spice_path(circuit_block(conditions))}"
.include "{spice_path(BLOCK)}"
.include "{spice_path(stimulus_path)}"
.param {assignments}
.temp {conditions.temperature_c}
VDD vdd 0 {{VDD_VAL}}
VCLK clk 0 PULSE(0 {{VDD_VAL}} 100p 5p 5p 95p 200p)
RSP chp_src inp {conditions.source_resistance_per_leg_ohm}
RSN chn_src inn {conditions.source_resistance_per_leg_ohm}
RTERM inp inn {conditions.receiver_termination_diff_ohm}
CINP inp 0 {max(conditions.input_parasitic_f,1e-18)}
CINN inn 0 {max(conditions.input_parasitic_f,1e-18)}
CLP outp 0 {{CLOAD}}
CLN outn 0 {{CLOAD}}
XCTLE inp inn outp outn vdd 0 CTLE RLOAD={{RLOAD}} RDEG={{RDEG}} CDEG={{CDEG}} ITAIL_VAL={{ITAIL_VAL}} WIN={{WIN}} LIN={{LIN}} WTAIL={{WTAIL}} LTAIL={{LTAIL}} RBIAS={{RBIAS}}
XRX {positive} {negative} clk q qb sump sumn {fpos} {fneg} vdd 0 NEBULA_SAMPLER_DFE RIN={rin} RFB={rfb}
.control
set wr_singlescale
set wr_vecnames
option numdgt=15
tran {step_s} {end_s} 0 {step_s}
let supply_current=i(VDD)
let sum_diff=v(sump)-v(sumn)
wrdata @@PHYSICAL_OUTPUT_FILE@@ v(q) v(qb) sum_diff supply_current v(xrx.cp) v(xrx.d)
quit
.endc
.end
"""

def evaluate_physical_receiver(parameters, conditions=SimulationConditions(), *, channel_path=SYNTHETIC_CHANNEL,
                               sky130=None, ngspice=None, bit_count=128):
    model=(sky130 or Sky130Config()).resolve_model_library()
    config=NRZStimulusConfig(bit_count=bit_count,pattern=conditions.stimulus_pattern,
        jitter_rms_s=conditions.stimulus_jitter_rms_s,common_mode_v=conditions.input_common_mode_v)
    stimulus=generate_nrz(config)
    channel=load_s4p(channel_path)
    waveform=filter_channel(channel,stimulus.time_s,stimulus.differential_v)
    compensation=1+2*conditions.source_resistance_per_leg_ohm/conditions.receiver_termination_diff_ohm
    with TemporaryDirectory(prefix="nebula_physical_") as temp:
        root=Path(temp); source=root/"stimulus.inc"; bench=root/"physical.cir"
        source.write_text(stimulus_include(stimulus.time_s,waveform,conditions.input_common_mode_v,compensation,tolerance_v=conditions.stimulus_pwl_tolerance_v))
        bench.write_text(render_bench(parameters,conditions,model,source,end_s=stimulus.time_s[-1]))
        result=run_simulation(SimulationRequest(bench,output_files={"PHYSICAL_OUTPUT_FILE":"physical.dat"}),
                              config=ngspice or NgSpiceConfig(timeout_s=60))
    if not result.success:
        return {"success":False,"errors":result.errors,"metrics":{},"provenance":result.provenance}
    trace=parse_wrdata(result.artifacts["PHYSICAL_OUTPUT_FILE"])
    delay=max(0.,channel.bulk_delay_s())
    sample_times=(np.arange(bit_count)+.9)*config.ui_s+delay
    valid=(np.arange(bit_count)>=config.warmup_bits)&(np.arange(bit_count)<bit_count-config.tail_bits)&(sample_times<=trace.scale[-1])
    measured=np.interp(sample_times[valid],trace.scale,trace.column("v(q)"))
    expected=stimulus.bits[valid]
    errors=int(np.sum((measured>=conditions.supply_v/2)!=expected))
    power=float(-np.mean(trace.column("supply_current"))*conditions.supply_v)
    return {"success":len(expected)>0 and errors==0 and math.isfinite(power) and 0<power<.015,
            "metrics":{"physical_receiver_power_w":power,"physical_error_count":errors,
                       "physical_measured_bits":len(expected),
                       "q_min_v":float(np.min(trace.column("v(q)"))),
                       "q_max_v":float(np.max(trace.column("v(q)"))),
                       "comparator_min_v":float(np.min(trace.column("v(xrx.cp)"))),
                       "comparator_max_v":float(np.max(trace.column("v(xrx.cp)")))},"provenance":result.provenance,
            "limitations":["experimental sampler/DFE", "external clock-driver power excluded", "no layout parasitics", "fixed sampling clock phase"]}

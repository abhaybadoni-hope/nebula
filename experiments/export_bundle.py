"""Export editable SPICE sources, all verification benches, model closure and stimulus."""
from pathlib import Path
import argparse
import json
import re
from dataclasses import asdict
from simulator.config import SimulationConditions, Sky130Config
from simulator.receiver import ReceiverParameters, circuit_block, BENCHES, SYNTHETIC_CHANNEL
from simulator.ngspice import parameterize_netlist, render_template
from simulator.stimulus import NRZStimulusConfig, generate_nrz, stimulus_include
from simulator.channel import load_s4p, filter_channel
from simulator.provenance import sha256_file

INCLUDE=re.compile(r"(?im)^\s*\.include\s+(?:\"([^\"]+)\"|'([^']+)'|([^\s;]+))[^\r\n]*")

def flatten_model(path, stack=()):
    path=Path(path).resolve()
    if path in stack: raise ValueError("cyclic model include")
    text=path.read_text(encoding="utf-8")
    def expand(match):
        child=next(v for v in match.groups() if v is not None)
        if "{" in child or "$" in child: raise ValueError("dynamic model include cannot be bundled")
        return flatten_model(path.parent/child,stack+(path,))
    return "* Bundled source: "+path.name+" SHA256="+sha256_file(path)+"\n"+INCLUDE.sub(expand,text)

def export_bundle(parameters, output, *, conditions=SimulationConditions(), channel_path=SYNTHETIC_CHANNEL, model=None):
    output=Path(output)
    if output.exists(): raise FileExistsError("bundle destination must be new")
    parameters.validate(); conditions.validate()
    model=Path(model) if model else Sky130Config().resolve_model_library()
    model_text=flatten_model(model)
    config=NRZStimulusConfig(bit_count=1024,common_mode_v=conditions.input_common_mode_v,
                             pattern=conditions.stimulus_pattern,jitter_rms_s=conditions.stimulus_jitter_rms_s)
    stimulus=generate_nrz(config)
    waveform=filter_channel(load_s4p(channel_path),stimulus.time_s,stimulus.differential_v)
    output.mkdir(parents=True)
    (output/"models.spice").write_text(model_text)
    (output/"ctle.spice").write_text(circuit_block(conditions).read_text())
    (output/"channel.s4p").write_bytes(Path(channel_path).read_bytes())
    compensation=1+2*conditions.source_resistance_per_leg_ohm/conditions.receiver_termination_diff_ohm
    (output/"stimulus.inc").write_text(stimulus_include(stimulus.time_s,waveform,conditions.input_common_mode_v,compensation,tolerance_v=conditions.stimulus_pwl_tolerance_v))
    values=parameters.spice_parameters(conditions)
    values.update(SOURCE_LEG_R=conditions.source_resistance_per_leg_ohm,RTERM_DIFF=conditions.receiver_termination_diff_ohm,CIN=conditions.input_parasitic_f)
    for bench in BENCHES.glob("*.cir"):
        source=bench.read_text()
        declared={name:value for name,value in values.items() if re.search(r"(?im)^\s*\.param[^\n]*\b"+name+r"=",source)}
        source=parameterize_netlist(source,declared)
        templates={"SKY130_MODEL_LIBRARY":"models.spice","CTLE_BLOCK_FILE":"ctle.spice",
                   "STIMULUS_INCLUDE_FILE":"stimulus.inc","PROCESS_CORNER":conditions.process_corner.value,
                   "TEMPERATURE_C":str(conditions.temperature_c),"TRANSIENT_STEP":str(config.time_step_s),
                   "TRANSIENT_END":str(stimulus.time_s[-1])}
        for marker in re.findall(r"@@([A-Z0-9_]+)@@",source):
            if marker.endswith("OUTPUT_FILE"): templates[marker]=bench.stem+".dat"
        (output/bench.name).write_text(render_template(source,{k:v for k,v in templates.items() if "@@"+k+"@@" in source}))
    init=Path(__file__).resolve().parents[1]/"simulator/resources/sky130.spiceinit"
    (output/".spiceinit").write_bytes(init.read_bytes())
    manifest={"parameters":asdict(parameters),"conditions":conditions.to_dict(),
              "dfe":"behavioral one-tap; tap recorded in parameters; not present in CTLE benches",
              "files":{p.name:sha256_file(p) for p in output.iterdir() if p.is_file()}}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    (output/"README.md").write_text("# NEBULA verification bundle\n\nEditable SPICE source files and all model includes are local. Run from this directory, e.g. `ngspice -b ctle_dc.cir`. The receiver transient exports the CTLE waveform; behavioral DFE metrics require Nebula. Model license notices are retained. This bundle is not a layout or a physical-DFE compliance claim.\n")
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--design",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--model",type=Path); p.add_argument("--channel",type=Path,default=SYNTHETIC_CHANNEL)
    a=p.parse_args(); d=json.loads(a.design.read_text()); export_bundle(ReceiverParameters(**d["parameters"]),a.output,conditions=SimulationConditions(**d.get("conditions",{})),channel_path=a.channel,model=a.model)

if __name__=="__main__": main()

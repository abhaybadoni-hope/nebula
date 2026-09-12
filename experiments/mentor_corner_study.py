"""HD3/eye-led TT study followed by fixed-design SS and FF checks.

Power and area are reported, not optimized. This is not full compliance.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time
from simulator.config import SimulationConditions, ProcessCorner, Sky130Config
from simulator.ngspice import NgSpiceConfig
from simulator.receiver import (ReceiverParameters, EvaluationFidelity, SYNTHETIC_CHANNEL,
    _run_dc, _run_ac, _run_hd3, _run_transient)
from analysis.physical_area import estimate_area


def priority(row):
    m=row["metrics"]
    checks=(("hd3_db",lambda x:x < -30),("dfe_locked_phase_eye_height_v",lambda x:x>.1),
            ("dfe_eye_width_ui",lambda x:x>.4),("dfe_min_margin_v",lambda x:x>0),
            ("dfe_error_count",lambda x:x==0),("peaking_db",lambda x:3<=x<=12))
    failures=sum(not (isinstance(m.get(k),(float,int)) and math.isfinite(m[k]) and test(m[k])) for k,test in checks)
    failures+=sum(bool(s["errors"]) or bool([v for v in s["violations"] if "power" not in v.lower()]) for s in row["stages"])
    return (failures,m.get("hd3_db",float("inf")),-m.get("dfe_locked_phase_eye_height_v",0),
            -m.get("dfe_eye_width_ui",0),abs(m.get("peaking_db",0)-6))


def evaluate(name,parameters,corner,output,model,deadline):
    conditions=SimulationConditions(process_corner=ProcessCorner(corner))
    stages=[]; metrics={}
    for label,run in (("dc",_run_dc),("ac",_run_ac),("hd3",_run_hd3),("eye",None)):
        remaining=deadline-time.monotonic()
        if remaining<=0: break
        config=NgSpiceConfig(timeout_s=min(30,remaining))
        if run:
            result=run(parameters,conditions,model,config)
        else:
            result=_run_transient(parameters,conditions,model,config,EvaluationFidelity.TRAINING,
                SYNTHETIC_CHANNEL,waveform_output=output/f"{name}-{corner}-eye.npz")
        stages.append(asdict(result)); metrics.update(result.metrics)
        # Stop broken simulations or unusable DC; retain AC/HD3 failures for diagnosis.
        if result.errors or (label=="dc" and any("power" not in v.lower() for v in result.violations)): break
    return {"name":name,"parameters":asdict(parameters),"conditions":conditions.to_dict(),
            "metrics":metrics,"stages":stages,"area":estimate_area(parameters),
            "completed_stages":len(stages),"full_compliance_claimed":False}


def write_visual_report(output,report):
    import html
    import numpy as np
    figures=[]
    rows=[]
    for row in report["tt"]+report["corners"]:
        corner=row["conditions"]["process_corner"]
        label=row["name"]+"-"+corner
        path=output/(label+"-eye.npz")
        m=row["metrics"]
        keys=("hd3_db","dfe_locked_phase_eye_height_v","dfe_eye_width_ui","peaking_db","gain_2p5ghz_db","nyquist_boost_db","ctle_power_w")
        rows.append("<tr><td>"+html.escape(label)+"</td>"+"".join("<td>"+(f"{m[k]:.5g}" if k in m else "not measured")+"</td>" for k in keys)+"</tr>")
        if not path.exists(): continue
        with np.load(path) as trace:
            ui=float(trace["ui_s"]); t=trace["time_s"]; v=trace["vout_diff"]
            start=int(trace["warmup_bits"]); stop=len(trace["bits"])-int(trace["tail_bits"])-2
            delay=float(trace["bulk_delay_s"])
            x=np.linspace(0,2,201)
            limit=max(.1,float(np.max(np.abs(v))))
            curves=[]
            for bit in range(start,min(stop,start+128)):
                times=(bit+x)*ui+delay
                if times[-1]>t[-1]: break
                y=np.interp(times,t,v)
                points=" ".join(f"{50+250*xx:.2f},{140-100*yy/limit:.2f}" for xx,yy in zip(x,y))
                curves.append('<polyline points="'+points+'" fill="none" stroke="#1267a8" stroke-opacity=".12" stroke-width="1"/>')
        svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 310"><rect width="600" height="310" fill="white"/><text x="50" y="20">'+html.escape(label)+' - CTLE output before DFE</text><path d="M50 35V245H550M50 140H550" fill="none" stroke="#888"/>'+"".join(curves)+f'<text x="5" y="43">{limit:.2f}V</text><text x="5" y="243">-{limit:.2f}V</text><text x="50" y="270">0</text><text x="295" y="270">1</text><text x="540" y="270">2 UI</text><text x="50" y="296">Measured transient overlay; up to 128 traces</text></svg>'
        (output/(label+"-eye.svg")).write_text(svg,encoding="utf-8")
        figures.append(svg)
    document='<meta charset="utf-8"><title>Mentor corner study</title><style>body{font:16px system-ui;margin:32px;max-width:1100px}td,th{padding:9px;border-bottom:1px solid #ccc}svg{width:540px;max-width:100%}</style><h1>HD3 and eye-led corner study</h1><p>TT: 1.8 V, 27 C. SS and FF use the selected TT parameters unchanged. Nyquist frequency: 2.5 GHz for 5 Gb/s NRZ. Boost is relative to 100 MHz. Power and area are secondary. No full compliance claim.</p><p>HD3: 100 MHz, 100 mVpp differential. Eye metrics below include behavioral DFE; plots show the actual CTLE waveform before DFE. Synthetic channel, 128-bit PRBS7.</p><table><tr><th>Design/corner</th><th>HD3 dB</th><th>DFE eye V</th><th>DFE width UI</th><th>Peak boost (1.25-2.5 GHz), dB</th><th>Gain at 2.5 GHz, dB</th><th>Boost at 2.5 GHz, dB</th><th>CTLE power W</th></tr>'+"".join(rows)+'</table>'+"".join(figures)
    (output/"report.html").write_text(document,encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seconds",type=float,default=240)
    args=parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds<=0: raise ValueError("positive time budget required")
    args.output.mkdir(parents=True,exist_ok=False)
    base=ReceiverParameters(**json.loads(args.design.read_text())["parameters"])
    model=Sky130Config().resolve_model_library(); deadline=time.monotonic()+args.seconds
    choices=[("baseline",base),("more_degeneration",replace(base,rdeg_ohm=base.rdeg_ohm*1.5)),
             ("lower_capacitance",replace(base,cdeg_f=base.cdeg_f*.7))]
    report={"assumptions":{"supply_v":1.8,"temperature_c":27,"peaking_target_db":6,
            "nyquist_frequency_hz":2.5e9,"boost_reference_hz":1e8,"corner_strategy":"same TT-selected parameters at SS and FF",
            "hd3_stimulus":"100 MHz, 100 mVpp differential (repository convention)",
            "eye":"128-bit PRBS7, synthetic channel, behavioral DFE"},"tt":[],"corners":[]}
    def save(): (args.output/"study.json").write_text(json.dumps(report,indent=2,default=str))
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending=[pool.submit(evaluate,n,p,"tt",args.output,model,deadline) for n,p in choices]
        for future in pending:
            report["tt"].append(future.result()); save()
        best=min(report["tt"],key=priority)
        report["selected_for_corner_diagnosis"]=best["name"]
        report["tt_priority_failures"]=priority(best)[0]
        selected=ReceiverParameters(**best["parameters"])
        pending=[pool.submit(evaluate,best["name"],selected,c,args.output,model,deadline) for c in ("ss","ff")]
        for future in pending:
            report["corners"].append(future.result()); save()
    write_visual_report(args.output,report)
    print(json.dumps({"selected":best["name"],"tt_priority_failures":priority(best)[0],"output":str(args.output)}))

if __name__=="__main__": main()

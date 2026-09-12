"""Noise corner checks and sampled tuning evidence, separate from RL training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time
from simulator.config import SimulationConditions, ProcessCorner, Sky130Config
from simulator.receiver import ReceiverParameters,_run_dc,_run_ac,_run_noise,_run_hd3,_run_transient,EvaluationFidelity,SYNTHETIC_CHANNEL
from simulator.ngspice import NgSpiceConfig

TARGETS=(1.25e9,1.875e9,2.5e9)


def tuning_match(row,target,tolerance=.05):
    m=row["ac"]["metrics"]
    return (row["dc"]["success"] and row["ac"]["success"] and m.get("is_local_peak")==1
            and math.isfinite(m.get("peak_frequency_hz",float("nan")))
            and abs(m["peak_frequency_hz"]-target)<=target*tolerance)


def write_report(output,report):
    import html
    noise_rows=[];tuning_rows=[]
    for r in report["noise"]:
        value=r["noise"]["metrics"].get("input_referred_noise_vrms")
        noise_rows.append(f"<tr><td>{r['corner'].upper()}</td><td>{value*1000:.4f}</td><td>{'PASS' if r['passes_noise'] else 'NOT PASSED'}</td></tr>" if value is not None else "<tr><td>Not measured</td></tr>")
    for r in report["targets"]:
        m=r.get("ac",{}).get("metrics",{});p=r.get("parameters",{});eye=r.get("eye",{}).get("metrics",{})
        values=[r["frequency_hz"]/1e9,m.get("peak_frequency_hz",0)/1e9,p.get("rload_ohm"),p.get("cdeg_f",0)*1e12,
                m.get("peaking_db"),r.get("hd3",{}).get("metrics",{}).get("hd3_db"),
                eye.get("dfe_locked_phase_eye_height_v"),eye.get("dfe_eye_width_ui"),r["validated"]]
        tuning_rows.append("<tr>"+"".join("<td>"+html.escape(f"{v:.5g}" if isinstance(v,float) else str(v))+"</td>" for v in values)+"</tr>")
    body='<meta charset="utf-8"><title>Noise and sampled tuning</title><style>body{font:16px system-ui;max-width:1200px;margin:35px auto}td,th{padding:10px;border-bottom:1px solid #ccc}</style><h1>Noise and sampled tuning validation</h1><p>1.8 V, 27 C. Legacy ideal-tail CTLE, behavioral DFE. Not full PVT or physical-receiver compliance.</p><h2>Selected-design noise: 10 MHz to 5 GHz</h2><table><tr><th>Corner</th><th>Input noise mVrms</th><th>Below 1.5 mVrms</th></tr>'+"".join(noise_rows)+'</table><h2>TT tuning settings</h2><p>Targets matched within 5%; measured local peaks and 3-12 dB boost required. Rdeg, tail current and DFE tap remain at the base design values. Tuned settings receive noise, HD3 and 128-bit synthetic-channel eye checks. This demonstrates sampled settings, not continuous tunability or exact endpoint coverage.</p><table><tr><th>Target GHz</th><th>Measured GHz</th><th>Rload ohm</th><th>Cdeg pF</th><th>Boost dB</th><th>HD3 dB</th><th>Eye V</th><th>Eye UI</th><th>Validated</th></tr>'+"".join(tuning_rows)+'</table>'
    (Path(output)/"report.html").write_text(body,encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    data=json.loads(args.design.read_text());p=ReceiverParameters(**data["parameters"])
    c=SimulationConditions(**data.get("conditions",{}));model=Sky130Config().resolve_model_library()
    config=NgSpiceConfig(timeout_s=30)
    report={"parameters":asdict(p),"base_conditions":c.to_dict(),"noise":[],"tuning":[],"targets":[],
            "frequency_tolerance_fraction":.05,"continuous_tunability_proven":False,
            "scope":"nominal voltage/temperature; synthetic channel; ideal-tail CTLE and behavioral DFE"}
    def save(): (args.output/"study.json").write_text(json.dumps(report,indent=2,default=str))
    def noise(corner):
        condition=replace(c,process_corner=ProcessCorner(corner))
        dc=_run_dc(p,condition,model,config)
        stage=_run_noise(p,condition,model,config)
        value=stage.metrics.get("input_referred_noise_vrms")
        return {"corner":corner,"dc":asdict(dc),"noise":asdict(stage),
                "passes_noise":bool(dc.success and stage.success and value is not None and math.isfinite(value) and 0<value<.0015)}
    def sweep(pair):
        load,cap=pair;parameters=replace(p,rload_ohm=p.rload_ohm*load,cdeg_f=p.cdeg_f*cap)
        condition=replace(c,process_corner=ProcessCorner.TT)
        dc=_run_dc(parameters,condition,model,config);ac=_run_ac(parameters,condition,model,config)
        return {"parameters":asdict(parameters),"load_scale":load,"cap_scale":cap,"dc":asdict(dc),"ac":asdict(ac)}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for row in pool.map(noise,("tt","ss","ff")):
            report["noise"].append(row);save()
        print("Noise checks complete",flush=True)
        for row in pool.map(sweep,[(r,k) for r in (.7,1.,1.4) for k in (.25,.4,.6,.8,1.,1.3,1.7,2.5,4.)]):
            report["tuning"].append(row);save()
    if any(not any(tuning_match(r,t) for r in report["tuning"]) for t in TARGETS):
        with ThreadPoolExecutor(max_workers=2) as pool:
            for row in pool.map(sweep,[(.7,k) for k in (.425,.45,.475)]+[(1.,k) for k in (.3,.325,.35)]+[(1.4,k) for k in (.2,.225,.24)]):
                report["tuning"].append(row);save()
    for target in TARGETS:
        matches=[r for r in report["tuning"] if tuning_match(r,target)]
        target_row={"frequency_hz":target,"ac_covered":bool(matches),"validated":False}
        if matches:
            best=min(matches,key=lambda r:abs(r["ac"]["metrics"]["peak_frequency_hz"]-target))
            parameters=ReceiverParameters(**best["parameters"]);condition=replace(c,process_corner=ProcessCorner.TT)
            hd3=_run_hd3(parameters,condition,model,config)
            noise_stage=_run_noise(parameters,condition,model,config)
            eye=_run_transient(parameters,condition,model,config,EvaluationFidelity.TRAINING,SYNTHETIC_CHANNEL)
            m=eye.metrics
            eye_pass=eye.success and m.get("dfe_locked_phase_eye_height_v",0)>.1 and m.get("dfe_eye_width_ui",0)>.4
            target_row.update(parameters=asdict(parameters),ac=best["ac"],hd3=asdict(hd3),noise=asdict(noise_stage),eye=asdict(eye),
                validated=bool(hd3.success and noise_stage.success and eye_pass))
        report["targets"].append(target_row);save()
    report["all_sampled_targets_validated"]=all(r["validated"] for r in report["targets"])
    save();write_report(args.output,report);print(json.dumps({"noise_passes":[r["passes_noise"] for r in report["noise"]],"targets":[{k:r[k] for k in ("frequency_hz","ac_covered","validated")} for r in report["targets"]]}))

if __name__=="__main__":main()

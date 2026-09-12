"""Finalist-only HD3 convergence and longer eye validation; never runs in training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from simulator.config import SimulationConditions, ProcessCorner, Sky130Config
from simulator.receiver import ReceiverParameters, BENCHES, _templates, _run_transient, EvaluationFidelity, SYNTHETIC_CHANNEL
from simulator.models import SimulationRequest
from simulator.ngspice import NgSpiceConfig, run_simulation
from simulator.waveform import parse_wrdata, hd3_db


def hd3_check(parameters,conditions,model,amplitude,step_ps,tight):
    source=(BENCHES/"ctle_hd3.cir").read_text()
    source=source.replace("tran 20p 500n 0 20p",f"tran {step_ps}p 500n 0 {step_ps}p")
    if tight: source=source.replace(".control",".options reltol=1e-5 abstol=1e-12 vntol=1e-8\n.control")
    start=time.perf_counter()
    with TemporaryDirectory(prefix="nebula_hd3_check_") as temp:
        bench=Path(temp)/"hd3.cir";bench.write_text(source)
        values={**parameters.spice_parameters(conditions),"HD3_DIFF_PP":amplitude}
        result=run_simulation(SimulationRequest(bench,values,(),_templates(model,conditions),
            {"HD3_OUTPUT_FILE":"hd3.dat"}),config=NgSpiceConfig(timeout_s=30))
    row={"corner":conditions.process_corner.value,"differential_pp_v":amplitude,
         "maximum_timestep_ps":step_ps,"tight_tolerances":tight,"elapsed_s":time.perf_counter()-start,
         "simulation_success":result.success,"errors":result.errors,"provenance":result.provenance}
    if result.success:
        trace=parse_wrdata(result.artifacts["HD3_OUTPUT_FILE"])
        row["hd3_by_discard_ns"]={str(n):hd3_db(trace,discard_before_s=n*1e-9) for n in (100,200,250)}
        row["all_windows_below_minus30_db"]=all(x < -30 for x in row["hd3_by_discard_ns"].values())
    return row


def summarize_hd3(rows):
    summaries=[]
    for corner in ("tt","ss","ff"):
        for amplitude in (.1,.2):
            matches=[r for r in rows if r["corner"]==corner and r["differential_pp_v"]==amplitude]
            complete=len(matches)==3 and {(r["maximum_timestep_ps"],r["tight_tolerances"]) for r in matches}=={(20,False),(10,True),(5,True)}
            complete=complete and all(r.get("simulation_success") and len(r.get("hd3_by_discard_ns",{}))==3 for r in matches)
            values=[v for r in matches for v in r.get("hd3_by_discard_ns",{}).values()]
            import math
            finite=bool(values) and all(math.isfinite(v) for v in values)
            spread=max(values)-min(values) if finite else None
            summaries.append({"corner":corner,"differential_pp_v":amplitude,
                "complete":complete,"worst_hd3_db":max(values) if finite else None,
                "spread_db":spread,"stable_within_0p1_db":bool(complete and finite and spread<.1),
                "passes_hd3":bool(complete and finite and max(values)<-30)})
    return summaries


def write_report(output,report):
    import html
    rows=[]
    for r in report.get("hd3_summary",[]):
        rows.append("<tr>"+"".join("<td>"+html.escape(str(v))+"</td>" for v in (
            r["corner"],r["differential_pp_v"],r["worst_hd3_db"],r["spread_db"],r["passes_hd3"]))+"</tr>")
    eyes=[]
    for row in report["eye"]:
        m=row["result"]["metrics"]
        eyes.append("<tr>"+"".join("<td>"+html.escape(str(v))+"</td>" for v in (
            row["bits"],row["result"]["success"],m.get("dfe_locked_phase_eye_height_v","not measured"),
            m.get("dfe_eye_width_ui","not measured"),m.get("dfe_error_count","not measured"),round(row["elapsed_s"],2)))+"</tr>")
    body='<meta charset="utf-8"><title>Finalist verification</title><style>body{font:16px system-ui;max-width:1050px;margin:35px auto}td,th{padding:10px;border-bottom:1px solid #ccc}pre{white-space:pre-wrap}</style><h1>Finalist HD3 and eye verification</h1><p>Legacy CTLE, ideal tail current, behavioral DFE; synthetic channel. Nominal 1.8 V, 27 C. This is not full PVT or physical-receiver compliance.</p><h2>HD3 convergence</h2><p>100 MHz; 20/10/5 ps maximum timesteps; tighter tolerances at 10/5 ps; three steady-state analysis windows per run. Empty results mean not measured in this run.</p><table><tr><th>Corner</th><th>Input Vpp differential</th><th>Worst HD3 dB</th><th>Spread dB</th><th>Below -30 dB</th></tr>'+"".join(rows)+'</table><h2>Long eye checks at TT</h2><p>2 ps simulation timestep. Zero observed errors is a finite-pattern result, not a statistical BER guarantee. Timeouts are missing measurements.</p><table><tr><th>Bits</th><th>Stage passed</th><th>DFE eye V</th><th>DFE eye UI</th><th>Errors</th><th>Seconds</th></tr>'+"".join(eyes)+'</table><h2>Parameters</h2><pre>'+html.escape(json.dumps(report["parameters"],indent=2))+'</pre>'
    (Path(output)/"report.html").write_text(body,encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--eye-bits",type=int,nargs="+",default=[512,1024])
    parser.add_argument("--pwl-tolerance",type=float,default=0.)
    parser.add_argument("--skip-hd3",action="store_true",help="eye-only rerun; does not claim HD3 validation")
    parser.add_argument("--eye-timeout",type=float,default=120)
    args=parser.parse_args()
    if any(n<128 or n>65536 for n in args.eye_bits): raise ValueError("eye bits must be 128..65536")
    args.output.mkdir(parents=True,exist_ok=False)
    data=json.loads(args.design.read_text());params=ReceiverParameters(**data["parameters"])
    base=replace(SimulationConditions(**data.get("conditions",{})),stimulus_pwl_tolerance_v=args.pwl_tolerance);base.validate();model=Sky130Config().resolve_model_library()
    report={"parameters":asdict(params),"conditions":base.to_dict(),"hd3":[],"eye":[],
            "amplitude_note":"Both 100 mVpp and 100 mV peak differential (200 mVpp) checked; official convention still requires confirmation."}
    def save(): (args.output/"verification.json").write_text(json.dumps(report,indent=2,default=str))
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(hd3_check,params,replace(base,process_corner=ProcessCorner(c)),model,a,step,tight)
              for c in (() if args.skip_hd3 else ("tt","ss","ff")) for a in (.1,.2) for step,tight in ((20,False),(10,True),(5,True))]
        for job in jobs:
            report["hd3"].append(job.result());save()
    report["hd3_summary"]=summarize_hd3(report["hd3"]);save()
    print("HD3 skipped" if args.skip_hd3 else "HD3 checks complete",flush=True)
    # Run long waveforms serially to avoid competing for memory/CPU.
    for bits in args.eye_bits:
        conditions=replace(base,validation_bit_count=bits)
        start=time.perf_counter()
        result=_run_transient(params,conditions,model,NgSpiceConfig(timeout_s=args.eye_timeout),
            EvaluationFidelity.FINAL,SYNTHETIC_CHANNEL,waveform_output=args.output/f"tt-{bits}-eye.npz")
        report["eye"].append({"bits":bits,"elapsed_s":time.perf_counter()-start,"result":asdict(result)})
        save();write_report(args.output,report);print(json.dumps({"bits":bits,"success":result.success,"seconds":report["eye"][-1]["elapsed_s"]}),flush=True)

if __name__=="__main__":main()

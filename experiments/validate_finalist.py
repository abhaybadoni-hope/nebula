"""Separate, resumable finalist validation. Never runs during quick inference."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from analysis.acceptance import acceptance_violations
from simulator.config import PVT_GRID, SimulationConditions
from simulator.receiver import ReceiverParameters, EvaluationFidelity, SYNTHETIC_CHANNEL
from simulator.runtime import BudgetedEvaluator, SearchLimits
from simulator.physical_receiver import evaluate_physical_receiver
from rl.target_spec import TargetSpec


def scenarios(base, channels, *, full_pvt=False, stress=True):
    jobs=[]
    for channel in channels:
        for condition in (PVT_GRID if full_pvt else (base,)):
            conditions=replace(base,process_corner=condition.process_corner,supply_v=condition.supply_v,temperature_c=condition.temperature_c)
            jobs.append((str(channel),conditions))
        if stress:
            jobs.append((str(channel),replace(base,stimulus_pattern="prbs15",validation_bit_count=4096)))
            jobs.append((str(channel),replace(base,stimulus_pattern="prbs15",stimulus_jitter_rms_s=5e-12,validation_bit_count=4096)))
    return jobs


def validate(parameters, output, *, conditions=SimulationConditions(), channels=(SYNTHETIC_CHANNEL,),
             full_pvt=False, stress=True, limits=SearchLimits(600,8,300), target=None):
    target=target or TargetSpec.from_existing_thresholds()
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    evaluator=BudgetedEvaluator(limits, charge_cache_hits=False)
    jobs=scenarios(conditions,channels,full_pvt=full_pvt,stress=stress)
    rows=[]
    report={"parameters":asdict(parameters),"planned":len(jobs),"completed":0,"all_requested_pass":False,"rows":rows}
    # Persistent evaluator cache resumes identical successful or deterministic failed jobs.
    for channel,condition in jobs:
        if not evaluator.available(): break
        result=evaluator(parameters,condition,EvaluationFidelity.FINAL,channel_path=channel)
        violations=acceptance_violations(result.metrics,target,final=True)
        rows.append({"channel":channel,"conditions":condition.to_dict(),"success":result.success and not violations,
                     "violations":violations,"evaluation":result.to_dict()})
        report={"parameters":asdict(parameters),"target":target.as_dict(),"planned":len(jobs),"completed":len(rows),
                "all_requested_pass":len(rows)==len(jobs) and all(r["success"] for r in rows),
                "runtime":evaluator.summary(),"rows":rows}
        temporary=output.with_suffix(output.suffix+".tmp")
        temporary.write_text(json.dumps(report,indent=2,default=str)); temporary.replace(output)
    report["runtime"] = evaluator.summary()
    output.write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design",type=Path,required=True,help="JSON containing parameters and optional conditions")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--channel",type=Path,action="append")
    p.add_argument("--full-pvt",action="store_true")
    p.add_argument("--seconds",type=float,default=600)
    p.add_argument("--stage-timeout",type=float,default=300,help="finalist-only maximum seconds per simulator stage")
    p.add_argument("--max-evaluations",type=int,default=8)
    p.add_argument("--report",type=Path,help="write a measured nominal HTML report")
    p.add_argument("--tuning",action="store_true",help="separate 120-second sampled tuning check")
    p.add_argument("--physical-receiver",action="store_true",help="also run the experimental transistor sampler/DFE")
    args=p.parse_args(); data=json.loads(args.design.read_text())
    params=ReceiverParameters(**data["parameters"])
    conditions=SimulationConditions(**data.get("conditions",{}))
    result=validate(params,args.output,conditions=conditions,channels=args.channel or (SYNTHETIC_CHANNEL,),
                    full_pvt=args.full_pvt,limits=SearchLimits(args.seconds,args.max_evaluations,args.stage_timeout))
    if args.report and result["rows"]:
        from analysis.design_report import write_report
        write_report(params,conditions,result["rows"][0]["evaluation"]["metrics"],args.report,
                     source="Final-fidelity nominal evaluation; see JSON for full scenario status",runtime=result["runtime"])
    if args.tuning:
        from experiments.tuning_sweep import characterize_tuning
        result["tuning"] = characterize_tuning(params,conditions)
        args.output.write_text(json.dumps(result,indent=2,default=str))
    if args.physical_receiver:
        result["physical_receiver"]=evaluate_physical_receiver(params,conditions,channel_path=(args.channel or [SYNTHETIC_CHANNEL])[0])
        args.output.write_text(json.dumps(result,indent=2,default=str))
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))

if __name__ == "__main__": main()

"""Offline, budget-matched RS/CEM/PPO benchmark for the nine-variable sizing space."""
import argparse
import json
from pathlib import Path
import random
from simulator.runtime import BudgetedEvaluator, SearchLimits
from simulator.receiver import EvaluationFidelity
from analysis.acceptance import acceptance_violations
from rl.sized_receiver import BOUNDS, SizedReceiverEnv, decode, load_policy
from rl.target_spec import TargetSpec
from rl.autockt_state import spec_error_vector


def trial(method,seed,target,limits,*,checkpoint=None,evaluator=None):
    rng=random.Random(seed)
    evaluator=evaluator or BudgetedEvaluator(limits)
    rows=[]
    mean=[0.]*len(BOUNDS); std=[1/(3**.5)]*len(BOUNDS); population=[]
    if method=="ppo":
        if checkpoint is None: raise ValueError("PPO comparison requires a trained sizing checkpoint")
        agent=load_policy(checkpoint,seed); env=SizedReceiverEnv((target,),evaluator=evaluator,seed=seed,step_size=agent.step_size)
        state,_=env.reset()
    while evaluator.available():
        if method=="ppo":
            choices,_,_=agent.act(state,deterministic=True); step=env.step(choices)
            state=step.state; metrics=step.info["metrics"]; success=step.done
            if step.done or step.truncated: state,_=env.reset()
        else:
            action=[rng.uniform(-1,1) for _ in BOUNDS] if method=="random" else [max(-1,min(1,rng.gauss(m,s))) for m,s in zip(mean,std)]
            parameters,conditions=decode(action)
            result=evaluator(parameters,conditions,EvaluationFidelity.TRAINING)
            metrics=result.metrics; success=result.success and not acceptance_violations(metrics,target)
            score=sum(max(-1,min(0,e)) for e in spec_error_vector(metrics,target))-(0 if result.success else 1)
            population.append((score,action))
            if method=="cem" and len(population)==5:
                elite=sorted(population,key=lambda x:x[0],reverse=True)[:2]
                mean=[sum(a[i] for _,a in elite)/len(elite) for i in range(len(BOUNDS))]
                std=[max(.05,(sum((a[i]-mean[i])**2 for _,a in elite)/len(elite))**.5) for i in range(len(BOUNDS))]
                population=[]
        rows.append({"evaluation":len(rows)+1,"success":bool(success),"metrics":metrics,
                     "elapsed_s":evaluator.summary()["elapsed_s"]})
    successes=[r for r in rows if r["success"]]
    return {"method":method,"seed":seed,"target":target.as_dict(),"runtime":evaluator.summary(),
            "first_success_evaluation":successes[0]["evaluation"] if successes else None,
            "first_success_s":successes[0]["elapsed_s"] if successes else None,
            "success_rate":len(successes)/len(rows) if rows else None,"rows":rows,
            "criterion":"strict nominal TRAINING-fidelity constraints; final validation separate"}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--targets",type=Path,help="JSON list of held-out target dictionaries; keep separate from training targets")
    p.add_argument("--checkpoint",type=Path); p.add_argument("--seeds",type=int,nargs="+",default=[11,23,47])
    p.add_argument("--seconds-per-trial",type=float,default=120); p.add_argument("--evaluations",type=int,default=20)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    rows=[]; methods=("random","cem","ppo") if a.checkpoint else ("random","cem")
    targets=[TargetSpec(**t) for t in json.loads(a.targets.read_text())] if a.targets else [TargetSpec.from_existing_thresholds()]
    for seed in a.seeds:
        for target in targets:
          for method in methods:
            rows.append(trial(method,seed,target,SearchLimits(a.seconds_per_trial,a.evaluations),checkpoint=a.checkpoint))
            a.output.parent.mkdir(parents=True,exist_ok=True)
            a.output.write_text(json.dumps({"trials":rows,"bounds":BOUNDS,
                "caveats":["PPO training cost is excluded from inference trials and must be reported separately", "PPO takes local actions; RS/CEM propose global points", "cache hits reported separately; compare cold runs separately", "no generalization claim without held-out targets"]},indent=2))

if __name__=="__main__": main()

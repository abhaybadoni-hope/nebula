"""Fast inference for an explicitly trained grouped-sizing policy."""
import argparse
import json
from pathlib import Path
from rl.sized_receiver import SizedReceiverEnv, load_policy
from rl.target_spec import TargetSpec
from simulator.runtime import BudgetedEvaluator, SearchLimits

def design(checkpoint, *, target=None, limits=SearchLimits(), seed=42, step_size=None):
    agent=load_policy(checkpoint,seed)
    evaluator=BudgetedEvaluator(limits)
    env=SizedReceiverEnv((target or TargetSpec.from_existing_thresholds(),), evaluator=evaluator,seed=seed,step_size=agent.step_size if step_size is None else step_size)
    state,_=env.reset()
    candidates=[]
    while evaluator.available():
        choices,_,_=agent.act(state,deterministic=True)
        step=env.step(choices)
        candidates.append(step.info)
        state=step.state
        if step.done: break
        if step.truncated: state,_=env.reset()
    selected=next((c for c in candidates if c["spec_satisfied"]),None)
    return {"selected":selected,"candidates":candidates,"runtime":evaluator.summary(),
            "step_size":env.step_size, "validation":"nominal training fidelity only; DFE is behavioral; full validation required"}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--design-output",type=Path,help="export selected parameters/conditions for validation")
    p.add_argument("--step-size",type=float,help="override checkpoint step size; may affect policy quality")
    p.add_argument("--seconds",type=float,default=120)
    p.add_argument("--max-evaluations",type=int,default=4)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result=design(args.checkpoint,limits=SearchLimits(args.seconds,args.max_evaluations),step_size=args.step_size)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    if args.design_output and result["selected"]:
        args.design_output.parent.mkdir(parents=True,exist_ok=True)
        args.design_output.write_text(json.dumps({k:result["selected"][k] for k in ("parameters","conditions")},indent=2))
    print(json.dumps(result["runtime"]))

if __name__ == "__main__": main()

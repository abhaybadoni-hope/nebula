"""Offline training for the physical-bias, grouped-sizing policy."""
import argparse
import json
from pathlib import Path
from rl.sized_receiver import SizedReceiverEnv, STATE_DIM, BOUNDS, save_policy, DEFAULT_STEP_SIZE
from rl.ppo_agent import PPOAgent
from rl.trainer import train, DEFAULT_PPO_EPOCHS, DEFAULT_MINIBATCH_SIZE
from rl.target_spec import TargetSpec
from simulator.runtime import BudgetedEvaluator, SearchLimits, SearchBudgetExpired

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers",type=int,choices=(1,2),default=2,help="independent SPICE episodes in parallel")
    p.add_argument("--updates", type=int, default=4)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--seconds", type=float, default=600)
    p.add_argument("--max-evaluations", type=int, default=64)
    p.add_argument("--step-size",type=float,default=DEFAULT_STEP_SIZE)
    p.add_argument("--seed", type=int, default=42)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    evaluator=BudgetedEvaluator(SearchLimits(args.seconds,args.max_evaluations))
    env=SizedReceiverEnv((TargetSpec.from_existing_thresholds(),TargetSpec.from_hard_target()),
                         evaluator=evaluator, seed=args.seed,horizon=args.horizon,step_size=args.step_size)
    agent=PPOAgent(STATE_DIM,len(BOUNDS),seed=args.seed)
    rows=[]
    updates=[]
    try:
        if args.workers == 1:
            train(env,agent,num_updates=args.updates,episodes_per_update=args.episodes,on_step=rows.append,on_update=updates.append)
        else:
            from rl.parallel_rollout import collect_parallel
            import time
            for update in range(args.updates):
                started=time.perf_counter()
                def factory(index):
                    return SizedReceiverEnv(env.target_pool,evaluator=evaluator,
                        seed=args.seed+update*args.episodes+index,horizon=args.horizon,step_size=args.step_size)
                transitions=collect_parallel(factory,agent,episodes=args.episodes,workers=args.workers,on_step=rows.append)
                stats=agent.update(transitions,0.,epochs=DEFAULT_PPO_EPOCHS,minibatch_size=DEFAULT_MINIBATCH_SIZE)
                updates.append({"update":update,"transitions":len(transitions),
                                "wall_clock_s":time.perf_counter()-started,**stats})
    except SearchBudgetExpired:
        pass
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if updates:
        save_policy(agent,args.output,step_size=args.step_size)
    args.output.with_suffix(".json").write_text(json.dumps({"runtime":evaluator.summary(),"seed":args.seed,"step_size":args.step_size,"workers":args.workers,"completed_updates":len(updates),"checkpoint_saved":bool(updates),"updates":updates,"steps":rows},indent=2))
    print(json.dumps(evaluator.summary()))

if __name__ == "__main__": main()

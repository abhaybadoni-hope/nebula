"""Parallel independent episodes; policy sampling and updates stay on the caller thread."""
from concurrent.futures import ThreadPoolExecutor
from .ppo_agent import Transition


def collect_parallel(env_factory, agent, *, episodes, workers=2, on_step=None):
    if workers not in (1, 2) or episodes < 1:
        raise ValueError("positive episodes and one or two workers required")
    transitions=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, episodes, workers):
            envs=[env_factory(i) for i in range(start,min(start+workers,episodes))]
            resets=[env.reset() for env in envs]
            states=[r[0] for r in resets]
            active=list(range(len(envs)))
            trajectories=[[] for _ in envs]
            while active:
                # Sample serially: no shared torch RNG or policy mutation in workers.
                actions={i:agent.act(states[i]) for i in active}
                pending={i:pool.submit(envs[i].step,actions[i][0]) for i in active}
                remaining=[]
                for i in active:
                    step=pending[i].result()
                    choices,log_prob,value=actions[i]
                    bootstrap=agent.act(step.state)[2] if step.truncated and not step.done else 0.
                    trajectories[i].append(Transition(states[i],choices,log_prob,value,step.reward,
                                                       step.done,step.truncated,bootstrap))
                    if on_step:
                        on_step({"episode":start+i,"reward":step.reward,"target":resets[i][1]["target"],**step.info})
                    states[i]=step.state
                    if not step.done and not step.truncated: remaining.append(i)
                active=remaining
            # Keep each trajectory contiguous so GAE cannot mix independent episodes.
            for trajectory in trajectories: transitions.extend(trajectory)
    return transitions

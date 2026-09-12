"""Versioned PPO environment for grouped MOS sizing and physical CTLE bias.

This is a separate nine-head policy: legacy five-head checkpoints are rejected.
Training is offline. Inference uses a bounded number of real SPICE evaluations.
"""
from dataclasses import asdict
import math
import random
from simulator.config import SimulationConditions
from simulator.receiver import ReceiverParameters, EvaluationFidelity
from simulator.sizing import CircuitSizing
from simulator.runtime import BudgetedEvaluator, SearchLimits
from analysis.acceptance import acceptance_violations
from rl.autockt_env import AutoCktStep
from rl.autockt_state import spec_error_vector, target_reference_vector
from rl.target_spec import TargetSpec

SCHEMA = "nebula_physical_bias_sizing_v1"
BOUNDS = (
    ("rload_ohm", 100., 10000., "log"),
    ("rdeg_ohm", 10., 10000., "log"),
    ("cdeg_f", 1e-14, 1e-11, "log"),
    ("dfe_tap_v", -.4, .4, "linear"),
    ("input_width_um", .42, 100., "log"),
    ("input_length_um", .15, 2., "log"),
    ("tail_width_um", .42, 100., "log"),
    ("tail_length_um", .15, 2., "log"),
    ("bias_resistance_ohm", 100., 1e6, "log"),
)
DEFAULT_STEP_SIZE = 0.2

STATE_DIM = 8 + len(BOUNDS) + 1
STAGES = (None, "setup", "dc", "ac", "ctle_transient", "channel", "noise", "hd3", "transient", "budget")

def decode(action):
    if len(action) != len(BOUNDS): raise ValueError("sizing action requires nine dimensions")
    values = {}
    for x, (name, low, high, scale) in zip(action, BOUNDS):
        if not math.isfinite(x) or not -1 <= x <= 1: raise ValueError("action outside [-1, 1]")
        fraction = (x+1)/2
        values[name] = low * (high/low)**fraction if scale == "log" else low + fraction*(high-low)
    parameters = ReceiverParameters(**{k: values[k] for k in ("rload_ohm", "rdeg_ohm", "cdeg_f", "dfe_tap_v")})
    sizing = CircuitSizing(**{k: v for k, v in values.items() if k not in parameters.__dict__}, physical_bias=True)
    return parameters, SimulationConditions(circuit_sizing=sizing)

class SizedReceiverEnv:
    def __init__(self, target_pool, *, evaluator=None, seed=42, horizon=8, step_size=DEFAULT_STEP_SIZE):
        if not math.isfinite(step_size) or not 0 < step_size <= 2:
            raise ValueError("step_size must be finite and in (0, 2]")
        self.step_size = step_size
        self.target_pool = tuple(target_pool)
        if not self.target_pool or horizon < 1: raise ValueError("targets and positive horizon required")
        self.evaluator = evaluator or BudgetedEvaluator(SearchLimits())
        self.rng = random.Random(seed)
        self.horizon = horizon
        self.evaluations = 0

    def state(self, metrics, failed_stage):
        stage = STAGES.index(failed_stage) if failed_stage in STAGES else len(STAGES)
        return spec_error_vector(metrics, self.target) + target_reference_vector(self.target) + tuple(self.action) + (stage/len(STAGES),)

    def reset(self, *, target=None):
        self.target = target or self.rng.choice(self.target_pool)
        self.action = [self.rng.uniform(-1., 1.) for _ in BOUNDS]
        self.steps = 0
        return self.state({}, "setup"), {"target": self.target.as_dict()}

    def step(self, choices):
        if len(choices) != len(BOUNDS) or any(c not in (0,1,2) for c in choices):
            raise ValueError("nine discrete choices (0,1,2) required")
        self.action = [max(-1., min(1., x + self.step_size*(c-1))) for x,c in zip(self.action,choices)]
        parameters, conditions = decode(self.action)
        evaluation = self.evaluator(parameters, conditions, EvaluationFidelity.TRAINING)
        self.evaluations += 1
        self.steps += 1
        metrics = evaluation.metrics
        done = evaluation.success and not acceptance_violations(metrics, self.target)
        # Graded valid-spec shortfall; simulator failures remain unsuccessful.
        errors = spec_error_vector(metrics, self.target)
        penalty = sum(max(-1., min(0., e)) if math.isfinite(e) else -1. for e in errors)
        reward = 10. if done else penalty - (0. if evaluation.success else 1.)
        info = dict(success=evaluation.success, spec_satisfied=done, step_count=self.steps,
            failure_stage=evaluation.failed_stage, total_evaluation_count=self.evaluations,
            parameters=asdict(parameters), conditions=conditions.to_dict(),
            indices=tuple(self.action), metrics=dict(metrics))
        return AutoCktStep(self.state(metrics, evaluation.failed_stage), reward, done,
                          self.steps >= self.horizon, info)

def save_policy(agent, path, *, step_size=DEFAULT_STEP_SIZE):
    import torch
    torch.save({"schema": SCHEMA, "bounds": BOUNDS, "step_size": step_size, "policy": agent.policy.state_dict()}, path)

def load_policy(path, seed=42):
    import torch
    from rl.ppo_agent import PPOAgent
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != SCHEMA or tuple(checkpoint.get("bounds", ())) != BOUNDS:
        raise ValueError("checkpoint is not a compatible grouped-sizing policy; train a new sizing policy")
    agent = PPOAgent(STATE_DIM, len(BOUNDS), seed=seed)
    agent.step_size = checkpoint.get("step_size", 0.1)
    if not math.isfinite(agent.step_size) or not 0 < agent.step_size <= 2:
        raise ValueError("invalid checkpoint step size")
    agent.policy.load_state_dict(checkpoint["policy"])
    return agent

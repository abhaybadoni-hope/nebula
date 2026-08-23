"""Simulator interfaces used by circuit optimizers."""

from .models import FailureCode, FailureStage, SimulationRequest, SimulationResult, SimulationStatus
from .ngspice import NgSpiceConfig, ngspice_identity, run_ngspice, run_simulation
from .config import OUTPUT_LOAD_GRID_F, PVT_GRID, ProcessCorner, SimulationConditions, Sky130Config
from .ctle import CTLEEvaluation, evaluate_ctle
from .cache import EvaluationCache
from .channel import ChannelPortMap, validate_s4p_channel
from .receiver import (
    EvaluationFidelity,
    ReceiverEvaluation,
    ReceiverParameters,
    evaluate_pvt_grid,
    evaluate_receiver,
)
from .rl_adapter import (
    ACTION_BOUNDS, ACTION_SCHEMA_VERSION, CONSTRAINT_NAMES, OBSERVATION_NAMES,
    OBSERVATION_SCHEMA_VERSION, REWARD_VERSION, RLBudget, RLStep,
    ReceiverRLAdapter, normalized_action_to_parameters,
)

__all__ = [
    "FailureCode",
    "FailureStage",
    "NgSpiceConfig",
    "ngspice_identity",
    "SimulationRequest",
    "SimulationResult",
    "SimulationStatus",
    "Sky130Config",
    "ProcessCorner",
    "SimulationConditions",
    "PVT_GRID",
    "OUTPUT_LOAD_GRID_F",
    "EvaluationCache",
    "ChannelPortMap",
    "validate_s4p_channel",
    "CTLEEvaluation",
    "evaluate_ctle",
    "EvaluationFidelity",
    "ReceiverEvaluation",
    "ReceiverParameters",
    "evaluate_receiver",
    "evaluate_pvt_grid",
    "run_ngspice",
    "run_simulation",
    "ACTION_BOUNDS",
    "ACTION_SCHEMA_VERSION",
    "CONSTRAINT_NAMES",
    "OBSERVATION_NAMES",
    "OBSERVATION_SCHEMA_VERSION",
    "REWARD_VERSION",
    "RLBudget",
    "RLStep",
    "ReceiverRLAdapter",
    "normalized_action_to_parameters",
]

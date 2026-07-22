"""Collision-anchored accident reconstruction.

The package deliberately keeps its geometry and optimisation core independent
from CARLA.  CARLA objects are accepted through duck-typed adapters at the
boundary, which makes the fitting logic deterministic and unit-testable without
starting the simulator.
"""

from .models import (
    AccidentSpecification,
    ActorBehaviorParameters,
    ActorDimensions,
    ActorState,
    CollisionAnchor,
    CollisionPairHypothesis,
    PipelineResult,
    ReferencePath,
    RolloutResult,
    SceneState,
    SolverResult,
)
from .pipeline import CollisionAnchoredFittingPipeline
from .specification import build_accident_specification, build_scene_state

__all__ = [
    "AccidentSpecification",
    "ActorBehaviorParameters",
    "ActorDimensions",
    "ActorState",
    "CollisionAnchor",
    "CollisionPairHypothesis",
    "CollisionAnchoredFittingPipeline",
    "PipelineResult",
    "ReferencePath",
    "RolloutResult",
    "SceneState",
    "SolverResult",
    "build_accident_specification",
    "build_scene_state",
]

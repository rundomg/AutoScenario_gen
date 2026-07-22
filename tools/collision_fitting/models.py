"""Shared data contracts for collision-anchored fitting."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


Point2D = Tuple[float, float]


@dataclass
class ActorDimensions:
    """Full vehicle dimensions in metres (not CARLA half-extents)."""

    length: float = 4.6
    width: float = 1.9
    height: float = 1.6


@dataclass
class ActorState:
    actor_id: str
    x: float
    y: float
    yaw_deg: float
    speed_mps: float = 0.0
    category: str = "vehicle"
    lane_id: Optional[int] = None
    road_id: Optional[int] = None
    speed_limit_mps: Optional[float] = None
    dimensions: ActorDimensions = field(default_factory=ActorDimensions)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollisionPairHypothesis:
    actor_a: str
    actor_b: str
    accident_type: str
    confidence: float = 0.5
    striking_actor_id: Optional[str] = None
    struck_actor_id: Optional[str] = None
    evidence: str = ""


@dataclass
class AccidentSpecification:
    accident_type: str
    striking_actor_id: str
    struck_actor_id: str
    actor_maneuvers: Dict[str, str] = field(default_factory=dict)
    behavior_sequences: Dict[str, List[str]] = field(default_factory=dict)
    relative_motion_constraints: List[Dict[str, Any]] = field(default_factory=list)
    event_order: List[Dict[str, Any]] = field(default_factory=list)
    collision_time_range: Tuple[float, float] = (2.0, 4.0)
    confidence: Dict[str, float] = field(default_factory=dict)
    pair_hypotheses: List[CollisionPairHypothesis] = field(default_factory=list)
    expected_contact_sides: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReferencePath:
    actor_id: str
    waypoints: List[Point2D]
    cumulative_distances: List[float]
    lane_ids: List[Optional[int]] = field(default_factory=list)
    road_ids: List[Optional[int]] = field(default_factory=list)
    maneuver: str = "keep_lane"
    source: str = "fallback_heading"

    @property
    def length(self) -> float:
        return self.cumulative_distances[-1] if self.cumulative_distances else 0.0


@dataclass
class CollisionAnchor:
    accident_type: str
    conflict_position: Point2D
    actor_path_distances: Dict[str, float]
    target_time_range: Tuple[float, float]
    expected_contact_sides: Dict[str, str] = field(default_factory=dict)
    conflict_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    actor_start_offsets: Dict[str, float] = field(default_factory=dict)
    initial_clearance_m: Optional[float] = None
    feasible: bool = True
    reason: str = ""


@dataclass
class ActorBehaviorParameters:
    initial_speed: float
    target_speed: float
    acceleration: float = 0.0
    deceleration: float = 0.0
    brake_start_time: float = math.inf
    start_delay: float = 0.0
    phase_durations: List[float] = field(default_factory=list)
    lane_change_start_time: Optional[float] = None
    lane_change_duration: Optional[float] = None


@dataclass
class SceneState:
    actors: Dict[str, ActorState]
    duration_s: float = 6.0
    fixed_delta_seconds: float = 0.05
    map_name: Optional[str] = None
    default_speed_limit_mps: float = 13.9
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutResult:
    collided: bool
    collision_pair: Optional[Tuple[str, str]] = None
    collision_time: Optional[float] = None
    collision_location: Optional[Point2D] = None
    relative_impact_speed: Optional[float] = None
    trajectory_logs: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    minimum_pair_distance: float = math.inf
    relation_errors: Dict[str, float] = field(default_factory=dict)
    arrival_times: Dict[str, Optional[float]] = field(default_factory=dict)
    non_target_collisions: List[Tuple[str, str]] = field(default_factory=list)
    contact_sides: Dict[str, str] = field(default_factory=dict)
    control_smoothness: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SolverResult:
    parameters: Dict[str, ActorBehaviorParameters]
    rollout: RolloutResult
    loss: float
    evaluations: int
    converged: bool
    loss_components: Dict[str, float] = field(default_factory=dict)
    repair_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PipelineResult:
    specification: AccidentSpecification
    paths: Dict[str, ReferencePath]
    anchor: CollisionAnchor
    solver_result: SolverResult
    hypotheses_evaluated: int = 1
    hypothesis_summaries: List[Dict[str, Any]] = field(default_factory=list)


def to_jsonable(value: Any) -> Any:
    """Recursively convert contracts to strict JSON-compatible values."""

    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def pair_equal(
    pair_a: Optional[Sequence[str]], pair_b: Optional[Sequence[str]]
) -> bool:
    if pair_a is None or pair_b is None or len(pair_a) != 2 or len(pair_b) != 2:
        return False
    return {str(pair_a[0]), str(pair_a[1])} == {str(pair_b[0]), str(pair_b[1])}

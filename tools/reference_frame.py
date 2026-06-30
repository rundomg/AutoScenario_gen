"""Structural reference frames for actor placement.

This module decouples *where the scene matched* from *where each actor is
placed*.  The map matcher (Layer 1) is expected to emit a ``matched_structure``
describing the matched road structure -- either a junction (its centre plus the
enumerated legs) or an open road segment (a centreline with a longitudinal ``s``
axis).  Given such a structure we build a :class:`ReferenceFrame` and place each
actor *inside* that frame instead of smearing every actor along a single anchor
lane axis.

Two design rules underpin everything here:

1.  **Heading is never extrapolated geometrically.**  An actor's yaw is the
    legal travel direction of the lane it occupies (``legal_flow_heading``),
    optionally flipped 180 degrees when the actor is genuinely driving against
    that flow (``flow_compliance == "wrong_way"``).  We never synthesise
    ``anchor_yaw + 90`` for crossing traffic -- crossing traffic is placed on a
    real side leg whose own direction supplies the yaw.

2.  **"Which lane band" and "which direction it faces" are independent.**  A
    wrong-way motorcycle occupies the lane band implied by its image position
    but faces against that band's legal flow.  Conflating the two is what
    erased real wrong-way actors (and what put cross-leg cars sideways in front
    of ego) in the single-axis design.

Coordinate convention matches the rest of the codebase / CARLA: x east, y
south, yaw clockwise in degrees, so the right-hand perpendicular of a heading
``h`` is the direction ``h + 90``.

No CARLA import here -- the frame is built from plain numbers that the matcher
fills from real waypoints; this keeps the whole module unit-testable offline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def normalize_angle(deg: float) -> float:
    """Wrap an angle to (-180, 180]."""
    angle = float(deg) % 360.0
    if angle > 180.0:
        angle -= 360.0
    return angle


def angle_difference(a: float, b: float) -> float:
    """Absolute smallest difference between two headings, in [0, 180]."""
    return abs(normalize_angle(a - b))


def _unit(heading_deg: float) -> Tuple[float, float]:
    rad = math.radians(heading_deg)
    return math.cos(rad), math.sin(rad)


def _offset_point(
    base: Tuple[float, float, float],
    along_heading: float,
    along: float,
    lateral: float,
) -> Tuple[float, float, float]:
    """Move ``base`` ``along`` metres in ``along_heading`` then ``lateral``
    metres to the right of that heading (right = heading + 90)."""
    fx, fy = _unit(along_heading)
    rx, ry = _unit(along_heading + 90.0)
    return (
        base[0] + fx * along + rx * lateral,
        base[1] + fy * along + ry * lateral,
        base[2],
    )


def heading_for(legal_flow_heading: float, flow_compliance: Any) -> float:
    """Resolve an actor yaw from the legal flow of its lane.

    ``flow_compliance``:
        ``"wrong_way"`` -> face against the lane (legal flow + 180).
        anything else (``"legal"`` / ``"unknown"`` / None) -> follow the lane.
    """
    yaw = float(legal_flow_heading)
    if str(flow_compliance or "").lower() == "wrong_way":
        yaw += 180.0
    return normalize_angle(yaw)


# Result of placing a single actor inside a reference frame.
@dataclass
class Placement:
    location: Dict[str, float]
    yaw: float
    legal_flow_heading: float
    flow_compliance: str
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "location": dict(self.location),
            "rotation": {"pitch": 0.0, "yaw": self.yaw, "roll": 0.0},
            "legal_flow_heading": self.legal_flow_heading,
            "flow_compliance": self.flow_compliance,
            "debug": dict(self.debug),
        }


# --------------------------------------------------------------------------- #
# Junction frame
# --------------------------------------------------------------------------- #
@dataclass
class JunctionLeg:
    """One arm of a junction.

    ``heading_out_deg`` points from the junction centre outward along the leg.
    A vehicle *approaching* the junction therefore travels at
    ``heading_out_deg + 180`` (inbound); a vehicle *leaving* travels at
    ``heading_out_deg``.
    """

    name: str
    heading_out_deg: float
    inbound_lanes: List[Dict[str, Any]] = field(default_factory=list)
    outbound_lanes: List[Dict[str, Any]] = field(default_factory=list)

    def legal_flow_heading(self, motion: str) -> float:
        """Legal travel heading for the lane an actor uses on this leg.

        ``motion == "leaving"`` -> outbound; otherwise inbound (approaching).
        """
        if str(motion or "approaching").lower() == "leaving":
            return normalize_angle(self.heading_out_deg)
        return normalize_angle(self.heading_out_deg + 180.0)

    def lanes_for_motion(self, motion: str) -> List[Dict[str, Any]]:
        if str(motion or "approaching").lower() == "leaving":
            return list(self.outbound_lanes or [])
        return list(self.inbound_lanes or [])


@dataclass
class JunctionFrame:
    center: Tuple[float, float, float]
    legs: List[JunctionLeg]
    ego_approach_heading_deg: float  # the heading at which ego approaches (inbound)

    def assign_leg(self, direction: str) -> Optional[JunctionLeg]:
        """Map an ego-relative direction to the nearest real leg.

        ``direction`` is one of ``ahead`` / ``opposite`` / ``left`` / ``right``,
        interpreted from ego's point of view.  Returns ``None`` when no leg is
        close enough (e.g. a missing arm on a T-junction).
        """
        target_out = self._target_out_heading(direction)
        if target_out is None:
            return None
        best: Optional[JunctionLeg] = None
        best_diff = 60.0  # tolerance: legs further than this are not a match
        for leg in self.legs:
            diff = angle_difference(leg.heading_out_deg, target_out)
            if diff < best_diff:
                best_diff = diff
                best = leg
        return best

    def _target_out_heading(self, direction: str) -> Optional[float]:
        # Ego approaches inbound at ego_approach_heading_deg, so ego's own leg
        # points outward in the opposite direction.
        ego_leg_out = normalize_angle(self.ego_approach_heading_deg + 180.0)
        d = str(direction or "").lower()
        if d in {"ahead", "through", "opposite", "oncoming"}:
            # The leg directly across the junction points the same way ego travels.
            return normalize_angle(self.ego_approach_heading_deg)
        if d == "left":
            return normalize_angle(ego_leg_out + 90.0)
        if d == "right":
            return normalize_angle(ego_leg_out - 90.0)
        if d in {"ego", "behind"}:
            return ego_leg_out
        return None

    def place(
        self,
        *,
        direction: Optional[str] = None,
        leg: Optional[JunctionLeg] = None,
        distance_m: float,
        lateral_m: float = 0.0,
        motion: str = "approaching",
        flow_compliance: Any = "legal",
        z: float = 0.3,
    ) -> Optional[Placement]:
        """Place an actor on a junction leg.

        Position is ``distance_m`` from the centre along the chosen leg, offset
        ``lateral_m`` to the right of the *outbound* direction (so lane-band
        semantics stay in the road frame, not the actor frame).  Yaw comes from
        the leg's legal flow, flipped for wrong-way actors.
        """
        chosen = leg if leg is not None else (self.assign_leg(direction) if direction else None)
        if chosen is None:
            return None
        x, y, _z = _offset_point(
            (self.center[0], self.center[1], self.center[2]),
            chosen.heading_out_deg,
            float(distance_m),
            float(lateral_m),
        )
        legal_flow = chosen.legal_flow_heading(motion)
        yaw = heading_for(legal_flow, flow_compliance)
        return Placement(
            location={"x": x, "y": y, "z": float(z)},
            yaw=yaw,
            legal_flow_heading=legal_flow,
            flow_compliance=str(flow_compliance or "legal"),
            debug={
                "frame": "junction",
                "leg": chosen.name,
                "leg_heading_out": chosen.heading_out_deg,
                "direction": direction,
                "motion": motion,
                "distance_m": float(distance_m),
                "lateral_m": float(lateral_m),
            },
        )


# --------------------------------------------------------------------------- #
# Road segment frame (straight / curve)
# --------------------------------------------------------------------------- #
@dataclass
class RoadFrame:
    """Open road reference frame anchored at the ego point.

    ``forward_heading_deg`` is the lane tangent at the ego point (s = 0).  For a
    curve, supply ``samples`` as a sorted list of ``(s_metres, heading_deg)``
    pairs; the tangent used at a given longitudinal position is the nearest
    sample, so vehicles follow the bend instead of a single straight axis.
    """

    origin: Tuple[float, float, float]
    forward_heading_deg: float
    samples: List[Tuple[float, float]] = field(default_factory=list)

    def tangent_at(self, s: float) -> float:
        if not self.samples:
            return normalize_angle(self.forward_heading_deg)
        nearest = min(self.samples, key=lambda pair: abs(pair[0] - s))
        return normalize_angle(nearest[1])

    def _point_at(self, s: float, lateral: float) -> Tuple[float, float, float]:
        """Integrate along sampled tangents up to ``s`` then offset laterally."""
        if not self.samples:
            return _offset_point(self.origin, self.forward_heading_deg, s, lateral)
        # Walk forward (or backward) accumulating segment vectors so a curved
        # centreline bends rather than drifting off a fixed axis.
        x, y, z = self.origin
        remaining = s
        prev_s = 0.0
        ordered = sorted(self.samples, key=lambda pair: pair[0])
        if s >= 0:
            for sample_s, heading in ordered:
                if sample_s <= 0:
                    continue
                step = min(sample_s, s) - prev_s
                if step <= 0:
                    prev_s = sample_s
                    continue
                fx, fy = _unit(heading)
                x += fx * step
                y += fy * step
                prev_s = sample_s
                remaining = s - prev_s
                if prev_s >= s:
                    remaining = 0.0
                    break
            if remaining > 0:
                fx, fy = _unit(self.tangent_at(s))
                x += fx * remaining
                y += fy * remaining
        else:
            fx, fy = _unit(self.tangent_at(s))
            x += fx * s
            y += fy * s
        tangent = self.tangent_at(s)
        rx, ry = _unit(tangent + 90.0)
        return (x + rx * lateral, y + ry * lateral, z)

    def place(
        self,
        *,
        longitudinal_m: float,
        lateral_m: float = 0.0,
        side: str = "same",
        flow_compliance: Any = "legal",
        z: float = 0.3,
    ) -> Placement:
        """Place an actor along the road.

        ``side == "opposing"`` puts the actor in the opposing carriageway whose
        legal flow is reversed; ``"same"`` keeps ego's direction.  Wrong-way
        actors stay on whichever side their lane band implies and only flip yaw.
        """
        x, y, zz = self._point_at(float(longitudinal_m), float(lateral_m))
        tangent = self.tangent_at(float(longitudinal_m))
        legal_flow = (
            normalize_angle(tangent + 180.0)
            if str(side or "same").lower() == "opposing"
            else normalize_angle(tangent)
        )
        yaw = heading_for(legal_flow, flow_compliance)
        return Placement(
            location={"x": x, "y": y, "z": float(z if z is not None else zz)},
            yaw=yaw,
            legal_flow_heading=legal_flow,
            flow_compliance=str(flow_compliance or "legal"),
            debug={
                "frame": "road",
                "side": side,
                "longitudinal_m": float(longitudinal_m),
                "lateral_m": float(lateral_m),
                "tangent_heading": tangent,
            },
        )


# --------------------------------------------------------------------------- #
# Factory + consistency guard
# --------------------------------------------------------------------------- #
def build_reference_frame(matched_structure: Dict[str, Any]) -> Any:
    """Build a junction or road frame from a matcher ``matched_structure``.

    Expected schema (see map_match_structural_anchor_plan.md section 4.1):

      kind == "junction":
        center: {x,y,z}
        ego_approach_heading_deg: float
        legs: [{name, heading_out_deg}, ...]

      kind == "road_segment":
        origin: {x,y,z}                 (ego point)
        forward_heading_deg: float
        curve_samples: [[s, heading], ...]   (optional, for curves)
    """
    kind = str(matched_structure.get("kind") or "").lower()
    if kind == "junction":
        center = matched_structure.get("center") or {}
        legs = [
            JunctionLeg(
                name=str(leg.get("name") or f"leg_{i}"),
                heading_out_deg=float(leg.get("heading_out_deg", 0.0) or 0.0),
                inbound_lanes=list(leg.get("inbound_lanes") or []),
                outbound_lanes=list(leg.get("outbound_lanes") or []),
            )
            for i, leg in enumerate(matched_structure.get("legs") or [])
        ]
        return JunctionFrame(
            center=(
                float(center.get("x", 0.0) or 0.0),
                float(center.get("y", 0.0) or 0.0),
                float(center.get("z", 0.0) or 0.0),
            ),
            legs=legs,
            ego_approach_heading_deg=float(
                matched_structure.get("ego_approach_heading_deg", 0.0) or 0.0
            ),
        )
    if kind in {"road_segment", "road", "straight", "curve"}:
        origin = matched_structure.get("origin") or {}
        samples = [
            (float(pair[0]), float(pair[1]))
            for pair in matched_structure.get("curve_samples") or []
            if isinstance(pair, (list, tuple)) and len(pair) >= 2
        ]
        return RoadFrame(
            origin=(
                float(origin.get("x", 0.0) or 0.0),
                float(origin.get("y", 0.0) or 0.0),
                float(origin.get("z", 0.0) or 0.0),
            ),
            forward_heading_deg=float(
                matched_structure.get("forward_heading_deg", 0.0) or 0.0
            ),
            samples=samples,
        )
    raise ValueError(f"Unknown matched_structure kind: {kind!r}")


def check_heading_consistency(
    yaw: float,
    ego_heading: float,
    heading_relation: Any,
    flow_compliance: Any,
) -> Tuple[bool, str]:
    """Validate a resolved yaw against what the VLM observed.

    The guard enforces *fidelity to the observation*, not traffic legality:

    * ``flow_compliance == "wrong_way"`` actors are expected to oppose their
      lane, so we only confirm the placement is non-degenerate -- never "fix"
      them back to legal flow.
    * ``legal`` actors must match the ego-relative relation the VLM reported,
      catching pipeline bugs (e.g. an oncoming car snapped facing ego's way).

    Returns ``(ok, reason)``.
    """
    relation = str(heading_relation or "unknown").lower()
    delta = angle_difference(yaw, ego_heading)

    if str(flow_compliance or "").lower() == "wrong_way":
        # Real wrong-way actor: accept by construction, just record it.
        return True, "wrong_way_preserved"

    if relation in {"same_direction", "same"}:
        ok = delta < 90.0
        return ok, ("ok" if ok else "expected_same_direction_but_yaw_opposes_ego")
    if relation in {"opposite_direction", "opposite", "oncoming"}:
        ok = delta > 90.0
        return ok, ("ok" if ok else "expected_oncoming_but_yaw_follows_ego")
    if relation == "crossing":
        ok = 45.0 <= delta <= 135.0
        return ok, ("ok" if ok else "expected_crossing_but_yaw_is_parallel")
    return True, "unconstrained"

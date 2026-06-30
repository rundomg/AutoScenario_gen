"""Stage 2 of the pure-LLM (no template library) accident experiment.

Given a predicted accident candidate (from ``LlmAccidentPredictor``) plus the
reconstructed executable actor table (``actors.json`` / spawn_payload), an LLM
writes the Scenic *body* (constants, behaviors, actor placement,
require/terminate). The map/model header is composed deterministically by the
runner from ``match.json`` so the map path is always correct.

Placement is **relative geometry**, not absolute coordinates: the ego is spawned
on a lane and the other actors are placed relative to the ego using the
``relative_to_ego`` longitudinal/lateral offsets from the actor table. Absolute
``x @ y`` placement was verified to crash this Scenic build (scenic 3.0.0b2 +
shapely 2.1.2) when it computes the road direction at the point, whereas
ego-relative placement compiles. The relative layout still preserves the accident
interaction geometry (e.g. lead vehicle ~10 m ahead in the same lane).

This intentionally lets the LLM write Scenic directly, which the design memo
``scenicnl_internalization_design.md`` advises against. The experiment exists to
measure how well that pure-LLM route actually works (compile pass rate, etc.).
"""

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.task_agent import TaskAgent


# Concise, original Scenic tutorial (public Scenic library API names only) so we
# do not depend on or copy scenicNL's prompt files at runtime. The placement
# constructs below were each verified to compile on scenic 3.0.0b2 + shapely 2.x.
SCENIC_TUTORIAL = """
Scenic is a probabilistic scene-description language. A program body (the map and
model header is already provided, do NOT repeat it) has these parts:

1. CONSTANTS, e.g. `EGO_SPEED = 10`.
2. BEHAVIOR DEFINITIONS using built-in dynamic behaviors:
   - `FollowLaneBehavior(speed)` keep driving in the current lane at `speed`.
   - `take SetBrakeAction(intensity)` apply braking (0..1).
   - `take SetThrottleAction(x)`, `take SetSteerAction(steer)` (steer in -1..1).
     Use throttle+steer for cut-in / crossing / turning / evasive maneuvers.
   - AVOID `LaneChangeBehavior`: its first argument must be a real LaneSection
     object (e.g. `network.laneSectionAt(self).laneToLeft`), never an integer or
     lane index -- passing a number compiles but crashes at simulation time.
     Express lateral/turning motion with `SetSteerAction` + `SetThrottleAction`
     instead.
   Pattern:
     behavior EgoBehavior(speed=10):
         do FollowLaneBehavior(speed)
     behavior LeadBrake(speed=8):
         try:
             do FollowLaneBehavior(speed)
         interrupt when (distance to ego) < TRIGGER:
             take SetBrakeAction(1.0)
   Every `try` must have at least one `interrupt when` clause.
3. ACTOR PLACEMENT -- use RELATIVE geometry, never absolute `x @ y` (absolute
   placement crashes this Scenic build). Spawn the ego on a lane, then place the
   other actors relative to the ego:
     lane = Uniform(*network.lanes)
     egoSpawn = new OrientedPoint on lane.centerline
     ego = new Car at egoSpawn, with behavior EgoBehavior()
   - Ahead/behind in the same lane (D metres, negative = behind):
       lead = new Car following roadDirection from ego for 10
   - Lateral / adjacent-lane actors (ego-local `lateral @ longitudinal`, +lateral
     is to the ego's right, +longitudinal is ahead):
       cutin = new Car at ego offset by -3.5 @ 8
     or `new Car left of ego by 3.5` / `new Car right of ego by 3.5`.
   There must be exactly one `ego`. Other actors are `new Car`, `new Pedestrian`,
   `new Bicycle`, etc. All variables must be defined before they are used.
4. Optional `require ...` and `terminate when ...` lines at the end.

Do not emit `param map`, `param carla_map`, `model ...`, or `simulate()`.
Output ONLY the Scenic body as plain text, no markdown fences, no prose.
""".strip()


EXAMPLE_BODY = """
EGO_SPEED = 10
LEAD_SPEED = 8
TRIGGER_DIST = 12
BRAKE = 1.0

behavior EgoBehavior(speed=EGO_SPEED):
    do FollowLaneBehavior(speed)

behavior LeadHardBrakeBehavior(speed=LEAD_SPEED):
    try:
        do FollowLaneBehavior(speed)
    interrupt when (distance to ego) < TRIGGER_DIST:
        take SetBrakeAction(BRAKE)

lane = Uniform(*network.lanes)
egoSpawn = new OrientedPoint on lane.centerline
ego = new Car at egoSpawn,
    with behavior EgoBehavior()

# lead vehicle ~10 m ahead in the same lane (relative_to_ego.longitudinal_m ~= 10)
lead = new Car following roadDirection from ego for 10,
    with behavior LeadHardBrakeBehavior()

require (distance to intersection) > 50
terminate when (distance to lead) < 2
""".strip()


class LlmScenicGenerator(TaskAgent):
    """LLM that writes a Scenic body for one predicted accident candidate."""

    def __init__(self, ego_speed_mps: float = 10.0) -> None:
        super().__init__()
        self.ego_speed_mps = float(ego_speed_mps)

    def generate(
        self,
        *,
        scene_id: str,
        candidate: Dict[str, Any],
        actor_context: Dict[str, Any],
        map_name: str,
        prior_attempt: Optional[str] = None,
        compile_error: Optional[str] = None,
    ) -> str:
        add_info = {
            "scene_id": scene_id,
            "candidate": candidate,
            "actor_context": actor_context,
            "map_name": map_name,
            "prior_attempt": prior_attempt,
            "compile_error": compile_error,
            "request_label": "Scenic generation",
            "request_timeout": 180,
            "request_max_tokens": 3000,
        }
        response = self.send_request("", add_info)
        return self._strip_body(response)

    def refine_request(self, user_request, add_info=None):
        assert add_info is not None, "Missing scenic generator add_info."
        candidate = add_info.get("candidate") or {}
        actor_context = add_info.get("actor_context") or {}

        prompt = (
            "You write a CARLA Scenic scenario body that reproduces ONE predicted "
            "accident on a reconstructed scene.\n\n"
            f"Ego drives forward at a fixed {self.ego_speed_mps:.1f} m/s unless it "
            "must react.\n\n"
            f"Scenic tutorial:\n{SCENIC_TUTORIAL}\n\n"
            f"Example body (style reference, different scene):\n{EXAMPLE_BODY}\n\n"
            f"CARLA map: {add_info.get('map_name')} (the map/model header is added "
            "automatically -- do NOT write it).\n\n"
            "Executable actor table. Use each actor's `relative_to_ego` "
            "(longitudinal_m ahead/+, lateral_m right/+) to place it relative to "
            "the ego -- do NOT use the absolute x/y/yaw fields for placement:\n"
            f"{json.dumps(actor_context, indent=2, sort_keys=True)}\n\n"
            "Predicted accident to reproduce:\n"
            f"{json.dumps(candidate, indent=2, sort_keys=True)}\n\n"
            "Instructions:\n"
            "- Spawn the ego (id \"ego\") on a lane "
            "(`new OrientedPoint on lane.centerline`) and give it a forward-driving "
            f"behavior at {self.ego_speed_mps:.1f} m/s.\n"
            "- Choose the actor(s) from the actor table that best realise the "
            "predicted accident and place them RELATIVE to the ego using their "
            "`relative_to_ego` offsets: `following roadDirection from ego for "
            "<longitudinal_m>` for same-lane ahead/behind, or "
            "`new Car at ego offset by <lateral_m> @ <longitudinal_m>` for "
            "adjacent/lateral actors.\n"
            "- Give them behaviors that produce the described mechanism (hard "
            "brake, cut-in, cross). For lateral / crossing / turning motion use "
            "`SetSteerAction` + `SetThrottleAction`; do NOT call "
            "`LaneChangeBehavior` with an integer. Use realistic CARLA blueprints "
            "(vehicle.* / walker.*).\n"
            "- Do NOT use absolute `x @ y` positions (they fail to compile).\n"
            "- Output ONLY the Scenic body, no header, no fences, no commentary.\n"
        )

        prior_attempt = add_info.get("prior_attempt")
        compile_error = add_info.get("compile_error")
        if prior_attempt and compile_error:
            prompt += (
                "\nYour previous Scenic body failed to compile. Fix it.\n"
                "Previous body:\n"
                f"{prior_attempt}\n\n"
                "Compiler error:\n"
                f"{compile_error}\n\n"
                "Output a corrected Scenic body that compiles. Preserve the same "
                "actors and accident intent.\n"
            )

        return [{"type": "text", "text": prompt}]

    @staticmethod
    def _strip_body(response: str) -> str:
        if not isinstance(response, str):
            return ""
        text = response.strip()
        fenced = re.search(r"```(?:python|scenic)?\s*(.*?)\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()
        # Drop any header lines the model emitted despite instructions.
        kept: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("param map") or stripped.startswith("param carla_map"):
                continue
            if stripped.startswith("model scenic"):
                continue
            if stripped == "simulate()":
                continue
            kept.append(line)
        return "\n".join(kept).strip()

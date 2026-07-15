# CARLA vehicle atomic behavior design

## Goal and boundary

The VLM may infer an arbitrary accident *intent*, but it may only execute that
intent by composing a closed catalog of validated vehicle behaviors. The catalog
covers the CARLA road-vehicle motion surface needed for accident reconstruction:
longitudinal motion, lateral/route motion, interaction with other actors,
reversing, signalling, and bounded low-level `VehicleControl`.

Teleportation, direct impulses, physics mutation, actor destruction, and raw
Python are deliberately excluded. They are simulator/debug operations rather
than vehicle behavior and would let a model fake an accident instead of
reconstructing it through vehicle dynamics.

## Representation

`video-trajectory-dsl-v2` keeps one non-overlapping timeline per actor:

```json
{
  "actor_id": "veh_1",
  "role": "cut_in_vehicle",
  "segments": [
    {
      "start_s": 0.0,
      "end_s": 1.5,
      "action": "lane_follow_speed",
      "target_speed_mps": 8.0
    },
    {
      "start_s": 1.5,
      "end_s": 3.0,
      "action": "lane_change",
      "direction": "left",
      "target_speed_mps": 7.0,
      "lights": ["left_blinker"]
    }
  ]
}
```

Only one motion atom owns a vehicle at a time. Lights are a side effect on that
atom and can therefore run concurrently without creating conflicting controls.
When the next segment starts, its atom supersedes the previous atom. This makes
composition deterministic and prevents two simultaneous motion controllers from
fighting over the same `VehicleControl`.

## Atomic behavior catalog

### Longitudinal and lane keeping

| Atom | Required intent | Main bounds |
|---|---|---|
| `lane_follow_speed` | keep lane at speed | speed ≥ 0 |
| `accelerate_to_speed` | controlled acceleration | speed ≥ 0, acceleration 0.1–12 m/s² |
| `decelerate_to_speed` | controlled deceleration | speed ≥ 0, deceleration 0.1–15 m/s² |
| `brake` | fixed service braking | intensity 0–1 |
| `emergency_brake` | maximum braking | no free parameters |
| `coast` | release pedals | no free parameters |
| `stop` | brake to zero | optional hand brake |
| `hold_position` | remain stationary | brake + hand brake |

### Lateral and route control

| Atom | Required intent | Execution |
|---|---|---|
| `steer_offset` | short swerve/encroachment | bounded steer/throttle/brake |
| `lane_change` | move 1–3 adjacent lanes | target-lane CARLA waypoint route + tracking |
| `junction_maneuver` | left/right/straight/U-turn | branch selection + waypoint tracking |
| `drive_to_location` | reach a known world target | pure-pursuit-style target tracking |
| `reverse` | backward motion | bounded reverse speed and steer |

### Closed-loop interaction

| Atom | Required intent | Feedback |
|---|---|---|
| `follow_actor` | maintain a gap | target speed and current actor distance |
| `approach_actor` | close forward/reverse to a selected gap | target gap; zero explicitly permits collision |
| `yield_to_actor` | stop while target is near | target distance and resume speed |

### Low-level escape hatch

`raw_vehicle_control` exposes only the shape of `carla.VehicleControl`:
`throttle`, `steer`, `brake`, `hand_brake`, `reverse`,
`manual_gear_shift`, and `gear`. Numeric controls are clamped to CARLA ranges.
The prompt requires the VLM to prefer semantic atoms; raw control is reserved for
loss-of-control motion that cannot be expressed safely otherwise.

Any atom may include validated lights such as `left_blinker`, `right_blinker`,
`brake`, `reverse`, and head/fog lights.

## Derived complex behaviors

Derived names are not executable actions. They are planning recipes expanded by
the VLM into atoms:

| Complex behavior | Atomic composition |
|---|---|
| rear-end a lead vehicle | `lane_follow_speed → approach_actor(gap=0)` |
| lead vehicle emergency stop | `lane_follow_speed → emergency_brake → hold_position` |
| cut in then brake-check | `lane_follow_speed → lane_change → emergency_brake` |
| overtake and return | `lane_change(left) → accelerate_to_speed → lane_change(right)` |
| sideswipe | `approach_actor → lane_change/steer_offset → brake` |
| junction crossing collision | `junction_maneuver/drive_to_location → approach_actor(gap=0)` |
| red-light running | `lane_follow_speed → junction_maneuver` without a yield atom |
| fail-to-yield | `junction_maneuver` instead of `yield_to_actor` |
| backing collision | `approach_actor(gap=0, travel_direction=reverse) → emergency_brake` |
| loss of control | `steer_offset/raw_vehicle_control → coast → emergency_brake` |
| parked pull-out | `hold_position → lane_change → accelerate_to_speed` |

The structured video understanding supplies actor identity and semantic event
order. The trajectory VLM chooses atoms and bounded parameters. The validator
then checks actor IDs, target actor IDs, timing, non-overlap, parameter ranges,
direction enums, target coordinates, and lights before any script is generated.

## Execution guarantees

- VLM output cannot introduce a new actor or controller.
- No arbitrary CARLA/Python code is accepted.
- Each motion parameter is normalized into a documented safe range.
- Lane changes and junction turns use map waypoints rather than fixed steering.
- Actor interactions are recomputed every simulation tick.
- Old `video-trajectory-dsl-v1` documents remain accepted.
- Lowered programs still use the existing deterministic event executor.

## Known physical limits

This catalog is complete for ordinary road-vehicle commands, not a guarantee
that every visually inferred trajectory is physically reachable. Success still
depends on map topology, the actor's spawn lane, tire/vehicle physics, available
distance, and the quality of monocular VLM timing/speed estimates. If an adjacent
driving lane or requested junction branch does not exist, the executor falls
back to the current forward lane rather than teleporting the vehicle. Current
outcome metrics catch collision/non-collision mismatch; per-atom fallback
diagnostics are the next required observability improvement.

Future fidelity work should add per-atom execution diagnostics and event/geometry
triggers (TTC, actor distance, waypoint reached) so CARLA timing can adapt when
the static reconstruction scale differs from the source video.

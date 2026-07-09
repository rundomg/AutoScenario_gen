# VLM Static Reconstruction Pipeline Logic

This document records the current image-to-CARLA static reconstruction flow used by
`experiments/auto_generate_all_vlm.py`. It is written as an execution trace: input,
intermediate artifacts, transformation logic, and final executable output.

## Scope

Primary path covered here:

```text
image + optional user text
  -> scene understanding JSON
  -> CARLA map/topology match
  -> relation DSL
  -> local coordinates
  -> CARLA-projected actor payload
  -> executable CARLA Python script
```

The final executable file is `{scene_id}_static.py`; the runtime actor table it
loads is `{scene_id}_actors.json`.

Related but secondary systems such as video trajectory, risk DSL, Scenic export,
and legacy command/text generation reuse parts of this output, especially
`actors.json`, but are not the primary path for `auto_generate_all_vlm.py`.

## Entry Point

Main script:

```bash
python experiments/auto_generate_all_vlm.py \
  --image-path <input image> \
  --output-folder <result dir> \
  --user-input "<optional scene description>"
```

Default CLI values:

- `--image-path`: `data/0107.jpg`
- `--output-folder`: `results/auto_result_<timestamp>`
- `--user-input`: `""` (empty by default; user description merge only runs when non-empty)

The `__main__` block builds:

- `input_info`
- `AutoGenerator(output_folder, input_info)`
- `additional_info`

Important `input_info` values:

- `input_type="image"`
- `require_carla_connection=True`
- `enable_scene_match=True`
- `enable_scene_verify=False`
- `merge_user_description=True`
- `topology_cache_dir=data/map_cache`
- `map_match_*` weights

The main mode is hard-coded:

```python
mode = "FullPipeline"
```

## Output Files

For base `scene_id = s0000`:

- `s0000_su.json`: normalized scene understanding.
- `s0000_road_scene.json`: road-only VLM output.
- `s0000_orientation.json`: detected-vehicle orientation output.
- `s0000_position.json`: detected-vehicle position/layout output.
- `s0000_position_raw.txt`: raw VLM text for position agent.
- `s0000_detections.jpg`: image annotated with detector boxes.
- `s0000_det_<n>_crop.jpg`: detector crops.

For matched candidate `scene_id = s0000_c0`:

- `s0000_c0_match.json`: map match report.
- `s0000_c0_matched_structure.json`: matched road/junction structure summary.
- `s0000_c0_topo.json`: topology/cache debug artifact when available.
- `s0000_c0_actors.json`: final spawn payload.
- `s0000_c0_static.py`: executable CARLA scene script.
- `s0000_c0_quick_bev.png`: quick BEV preview if CARLA run succeeds.

Optional verification outputs when enabled:

- `s0000_c0_verify_r<n>.json`
- `s0000_c0_bev_r<n>.png`
- `s0000_c0_ego_r<n>.png`
- `s0000_c0_render_actor_graph_r<n>.json`
- spawn layout repair summaries.

## High-Level Control Flow

`AutoGenerator._build_scenario_for_match()` is the downstream backbone after map
matching:

```text
generate_scene_understanding()
analyze_topology_scene_match()
_apply_match_to_spawn_context()
_ensure_matched_world_loaded()
_sample_dense_local_waypoints()
_ensure_matched_structure()
_index_ego_on_candidate_lane()
generate_relation_dsl()
generate_initial_coordinates()
apply_pairwise_ordering_step()
project_entities_step()
refine_coordinates_step()
validate_layout_step()
reproject_junction_step()
validate_structural_reprojection()
build_spawn_payload_from_match_or_fallback()
generate_final_scene_script()
_capture_quick_bev_preview()
```

## Stage 1: Scene Understanding

Function:

```python
AutoGenerator.generate_scene_understanding()
```

Calls:

```python
SceneUnderstandingInterpreter.call_agent()
```

By default, `split_scene_agents=True`, so this uses:

```text
_call_split_scene_agents()
```

### 1.1 Detection Evidence

Function:

```python
SceneUnderstandingInterpreter._prepare_detection_evidence()
```

It calls `tools.vehicle_detector`:

- `detect_vehicles(image_path)`
- `select_representative_detections()`
- `infer_row_group_hints()`
- `annotate_image()`
- `write_detection_crops()`

Outputs:

- detector list with `id`, `label`, `bbox_norm`, `center_norm`, `conf`, `crop_path`
- row hints for possible queue/row ordering
- annotated image path, usually `s0000_detections.jpg`
- crop files

Important: detector boxes are evidence of actor presence and image position.
They are not road topology. A bbox center-x is not a junction arm label.

### 1.2 Road Scene Agent

Function:

```python
SceneUnderstandingInterpreter._call_road_scene_agent()
```

Output file:

```text
s0000_road_scene.json
```

The road scene prompt asks for:

- `road_network.map_matching`
- `road_network.lane_groups`
- `actor_layout` anchors
- `general_environment`
- `metadata.ego_localization`

It explicitly should not output vehicle actors.

Core fields:

- `topology_type`: `straight_road`, `straight_two_way`, `curve`,
  `t_junction`, `cross_intersection`, `multi_branch`, `roundabout`, `unknown`
- `junction_visible`
- `junction_type`
- `junction_branches`
- `target_branch_count`
- `forward_lane_count`
- `opposing_lane_count`
- `driving_lane_count`
- `ego_lane_from_right`
- `ego_to_junction_distance_m`
- `left_parking_presence`
- `right_parking_presence`

The road scene agent classifies road geometry. It should not classify a scene as
a junction merely because there are traffic lights, crosswalks, lane arrows, or
crossing-looking vehicles.

### 1.3 Vehicle Orientation Agent

Function:

```python
SceneUnderstandingInterpreter._call_vehicle_orientation_agent()
```

Agent:

```python
VehicleOrientationInterpreter
```

Output file:

```text
s0000_orientation.json
```

Inputs:

- annotated full image
- detector crops
- road scene JSON
- user description

Outputs per detected vehicle:

- `visible_end`
- `front_points_image_direction`
- `vehicle_region_hint` for junction scenes
- `road_region_hint` for open-road scenes
- `junction_center_relative_to_vehicle`
- `heading_relation_to_ego`
- `junction_travel_direction`
- confidence/evidence

Critical rule:

For junction scenes, `vehicle_region_hint=left_arm/right_arm` means the vehicle
is physically on the cross street or side-road branch. A vehicle in the ego
approach left/right/same lane must remain `vehicle_region_hint=ego_approach`.

### 1.4 Vehicle Position Agent

Function:

```python
SceneUnderstandingInterpreter._call_vehicle_position_agent()
```

Agent:

```python
VehiclePositionInterpreter
```

Output files:

- `s0000_position_raw.txt`
- `s0000_position.json`

Inputs:

- annotated image
- detector list
- road scene JSON
- orientation payload
- row hints
- user description

Outputs:

- `vehicle_position_brief`
- `traffic_subjects`
- `key_pairwise_relations`

Each `traffic_subject` may contain:

- `id`
- `category`
- `motion_state`
- `heading_relation_to_ego`
- `lane_side_relation`
- `lane_index_relation`
- `longitudinal_relation`
- `longitudinal_proximity`
- `placement_mode_hint`
- grouping fields
- optional `layout_anchor_id`
- optional `anchor_relation`

Critical boundary:

- `lane_side_relation` and `lane_index_relation` encode lateral lane position
  relative to ego or to the selected arm.
- `layout_anchor_id` encodes the road anchor: `ego_approach`, `left_arm`,
  `right_arm`, `ahead_arm`, `oncoming_arm`, etc.
- In a junction, a vehicle can still be on `layout_anchor_id=ego_approach` and
  have `lane_side_relation=left_lane` or `right_lane`.
- `left_lane` does not mean `left_arm`.
- `right_lane` does not mean `right_arm`.

### 1.5 Position Fallback

If the position agent fails, the pipeline falls back to:

```python
SceneUnderstandingInterpreter._fallback_vehicle_position_payload()
SceneUnderstandingInterpreter._subject_from_detection()
```

The fallback uses detector center-x only as a weak lane-side estimate:

```text
cx < 0.38 -> lane_index_relation=-1 -> left_lane
cx > 0.62 -> lane_index_relation=+1 -> right_lane
otherwise -> same_lane
```

Current rule: fallback must not use bbox center-x to assign `left_arm` or
`right_arm`. Arm assignment requires explicit arm evidence or VLM arm hint.

History of the removed wrong fallback:

- Commit: `94850e6`
- Date: `2026-07-04 12:54:47 +0800`
- Commit message: `Update VLM scenario generation and map matching`
- Original logic:

```python
if cx < 0.35:
    return "left_arm"
if cx > 0.65:
    return "right_arm"
```

Why it was wrong:

It conflated image screen position with road topology. In a dashcam image, an
object on the left side of the frame can simply be in the ego approach's left
lane. It is not necessarily on the left branch of an intersection.

### 1.6 User Description Merge

After split-scene output, `AutoGenerator.generate_scene_understanding()` calls:

```python
SceneUnderstandingInterpreter.merge_with_user_description()
```

when `merge_user_description=True` and `--user-input` is non-empty.

The user description is authoritative when it conflicts with VLM output.

Current rule:

If the user says actors are in ego's same/left/right lane or ahead of ego, keep
them on `layout_anchor_id=ego_approach` in junction scenes and encode lateral
position through `lane_side_relation` / `lane_index_relation`.

### 1.7 Normalization

Function:

```python
tools.structured_pipeline.normalize_scene_understanding()
```

Responsibilities:

- Validate required top-level keys (`traffic_subjects`, `road_network`,
  `general_environment`, `metadata`). `background_traffic` is optional; the
  normalizer creates an empty list when absent and promotes supported background
  entries into `traffic_subjects`.
- Normalize entity categories/subtypes.
- Normalize lane fields.
- Normalize heading/motion/turn/flow fields.
- Normalize `road_network.map_matching`.
- Normalize `road_network.lane_groups`.
- Normalize key pairwise relations.
- Merge background traffic into `traffic_subjects`.
- Apply parking symmetry where appropriate.
- Tolerate unknown numeric strings such as `"unknown"` by falling back to
  defaults.

Output:

```text
s0000_su.json
```

## Stage 2: Map Matching

The pipeline then creates candidate matched scenes. Current `num_candidates` is
1 in the main script.

Function:

```python
AutoGenerator.analyze_topology_scene_match()
```

Calls:

```python
SceneMapMatcher.analyze_topology_scene_assets()
```

Output:

```text
s0000_c0_match.json
```

### 2.1 Topology Signature

Function:

```python
SceneMapMatcher.build_road_topology_signature()
```

Input:

```text
s0000_su.json
```

Builds a compact signature from:

- `road_network.map_matching`
- `road_network.lane_groups`
- `metadata.ego_localization`
- side/environment hints

Important fields:

- topology type
- junction visibility/type/branches
- forward/opposing/driving lane counts
- parking presence
- center median
- crosswalk/traffic light
- curve direction
- ego lane from right
- ego-to-junction distance

### 2.2 Candidate Search

`SceneMapMatcher` can work from:

- cached topology in `data/map_cache`
- live CARLA topology if available

Candidate scoring considers:

- topology type alignment
- junction branch directions
- target branch count
- lane counts
- center median
- parking/side context
- curve direction
- ego distance to junction
- auxiliary context

Output match report contains:

- `status`
- `world_name`
- `scene_features`
- `candidate_summary`
- `best_match`
- `best_match.candidate_lane`
- `best_match.candidate_features`
- `best_match.matched_structure`
- optional `projected_layout`

### 2.3 Spawn Context Injection

Function:

```python
AutoGenerator._apply_match_to_spawn_context()
```

It injects the selected candidate lane into:

```python
self.carla_spawn_context["topology_sample"]
```

It also stores:

```python
self.carla_spawn_context["matched_structure"]
self.carla_spawn_context["matched_structure_source"]
```

`matched_structure` is the reference frame used later by
`tools.junction_placement`.

## Stage 3: Matched World and Structure

Function:

```python
AutoGenerator._build_scenario_for_match()
```

If CARLA is required:

- `_ensure_matched_world_loaded()` loads the matched CARLA map.
- `_sample_dense_local_waypoints()` samples dense waypoints near the anchor.
- `_ensure_matched_structure()` builds or recovers a real structure:
  - junction center + legs for junction scenes
  - road segment/centerline for road scenes

The structure is written to:

```text
s0000_c0_matched_structure.json
```

For junctions, downstream placement requires this structure. Without it, the
pipeline should fail instead of silently dropping actors onto a fake frame.

## Stage 4: Relation DSL

Function:

```python
AutoGenerator.generate_relation_dsl()
tools.structured_pipeline.build_relation_dsl()
```

Input:

```text
s0000_su.json
carla_spawn_context
```

Output:

debug artifact when enabled:

```text
s0000_c0_relation_dsl.json
```

Responsibilities:

- Select anchor lane.
- Build `lane_context`.
- Expand counted actors.
- Convert `traffic_subjects` to relation entities.
- Attach blueprint category/subtype.
- Attach lane anchor and lane offsets.
- Build pairwise relations from explicit key relations and derived relations.
- Preserve junction fields such as `layout_anchor_id`, `anchor_relation`,
  `lane_side_relation`, `lane_index_relation`, and turn intent.

## Stage 5: Initial Coordinates

Function:

```python
AutoGenerator.generate_initial_coordinates()
tools.structured_pipeline.generate_initial_coordinates_from_relation_dsl()
```

The generated local frame is ego-relative:

- longitudinal positive is ego forward
- lateral negative is ego-left
- lateral positive is ego-right

Actors get initial:

- `location`
- `rotation`
- `longitudinal_m`
- `lateral_m`
- lane anchor metadata

This stage is still abstract/local and not yet guaranteed to lie on a CARLA lane.

## Stage 6: Pairwise Ordering

Function:

```python
AutoGenerator.apply_pairwise_ordering_step()
tools.structured_pipeline.apply_pairwise_ordering()
```

Uses `key_pairwise_relations` to adjust longitudinal positions:

- `ahead_of_other`
- `behind_other`
- `aligned_with_other`
- `longitudinal_gap_band`

Purpose:

- avoid collapsing multiple actors into the same coordinate
- preserve explicit queue/row order

## Stage 7: CARLA Projection

Function:

```python
AutoGenerator.project_entities_step()
tools.structured_pipeline.project_entities_to_carla_context()
```

Inputs:

- ordered local coordinates
- scene understanding
- `carla_spawn_context`

Responsibilities:

- Choose semantic lane for each actor.
- Project vehicles to driving lanes, opposing lanes, parking lanes, or direct
  coordinates depending on category and semantic fields.
- Compute yaw from lane tangent and heading relation.
- Enrich topology sample if needed.
- Preserve junction seed fields for later structural reprojection.

Important projected fields:

- `projected_lane`
- `heading_relation`
- `lane_side_relation`
- `lane_index_relation`
- `layout_scene_kind`
- `placement_reason`
- `degraded_reason`

## Stage 8: Pairwise Refinement

Function:

```python
AutoGenerator.refine_coordinates_step()
tools.structured_pipeline.refine_projected_coordinates_with_pairwise_relations()
```

Purpose:

- resolve pairwise layout constraints after CARLA projection
- keep relative ordering after snapping/projecting
- mitigate collisions and overlap

## Stage 9: Layout Validation

Function:

```python
AutoGenerator.validate_layout_step()
tools.structured_pipeline.validate_relation_layout()
```

Checks:

- lane-side consistency
- pairwise relation consistency
- rough distance/gap consistency
- category/placement legality

This validation runs before structural reprojection. For junction/road-structure
placement, `reproject_junction_step()` now also records a lightweight structural
sanity validation after reprojection so final `junction_direction` / arm anchors
can be inspected before `actors.json` is built.

Output debug artifact when enabled:

```text
s0000_c0_relation_validation.json
```

## Stage 10: Structural Reprojection

Function:

```python
AutoGenerator.reproject_junction_step()
tools.junction_placement.reproject_actors_for_structure()
```

If `matched_structure.kind == "junction"`:

```python
reproject_actors_for_junction()
```

If road/curve:

```python
reproject_actors_for_road()
```

### Junction Reprojection Logic

The junction structure contains legs:

- ego approach
- left arm
- right arm
- ahead/oncoming arm
- each with inbound/outbound lanes where available

For each actor:

1. Determine direction and motion:
   - `layout_anchor_id`
   - `heading_relation`
   - `anchor_relation.travel_direction`
   - `turn_intent`
2. Select leg:
   - `ego_approach` -> ego leg
   - `left_arm` -> left leg
   - `right_arm` -> right leg
   - `ahead_arm` / `oncoming_arm` -> opposite/ahead leg
3. Select inbound/outbound motion:
   - approaching/toward junction
   - leaving/away from junction
4. Select lane slot:
   - explicit `anchor_relation.lane_from_right` wins
   - ego arm actors use ego lane slot as reference
   - opposite arm can count lanes from median where needed
5. Compute distance from junction center.
6. Compute lateral offset using real lane anchor geometry when available.
7. Write:
   - `junction_direction`
   - `junction_leg`
   - `junction_motion`
   - `junction_distance_m`
   - `projected_lane`
   - updated `location` and `rotation`

Important:

For ego-approach vehicles, adjacent lanes should stay on `junction_direction="ego"`
(and the corresponding ego approach leg). They should not become left/right side
arms merely because `lane_side_relation` is `left_lane` or `right_lane`.

Post-reprojection sanity output is attached to coordinate metadata and, when
`debug_artifacts=True`, written as:

```text
s0000_c0_structural_validation.json
```

## Stage 11: Spawn Payload

Function:

```python
AutoGenerator.build_spawn_payload_from_match_or_fallback()
tools.structured_pipeline.build_projected_spawn_payload()
```

Output:

```text
s0000_c0_actors.json
```

Before building payload:

- `_prepend_ego_vehicle()` adds ego actor.
- `_apply_projected_layout_from_match()` may apply matched projected layout.
- If matched, coordinates are translated by match anchor dx/dy.

Each payload entity includes:

- `id`
- `spawn_kind`
- `blueprint_name`
- `category`
- `lane_side_relation`
- `lane_index_relation`
- `heading_relation`
- `flow_compliance`
- `motion_state`
- `projected_lane`
- `junction_direction`
- `junction_leg`
- `junction_motion`
- `junction_distance_m`
- `location`
- `rotation`
- `color`
- `placement_mode`
- grouping and layout metadata

### Placement Mode Selection

`build_projected_spawn_payload()` chooses:

- `direct`: static objects such as cone/barrier groups.
- `project_to_parking_lane`: parking-lane actors.
- `project_to_junction_lane`: vehicle with junction projection fields.
- `project_to_opposing_lane`: legal opposite-direction vehicle.
- `project_to_lane`: normal vehicle default.

The generated CARLA script uses `placement_mode`, not only semantic fields.
For junction actors, runtime spawn uses `projected_lane` and `junction_distance_m`.
For parking-lane actors, any concrete `projected_lane` is preserved so runtime can
prefer that target parking lane; when no concrete lane is known, placement falls
back to the seed XY location near the curb/parking strip.

## Stage 12: Static CARLA Script Generation

Function:

```python
AutoGenerator.generate_final_scene_script()
ExistingWorldScenarioGenerator.build_existing_world_scene_script()
```

Output:

```text
s0000_c0_static.py
```

The static script contains:

- CARLA client setup
- world loading
- helper import/bootstrap
- spawn payload filename
- spawn loop
- optional image/actor-graph capture

The script references:

```python
_AUTOSCENARIO_SPAWN_PAYLOAD = "s0000_c0_actors.json"
```

It imports or embeds helper functions from:

```text
autoscenario_scene_helpers.py
```

## Stage 13: Runtime Spawn Logic

In the generated script, `_build_spawn_payload_loop()` does:

```text
load actors.json
read ego yaw
for each entity:
  build carla.Location and carla.Rotation
  adjust heading relation if needed
  branch on spawn_kind and placement_mode
  call helper spawn function
  store actor_by_id
capture render actor graph / ego view / BEV if requested
sleep(2)
```

Runtime placement functions:

- `spawn_kind == pedestrian`:
  - `_autoscenario_spawn_pedestrian()`
- `spawn_kind == static`:
  - `_autoscenario_spawn_static_prop()`
- `placement_mode in {"direct", "preserve_xy"}`:
  - `_autoscenario_spawn_vehicle_direct()`
- `placement_mode == "project_to_opposing_lane"`:
  - `_autoscenario_spawn_vehicle_opposing()`
- `placement_mode == "project_to_parking_lane"`:
  - `_autoscenario_spawn_vehicle_parking_lane()`
- `placement_mode == "project_to_junction_lane"`:
  - `_autoscenario_spawn_vehicle_junction_lane()`
- otherwise:
  - `_autoscenario_spawn_vehicle()`

### Junction Runtime Spawn

For `project_to_junction_lane`:

```python
_autoscenario_spawn_vehicle_junction_lane(
    blueprint_name,
    location,
    rotation,
    projected_lane,
    color,
    junction_distance_m,
)
```

This calls:

```python
_autoscenario_project_vehicle_to_specific_lane()
```

The runtime does not reinterpret `lane_side_relation` to choose a junction arm.
It uses the already computed `projected_lane`.

Therefore, if an actor is wrong in final CARLA placement, inspect in this order:

1. `s0000_position.json`: did VLM assign wrong `layout_anchor_id`?
2. `s0000_su.json`: did merge preserve or correct it?
3. `s0000_c0_actors.json`: what are `junction_leg` and `projected_lane`?
4. `s0000_c0_static.py`: what `placement_mode` is used?

## Stage 14: Optional Verification and Repair

Controlled by:

```python
enable_scene_verify
verify_mode
verify_max_rounds
verify_min_score
```

When enabled, the final scene script can be run to capture:

- ego view
- BEV view
- render actor graph

Verification can use:

- VLM verification agent
- static actor graph checks

Repair paths can revise:

- scene understanding
- relation DSL
- spawn payload / coordinates

Current default in `auto_generate_all_vlm.py`:

```python
enable_scene_verify=False
```

So the normal run only generates quick BEV preview after static script creation.

## Important Field Semantics

### `layout_anchor_id`

Road/topology anchor for actor placement.

Common values:

- empty string for open-road non-junction layout
- `ego_approach`
- `junction_center`
- `left_arm`
- `right_arm`
- `ahead_arm`
- `oncoming_arm`

It should answer: "Which road piece is this actor on?"

### `lane_side_relation`

Ego-centric or anchor-centric lateral lane relation.

Common values:

- `same_lane`
- `left_lane`
- `right_lane`
- `opposing_lane`
- `left_parking_lane`
- `right_parking_lane`
- `left_edge`
- `right_edge`
- `sidewalk_left`
- `sidewalk_right`
- `crosswalk`

It should answer: "Which lane band relative to the anchor is this actor in?"

### `lane_index_relation`

Numeric lane offset:

- `0`: same lane
- `-1`: first left lane / adjacent left
- `+1`: first right lane / adjacent right

For some opposite-direction logic, negative indices may encode opposing lanes
before correction.

### `junction_leg`

Final actor payload field after structural reprojection.

Values:

- `ego`
- `left`
- `right`
- `ahead` / `opposite`
- empty for non-junction actors

This is consumed indirectly via `projected_lane`.

### `projected_lane`

Concrete CARLA lane selected by projection/reprojection.

Common keys:

- `road_id`
- `lane_id`
- `role`
- `yaw`
- `source`
- `anchor`

Runtime spawn snaps to this lane for `project_to_junction_lane` and parking
placement modes.

### `placement_mode`

Final executable spawn mode in `actors.json`.

This is the field the generated script branches on. Semantic fields such as
`lane_side_relation` do not by themselves determine runtime spawn behavior once
`placement_mode` and `projected_lane` exist.

## Debugging the Left-Lane vs Left-Arm Failure

Symptom:

```text
det_2 lane_side_relation=left_lane but layout_anchor_id=left_arm
det_3 lane_side_relation=right_lane but layout_anchor_id=right_arm
```

For an ego-approach three-lane case, this is wrong. Correct representation:

```json
{
  "layout_anchor_id": "ego_approach",
  "lane_side_relation": "left_lane",
  "lane_index_relation": -1
}
```

and:

```json
{
  "layout_anchor_id": "ego_approach",
  "lane_side_relation": "right_lane",
  "lane_index_relation": 1
}
```

Final `actors.json` should then show:

```text
junction_leg = "ego"
projected_lane = lane on ego approach
```

Why the old bug happened:

1. Road scene correctly classified a cross intersection and exposed left/right
   arms.
2. Orientation/position VLM saw vehicles on image left/right.
3. The position prompt did not receive the user description strongly enough.
4. The VLM treated screen-left/screen-right vehicles as side-road-arm vehicles.
5. Old fallback also encoded the same wrong assumption:

```python
cx < 0.35 -> left_arm
cx > 0.65 -> right_arm
```

Current protection:

- user text is passed into orientation and position agents
- junction prompts explicitly distinguish ego-approach lanes from junction arms
- fallback no longer maps bbox x to `left_arm/right_arm`
- user-described ego lane layout can correct `left_arm/right_arm` back to
  `ego_approach`

## Where to Inspect a Bad Run

Use this order:

1. `s0000_detections.jpg`
   - confirm detector ids and image positions
2. `s0000_road_scene.json`
   - confirm road topology and lane counts
3. `s0000_orientation.json`
   - check `vehicle_region_hint` and heading relation
4. `s0000_position_raw.txt`
   - see raw VLM reasoning and whether it confused lanes with arms
5. `s0000_position.json`
   - check `layout_anchor_id`, `lane_side_relation`, `lane_index_relation`
6. `s0000_su.json`
   - check user-merge result
7. `s0000_c0_match.json`
   - check CARLA map, candidate lane, matched structure
8. `s0000_c0_matched_structure.json`
   - check junction legs and lane ids
9. `s0000_c0_actors.json`
   - check `placement_mode`, `junction_leg`, `projected_lane`
10. `s0000_c0_static.py`
   - confirm generated script references the expected actors JSON
11. `s0000_c0_quick_bev.png`
   - visual sanity check after CARLA spawn

## Legacy and Extension Paths

`auto_generate_all_vlm.py` still contains legacy modes:

- `AfterInterpreter`
- `AfterNet`
- `AfterObject`

These call older net/object/scenario generators and scene matching, but the
default path is `FullPipeline`.

Other extensions can consume `actors.json`:

- risk scenario pipeline
- pure LLM risk/scenic runners
- video trajectory reconstruction

These systems assume the static reconstruction has already produced a valid
spawn payload.

## Practical Invariants

Keep these true:

1. User description overrides conflicting VLM fields.
2. Road topology fields describe road geometry, not actor screen position.
3. Detector bbox center-x is never sufficient to assign a junction arm.
4. `left_lane/right_lane` are lane relations, not side-road branch names.
5. `layout_anchor_id=left_arm/right_arm` requires physical side-road evidence.
6. In `actors.json`, runtime behavior follows `placement_mode` and
   `projected_lane`.
7. Junction scenes require a valid `matched_structure`.
8. Existing result files are not retroactively updated after code fixes.

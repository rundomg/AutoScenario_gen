# Implementation Plan: Structured VLM Pipeline for `auto_generate_all_vlm.py`

## Goal

Revise `experiments/auto_generate_all_vlm.py` so its default image pipeline uses
the strongest parts of the OpenDRIVE experiment pipeline without generating a
custom OpenDRIVE map or SUMO network.

The new default flow is:

```text
image
  -> structured scene understanding JSON
  -> relation DSL and coordinates in memory
  -> CARLA existing-map region match
  -> final spawn payload
  -> deterministic CARLA scene script
```

The implementation should keep the main run compact and inspectable. By
default, only the core artifacts are written. Full intermediate JSON files are
written only when `debug_artifacts=True`.

## Why This Change

| Area | Current main flow | New main flow | Reason |
|---|---|---|---|
| Image interpretation | `UniInterpreter` returns 5-section free text | `SceneUnderstandingInterpreter` returns validated JSON | Structured data is easier to validate and reuse |
| Road network | `NetGenerator` creates SUMO `.nod.xml`, `.edg.xml`, `.net.xml` | Removed from default path | The target workflow is CARLA existing-map spawning, not SUMO |
| Object coordinates | `ObstacleGenerator` asks an LLM to emit Python dictionaries | Deterministic relation/coordinate functions | Avoid hallucinated coordinates and make placement reproducible |
| CARLA map usage | Generated layout is later matched to CARLA | Match before final spawn payload generation | The final payload should already contain matched or fallback coordinates |
| Final script | LLM-generated CARLA Python then patched by matcher | Deterministic script reading `spawn_entities.json` | Avoid fragile text patching and reduce LLM calls |

## Default Artifacts

Default mode writes exactly these core files:

| File | Content |
|---|---|
| `{scene_id}_scene_understanding.json` | Validated VLM scene understanding: traffic subjects, background traffic, pairwise relations, road network, environment, metadata |
| `{scene_id}_scene_match.json` | CARLA matching report: status, world name, scene features, candidate summary, best match, matched layout or fallback/unavailable reason |
| `{scene_id}_spawn_entities.json` | Final spawn payload consumed by the CARLA script: entity IDs, spawn kind, blueprint name, category, location, rotation, color, placement mode, appearance |
| `{scene_id}_scene_final.py` | Deterministic CARLA script that loads the existing world, reads the spawn payload, projects actors when needed, and spawns the scene |

When `debug_artifacts=True`, additionally write:

- `{scene_id}_relation_dsl.json`
- `{scene_id}_coordinates_raw_initial.json`
- `{scene_id}_coordinates_raw_ordered.json`
- `{scene_id}_coordinates_projected.json`
- `{scene_id}_coordinates_projected_refined.json`
- `{scene_id}_relation_validation.json`

Do not write `{scene_id}_coordinate_program.py` in the default path. The new
coordinate calculation should be a direct in-memory function call.

## New Pipeline Flow

```text
Image
  |
  v
[1] SceneUnderstandingInterpreter
    -> save {scene_id}_scene_understanding.json
  |
  v
[2] Load CARLA context, or fallback context if CARLA is unavailable
    -> in memory: map_name, spawn_points, topology_sample, optional failure reason
  |
  v
[3] build_relation_dsl(scene_understanding, spawn_context)
    -> in memory relation DSL
    -> debug only: {scene_id}_relation_dsl.json
  |
  v
[4] generate_initial_coordinates_from_relation_dsl(relation_dsl)
    -> in memory raw initial coordinates
    -> debug only: {scene_id}_coordinates_raw_initial.json
  |
  v
[5] apply_pairwise_ordering(raw_initial, relation_dsl)
    -> in memory ordered coordinates
    -> debug only: {scene_id}_coordinates_raw_ordered.json
  |
  v
[6] project_entities_to_xodr(ordered, scene_understanding, spawn_context)
    -> in memory projected coordinates
    -> debug only: {scene_id}_coordinates_projected.json
  |
  v
[7] refine_projected_coordinates_with_pairwise_relations(projected, relation_dsl)
    -> in memory refined coordinates
    -> debug only: {scene_id}_coordinates_projected_refined.json
  |
  v
[8] validate_relation_layout(relation_dsl, refined)
    -> in memory validation report
    -> debug only: {scene_id}_relation_validation.json
  |
  v
[9] SceneMapMatcher.analyze_structured_scene_assets(...)
    -> save {scene_id}_scene_match.json
  |
  v
[10] build_spawn_payload_from_match_or_fallback(...)
     -> save {scene_id}_spawn_entities.json
  |
  v
[11] build_existing_world_scene_script(...)
     -> save {scene_id}_scene_final.py
```

Important ordering rule: scene matching happens before writing the final spawn
payload and final script. The new main path must not generate a script first and
then patch it with `_rewrite_scene_script_with_match()`.

## Step-by-Step Implementation

### Step 1: Add CARLA Context With Fallback

File: `experiments/auto_generate_all_vlm.py`

Extend `_load_carla_spawn_context()` so a successful CARLA connection returns:

```python
{
    "status": "available",
    "map_name": world_map.name,
    "spawn_points": sampled_spawn_points,
    "topology_sample": topology_sample,
}
```

`topology_sample` should be sampled from `world_map.get_topology()` and should
match the shape already consumed by `structured_pipeline`:

```python
{
    "road_id": int(start_wp.road_id),
    "lane_id": int(start_wp.lane_id),
    "start": {
        "x": float(start_wp.transform.location.x),
        "y": float(start_wp.transform.location.y),
        "z": float(start_wp.transform.location.z),
        "yaw": float(start_wp.transform.rotation.yaw),
        "is_junction": bool(start_wp.is_junction),
    },
    "end": {
        "x": float(end_wp.transform.location.x),
        "y": float(end_wp.transform.location.y),
        "z": float(end_wp.transform.location.z),
        "yaw": float(end_wp.transform.rotation.yaw),
        "is_junction": bool(end_wp.is_junction),
    },
}
```

If CARLA import or connection fails, do not terminate the pipeline. Return a
fallback context:

```python
{
    "status": "unavailable",
    "map_name": None,
    "spawn_points": [],
    "topology_sample": [
        {
            "road_id": 1,
            "lane_id": -1,
            "start": {"x": 0.0, "y": -1.75, "z": 0.0, "yaw": 0.0, "is_junction": False},
            "end": {"x": 80.0, "y": -1.75, "z": 0.0, "yaw": 0.0, "is_junction": False},
        }
    ],
    "failure_reason": "...",
}
```

The failure reason must later be included in `{scene_id}_scene_match.json`.

### Step 2: Replace the Default Interpreter

File: `experiments/auto_generate_all_vlm.py`

Use `SceneUnderstandingInterpreter` as the default image interpreter:

```python
from opendrive_experiment.agents.scene_understanding_interpreter import (
    SceneUnderstandingInterpreter,
)
```

Add a `generate_scene_understanding()` method that:

- calls `SceneUnderstandingInterpreter.call_agent()`
- writes `{scene_id}_scene_understanding.json`
- returns the normalized scene understanding dict
- calls `adapt_generation_params_for_scene()` if that logic is still useful for
  downstream lane width/roadside placement assumptions

Keep legacy text extraction helpers only for old compatibility modes. They
should not be called by the new default `FullPipeline`.

### Step 3: Add `debug_artifacts`

File: `experiments/auto_generate_all_vlm.py`

Add:

```python
self.debug_artifacts = info_dict.get("debug_artifacts", False)
```

Provide a small helper:

```python
def _write_debug_json(self, scene_id: str, suffix: str, payload: dict) -> None:
    if not self.debug_artifacts:
        return
    write_to_file(
        join(self.output_folder, f"{scene_id}_{suffix}.json"),
        json.dumps(payload, indent=2, sort_keys=True),
    )
```

Use this only for debug artifacts. Core artifacts are always written explicitly.

### Step 4: Add In-Memory Coordinate Generation

File: `opendrive_experiment/tools/structured_pipeline.py`

Do not use `write_coordinate_program()` and `evaluate_coordinate_program()` in
the new default path. Add a direct pure function:

```python
def generate_initial_coordinates_from_relation_dsl(
    relation_dsl: Dict[str, Any],
) -> Dict[str, Any]:
    ...
```

It must produce the same payload shape as the current coordinate program output:

```python
{
    "selected_anchor_lane": ...,
    "entities": [...],
    "metadata": {
        ...,
        "coordinate_stage": "initial",
    },
}
```

The function should reuse the same math currently embedded in
`build_coordinate_program_source()`:

- lane width from `LANE_WIDTH_METERS`
- forward/right vectors from `selected_anchor_lane`
- longitudinal distance from `distance_band`, `order_relation`, rank, and group spacing
- lateral offset from `lane_anchor` and `lane_index_relation`
- yaw from anchor heading and `heading_relation`
- z defaults by category

Keep existing `write_coordinate_program()` and `evaluate_coordinate_program()`
for legacy/debug compatibility, but do not call them from the new main flow.

### Step 5: Add Structured Pipeline Methods

File: `experiments/auto_generate_all_vlm.py`

Import:

```python
from opendrive_experiment.tools.structured_pipeline import (
    apply_pairwise_ordering,
    build_projected_spawn_payload,
    build_relation_dsl,
    generate_initial_coordinates_from_relation_dsl,
    project_entities_to_xodr,
    refine_projected_coordinates_with_pairwise_relations,
    validate_relation_layout,
)
```

Add methods that run each stage in memory and optionally write debug JSON:

- `generate_relation_dsl(scene_id, scene_understanding)`
- `generate_initial_coordinates(scene_id, relation_dsl)`
- `apply_pairwise_ordering_step(scene_id, raw_initial, relation_dsl)`
- `project_entities_step(scene_id, ordered, scene_understanding)`
- `refine_coordinates_step(scene_id, projected, relation_dsl)`
- `validate_layout_step(scene_id, relation_dsl, refined)`

These methods should not create temporary Python files.

### Step 6: Add Structured Scene Matching

File: `tools/scene_map_matcher.py`

Add a new entry point:

```python
def analyze_structured_scene_assets(
    self,
    scene_id: str,
    output_folder: str,
    scene_understanding: Dict[str, Any],
    refined_coordinates: Dict[str, Any],
    validation: Optional[Dict[str, Any]] = None,
    spawn_context: Optional[Dict[str, Any]] = None,
) -> str:
    ...
```

This method writes `{scene_id}_scene_match.json`.

Do not create a fake `sumo_topology` dict with missing legacy fields. Instead,
add a dedicated structured feature extractor, for example:

```python
def _extract_structured_scene_features(
    self,
    scene_understanding: Dict[str, Any],
    refined_coordinates: Dict[str, Any],
    validation: Optional[Dict[str, Any]],
    spawn_context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    ...
```

The extracted features should include the fields consumed by existing CARLA
candidate scoring:

- `semantic_hints`
- `layout_summary.anchor_heading`
- `layout_summary.anchor_lane_count`
- `layout_summary.anchor_near_junction`
- `layout_summary.target_branch_count`
- `layout_summary.dynamic_vehicle_count`
- `layout_summary.parked_vehicle_count`
- `layout_summary.pedestrian_count`
- `layout_summary.static_object_count`
- `layout_summary.relative_layout`
- `entities`

For lane count, derive from `scene_understanding["road_network"]["lane_groups"]`
instead of SUMO XML. For junction hints, derive from `road_network["junctions"]`
and any junction information in `spawn_context["topology_sample"]`.

If `spawn_context["status"] == "unavailable"`, write a report with:

```python
{
    "status": "unavailable",
    "reason": spawn_context.get("failure_reason"),
    ...
}
```

and skip CARLA candidate scoring.

Keep legacy `analyze_scene_assets()` untouched for old SUMO-based callers.

### Step 7: Build Spawn Payload From Match or Fallback

File: `experiments/auto_generate_all_vlm.py`

Add a method:

```python
def build_spawn_payload_from_match_or_fallback(
    self,
    scene_id: str,
    refined_coordinates: Dict[str, Any],
    match_report_path: str,
) -> Dict[str, Any]:
    ...
```

Behavior:

- read `{scene_id}_scene_match.json`
- branch on `status`:
  - if `status == "matched"` and the report contains a usable `best_match`:
    apply the anchor offset from `best_match` to the entity locations in
    `refined_coordinates`, then call `build_projected_spawn_payload()` on the
    offset-applied coordinates. Concretely:
    ```
    for each entity in refined_coordinates["entities"]:
        entity["location"]["x"] += best_match["anchor"]["dx"]
        entity["location"]["y"] += best_match["anchor"]["dy"]
    ```
    where `dx`/`dy` are the delta from the match report's projected anchor to
    the CARLA world anchor. Preserve all other entity fields unchanged.
  - otherwise (status is `"unavailable"`, `"no_match"`, `"skipped"`, etc.):
    call `build_projected_spawn_payload(refined_coordinates)` directly without
    any offset
- call `build_projected_spawn_payload()` on the chosen coordinate set
- write `{scene_id}_spawn_entities.json`
- return the payload

This replaces the old pattern of generating a script and then patching it with
`SceneMapMatcher.apply_match_to_scene_script()`.

### Step 8: Add Existing-World Deterministic Script Builder

File: `opendrive_experiment/agents/scenario_generator_xodr.py`

Extract the mature spawn-payload loop and helper usage from
`XODRScenarioGenerator._build_deterministic_scene_script()` into reusable pieces.

Add an existing-world builder:

```python
def build_existing_world_scene_script(
    self,
    spawn_payload_filename: str,
    carla_host: str = "localhost",
    carla_port: int = 2000,
    carla_map: Optional[str] = None,
    scene_match_status: Optional[str] = None,
    scene_match_reason: Optional[str] = None,
) -> str:
    ...
```

Requirements:

- do not load a `.xodr`
- use `client.load_world(carla_map)` only when `carla_map` is provided
- otherwise use `client.get_world()`
- read `spawn_payload_filename`
- spawn entities using the same helper behavior as the deterministic XODR
  script: blueprint selection, vehicle color, lane projection, static ground
  projection, pedestrian spawning, spectator focus
- include a leading comment such as:

```python
# scene_match_status: matched
```

or:

```python
# scene_match_status: unavailable; reason: ...
```

The new main path should call this builder directly and write
`{scene_id}_scene_final.py`. It should not rely on
`SceneMapMatcher.apply_match_to_scene_script()` or
`_rewrite_scene_script_with_match()`.

### Step 9: Rewrite `FullPipeline`

File: `experiments/auto_generate_all_vlm.py`

The new default `FullPipeline` should be:

```python
scene_understanding = auto_generator.generate_scene_understanding(
    user_request,
    additional_info,
)
relation_dsl = auto_generator.generate_relation_dsl(scene_id, scene_understanding)
raw_initial = auto_generator.generate_initial_coordinates(scene_id, relation_dsl)
ordered = auto_generator.apply_pairwise_ordering_step(
    scene_id,
    raw_initial,
    relation_dsl,
)
projected = auto_generator.project_entities_step(
    scene_id,
    ordered,
    scene_understanding,
)
refined = auto_generator.refine_coordinates_step(
    scene_id,
    projected,
    relation_dsl,
)
validation = auto_generator.validate_layout_step(scene_id, relation_dsl, refined)
match_report_path = auto_generator.analyze_structured_scene_match(
    scene_id,
    scene_understanding,
    refined,
    validation,
)
auto_generator.build_spawn_payload_from_match_or_fallback(
    scene_id,
    refined,
    match_report_path,
)
auto_generator.generate_final_scene_script(scene_id, match_report_path)
```

Update default `input_info`:

```python
input_info = {
    "generation_mode": "generation",
    "input_type": "image",
    "require_carla_connection": True,
    "spawn_point_limit": 12,
    "enable_scene_match": True,
    "debug_artifacts": False,
}
```

Legacy modes may stay as compatibility code, but document that they are not the
recommended default path.

## Files Modified Summary

| File | Change |
|---|---|
| `experiments/auto_generate_all_vlm.py` | New default structured pipeline, fallback CARLA context, debug artifact policy, final payload/script writing |
| `tools/scene_map_matcher.py` | New structured matching entry point and feature extractor; legacy SUMO matching remains |
| `opendrive_experiment/agents/scenario_generator_xodr.py` | Reusable existing-world deterministic CARLA script builder |
| `opendrive_experiment/tools/structured_pipeline.py` | New in-memory initial coordinate generation function |

Tests should be added or updated. Do not leave the migration untested.

Recommended test coverage:

- default mode writes only the four core artifacts
- `debug_artifacts=True` writes additional intermediate JSON files
- CARLA unavailable produces `scene_match.status == "unavailable"` and still
  writes spawn payload and final script
- structured matcher does not require SUMO `.net.xml`, `.nod.xml`, or `.edg.xml`
- existing-world scene script has valid Python syntax and reads
  `spawn_entities.json`
- legacy `SceneMapMatcher.analyze_scene_assets()` still supports the old SUMO
  input format

## Execution Order

1. Add `generate_initial_coordinates_from_relation_dsl()` and unit-test it
   against the current coordinate-program output shape. Use the output of the
   existing `evaluate_coordinate_program()` on a representative
   `relation_dsl.json` fixture as the golden reference — the new function must
   produce numerically identical `entities[*].location` and `rotation` values
   for the same input.
2. Add CARLA context fallback and topology sampling.
3. Wire the structured scene understanding and in-memory coordinate stages.
4. Add structured scene matching and its report writer.
5. Add spawn-payload-from-match-or-fallback logic.
6. Add existing-world deterministic script builder.
7. Rewrite the default `FullPipeline`.
8. Add tests for default artifacts, debug artifacts, fallback behavior, matcher
   v2 input, generated script syntax, and legacy matcher compatibility.

## Known Risks and Mitigations

| Risk | Mitigation |
|---|---|
| CARLA is not running during generation | Return fallback context and write `scene_match.status = "unavailable"` instead of terminating |
| Structured matcher accidentally depends on old SUMO topology shape | Use a dedicated structured feature extractor; do not fake partial SUMO dictionaries |
| Final script patching becomes fragile | Do not use `_rewrite_scene_script_with_match()` in the new main path; generate final payload after matching |
| Temporary coordinate files cause cleanup/race issues | Do not write `coordinate_program.py`; compute initial coordinates in memory |
| `project_entities_to_xodr()` name is OpenDRIVE-specific | Reuse it initially because it already handles topology/lane projection; rename later only as a cleanup |
| Existing-world script builder diverges from tested XODR script helpers | Extract reusable helper/payload-loop logic from the deterministic XODR builder instead of writing a new unrelated template |

## Acceptance Criteria

- Running the new default path no longer produces SUMO road-network files.
- Running with `debug_artifacts=False` writes only the four core artifacts.
- Running with `debug_artifacts=True` writes the documented intermediate JSON
  artifacts.
- CARLA connection failure does not stop the pipeline before script generation.
- `scene_match.json` is written before `spawn_entities.json`.
- `spawn_entities.json` reflects matched coordinates when matching succeeds and
  refined fallback coordinates otherwise.
- `scene_final.py` is deterministic and does not contain LLM-generated spawn
  code.

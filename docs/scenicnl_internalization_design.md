# ScenicNL Design Internalization for AutoScenario_gen

## Summary

本文档说明如何把 `/home/zx/code/scenicNL` 中“事故报告到 CARLA/Scenic 动态场景”的优秀设计内化到 `AutoScenario_gen` 的风险生成主流程中。

核心结论：

- scenicNL 不应直接替换 `AutoScenario_gen` 当前的 risk spec / template / CARLA Python 生成链路。
- scenicNL 最值得吸收的是：事故语义分解、缺失信息概率化、受限行为库、分阶段生成、编译/验证反馈修复。
- `AutoScenario_gen` 当前应继续以 `actors.json` / `match.json` 为可执行事实来源，以原图作为主体识别和速度假设来源。
- VLM 的输出应从“直接生成 risk candidates”升级为“先输出 ScenicNL-style accident reasoning，再输出受模板约束的 risk spec”。
- 后续若要生成 Scenic 代码，应以 `risk_spec + risk_sample + spawn_payload + map_match` 为输入，从结构化事实生成 Scenic，而不是让 VLM 自由写 Scenic。

## Review And Corrections (2026-06-15)

本节记录本设计与 `AutoScenario_gen` 当前代码的核对结论。结论：战略方向正确，但有若干与现状不一致或与既有计划重叠之处，需在落地前对齐。下文各处已据此修订。

### A. 与 `raw_image_three_speed_risk_plan.md` 的关系（必须先对齐）

`raw_image_three_speed_risk_plan.md` 已经独立提出了三件与本文重叠的事：

- 在 `ego` 下新增 `speed_hypotheses_mps`（low/medium/high）。
- 重新引入原图，仅用于识别主体参与者与估计三档 ego 速度。
- 按速度档把每个 candidate 展开为 low/medium/high。

因此本文 **不应** 把这三项当作全新设计，而应视为对该计划的“ScenicNL reasoning 增量”：本文真正新增的只有 `risk_evidence`（accident reasoning 中间层）与未来的 Scenic 后端。落地顺序建议：先实现 `raw_image_three_speed_risk_plan.md`，再在其 schema 之上叠加 `risk_evidence`。两份文档对 schema 版本号的处理必须统一（见 B）。

### B. Schema 版本现状（本文原文有误）

当前代码 `tools/risk_scenario_pipeline.py` 中 `RISK_SCHEMA_VERSION = "risk-scenario-v1"`，且 `validate_risk_scenario_spec` 对非该版本号直接报错拒绝。直接把输出 schema 改成 `risk-scenario-v2` 会被现有校验拒绝。因此：

- 若与 `raw_image_three_speed_risk_plan.md` 协同，倾向 **保持 `risk-scenario-v1`，向后兼容地新增字段**（`ego.speed_hypotheses_mps`、`risk_evidence`），而不是直接跳版本。
- 若确需 v2，必须同时修改 `RISK_SCHEMA_VERSION` 常量与校验分支，并保留对 v1 的读取兼容。
- 本文后文 schema 示例中的 `"schema_version": "risk-scenario-v2"` 仅为说明目标结构，不代表必须升版。

### C. 原图用途与当前 prompt 冲突（前置条件）

当前 `agents/risk_scenario_interpreter.py` 的 `pre_prompt` 明确写着 “do not rely on raw images or BEV renders”，且该 interpreter 目前只发送纯文本 message，**没有图像编码**。本文“Raw Image”输入设计要落地，前提是：

- 反转该 prompt 约束（允许原图，但限定用途）。
- 在 `refine_request` 中加入图像编码与多模态 message。

这正是 `raw_image_three_speed_risk_plan.md` 已规划的改动，本文依赖其先落地。

### D. Phase 1 校验大多已存在（避免重复造轮子）

`validate_risk_scenario_spec` 已经实现：每个 candidate 必须含 `ego`、`involved_actor_ids` 必须是 spawn payload 中存在的 id、`template_id` 必须在库内、至少两个 actor、`confidence` 截断到 [0,1]、parameter ranges 规范化。因此 Phase 1 的真正新增项只有：

- `ego.speed_hypotheses_mps` 的 `low < medium < high` 且非负校验。
- `risk_evidence` 中引用的 actor id 是否存在于 actor table（建议软校验：不存在则降级为 `unsupported`，而非整体 reject）。

### E. 去除 `ego_target_speed_mps` 有未声明的依赖

`ego_target_speed_mps` 当前是 `tools/accident_template_library.py` 中多个模板（至少 3 个）的 `parameter_ranges` 字段，并由 `sample_risk_scenario` 采样。因此“三速展开后 candidate 不再含 `ego_target_speed_mps` 采样范围”需要：

- 从受影响模板的 `parameter_ranges` 中移除该键，或在展开/采样阶段以固定 `ego_target_speed_mps` 覆盖它。
- 同步调整 `sample_risk_scenario`，使展开后的固定速度不被重新随机采样。

### F. 数据来源命名需精确

本文用 “actors.json / match.json” 作为简称。实际：

- 内存结构是 `spawn_payload`，actor 列表在 `spawn_payload["entities"]`（键名是 `entities`，不是 `actors`）。
- 文件由 `RiskScenarioRunner` 按候选名加载：`{scene_id}_actors.json` 或 `{scene_id}_spawn_entities.json`；`{scene_id}_match.json` 或 `{scene_id}_scene_match.json`。
- 本文第 2 节那张 “actor table” 是 `build_actor_context(spawn_payload)` 的派生输出（`{"ego_actor_id": "ego", "actors": [...]}`），不是磁盘文件原貌。
- 采样已有 `sample_risk_scenario` / `sample_all_risk_scenarios`，三速展开应与它们衔接，而非另起一套。

## Why ScenicNL Is Useful

scenicNL 的目标是：

> Natural language crash report -> Scenic program -> CARLA executable dynamic scene.

它的关键设计不是“生成 Scenic 代码”本身，而是把事故文本逐步拆成：

1. 主要交通参与者
2. 参与者之间的空间关系
3. 事故事件序列
4. 缺失但影响还原的事实
5. 对缺失事实的概率分布
6. 可用行为库中的动作选择
7. 地图/道路环境选择
8. Scenic 程序分段生成
9. 编译检查和错误驱动修复

这些能力正好可以补强 `AutoScenario_gen` 当前的 VLM 风险生成阶段。

## Difference From Current AutoScenario_gen

`AutoScenario_gen` 当前流程更适合事故预测：

```text
raw image / scene understanding
        +
CARLA reconstructed actors.json
        +
scene match / map topology
        ->
VLM risk spec
        ->
template-constrained sampling
        ->
CARLA Python dynamic scenario
```

scenicNL 的流程更适合事故复现：

```text
crash report text
        ->
LLM generates Scenic code
        ->
Scenic compiler checks program
        ->
CARLA simulation
```

因此，scenicNL 应作为设计参考和中间推理模板，而不是作为当前主流程的直接执行后端。

## Current VLM Input Should Be

当前 VLM 输入应分成五类，且明确每类信息的优先级。

### 1. Raw Image

用途只限于：

- 识别图像中的主体道路参与者
- 判断哪些 actor 是视觉上重要的风险对象
- 辅助估计 ego 的 low / medium / high 三档速度
- 识别交通语义线索，例如拥堵、路口、停车、行人横穿、遮挡、车道切入趋势

禁止用途：

- 不允许用原图覆盖 `actors.json` 中的可执行位置
- 不允许根据原图发明不存在的 CARLA actor
- 不允许生成自由轨迹或 CARLA 代码

### 2. Actor Table From CARLA Spawn Payload

这是 VLM 可引用 actor 的唯一集合。它由 `build_actor_context(spawn_payload)` 从 `spawn_payload["entities"]` 派生（注意磁盘上的键是 `entities`，不是 `actors`）。

派生后的结构形如：

```json
{
  "ego_actor_id": "ego",
  "actors": [
    {
      "id": "ego",
      "category": "car",
      "x": 0.0,
      "y": 0.0,
      "yaw": 0.0,
      "lane_side_relation": "ego_lane",
      "relative_to_ego": {
        "longitudinal_m": 0.0,
        "lateral_m": 0.0,
        "distance_m": 0.0
      }
    }
  ]
}
```

这是 geometry 和可执行 actor id 的最高优先级来源。

### 3. Scene Match / Map Context

应输入：

- CARLA world name
- road / lane / junction 匹配状态
- ego 所在 lane
- actor 与 lane 的匹配关系
- 是否接近 junction / crosswalk / traffic light / stop line
- 匹配失败原因

用途：

- 判断风险模板是否物理可执行
- 判断 crossing / turning / cut-in / rear-end 等模板是否适配
- 约束 Scenic 或 CARLA Python 后端的道路选择

### 4. Scene Understanding Summary

应输入来自现有 scene understanding 的高层语义：

- road type
- weather / lighting if available
- traffic density
- visible road users
- occlusions
- traffic control devices
- image-level safety cues

注意：它的优先级低于 `actors.json` 和 `match.json`，但高于 VLM 自由推测。

### 5. Accident Template Library

继续输入当前 `tools/accident_template_library.py` 中的模板摘要。

VLM 只能从库中选择 `template_id`，不能发明事故类型。

## Current VLM Output Should Be

VLM 输出应从当前的单层 risk spec 升级为两层结构：

1. `risk_evidence`: ScenicNL-style accident reasoning
2. `risk_candidates`: AutoScenario_gen executable risk candidates

推荐 schema：

注意：`schema_version` 示例写作 `risk-scenario-v2` 仅用于展示目标结构。当前代码只接受 `risk-scenario-v1`（见上文 Review 节 B）；落地时优先在 v1 上向后兼容地新增 `risk_evidence` 与 `ego.speed_hypotheses_mps`，除非同步修改 `RISK_SCHEMA_VERSION` 与校验。

```json
{
  "schema_version": "risk-scenario-v2",
  "source_scene_id": "scene_001",
  "ego": {
    "actor_id": "ego",
    "controller_mode": "scripted",
    "route_source": "map_forward_waypoints",
    "behavior_profile": "accident_reproduction",
    "speed_hypotheses_mps": {
      "low": 4.0,
      "medium": 8.0,
      "high": 12.0
    },
    "speed_rationale": "brief reason based on road context and image cues"
  },
  "risk_evidence": {
    "main_objects": [
      {
        "actor_id": "ts1",
        "role": "lead_vehicle",
        "visual_description": "vehicle ahead of ego",
        "matched_from_actor_table": true
      }
    ],
    "spatial_relations": [
      {
        "subject_actor_id": "ts1",
        "relation": "ahead_of",
        "object_actor_id": "ego",
        "evidence": "positive longitudinal offset in actor table"
      }
    ],
    "event_hypotheses": [
      {
        "event": "lead vehicle may brake hard",
        "actors": ["ego", "ts1"],
        "evidence": "same lane, lead vehicle ahead, traffic context"
      }
    ],
    "missing_or_uncertain_facts": [
      {
        "fact": "lead vehicle future braking intensity",
        "impact": "affects rear-end collision timing",
        "modeled_as": "parameter_range"
      }
    ],
    "template_mapping": [
      {
        "template_id": "lead_vehicle_hard_brake",
        "actors": ["ego", "ts1"],
        "why_template_fits": "lead actor is ahead and close enough for rear-end risk",
        "why_executable": "both actor ids exist in actor table"
      }
    ]
  },
  "risk_candidates": [
    {
      "id": "risk_001",
      "accident_type": "rear_end",
      "template_id": "lead_vehicle_hard_brake",
      "involved_actor_ids": ["ego", "ts1"],
      "confidence": 0.75,
      "rationale": "lead vehicle ahead of ego in same lane; hard braking can create rear-end risk",
      "preconditions": [
        "ts1 remains ahead of ego",
        "ego follows map forward waypoints",
        "ts1 brakes when ego reaches trigger distance"
      ],
      "parameter_ranges": {
        "ego_reaction_delay_s": [0.8, 2.5],
        "npc_trigger_distance_m": [8.0, 18.0],
        "npc_brake_intensity": [0.6, 1.0]
      }
    }
  ],
  "unsupported_risk_hypotheses": []
}
```

Important details:

- `speed_hypotheses_mps` belongs to `ego`, not to individual candidates.
- `risk_candidates` should not include `ego_target_speed_mps` as a sampled range after three-speed expansion is introduced.
  - 注意依赖：`ego_target_speed_mps` 目前是 `tools/accident_template_library.py` 中多个模板的 `parameter_ranges` 字段，并由 `sample_risk_scenario` 采样。要去除它，必须同步修改模板库与采样逻辑（见 Review 节 E），不能只改 schema。
- The pipeline should expand each candidate into low / medium / high versions.
- Each expanded candidate should set fixed `ego_speed_label` and `ego_target_speed_mps`.
- `risk_evidence` is for traceability, debugging, future Scenic generation, and prompt self-consistency.

## ScenicNL Parts We Can Directly Reuse

### 1. Prompt Decomposition

Reusable source:

- `/home/zx/code/scenicNL/src/scenicNL/adapters/prompts/question_reasoning.txt`
- `/home/zx/code/scenicNL/src/scenicNL/adapters/prompts/tot_questions.txt`

What to reuse:

- Main objects
- Spatial relations
- Event sequence
- Missing details
- Probability distributions over missing values
- Behavior selection from a limited behavior set

How to adapt:

- Replace Scenic-specific output with JSON fields.
- Replace free map selection with `match.json` / `world_name`.
- Replace generated object names with existing CARLA actor ids.
- Replace LTL output with optional `event_hypotheses` and `termination_conditions`.

### 2. Scenic Template Structure

Reusable source:

- `/home/zx/code/scenicNL/src/scenicNL/constraints/lmql_template_limited.scenic`

Useful sections:

```text
SET MAP AND MODEL
CONSTANTS
DEFINING BEHAVIORS
DEFINING SPATIAL RELATIONS
```

How to adapt:

- Keep this section layout for future Scenic backend.
- Do not let VLM fill it freely at first.
- Generate each section from structured `risk_spec` and `risk_sample`.

### 3. Compile Feedback Loop

Reusable source:

- `construct_scenic_program`
- `construct_scenic_program_tot`
- `check_compile`

What to reuse conceptually:

- Generate in sections.
- Check after each section.
- Feed compiler error back into regeneration.
- Keep a known-good partial program.

How to adapt:

- Implement a local `ScenicScenarioGenerator` in `AutoScenario_gen`.
- Add a `check_scenic_compile()` helper if Scenic is installed.
- Use compile feedback only after deterministic generation fails.

### 4. Crash Report Dataset As Evaluation Material

Reusable source:

- `/home/zx/code/scenicNL/eval_txts`
- `/home/zx/code/scenicNL/examples`

Useful for:

- Building few-shot examples of accident reasoning.
- Expanding the accident template library.
- Testing whether the VLM can map accident descriptions to our risk templates.

Caution:

- Check licensing before copying large amounts of data.
- Prefer using small internal derived examples or manually rewritten examples.

## ScenicNL Parts We Should Not Directly Reuse

### 1. Direct LLM-to-Scenic Generation

Reason:

- It may invent actors, lanes, maps, and behaviors.
- It does not respect our `actors.json` as the executable source of truth.
- It solves reconstruction from text, while we solve prediction from an already reconstructed CARLA static scene.

### 2. Dependency Stack

scenicNL depends on older or heavy packages:

- `openai<=0.28.1`
- `scenic==3.0.0b2`
- `lmql`
- `transformers`
- local vector DB / Pinecone stubs

Do not merge its environment directly into `AutoScenario_gen`.

### 3. Free Map Selection

scenicNL asks the model to choose `Town01` / `Town02` / etc.

In our flow:

- map should come from `match.json`
- VLM may comment on map suitability
- VLM should not override the executable world unless explicitly in a separate map-selection stage

## Proposed AutoScenario_gen Architecture

Recommended staged flow:

```text
1. Static reconstruction
   image -> scene understanding -> actors.json -> match.json

2. ScenicNL-style risk reasoning
   image + actor_context + match + templates -> risk_evidence + risk_spec

3. Risk validation
   validate actor ids
   validate template ids
   validate speed hypotheses
   validate candidate parameters

4. Three-speed expansion
   each risk candidate -> low / medium / high candidate variants

5. Sampling
   fixed ego speed per speed label
   sample other parameter ranges

6. Dynamic backend
   current: CARLA Python scenario script
   future: Scenic program generation

7. Execution and metrics
   run scenario
   collect collision / TTC / distance / impact speed
```

## How To Generate Scenic Code Later

The future Scenic backend should be deterministic-first.

### Scenic Generator Input

Use:

```text
spawn_payload
match_report
risk_spec
risk_sample
map_topology
template_library
```

Do not ask VLM to write Scenic from scratch.

### Scenic Generator Output

Output one `.scenic` file per expanded risk sample:

```text
scene_001_r000_low_s000.scenic
scene_001_r000_medium_s000.scenic
scene_001_r000_high_s000.scenic
```

Each Scenic file should include:

```text
1. SET MAP AND MODEL
2. CONSTANTS
3. BEHAVIOR DEFINITIONS
4. ACTOR PLACEMENT
5. REQUIRE CONSTRAINTS
6. TERMINATION / METRIC CONDITIONS
```

### Scenic Code Generation Strategy

Use a template-per-risk-template design:

```text
lead_vehicle_hard_brake.scenic.j2
adjacent_vehicle_cut_in.scenic.j2
roadside_vehicle_pull_out.scenic.j2
straight_crossing_path.scenic.j2
pedestrian_nearside_crossing.scenic.j2
```

Each Jinja-style template maps structured fields:

- ego actor pose
- risk actor pose
- ego speed label
- ego target speed
- NPC trigger distance
- NPC behavior parameters
- map path
- route / lane constraints

The VLM may generate `risk_evidence`, but the Scenic code should be produced by deterministic Python templates.

### Example Scenic Skeleton

```scenic
param map = "/path/to/TownXX.xodr"
param carla_map = "TownXX"
model scenic.simulators.carla.model

EGO_MODEL = "vehicle.lincoln.mkz_2017"
EGO_SPEED = 8.0
NPC_BRAKE_INTENSITY = 0.8
NPC_TRIGGER_DISTANCE = 12.0

behavior EgoBehavior(speed=EGO_SPEED):
    do FollowLaneBehavior(speed)

behavior LeadHardBrakeBehavior(speed=5.0):
    try:
        do FollowLaneBehavior(speed)
    interrupt when distance to ego < NPC_TRIGGER_DISTANCE:
        take SetBrakeAction(NPC_BRAKE_INTENSITY)

ego = new Car at OrientedPoint(...)
lead = new Car at OrientedPoint(...),
    with behavior LeadHardBrakeBehavior()

require distance from ego to lead < 40
terminate when distance from ego to lead < 1.5
```

This skeleton should not be emitted by VLM directly. It should be generated from a controlled template.

## Required Changes To Current Project

### Phase 1: Improve VLM Contract

前置依赖：先反转 `RiskScenarioInterpreter.pre_prompt` 中 “do not rely on raw images” 的约束并加入图像编码（即先落地 `raw_image_three_speed_risk_plan.md`）。

Update `RiskScenarioInterpreter` prompt to require:

- `risk_evidence`
- `ego.speed_hypotheses_mps`
- actor-id-grounded spatial relations
- template mapping rationale
- unsupported hypotheses when actor mapping is not reliable

校验现状：`validate_risk_scenario_spec` 已经覆盖 “candidate 必含 ego / actor id 必须存在 / template id 必须在库 / ≥2 actor / confidence 截断” 等。Phase 1 真正 **新增** 的校验只有：

- `ego.speed_hypotheses_mps` 的 `low < medium < high` 且非负。
- `risk_evidence` 中引用的 actor id 存在性（建议软校验：不存在则降级为 unsupported，而非整体 reject，以免新增 reasoning 层降低产出率）。

### Phase 2: Add Three-Speed Expansion

Implement:

```text
expand_candidates_by_speed(spec)
```

Behavior:

- copy every original candidate into low / medium / high variants
- assign `source_candidate_index`
- assign `ego_speed_label`
- assign fixed `ego_target_speed_mps`
- remove random sampling for `ego_target_speed_mps`

衔接现状：展开应发生在 `validate_risk_scenario_spec` 之后、`sample_all_risk_scenarios` 之前，使展开产生的每个速度变体再各自走既有采样。固定 `ego_target_speed_mps` 需在 `sample_risk_scenario` 中被识别为固定值（而非两元素 range）以跳过随机采样；或在模板库中移除该参数后由展开阶段注入。

### Phase 3: Add Traceable Risk Evidence Artifacts

Persist:

```text
scene_id_risk_evidence.json
scene_id_risk_spec.json
scene_id_risk_actors.json
scene_id_risk_summary.json
```

This makes VLM errors easier to diagnose.

### Phase 4: Add Optional Scenic Backend

Add a new module:

```text
tools/scenic_scenario_generator.py
```

Responsibilities:

- load `risk_sample`
- load `spawn_payload`
- select Scenic template by `template_id`
- render `.scenic`
- optionally call Scenic compiler

Add output paths:

```text
scene_id_r000_low_s000.scenic
scene_id_r000_low_s000_scenic_compile.json
```

### Phase 5: Add Compile/Repair Loop

Initial version:

- deterministic Scenic generation
- compile with Scenic if available
- store compiler error

Later version:

- if deterministic template fails, call LLM only to repair the failing section
- provide:
  - known-good partial Scenic
  - failing section
  - compiler error
  - original structured risk sample
- require repaired output to preserve actor ids and template semantics

## Recommended Prompt Contract

The VLM should be told:

```text
You are not generating CARLA code or Scenic code.
You are producing a structured accident-risk interpretation.

Use the raw image only to identify salient road users, visual traffic cues,
and ego speed hypotheses.

Use the actor table as the only executable actor set.
Every risk candidate must reference existing actor ids.

Before writing risk_candidates, first produce risk_evidence:
- main objects
- spatial relations
- event hypotheses
- uncertain facts
- template mapping rationale

Then choose risk_candidates only from the accident template library.
If a visually plausible risk cannot be mapped to existing actor ids or templates,
put it under unsupported_risk_hypotheses.
```

## Key Design Principle

The main design principle should be:

> VLM reasons; deterministic code validates, expands, samples, and executes.

scenicNL shows that LLM-generated accident scenarios become much better when the model is forced to reason through objects, relations, events, uncertainty, and behavior constraints. `AutoScenario_gen` should absorb that reasoning structure while keeping execution grounded in its own actor table, template library, validators, and CARLA/Scenic generators.


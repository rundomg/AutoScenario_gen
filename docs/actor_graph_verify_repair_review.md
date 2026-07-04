# Actor Graph Verify-Repair 更新实施计划

## 目标

将当前 `原图 + ego-view + 当前 JSON → VLM check` 的 verify-repair，升级为一个更确定、可解释、可调试的 actor graph 验证闭环：

```text
scene_understanding / relation_dsl → source_actor_graph
spawn_entities + CARLA actual transforms → render_actor_graph
compare(source_actor_graph, render_actor_graph) → repair_plan
repair_plan → 路由到 spawn_payload / scene_understanding / projection_logic
```

v1 不重新从原图独立抽取 reference，不验证道路、天气、建筑、地图外观，只关注车辆类 actor 的：

- 类别是否一致
- 数量是否一致
- ego-centric 左右/前后/远近关系是否一致
- heading 是否一致
- 是否发生 overlap 或 CARLA spawn failure

VLM 不再作为主 verifier。它保留为 fallback、低置信 case 的辅助解释器，以及原方案兼容路径。

---

## 总体设计

### 1. Source Actor Graph

`source_actor_graph` 从当前已有的 `scene_understanding → relation_dsl` 派生，不额外调用 VLM。

数据来源：

- `tools/structured_pipeline.py::build_relation_dsl()`
- `relation_dsl.ego_relations`
- `relation_dsl.pairwise_relations`

每个 source actor 至少包含：

```json
{
  "id": "car_row_0_1",
  "group_id": "car_row_0",
  "category": "car",
  "subtype": "car",
  "spawn_kind": "vehicle",
  "lane_side_relation": "right_edge",
  "longitudinal_relation": "ahead",
  "distance_band": "near",
  "heading_relation": "same_direction",
  "visual_confidence": "medium"
}
```

输出文件：

```text
{scene_id}_source_actor_graph.json
```

v1 只纳入车辆类 actor：

```text
car, truck, bus, motorcycle, bicycle
```

pedestrian、cone、barrier 等暂不进入主评分，可以保留在 graph 中作为 `ignored_actor` 或 `out_of_scope_actor`。

---

### 2. Render Actor Graph

`render_actor_graph` 描述 CARLA 中实际生成出来的 actor。优先使用 CARLA truth，而不是从 ego-view 图片里重新识别。

新增环境变量：

```text
AUTOSCENARIO_RENDER_ACTOR_GRAPH_OUTPUT=/path/to/{scene_id}_render_actor_graph_r1.json
```

final scene script 在 spawn 完成后写出 render graph。

关键要求：必须遍历原始 `spawn_payload.entities`，不能只遍历 `_autoscenario_actor_by_id`。因为 `_autoscenario_actor_by_id` 只包含 spawn 成功的 actor；spawn 失败的 entity 如果不遍历 payload 会直接消失，导致无法区分“少生成”和“spawn 失败”。

每个 render actor 至少包含：

```json
{
  "id": "car_row_0_1",
  "category": "car",
  "blueprint_name": "vehicle.tesla.model3",
  "spawn_kind": "vehicle",
  "spawned": true,
  "spawn_failure_reason": null,
  "location": {"x": 12.3, "y": 45.6, "z": 0.3},
  "yaw": 91.2,
  "ego_frame": {
    "longitudinal_m": 18.4,
    "lateral_m": 3.2,
    "distance_band": "mid",
    "lane_side_relation": "right_lane"
  },
  "heading_relation_to_ego": "same_direction",
  "truth_source": "carla_actor_transform"
}
```

如果某个 payload entity 没有对应 actor：

```json
{
  "id": "car_row_0_1",
  "category": "car",
  "spawned": false,
  "spawn_failure_reason": "try_spawn_actor_returned_none_or_collision",
  "truth_source": "carla_actor_transform"
}
```

输出文件：

```text
{scene_id}_render_actor_graph_r{round_index}.json
```

---

### 3. Fallback 模式限制

当 CARLA truth 文件没有生成时，可以从 `spawn_entities` 生成 fallback render graph，但该模式不能冒充完整验证。

fallback graph 必须标记：

```json
{
  "truth_source": "spawn_payload_fallback",
  "position_checks_enabled": false,
  "spawn_checks_enabled": false
}
```

fallback 模式只允许：

- count/category 静态一致性检查
- schema / ID 完整性检查
- spawn payload 中明显 overlap 的 warning

fallback 模式必须跳过：

- lane-side 位置检查
- longitudinal/lateral 误差检查
- CARLA spawn failure 检查
- blueprint 实际生成结果检查

fallback 模式下不要输出强语义的 `passed=true`。建议输出：

```json
{
  "status": "static_checked",
  "passed": false,
  "score": null,
  "truth_unavailable": true
}
```

这样避免离线环境里出现虚假高分。

---

## Entity ID 链路要求

这是实现前必须先确认的 P0 项。

ID 必须在以下链路中保持一致：

```text
relation_dsl.ego_relations[].entity_id
  → coordinates.entities[].id
  → spawn_entities.entities[].id
  → final_scene_script._autoscenario_actor_by_id key
  → render_actor_graph.actors[].id
```

当前代码形状看，链路大概率已通：

- `build_relation_dsl()` 展开 count 后生成 `entity_id`
- 坐标生成阶段写成 `entities[].id`
- `build_projected_spawn_payload()` 保留 `entity["id"]`
- final scene script 使用 `entity_id = str(entity.get('id') or '')`

但实现前仍需增加一个非侵入式 debug/dump 测试，至少覆盖：

- `count=1`
- `count>1`
- curbside row
- motorcycle / bicycle

需要确认 count 展开命名，例如：

```text
count=1: car_row_0
count=3: car_row_0_0, car_row_0_1, car_row_0_2
```

如果任何一环 ID 不一致，compare 不应退化为无提示的 nearest-neighbor 匹配，而应输出：

```json
{
  "issue_type": "id_chain_mismatch",
  "requires_code_fix": true
}
```

---

## Graph Compare 与 Repair Plan

新增模块建议：

```text
tools/actor_graph_verifier.py
```

核心函数：

```python
build_source_actor_graph(scene_understanding, relation_dsl) -> dict
build_render_actor_graph_from_spawn_payload(spawn_payload) -> dict
compare_actor_graphs(source_graph, render_graph, validation=None) -> dict
```

`compare_actor_graphs()` 输出 repair plan：

```json
{
  "passed": false,
  "score": 0.68,
  "truth_source": "carla_actor_transform",
  "systematic_error": false,
  "requires_code_fix": false,
  "issues": [
    {
      "issue_type": "lane_side_mismatch",
      "target_artifact": "spawn_payload",
      "operation": "shift_lateral",
      "source_entity_id": "car_1",
      "render_entity_id": "car_1",
      "severity": "medium",
      "evidence": "source expects right_edge, render actor is near same_lane"
    }
  ],
  "repair_actions": []
}
```

### Issue 类型

v1 支持：

```text
missing_actor
extra_actor
count_mismatch
category_mismatch
lane_side_mismatch
longitudinal_mismatch
heading_mismatch
overlap
spawn_failure
id_chain_mismatch
truth_unavailable
systematic_projection_error
```

---

## Repair 路由规则

不要简单按“语义/几何”二分，尤其是 `lane_side_mismatch` 需要更细。

### 1. 修 scene_understanding

路由到 `scene_understanding` 的情况：

- `count_mismatch`
- `category_mismatch`
- `missing_actor` 且 source graph 确认应存在
- source graph 的 `lane_side_relation` 本身明显不合理

处理方式：

```text
repair_plan → revise_with_verification_feedback()
→ 输出新的 scene_understanding JSON
→ 重跑 relation_dsl / coordinates / projection / validation / spawn_payload / final scene script
```

### 2. 修 spawn_payload

路由到 `spawn_payload` 的情况：

- `heading_mismatch`
- `overlap`
- 小范围 `longitudinal_mismatch`
- 小范围 `lane_side_mismatch`，且 actor 使用 `preserve_xy` / `direct` 或确认不会被后续 CARLA snapping 覆盖

处理方式：

- 将 repair plan 转成当前 `_apply_layout_repair_actions()` 支持的结构化 action。
- 继续复用现有确定性修复逻辑。
- 修复后重新生成 final scene script，再进入下一轮 verify。

### 3. 标记 projection_logic

路由到 `projection_logic` 的情况：

- source graph 与 spawn payload 的 lane-side 一致，但 CARLA truth 位置系统性错误
- actor 使用 `project_to_lane`，修改 x/y 后会被 final scene script 再次吸附回错误 lane
- 连续两轮同一 entity 同类 lane-side/longitudinal issue 未消失
- 同类 issue 同一方向出现在多个不同 entity 上

v1 不自动修改 Python 投影代码，只输出：

```json
{
  "requires_code_fix": true,
  "target_artifact": "projection_logic",
  "repair": "blocked_for_code_fix"
}
```

并立即停止当前 verify-repair 循环，避免空转。

---

## Systematic Error 检测

`compare_actor_graphs()` 必须检测系统性错误，避免同一个 projection bug 被当成多个局部坐标问题反复修。

触发条件：

```text
同一 issue_type + 同一 direction 出现在 >= 3 个不同 entity
```

或：

```text
连续两轮 repair 后，同一 entity 的同类 issue 仍然存在
```

触发后输出：

```json
{
  "systematic_error": true,
  "requires_code_fix": true,
  "stop_repair_loop": true,
  "suspected_module": "tools/structured_pipeline.py::_project_vehicle_entity"
}
```

其中 `N=3` 是默认值，但不能只依赖该阈值；小场景要靠“连续两轮同类问题仍存在”来触发。

---

## 接入主流程

修改 `experiments/auto_generate_all_vlm.py::verify_and_repair_spawn_layout()`。

历史上按分阶段接入推进；当前实现已进入 actor graph 默认主控阶段。

### 阶段 1：Artifact-only

历史阶段：默认仍使用当前 VLM verify-repair。

额外生成：

```text
{scene_id}_source_actor_graph.json
{scene_id}_render_actor_graph_r{round}.json
{scene_id}_actor_graph_repair_plan_r{round}.json
```

用于观察 actor graph verifier 与 VLM verifier 的差异。

### 阶段 2：Hybrid

新增模式：

```text
verify_mode="actor_graph"
```

actor graph repair plan 作为主 decision source。

VLM 只在以下情况调用：

- `truth_unavailable`
- graph compare low-confidence
- ID 链不完整但仍需要自然语言解释
- 用户显式要求 VLM judge

### 阶段 3：Actor Graph Default（当前默认）

当前 `AutoGenerator` 默认：

```text
verify_mode="actor_graph"
```

保留：

```text
verify_mode="vlm"
```

作为回退路径。

---

## 测试计划

### 单元测试

新增 `tests/test_actor_graph_verifier.py`：

- `count=1` 与 `count>1` source graph 展开正确
- source graph 保留 `entity_id/group_id/category/lane_side_relation/distance_band`
- spawn payload fallback graph 正确标记 `truth_source=spawn_payload_fallback`
- fallback 模式跳过位置检查，不输出强 `passed=true`
- source/render 数量不同输出 `count_mismatch`
- source/render 类别不同输出 `category_mismatch`
- source/render lane-side 不同输出 `lane_side_mismatch`
- heading 差约 180 度输出 `heading_mismatch`
- spawn payload 中有 entity 但 actor truth 缺失时输出 `spawn_failure`
- 同类同方向 issue >= 3 个时输出 `systematic_error`
- 连续两轮同一 issue 未修复时输出 `requires_code_fix`

### 集成测试

扩展现有 pipeline tests：

- mock final scene script 写出 render actor graph，验证主循环能读入并生成 repair plan
- mock CARLA truth 缺失，验证 fallback scope 被限制
- 验证 actor graph repair plan 可以转成 `_apply_layout_repair_actions()` 所需格式
- 验证 `verify_mode="vlm"` 仍保持原行为

---

## 实施优先级

1. 打通并测试 entity ID 链，覆盖 `count>1` 场景。
2. 实现 source/render actor graph artifact 写出，但暂不接管 repair。
3. 实现 fallback scope 限制，避免离线虚假通过。
4. 实现 `compare_actor_graphs()` 与 repair plan 输出。
5. 实现 systematic error 检测。
6. 将 repair plan 路由到现有 `_apply_layout_repair_actions()` / `revise_with_verification_feedback()`。
7. 接入 `verify_mode="actor_graph"`。
8. 观察稳定后再考虑设为默认。

---

## 相关代码位置

| 文件 | 关键函数 | 说明 |
|---|---|---|
| `experiments/auto_generate_all_vlm.py` | `_verify_scene_round()` | 当前 VLM 验证入口，actor graph 模式下改为 fallback |
| `experiments/auto_generate_all_vlm.py` | `_apply_layout_repair_actions()` | 现有确定性修复，继续复用 |
| `experiments/auto_generate_all_vlm.py` | `verify_and_repair_spawn_layout()` | 新 verify-repair 主循环接入点 |
| `tools/structured_pipeline.py` | `build_relation_dsl()` | source actor graph 数据来源 |
| `tools/structured_pipeline.py` | `_project_vehicle_entity()` | systematic projection issue 的高概率位置 |
| `tools/structured_pipeline.py` | `build_projected_spawn_payload()` | spawn payload / fallback graph 数据来源 |
| `agents/existing_world_scenario_generator.py` | `_build_spawn_payload_loop()` | 写出 render actor graph 的位置 |
| `agents/scene_verification_agent.py` | `SceneVerificationAgent` | VLM fallback verifier |

---

## 运行案例记录：`auto_result_20260609_152144`

结果目录：

```text
results/auto_result_20260609_152144
```

这次运行已经生成了更新后的 artifacts：

```text
s0000_c0_source_actor_graph.json
s0000_c0_render_actor_graph_r1.json
s0000_c0_render_actor_graph_r2.json
s0000_c0_actor_graph_repair_plan_r1.json
s0000_c0_actor_graph_repair_plan_r2.json
s0000_c0_spawn_repair.json
```

当时主流程仍在 `verify_mode="vlm"` 下运行，因此 actor graph repair plan 只作为 artifact/辅助信息存在，没有接管主决策。该问题已在后续修正：默认 `verify_mode` 已切到 `actor_graph`，且显式 `vlm` 模式也会尊重 actor graph hard-stop。

### 现象

第一轮 VLM verify：

```text
s0000_c0_verify_r1.json
passed=false
score=0.45
capture_mode=ego_view
recommended_stage=scene_understanding
```

VLM 发现的主要问题：

- `ts1` motorcycle 应在近前景略偏左，但生成结果太远、偏中间。
- `ts4` orange scooter 应停在近右侧 curb，生成结果在/靠近 driving lane，且像 moving rider。
- `ts3` 右侧 parked white car row 缺失或严重不足。
- `ts6` 远处 silver oncoming car 没有正确生成。
- `ts7` 远处 red car 的位置和与 `ts6` 的左右关系不对。
- `ts4` 与 `ts3` 的前后关系没有保持。

第一轮之后主循环因为存在 geometry actions，优先执行：

```text
repair=deterministic_layout_repair
```

而不是根据 `recommended_stage=scene_understanding` 先修 scene understanding。

第二轮 VLM verify：

```text
s0000_c0_verify_r2.json
passed=false
score=0.22
repair=max_rounds_reached
```

第二轮画面问题更严重：

- `ts1` 近前景 scooter 直接不可见。
- `ts3` 右侧 parked car row 不可见。
- `ts4` 近右侧 orange scooter 不可见。
- `ts5` 中右侧 scooter 不可见。
- `ts6` 与 `ts7` 没有作为两个独立远处 actor 正确恢复。

### Actor Graph 发现的问题

第一轮 actor graph repair plan：

```text
s0000_c0_actor_graph_repair_plan_r1.json
passed=false
score=0.0
stop_repair_loop=true
requires_code_fix=true
systematic_reason=lane_side_mismatch:right_edge->right_lane:multiple_entities
```

典型 issues：

- `ts3_0/ts3_1/ts3_2/ts3_3/ts4/ts5`：source 期望 `right_edge`，render 落到 `right_lane`。
- `ts1/ts6`：source 期望 `left_lane`，render 落到 `same_lane`。
- `ts7`：source 期望 `same_lane`，render 落到 `left_edge`。
- 多个同方向 lane-side mismatch 触发 `systematic_projection_error`。

第二轮 actor graph repair plan：

```text
s0000_c0_actor_graph_repair_plan_r2.json
passed=false
score=0.04
stop_repair_loop=true
requires_code_fix=true
systematic_reason=longitudinal_mismatch:mid->far:multiple_entities
```

新增关键问题：

- `ts2` spawn failed：`try_spawn_actor_returned_none_or_collision`
- `ts3_2/ts3_3/ts5` 仍从 source `mid` 变成 render `far`
- `ts4` 从 source `near` 变成 render `mid`
- 多个同方向 longitudinal mismatch 触发新的 `systematic_projection_error`

### 修复实际改动

第一轮 deterministic repair 修改的是 spawn payload：

```text
s0000_c0_actors.json
```

`repair_metadata.last_applied_repairs` 显示执行了大量 lateral 修正：

```text
lane_side_validation:
ts1 -7.0m
ts2 -4.0m
ts3_0 +4.0m
ts3_1 +4.0m
ts3_2 +4.0m
ts3_3 +4.0m
ts4 +4.0m
ts5 +4.0m
ts6 -7.0m
ts7 +6.436m
```

随后 `_apply_layout_repair_actions()` 又对 `current_validation["pairwise_results"]` 中所有 failed pairs 执行 pairwise lateral 修复，导致同一 actor 被多次累积移动。例如：

```text
ts1: 多次 -2.0m，并额外 -11.436m
ts2: -4.936m，并多次 -1.375m
ts3_0/ts3_1/ts3_2/ts3_3: 各 +2.0m
ts4: +2.0m
ts5: +2.0m
ts7: 多次 +2.186m
ts6: -11.436m
```

第二轮 render actor graph 证明这些修复出现过度移动或副作用：

```text
r1 ts1 lateral ~= -0.001m
r2 ts1 lateral ~= -41.552m

r1 ts2 spawned=true
r2 ts2 spawned=false, try_spawn_actor_returned_none_or_collision

r1 ts3_*/ts4/ts5 lateral ~= 3.0m
r2 ts3_*/ts4/ts5 lateral ~= 9.0m
```

这解释了为什么第二轮 VLM 看到多个 actor 消失：它们不是语义上被删除，而是被修复逻辑推到视野外、不可见区域，或导致 spawn collision。

### 暴露的设计缺陷

1. VLM 模式下没有尊重 actor graph 的 `stop_repair_loop`（已修复）

历史问题：当时逻辑只有在 `verify_mode=="actor_graph"` 时才处理：

```text
actor_graph_plan.stop_repair_loop
actor_graph_plan.truth_unavailable
```

因此即使 actor graph 已经检测到 `requires_code_fix=true` 和 systematic projection error，VLM 模式仍继续执行 deterministic layout repair，造成空转和过度修复。

当前修复：即使 `verify_mode="vlm"`，每轮 repair 前也会检查 actor graph plan；只要 `stop_repair_loop` 或 `requires_code_fix` 为真，就记录 `blocked_for_code_fix` 并停止，不再执行 VLM verify 或 deterministic repair。

改进建议：

- 在 `verify_mode="vlm"` 下也读取 actor graph plan 的 hard stop 信号。
- 如果 `systematic_error=true` 且 `requires_code_fix=true`，应优先记录 `blocked_for_code_fix`，停止继续挪 spawn payload。
- 或至少禁止对同方向系统性 mismatch 执行局部 deterministic repair。

2. Geometry action 优先级过高，盖过了 scene-understanding 级问题（已修复）

第一轮 VLM 同时报告了：

```text
count_mismatch
category_mismatch
lane_side_mismatch
pairwise_mismatch
```

历史问题：由于存在 geometry actions，流程直接进入 deterministic repair，没有先处理 `recommended_stage=scene_understanding` 和高严重度语义问题。

当前修复：当 `recommended_stage=scene_understanding`，或 semantic actions 中存在 high severity `count_mismatch` / `category_mismatch` 时，优先执行 scene-understanding revision，即使同一轮同时存在 geometry actions。

改进建议：

- 当 `semantic_actions` 中存在 high severity `count_mismatch` / `category_mismatch`，且 VLM `recommended_stage=scene_understanding` 时，不应因为同时存在 geometry action 就优先挪坐标。
- repair routing 应先判断问题是否可通过移动已有 actor 解决。缺 actor、错类别、visible parked row 缺失这类问题应路由到 scene understanding 或 source graph 修复。

3. Pairwise validation fallback 会对所有 failed pairs 无差别累积修复（已修复）

历史问题：`_apply_layout_repair_actions()` 会扫描 `current_validation["pairwise_results"]`，对所有 failed pairs 执行 lateral/longitudinal adjustment。若一个 actor 同时参与多个 failed pairs，就会被重复移动。

这次 `ts1` 被多次 lateral correction 后，render lateral 变成约 `-41.552m`，明显超出合理 lane/edge 范围。

当前修复：

- 每个 entity 每轮最多执行一次 lateral correction 和一次 longitudinal correction。
- 对同一 entity 的多个 pairwise failures 先聚合为目标约束，再求一个 bounded correction。
- 单轮 lateral correction 上限为一个 lane-width 级别，即 `3.5m`。
- 超过上限时不执行该 lateral repair，写入 `repair_metadata.blocked_repairs`，并设置 `repair_metadata.blocked_for_code_fix=true`。

4. VLM target_relation 没有规范化，部分 action 实际不可执行

第一轮 VLM 输出了：

```text
target_relation=near_ahead_left
target_relation=before_right_curbside_row
```

当前执行器只识别有限集合：

```text
ahead_of
behind_of
left_of
right_of
```

因此这些复合关系没有被精确执行，repair 退化为 validation-driven pairwise adjustment。

改进建议：

- 在 `SceneVerificationAgent.normalize_report()` 中将 target_relation 限制为可执行枚举。
- 复合关系应拆成结构化字段，例如：

```json
{
  "type": "pairwise_mismatch",
  "entity_id": "ts1",
  "reference_entity_id": "ego",
  "target_longitudinal_relation": "ahead_of",
  "target_lateral_relation": "left_of",
  "target_distance_band": "near"
}
```

- 不可执行的 action 应标记为 `unsupported_repair_action`，不要静默忽略。

5. Spawn failure 应作为 hard issue 参与下一步路由

第二轮 `ts2` 出现：

```text
spawn_failure: try_spawn_actor_returned_none_or_collision
```

这通常说明修复后坐标不可用或和环境/其他 actor 冲突。继续 VLM 验证只会看到 actor 缺失。

改进建议：

- actor graph plan 出现 `spawn_failure` 时，应优先路由到 overlap/spacing repair 或 `blocked_for_code_fix`。
- VLM 报告中的 missing actor 应与 render graph 的 `spawned=false` 对齐，区分“未规划生成”和“规划了但 spawn 失败”。

### 后续改进任务

- 已完成：在 VLM 模式下增加 actor graph hard-stop gate，`stop_repair_loop || requires_code_fix` 会中断局部 repair。
- 已完成：重写 repair routing，`recommended_stage=scene_understanding` 或 high severity semantic mismatch 优先于 geometry movement。
- 已完成：修改 `_apply_layout_repair_actions()`，对 pairwise repairs 做 per-entity aggregation、dedup、bounded correction。
- 将 VLM `repair_actions.target_relation` schema 改为可执行枚举或拆分字段。
- 已完成：增加 regression test，验证同一 entity 的多个 pairwise lateral failures 只触发一次聚合修复，且超过 `3.5m` 时会 block。
- 已完成：增加 integration test，验证 actor graph plan 已经 `requires_code_fix=true` 时，即使 `verify_mode="vlm"` 也不继续执行 VLM verify 或 deterministic layout repair。

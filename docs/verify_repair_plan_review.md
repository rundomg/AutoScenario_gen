# Verify-Repair 改造方案 v3

## 目标

将当前 verify-repair 从“BEV 验证 + 自然语言 repair hints + 少量 pairwise nudge”升级为一个真正可闭环的布局修复模块：

1. Actor layout 验证优先使用 ego 第一人称前视图，BEV 只作为 fallback。
2. VLM 输出结构化 `repair_actions`，不再依赖解析自然语言 `repair_hints`。
3. Deterministic repair 分别处理 lane-side、pairwise、heading、overlap 等问题。
4. `scene_understanding_revision` 只处理 actor 数量/类别等语义错误。
5. 生成脚本默认保留 CARLA 世界已有车辆，并在 spawn 前跳过重叠 actor。

当前代码尚未实现本方案。现有实现仍使用 `_capture_bev_image()`、`bev_image_path`、`_shift_spawn_payload_coordinates()`，并且生成脚本仍无条件调用 `_autoscenario_clear_existing_dynamic_actors()`。

---

## 1. Capture：ego-view 优先，BEV fallback

### 当前问题

现有 `_capture_bev_image()` 只设置：

```python
env["AUTOSCENARIO_BEV_OUTPUT"] = bev_path
```

然后运行最终场景脚本，得到一张 BEV 图。VLM 必须从 BEV 中推断 ego 朝向和左右关系，容易误判。

### 新方案

新增 `_capture_layout_images()`，替代 spawn layout verify-repair 阶段的 `_capture_bev_image()`。

返回结构：

```python
{
    "layout_image_path": ".../s0000_c0_ego_r1.png",
    "ego_view_path": ".../s0000_c0_ego_r1.png",
    "bev_path": ".../s0000_c0_bev_r1.png",
    "capture_mode": "ego_view",
    "error": None,
}
```

运行 scene script 时同一 subprocess 同时设置：

```python
env["AUTOSCENARIO_EGO_VIEW_OUTPUT"] = ego_view_path
env["AUTOSCENARIO_BEV_OUTPUT"] = bev_path
```

优先级：

1. 如果 ego-view 文件存在，`layout_image_path = ego_view_path`，`capture_mode = "ego_view"`。
2. 如果 ego-view 不存在但 BEV 存在，`layout_image_path = bev_path`，`capture_mode = "bev_fallback"`。
3. 如果两者都不存在，返回 capture error。

这样 ego spawn 失败时不会让整轮 verify 完全没有反馈。

---

## 2. 生成脚本：新增 ego-view capture 分支

### 需要修改的地方

这不是普通 Python runtime 方法改动，而是 `agents/existing_world_scenario_generator.py` 中的字符串生成逻辑。

当前生成脚本只有：

```python
def _autoscenario_capture_bev_if_requested():
    output_path = os.environ.get("AUTOSCENARIO_BEV_OUTPUT")
    ...
```

需要在生成出来的脚本中新增：

```python
def _autoscenario_capture_ego_view_if_requested(actor_by_id=None):
    output_path = os.environ.get("AUTOSCENARIO_EGO_VIEW_OUTPUT")
    if not output_path:
        return
    ego_actor = None
    if actor_by_id:
        ego_actor = actor_by_id.get("ego")
    if ego_actor is None:
        # fallback path: do not fail script; caller will use BEV if it exists.
        return
    # attach RGB camera to ego and save one frame
```

默认 ego camera：

```text
Location(x=1.2, y=0.0, z=1.6)
Rotation(pitch=-5.0, yaw=0.0, roll=0.0)
FOV=90
image_size=1024
```

静态重建脚本需要保留 `actor_by_id`，至少要能定位 `ego` actor。风险场景路径已经有 `actor_by_id`，静态路径也需要同步。

### 待补充：静态脚本 `actor_by_id` 构建

当前 `_build_spawn_payload_loop()` 的 spawn 循环只向 `_AUTOSCENARIO_SPAWNED_ACTORS` list 追加，没有建 id→actor 映射：

```python
# 当前生成代码：只有列表，无 id 映射
_AUTOSCENARIO_SPAWNED_ACTORS.append(actor)
```

ego-view capture 调用点在 spawn 循环结束后（紧跟 `_autoscenario_capture_bev_if_requested()` 旁边），此时没有 id→actor 映射可用。

需要在生成脚本的 spawn 循环里同时维护 `_autoscenario_actor_by_id = {}`，成功 spawn 后写入：

```python
_autoscenario_actor_by_id[str(entity.get('id', ''))] = actor
```

spawn 循环结束后调用：

```python
_autoscenario_capture_ego_view_if_requested(_autoscenario_actor_by_id)
```

这是 `_build_spawn_payload_loop()` 里的字符串生成改动，不是 runtime Python 方法改动。

### 解决方案：静态脚本同步建立 actor 映射

在静态重建脚本的生成字符串中，spawn payload 循环开始前创建：

```python
_autoscenario_actor_by_id = {}
```

每个实体 spawn 后统一执行：

```python
entity_id = str(entity.get('id') or '')
if actor is not None and entity_id:
    _autoscenario_actor_by_id[entity_id] = actor
```

该逻辑需要覆盖三条 spawn 分支：

- pedestrian
- static prop
- vehicle，包括 `direct/preserve_xy` 和 `project_to_lane`

循环结束后按顺序调用：

```python
_autoscenario_focus_spectator()
_autoscenario_capture_ego_view_if_requested(_autoscenario_actor_by_id)
_autoscenario_capture_bev_if_requested()
```

ego-view capture 失败不能让脚本退出；只要 BEV 成功，调用方仍可使用 `bev_fallback`。

---

## 3. Verification prompt：按 capture_mode 切换

### 接口改动

`_verify_scene_round()` 的签名从：

```python
def _verify_scene_round(..., bev_path, ...)
```

改为：

```python
def _verify_scene_round(..., layout_image_path, capture_mode, ...)
```

传给 `SceneVerificationAgent` 的 `add_info` 使用：

```python
{
    "source_image_path": image_path,
    "layout_image_path": layout_image_path,
    "capture_mode": capture_mode,
    ...
}
```

为了兼容旧代码，`SceneVerificationAgent.refine_request()` 可以暂时 fallback 到 `bev_image_path`，但新路径应优先使用 `layout_image_path`。

### 解决方案：接口统一为 layout image

`_capture_layout_images()` 是唯一的 spawn-layout capture 入口，返回：

```python
{
    "layout_image_path": selected_path,
    "ego_view_path": ego_view_path_or_none,
    "bev_path": bev_path_or_none,
    "capture_mode": "ego_view" 或 "bev_fallback",
    "error": None,
}
```

`_verify_scene_round()` 必须重命名参数：

```python
def _verify_scene_round(
    self,
    scene_id,
    round_index,
    image_path,
    layout_image_path,
    capture_mode,
    user_scene_description,
    scene_understanding,
    match_report_path,
):
```

传给 VLM agent 的 `add_info` 使用：

```python
{
    "source_image_path": image_path,
    "layout_image_path": layout_image_path,
    "capture_mode": capture_mode,
    ...
}
```

`SceneVerificationAgent.refine_request()` 中读取第二张图的优先级：

```python
layout_image_path = add_info.get("layout_image_path") or add_info.get("bev_image_path")
```

所有新 prompt 分支只看 `capture_mode`，不再通过文件名判断图片类型。

### ego-view prompt

当 `capture_mode == "ego_view"` 时，prompt 描述为：

```text
The first image is the original ego-view input.
The second image is the generated CARLA ego-view image.
```

必须加入盲区约束：

```text
Do NOT penalize actors not visible in the forward ego-view field of view.
Only judge actors that appear in both images or are expected to appear in the forward cone based on the scene description.
```

原因：ego-view 前视图看不到 ego 后方、侧后方或被遮挡 actor。不能因为生成图看不到这些 actor，就直接判定 actor count 不足。

### BEV fallback prompt

当 `capture_mode == "bev_fallback"` 时，沿用 BEV 语义：

```text
The second image is the generated CARLA BEV.
Use ego actor yaw/location and spawn_entities to infer ego-centric left/right/ahead/behind.
```

Clean map match 阶段继续使用原本 BEV prompt，不受本改造影响。

---

## 4. `repair_actions` schema

### DEFAULT_REPORT

`SceneVerificationAgent.DEFAULT_REPORT` 增加：

```python
"repair_actions": []
```

### VLM 输出字段

VLM report 增加可选字段：

```json
{
  "repair_actions": [
    {
      "type": "lane_side_mismatch",
      "entity_id": "ts3",
      "target_lane_side": "left_edge",
      "severity": "major",
      "evidence": "The vehicle appears in the adjacent driving lane instead of the roadside parking band."
    }
  ]
}
```

支持类型：

```text
lane_side_mismatch
pairwise_mismatch
category_mismatch
count_mismatch
overlap
heading_mismatch
```

`pairwise_mismatch` 必须包含 `reference_entity_id`：

```json
{
  "type": "pairwise_mismatch",
  "entity_id": "ts1",
  "reference_entity_id": "ts3",
  "target_relation": "ahead_of",
  "severity": "major",
  "evidence": "ts1 should be ahead of ts3 but appears behind it."
}
```

### normalize 规则

`normalize_report()` 按 action type 分别校验：

必须有 `entity_id`：

```text
lane_side_mismatch
pairwise_mismatch
overlap
heading_mismatch
```

`pairwise_mismatch` 还必须有：

```text
reference_entity_id
```

`entity_id` 可选：

```text
count_mismatch
category_mismatch
```

非法 action 过滤掉。`repair_hints` 保留给人看，但机器 repair 不解析自然语言 hints。

---

## 5. Validation 输出补强

### 当前问题

当前 `validate_relation_layout()` 的 lateral 检查中写死了 `3.5`：

```python
lateral_ok = abs((lateral / 3.5) - expected_lateral_band) <= 0.8
```

而 `_expected_lateral_band()` 返回的是 band 乘数，不是米：

```text
same_lane = 0.0
left_lane_center = -1.0
left_curbside = -2.0
right_curbside = 2.0
```

repair 需要米制目标值，不能继续猜 `3.5m`。

### 新输出

`validate_relation_layout()` 中从 `relation_dsl` 推导：

```python
lane_context = relation_dsl.get("lane_context") or {}
lane_width_m = ...
```

`lane_width_m` 从 lane context 或 lane width class 计算，不能硬编码。

`entity_results` 增加：

```json
{
  "entity_id": "ts3",
  "lane_side_relation": "left_edge",
  "status": "fail",
  "longitudinal_check": true,
  "lateral_check": false,
  "actual_longitudinal_m": 5.3,
  "expected_longitudinal_m": 4.0,
  "actual_lateral_m": -3.0,
  "expected_lateral_band": -1.857,
  "expected_lateral_m": -6.5,
  "lateral_error_m": 3.5
}
```

顶层增加：

```json
{
  "lane_context": {...},
  "lane_width_m": 3.5
}
```

`entity_results` 负责 ego-relative 位置精度；`lane_results` 只负责 actor 类型是否允许出现在对应 lane role。repair 优先响应 `entity_results` 中的 lane-side fail。

### 待补充：`expected_lateral_band` 示例值与换算公式

JSON 示例中的 `-1.857` 是一个任意小数，与 `_expected_lateral_band()` 实际返回值不符。该函数返回整数或 0.5 步长的乘数（`same_lane=0.0, left_lane_center=-1.0, left_curbside=-2.0` 等），不会出现 `-1.857`。示例应改为合法值：

```json
"expected_lateral_band": -2.0,
"expected_lateral_m": -7.0
```

换算公式需在实现说明中写明：

```python
expected_lateral_m = expected_lateral_band * lane_width_m
# e.g. -2.0 * 3.5 = -7.0m  (两车道路两侧有停车带时的 left_edge 目标位置)
```

`lane_width_m` 从 `relation_dsl["lane_width_class"]` 通过 `_lane_width_for_class()` 得到，不硬编码 `3.5`。测试用例中的期望值也必须通过同一函数推导，不能在测试里直接写死 `-6.5m`。

### 解决方案：米制 lateral 统一由 validation 产出

在 `validate_relation_layout()` 内新增局部计算：

```python
lane_width_class = relation_dsl.get("lane_width_class") or "standard"
lane_width_m = _lane_width_for_class(lane_width_class)
expected_lateral_band = _expected_lateral_band(ego_relation)
expected_lateral_m = expected_lateral_band * lane_width_m
lateral_error_m = lateral - expected_lateral_m
```

`entity_results` 同时保存 band 和 meters：

```json
{
  "expected_lateral_band": -2.0,
  "expected_lateral_m": -7.0,
  "actual_lateral_m": -3.0,
  "lateral_error_m": 4.0
}
```

repair 模块只使用 `expected_lateral_m` 与 `actual_lateral_m`，不再重复推导 band，也不再使用硬编码 lane width。

如果后续要把 `left_edge` 的目标从 `-2.0 * lane_width` 改为更精确的 “跨过 opposing lane + parking lane center”，也应先修改 validation 的 `expected_lateral_m` 产出，再让 repair 消费该值。repair 不应各自维护一套 lateral 公式。

---

## 6. Deterministic repair 执行器

### 当前问题

当前函数：

```python
_shift_spawn_payload_coordinates(scene_id, validation)
```

只处理 `validation["pairwise_results"]`，不会修：

- `left_edge` 是否落在 `left_lane` 上
- `left_parking_lane_count/opposing_lane_count` 导致的 edge offset 不足
- VLM 发现的 `lane_side_mismatch`
- `entity_results` 中的 ego-relative lateral fail

### 新函数

替换为：

```python
_apply_layout_repair_actions(scene_id, current_validation, report)
```

注意参数必须是当前轮最新的 `current_validation`。如果前面执行过 scene understanding revision，调用方必须传更新后的 validation，不能误传 `verify_and_repair_spawn_layout()` 入参里的初始 `validation`。

### 修复分工

Pairwise 修复：

- 输入：`current_validation["pairwise_results"]` 和 `repair_actions[type=pairwise_mismatch]`
- 必须使用 `reference_entity_id`
- 只修 ahead/behind/left/right 等实体间关系

Lane-side 修复：

- 输入：`current_validation["entity_results"]` 和 `repair_actions[type=lane_side_mismatch]`
- 根据 `expected_lateral_m - actual_lateral_m` 沿 anchor right vector 调整 actor location
- 例如 `left_edge` 当前在 `-3.0m`，目标是 `-6.5m`，则继续向 ego-left 移动 `3.5m`

Heading 修复：

- 输入：`repair_actions[type=heading_mismatch]`
- 结合 `heading_relation`、ego yaw、projected lane yaw 修正 actor yaw

Overlap 修复：

- 输入：`repair_actions[type=overlap]` 或内部 spawn spacing 检查
- 优先沿 lane forward/backward 小幅移动本轮新 actor
- 若与已有 CARLA actor 冲突，跳过新 actor，不移动已有 actor

Count/category 修复：

- `count_mismatch` 和 `category_mismatch` 不做几何修复
- 触发 `scene_understanding_revision`

### repair 优先级

如果 VLM 推荐 `scene_understanding`，但 `repair_actions` 中存在：

```text
lane_side_mismatch
pairwise_mismatch
heading_mismatch
overlap
```

则先执行 deterministic repair，再进入下一轮验证。

只有出现：

```text
count_mismatch
category_mismatch
```

才优先重跑 scene understanding。

---

## 7. 保留已有 CARLA 车辆与 overlap skip

### 当前问题

生成脚本字符串中有无条件调用：

```python
_autoscenario_clear_existing_dynamic_actors()
```

这会删除 `vehicle.*`，破坏当前 CARLA 世界中已有车辆。

### 新策略

默认保留已有车辆。旧清空行为只通过 env var 显式开启：

```bash
AUTOSCENARIO_CLEAR_EXISTING=1
```

生成脚本中的逻辑：

```python
if os.environ.get("AUTOSCENARIO_CLEAR_EXISTING") == "1":
    _autoscenario_clear_existing_dynamic_actors()
```

同时在生成出来的脚本中新增：

```python
def _autoscenario_collect_existing_vehicles():
    return list(world.get_actors().filter("vehicle.*"))

def _autoscenario_spawn_location_is_clear(location, min_distance_m):
    # check existing vehicles + already spawned actors
```

在 `_autoscenario_try_spawn_vehicle_actor()` 和 direct spawn 路径中，`world.try_spawn_actor()` 前先检查：

1. 与已有 CARLA `vehicle.*` 是否过近
2. 与本轮 `_AUTOSCENARIO_SPAWNED_ACTORS` 是否过近

冲突时跳过当前候选点或当前实体。不要删除已有 actor，不要移动已有 actor。

### 待补充：existing vehicles 预收集位置

`_autoscenario_collect_existing_vehicles()` 不能在每次 `_autoscenario_try_spawn_vehicle_actor()` 内部调用——场景里 10 个 actor 就会发 10 次 `world.get_actors()` API 请求，开销成倍增加。

正确做法是在 spawn 循环**开始前**调用一次，结果存入模块级变量：

```python
_autoscenario_existing_vehicles = _autoscenario_collect_existing_vehicles()
# 然后开始 spawn 循环
for entity in spawn_payload.get('entities', []):
    ...
```

`_autoscenario_spawn_location_is_clear()` 检查时直接用这个预收集的列表，不重复调用 API。这行初始化也是 `_build_spawn_payload_loop()` 生成代码的一部分。

### 解决方案：已有车辆缓存与本轮车辆分开维护

生成脚本中使用两个集合：

```python
_AUTOSCENARIO_EXISTING_VEHICLES = []
_AUTOSCENARIO_SPAWNED_ACTORS = []
```

在 spawn payload 循环前执行一次：

```python
_AUTOSCENARIO_EXISTING_VEHICLES = _autoscenario_collect_existing_vehicles()
```

`_autoscenario_spawn_location_is_clear(location, min_distance_m)` 只读取这两个集合：

```python
for actor in list(_AUTOSCENARIO_EXISTING_VEHICLES) + list(_AUTOSCENARIO_SPAWNED_ACTORS):
    ...
```

成功 spawn 的新 actor 继续追加到 `_AUTOSCENARIO_SPAWNED_ACTORS`。这样后续实体会避让本轮已经生成的 actor，同时不会重复调用 `world.get_actors()`。

如果设置 `AUTOSCENARIO_CLEAR_EXISTING=1`，应先执行旧清场逻辑，再收集已有车辆；此时 `_AUTOSCENARIO_EXISTING_VEHICLES` 通常为空。

---

## 8. Verify-repair 主循环更新

`verify_and_repair_spawn_layout()` 的每轮逻辑改成：

```text
1. _capture_layout_images()
2. _verify_scene_round(layout_image_path, capture_mode)
3. if passed: stop
4. classify repair_actions
5. if geometry actions exist:
       _apply_layout_repair_actions(scene_id, current_validation, report)
       regenerate final scene script
   elif count/category mismatch exists:
       revise scene_understanding
       rerun structured tail without remap
       update current_validation
   else:
       fallback to existing recommended_stage behavior
6. next round
```

`current_validation` 必须随每次 rerun structured tail 更新：

```python
current_relation_dsl, _refined, current_validation, current_final_path = ...
```

随后 repair 必须使用这个最新 `current_validation`。

### 待补充：repair 分支歧义

**geometry 和 count/category 同时存在时**，`elif` 结构会让 count/category 在本轮被静默跳过。需要明确这是有意的：几何修复优先，count/category 延迟到下一轮验证再处理。建议在注释里写明：

```text
# count/category mismatch present but geometry actions take priority this round;
# SU revision deferred to next verify round if geometry repair does not fix them.
```

**`else` 分支的降级行为**需要明确。当 `repair_actions` 为空但 `current_validation` 仍有 pairwise fail 时，不能直接跳过。建议改为：

```text
if geometry actions exist:
    _apply_layout_repair_actions(scene_id, current_validation, report)
    # count/category 延迟到下一轮，不在本轮处理
elif count/category mismatch exists:
    revise scene_understanding + rerun structured tail
    update current_validation
else:
    # repair_actions 为空，降级到 validation pairwise_results 驱动的 deterministic shift
    if current_validation has pairwise fail:
        _apply_layout_repair_actions(scene_id, current_validation, report={})
```

### 解决方案：主循环分支固定为四类

每轮 verify 后先分类：

```python
geometry_actions = lane_side + pairwise + heading + overlap
semantic_actions = count + category
validation_has_pairwise_fail = any(pair.status == "fail")
validation_has_entity_lateral_fail = any(entity.lateral_check is False)
```

分支顺序固定如下：

```text
1. passed -> stop
2. geometry_actions 非空 -> deterministic repair
3. semantic_actions 非空 -> scene_understanding_revision
4. validation_has_entity_lateral_fail 或 validation_has_pairwise_fail -> validation-driven deterministic repair
5. 否则按 recommended_stage 兜底；仍无可执行动作则记录 no_repair_action 并停止或进入 max_rounds
```

如果 geometry 和 semantic 同时存在，执行第 2 步，semantic 留到下一轮验证后再决定。这样避免同一轮既改语义又改坐标，导致无法判断哪一步造成新布局变化。

`_apply_layout_repair_actions(scene_id, current_validation, report)` 必须接受空 report：

```python
report = report or {"repair_actions": []}
```

这样 validation-driven fallback 可以复用同一个 repair 执行器。

这样避免 `repair_actions` 为空时整轮 repair 变成 no-op，保留了旧 pairwise shift 的兜底。

---

## 9. 测试计划

### SceneVerificationAgent

- `DEFAULT_REPORT` 包含 `repair_actions`。
- `normalize_report()`：
  - `count_mismatch` 无 `entity_id` 不被过滤。
  - `pairwise_mismatch` 缺 `reference_entity_id` 会被过滤。
  - 非 list 的 `repair_actions` 归一化为空 list。
- ego-view prompt 包含 forward FOV 盲区约束。
- BEV fallback prompt 仍包含 BEV ego-frame 推断说明。

### Capture and generated script

- 生成脚本包含 `AUTOSCENARIO_EGO_VIEW_OUTPUT` 分支。
- `_capture_layout_images()` 同时设置 ego-view 和 BEV env var。
- ego-view 成功时 `capture_mode == "ego_view"`。
- ego-view 缺失但 BEV 成功时 `capture_mode == "bev_fallback"`。

### Validation

- `validate_relation_layout()` 输出：
  - `lane_context`
  - `lane_width_m`
  - `actual_lateral_m`
  - `expected_lateral_band`
  - `expected_lateral_m`
  - `lateral_error_m`
- `expected_lateral_m` 从 `lane_width_m` 动态计算，不硬编码 `-6.5m`。

### Repair

- `left_edge` 落在相邻行车道时触发 lane-side repair，并移动到 `expected_lateral_m`。
- `right_edge` 当前成功案例不被错误扩大。
- `pairwise_mismatch` 必须带 `reference_entity_id` 才执行 pairwise repair。
- `count_mismatch/category_mismatch` 触发 `scene_understanding_revision`，不执行坐标修。
- SU revision 后下一轮使用更新后的 `current_validation`。

### Existing vehicle safety

- 生成脚本默认不无条件清空 `vehicle.*`。
- 设置 `AUTOSCENARIO_CLEAR_EXISTING=1` 时恢复旧清空行为。
- spawn 前冲突检测存在，冲突时跳过新 actor。

### Regression

运行：

```bash
python -m pytest tests/test_generation_pipeline_helpers.py tests/test_auto_generate_all_vlm.py
```

验收当前问题：

- 左侧 `left_edge` 停车 actor 不再落到 ego 左侧行车道。
- 右侧停车 actor 仍保持在右侧停车带。
- verify-repair summary 中记录 `capture_mode`、`repair_actions` 和实际执行的 deterministic repair。

---

## 10. 实施优先级

| 优先级 | 任务 | 文件 |
| --- | --- | --- |
| P0 | `DEFAULT_REPORT` / `normalize_report()` 支持 `repair_actions` | `agents/scene_verification_agent.py` |
| P0 | ego-view / BEV fallback prompt 和 `capture_mode` 接口 | `agents/scene_verification_agent.py`, `experiments/auto_generate_all_vlm.py` |
| P0 | `validate_relation_layout()` 输出 lane context 与米制 lateral 诊断 | `tools/structured_pipeline.py` |
| P1 | 生成脚本支持 `AUTOSCENARIO_EGO_VIEW_OUTPUT` | `agents/existing_world_scenario_generator.py` |
| P1 | `_capture_layout_images()` 替代 spawn layout 阶段 BEV-only capture | `experiments/auto_generate_all_vlm.py` |
| P1 | `_apply_layout_repair_actions()` 实现 lane-side + pairwise + heading 修复 | `experiments/auto_generate_all_vlm.py` |
| P1 | 生成脚本默认保留已有车辆，并做 spawn 前 overlap skip | `agents/existing_world_scenario_generator.py` |
| P2 | 测试覆盖 prompt、capture、validation、repair、existing vehicle safety | `tests/` |

---

## 核心原则

- Actor layout verification 优先使用 ego-view；BEV 只作为 fallback。
- Machine repair 不解析自然语言 `repair_hints`。
- `repair_actions` 只负责表达诊断；具体几何量以 `current_validation` 为准。
- Pairwise 修复放在 pairwise repair 里，lane-side 修复放在 lane-side repair 里。
- 语义错误修 scene understanding，几何错误修 spawn payload。
- 默认不破坏 CARLA 世界中已有车辆。

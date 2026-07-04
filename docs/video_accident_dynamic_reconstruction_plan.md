# 事故视频到 CARLA 动态重建方案

## Summary

目标是把“单帧静态重建 + 视频事故理解 + 可执行轨迹控制”串成一条动态事故复现流水线。推荐采用：**自动选择事故前起始帧作为静态锚点，最后事故帧作为软约束验证锚点，动态过程用结构化时序轨迹 DSL 驱动 CARLA 控制器**。

这比直接让 `video_interpreter.py` 输出自由文本更稳，因为 CARLA 可执行部分必须绑定到已经生成的 `actors.json`、CARLA map waypoint 和具体 actor id。

## Key Changes

- 新增视频预处理阶段：
  - 从事故视频抽帧，保存帧图像和元信息：`frame_index`、`timestamp_s`、`fps`、`is_candidate_start`、`is_candidate_collision/end`。
  - 默认自动选起始帧：事故尚未发生、ego 和关键参与者清晰、道路结构可见的一帧。
  - 默认自动选终点帧：碰撞或冲突结果最清楚的一帧，只作为软约束。

- 静态场景仍复用 `auto_generate_all_vlm.py` 的主链路：
  - 输入为起始帧图片。
  - 输出起始时刻的 CARLA 静态场景：`scene_understanding`、`relation_dsl`、`actors.json`、match report、spawn script。
  - 起始帧是唯一的 spawn anchor，避免中途/终局事故帧导致车辆已经重叠、遮挡或姿态不可恢复。

- 替换/升级当前 `video_interpreter.py` 的输出形态：
  - 不再只输出自由文本事故描述。
  - 新增结构化视频事故理解 JSON，包含：
    - 事故类型和关键过程摘要。
    - 每个关键帧中可见道路参与者的相对位置、动作、可见性和不确定性。
    - 关键事件序列，例如减速、变道、切入、横穿、追尾、碰撞。
    - 参与者到起始静态场景 actor id 的映射建议。

- 新增 `video-trajectory-dsl-v1`：
  - 以 `actors.json` 中的真实 actor id 为唯一执行对象。
  - 每个 actor 输出一组按时间排序的控制片段，而不是直接生成 CARLA Python。
  - 控制片段使用可执行动作：`lane_follow_speed`、`brake`、`stop`、`steer_offset`、`lane_change`、`hold_position`、`target_waypoint`。
  - 每个片段包含 `start_s`、`end_s`、`target_speed_mps`、`lane_relation/waypoint_constraint`、`confidence`、`source_frames`。

- 扩展最终 CARLA 脚本生成器：
  - 在现有 `ExistingWorldScenarioGenerator.build_dsl_risk_scene_script` 思路上新增视频轨迹执行 loop。
  - ego 和 NPC 都从起始帧 spawn。
  - 默认沿 CARLA waypoint lane-follow，LLM 只决定速度、触发时间、车道/横向动作，不直接写 Python。
  - 背景车辆若没有轨迹 DSL，则根据起始帧 `motion_state` 决定停车、低速巡航或 Traffic Manager autopilot。

- 最后事故帧处理策略：
  - 第一版使用“软约束”。
  - 终点帧不用于重新 spawn 一个最终静态场景。
  - 它用于检查仿真末端是否满足：关键 actor 相对顺序、是否碰撞、冲突位置、最终朝向/停止状态是否接近视频。
  - 如果偏差过大，再反馈给 LLM 修正 `video-trajectory-dsl-v1` 的时间、速度、制动和横向动作。

## Public Interfaces

- 新增入口脚本建议命名：
  - `experiments/auto_generate_all_video_reconstruction.py`

- 主要输入：
  - `--video-path`
  - `--output-folder`
  - `--start-frame auto|<frame_index>|<timestamp_s>`
  - `--end-frame auto|<frame_index>|<timestamp_s>`
  - `--frame-sample-rate`
  - `--carla-map optional`

- 主要输出：
  - `frames/`: 抽帧图片和 `frames.json`
  - `<scene_id>_start_frame.jpg`
  - `<scene_id>_end_anchor_frame.jpg`
  - `<scene_id>_actors.json`
  - `<scene_id>_video_understanding.json`
  - `<scene_id>_video_trajectory_dsl.json`
  - `<scene_id>_dynamic_reconstruction.py`
  - `<scene_id>_video_reconstruction_summary.json`

- `video-trajectory-dsl-v1` 必须经过 validator：
  - actor id 必须存在于 `actors.json`
  - 时间段不能为负，且必须在视频时长范围内
  - 动作类型必须在执行器支持列表中
  - 不允许 LLM 输出任意 CARLA Python 代码
  - 允许 `unknown` 和低置信度，但执行时必须有保守 fallback

## Test Plan

- 单元测试：
  - 视频抽帧保留正确 `frame_index/timestamp_s`。
  - 起始帧/终点帧自动选择结果结构合法。
  - `video-trajectory-dsl-v1` validator 拒绝未知 actor、未知 action、非法时间段。
  - DSL 执行器能把 `lane_follow_speed`、`brake`、`stop`、`lane_change/steer_offset` 转成 CARLA 控制逻辑。

- 集成测试：
  - 用一个短视频或 mock frames 跑完整链路，确认生成 `actors.json`、视频理解 JSON、轨迹 DSL 和最终 `.py`。
  - 对现有 `results/auto_result_20260618_192858/accidents_1` 类似场景，验证新的轨迹 DSL 能表达“前车/摩托低速行驶后停止，ego 追尾”。
  - 最终脚本 `python -m py_compile` 通过。
  - 若 CARLA 可用，运行后记录 metrics：碰撞、最小距离、TTC、关键事件时间。

- 验证指标：
  - 起始帧静态布局与原始起始帧相似。
  - 事故参与者与视频关键帧中的相对前后/左右关系一致。
  - 事故机制一致，例如追尾、切入、横穿、侧擦。
  - 终点软锚点检查通过或给出可修复偏差报告。

## Assumptions

- 起始帧默认自动选择，但允许用户用帧号或时间戳覆盖。
- 终点事故帧默认作为软约束，不作为第二个强制静态 spawn 场景。
- 第一版不做逐帧物体检测和多目标跟踪；用 VLM 结构化理解关键帧，再由 LLM 生成时序控制 DSL。
- 第一版优先保证 CARLA 可执行和事故机制可复现，精确像素级轨迹对齐作为后续增强。
- 当前 `risk-dsl-v1` 的事件表达能力不足以“尽可能还原视频”，因此新增视频轨迹 DSL，而不是直接复用原 DSL。

## 审阅意见（Claude, 2026-06-24）

总体判断：方向正确，“起始帧 spawn + 时序控制 DSL + 终点软约束”是比让 VLM 直接吐 CARLA Python 更稳的路线，且与现有两层架构兼容。但计划把若干**最难的问题当成了普通条目**，并对“视频时间轴 ↔ CARLA 仿真时间轴”这一核心标定问题只字未提。以下按严重度排列。已对照 `tools/risk_dsl.py`、`agents/existing_world_scenario_generator.py`、`tools/pure_llm_risk_runner.py`、`agents/video_interpreter.py` 核实。

### P0 — 必须先解决，否则方案不成立

1. **视频时间轴 ↔ CARLA 时间轴没有标定（最大隐患）。**
   `video-trajectory-dsl-v1` 用绝对 `start_s/end_s`（视频秒）驱动 CARLA。但起始帧静态重建是把所有 actor **投影到起始帧 lane waypoint** 上，CARLA 里 actor 之间的初始间距、绝对位置 ≠ 视频真实世界的距离/速度。把视频里“3 秒后追尾”原样搬进 CARLA，会因初始间距不同而**碰不上或提前碰**。
   - 建议：关键事件锚定改用**事件/几何触发**（复用现有 `distance_to_ego_below`、`time_elapsed_above`），而不是依赖绝对视频时间；绝对时间只作段内 fallback。
   - `video_interpreter.py` 里已有 depth + 光流测距（`calculate_distance_from_optical_flow`）可用来估每个参与者的真实速度，进而做一次“视频距离 → CARLA 初始间距”的缩放标定 —— 计划完全没提，应纳入。
   - 需要明确：标定单位是“复现事故机制”还是“复现绝对时刻”。Assumptions 已说优先机制，但 DSL 又用绝对秒，两者要对齐。

2. **与现有 `risk-dsl-v1` 执行器的重叠与协同未说明，存在重写一整套并行执行器的风险。**
   `build_dsl_risk_scene_script` 已经实现：waypoint lane-follow 路由（`_autoscenario_sample_forward_route`）、纵向速度控制（`_autoscenario_apply_speed_control` / `_apply_target_velocity`）、`hold_vehicle_stationary`、`EgoController` / `NpcRiskController`，以及 trigger（`immediate` / `time_elapsed_above` / `distance_to_ego_below`）+ action（`brake` / `set_speed` / `accelerate` / `steer` / `cross` / `stop`）。新 DSL 的 `lane_follow_speed` / `brake` / `stop` / `steer_offset` 基本能**复用**这套。
   - 建议：把 `video-trajectory-dsl-v1` 定位为 `risk-dsl-v1` 的“**每 actor 多段 timeline**”扩展（在现有 action/trigger 词表上加时间分段），而不是全新 DSL + 全新执行器。否则两套执行器并行维护、行为漂移。计划应逐条标注：哪些动作复用、哪些新增。

3. **`lane_change` 和 `target_waypoint` 是全场最难的两个动作，却被当普通条目列出。**
   当前执行器只有纵向速度控制 + 简易 `steer`，**没有横向 lane change 控制器，也没有路径跟踪器**。CARLA 原生没有可控的横向变道；用 Traffic Manager 的 `force_lane_change` 方向/时机不可控，`target_waypoint` 需要 pure-pursuit/Stanley 之类的跟踪器。
   - 建议：第一版把 `lane_change` / `target_waypoint` 明确降级为“后续增强”，MVP 只保证纵向动作（`lane_follow_speed` + `brake` + `stop` + `hold_position`）+ 触发时机，事故机制聚焦**追尾 / 被切入后减速 / 前车急停**这类纵向主导的类型。横向切入用“切入车提前 spawn 在目标车道 + 纵向接近”近似，而非真做变道。

### P1 — 影响可用性与闭环收敛

4. **终点“软约束 → 反馈 LLM 修正”闭环缺少可计算偏差度量与收敛准则。**
   计划说“偏差过大就反馈 LLM 修正时间/速度/制动”，但没定义：偏差怎么算（CARLA 末态如何与一张终点帧图像比？像素？还是符号化的相对顺序/碰撞布尔？）、最多迭代几次、何时判“不可达”放弃。而且要对比，必须先把终点帧也做一次结构化理解（碰撞对、相对前后/左右顺序、末态停止与否）—— 这等于第二次 VLM 理解，但 Public Interfaces 里没有它的产物 schema。
   - 建议：终点检查只用**离散符号指标**（是否碰撞 / 碰撞 actor 对 / 末态相对顺序 / 是否停止），做布尔或分类比对；修正闭环设硬上限（≤2 次），并允许“机制达成但时空不精确”即判通过。新增产物 `<scene_id>_end_anchor_understanding.json` 并定义其 schema。

5. **起始帧自动选择是 VLM 单点判断，失败成本高且无回退。**
   “事故尚未发生 + 参与者清晰 + 道路可见”这三条常互相冲突（最清晰的帧往往已接近事故）。计划给了手动覆盖，但没有自动选失败时的诊断或候选集。
   - 建议：自动选 **top-k 候选帧 + 质量分**（含“事故未发生”置信度），写进 `frames.json` 供下游/人工挑选，而非单帧硬选。

6. **起始帧外不可见的关键参与者无法 spawn，但事故机制可能正依赖它。**
   起始帧是唯一 spawn anchor（合理），但事故中途“横穿冲出的人/车”在起始帧不可见 → 无法 spawn → 该类事故无法复现。
   - 建议：显式声明这是第一版限制；或允许从起始帧**外推一个隐含 actor**（在画面外的合理 lane 位置 spawn，再用轨迹 DSL 让它驶入）。同时映射阶段要定义“映射不上 / 低置信度”的处理（validator 提了 `unknown`，但映射阶段没说兜底）。

5和6暂时不考虑，输入里先加入起始帧参数和结束帧参数，由用户提供输入，，目前先不考虑让agent去自行确定起始帧和结束帧

### P2 — 细节与一致性

7. **背景车 fallback 用 Traffic Manager autopilot 与“可复现”目标矛盾。**
   TM 行为随机、跨版本不稳定，可能让背景车乱入事故区。建议事故相关区域禁用 TM；背景车默认 `hold` 或低速直行 lane-follow，autopilot 只用于远处无关背景。

8. **长视频把所有帧 base64 塞进单次请求会 token 爆炸。**
   现 `video_interpreter.video_encode` 即是“全帧进一次请求”。新链路需明确采样上限 / 分段理解策略，并对“关键帧”和“全帧”分级。

9. **入口与参数应复用 Layer1/Layer2，避免第三条平行链路。**
   建议 `auto_generate_all_video_reconstruction.py` 内部直接调用 `auto_generate_all_vlm.py` 的静态主链 + Layer2 runner，而不是新写一条端到端流水线，减少分叉与维护面。
   - `--start-frame auto|<frame_index>|<timestamp_s>` 三义同字段：纯整数到底当帧号还是秒需消歧（例如 `120` vs `120s`），否则解析有歧义。

10. **术语对齐（非阻塞）。** 计划文字用 “actors.json / spawn_payload”，实际 schema 顶层是 `entities[].id`（`tools/risk_dsl.py:_collect_actor_ids` 读的就是 `entities`）。新 validator 读取路径保持一致即可，文档可注明二者等价。

### 建议的 MVP 收敛范围

只做：起始帧 top-k 自动选 + 手动覆盖 → 复用 Layer1 静态重建 → 结构化视频理解 JSON（含每参与者真实速度估计）→ `risk-dsl-v1` 的“多段 timeline”扩展（仅纵向动作 + 事件触发）→ 复用现有执行器生成 `.py` → 终点**符号化**检查（碰撞/顺序/停止）+ ≤2 次修正。先把“追尾/前车急停/被切入后减速”跑通并 `py_compile` 通过，再迭代横向变道与绝对时间对齐。

## 实现状态（Claude, 2026-06-24）

已按上述 MVP 范围实现并通过测试（新增 22 个单测，全量 258 个测试 OK；lower 后的脚本经真实生成器 `py_compile` 通过）。采用的关键决策与商定一致：起止帧由用户用**秒**提供（不自动选帧，P1#5/#6 暂不做）；只实现纵向 + 软横向动作，`lane_change`/`target_waypoint` 被 validator 显式拒绝并给修复提示；**不新增执行器**——轨迹 DSL lower 成 `risk-dsl-v1` 多段 `time_elapsed_above` 事件复用 `DslEventController`。

新增文件：
- `tools/video_frames.py` —— 按秒抽取起止/上下文帧 + `frames.json`（cv2 注入式，可单测）。
- `tools/video_trajectory_dsl.py` —— `video-trajectory-dsl-v1` 的 schema/validator + `lower_to_risk_dsl`。
- `agents/video_accident_interpreter.py` —— 结构化视频事故理解（含 `end_state`，供符号化终点检查复用）。
- `agents/llm_video_trajectory_generator.py` —— 视频理解 → 轨迹 DSL（带 schema 修复回路）。
- `tools/video_reconstruction_runner.py` —— 编排器 + `evaluate_end_anchor` 符号化软约束检查。
- `experiments/auto_generate_all_video_reconstruction.py` —— CLI（`--start-frame`/`--end-frame` 单位为秒）。
- `tests/test_video_frames.py`、`tests/test_video_trajectory_dsl.py`、`tests/test_video_reconstruction_runner.py`。

未做（留待后续）：起止帧自动选择（信号/检测）、横向真变道与 `target_waypoint`、视频↔CARLA 绝对时间标定（P0#1，当前用事件/分段时间近似）、运行 CARLA 后回灌 metrics 触发终点修复回路（`evaluate_end_anchor` 已就绪，目前无 metrics 时返回 `pending`）。静态重建沿用 Layer-2 既有约定：**复用** `{scene_id}_actors.json` / `_match.json`，需先对起始帧跑 Layer-1。

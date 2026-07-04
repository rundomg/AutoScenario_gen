# 设计方案:按「所属道路」放置 actor(路口分支 + 对向 + 转向意图)

状态:设计草案(未实现)。对应用户反馈中的「问题 2」——T 字路口侧路上的车被横在 ego 车道里。
关联已落地修复:对向车 heading 兜底(问题 3)、ego 车道 min 重排(问题 1)。

## 1. 问题

当前 Layer-1 重建是**纯 ego 车道相对**的(`tools/structured_pipeline.py`):

- 横向 = `lane_index_relation` × 车道宽,相对 ego 锚点车道左右偏移;
- 纵向 = 沿 ego 锚点车道的距离;
- 朝向 = `heading_relation`:`same_direction`→ego 朝向,`opposite_direction`→+180°,`crossing`→**+90°**。

后果:

1. **侧路车被横在 ego 车道**:T 字路口侧支路上等待汇入的车(`heading=crossing`, `lane_index=1`)只是被 +90° 旋转并塞进 ego 右车道,而不是放到**侧支路的行车道**上、顺着侧路朝向主路。
2. **侧路车污染 ego 车道判定**:VLM 把侧路车编码成 ego 同向 `lane_index +1/+2`,把 ego 的 `ego_lane_from_right` 撑大(问题 1 的根因之一)。
3. **对向/转向表达不了**:对向左转车没有「转向意图」概念,只能摆一辆直朝的对向车。

根因:actor 没有「我属于哪条路」的语义,所有东西都被迫挤进 ego 这一条车道的相对坐标系。

## 2. 目标

给每个 actor 一个**所属道路(road membership)**,并据此:

- 选中该道路在匹配地图上的**真实车道**;
- 让 actor 顺着**那条道路**的方向摆放(进入路口的横向车应朝向路口/主路,而非垂直于 ego);
- 支持**转向意图**(直行/左转/右转),用于选择转向车道与朝向角。

非目标:动态轨迹(仍是静态重建);非路口的任意 landmark 几何对齐(无地图坐标)。

## 3. SU 契约扩展(`agents/scene_understanding_interpreter.py`)

每个 `traffic_subjects` 项新增可选字段:

```jsonc
{
  "road_membership": "ego_carriageway",   // 见下枚举
  "turn_intent": "through",               // through | left | right | unknown
  "branch_side": "right"                  // 仅 cross_branch_* 时:left|right|ahead
}
```

`road_membership` 枚举:

| 值 | 含义 | 放置目标 |
|---|---|---|
| `ego_carriageway` | ego 同向车道(默认) | 现有 ego 相对放置 |
| `opposing_carriageway` | 对向车道 | 现有 `project_to_opposing_lane` |
| `cross_branch` | 路口分支道(侧路)上的车 | 新增:分支道车道 |
| `roadside_off_road` | 路边停车/店前(非行车道) | 现有 parked 处理 / 可选过滤 |

向后兼容:字段缺省时,从 `heading_relation_to_ego` 推断(`crossing`→`cross_branch`,`opposite_direction`→`opposing_carriageway`,否则 `ego_carriageway`),不破坏旧数据。

提示词要点:明确「侧路/驶出口/横穿车」用 `cross_branch` 且**不要**计入 ego 同向车道(与问题 1 prompt 修复一致);对向左转车 `road_membership=opposing_carriageway` + `turn_intent=left`。

## 4. 管线设计

### 4.1 路口分支拓扑(需要 CARLA)

新增 `tools/junction_topology.py`(或并入 `_sample_dense_local_waypoints`):

- 输入:匹配锚点 location、朝向。
- 用 `world_map.get_waypoint(...)` 沿前向走到 `is_junction` 的 waypoint,取其 `get_junction()`;
- `junction.get_waypoints(carla.LaneType.Driving)` 枚举进出对,按**进入方向相对 ego 的角度**聚类成 `ahead / left / right` 分支;
- 每个分支输出一条代表性「入口车道」(road_id, lane_id, 朝向路口的 start/end),写入 `spawn_context["junction_branches"]`。

缓存模式(无在线 CARLA):无法得到分支几何 → `cross_branch` actor **降级**为现有 `crossing` 行为,但**不计入 ego 车道**(语义仍生效),并在 artifact 里标注 `branch_placement="degraded_no_carla"`。

### 4.2 放置解析(`_select_semantic_lane` 扩展)

在 `_select_semantic_lane` 增加按 `road_membership` 的分派:

- `cross_branch` + 有 `junction_branches`:按 `branch_side` 选对应分支入口车道;朝向 = 该车道**指向路口**的方向(让等待汇入的车面向主路),而非 ego+90°。
- `opposing_carriageway`:走现有对向逻辑;若 `turn_intent=left`,优先选对向的**左转车道**(最靠中心线的对向车道),朝向可加一个朝路口中心的偏角(如 +20°)表达「正在左转」。
- 其余:现有行为。

朝向计算集中到一个 `resolve_actor_yaw(membership, turn_intent, lane_yaw, ego_yaw)`,替换分散的 `+90/+180`。

### 4.3 ego 车道(与问题 1 协同)

`cross_branch`/`opposing_carriageway` 的 actor 已通过 heading 在 `_infer_ego_lane_offset` 被排除;加入 `road_membership` 后用它做更稳的排除条件(不依赖 heading 标注准确)。

## 5. 新增/变更产物

- `spawn_context.junction_branches`(运行期);
- `{scene_id}_actors.json` 每个 actor 增加 `road_membership` / `branch_placement` 便于审查;
- verify 增加一条软检查:`cross_branch` actor 是否落在非 ego road_id 上(落在 ego road 视为放置失败)。

## 6. 测试策略

纯函数优先(无 CARLA):

- `resolve_actor_yaw`:各 membership/turn_intent 的朝向;
- 分支选择:给定 mock `junction_branches` + `branch_side`,选对入口车道;
- 降级路径:无 `junction_branches` 时 `cross_branch` 回退且不污染 ego 车道;
- 回归:`road_membership` 缺省时行为与现状一致。

CARLA 相关的分支拓扑提取做集成测试 / 手动验证。

## 7. 分期

1. **P1**:SU 契约 + `resolve_actor_yaw` + `_infer_ego_lane_offset` 用 `road_membership` 排除侧路(纯函数,可测,立即缓解问题 1/2 的「污染」)。
2. **P2**:`junction_topology` 分支提取 + `cross_branch` 真实分支道放置(需 CARLA)。
3. **P3**:`turn_intent` 左转车道选择 + 朝向偏角。

## 8. 风险

- 分支聚类对复杂/多支路口可能不稳 → 先只支持 `ahead/left/right` 三分支,其余降级。
- VLM 误标 `road_membership` → 保留从 heading 的推断兜底 + verify 软检查兜底。
- 缓存模式无分支几何 → 明确降级,不追求该模式下的分支道精确放置。

# 地图匹配重构计划：结构级匹配 + 参考系解耦 + 朝向保真

> 状态：设计草案（2026-06-27）
> 关联代码：`tools/scene_map_matcher.py`、`tools/structured_pipeline.py`、`agents/scene_understanding_interpreter.py`、`tools/cache_map_topology.py`、`agents/map_match_reranker.py`

## 1. 背景与动机

当前 Layer-1 的地图匹配把整个场景压成一个**单点截面的标量/布尔签名**（`build_road_topology_signature`，`scene_map_matcher.py:473`），在 CARLA 地图上撒 waypoint 逐点打分（`_score_topology_candidate_features:3288`），最后**直接把匹配到的那个 waypoint 当作 ego 锚点**——下游用这条单一 anchor lane 的 `start` + 一组固定 forward/right 基向量，把所有 actor 摊在一条直线坐标轴上（`_build_entity_coordinates`，`structured_pipeline.py:1502`，朝向 `yaw = anchor_yaw (±180/±90)`，`:1524`）。

### 已识别的核心缺陷

1. **签名丢结构**：只有"局部有几条路/几条车道 + 几个布尔"，没有道路连通关系，也没有沿路顺序与距离。
2. **大图退化**：Town12 上"双向 2 车道城市直路"有几百个等价点，签名无法区分 → 锚点近乎随机选，下游几何全继承这个噪声点。
3. **road_id/lane_id 只当局部计数用**，没用到它们承载的图结构：无连通性匹配、不按 road_id 折叠候选、不用 `s` 做纵向对齐。
4. **单轴坐标系表达不了多腿路口**：十字/T 路口的横穿、对向、转弯车被投影到 ego 一条车道的轴上 → 落错车道/朝向错。
5. **朝向靠几何外推**（`anchor_yaw ± k`）：锚轴与真实车道方向不一致时（弯道、路口、对向带）直接逆行。
6. **逆行语义被抹掉**：`heading_relation_to_ego` 把"对向车道合法对向"与"我车道里逆行"混为一谈；一致性校验会把真实逆行（尤其摩托车）"掰正"，毁掉安全关键信号。

## 2. 目标 / 非目标

**目标**
- 把"匹配"与"摆放"解耦：匹配只锁定**结构**，ego 和所有车辆的位置/朝向在**参考系内重算**。
- 提升匹配 specificity：以 road/junction 为单位建候选，利用 OpenDRIVE 连通图。
- 朝向保真：朝向取自吸附到的真实车道，且**完整保留真实逆行**。

**非目标**
- 不改 Layer-2 风险生成。
- 不要求 VLM 输出绝对坐标。
- 不追求像素级精确，匹配只需"选对结构类型 + 对的腿/车道"。

## 3. 总体架构：三层解耦

```
① 匹配（结构级）   找结构最像的"路段→路口"链，按 road_id/junction_id 折叠候选
                  输出结构对象（matched_structure），不是单个 waypoint
② 建参考系         路口场景 = junction 中心 + 枚举各腿（朝向/lane）
                  直路/弯道 = lane segment + s 纵向坐标
③ 摆放（重算）     用 VLM 有序前方特征 + actor 相对方向，在参考系内重算
                  每辆车：选车道带 → 吸附真实 lane → 定朝向（含逆行）→ 一致性校验
```

关键：**匹配点不再直接当 ego 锚点**，只用来锁定结构。匹配精度要求从"选对那个点"降到"选对结构类型"，与匹配本身低 specificity 的现实相符。

## 4. 组件设计

### 4.1 匹配器：route-signature + 候选折叠

- **VLM 侧**新增"沿 ego 前进方向的有序前方特征序列"（见 4.4），把单点属性升级为路线签名。
- **CARLA 侧**：
  - 用 `map.get_topology()` / junction 连通性，把候选**按 road_id 折叠成路段候选、按 junction_id 折叠成路口候选**（替代现在"30m 球里数 road_id"）。
  - 沿 successor / junction 连通关系 walk 出"路段→路口"链，与 VLM 有序特征序列做序列比对（特征类型 + side + 粗距离）。
  - 曲率补**方向 + 量级（半径/锐度）**，替代现在只有 `is_curve + curve_score`。
- **输出结构对象**（替换 `_build_topology_match_record:2272` 的单点记录）：

```jsonc
matched_structure = {
  "kind": "junction" | "road_segment",
  "junction_id": 123,                       // kind=junction
  "center": {"x","y","z"},
  "legs": [                                  // 枚举各腿
    {"heading": deg, "entry_wp": {...}, "exit_wp": {...}, "lane_ids": [...]}
  ],
  "road_id": 45, "lane_segment": {"lane_id": -1, "s_range": [s0, s1]},  // kind=road_segment
  "curvature": {"direction": "left|right|straight", "radius_m": 80.0}
}
```

### 4.2 参考系构建（scene-dependent 锚点）

| 场景 | 参考系 | ego 定位 |
|---|---|---|
| 路口（十字/T） | junction 中心 + 枚举的腿 | 沿进近腿回退 `ego_to_junction_distance_m`，吸附进近腿 lane |
| 直路/弯道 | lane segment + `s` 坐标 | 沿 `s` 摆放，前方 `forward_reference` 对齐；弯道 forward 随 s 变 |

附带好处：路口中心对"具体选了哪个 waypoint"不敏感（周围 waypoint 都折叠到同一中心），与候选折叠互相加强；即使匹配质量一般，只要结构类型对，布局仍可用。

### 4.3 摆放与朝向（核心：杜绝假逆行 + 保真真逆行）

**统一原则：朝向永不几何外推，一律读"吸附到的那条真实车道"的 yaw。** 每条可行驶 lane 自带唯一合法方向（waypoint forward = 车道行车方向）。

把"占哪条车道"与"朝哪个方向"彻底拆开：

```
占哪条车道带  = lane_index_relation（横向位置，VLM 已有：0=ego道,+1右,-1左）
base_yaw     = 吸附车道的合法 forward yaw
yaw          = base_yaw + 180   if flow_compliance == "wrong_way"
               base_yaw         otherwise
```

VLM 的 `heading_relation_to_ego` / `turn_intent` **不直接当 yaw**，只用来选哪条 lane / 哪条腿。

**一致性校验按 `flow_compliance` 分流**（守"符合 VLM 观测"，不是"符合交规"）：
- `legal`：校验顺向（same→Δyaw<90，opposite→Δyaw>90，crossing→≈90）→ 抓**管线 bug 导致的意外逆行**，错则重选同向车道，仍不行丢弃并记日志。
- `wrong_way`：校验它**确实逆向** → 防止真实逆行被"掰正"。
- `unknown`：默认合法，**不凭空制造逆行**，但绝不抹掉已明确标注的逆行。

### 4.4 VLM 输出扩展（`scene_understanding_interpreter.py`）

`road_network` 新增 / 强化：
- **`forward_features`（有序）**：`[{type: crosswalk|t_junction|cross_intersection|traffic_light|..., side: left|right|ahead, distance_m: 粗估}, ...]` —— 沿 ego 前进方向的特征序列，喂给 route-signature。
- **junction 几何**：各腿大致夹角、臂数、是否信号化（不只 branch_count）。
- **曲率**：`curve_direction`（left/right）+ `curve_sharpness`（gentle/sharp 或估计半径）。
- 每字段 `confidence`（已部分有），供 scorer 降权而非硬拒。

每个 actor 新增：
- **`flow_compliance: legal | wrong_way | unknown`** —— 相对其**所在车道合法流向**是否逆行，独立于 `heading_relation_to_ego`。prompt 明确提示**摩托车/两轮车常见逆行、贴边逆向、横穿，需重点判断**，逆行务必标 `wrong_way`，不得因"看起来不合规"改判车道。

### 4.5 软化硬约束（顺带修 #2 车道误判翻车）

- 把 `driving_lane_count≤2 + highway shoulder`、直路车道过宽、curve_score≥0.45 等 **hard-reject（`:3312`–`:3338`）降级为带 confidence 的软惩罚**。
- reranker（`map_match_reranker.py`）暴露给"被惩罚但高分"的候选，避免 VLM 数错车道把真值点直接淘汰且无法挽回。

## 5. 四种场景端到端走查

### 十字路口
- 匹配：VLM 给"前方四岔口 + 距离 + 哪条腿可左/右转" → 匹配 4 腿 junction，路口 waypoint 折叠为一个候选。
- 参考系：junction 中心 + 4 条腿。
- 摆放：ego = 中心 − 进近腿方向 × `ego_to_junction_distance_m`；对向车→对面腿，横穿车→左/右腿，转弯车→吸附 connecting road（自带正确弯曲朝向）。每车 yaw 读所在 lane；`wrong_way` 则 +180。

### T 字路口
- 匹配：VLM 给 3 腿 + 缺哪个方向 → 匹配 3 腿 junction，用"腿的有无方向"过滤掉四岔口。
- 摆放：同十字，少一条腿；**腿分配是关键**——把"右侧有支路"映射到真实侧腿，侧腿汇入车朝路口。

### 双向直道
- 匹配：找直的、双向、车道数匹配的 lane segment（沿 successor 确认前方无路口），按 road_id 折叠。
- 参考系：lane segment + `s`。
- 摆放：同向车→ego 半幅 lane；合法对向车→对向 lane（lane_id 反号，yaw 自然≈ego+180，**不手动加**）；逆行摩托→`lane_index=0` ego 道 + `wrong_way` → 落 ego 车道带，yaw = ego道合法向 +180（迎面冲来）。合法对向与逆行清楚分开。

### 弯道
- 匹配：VLM 给左/右弯 + 缓/急 → 匹配曲率方向与量级一致的弯段。
- 参考系：弯段 lane + `s`，forward 随 s 变。
- 摆放：yaw = 该 s 处局部切线；逆行 +180。车队顺弯排布，不再用固定轴越摆越偏。

## 6. 实施阶段

**Phase 0 — 准备**
- 给 `matched_structure` 定 schema 与 `_match.json` 字段；保持向后兼容（旧单点字段保留过渡）。

**Phase 1 — 候选折叠 + 结构输出**（`scene_map_matcher.py`）
- 按 road_id/junction_id 折叠候选；`_build_topology_match_record` 改吐 `matched_structure`。
- 用 `cache_map_topology.py` 预存连通图（`get_topology()`、junction connecting roads）。

**Phase 2 — VLM 字段扩展**（`scene_understanding_interpreter.py`）
- 新增 `forward_features`、junction 几何、曲率方向/锐度、actor `flow_compliance`，更新 prompt（摩托逆行提示）。

**Phase 3 — route-signature 匹配**
- 序列比对 + 曲率量级匹配；硬约束降级为软惩罚 + confidence。

**Phase 4 — 参考系 + 摆放重写**（`structured_pipeline.py`）
- `select_anchor_lane`/`_build_entity_coordinates` 从"单 lane 单轴"升级为"参考系 + 多腿"。
- 新增 `build_reference_frame(matched_structure)` 与 `place_actors_in_frame(...)`。

**Phase 5 — 朝向 + 逆行**
- 朝向读吸附车道 yaw + `flow_compliance` 控制 ±180；一致性校验按标志分流。

**Phase 6 — 联调 + 旧路径下线**
- 四类场景回归；确认 actor-graph 验证/修复仍可用。

## 7. 测试

- 单测：候选折叠、route-signature 序列比对、`build_reference_frame`（路口枚举腿）、朝向规则（含 `wrong_way` ±180）、一致性校验分流。
- 场景级：十字/T/直道/弯道各一组固定输入，断言锚点结构类型、每车所在腿/车道、朝向符号。
- 逆行专项：合法对向 vs 我车道逆行摩托，断言两者落在不同车道带、朝向相反，且逆行未被"掰正"。
- 回归：现有 `tests/test_ego_longitudinal_alignment.py`、`test_ego_lane_indexer.py`、`test_actor_graph_verifier.py`。

## 8. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 腿分配错（斜交/3 腿/腿数≠4） | 在小离散腿集合里选，比连续 lane 吸附好定义；加 VLM↔腿一致性日志 |
| VLM 距离/曲率粗估带误差 | clamp 到进近路实际长度；曲率只做量级分档不求精确 |
| `forward_features` 漏标/错标 | confidence 加权 + 软惩罚，不硬拒；序列允许部分缺失 |
| 地图无对应结构 | 显式返回低置信而非静默用最佳被拒候选；记录 `used_rejected_candidate` |
| `flow_compliance` 误判 | `unknown` 默认合法，绝不凭空造逆行；真实逆行不被校验掰正 |

## 9. 实施进度（2026-06-27）

| 阶段 | 状态 | 产出 / 验证 |
|---|---|---|
| Phase 4-5 核心 | ✅ 完成 | `tools/reference_frame.py`（参考系/摆放/朝向/逆行 + 一致性校验）；`tests/test_reference_frame.py` 13 项（含路口右臂车不横放、逆行摩托保真）；Town05 实地验证 |
| Phase 1 结构化匹配 | ✅ 完成 | `tools/map_structure.py`（junction 腿枚举 + road_segment 曲线采样 + 派发）；`SceneMapMatcher._build_matched_structure` 把 `matched_structure` 加法式写入 `_match.json`；`tests/test_map_structure.py` 8 项（mock，CI 无 CARLA 可跑）；Town05 实地 junction/road 验证 |
| Phase 2 `flow_compliance` | ✅ 完成 | VLM prompt 新增字段 + 摩托逆行提示；`_canonical_flow_compliance` 三处实体透传到 placement DSL |
| Phase 4 接线（路口） | ✅ 完成 | `tools/junction_placement.py` `reproject_actors_for_junction`；`AutoGenerator.reproject_junction_step` 在 refine 之后、spawn 之前应用（非路口 no-op）；`_apply_match_to_spawn_context` 透传 `matched_structure`。`tests/test_junction_placement.py` 10 项；**live CARLA(Town05 4 腿路口)验证**：ego/对向/左穿/右穿/逆行摩托全部落在正确车道、朝向与车道一致、逆行保真；**物理 spawn 5/5 成功** |
| Phase 4 接线（直路/弯道） | ✅ 完成 | `reproject_actors_for_road` + `reproject_actors_for_structure` 派发；管线步骤泛化为同时处理 junction/road。复用现有横向偏移、沿真实中心线（细采样 1m）重铺、朝向取车道切线、逆行特判拉回本车道。`tests/test_junction_placement.py` 共 16 项；**live CARLA 验证（Town05 直路+弯道）**：lead/对向/逆行摩托全部落对车道、朝向正确、弯道精确跟随（曲率 21°，d≈0） |
| Phase 2 召回（junction-anchor 门控） | ✅ 完成 | 见下方诊断与修复 |
| Phase 2 其余 | ⬜ 待办 | `forward_features` 有序前方特征序列、junction 臂夹角（3 vs 4 腿精配）、曲率方向/锐度 |
| Gap A：缓存流程透传 matched_structure | ✅ 完成 | 见下方 |

全套 338 项单测通过，零回归。

### Gap A 修复（端到端打通）
- **问题**：大图缓存匹配 `_match_from_cache` 提前 return，且缓存只存候选特征无 junction 几何；`_sample_dense_local_waypoints` 用的是当前已加载世界（非命中世界）。结果 `matched_structure` 始终 None，重投影不触发。
- **修复**（`AutoGenerator`，`_build_scenario_for_match` 内）：
  1. `_ensure_matched_world_loaded(match_report)`：命中世界与当前不一致时 `client.load_world(...)`。
  2. `_ensure_matched_structure(anchor, best_match)`：从加载后世界的真实 waypoint 经 `build_matched_structure_from_waypoint` 构建结构，存入 `spawn_context['matched_structure']`（已有则跳过）。
- **路口框 clearance**：`build_junction_structure` 增 `junction_radius_m`（包围盒半径）；进近车的沿腿距离取 `max(longitudinal, 半径+4m)`，避免落入路口内部车道不清的区域。
- **端到端实跑验证**（`data/start.jpg` T 路口）：日志依次出现 `Loading matched world Town12` → `Built matched_structure (junction)` → `Re-placing actors in junction reference frame`；最终 `_actors.json` 在 Town12 实测：ego 与对向车均 **dToLane=0**（朝向与车道完全一致）、**offroad≤1.0m**、对向 **180°**、对向车 **越出路口框**（`in_junction=False`）。

### Phase 2 召回修复（实跑诊断驱动）
- **症状**：在 T 字路口图上端到端实跑，VLM 正确识别 T_junction、签名 `topology_type=t_junction` 也对，但匹配把它配到 Town12 一条**直路**（`is_junction=False`、前方无路口）。
- **根因**：缓存里 25779 个候选中 5636 个真路口，但**得分最高(0.833)的是 `is_junction=False` 却被分类成 `t_junction` 的非路口点**（仅凭 `nearby_road_count=5`/`heading_cluster_count=3` 误判，`distance_to_junction_ahead=None`）；真路口只有 0.722，被压在下面。
- **修复**：`_score_topology_candidate_features` 增加 `junction_anchor_quality` 门控——路口目标下，`is_junction=True`→1.0、前方≤`JUNCTION_TARGET_REACH_M`(40m)有路口→0.9、否则→0.30，乘到 topology_score。常量与 `build_matched_structure_from_waypoint` 的 lookahead 对齐，使被接受的 approach 候选必能构出 junction 结构。
- **验证**：经真实匹配入口（含缓存路径）重跑，结果落到 **Town06 真路口** approach（road 50，前方有路口），`build_matched_structure` 产出 **4 腿 junction 结构**（junction 582，ego 距中心 25m）。新增回归单测。

### 直路/弯道接线的实现要点
- 复用现有（已处理对向车道/中央带几何的）(longitudinal, lateral) 偏移：把实体世界坐标在直线锚架里分解，再沿真实 `curve_samples` 重铺——直路几何上是 no-op，弯道则跟随弯曲，避免单轴越摆越偏。
- 朝向取所在位置的车道**局部切线**（对向 +180、逆行 +180），不再用固定 `anchor_yaw±90/180`。
- 逆行特判：把 wrong-way actor 从"对向偏移"拉回**自身车道带**（lane_index×车道宽），仅翻转朝向，保真逆行。
- 曲率采样步长 1m：5m 步长在急弯上累积 ~0.3m 横向漂移会把车 snap 到对向车道（实测），1m 步长稳定落在正确车道；采样每候选只算一次，开销可忽略。

### 路口接线的实现要点
- 接入位置：`refine` 之后、`validate` 之后、`build_spawn_payload` 之前——validation 仍跑在它熟悉的旧布局上，重投影只改最终 spawn 的坐标，降低回归风险。
- 腿分配：`direction_and_motion_for_entity` 把 `heading_relation_to_ego`+`turn_intent`+`lane_index` 映射到 ego/opposite/left/right 腿 + approaching/leaving。
- 横向：按 motion 把车偏移半车道进**正确行驶车道**（approaching=行进方向右侧车道），否则落在进/出分隔线会被 lane-snap 吸到反向车道（实测出现过 180° 错向，已修）。
- 缺腿：implied 腿不存在时保留原位并标 `junction_placement=unassigned`，不凭空造横放车。
- ego 仍由既有 ego 纵向索引在 candidate lane 上摆放；actor 用 junction 真实世界坐标，二者同图一致。`flow_compliance` 仅透传备用，未接入旧 `_yaw_for_heading_relation`（避免与现有 opposite_direction 翻转叠加导致双重翻转）；正确消费点是 Phase 4 改用 reference_frame 后。

## 10. 一句话总结

把地图匹配从"选对一个点、再把点当锚摊车"改成"**选对一个结构、在结构参考系里重算每辆车**"：匹配按 road_id/junction_id 折叠并走连通图做 route-signature；锚点对路口取中心+枚举腿、对直路弯道取 lane+s；朝向一律读吸附车道的真实方向，并用独立的 `flow_compliance` 字段**保真真实逆行、杜绝管线假逆行**。

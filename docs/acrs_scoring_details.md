# ACRS 静态场景评估：完整打分细节与讲解手册

本文档用于解释 ACRS（Accident-related static scene Reconstruction Score）的设计、证据来源、逐层计算方法和结果含义。所有分数最终转换为 `0–100` 分。

## 1. ACRS 总体定义

ACRS 由三个维度组成：

| 维度 | 记号 | 总权重 | 评估目标 |
|---|---:|---:|---|
| 道路拓扑保真度 | `R` | 0.40 | 道路类型、车道组织和交叉口连通性 |
| 背景环境保真度 | `B` | 0.20 | 天气、照明、时段、路面和周围环境语义 |
| 交通参与者保真度 | `T` | 0.40 | 关键及背景参与者的恢复和道路相对布局 |

总分公式：

```text
ACRS = 100 × (0.40R + 0.20B + 0.40T)
```

代码内部的 `R/B/T` 均在 `[0, 1]` 范围内，最终报告中的三个维度和 ACRS 会乘以 100。

例如：

```text
R = 0.94
B = 0.75
T = 0.820357

ACRS = 100 × (0.40×0.94 + 0.20×0.75 + 0.40×0.820357)
     = 85.4143
```

## 2. 评估证据来自哪里

### 2.1 人工参考真值

参考真值是 `{scene_id}_acrs_reference.json`，schema 为 `acrs-reference-v1`。它由 `_su.json` 自动预填，但必须人工检查，特别是：

- `critical_actor_ids`：事故关键参与者 ID；
- `road_topology`：道路、车道及分支；
- `environment`：原图中的环境事实；
- `actors`：参与者类别、车道、分支、朝向和相对位置；
- `pairwise_relations`：非 ego 参与者之间的空间关系。

### 2.2 候选场景道路证据

道路候选事实只读取 `_match.json.best_match.candidate_features` 以及其中的候选分支证据。以下内容禁止作为候选真值：

- `_topo.json`；
- `_match.json` 中的所有 `target_*` 字段。

原因是它们描述的是原始目标，而不是 CARLA 中实际匹配到的道路；使用它们会形成循环评分。

### 2.3 候选参与者证据

参与者实际状态读取编号最大的 `{scene_id}_render_actor_graph_rN.json`。正式评估要求：

```json
"truth_source": "carla_actor_transform"
```

只有 `spawned=true` 的 actor 才视为成功恢复。车道和空间关系优先从 CARLA 实际的 `ego_frame`、`actual_waypoint` 和 transform 推导，不把 `_actors.json` 中的预期布局直接当作实际结果。

### 2.4 候选环境证据

天气配置可从 `_actors.json.metadata.carla_weather_preset` 取得；正式 hybrid 模式还会让 VLM 从 CARLA ego-view/BEV 中提取：

- 天气、照明、时段和路面状态；
- 城市密度和左右道路环境；
- 地标与交通控制设施。

VLM 只提取事实，不直接给相似度分。中高置信度视觉事实优先；视觉输出为 `unknown` 时才回退到结构化信息。

## 3. 通用比较规则

### 3.1 精确类别比较

布尔值和规范化后的类别通常采用精确比较：

```text
一致 → 1
不一致或候选缺失 → 0
```

系统会先处理部分同义词，例如：

```text
straight_road / straight_two_way / line → straight
daytime → day
nighttime → night
traffic_lights / overhead_traffic_lights → traffic_light
buildings → building
```

### 3.2 数值或计数比较

车道数、分支数和车道编号采用连续相似度：

```text
score = max(0, 1 - |actual-reference| / max(|actual|, |reference|, 1))
```

示例：参考车道数为 2、实际为 3 时，得分为 `1-1/3=0.6667`。

### 3.3 集合比较

道路分支、周围环境标签和地标集合采用 Jaccard 相似度：

```text
J(A,B) = |A∩B| / |A∪B|
```

例如参考为 `{vegetation, sidewalk}`，实际为 `{vegetation, building, sidewalk}`，得分为 `2/3`。两个集合均为空时得 1；参考为空但实际非空时得 0。

### 3.4 `unknown` 和空数组的区别

这是人工标注中最重要的规则之一：

```json
"roadside_context_left": "unknown"
```

表示原图无法判断，该字段不进入分母。

```json
"roadside_context_left": []
```

表示人工确认左侧不存在任何相关环境元素，该字段正常参与比较。因此候选若出现元素，集合得分为 0。

候选缺失一个人工已知字段时，该字段得 0。

### 3.5 环境标签与交通参与者的边界

停车车辆属于 `actors`，参与交通参与者保真度 T 的评分，不属于左右道路
环境。评估器会从 `roadside_context_left`、`roadside_context_right` 和
`landmarks_and_controls` 中自动移除 `parkingcar`、`parked_car(s)`、
`parked_vehicle(s)`、`car(s)` 和 `vehicle(s)` 等参与者标签，防止在 B 和 T
中重复计分。

在 ACRS 第一版的语义粒度下，`residential_houses`、`residential_house`、
`residential_buildings` 和 `residential_building` 统一规范化为
`residential_building`。如果人工明确需要表达公寓楼，可使用更具体且独立的
`apartment_building` 标签。

## 4. 道路拓扑保真度 R

道路拓扑公式：

```text
R = 0.30 × Geometry
  + 0.40 × LaneOrganization
  + 0.30 × Connectivity
```

### 4.1 道路几何类型 Geometry

支持的规范类型包括 `straight`、`curve`、`t_junction`、`cross_intersection`、`roundabout` 和 `multi_branch`。规范化后完全一致得 1，否则得 0。

### 4.2 车道组织 LaneOrganization

| 字段 | 权重 | 比较方式 |
|---|---:|---|
| `directionality` | 0.20 | 精确类别比较 |
| `forward_lane_count` | 0.25 | 连续计数相似度 |
| `opposing_lane_count` | 0.20 | 连续计数相似度 |
| `ego_lane_from_right` | 0.15 | 连续计数相似度 |
| `has_center_median` | 0.10 | 布尔精确比较 |
| `left_parking_presence` | 0.05 | 布尔精确比较 |
| `right_parking_presence` | 0.05 | 布尔精确比较 |

公式：

```text
LaneOrganization =
  0.20×directionality
+ 0.25×forward_lanes
+ 0.20×opposing_lanes
+ 0.15×ego_lane
+ 0.10×median
+ 0.05×left_parking
+ 0.05×right_parking
```

### 4.3 交叉口连通性 Connectivity

参考场景存在交叉口时：

| 字段 | 权重 | 比较方式 |
|---|---:|---|
| `junction_visible` | 0.25 | 布尔精确比较 |
| `junction_type` | 0.25 | 规范化类别比较 |
| `junction_branches` | 0.35 | 分支集合 Jaccard |
| `branch_count` | 0.15 | 连续计数相似度 |

分支集合只统计 `ahead/left/right/uturn`。`branch_count` 表示从 ego 视角可选的机动方向数量，不包含 ego 自己的来路。

参考场景明确无交叉口时，只检查：

```text
实际也无交叉口 → Connectivity=1
实际错误生成交叉口 → Connectivity=0
```

## 5. 背景环境保真度 B

背景环境公式：

```text
B = 0.25 × Weather
  + 0.25 × Lighting
  + 0.10 × TimeOfDay
  + 0.15 × RoadSurface
  + 0.25 × SurroundingContext
```

前四项均采用规范化类别精确比较。

### 5.1 周围环境 SurroundingContext

```text
SurroundingContext =
  0.35  × UrbanDensity
+ 0.225 × RoadsideLeft
+ 0.225 × RoadsideRight
+ 0.20  × LandmarksAndControls
```

| 字段 | 比较方式 |
|---|---|
| 城市密度 | 精确类别比较 |
| 左侧环境标签 | Jaccard |
| 右侧环境标签 | Jaccard |
| 地标与控制设施 | Jaccard |

这里评估的是语义保留，不比较建筑纹理、颜色、像素位置或图像 SSIM。

## 6. 交通参与者保真度 T

```text
T = 0.70 × CriticalRole
  + 0.30 × BackgroundRole
```

如果参考场景没有背景参与者，则跳过背景项并将有效角色权重重新归一化。正式参考必须至少包含一个非 ego 的关键参与者。

### 6.1 Actor 如何匹配

系统使用 Hungarian assignment 在参考 actor 和 CARLA 实际 actor 之间寻找全局最优一对一匹配，不要求两边 ID 相同。

单对 actor 的匹配相似度：

```text
MatchSimilarity =
  0.35 × Identity
+ 0.40 × RoadRelativeLayout
+ 0.25 × Spatial
```

匹配相似度低于 `0.5` 时视为没有匹配成功。

#### Identity

- 类别完全一致得 1；
- `car/truck/bus` 间错误匹配可得到 0.4 的弱兼容分；
- `motorcycle/bicycle` 间错误匹配可得到 0.4；
- 机动车与行人等明显不兼容类别得 0；
- 人工填写了具体 subtype 时，类别占 0.8、subtype 占 0.2。

#### RoadRelativeLayout

在人工已知的字段中等权比较：

```text
lane_assignment
lane_from_right
branch_assignment
heading_relation
```

#### Spatial

在人工已知的字段中等权比较：

```text
longitudinal_relation
distance_band
```

### 6.2 每种角色的最终分数

关键和背景角色内部均采用：

```text
RoleScore =
  0.35 × Recovery
+ 0.40 × RoadRelativeLayout
+ 0.25 × SpatialRelations
```

关键参与者的 Recovery 使用带类别相似度的召回率。漏掉关键 actor 会直接降低该项，额外背景 actor 不会惩罚关键参与者得分。

背景参与者的 Recovery 使用 soft precision、soft recall 的 F1：

```text
precision = softTP / background_candidate_count
recall    = softTP / background_reference_count
F1        = 2PR / (P+R)
```

因此背景车漏建和无依据的额外生成都会被惩罚。

### 6.3 两两空间关系

人工 `pairwise_relations` 支持：

```text
longitudinal_relation
lateral_relation
lane_relation
```

实际纵向/横向关系通过 CARLA transform 在 ego 坐标系中的米制位置计算。涉及 ego 的关系已经由 actor 自身字段评分，不会作为两个参与者间关系重复计算。

`lane_relation` 使用受控词表：

```text
same_lane
same_parking_lane
adjacent_left_lane
adjacent_right_lane
cross_lane
different_lane
```

推导规则为：两车位于同侧同一停车带时得到 `same_parking_lane`；普通车道标签相同时得到 `same_lane`；一车为对向且另一车非对向时得到 `cross_lane`；同向车辆横向间隔在 `1–5.5m` 时，根据另一辆车位于 entity 左侧还是右侧得到 `adjacent_left_lane/adjacent_right_lane`；其余情况为 `different_lane`。若运行时缺少车道证据则为 `unknown`，不会把未知车道错误判成 `different_lane`。

关系报告同时记录 `relation_id`、`entity_id` 和 `other_entity_id`，关系方向始终是“`entity_id` 相对于 `other_entity_id`”。参考文件使用不受支持的关系标签时会在评估开始前报错，不再静默产生技术性 0 分。

## 7. Coverage 的含义

`coverage` 不是准确率，而是人工参考中有多少预定权重拥有可评分真值：

```text
coverage = 有效字段权重 / 全部预定字段权重
```

例如某维度中 20% 权重的字段被人工标为 `unknown`，coverage 为 0.8。有效字段会重新归一化计算分数，因此低 coverage 下的高分不能与高 coverage 下的高分等价解释。报告时建议同时展示 `ACRS + R/B/T + 三个 coverage`。

## 8. `complete`、`official` 与诊断分

正式 hybrid 评估要求：

- CARLA runtime actor graph 可用；
- CARLA ego-view 或 BEV 可用；
- VLM 渲染环境事实提取成功；
- `mode=hybrid`。

全部满足时为 `status=complete, official=true`。缺少正式证据时仍可能计算开发诊断分，但会标记 `status=incomplete, official=false`。正式批量汇总只统计 `official=true` 的场景。

## 9. 实例：修复关系词表后 86.8512 分是怎样得到的

实例报告：`results/006_success/s0000_c0_acrs.json`。

### 9.1 道路拓扑 94.0

```text
Geometry     = 1.00
Lane         = 0.85
Connectivity = 1.00

R = 0.30×1 + 0.40×0.85 + 0.30×1
  = 0.94
  = 94.0分
```

车道组织的唯一扣分来自 `ego_lane_from_right`：参考为 0，实际为 `unknown`，得分为 0，权重为 0.15。其余方向性、同向/对向车道数、中央隔离及左右停车带全部一致：

```text
Lane = 0.20 + 0.25 + 0.20 + 0 + 0.10 + 0.05 + 0.05
     = 0.85
```

### 9.2 背景环境 78.4821

```text
Weather       = 1
Lighting      = 1
TimeOfDay     = 1
RoadSurface   = 1
Surroundings  = 0.1392857

B = 0.25×1 + 0.25×1 + 0.10×1 + 0.15×1 + 0.25×0.1392857
  = 0.784821
  = 78.4821分
```

周围环境仅为 0.1392857 的原因：

- 参考城市密度为 `urban`，渲染观察为 `suburban`；
- 左右环境参考只标注住宅和植被，渲染图还提取到人行道、路灯、围栏等，因此 Jaccard 分别为 `1/3` 和 `2/7`；
- 地标参考写为 `[]`，但渲染图提取到多个非空元素，因此该项为 0。

如果参考中的 `[]` 实际意思是“没有人工标注”，应改成 `"unknown"`；如果意思是“确认原图不存在”，则当前 0 分符合定义。

### 9.3 交通参与者 83.8869

5 个参考 actor 全部匹配成功，没有漏建或额外 actor。

关键参与者：

```text
Recovery = 1
Layout   = 0.6666667
Spatial  = 0.75

Critical = 0.35×1 + 0.40×0.6666667 + 0.25×0.75
         = 0.804167
```

关键参与者的丰富车道关系已经正确识别：`adjacent_right_lane` 和 `cross_lane` 均通过；仍有一项纵向 `aligned_with_other` 与实际 `behind_other` 不一致。空间子项综合为 0.75。

背景参与者：

```text
Recovery = 1
Layout   = 0.8888889
Spatial  = 0.8571429

Background = 0.35×1 + 0.40×0.8888889 + 0.25×0.8571429
           = 0.919841
```

最终参与者得分：

```text
T = 0.70×0.804167 + 0.30×0.919841
  = 0.838869
  = 83.8869分
```

### 9.4 ACRS 总分

```text
ACRS = 0.40×94.0 + 0.20×78.4821 + 0.40×83.8869
     = 37.6 + 15.69642 + 33.55476
     = 86.85118
     ≈ 86.8512分
```

## 10. 面向询问者的简短讲解模板

> ACRS 不做原图与仿真图的像素级相似度，而是评估事故相关静态结构是否被保留。道路、背景和参与者分别占 40%、20% 和 40%。道路比较道路类型、车道组织和交叉口分支；背景比较天气、照明、路面与周围环境语义；参与者通过一对一最优匹配比较关键车辆和背景车辆是否恢复、是否位于正确车道或分支、空间关系是否正确。人工无法判断的字段不进入分母，同时用 coverage 报告证据完整度。最终分数完全由确定性公式计算，VLM 只负责从 CARLA 渲染图中提取环境事实。

## 11. 查看单场景详细证据

每个 `{scene_id}_acrs.json` 中：

- `scores`：百分制总分和三个维度分；
- `dimensions.*.components`：中间子项分；
- `dimensions.*.details`：每个字段的参考值、实际值、得分和权重；
- `traffic_participants.matches`：actor 匹配结果；
- `missing_reference_actors`：漏建参与者；
- `extra_actual_actors`：额外生成参与者；
- `coverage`：有效参考字段覆盖率；
- `evidence_conflicts`：结构化证据与 VLM 渲染观察的冲突；
- `inputs`：使用的参考文件、match、actor graph、渲染图、模型和 prompt 版本。

实现位置：

- `tools/acrs_evaluator.py`：全部确定性评分逻辑；
- `agents/acrs_visual_extractor.py`：环境事实提取；
- `experiments/evaluate_acrs.py`：单场景和批量运行入口。

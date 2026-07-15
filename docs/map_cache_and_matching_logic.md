# AutoScenario 地图 Cache 构建与地图匹配完整逻辑

> 用途：给 GPT 或新开发者快速理解本项目中“CARLA 地图如何离线缓存、场景如何生成匹配目标、如何跨地图选点、匹配结果如何进入后续生成”的完整链路。
>
> 本文以当前主流程（结构化场景理解 + topology cache + v2 matcher）为主，同时说明无 cache 时的 live CARLA 路径和 legacy 路径。

## 1. 一句话总览

系统先在 CARLA 中把每张地图采样成一组候选锚点，并为每个锚点缓存道路拓扑、车道、路口、曲率、停车带、交通设施和环境特征；推理时再把 VLM 的 `scene_understanding.road_network.map_matching` 规范化成目标拓扑签名，对所有地图中的候选执行“结构分类 → 硬过滤 → 分项加权评分”，选出全局最高分的合法候选，最后加载命中的 CARLA 地图并补采局部高密度 waypoint，供车辆定位和场景生成使用。

```text
离线阶段
CARLA map -> 候选 waypoint 采样 -> 候选特征提取 -> 每地图一个 JSON cache

推理阶段
scene_understanding
  -> road_topology_signature
  -> 扫描所有 cache JSON
  -> candidate 结构分类
  -> hard gate
  -> v2 score
  -> 全局 best_match
  -> 加载命中地图
  -> matched_structure + dense_local_waypoints
  -> ego/其他 actor 投影与 spawn
```

## 2. 核心文件与职责

| 文件 | 职责 |
|---|---|
| `tools/cache_map_topology.py` | 连接 CARLA、加载单张地图、采样候选点、提取环境上下文、写入 cache JSON |
| `tools/scene_map_matcher.py` | 构建目标拓扑签名；提取候选特征；读取全部 cache；组织匹配、报告和 live CARLA 回退 |
| `tools/map_matcher_v2.py` | CARLA 无关的 v2 核心：目标规范化、候选分类、硬过滤、固定权重评分 |
| `experiments/auto_generate_all_vlm.py` | 主流程入口；配置 matcher；调用匹配；把结果写入 spawn context；加载命中地图并补建局部几何 |
| `tools/map_structure.py` | 从命中 waypoint 构造路口中心/分支或普通道路段的 `matched_structure` |
| `tools/structured_pipeline.py` | 使用 candidate lane、matched structure 和 dense waypoints 将语义布局投影到 CARLA |

## 3. 地图 Cache 如何构建

### 3.1 入口与输出

入口是：

```bash
python tools/cache_map_topology.py --map Town01
python tools/cache_map_topology.py --map Town03 --sample-step 5.0
python tools/cache_map_topology.py --map Town10HD_Opt --large-map
python tools/cache_map_topology.py --map /Game/Carla/Maps/Town12 --large-map \
  --road-walk-steps 8 --road-walk-dist 15
```

默认输出目录为 `data/map_cache/`。每张地图生成一个 JSON，文件名由 CARLA 的 `world_map.name` 将 `/`、`\` 替换成 `_` 后追加 `.json`。

顶层结构：

```json
{
  "world_name": "Town01",
  "sample_step": 5.0,
  "large_map": false,
  "candidate_count": 1234,
  "cached_at": "YYYY-MM-DDTHH:MM:SS",
  "candidates": []
}
```

构建过程必须有 CARLA Python API 和运行中的 CARLA server，因为函数会调用 `client.load_world(map_name)`。

### 3.2 预计算全图环境数据

加载地图后先执行两个全图预计算，避免对每个候选重复调用昂贵的 CARLA API。

1. 环境物体：通过 `world.get_environment_objects(label)` 读取以下标签的位置：
   `Buildings`、`Sidewalks`、`TrafficSigns`、`TrafficLight`、`Poles`、`Walls`、`Fences`、`Vegetation`、`Terrain`、`Ground`、`Water`。
2. 人行横道：调用 `world_map.get_crosswalks()`，并将 CARLA 可能返回的扁平或嵌套位置列表统一展平。

对每个候选点，在 60 m 半径内统计上述环境物体，并记录两倍半径内各类物体的最近距离。环境摘要包含：

- `counts`、`nearest_m`；
- `urban_score`：建筑、人行道、交通控制物、杆/墙/栅栏的加权饱和计分；
- `natural_score`：植被、地形、地面、水体的加权饱和计分；
- `environment_class`：`urban_like` / `natural_like` / `mixed` / `unknown`；
- `water_nearby`、`buildings_nearby`、`sidewalks_nearby`、`traffic_control_nearby`。

### 3.3 两种候选采样模式

#### 普通地图模式

调用 `world_map.generate_waypoints(sample_step)`，默认每 5 m 采一个 waypoint。对每个点调用：

```python
SceneMapMatcher._extract_candidate_features(all_sampled_waypoints, center_wp)
```

该方法以候选点为中心，在 `search_radius`（默认 30 m）内从全量采样点中收集邻居并计算局部拓扑。

#### 大地图模式

不生成全路网 waypoint，而是从 `world_map.get_spawn_points()` 开始，将 spawn point 投影到道路后调用：

```python
SceneMapMatcher._extract_local_candidate_features(world_map, center_wp)
```

它通过 waypoint 的相邻关系收集局部点，更适合 Town12 等大地图。

可选 `--road-walk-steps N`：从每个 spawn waypoint 分别向前、向后走 N 步，每步 `--road-walk-dist`（默认 15 m）；遇到分叉只取第一个候选，保证确定性。去重键为：

```text
(road_id, section_id, lane_id, int(s / bucket_m))
```

启用 road walk 时 bucket 为 `road_walk_dist`，否则为 1 m。注意：大地图默认 `road_walk_steps=0`，因此 cache 的空间覆盖主要由 CARLA spawn points 决定。

### 3.4 每个候选缓存什么

基础定位信息：

- `location: {x,y,z}`、`yaw`；
- `candidate_lane`：由 `_waypoint_to_lane_dict()` 生成，包含 road/lane/section/s、lane type/width、transform 等下游定位数据。

结构特征：

- `is_junction`；
- `junction_waypoint_ratio`：局部 waypoint 中路口点比例；
- `nearby_road_count`、`nearby_lane_count`；
- `heading_cluster_count`：局部航向按 30° 容差聚类后的数量；
- `estimated_junction_degree = max(nearby_road_count, heading_cluster_count)`；
- `candidate_topology_type`；
- `same_road_lane_count`；
- `same_direction_lane_count`、`has_parallel_same_direction_lanes`、`same_direction_lane_evidence`。

道路形态与设施特征：

- `is_curve`、`curve_yaw_delta_deg`、`curve_abs_yaw_delta_deg`、`curve_direction`、`curve_score`、`curve_sample_distance_m`；
- `left_parking_lane_present`、`right_parking_lane_present`；
- `has_crosswalk_nearby`；
- `has_center_median_candidate`、`center_median_evidence`；
- `has_highway_shoulder`；
- `distance_to_junction_ahead`；
- `distance_to_traffic_light_ahead`；
- `physical_junction_arms`：路口物理方向（ahead/left/right），用于和图像中的路口结构比较；
- `lane_maneuver_dirs` / `junction_branch_dirs`：当前车道可达的转向动作；它和物理路口分支不是同一个语义。

此外还有前述 `environment_context`。

## 4. 匹配目标如何从场景理解生成

当前 topology-first 入口是：

```text
AutoGenerator.analyze_topology_scene_match()
  -> SceneMapMatcher.analyze_topology_scene_assets()
  -> SceneMapMatcher.build_road_topology_signature()
```

输入主要来自：

```text
scene_understanding.road_network.map_matching
scene_understanding.road_network.lane_groups
scene_understanding.metadata.ego_localization
以及 side/environment 的结构化描述
```

优先级原则是：显式 `road_network.map_matching` 字段优先；缺失字段才从 `lane_groups`、控制元素、特殊道路区域和 metadata 回填。

生成的 `road_topology_signature` 主要包含：

```json
{
  "topology_type": "straight_road | straight_two_way | curve | t_junction | cross_intersection | multi_branch | roundabout | unknown",
  "junction_type": "...",
  "junction_visible": false,
  "target_branch_count": 1,
  "junction_branches": {"ahead": true, "left": false, "right": false, "known": true},
  "directionality": "...",
  "driving_lane_count": 2,
  "forward_lane_count": 1,
  "opposing_lane_count": 1,
  "left_parking_presence": false,
  "right_parking_presence": false,
  "has_crosswalk": false,
  "has_traffic_light": false,
  "has_traffic_sign": false,
  "has_center_median": false,
  "curve_direction": "straight | left | right | unknown",
  "side_context": {},
  "environment_context": {},
  "ego_lane_from_right": 0,
  "ego_to_junction_distance_m": 20.0
}
```

`analyze_topology_scene_assets()` 会把它写入 `<scene_id>_topo.json`，匹配报告写入 `<scene_id>_match.json`，候选调试信息写入 `<scene_id>_mm_candidates.json`。

## 5. Cache 与 Live CARLA 的路由规则

`SceneMapMatcher._match_to_carla()` 的第一条规则是：

```text
topology_cache_dir 已配置
AND 目录存在
AND 至少有一个 *.json
=> 立即调用 _match_from_cache()
```

因此 cache 路径优先，且匹配阶段不会调用 `CARLA load_world()`。它会跨 cache 目录中的所有地图做全局选择。

只有没有可用 cache 时才：

1. 导入 CARLA；
2. 连接 server；
3. 加载 `load_world_name` 或使用当前 world；
4. 在当前地图的 spawn points / waypoint 上实时提取和评分。

主实验默认配置把 `topology_cache_dir` 设为：

```python
os.path.join(os.getcwd(), "data", "map_cache")
```

## 6. v2 地图匹配算法

当前只要存在 `road_topology_signature` 且 `legacy_map_match=False`（默认），cache 和 live candidate 都使用 `tools/map_matcher_v2.py`。

### 6.1 目标规范化

`build_target_signature()` 将原始 signature 变成 v2 target：

- 场景类型 `scene_kind`：`junction` / `open_road` / `unknown`；
- 道路形态 `road_shape`：`straight` / `curve_left` / `curve_right` / `curve_unknown` / `unknown`；
- 路口类型：`t` / `cross` / `multi` / `roundabout` / `none`；
- 分支、车道数、中央隔离、停车、信号灯、路口距离、环境期望等。

判定路口的主要条件：`junction_visible=true`，或 topology 本身属于 T/cross/multi/roundabout；否则明确的直路/曲路且 `junction_visible=false` 才是 open road。

### 6.2 候选结构分类

`classify_candidate()` 先将候选放入结构池：

- `junction`：`is_junction=true`，或 `junction_waypoint_ratio >= 0.18`；
- `junction_approach`：前方 45 m 内有路口，同时 heading cluster 或 nearby road 至少为 3；
- `open_road`：局部方向和道路数量较少，且 junction ratio < 0.12；
- 其余为 `invalid`。

### 6.3 硬过滤（hard gate）

评分之前先剔除结构不兼容候选。硬过滤失败的候选仍保留在 debug/rejected 列表中，但不能成为最终 best match。

Open-road target 的主要拒绝条件：

- 候选为 junction / junction approach、位于路口内，或 junction ratio >= 0.18；
- 直路目标要求同向车道数和总行驶车道数精确匹配；
- 明确的曲路目标会拒绝显著过宽的候选；
- 直路/明确曲路要求中央隔离证据与目标一致，缺证据也拒绝；
- 若目标明确左右均无停车带，则候选任一侧出现停车带即拒绝。

Junction target 的主要拒绝条件：

- 候选不是 junction / junction approach；
- `physical_junction_arms` 与目标已知 ahead/left/right 分支不一致；
- 多车道目标遇到严重的 approach lane / total lane 不足；
- 信号灯路口目标的候选在 60 m 内没有交通灯证据。

Blacklist 也作为 hard reject：

- 带 `world` 的 blacklist 项跳过整张地图；
- 位置 blacklist 使用 `blacklist_radius_m`（主流程默认 35 m）排除该点附近候选；
- 用于重匹配时避免再次选到同一区域。

### 6.4 Open-road 固定权重评分

```text
score =
  0.24 * topology
+ 0.20 * curve
+ 0.16 * forward_lanes
+ 0.12 * driving_lanes
+ 0.09 * median
+ 0.10 * parking
+ 0.09 * context
```

额外乘法修正：

- 窄路目标若候选有 highway shoulder：`score *= 0.65`；
- 目标明确无信号灯，但候选 45 m 内有灯：`score *= 0.80`。

最低置信度为 `0.52`，低于阈值也视为 hard reject。

### 6.5 Junction 固定权重评分

```text
score =
  0.21 * topology
+ 0.20 * junction_type
+ 0.22 * branch_direction
+ 0.13 * ego_distance
+ 0.10 * approach_lanes
+ 0.05 * median
+ 0.06 * context
+ 0.03 * parking
```

信号灯目标中，灯距 <= 35 m 会最多乘 `1.05`；无灯或灯距 > 75 m 会乘 `0.70`（通常前面的 60 m hard gate 已先拒绝）。最低置信度为 `0.50`。

### 6.6 各分项的含义

- lane fit：完全一致为 1；相差 1 通常为 0.72；差距更大继续衰减；
- median fit：一致 1，冲突 0.25，未知给予中性分；
- parking fit：左右分别比较；目标明确“无”而候选“有”的惩罚强于目标“有”而候选“无”；
- curve fit：比较是否弯曲以及左右弯方向；
- branch direction：只用 `physical_junction_arms` 比物理分支，旧 cache 只有 lane maneuver 时不做错误的等价比较；
- junction type：根据 `candidate_topology_type` 和 `estimated_junction_degree` 比 T/cross/multi；
- ego distance：目标与候选到路口距离误差 <= 8 m 为 1，<= 20 m 为 0.72，否则 0.35；
- context：根据目标 urban/natural 期望与 cache 的环境分、建筑/人行道存在性计算。

### 6.7 跨地图全局选优

`_match_from_cache()` 按文件名顺序加载目录下所有 JSON，对每个 candidate 调用 v2：

1. 每张地图分别记录最高分 candidate、最高分 accepted、最高分 rejected；
2. 全局只在非 hard-reject 候选中比较分数；
3. 全局最高分成为 `best_match`；
4. 若没有任何 accepted，但存在 rejected，状态为 `unmatched`；完全没有候选则为 `no_candidates`。

旧 cache 在评分前还会调用 `_normalize_cached_parking_sides()`：按 road_id 汇总双向道路两侧停车信息，修复“从单个 waypoint 沿固定 left/right 链无法看到对向外侧停车带”的历史缺陷。

## 7. 匹配输出结构

成功时 `<scene_id>_match.json` 的核心字段：

```json
{
  "status": "matched",
  "world_name": "TownXX",
  "scene_features": {"road_topology_signature": {}},
  "candidate_summary": {
    "accepted": [],
    "rejected": [],
    "top_candidates": []
  },
  "best_match": {
    "search_strategy": "cache_v2",
    "location": {"x": 0, "y": 0, "z": 0},
    "yaw": 0,
    "coarse_score": 0.8,
    "refined_score": 0.8,
    "score_details": {},
    "candidate_features": {},
    "candidate_lane": {},
    "world": "TownXX"
  }
}
```

cache topology-only 路径的 `projected_layout` 为空；它只选择真实地图锚点，具体 actor 布局在后续结构化投影阶段完成。

## 8. 匹配结果如何进入后续生成

### 8.1 写入 spawn context

`AutoGenerator._apply_match_to_spawn_context()` 将 `best_match.candidate_lane` 放到 `carla_spawn_context.topology_sample` 首位，并补入：

- `distance_to_junction_ahead`，供无 CARLA 时的 ego 纵向对齐；
- `same_direction_lane_evidence`，供无 CARLA 时的 ego 横向换道/选道；
- 若报告自带有效 `matched_structure`，也写入 spawn context。

### 8.2 加载真正命中的地图

cache 匹配本身不加载 CARLA world。进入 `_build_scenario_for_match()` 后，如果 `require_carla_connection=true`：

1. `_ensure_matched_world_loaded()` 比较当前 world 和报告中的 `world_name`；
2. 不同则 `client.load_world(world_name)`；
3. 后续所有 live 几何必须从这张命中地图读取，避免“匹配 TownA、却在当前 TownB 上取 waypoint”。

### 8.3 构建 matched_structure

如果 cache 报告没有结构，`_ensure_matched_structure()` 会在命中位置获取 Driving waypoint，再调用：

```python
build_matched_structure_from_waypoint(waypoint, ego_inbound_yaw)
```

输出两类结构：

- 路口：路口中心、入口/出口 legs、方向和 lane 信息；
- 普通道路：沿道路前后采样的 road segment。

若锚点是路口前 approach lane，该模块会向前查看约 40 m，找到近端路口后构建路口结构。

### 8.4 补采 dense local waypoints

`_sample_dense_local_waypoints()` 从命中锚点开始 BFS：

- 默认半径 80 m、步长 2 m；
- 扩展 `next()`、`previous()`；
- 同时扩展左右相邻 Driving lane；
- 去重键 `(road_id, lane_id, round(s))`；
- 保存 x/y/z/yaw/lane_width/is_junction。

结果写入 `spawn_context["dense_local_waypoints"]`，用于：

- 选择 ego 的正确同向车道和纵向位置；
- actor 精确吸附到真实道路；
- 修正 z 高度；
- 在弯道上使用真实切线 yaw，而不是简单直线外推。

之后主流程才运行 relation DSL、初始坐标、排序、CARLA 投影、细化、校验、路口重投影和最终 spawn payload。

## 9. Live CARLA 路径与 Cache 路径的差异

| 项目 | Cache 路径 | Live CARLA 路径 |
|---|---|---|
| 搜索地图范围 | cache 目录中所有地图 | 当前或指定的一张 world |
| 匹配时需要 server | 不需要 | 需要 |
| v2 特征 | 构建 cache 时固化 | 匹配时实时提取 |
| topology-only | 直接返回最佳合法锚点 | 直接返回最佳合法锚点并可构建 structure |
| structured/legacy layout | cache 主路径不在匹配阶段 refine layout | 对 coarse top-K 做 `_refine_candidate_layout()` |
| 匹配后的 live 几何 | 后续再加载命中地图补建 | world 已加载 |

无 cache 且非 topology-only 时，live 路径先得到 coarse candidates，再对前 `coarse_top_k`（默认 20）执行实体布局投影和吸附，检查平均 snap distance 等质量指标，最后按 `refined_score` 选取。

## 10. Legacy 评分与配置注意事项

`SceneMapMatcher` 构造器仍保留：

- `topology_weight`；
- `side_context_weight`；
- `auxiliary_weight`；
- `min_refined_score`；
- `max_average_snap_distance`。

主实验传入的前三项默认是 0.70 / 0.20 / 0.10。但当前有 `road_topology_signature` 且 `legacy_map_match=False` 时，实际调用 `score_candidate_v2()`，使用本文第 6 节的固定权重；上述三项主要服务旧的 `_score_candidate_features()` 路径。不要误以为修改 `map_match_topology_weight` 会改变 v2 的 0.24/0.20 等权重。

## 11. 当前实现的重要边界和风险

1. Cache 不做版本校验。JSON 没有 schema/version/hash；特征提取逻辑改变后，旧 cache 仍会被读取，需要人工重建。
2. 单个坏 JSON 会被静默跳过。若文件可读但字段缺失，v2 多数会给未知/中性分或 hard reject。
3. 只要目录内有任意 JSON 就完全优先 cache，不会将 cache 候选与 live 候选混合，也不会在 cache 全部 unmatched 后自动回退 live。
4. 大地图不启用 road walk 时覆盖依赖 spawn points；某些很适合的道路区域可能根本没有候选。
5. 普通模式为每个候选扫描 `search_radius` 内的全量 sampled waypoints；地图大或 sample step 很小时，构建成本明显增加。
6. 环境上下文只按二维欧氏距离统计，不考虑遮挡、道路连通性或候选朝向左右侧。
7. 目标 signature 的显式布尔值很重要。例如明确 `junction_visible=false`、无停车带、无中央隔离会触发严格 hard gate；VLM 的错误确定性可能导致所有候选被拒绝。
8. v2 的直路 lane count 是严格硬约束；图像中车道数不可见或解析错误时，匹配召回率会下降。
9. Cache best match 通常没有 `matched_structure`，必须在后续成功加载命中 CARLA 地图才能获得高质量路口结构和 dense waypoints。
10. Cache 写文件不是增量的；每次构建一张地图会整体覆盖该地图对应 JSON。

## 12. 给 GPT 排查问题时建议同时提供的材料

至少提供：

1. 本文档；
2. 对应场景的 `<scene_id>_su.json`；
3. `<scene_id>_topo.json`；
4. `<scene_id>_match.json`；
5. `<scene_id>_mm_candidates.json`；
6. 命中地图 cache 中最佳候选附近的若干 candidate；
7. 若是投影/生成错误，再提供 `matched_structure`、spawn context 的 `dense_local_waypoints` 摘要和最终 spawn payload。

排查顺序建议：先看目标 signature 是否正确，再看候选为何 hard reject，然后看合法候选的分项评分，最后才看命中后的 lane indexing、matched structure 和 actor 投影。

## 13. 最关键的函数调用链

```text
# 离线构建
cache_map_topology.build_cache
  -> CARLA client.load_world
  -> get environment objects / crosswalks
  -> generate_waypoints 或 spawn_points + road walk
  -> SceneMapMatcher._extract_candidate_features
     或 SceneMapMatcher._extract_local_candidate_features
  -> _summarize_environment_context
  -> JSON dump

# topology-first 匹配
AutoGenerator.analyze_topology_scene_match
  -> SceneMapMatcher.analyze_topology_scene_assets
  -> build_road_topology_signature
  -> _match_to_carla
     -> [cache 存在] _match_from_cache
        -> map_matcher_v2.score_candidate_v2
           -> build_target_signature
           -> classify_candidate
           -> gate_candidate
           -> weighted score + confidence gate
        -> global best accepted candidate
     -> [无 cache] live CARLA candidate extraction/scoring

# 匹配后
AutoGenerator._apply_match_to_spawn_context
  -> _build_scenario_for_match
     -> _ensure_matched_world_loaded
     -> _sample_dense_local_waypoints
     -> _ensure_matched_structure
     -> _index_ego_on_candidate_lane
     -> structured coordinate projection
     -> spawn payload / scene script
```


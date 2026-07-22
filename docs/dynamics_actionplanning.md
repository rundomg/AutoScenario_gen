请分析当前项目代码，并为“借鉴 DeepAccident 的事故生成思想，提高真实事故动态重建准确度”制定一份可执行的工程实现计划。当前阶段只输出计划、接口设计和伪代码，不要直接修改代码。

## 项目背景

当前项目将真实事故视频重建为 CARLA 0.9.16 中可执行的事故场景。

现有动态事故重建流程大致为：

1. 对事故视频进行非均匀时间采样；
2. 使用 VLM 分析采样帧；
3. VLM识别事故主体、事故类型和车辆运动；
4. VLM直接生成基于时间的行为计划，例如：

   * 0–1秒保持速度；
   * 1–3秒加速；
   * 3秒后制动；
5. 在CARLA中执行；
6. 如果碰撞未发生或碰撞主体不正确，则重新调整行为。

目前的问题是：VLM能够判断“谁撞谁、谁在减速、谁在接近”，但难以准确估计初速度、加速度、制动时刻和碰撞位置。因此，直接由VLM生成精确时间表容易导致：

* 没有发生碰撞；
* 碰撞时间不准确；
* 撞击位置不正确；
* 碰撞主体错误；
* 碰撞方向与视频不一致；
* 同一提示多次生成结果不稳定。

## 希望借鉴的 DeepAccident 思想

DeepAccident公开仓库没有提供事故场景生成代码，因此不要尝试复制其实现。请根据论文中描述的核心思想自行设计：

1. 先确定事故车辆的计划路径；
2. 对交叉路径事故，计算两条路径的冲突点；
3. 计算车辆从初始位置到冲突点的沿路径距离；
4. 根据目标碰撞时间同步两辆车的到达时间；
5. 使用规则控制器执行车辆；
6. 根据实际碰撞时间和到达时间误差继续调整参数。

需要将这一思想扩展到本项目常见的三类事故：

* 追尾事故；
* 路口交叉或转向冲突；
* 换道或侧擦事故。

## 目标方法

请设计一个“Collision-Anchored / Simulation-Guided Behavior Fitting”模块。

新的职责划分应为：

### VLM负责离散语义

VLM只输出：

* 事故主体；
* striking actor和struck actor；
* 事故类型；
* 每辆车所在车道或路口分支；
* 直行、转弯、换道等maneuver；
* keep、accelerate、decelerate、brake、stop等行为阶段顺序；
* closing、separating、stopped等相对运动趋势；
* 大致碰撞时间区间；
* 关键事件顺序；
* 各判断的置信度。

VLM不再直接负责精确生成：

* 初速度；
* 精确加速度；
* 精确制动时刻；
* 精确碰撞坐标；
* 完整低层控制序列。

### 参数拟合模块负责连续数值

根据VLM语义、CARLA地图、车辆初始状态和碰撞约束，自动确定：

* 初速度；
* 目标速度；
* 加速度或减速度；
* 制动开始时间；
* 启动延迟；
* 行为阶段切换时间；
* 必要时的路径起点偏移。

## 请先检查现有代码

请先系统检查当前仓库，并回答：

1. 静态场景重建结果保存在哪里；
2. actor的初始transform、lane association、速度和ID如何表示；
3. 当前behavior schedule的数据结构在哪里定义；
4. 当前车辆控制使用Traffic Manager、BasicAgent、BehaviorAgent、自定义PID还是直接apply_control；
5. CARLA同步模式和固定时间步如何配置；
6. 当前collision sensor如何创建和记录；
7. 是否已经记录每一帧的：

   * transform；
   * velocity；
   * acceleration；
   * lane waypoint；
   * bounding box；
   * collision event；
8. 当前dynamic verify–repair代码位于哪些文件；
9. 哪些现有模块可以复用，哪些需要新增。

必须给出具体文件路径、类名和函数名。不要只做抽象描述。

## 需要设计的核心模块

请规划以下模块，并给出建议的文件路径、类、函数签名、输入输出和依赖关系。

### 1. Structured Accident Specification

设计统一的数据结构，例如：

```python
@dataclass
class AccidentSpecification:
    accident_type: str
    striking_actor_id: str
    struck_actor_id: str
    actor_maneuvers: dict
    behavior_sequences: dict
    relative_motion_constraints: list
    event_order: list
    collision_time_range: tuple[float, float]
    confidence: dict
```

需要说明如何兼容当前VLM输出，以及如何处理不确定候选事故主体。

### 2. Map-Constrained Path Builder

为每个事故主体生成CARLA可执行参考路径。

至少支持：

* 当前车道保持；
* 当前车道向前行驶；
* 路口左转；
* 路口右转；
* 路口直行；
* 必要时的换道路径。

路径建议表示为：

```python
@dataclass
class ReferencePath:
    waypoints: list
    cumulative_distances: list[float]
    lane_ids: list
    road_ids: list
```

需要说明：

* 如何从CARLA waypoint拓扑构建路径；
* 如何计算沿路径弧长；
* 如何查询车辆在路径中的当前位置；
* 如何避免直接依赖欧氏距离；
* 如何处理路口内waypoint连接；
* 如何处理两条路径只有车辆包围盒冲突、中心线并不严格相交的情况。

### 3. Collision Anchor Builder

分别为三类事故设计collision anchor。

#### 追尾事故

目标条件应基于前后车定向包围盒首次接触，而不是车辆中心点重合。

需要使用：

* 同车道或近似同路径；
* 初始纵向净间距；
* 两车沿路径位移；
* 前车后保险杠和后车前保险杠的位置。

#### 路口交叉事故

需要：

* 计算两条参考路径的冲突区域；
* 得到双方到冲突区域的沿路径距离；
* 估计目标同步到达时间；
* 允许车辆尺寸产生的冲突区域，而非仅单点交点。

#### 换道或侧擦事故

需要：

* 检测换道车辆进入目标车道的时刻；
* 检测两车纵向区间重叠；
* 使用两个定向bounding box首次相交作为碰撞条件。

建议统一接口：

```python
@dataclass
class CollisionAnchor:
    accident_type: str
    conflict_position: object
    actor_path_distances: dict
    target_time_range: tuple[float, float]
    expected_contact_sides: dict
```

### 4. Behavior Parameterization

不要让优化器直接搜索每一帧的throttle、brake和steer。

请设计低维行为参数，例如：

```python
@dataclass
class ActorBehaviorParameters:
    initial_speed: float
    target_speed: float
    acceleration: float
    deceleration: float
    brake_start_time: float
    start_delay: float
    phase_durations: list[float]
```

行为结构仍由VLM给出，例如：

* keep → brake；
* keep → accelerate → brake；
* decelerate → stop；
* keep → lane change；
* turn → brake。

参数模块只优化连续数值。

### 5. Initial Analytical Estimation

在运行CARLA搜索之前，先使用运动学关系计算初始值。

对于追尾：

```text
initial gap
+ lead displacement
- following displacement
= 0 at collision time
```

对于交叉事故：

```text
arrival_time_A = distance_A / estimated_speed_A
arrival_time_B = distance_B / estimated_speed_B
```

要求给出：

* 追尾事故的初始值计算方法；
* 交叉冲突的同步到达计算方法；
* 有加减速阶段时的分段运动学计算；
* 速度等级low/medium/high如何转换为合理的数值区间；
* 如何结合CARLA道路限速；
* 如何保证参数处于物理合理范围。

### 6. Black-Box Parameter Search

CARLA是黑盒仿真环境，请设计一个低成本的参数搜索模块。

MVP优先考虑：

1. 解析初值；
2. 粗到细网格搜索；
3. 必要时再考虑CMA-ES、Optuna或differential evolution。

请不要一开始引入过于复杂的可微优化。

建议搜索参数：

#### 追尾事故

* 后车初速度；
* 前车减速度；
* 后车制动开始时间；
* 可选：前车制动开始时间。

#### 路口事故

* 两车初速度；
* 启动延迟；
* 可选：进入路口后的目标速度。

#### 换道事故

* 换道开始时间；
* 换道持续时间；
* 两车纵向速度；
* 必要时的启动延迟。

请设计统一接口，例如：

```python
class BehaviorParameterSolver:
    def solve(
        self,
        scene_state,
        accident_spec,
        paths,
        collision_anchor,
    ) -> SolverResult:
        ...
```

### 7. CARLA Rollout Evaluator

每次参数候选执行后，需要记录：

* 是否发生碰撞；
* 首次碰撞时间；
* 首次碰撞位置；
* 碰撞参与者；
* 碰撞前相对速度；
* 接触面或碰撞方向；
* 是否先发生了其他非目标碰撞；
* 各车辆逐帧transform、速度和bounding box；
* 各关键帧时刻的lane和相对关系；
* 两个目标actor的最小bounding-box距离；
* 两车到冲突点的实际到达时间。

建议数据结构：

```python
@dataclass
class RolloutResult:
    collided: bool
    collision_pair: tuple[str, str] | None
    collision_time: float | None
    collision_location: object | None
    relative_impact_speed: float | None
    trajectory_logs: dict
    minimum_pair_distance: float
    relation_errors: dict
```

### 8. Objective Function

设计用于评价参数候选的损失函数。

必须包含：

* 目标碰撞是否发生；
* 碰撞主体是否正确；
* 是否发生非目标碰撞；
* 碰撞时间误差；
* 碰撞位置或道路相对位置误差；
* 碰撞类型或接触方向误差；
* 视频关键帧中的相对关系误差；
* 参数的物理合理性；
* 控制平滑性。

要求明确设计优先级：

1. 正确碰撞主体；
2. 正确事故类型；
3. 碰撞成功；
4. 碰撞时间和位置；
5. 中间轨迹关系；
6. 控制平滑性。

避免出现“错误车辆发生碰撞，但因为时间准确所以总分较高”的情况。

### 9. Directed Repair Rules

除了通用搜索，还应设计基于仿真结果的定向修复规则，例如：

* 未碰撞且后车始终落后：

  * 提高后车速度；
  * 推迟后车制动；
  * 增大前车减速度；
* 碰撞过早：

  * 降低后车速度；
  * 提前后车制动；
  * 减小前车减速度；
* 路口A车先通过：

  * 提高B速度；
  * 延迟A启动；
* 碰撞主体错误：

  * 暂时隔离背景actor；
  * 调整目标actor路径；
  * 修正候选事故主体；
* 碰撞方向错误：

  * 调整两车到达时间差；
  * 调整目标接触位置；
* 换道完成后才发生碰撞：

  * 提前纵向重叠；
  * 推迟换道完成时间。

请设计一个可扩展的repair action枚举和接口。

### 10. Candidate Accident-Pair Handling

VLM对事故主体可能不确定。

请设计top-K候选机制：

```python
@dataclass
class CollisionPairHypothesis:
    actor_a: str
    actor_b: str
    accident_type: str
    confidence: float
```

对多个候选分别进行：

* 路径可行性检查；
* 视频相对运动一致性检查；
* collision anchor构建；
* 参数搜索；
* CARLA执行验证。

最后综合：

* VLM置信度；
* 视频关系一致性；
* 几何可碰撞性；
* 仿真重建分数；

选择最优事故解释。

## MVP范围

请优先设计一个能快速实现的MVP，不要一次覆盖所有复杂事故。

建议MVP按以下顺序：

### Phase 1：追尾事故

只支持：

* 同车道；
* 两辆关键车辆；
* 前车keep/decelerate/stop；
* 后车keep/late brake；
* 优化后车初速度、前车减速度和后车制动时刻。

### Phase 2：路口交叉事故

支持：

* 两条固定CARLA路径；
* 冲突区域计算；
* 到达时间同步；
* 初速度和启动延迟优化。

### Phase 3：换道或侧擦事故

支持：

* 固定换道轨迹；
* 换道开始时间；
* 两车纵向同步；
* 定向bounding box碰撞验证。

## 工程约束

1. CARLA版本为0.9.16。
2. 优先复用当前代码，不要大规模重构。
3. 优先使用同步模式和固定delta seconds，保证重复执行稳定。
4. 搜索过程中必须支持场景快速reset。
5. 每次rollout尽量限制在4–6秒。
6. 背景actor在MVP中可以固定或临时禁用碰撞干扰。
7. 不要要求VLM输出精确米制速度。
8. 不要使用逐帧自由控制作为优化变量。
9. 不要假设DeepAccident公开了事故生成源码。
10. 方案必须能处理CARLA控制误差和车辆尺寸差异。
11. 方案需要兼容当前的atomic behavior primitives。
12. 最终输出应能保存为可复现的结构化scenario configuration。

## 期望输出格式

请按照以下结构输出完整计划。

### A. 当前代码审查结果

列出：

* 相关文件；
* 相关类和函数；
* 当前数据流；
* 可以复用的代码；
* 当前缺失能力；
* 潜在技术风险。

### B. 推荐总体架构

给出模块图和数据流：

```text
VLM Accident Specification
→ Path Construction
→ Collision Anchor
→ Analytical Initialization
→ Parameter Search
→ CARLA Rollout
→ Evaluation
→ Directed Repair
→ Final Scenario Configuration
```

### C. 文件级改动计划

用表格列出：

* 文件路径；
* 新增或修改；
* 主要类/函数；
* 输入；
* 输出；
* 依赖；
* 实现优先级。

### D. 关键数据结构

给出Python dataclass草案。

### E. 核心算法伪代码

至少给出：

1. 追尾事故参数拟合；
2. 路口冲突点和到达时间同步；
3. CARLA rollout evaluation；
4. 粗到细参数搜索；
5. 定向repair；
6. top-K事故主体候选选择。

### F. 分阶段实施路线

分别描述：

* Phase 1；
* Phase 2；
* Phase 3；

并为每一阶段给出：

* 实现内容；
* 验收标准；
* 需要的测试场景；
* 预计风险；
* 回退方案。

### G. 测试计划

至少包括：

* 单元测试；
* 不启动CARLA的几何测试；
* 运动学公式测试；
* CARLA集成测试；
* 重复执行稳定性测试；
* 失败案例测试；
* 与当前VLM时间表方案的对比实验。

### H. 最小可行实现建议

最后明确回答：

1. 第一版最少需要新增哪些文件；
2. 第一版只需要优化哪些参数；
3. 是否需要第三方优化库；
4. 一次场景大约需要多少次rollout；
5. 如何避免搜索空间爆炸；
6. 哪些功能应暂缓实现；
7. 如何在不修改现有pipeline图的情况下先接入当前系统。

请尽可能具体到当前仓库的实际代码结构。发现信息不足时，请基于仓库内容做最合理的工程假设，并明确标记这些假设，不要只提出问题。

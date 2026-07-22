# Collision-Anchored 事故还原 Pipeline

实现入口：`tools/collision_fitting/`。该模块把 VLM 的离散事故语义与连续运动参数解耦，数据流为：

```text
video_understanding.json
  -> AccidentSpecification / top-K collision pairs
  -> CARLA waypoint path (or deterministic offline fallback)
  -> vehicle-size-aware CollisionAnchor
  -> analytical kinematic initialization
  -> coarse-to-fine low-dimensional search
  -> kinematic OBB or CARLA rollout evaluation
  -> directed repair
  -> fitted actors + trajectory DSL + risk DSL + reproducible config
```

## 快速使用

只运行离线运动学和 OBB 搜索（不需要启动 CARLA）：

```bash
python experiments/fit_collision_anchored_scenario.py \
  --output-folder results/<run>/static \
  --risk-output-folder results/<run>/dynamic \
  --scene-id s0000_c0 \
  --understanding results/<run>/dynamic/s0000_c0_video_understanding.json \
  --base-trajectory-dsl results/<run>/dynamic/s0000_c0_video_trajectory_dsl.json
```

在现有视频重建命令中启用拟合：

```bash
python experiments/auto_generate_all_video_reconstruction.py \
  --frames-manifest <frames_manifest.json> \
  --output-folder <static-output> \
  --risk-output-folder <dynamic-output> \
  --scene-id s0000_c0 \
  --collision-anchored-fitting \
  --fitting-max-rollouts 48
```

CARLA 0.9.16 已运行且当前 Python 环境可以 `import carla` 时，可增加：

```bash
  --carla-rollouts --carla-host localhost --carla-port 2000 \
  --carla-python /path/to/carla/python
```

CARLA 模式为每个候选生成独立 DSL、脚本和 metrics；脚本使用同步模式、固定 `delta_seconds`、清场和重生。候选产物保存在 `collision_fit_rollouts/`，便于复现失败。

## 输出

- `*_collision_fitted_config.json`：事故规格、top-K 结果、参考路径、碰撞锚点、参数、损失分解和 repair 历史；
- `*_collision_fitted_actors.json`：必要的沿路径起点偏移后的 actor spawn payload；
- `*_collision_fitted_trajectory_dsl.json`：拟合参数转换成现有 atomic behaviors 后的程序；
- `*_collision_fitted_risk_dsl.json`：供现有 `DslEventController` 执行的 lowered DSL；
- `collision_fit_rollouts/*_metrics.json`：CARLA 中的碰撞对、时刻、位置以及逐帧 actor 状态。

## 当前范围

- 追尾：保险杠净间距、后车初速度、前车减速度、后车制动时刻；
- 路口：车辆宽度扫掠冲突区、双方速度和启动延迟同步；
- 换道/侧擦：目标车道进入时刻、换道持续时间、纵向同步和 OBB 首次接触；
- 优化器：无第三方依赖的粗到细网格 + 定向 repair；
- 最终精度仍应以 `--carla-rollouts` 结果为准，离线 backend 只作为快速初值和 CI 可测的 surrogate。

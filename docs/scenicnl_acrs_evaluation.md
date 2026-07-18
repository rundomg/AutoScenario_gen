# 使用 ACRS 评估 ScenicNL 场景

`experiments/evaluate_scenicnl_acrs.py` 用与 AutoScenario 相同的 ACRS 确定性
打分器评估 ScenicNL 生成的 `.scenic` 程序。由于 Scenic 程序包含随机分布，脚本会
对每个程序采样多次，并报告均值、标准差、最小值和最大值。

## 单场景

```bash
conda run -n autoscenario python experiments/evaluate_scenicnl_acrs.py \
  --scenic-path /home/zx/code/scenicNL/output/run/0-0.scenic \
  --reference annotations/s0000_acrs_reference.json \
  --samples 10 \
  --conda-executable /home/zx/miniconda3/bin/conda \
  --output-dir results/scenicnl_acrs
```

`--reference` 必须是人工复核过的 `acrs-reference-v1`，并且应与输入 ScenicNL 的
原始事故报告一一对应。不要用 ScenicNL 自己的输出反向生成参考真值。

ScenicNL 固定使用 `scenic==3.0.0b2`，同时应使用 `shapely<2.1`。Shapely 2.1
新增的 `sample` 属性会与该版 Scenic 的采样协议冲突，使编译成功的程序在
`scenario.generate()` 阶段报 `points() takes ...`。重新执行 `pip install -e .`
会应用仓库中的版本约束。

## 批量评估

manifest 可为 JSON 数组、`{"scenes": [...]}` 或 JSONL：

```json
[
  {
    "scene_id": "s0000",
    "scenic_path": "/home/zx/code/scenicNL/output/run/0-0.scenic",
    "reference": "/home/zx/code/AutoScenario_gen/annotations/s0000_acrs_reference.json",
    "samples": 10
  }
]
```

```bash
conda run -n autoscenario python experiments/evaluate_scenicnl_acrs.py \
  --manifest annotations/scenicnl_acrs_manifest.json \
  --conda-executable /home/zx/miniconda3/bin/conda \
  --output-dir results/scenicnl_acrs
```

输出包括每个程序的 `*_scenicnl_acrs.json`、批量 CSV 和 JSON 汇总。建议论文中同时
报告：

- `compiled_scene_count / scene_count`；
- `successful_samples / requested_samples`；
- 六项诊断分的 mean ± std：Road、Background Environment、Critical Traffic、
  Background Traffic、Traffic Total、ACRS；
- ACRS 各维度 coverage。

交通参与者同时报告三个字段：`critical_traffic_participants`、
`background_traffic_participants` 和兼容字段 `traffic_participants`。其中总交通
参与者分仍按 Critical 70% 与 Background 30% 加权，因此拆分不会改变既有 ACRS
总分或历史口径。

## 分数边界

当前脚本评估的是 Scenic 编译后采样出的静态初始状态：

- 道路事实来自 Scenic OpenDRIVE network；
- 交通参与者事实来自采样后的对象位置、朝向和类别；
- 天气只在 Scenic 参数已确定时使用；无法从地图可靠提取的环境字段保持
  `unknown`；
- 不执行 CARLA 动态仿真，不评价碰撞、TTC 或驾驶策略。

因此所有结果均为 `official=false`。正式 ACRS 还需要 CARLA 实际 actor graph、
ego/BEV 渲染图和基于渲染图的 VLM 环境事实；这些证据不能由 Scenic 源码或目标
参考真值替代。

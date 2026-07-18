# ACRS 静态场景评估

ACRS 将静态重建质量拆为道路拓扑、背景环境和交通参与者三部分，最终分数为：

```text
ACRS = 0.4 × Road + 0.2 × Background + 0.4 × Traffic
```

正式分数要求同时存在 CARLA 运行时 actor graph、CARLA 渲染图和渲染图的
VLM 环境事实提取结果。缺少任一证据时仍会生成诊断分，但报告的
`status=incomplete`、`official=false`。

## 1. 生成人工标注草稿

```bash
conda run -n autoscenario python experiments/evaluate_acrs.py \
  --create-reference-from-su results/crash_success/s0000_su.json \
  --scene-id s0000_c0 \
  --output-dir annotations
```

打开 `annotations/s0000_c0_acrs_reference.json`，检查道路与环境字段，把事故
关键参与者的 id 写入 `critical_actor_ids`。草稿在关键参与者未标记前会被评估器
拒绝，以避免将所有车辆误当成背景交通。

`unknown` 表示原始证据不足，该字段不会进入评分分母；空数组表示人工确认该类
对象不存在，会正常参与集合比较。

## 2. 单场景评估

```bash
conda run -n autoscenario python experiments/evaluate_acrs.py \
  --result-dir results/crash_success \
  --scene-id s0000_c0 \
  --reference annotations/s0000_c0_acrs_reference.json \
  --render-image auto \
  --render-graph auto \
  --mode hybrid
```

`auto` 会选择编号最大的 `{scene_id}_ego_rN.png`、`_bev_rN.png` 和
`_render_actor_graph_rN.json`。VLM 观察结果会缓存为
`{scene_id}_acrs_visual_observation.json`，之后可通过 `--visual-observation` 复用，
避免模型版本变化影响同一批实验。

使用 `--skip-vlm` 或 `--mode structured` 可以做离线诊断，但结果不是正式 ACRS。

## 3. 批量评估

manifest 可以是 JSON 数组、`{"scenes": [...]}` 或 JSONL。每项支持：

```json
{
  "scene_id": "s0000_c0",
  "result_dir": "results/crash_success",
  "reference": "annotations/s0000_c0_acrs_reference.json",
  "render_image": "auto",
  "render_graph": "auto"
}
```

```bash
conda run -n autoscenario python experiments/evaluate_acrs.py \
  --manifest annotations/acrs_manifest.json \
  --output-dir results/acrs_eval
```

输出包括逐场景 `{scene_id}_acrs.json`、`acrs_summary.csv` 和只统计
`official=true` 场景的 `acrs_summary.json`。

## 证据隔离原则

- 道路候选事实只来自 `_match.json.best_match.candidate_features` 和其中的
  candidate 分支证据。
- `_topo.json` 与任何 `target_*` 字段都不会作为候选事实进入评分。
- actor 只采用 `truth_source=carla_actor_transform` 的运行时图；spawn 失败的 actor
  计为漏建。
- 参与者匹配采用 Hungarian assignment，不要求参考与候选使用相同 id。
- VLM 仅输出受控环境事实，不输出也不影响任何主观相似度分。

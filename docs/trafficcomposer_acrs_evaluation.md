# 使用 ACRS 评估 TrafficComposer 场景

`experiments/evaluate_trafficcomposer_acrs.py` 将 TrafficComposer 的 textual/merged
YAML IR 转换为 ACRS 候选事实，并调用与 AutoScenario、ScenicNL 对比实验相同的
确定性评分器。

## 公平对比场景

`annotations/trafficcomposer_image_txts/trafficcomposer_acrs_manifest.json` 固定使用
ScenicNL 对比实验的 5 个 Nexar 场景及同一份人工复核
`acrs-reference-v1`。TrafficComposer 输出应写入：

```text
/home/zx/code/TrafficComposer/results/scenicnl_image_txts/<scene_id>.yaml
```

不要用 TrafficComposer 输出反向生成参考真值。

使用 AutoScenario `.env` 中的 `OPENAI_KEY`、`OPENAI_URL`、`OPENAI_MODEL`、超时和
重试设置生成 TrafficComposer IR：

```bash
conda run -n autoscenario python experiments/generate_trafficcomposer_ir.py
```

该入口保留 TrafficComposer 原始 system/few-shot/user messages 和原始
`post_process`，仅将 API 客户端配置替换为 AutoScenario 的配置。原始模型响应保存在
`TrafficComposer/results/scenicnl_image_txts/_raw/`，方便审计。

多模态模式使用仓库自带的 `multi_modal_gpt` prompt，并保留两个示例的文本与 gold
YAML 以及目标场景图像。由于两张 bundled few-shot PNG 合计约 12 MB、在当前 API
代理上会长时间阻塞，运行入口只移除示例图像本身；目标图像始终以 `detail=high`
发送。该兼容差异会写入实验说明，不与论文原配置混淆。

## 单场景

```bash
conda run -n autoscenario python experiments/evaluate_trafficcomposer_acrs.py \
  --traffic-ir /home/zx/code/TrafficComposer/results/scenicnl_image_txts/nexar_00006_t017133.yaml \
  --reference annotations/scenicnl_image_txts/nexar_00006_t017133_acrs_reference.json \
  --output-dir results/trafficcomposer_acrs
```

## 批量评估

```bash
conda run -n autoscenario python experiments/evaluate_trafficcomposer_acrs.py \
  --manifest annotations/trafficcomposer_image_txts/trafficcomposer_acrs_manifest.json \
  --output-dir results/trafficcomposer_acrs
```

输出包括逐场景 `*_trafficcomposer_acrs.json`、CSV 和 JSON 汇总。论文中应同时
报告 Road、Background Environment、Critical Traffic、Background Traffic、
Traffic Total、ACRS，以及各维度 coverage。

## 当前可复现实验结果（GPT-5.4）

五个场景均成功生成 IR。所有结果都是结构化诊断分，`official=false`：

| 变体 | Road | Background | Critical Traffic | Background Traffic | Traffic Total | ACRS |
|---|---:|---:|---:|---:|---:|---:|
| upstream 后处理输出 | 12.0000 | 57.5000 | 0.0000 | 13.2667 | 3.9800 | 17.8920 |
| 文本 IR + 后处理缺陷修复 | 12.0000 | 57.5000 | 62.8686 | 49.1667 | 58.7580 | **39.8032** |
| 仓库 multi_modal_gpt 兼容运行 | 12.0000 | 62.5000 | 63.5619 | 28.4722 | 53.0350 | 38.5140 |
| 文本关系 + 多模态 lane_idx 融合 | 12.0000 | 57.5000 | 62.8686 | 43.7222 | 57.1247 | 39.1499 |

后处理修复仅阻止 `post_process` 把模型已经生成的直接 ego/道路空间关系覆盖成
`None`，不使用参考真值。融合版本用文本 IR 保留行为与空间关系，用图像支持的
multi-modal IR 补 lane index 与额外可见参与者，匹配 TrafficComposer 的模态分工。

当前最佳的固定方案是修复后的文本 IR（ACRS 39.8032），但不能在论文主表中按
场景挑选四个变体的最高值。多模态和融合结果应作为独立消融项完整报告。

融合复现命令：

```bash
conda run -n autoscenario python experiments/merge_trafficcomposer_reproducible_ir.py
conda run -n autoscenario python experiments/evaluate_trafficcomposer_acrs.py \
  --manifest annotations/trafficcomposer_image_txts/trafficcomposer_reproducible_fused_acrs_manifest.json \
  --output-dir results/trafficcomposer_acrs_reproducible_fused
```

## 分数边界

当前公开 TrafficComposer 仓库仅包含文本/视觉 IR 生成与 IR 合并，没有发布
IR 到 CARLA 可执行场景的转换器。YAML IR 不能提供：

- CARLA 实际生成后的 actor transform；
- CARLA 地图中实际匹配到的道路与车道；
- CARLA ego/BEV 渲染图及 VLM 环境观察。

因此脚本输出的 `status=incomplete`、`official=false`，属于 IR 级结构化诊断
ACRS。即使分数数值完整，也不能与正式 ACRS 混报。TrafficComposer 的
`lane_number` 是视觉 IR 的总车道数，而 ACRS 分别评价同向与对向车道数；适配器
会保留总数作为诊断信息，但不会把它伪造成任一 ACRS 字段。

要得到正式 ACRS，需要 TrafficComposer 补齐 CARLA 场景生成与运行，并导出与
`experiments/evaluate_acrs.py` 相同的 runtime actor graph、ego/BEV render 和
VLM observation，再使用 `mode=hybrid` 评分。

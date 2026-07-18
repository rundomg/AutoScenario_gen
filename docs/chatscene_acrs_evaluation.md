# 使用 ACRS 评估 ChatScene 场景

本实验将 ChatScene 的公开 v1 文本检索流程接入与 ScenicNL 相同的 ACRS 静态场景
评测器。输入为 `scenicNL/eval_txts/image_txts/` 中同一批 6 个 Nexar 事故文本，参考
真值为同一批人工复核的 `acrs-reference-v1`；生成过程不读取参考真值。

## 生成协议

- ChatScene 数据库：`retrieve/database_v1.pkl`；
- LLM 仅按 ChatScene 原始 `extraction.txt` 把完整描述拆成 adversarial object、
  behavior、geometry 和 spawn；
- 分解模型：AutoScenario 当前统一配置的 `gpt-5.4`；
- 检索模型：ChatScene 原始配置的
  `sentence-transformers/sentence-t5-large`；
- `topk=3`，采用 ChatScene 默认的纯检索模式，即每类直接选择 top-1 snippet，
  不启用 `--use_llm` 改写 snippet；
- Scenic：ChatScene 自带的 `2.1.0b4` fork；
- 每个程序固定种子后请求 30 个静态样本。

ChatScene 上游示例默认使用 `gpt-4o`。这里仅把分解模型替换为与本工作区其他生成
基线一致的 `gpt-5.4`，原始 prompt、知识库、sentence-t5 检索和 top-1 拼接逻辑均
保持不变。

生成命令：

```bash
cd /home/zx/code/AutoScenario_gen
/home/zx/miniconda3/bin/conda run --no-capture-output -n autoscenario \
  python experiments/generate_chatscene_scenic.py
```

评测命令：

```bash
cd /home/zx/code/AutoScenario_gen
/home/zx/miniconda3/envs/autoscenario/bin/python \
  experiments/evaluate_scenicnl_acrs.py \
  --manifest annotations/chatscene_image_txts/chatscene_acrs_manifest.json \
  --conda-executable /home/zx/miniconda3/bin/conda \
  --conda-env autoscenario \
  --scenic-pythonpath /home/zx/code/ChatScene/Scenic/src \
  --source-id chatscene --source-display ChatScene \
  --output-dir results/chatscene_image_txts_gpt54_n30
```

不要从已激活的 `autoscenario` 环境中再用 `conda run -n autoscenario` 包裹评测
入口；评测器本身会为每个 Scenic 程序启动该环境，重复激活会污染 PATH。

## 结果

所有 6 个程序均可编译。5 个场景产生至少一个有效静态样本，总采样成功率为
`123/180 = 68.3333%`。仅在 5 个有效场景上计算的 scene-macro ACRS 为
`39.3232`；将完全失败的第 6 个场景按 0 分计入时，保守六场景均值为 `32.7693`。
按所有 123 个成功样本汇总的 pooled ACRS 为 `35.7081 ± 12.9270`。

| Scene | 成功样本 | Road | Background | Critical | Background traffic | Traffic total | ACRS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00003 | 26/30 | 21.0385 | 26.1538 | 76.6667 | 50.8974 | 68.9359 | 41.2205 |
| 00006 | 7/30 | 82.1429 | 37.1429 | 56.0119 | 19.6428 | 45.1012 | 58.3262 |
| 00160 | 30/30 | 24.5965 | 23.1667 | 0.0000 | 35.8333 | 10.7500 | 18.7719 |
| 00213 | 30/30 | 26.3000 | 51.8333 | 16.8667 | 60.0945 | 29.8350 | 32.8207 |
| 00234 | 30/30 | 80.0000 | 21.8333 | 9.8572 | 52.9167 | 22.7750 | 45.4767 |
| 00302 | 0/30 | — | — | — | — | — | — |
| 有效场景 macro | 123/180 | 46.8156 | 32.0260 | 31.8805 | 43.8769 | 35.4794 | **39.3232** |

采样失败不是编译器兼容错误。默认纯检索把独立选择的 geometry 与 spawn snippet
直接拼接；部分样本的 spawn snippet 请求不存在的相邻右车道，产生
`AttributeError: 'NoneType' object has no attribute 'lane'`。失败数分别为 00003 的
4 次、00006 的 23 次和 00302 的 30 次，均原样计入生成成功率，不做人工修复。

## 分数边界

结果为 `official=false` 的结构化诊断 ACRS。证据来自 Scenic 2.1 编译后的静态
初始状态，没有运行 CARLA 动态仿真，也没有 CARLA runtime actor graph、ego/BEV
渲染或 VLM 环境观察。因此它可与 ScenicNL 的结构化诊断结果按相同口径对照，不能
与 AutoScenario 的正式 hybrid ACRS 混报。

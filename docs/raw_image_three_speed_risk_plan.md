# Raw Image + Three-Speed Risk Scenario Plan

## Summary

在当前 risk 生成流程中重新引入原图，但用途限定为：让 VLM 识别原图中的主体道路参与者，并为自车给出低/中/高三档速度假设。CARLA `actors.json` / `match.json` 仍作为可执行约束来源，VLM 输出必须引用已有 CARLA actor id，后续仍走现有模板、采样、脚本生成链路。

> 命名说明：本文用 `actors.json` / `match.json` 作为简称。实际内存结构是 `spawn_payload`，actor 列表在 `spawn_payload["entities"]`（键名是 `entities`）；文件名由 `RiskScenarioRunner` 按候选名加载（`{scene_id}_actors.json` 或 `_spawn_entities.json`；`{scene_id}_match.json` 或 `_scene_match.json`）。VLM 看到的 actor table 是 `build_actor_context(spawn_payload)` 的派生输出。

## Review And Corrections (2026-06-15)

本节记录本计划与 `AutoScenario_gen` 当前代码核对后的修正。结论：方向正确、比 `scenicnl_internalization_design.md` 更接近可实现，但存在 2 个落地阻塞项与若干精度缺口，已在下文对应处修订。

### 阻塞项 1：`speed_hypotheses_mps` 必填与“现有 risk-spec 兼容”自相矛盾

原文一处要求该字段缺失时校验报错，另一处又承诺“现有 risk-spec 模式保持可用”。但磁盘上既有的 `risk-scenario-v1` risk spec **都没有** 该字段，强制必填会破坏被承诺保留的兼容路径。

**修正决策：** 仅对“本次新生成（VLM 路径）”的输出强制 `speed_hypotheses_mps`；对通过 `--risk-spec` 传入的已有 spec，缺失时不报错，而是注入常量默认三档 `4/8/12 m/s`，并在 summary 中标记 `speed_hypotheses_source: "default_fallback"`。校验层只在该字段“存在”时强制 `low < medium < high` 且非负。

### 阻塞项 2：三速展开缺少明确的接入点

`tools/risk_scenario_runner.py::build_artifacts` 中 `validate_risk_scenario_spec` 与 `sample_all_risk_scenarios` 是相邻调用，中间没有任何钩子，展开逻辑无处落脚。

**修正决策：** 在 `build_artifacts` 中、`validate_risk_scenario_spec` 之后、`sample_all_risk_scenarios` 之前插入 `expand_candidates_by_speed(normalized_spec)`，用展开后的 spec 再去采样。展开函数返回的 spec 中，每个速度变体是 `risk_candidates` 列表里的独立 candidate（其 `candidate_index` 自然唯一，详见下文文件名说明）。

### 精度缺口

- **`ego_target_speed_mps` 固定值机制（可行但原文未点明）：** 它是 `tools/accident_template_library.py` 多个模板的 `parameter_ranges` 字段。无需从模板移除——展开阶段把该 candidate 的 `parameter_ranges["ego_target_speed_mps"]` 写成 `[v, v]`，`_normalize_range` 会保留为定值区间，`sample_risk_scenario` 采样即返回 `v`，从而“固定且不随机”。
- **产出数量 3× 放大（需显式知情）：** `samples_per_candidate` 默认 3。三速展开后，1 个 VLM candidate → 3 个 candidate → 9 个 sample/script。这是预期行为（3 速度 × 3 随机参数采样），但属于成本决策，需明确接受或调小 `samples_per_candidate`。
- **文件名碰撞实为自解：** 现有命名 `{scene_id}_r{candidate_index:03d}_s{sample_index:03d}` 在展开重写 `candidate_index` 后天然唯一；`source_candidate_index` 仅用于回溯原始 VLM candidate，不是文件名唯一性所必需。
- **`ego_speed_label` 传播链需补全：** 展开阶段在每个速度 candidate 上打 `ego_speed_label` 与定值 `ego_target_speed_mps` → `sample_risk_scenario` 透传到 sample JSON → `build_artifacts` 的 `artifact` dict 增加 `ego_speed_label` 字段并写入 summary。三处都要改，缺一则 summary 无法区分速度档。

### 落地补充（解决实现歧义）

- **CLI 入口明确：** 当前独立 risk 入口是 `experiments/generate_risk_scenarios.py`。应在该文件增加 `--image-path` 参数；当未传 `--risk-spec` 时该参数必填，当传入 `--risk-spec` 时不要求。`image_path` 需继续传入 `RiskScenarioRunner.run(...)`，再由 runner 传给 `RiskScenarioInterpreter`。
- **默认三速策略定死：** 对 `--risk-spec` 传入的旧 v1 spec，如果缺少 `ego.speed_hypotheses_mps`，第一版统一注入常量 fallback：

```json
{
  "low": 4.0,
  "medium": 8.0,
  "high": 12.0
}
```

  同时至少在 summary 中记录 `speed_hypotheses_source: "default_fallback"`；如实现方便，也可同步写回规范化后的 spec。暂不从模板 `ego_target_speed_mps` 范围派生默认三速，避免第一版引入额外策略分支。
- **VLM 路径强校验位置：** 不在通用 `validate_risk_scenario_spec()` 中强制所有 spec 必含 `speed_hypotheses_mps`，否则会破坏旧 `--risk-spec` 兼容。强制逻辑应只放在 VLM 新生成路径：`RiskScenarioInterpreter.call_agent()` 成功解析后、返回 payload 前，或 `RiskScenarioRunner.run()` 的非 `risk_spec_path` 分支中。
- **sample 顶层字段明确：** `sample_risk_scenario()` 仍保留 `sampled_parameters["ego_target_speed_mps"]`，同时将速度信息额外顶层透传，便于 summary 和后续分析读取：

```json
{
  "ego_speed_label": "low",
  "ego_target_speed_mps": 4.0,
  "sampled_parameters": {
    "ego_target_speed_mps": 4.0
  }
}
```

## Key Changes

- CLI 入口 `experiments/generate_risk_scenarios.py` 重新增加参数 `--image-path`，仅在未传 `--risk-spec` 时需要；`--bev-path` 不恢复。
- `RiskScenarioRunner` 向 `RiskScenarioInterpreter` 传入原图 `image_path`、`spawn_payload`、派生 `actor_context`、`scene_match`、可选 `scene_understanding`。
- `RiskScenarioInterpreter` 恢复图片编码与多模态 message，但 prompt 明确：
  - 原图用于识别主体道路参与者与估计低/中/高 ego 速度。
  - `actors.json` 中的 actor id 是唯一可执行 actor 集合。
  - 每个 risk candidate 的 `involved_actor_ids` 必须从 actor table 中选择。
  - 不使用 BEV。
- 更新 risk spec schema 约定，新增 ego 三速假设字段，例如：

```json
"ego": {
  "actor_id": "ego",
  "controller_mode": "scripted",
  "route_source": "map_forward_waypoints",
  "behavior_profile": "accident_reproduction",
  "speed_hypotheses_mps": {
    "low": 4.0,
    "medium": 8.0,
    "high": 12.0
  }
}
```

- VLM 负责给出三个递增数值；校验层在该字段 **存在时** 要求三档速度均为非负且满足 `low < medium < high`（必填范围见阻塞项 1：仅 VLM 新生成路径强制）。
- 事故候选生成按速度档展开（接入点见阻塞项 2：在 `build_artifacts` 中 validate 之后、sample 之前调用 `expand_candidates_by_speed`）：
  - VLM 先输出主体 actor 与模板候选。
  - pipeline 将每个 candidate 复制为 low/medium/high 三个速度版本（每个为 `risk_candidates` 中独立 candidate，`candidate_index` 唯一）。
  - 每个展开 candidate 把 `parameter_ranges["ego_target_speed_mps"]` 写为定值区间 `[v, v]`，借 `_normalize_range` + `sample_risk_scenario` 实现固定不随机；无需改动模板库。
  - 其他参数仍按原有 `parameter_ranges` 采样。
  - 提醒：与 `samples_per_candidate`（默认 3）叠加后，单个 VLM candidate 产出 3×samples 个 sample/script。
- sample JSON 增加 `ego_speed_label` 与 `ego_target_speed_mps`，summary/artifacts 中保留该标签，方便区分同一事故假设在三种速度下的结果。

## Implementation Notes

- 保持 `--risk-spec` 兼容：如果用户提供 risk spec，则不需要 `--image-path`；risk spec 可直接包含 `speed_hypotheses_mps`。**缺失时不报错**（否则会破坏既有 v1 spec），而是注入常量默认三档 `4/8/12 m/s` 并在 summary 标记 `speed_hypotheses_source: "default_fallback"`（见阻塞项 1）。仅 VLM 新生成路径强制该字段。
- VLM 输出仍只允许选择现有 accident template library 中的 `template_id`，不允许输出自由轨迹或 CARLA 代码。
- 主体 actor 对齐采用“VLM 选 actor id”方案：prompt 同时给原图和 actor table，要求 VLM 用 rationale 说明为什么选择该 actor。
- 若 VLM 认为原图主体无法可靠映射到 actor id，应把该风险写入 `unsupported_risk_hypotheses`，不要生成不可执行 candidate。
- 三档速度展开后的 `candidate_index` 在展开重写后天然唯一，`rXXX_sXXX.json` 不会碰撞；`source_candidate_index` 仅用于回溯原始 VLM candidate（traceability），不是文件名唯一性所必需。
- `ego_speed_label` 传播链需补全：展开阶段打标签 → `sample_risk_scenario` 透传 → `build_artifacts` 的 artifact dict 与 summary 各加该字段（见精度缺口）。
- 顶层 sample 字段需补全：`sample_risk_scenario` 除了保留 `sampled_parameters["ego_target_speed_mps"]` 外，还应写出顶层 `ego_speed_label` 与 `ego_target_speed_mps`，避免分析脚本必须深入 sampled parameters 才能识别速度档。

## Test Plan

- CLI 测试：
  - 未传 `--risk-spec` 且缺少 `--image-path` 时失败。
  - 传 `--risk-spec` 时不要求 `--image-path`。
  - `experiments/generate_risk_scenarios.py` 将 `--image-path` 传入 `RiskScenarioRunner.run()`。
- Interpreter 测试：
  - `refine_request()` 输出包含原图 image message。
  - prompt 明确禁止使用 BEV，并要求 actor id 来自 actor table。
- Pipeline 测试：
  - 校验 `speed_hypotheses_mps` 三档递增。
  - 旧 v1 spec 缺失 `speed_hypotheses_mps` 时注入常量 fallback `4/8/12 m/s`，且记录 `speed_hypotheses_source: "default_fallback"`。
  - 一个 risk candidate 展开为 low/medium/high 三个可采样 candidate。
  - `ego_target_speed_mps` 固定为速度档数值，不被随机采样覆盖。
  - sample JSON 顶层包含 `ego_speed_label` 与 `ego_target_speed_mps`，且与 `sampled_parameters["ego_target_speed_mps"]` 一致。
- Runner 测试：
  - 生成 summary/artifacts 时包含 `ego_speed_label`。
  - 每个 implemented candidate 在三档速度下各生成 sample 和脚本。
- Regression：
  - 现有 risk-spec 模式、脚本 AST parse、metrics path 生成保持可用。
  - 不含 `speed_hypotheses_mps` 的旧 v1 risk spec 经 `--risk-spec` 传入时不报错，注入默认三档并标记 `default_fallback`（验证阻塞项 1 的兼容修正）。
  - `expand_candidates_by_speed` 在 `build_artifacts` 中位于 validate 与 sample 之间，展开后 candidate_index 唯一、样本文件不碰撞（验证阻塞项 2）。

## Assumptions

- 速度三档由 VLM 根据原图建议，但最终必须是确定的 m/s 数值。
- 原图只用于主体参与者识别和速度假设，不作为最终 CARLA 几何事实来源。
- BEV 不恢复。
- 本文档位于仓库当前目录：`raw_image_three_speed_risk_plan.md`。

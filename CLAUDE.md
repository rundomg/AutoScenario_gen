# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoScenario is a multi-agent LLM-driven framework that converts real-world safety-critical data (images, videos, crash reports, natural language commands) into structured, *executable* simulation scenarios for autonomous-vehicle testing in **CARLA** (and legacy **SUMO**).

The codebase has two layers that have grown on top of each other:

1. **Static reconstruction** — turn an input image into a faithful static scene placed in a *real* CARLA world map (real-world coordinates, verified actor layout).
2. **Risk / accident generation** — take a completed static reconstruction and script a dynamic safety-critical event on top of it, emitted either as CARLA Python or as **Scenic**.

## Setup

```bash
# Python 3.8 environment recommended
conda create --name autoscenario python=3.8
conda activate autoscenario
pip install -r requirements.txt

# Configure API credentials
cp config.example .env
# Edit .env: set OPENAI_KEY, OPENAI_URL, OPENAI_MODEL, etc.
```

Scenic compile-checking runs in a **separate** conda env (default `scenicNL`, with `scenic==3.0.0b2`); `tools/scenic_compile.py` shells out via `conda run -n scenicNL python`. The autoscenario env does NOT need scenic installed.

## Running the Pipelines

### Layer 1 — static reconstruction (primary)

```bash
# Image → structured static scene in a real CARLA map.
# Entry point + input image + run mode are hardcoded in the __main__ block.
python experiments/auto_generate_all_vlm.py
```

`__main__` selects a `mode`: `FullPipeline` runs the structured reconstruction flow; `AfterInterpreter` / `AfterNet` / `AfterObject` run the **legacy** SUMO 5-section flow (see below). `verify_mode` defaults to `"actor_graph"`.

Legacy input variants (still the old SUMO-style 5-section flow):

```bash
python experiments/auto_generate_all_video.py
python experiments/auto_generate_all_text.py
python experiments/auto_generate_all_command.py
python opendrive_experiment/auto_generate_all_vlm_xodr.py   # OpenDRIVE variant
```

### Layer 2 — risk / accident generation (consumes a Layer-1 output folder)

```bash
# Pure-LLM (default): accident → risk-dsl-v1 → executable CARLA Python
python experiments/generate_risk_scenarios.py \
    --output-folder results/auto_result_YYYYMMDD_HHMMSS \
    --scene-id s0000_c0 --image-path data/001.png --num-accidents 1

# Same, but legacy template-library backend:
python experiments/generate_risk_scenarios.py ... --use-template

# Pure-LLM Scenic branch: accident → .scenic (compile-checked in scenicNL env)
python experiments/generate_pure_llm_scenic.py \
    --output-folder results/auto_result_YYYYMMDD_HHMMSS \
    --scene-id s0000_c0 --image-path data/001.png --max-retries 2
```

Layer 2 reuses `{scene_id}_actors.json` (spawn_payload / real CARLA coordinates) and `{scene_id}_match.json` (chosen world map) produced by Layer 1 — it never re-derives geometry.

## Running Tests

```bash
python -m pytest tests/                       # all tests
python -m pytest tests/test_vlm_interpreter.py   # single file
```

Tests use `unittest` with mocked external deps (cv2, numpy, requests, dotenv). No pytest config file — discovery is automatic.

## Architecture

### Agent base

All agents inherit from `TaskAgent` (`agents/task_agent.py`): LLM API calls (HTTP POST, exponential backoff, up to 3 retries), config from `.env`, file output. Agents communicate **only** through LLM prompts and JSON files on disk — no direct inter-agent calls. Most generator stages re-attempt up to 3× on validation failure (missing sections, non-compiling Python/Scenic, schema errors), with the validation error fed back into the next prompt as a repair signal.

### Layer 1: structured static reconstruction (`FullPipeline`)

Driven by `experiments/auto_generate_all_vlm.py` (`AutoGenerator`) + `tools/structured_pipeline.py`:

```
image
 → SceneUnderstandingInterpreter   → Scene Understanding DSL ({scene_id}_su.json)
 → build_relation_dsl              → relational layout (who is near/ahead/left of whom)
 → generate_initial_coordinates    → 2D coords
 → apply_pairwise_ordering         → resolve relative ordering
 → project_entities_to_carla_context → place onto a real CARLA map's lanes/waypoints
 → refine + validate_relation_layout
 → SceneMapMatcher / topology match → {scene_id}_match.json (world_name + anchor lane)
 → ExistingWorldScenarioGenerator  → {scene_id}_static.py (CARLA spawn script)
 → actor-graph verify/repair loop  → repaired spawn_payload → {scene_id}_actors.json
```

**Scene Understanding DSL** (replaces the old 5 sections) has 5 required top-level keys
(`SCENE_UNDERSTANDING_REQUIRED_KEYS` in `tools/structured_pipeline.py`):
`traffic_subjects`, `background_traffic`, `road_network`, `general_environment`, `metadata`.

**Actor-graph verification** (`tools/actor_graph_verifier.py` + `agents/scene_verification_agent.py`): builds a source graph from the scene description and a render graph from the actual spawned transforms (truth = `carla_actor_transform`, fallback = `spawn_payload_fallback`), diffs them, and emits a repair plan. Loops while `verify_mode == "actor_graph"`.

### Layer 2: risk / accident generation

Orchestrators in `tools/`, CLIs in `experiments/`. All three branches share a common Stage-1:

- **Stage 1 — `agents/llm_accident_predictor.py`**: VLM reads the original image + fixed ego speed and freely predicts plausible accidents → `{scene_id}_candidates.json`. Intentionally NOT constrained by the template library.

Then one of:

| Branch | Stage 2 agent | Intermediate | Final artifact | Compile gate |
|--------|---------------|--------------|----------------|--------------|
| Pure-LLM Python (default) | `agents/llm_risk_dsl_generator.py` | `risk-dsl-v1` JSON (`tools/risk_dsl.py`) → `ExistingWorldScenarioGenerator.build_dsl_risk_scene_script` | `{scene_id}_rNNN.py` | `tools/python_compile.py` |
| Pure-LLM Scenic | `agents/llm_scenic_generator.py` | Scenic body (header composed by runner) | `{scene_id}_candN.scenic` | `tools/scenic_compile.py` (`scenicNL` env) |
| Template (legacy) | `agents/risk_scenario_interpreter.py` | `tools/accident_template_library.py` + `tools/risk_scenario_pipeline.py` | CARLA Python | — |

Runners: `tools/pure_llm_risk_runner.py`, `tools/pure_llm_scenic_runner.py`, `tools/risk_scenario_runner.py`. The repair loop in the pure-LLM Python branch targets **Stage 2a DSL schema validation** (`validate_risk_dsl`); the generated Python is deterministic, so `check_python_compile` is a final syntax gate, not part of the loop.

Scenic placement note (see `agents/llm_scenic_generator.py` docstring): actors are placed with **ego-relative** offsets, not absolute `x @ y` — absolute placement crashes this Scenic build (3.0.0b2 + shapely 2.1.2) when it computes road direction at the point.

### Legacy SUMO pipeline (still present)

The original `Input → Interpreter → NetGenerator → ObstacleGenerator → RouteGenerator → ScenarioGenerator` flow (5 sections: Road Net / Road Users / Static Objects / Vehicles' Locations & Behaviors / Scenario Description) is reachable via the non-`FullPipeline` modes and the video/text/command experiments. Files: `agents/universal_interpreter.py`, `net_generator.py`, `obstacle_generator.py`, `rou_generator.py`, `scenario_generator.py`.

### OpenDRIVE experiment

`opendrive_experiment/` is a self-contained parallel pipeline using OpenDRIVE instead of SUMO XML, with its own `agents/`, `tools/`, `tests/`.

### Output layout

Each run writes a `results/auto_result_*/` folder. Per scene (`{scene_id}`, e.g. `s0000_c0`):
`_su.json`, `_match.json`, `_actors.json`, `_static.py`, `_topo.json`, actor-graph artifacts
(`_source_actor_graph.json`, `_render_actor_graph_rN.json`, `_actor_graph_repair_plan_rN.json`, `_verify_rN.json`),
plus Layer-2 risk/scenic artifacts (`_candidates.json`, `_rNNN.py`, `_candN.scenic`, `*_summary.json`).

## Design docs (root `*.md`)

Read these before changing the corresponding subsystem — they record decisions and trade-offs:
`vlm_pipeline_migration_plan.md` (the move off the SUMO 5-section flow), `actor_graph_verify_repair_review.md` + `verify_repair_plan_review.md` (verification loop), `raw_image_three_speed_risk_plan.md` (risk generation), `scenicnl_internalization_design.md` (Scenic; notes that pure-LLM-writes-Scenic is an experiment the memo advises against, kept to measure compile pass-rate).

## Configuration (`.env`)

Loaded by `python-dotenv` in `TaskAgent.__init__`:

```
OPENAI_KEY            # API key
OPENAI_URL            # API endpoint (supports custom/proxy endpoints)
OPENAI_MODEL          # Model name (e.g., gpt-4o)
OPENAI_MAX_TOKENS
OPENAI_TIMEOUT        # Read timeout (seconds)
OPENAI_CONNECT_TIMEOUT
OPENAI_REQUEST_RETRIES
OPENAI_SYSTEM_PROMPT
```

Other env knobs: `CARLA_MAPS_DIR` (dir of `TownXX.xodr`; default in `tools/pure_llm_scenic_runner.py`), `--scenic-conda-env` (default `scenicNL`).

Always respond in Chinese

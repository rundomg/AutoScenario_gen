#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/zx/code/AutoScenario_gen"
SUMMARY_PATH="${1:-$ROOT_DIR/data/accid_1_keyframes/keyframe_selection_summary.json}"
OUTPUT_ROOT="${2:-$ROOT_DIR/results/accid_1_static_scenes}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$ROOT_DIR"
mkdir -p "$OUTPUT_ROOT"
SUCCESS_FILE="$OUTPUT_ROOT/successful_jobs.tsv"
FAILED_FILE="$OUTPUT_ROOT/failed_jobs.tsv"
: > "$SUCCESS_FILE"
: > "$FAILED_FILE"

PLAN_FILE="$(mktemp /tmp/accid1_static_plan.XXXXXX.tsv)"
trap 'rm -f "$PLAN_FILE"' EXIT

"$PYTHON_BIN" - "$SUMMARY_PATH" "$PLAN_FILE" <<'PY'
import json
import os
import sys

summary_path, plan_path = sys.argv[1], sys.argv[2]
with open(summary_path, "r", encoding="utf-8") as handle:
    summary = json.load(handle)

rows = []
for item in summary:
    scene_id = item.get("scene_id")
    output_dir = item.get("output_dir")
    collision_second = item.get("collision_second")
    if not scene_id or output_dir is None or collision_second is None:
        print(f"[skip] incomplete summary item: {item}", file=sys.stderr)
        continue

    manifest_path = os.path.join(output_dir, "sampled_frames", "frames_manifest.json")
    if not os.path.exists(manifest_path):
        print(f"[skip] missing manifest: {manifest_path}", file=sys.stderr)
        continue

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    frames = manifest.get("frames") or []
    if not frames:
        print(f"[skip] empty frames manifest: {manifest_path}", file=sys.stderr)
        continue

    target = max(0.0, float(collision_second) - 2.0)
    chosen = min(
        frames,
        key=lambda frame: abs(float(frame.get("timestamp_s", 0.0)) - target),
    )
    image_path = chosen.get("path")
    timestamp_s = chosen.get("timestamp_s")
    if not image_path or not os.path.exists(image_path):
        print(f"[skip] missing selected image for {scene_id}: {image_path}", file=sys.stderr)
        continue

    rows.append((scene_id, str(collision_second), str(target), str(timestamp_s), image_path))

with open(plan_path, "w", encoding="utf-8") as handle:
    for row in rows:
        handle.write("\t".join(row) + "\n")

print(f"Prepared {len(rows)} scene generation jobs from {summary_path}")
PY

while IFS=$'\t' read -r scene_id collision_second target_second image_second image_path; do
  scene_output="$OUTPUT_ROOT/${scene_id}_collision${collision_second}s_static${image_second}s"
  log_path="$scene_output/run.log"
  user_input="The input image is sampled from accident video ${scene_id} about 2 seconds before the predicted accident time ${collision_second}s. Reconstruct the static traffic scene visible in this pre-accident frame."

  echo
  echo "=== $scene_id ==="
  echo "collision_second: $collision_second"
  echo "target_static_second: $target_second"
  echo "selected_image_second: $image_second"
  echo "image_path: $image_path"
  echo "output_folder: $scene_output"

  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi

  mkdir -p "$scene_output"
  set +e
  "$PYTHON_BIN" experiments/auto_generate_all_vlm.py \
    --image-path "$image_path" \
    --output-folder "$scene_output" \
    --user-input "$user_input" 2>&1 | tee "$log_path"
  status=${PIPESTATUS[0]}
  set -e

  if [[ "$status" -eq 0 ]]; then
    echo -e "${scene_id}\t${collision_second}\t${image_second}\t${scene_output}" >> "$SUCCESS_FILE"
    echo "SUCCESS: $scene_id"
  else
    echo -e "${scene_id}\t${collision_second}\t${image_second}\t${status}\t${scene_output}\t${log_path}" >> "$FAILED_FILE"
    echo "FAILED: $scene_id (exit code $status). Continuing with next video."
  fi
done < "$PLAN_FILE"

echo
success_count=$(wc -l < "$SUCCESS_FILE")
failed_count=$(wc -l < "$FAILED_FILE")
echo "All requested scene generation jobs attempted. Output root: $OUTPUT_ROOT"
echo "Successful jobs: $success_count ($SUCCESS_FILE)"
echo "Failed jobs: $failed_count ($FAILED_FILE)"

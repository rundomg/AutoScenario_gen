# Monocular Depth KITTI Vehicle Distance Evaluation

This experiment evaluates monocular metric-depth models on KITTI Object vehicle
distance estimation. The active target source is YOLO instance segmentation:
each vehicle distance is sampled from a predicted instance mask, then compared
against KITTI 3D labels and projected LiDAR.

KITTI labels are still required as ground truth. They are not used as a
label-box sampling baseline in the current pipeline.

## Data

Expected local dataset:

```text
KITTI/kitti/training/image_2
KITTI/kitti/training/calib
KITTI/kitti/training/label_2
KITTI/kitti/training/velodyne
```

Only KITTI `training` is used because KITTI Object `testing` has no public
labels.

## Result Layout

Default outputs are grouped by purpose:

```text
experiments/monocular_depth_eval/results/
  runs/<depth-model>/yolo-seg/<N>_masked/
  smoke/yolo-seg/<N>_geometry/
  visuals/yolo-seg/
  config/ultralytics/
  legacy/
```

Old label-box and bbox-only result folders are kept under `results/legacy/`
only for traceability. They are no longer part of the active workflow.

## Quick Checks

Run geometry-only smoke first. This validates KITTI parsing, calibration,
LiDAR projection, YOLO instance masks, CSV writing, and overlays without
loading a depth model:

```sh
python experiments/monocular_depth_eval/evaluate_kitti_depth.py ^
  --model none ^
  --max-images 20
```

Default output:

```text
experiments/monocular_depth_eval/results/smoke/yolo-seg/20_geometry/
```

## Unlabeled Image Folder Inference

For deployment-style images without labels, use `process_image_folder.py`.
It detects vehicle instances with YOLO segmentation, predicts depth, samples
vehicle distance from each instance mask, and writes outputs under
`<input_dir>/output/` by default.

```sh
python experiments/monocular_depth_eval/process_image_folder.py ^
  H:\path\to\images ^
  --model metric3d ^
  --metric3d-offline ^
  --vehicle-classes car,bus,truck ^
  --yolo-seg-conf 0.15 ^
  --yolo-seg-imgsz 1280 ^
  --yolo-seg-max-count 50 ^
  --save-depth-vis
```

Output layout:

```text
<input_dir>/output/
  predictions.csv
  summary.json
  overlays/
  depth_vis/
  depth_npy/        # only when --save-depth-npy is set
```

Because these images have no labels, the script reports predicted vehicle
distances only; it does not compute accuracy metrics. If the camera focal
length is known, pass it with `--focal-length-px`. Otherwise the script uses
`--focal-ratio 0.58` as a fallback, so absolute metric scale should be treated
as approximate.

By default the detector keeps all COCO vehicle classes: bicycle, car,
motorcycle, bus, and truck. For car-only driving scenarios, set
`--vehicle-classes car,bus,truck`.

For distant small vehicles, lower the segmentation threshold and raise the
instance cap. `--yolo-seg-max-count 0` means no cap after deduplication:

```sh
--yolo-seg-conf 0.10 --yolo-seg-imgsz 1280 --yolo-seg-max-count 0
```

## Depth Anything V2 Metric

Install model dependencies if needed:

```sh
pip install transformers pillow
```

Run Depth Anything V2 Metric Outdoor through HuggingFace:

```sh
python experiments/monocular_depth_eval/evaluate_kitti_depth.py ^
  --model depth-anything-v2-hf ^
  --hf-model depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf ^
  --max-images 500
```

After the model has been downloaded once, add `--hf-offline`:

```sh
python experiments/monocular_depth_eval/evaluate_kitti_depth.py ^
  --model depth-anything-v2-hf ^
  --hf-offline ^
  --max-images 500
```

Default output:

```text
experiments/monocular_depth_eval/results/runs/depth-anything-v2-hf/yolo-seg/500_masked/
```

## Metric3D V2

Metric3D runs through the official Torch Hub entrypoint. The default comparison
uses `metric3d_vit_small`, which is Metric3D V2 and keeps runtime reasonable on
the KITTI subset.

```sh
python experiments/monocular_depth_eval/evaluate_kitti_depth.py ^
  --model metric3d ^
  --metric3d-model metric3d_vit_small ^
  --max-images 500
```

After the Torch Hub repo and checkpoint have been cached, add
`--metric3d-offline`:

```sh
python experiments/monocular_depth_eval/evaluate_kitti_depth.py ^
  --model metric3d ^
  --metric3d-model metric3d_vit_small ^
  --metric3d-offline ^
  --max-images 500
```

Default output:

```text
experiments/monocular_depth_eval/results/runs/metric3d/yolo-seg/500_masked/
```

## UniDepthV2

UniDepth runs through the official Torch Hub entrypoint. The default comparison
uses the small V2 backbone, `vits14`.

```sh
python experiments/monocular_depth_eval/evaluate_kitti_depth.py ^
  --model unidepth ^
  --unidepth-version v2 ^
  --unidepth-backbone vits14 ^
  --max-images 500
```

After the repo and checkpoint have been cached, add `--unidepth-offline`:

```sh
python experiments/monocular_depth_eval/evaluate_kitti_depth.py ^
  --model unidepth ^
  --unidepth-version v2 ^
  --unidepth-backbone vits14 ^
  --unidepth-offline ^
  --max-images 500
```

Default output:

```text
experiments/monocular_depth_eval/results/runs/unidepth/yolo-seg/500_masked/
```

The adapter installs small inference-only compatibility shims for optional
training/logging dependencies (`mmcv`, `wandb`) so the Windows evaluation path
does not require the full training stack.

## Instance-Mask ROI

The segmentation wrapper lives in `instance_segmenter.py`. It uses
`yolo11m-seg.pt` by default, keeps COCO vehicle instances, and matches each
predicted instance back to one KITTI vehicle label by IoU.

For each matched vehicle:

- build a base ROI from the YOLO-seg bbox
- intersect the ROI with the vehicle instance mask to remove bbox background
- subtract closer occluding vehicles, preferring their instance masks
- apply inward inset and quantile clipping
- take the median predicted depth inside the final visible mask

Important defaults:

```sh
--roi lower-half
--roi-inset-ratio 0.06
--depth-quantile-low 10
--depth-quantile-high 90
--min-visible-ratio 0.05
--yolo-seg-model yolo11m-seg.pt
--yolo-seg-conf 0.25
--yolo-seg-imgsz None
--yolo-seg-max-count 12
--match-iou 0.5
```

Low visible-area objects are kept by default with
`sampling_status=low_visible_area` or `no_visible_pixels`. Add
`--skip-low-visible` only when you intentionally want to exclude them.

## CSV Fields

Each row in `metrics.csv` represents one YOLO-seg/KITTI matched vehicle:

- frame id, class, occlusion/truncation, YOLO-seg bbox
- segmentation confidence and matched-label IoU
- KITTI 3D label location and label depth
- LiDAR median depth and number of LiDAR points in the final mask
- predicted median depth, absolute error, and relative error
- visible ROI ratio, occluder overlap ratio, ROI pixel counts, quantile depth
  stats, and sampling status
- instance-mask diagnostics: `mask_source`, `foreground_roi_ratio`,
  `background_removed_ratio`, `instance_mask_pixels`, and `seg_conf`

`summary.json` reports MAE, RMSE, AbsRel, distance-bin errors,
occlusion-group errors, overlap-ratio errors, sampling-status counts, and
mask-source counts.

## Visual Summaries

Generate visuals for one or more YOLO-seg result runs:

```sh
python experiments/monocular_depth_eval/make_visuals.py
```

For multiple depth models, repeat `--run`:

```sh
python experiments/monocular_depth_eval/make_visuals.py ^
  --run experiments/monocular_depth_eval/results/runs/depth-anything-v2-hf/yolo-seg/500_masked ^
  --run experiments/monocular_depth_eval/results/runs/metric3d/yolo-seg/500_masked ^
  --run experiments/monocular_depth_eval/results/runs/unidepth/yolo-seg/500_masked
```

Outputs go to:

```text
experiments/monocular_depth_eval/results/visuals/yolo-seg/
```

## Current 500-Frame Comparison

The latest local run compares all three models on the same first 500 KITTI
training frames, with YOLO instance masks as the only target source:

```text
Depth Anything V2: MAE 3.294m, RMSE 5.249m, AbsRel 0.1223
Metric3D V2 small: MAE 2.656m, RMSE 3.819m, AbsRel 0.1100
UniDepthV2 vits14: MAE 3.107m, RMSE 4.063m, AbsRel 0.1355
```

In this run, Metric3D V2 small gives the lowest MAE, RMSE, and AbsRel against
KITTI 3D label depth.

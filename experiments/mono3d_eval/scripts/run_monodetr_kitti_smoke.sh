#!/usr/bin/env bash
# MonoDETR KITTI smoke test 运行模板。
# 需要先完成 setup_monodetr_wsl.sh，并在 monodetr conda 环境中执行。

set -euo pipefail

MONODETR_DIR="${MONODETR_DIR:-$HOME/monodetr_workspace/MonoDETR}"
PROJECT_DIR="${PROJECT_DIR:-/mnt/h/AutoScenario_gen}"
KITTI_ROOT="$PROJECT_DIR/KITTI/kitti"
FRAME_LIST="$PROJECT_DIR/experiments/mono3d_eval/ImageSets/training_smoke_5.txt"

cd "$MONODETR_DIR"

# MonoDETR 配置中的 dataset/root_dir 需要指向 $KITTI_ROOT。
# checkpoint 路径需要按实际下载位置填入 configs/monodetr.yaml 的 tester/checkpoint。
echo "KITTI root: $KITTI_ROOT"
echo "Frame list: $FRAME_LIST"
echo "Edit configs/monodetr.yaml so dataset/root_dir and tester/checkpoint are correct, then run:"
echo "  bash test.sh configs/monodetr.yaml"
echo "After prediction txt files are generated, evaluate them from Windows with:"
echo "  python experiments/mono3d_eval/evaluate_monodetr_predictions.py --prediction-dir <MonoDETR-output-label-dir> --frame-list experiments/mono3d_eval/ImageSets/training_smoke_5.txt"


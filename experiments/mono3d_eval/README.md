# MonoDETR 单目 3D 车辆检测实验

这个实验目录负责 **MonoDETR 的输入准备、结果解析、轻量评估和可视化**。MonoDETR
本体建议安装在 WSL Ubuntu 的隔离环境中，避免污染当前 Windows Python 3.13 / Torch 2.9 环境。

## 当前状态

- 本地已有 KITTI Object 数据：`KITTI/kitti/training/{image_2,calib,label_2}`。
- WSL 命令存在，但当前未列出 Ubuntu 发行版；安装 Ubuntu 时出现 `0x80072f7d`，疑似系统安装源或网络问题。
- 本目录已提供项目侧工具，等 MonoDETR 预测 txt 生成后即可解析、评估和可视化。

## 1. 准备 KITTI 子集

```powershell
python experiments\mono3d_eval\prepare_kitti_subset.py --smoke-count 5 --eval-count 50
```

输出：

```text
experiments/mono3d_eval/ImageSets/training_smoke_5.txt
experiments/mono3d_eval/ImageSets/training_eval_50.txt
```

## 2. 在 WSL Ubuntu 中准备 MonoDETR

先解决 Ubuntu 安装问题，然后在 WSL 内执行：

```bash
cd /mnt/h/AutoScenario_gen
bash experiments/mono3d_eval/scripts/setup_monodetr_wsl.sh
```

然后按脚本提示安装 requirements、编译 deformable attention，并把
`configs/monodetr.yaml` 中的 `dataset/root_dir` 指向：

```text
/mnt/h/AutoScenario_gen/KITTI/kitti
```

## 3. 运行 MonoDETR 预测

在 WSL 的 MonoDETR conda 环境中运行：

```bash
cd /mnt/h/AutoScenario_gen
bash experiments/mono3d_eval/scripts/run_monodetr_kitti_smoke.sh
```

该脚本是模板，会提示需要修改的 config 和 checkpoint 位置。MonoDETR 生成的结果应为
每帧一个 KITTI 格式 txt。

## 4. 解析、评估和可视化预测结果

```powershell
python experiments\mono3d_eval\evaluate_monodetr_predictions.py ^
  --prediction-dir H:\path\to\monodetr\prediction_txt ^
  --frame-list experiments\mono3d_eval\ImageSets\training_smoke_5.txt ^
  --output-dir experiments\mono3d_eval\results\monodetr_kitti_smoke ^
  --save-visuals 5
```

输出：

```text
experiments/mono3d_eval/results/monodetr_kitti_smoke/
  predictions.csv
  predictions.json
  matches.csv
  summary.json
  visuals/image_overlay/
  visuals/bev/
```

## 输出字段

每个预测对象会输出：

```text
frame_id, class, score, bbox_2d, dimensions_hwl,
location_xyz, depth_m, distance_m, rotation_y, alpha
```

其中 `rotation_y` 是 KITTI 相机坐标系下的 yaw，后续接入主程序时需要再转换到 ego/world 坐标。


# MonoCon 当前主环境运行说明

当前主环境已经验证：

- `Python 3.13.7`
- `torch 2.9.1+cu130`
- CUDA 可用，GPU 为 `NVIDIA GeForce RTX 5080`
- MonoCon 代码可以在主环境中导入并初始化

缺少的 `yacs` 已安装到项目本地：

```powershell
python -m pip install yacs==0.1.8 --target experiments\mono3d_eval\.vendor
```

## 需要准备的权重

MonoCon 原仓库的预训练权重在 README 的 Google Drive 链接中。把 `.pth` 放到例如：

```text
experiments/mono3d_eval/checkpoints/monocon_pretrained.pth
```

`*.pth` 已加入 `.gitignore`，不会误提交大权重文件。

## 5 张 KITTI smoke test

```powershell
powershell -ExecutionPolicy Bypass -File experiments\mono3d_eval\scripts\run_monocon_current_env_smoke.ps1 `
  -CheckpointFile experiments\mono3d_eval\checkpoints\monocon_pretrained.pth
```

输出目录：

```text
experiments/mono3d_eval/results/monocon_current_env_smoke/
  prediction_txt/
  predictions.csv
  predictions.json
  matches.csv
  summary.json
  visuals/image_overlay/
  visuals/bev/
```

## 直接调用导出脚本

```powershell
python experiments\mono3d_eval\export_monocon_predictions.py `
  --checkpoint-file experiments\mono3d_eval\checkpoints\monocon_pretrained.pth `
  --config-file experiments\mono3d_eval\configs\monocon_current_env.yaml `
  --frame-list experiments\mono3d_eval\ImageSets\training_smoke_5.txt `
  --prediction-dir experiments\mono3d_eval\results\monocon_current_env_smoke\prediction_txt `
  --eval-output-dir experiments\mono3d_eval\results\monocon_current_env_smoke
```

导出的 KITTI txt 会转换 MonoCon 内部尺寸顺序：`length/height/width` -> KITTI 文本的 `height/width/length`。

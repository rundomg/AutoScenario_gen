param(
    [Parameter(Mandatory = $true)]
    [string]$CheckpointFile,

    [string]$ConfigFile = "experiments\mono3d_eval\configs\monocon_current_env.yaml",
    [string]$KittiRoot = "H:\AutoScenario_gen\KITTI\kitti",
    [string]$FrameList = "experiments\mono3d_eval\ImageSets\training_smoke_5.txt",
    [string]$OutputDir = "experiments\mono3d_eval\results\monocon_current_env_smoke",
    [int]$GpuId = 0
)

$ErrorActionPreference = "Stop"

# Use the current python.exe and add only project-local dependencies.
$ProjectRoot = (Resolve-Path ".").Path
$VendorDir = Join-Path $ProjectRoot "experiments\mono3d_eval\.vendor"
$MonoConRoot = Join-Path $ProjectRoot "experiments\mono3d_eval\external\monocon-pytorch"
$env:PYTHONPATH = "$VendorDir;$MonoConRoot;$env:PYTHONPATH"

if (!(Test-Path -LiteralPath $CheckpointFile)) {
    throw "MonoCon checkpoint not found: $CheckpointFile"
}

$PredictionDir = Join-Path $OutputDir "prediction_txt"

# Refresh small KITTI frame lists before running smoke tests.
python experiments\mono3d_eval\prepare_kitti_subset.py --smoke-count 5 --eval-count 50

# Run MonoCon inference and reuse the local KITTI txt evaluation/visualization tools.
python experiments\mono3d_eval\export_monocon_predictions.py `
    --checkpoint-file $CheckpointFile `
    --config-file $ConfigFile `
    --kitti-root $KittiRoot `
    --frame-list $FrameList `
    --prediction-dir $PredictionDir `
    --eval-output-dir $OutputDir `
    --gpu-id $GpuId `
    --batch-size 1 `
    --num-workers 0 `
    --save-visuals 5

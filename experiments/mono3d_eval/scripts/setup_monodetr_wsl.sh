#!/usr/bin/env bash
# MonoDETR WSL 环境准备脚本。
# 这个脚本应在 Ubuntu/WSL 内执行，不要在当前 Windows Python 环境里跑。

set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/monodetr_workspace}"
MONODETR_DIR="$WORKDIR/MonoDETR"

mkdir -p "$WORKDIR"
cd "$WORKDIR"

if [ ! -d "$MONODETR_DIR" ]; then
  git clone https://github.com/ZrrSkywalker/MonoDETR.git "$MONODETR_DIR"
fi

cd "$MONODETR_DIR"

# 官方 README 使用 Python 3.8 和 torch 1.9/cu111；这里使用 conda 隔离环境。
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda/Anaconda inside WSL first." >&2
  exit 2
fi

conda env list | grep -q '^monodetr ' || conda create -y -n monodetr python=3.8

echo "Activate manually before continuing:"
echo "  conda activate monodetr"
echo "Then install PyTorch/requirements and compile:"
echo "  pip install -r requirements.txt"
echo "  cd lib/models/monodetr/ops && bash make.sh"


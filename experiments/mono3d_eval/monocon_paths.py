"""MonoCon 路径适配。

这个模块只负责把第三方 MonoCon 仓库和本地 vendor 目录加入 sys.path。
这样主环境可以直接运行，同时不把依赖写进全局 Python。
"""

from __future__ import annotations

import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
DEFAULT_KITTI_ROOT = PROJECT_ROOT / "KITTI" / "kitti"
MONOCON_ROOT = EXPERIMENT_DIR / "external" / "monocon-pytorch"
VENDOR_DIR = EXPERIMENT_DIR / ".vendor"


def add_monocon_to_path() -> None:
    """把本地依赖和 MonoCon 仓库放到 import 搜索路径最前面。"""

    for path in (VENDOR_DIR, MONOCON_ROOT):
        if not path.exists():
            continue
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def require_monocon_repo() -> None:
    """提前给出清楚错误，避免后面 import 报一串难读堆栈。"""

    if not MONOCON_ROOT.exists():
        raise FileNotFoundError(
            f"MonoCon repo not found: {MONOCON_ROOT}. "
            "Expected experiments/mono3d_eval/external/monocon-pytorch."
        )

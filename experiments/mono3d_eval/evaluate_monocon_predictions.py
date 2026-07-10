"""MonoCon 预测结果评估入口。

评估逻辑本身是通用 KITTI txt 解析；这里保留一个 MonoCon 命名的入口，
避免后续命令里还出现 MonoDETR 字样。
"""

from __future__ import annotations

try:
    from .evaluate_monodetr_predictions import main
except ImportError:  # 允许直接执行脚本
    from evaluate_monodetr_predictions import main


if __name__ == "__main__":
    raise SystemExit(main())

"""Inference-only LAVIS bootstrap for LMDrive.

The upstream package imports training datasets (decord/webdataset) at package
import time. LMDrive evaluation only needs model registration, so skip those
optional training dependencies while keeping the original package modules.
"""

import os
import sys

from omegaconf import OmegaConf

_REAL_ROOT = "/home/zx/code/LMDrive/LAVIS/lavis"
__path__ = [os.path.dirname(__file__), _REAL_ROOT]

from lavis.common.registry import registry  # noqa: E402
from lavis.models.drive_models.drive import Blip2VicunaDrive  # noqa: E402,F401

default_cfg = OmegaConf.load(os.path.join(_REAL_ROOT, "configs", "default.yaml"))
registry.register_path("library_root", _REAL_ROOT)
repo_root = os.path.join(_REAL_ROOT, "..")
registry.register_path("repo_root", repo_root)
registry.register_path("cache_root", os.path.join(repo_root, default_cfg.env.cache_root))
registry.register("MAX_INT", sys.maxsize)
registry.register("SPLIT_NAMES", ["train", "val", "test"])

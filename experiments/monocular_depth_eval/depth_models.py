"""Depth-model adapters for KITTI vehicle distance evaluation.

Each adapter returns a metric depth map in the original image resolution. The
evaluator owns target matching and ROI sampling; this module only hides the
model-specific loading, preprocessing, and scale conventions.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_TMP = PROJECT_ROOT / ".cache" / "tmp"
LOCAL_APPDATA = PROJECT_ROOT / ".cache" / "appdata"
LOCAL_TMP.mkdir(parents=True, exist_ok=True)
LOCAL_APPDATA.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(LOCAL_TMP)
os.environ["TEMP"] = str(LOCAL_TMP)
os.environ["TMP"] = str(LOCAL_TMP)
os.environ["LOCALAPPDATA"] = str(LOCAL_APPDATA)
os.environ["APPDATA"] = str(LOCAL_APPDATA)
os.environ["XDG_CACHE_HOME"] = str(LOCAL_APPDATA)
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))

DEFAULT_HF_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
DEFAULT_METRIC3D_MODEL = "metric3d_vit_small"
DEFAULT_UNIDEPTH_VERSION = "v2"
DEFAULT_UNIDEPTH_BACKBONE = "vits14"


class DepthModel:
    name = "base"

    def predict_depth(
        self,
        image_path: Path,
        image_bgr: np.ndarray,
        calib: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        raise NotImplementedError


class DepthAnythingV2HF(DepthModel):
    """Depth Anything V2 metric model through HuggingFace transformers."""

    name = "depth-anything-v2-hf"

    def __init__(self, model_name: str, device: str = "auto", offline: bool = False) -> None:
        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            import torch
            from PIL import Image
            from transformers import pipeline
        except Exception as exc:  # noqa: BLE001 - surface actionable setup text
            raise RuntimeError(
                "Depth Anything V2 HF mode requires `transformers`, `Pillow`, "
                "and `torch`. Install transformers or run with `--model none` "
                "for geometry-only smoke tests."
            ) from exc

        self._torch = torch
        self._image_cls = Image
        self.model_name = model_name
        self.device = _resolve_hf_pipeline_device(device, torch)
        self.offline = offline
        self.pipe = pipeline("depth-estimation", model=model_name, device=self.device)

    def predict_depth(
        self,
        image_path: Path,
        image_bgr: np.ndarray,
        calib: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        image = self._image_cls.open(image_path).convert("RGB")
        target_height, target_width = image_bgr.shape[:2]
        output = self.pipe(image)
        pred = output.get("predicted_depth")
        if pred is None:
            raise RuntimeError("Depth pipeline did not return `predicted_depth`.")

        tensor = pred.float()
        while tensor.ndim < 4:
            tensor = tensor.unsqueeze(0)
        tensor = self._torch.nn.functional.interpolate(
            tensor,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        return tensor[0, 0].detach().cpu().numpy().astype(np.float32)


class Metric3DTorchHub(DepthModel):
    """Metric3D/Metric3Dv2 adapter using the official Torch Hub entrypoint."""

    name = "metric3d"

    def __init__(
        self,
        model_name: str = DEFAULT_METRIC3D_MODEL,
        device: str = "auto",
        offline: bool = False,
    ) -> None:
        try:
            import torch
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Metric3D requires `torch`.") from exc

        _patch_platformdirs_cache()
        _install_mmcv_compat_shim()
        self._torch = torch
        self.model_name = model_name
        self.device = _resolve_torch_device(device, torch)
        repo = _torch_hub_repo_or_local("yvanyin_metric3d_main", "yvanyin/metric3d", offline)
        source = "local" if Path(repo).exists() else "github"
        try:
            self.model = torch.hub.load(repo, model_name, pretrain=True, source=source, trust_repo=True)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to load Metric3D. The adapter uses the official Torch Hub "
                "entrypoint and needs the pretrained checkpoint. Re-run without "
                "`--metric3d-offline` once if the checkpoint is not cached."
            ) from exc
        self.model.to(self.device).eval()

    def predict_depth(
        self,
        image_path: Path,
        image_bgr: np.ndarray,
        calib: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        if calib is None or "P2" not in calib:
            raise RuntimeError("Metric3D requires KITTI P2 intrinsics for metric scale correction.")

        torch = self._torch
        rgb_origin = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        target_height, target_width = rgb_origin.shape[:2]
        input_size = _metric3d_input_size(self.model_name)
        scale = min(input_size[0] / target_height, input_size[1] / target_width)
        resized = cv2.resize(
            rgb_origin,
            (int(target_width * scale), int(target_height * scale)),
            interpolation=cv2.INTER_LINEAR,
        )

        # Metric3D predicts in canonical-camera space; the official example
        # rescales by the resized focal length divided by the 1000px canonical focal.
        fx = float(calib["P2"][0, 0]) * scale
        canonical_to_real_scale = fx / 1000.0

        padding_color = [123.675, 116.28, 103.53]
        pad_h = input_size[0] - resized.shape[0]
        pad_w = input_size[1] - resized.shape[1]
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        padded = cv2.copyMakeBorder(
            resized,
            pad_h_half,
            pad_h - pad_h_half,
            pad_w_half,
            pad_w - pad_w_half,
            cv2.BORDER_CONSTANT,
            value=padding_color,
        )

        mean = torch.tensor(padding_color, dtype=torch.float32, device=self.device)[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32, device=self.device)[:, None, None]
        tensor = torch.from_numpy(padded.transpose(2, 0, 1)).float().to(self.device)
        tensor = ((tensor - mean) / std).unsqueeze(0)

        with torch.no_grad():
            pred_depth, _confidence, _output_dict = self.model.inference({"input": tensor})

        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[
            pad_h_half : pred_depth.shape[0] - (pad_h - pad_h_half),
            pad_w_half : pred_depth.shape[1] - (pad_w - pad_w_half),
        ]
        pred_depth = torch.nn.functional.interpolate(
            pred_depth[None, None, :, :],
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        pred_depth = torch.clamp(pred_depth * canonical_to_real_scale, 0, 300)
        return pred_depth.detach().cpu().numpy().astype(np.float32)


class UniDepthV2TorchHub(DepthModel):
    """UniDepthV2 adapter using the official Torch Hub entrypoint."""

    name = "unidepth"

    def __init__(
        self,
        version: str = DEFAULT_UNIDEPTH_VERSION,
        backbone: str = DEFAULT_UNIDEPTH_BACKBONE,
        device: str = "auto",
        offline: bool = False,
    ) -> None:
        try:
            import torch
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("UniDepth requires `torch`.") from exc

        _install_wandb_compat_shim()
        self._torch = torch
        self.version = version
        self.backbone = backbone
        self.device = _resolve_torch_device(device, torch)
        repo = _torch_hub_repo_or_local("lpiccinelli-eth_UniDepth_main", "lpiccinelli-eth/UniDepth", offline)
        source = "local" if Path(repo).exists() else "github"
        try:
            self.model = torch.hub.load(
                repo,
                "UniDepth",
                version=version,
                backbone=backbone,
                pretrained=True,
                source=source,
                trust_repo=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to load UniDepthV2. It needs the official UniDepth repo, "
                "its Python dependencies, and the HuggingFace checkpoint. Re-run "
                "without `--unidepth-offline` once if they are not cached."
            ) from exc
        self.model.to(self.device).eval()

    def predict_depth(
        self,
        image_path: Path,
        image_bgr: np.ndarray,
        calib: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        torch = self._torch
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device)
        camera = self._unidepth_camera(calib)
        with torch.no_grad():
            predictions = self.model.infer(tensor, camera) if camera is not None else self.model.infer(tensor)
        depth = predictions["depth"]
        depth = depth.squeeze()
        return depth.detach().cpu().numpy().astype(np.float32)

    def _unidepth_camera(self, calib: dict[str, np.ndarray] | None) -> Any | None:
        if calib is None or "P2" not in calib:
            return None
        try:
            from unidepth.utils.camera import Pinhole
        except Exception:  # noqa: BLE001 - fall back to UniDepth self-prompted camera
            return None
        p2 = calib["P2"]
        intrinsics = self._torch.tensor(
            [[p2[0, 0], 0.0, p2[0, 2]], [0.0, p2[1, 1], p2[1, 2]], [0.0, 0.0, 1.0]],
            dtype=self._torch.float32,
            device=self.device,
        )
        return Pinhole(K=intrinsics)


def build_depth_model(args) -> DepthModel | None:
    if args.model == "none":
        return None
    if args.model == "depth-anything-v2-hf":
        return DepthAnythingV2HF(args.hf_model, args.device, args.hf_offline)
    if args.model == "metric3d":
        return Metric3DTorchHub(args.metric3d_model, args.device, args.metric3d_offline)
    if args.model == "unidepth":
        return UniDepthV2TorchHub(args.unidepth_version, args.unidepth_backbone, args.device, args.unidepth_offline)
    raise ValueError(f"Unsupported model: {args.model}")


def _resolve_hf_pipeline_device(device: str, torch_module) -> int:
    if device == "auto":
        return 0 if torch_module.cuda.is_available() else -1
    if device == "cpu":
        return -1
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        try:
            return int(device.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device: {device}") from exc
    raise ValueError("device must be one of: auto, cpu, cuda, cuda:<index>")


def _resolve_torch_device(device: str, torch_module):
    if device == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    return torch_module.device(device)


def _torch_hub_repo_or_local(local_dir_name: str, github_repo: str, offline: bool) -> str:
    local_path = PROJECT_ROOT / ".cache" / "torch" / "hub" / local_dir_name
    if local_path.exists():
        return str(local_path)
    if offline:
        raise RuntimeError(f"Torch Hub repo is not cached locally: {local_path}")
    return github_repo


def _metric3d_input_size(model_name: str) -> tuple[int, int]:
    if "convnext" in model_name:
        return 544, 1216
    return 616, 1064


def _patch_platformdirs_cache() -> None:
    """Keep third-party import caches inside the workspace on locked-down Windows.

    Metric3D imports mmengine, which imports YAPF. YAPF uses platformdirs to
    create a grammar pickle under the user cache; in this sandbox that path can
    hang while creating a temporary file. Patching the function before mmengine
    imports keeps that cache local to the experiment.
    """

    try:
        import platformdirs
    except Exception:  # noqa: BLE001
        return

    cache_root = LOCAL_APPDATA / "platformdirs"
    cache_root.mkdir(parents=True, exist_ok=True)

    def user_cache_dir(appname=None, appauthor=None, version=None, *args, **kwargs):  # noqa: ANN001
        parts = [str(value) for value in (appauthor, appname, version) if value]
        path = cache_root.joinpath(*parts) if parts else cache_root
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    platformdirs.user_cache_dir = user_cache_dir


def _install_mmcv_compat_shim() -> None:
    """Provide the tiny mmcv surface Metric3D imports during inference.

    The cached Metric3D repo can use `mmengine.Config`, but one utility module
    still imports `mmcv.utils.collect_env`. Installing full mmcv is heavy on
    Windows, and this evaluator never calls Metric3D training utilities, so a
    minimal shim keeps inference dependency-light.
    """

    if "mmcv.utils" in sys.modules:
        return

    utils_mod = types.ModuleType("mmcv.utils")

    def collect_env() -> dict[str, str]:
        return {}

    def get_git_hash() -> str:
        return "unknown"

    utils_mod.collect_env = collect_env
    utils_mod.get_git_hash = get_git_hash
    mmcv_mod = types.ModuleType("mmcv")
    mmcv_mod.utils = utils_mod
    sys.modules.setdefault("mmcv", mmcv_mod)
    sys.modules["mmcv.utils"] = utils_mod


def _install_wandb_compat_shim() -> None:
    """Avoid pulling UniDepth training-log dependencies into inference."""

    if "wandb" in sys.modules:
        return

    wandb_mod = types.ModuleType("wandb")

    class Image:  # noqa: D401 - mirrors the small constructor surface used by UniDepth
        """Placeholder for wandb.Image in unused training visualization paths."""

        def __init__(self, data):
            self.data = data

    def log(*args, **kwargs):  # noqa: ANN001
        return None

    wandb_mod.Image = Image
    wandb_mod.log = log
    sys.modules["wandb"] = wandb_mod

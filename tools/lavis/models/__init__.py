"""Expose only the LMDrive model instead of importing every LAVIS model."""

__path__ = ["/home/zx/code/LMDrive/LAVIS/lavis/models"]

from lavis.models.base_model import BaseModel  # noqa: E402,F401

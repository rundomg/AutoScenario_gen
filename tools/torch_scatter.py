"""Minimal torch_scatter compatibility used by LMDrive PointPillar.

The refined batch uses Python 3.10/CARLA 0.9.16, for which the machine has no
matching torch-scatter binary. These two dim=0 operations are the only APIs
LMDrive uses and map directly to native PyTorch scatter reductions.
"""

import torch


def _expanded_index(index, source):
    shape = [index.shape[0]] + [1] * (source.dim() - 1)
    return index.reshape(shape).expand_as(source)


def scatter_mean(source, index, dim=0, dim_size=None):
    if dim != 0:
        raise NotImplementedError("LMDrive compatibility supports dim=0 only")
    size = int(index.max().item()) + 1 if dim_size is None and index.numel() else int(dim_size or 0)
    output = torch.zeros((size,) + tuple(source.shape[1:]), dtype=source.dtype, device=source.device)
    expanded = _expanded_index(index, source)
    output.scatter_add_(0, expanded, source)
    counts = torch.zeros(size, dtype=source.dtype, device=source.device)
    counts.scatter_add_(0, index, torch.ones_like(index, dtype=source.dtype))
    counts = counts.clamp_min_(1)
    return output / counts.reshape((size,) + (1,) * (source.dim() - 1))


def scatter_max(source, index, dim=0, dim_size=None):
    if dim != 0:
        raise NotImplementedError("LMDrive compatibility supports dim=0 only")
    size = int(index.max().item()) + 1 if dim_size is None and index.numel() else int(dim_size or 0)
    output = torch.full(
        (size,) + tuple(source.shape[1:]),
        torch.finfo(source.dtype).min,
        dtype=source.dtype,
        device=source.device,
    )
    output.scatter_reduce_(0, _expanded_index(index, source), source, reduce="amax", include_self=True)
    return output, None

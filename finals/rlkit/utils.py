"""Small tensor-dict helpers shared by the buffer, the pool and the rollout."""
from __future__ import annotations

import torch


def slice_rows(d, rows):
    """Slice every tensor in a dict along dim 0 (non-tensors pass through)."""
    return {k: (v[rows] if torch.is_tensor(v) else v) for k, v in d.items()}


def to_device(d, device, non_blocking=False):
    """Move every tensor in a dict to ``device``.

    On a device the tensors already live on this is a no-op ALIAS, not a copy --
    which is exactly why anything stored into a rollout buffer must be a freshly
    allocated tensor rather than a view of mutable env state. A batched env that
    hands back one of its own buffers (a validity mask, an id table) and later
    overwrites it in place would otherwise silently rewrite history in every
    stored transition. ``.clone()`` such tensors where you build them.
    """
    return {k: (v.to(device, non_blocking=non_blocking) if torch.is_tensor(v) else v)
            for k, v in d.items()}


def write_rows(dest, rows, src, full_batch=False):
    """Scatter one opponent's action/extra dict into the full-batch dict.

    ``src`` is either row-sized (a net policy that ran on a sliced observation) or
    full-batch (a scripted bot that found it cheaper to compute every game and let
    us keep only its rows) -- ``full_batch`` says which.
    """
    for k, v in src.items():
        if not torch.is_tensor(v):
            continue
        if k not in dest:
            raise KeyError(f"opponent produced key {k!r} which the task's "
                           f"empty_opponent_out() template does not declare "
                           f"(template has {sorted(dest)})")
        dest[k][rows] = v[rows] if full_batch else v


def nbytes(d):
    """Total bytes of the tensors in a dict (used to size the rollout buffer)."""
    return sum(v.numel() * v.element_size() for v in d.values() if torch.is_tensor(v))


def stack_metrics(names, values, device):
    """Pack a metric dict into a fixed-order tensor for on-device accumulation.

    Reading metrics with ``.item()`` inside the minibatch loop would sync the CUDA
    stream hundreds of times per iteration for numbers that are printed once, so
    they are accumulated as a single tensor and read at the end.
    """
    return torch.stack([values[n] if torch.is_tensor(values[n])
                        else torch.as_tensor(float(values[n]), device=device)
                        for n in names])

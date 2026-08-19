"""Rollout buffer: storage, GAE, flattening, whitening, minibatching.

Nothing here knows what a transition contains. A transition is a dict of
[B, ...] tensors that must include ``value``, ``reward`` and ``done``; every other
key is passed through to the minibatches untouched.
"""
from __future__ import annotations

import torch

from .utils import nbytes, to_device


def resolve_store_device(store_device, compute_device, *, bytes_needed=None,
                         vram_frac=0.45):
    """Where the rollout buffer should live.

    Returns ``(device_str, spilled)``. ``spilled`` is True only when "auto" WANTED
    VRAM and could not have it, which is the one case worth warning about.

    Measured on a transformer policy at B=2048: host RAM and VRAM are within noise
    of each other, because the per-minibatch gather and host-to-device copy overlap
    with GPU compute. So "cpu" is a fine default and leaves the VRAM for a larger
    B, which does move the needle. What is NOT fine is pinning the host buffer:
    pinning thousands of small blocks costs far more than the copies it saves.
    """
    on_cuda = torch.device(compute_device).type == "cuda"
    if store_device == "cuda":
        return (compute_device if on_cuda else "cpu"), False
    if store_device != "auto":
        return store_device, False
    if not on_cuda or bytes_needed is None:
        return "cpu", False
    free, _total = torch.cuda.mem_get_info(torch.device(compute_device))
    # flatten() briefly holds the per-step tensors AND their concatenation, so the
    # fraction must stay under 0.5 of what is free.
    if bytes_needed < vram_frac * free:
        return compute_device, False
    return "cpu", True


class RolloutBuffer:
    """Per-step transition storage for one iteration.

    ``store_device="auto"`` is resolved on the FIRST ``add`` -- by then the true
    size of a transition is known by measurement, which beats any hand-written
    per-key estimate and cannot drift when a game adds a feature.
    """

    def __init__(self, store_device="cpu", *, compute_device="cpu",
                 expected_steps=None, vram_frac=0.45, verbose=False):
        self.want = store_device
        self.compute_device = compute_device
        self.expected_steps = expected_steps
        self.vram_frac = vram_frac
        self.verbose = verbose
        self.sdev = None            # resolved on the first add()
        self._resolved = False
        self.spilled = False
        self.bytes_per_step = 0
        self.buf = []

    # ---- storage ---------------------------------------------------------- #
    def _resolve(self, transition):
        self.bytes_per_step = nbytes(transition)
        need = (self.bytes_per_step * self.expected_steps
                if self.expected_steps else None)
        self.sdev, self.spilled = resolve_store_device(
            self.want, self.compute_device, bytes_needed=need,
            vram_frac=self.vram_frac)
        self._resolved = True
        if self.verbose:
            tot = (need or 0) / 2 ** 30
            print(f"rollout buffer: ~{tot:.2f} GiB/rank on "
                  f"{'GPU' if self.sdev != 'cpu' else 'host RAM'} "
                  f"({self.expected_steps} steps x "
                  f"{self.bytes_per_step / 2 ** 20:.1f} MiB)")
            if self.spilled:
                print("  (does not fit VRAM -- the per-minibatch gather and H2D "
                      "copy will show up in the update; lower steps_per_iter, "
                      "raise --gpus, or accept it)")

    def add(self, transition):
        if not self._resolved:
            self._resolve(transition)
        self.buf.append(to_device(transition, self.sdev))

    def __len__(self):
        return len(self.buf)

    def clear(self):
        self.buf.clear()

    # ---- advantages ------------------------------------------------------- #
    def compute_gae(self, last_value, gamma, lam):
        """Standard truncated GAE, writing ``adv`` and ``ret`` into every step.

        The effective horizon of the advantage estimate is ~1/(1 - gamma*lam)
        steps, NOT the rollout length -- which is why a short rollout (a big B for
        a fixed steps_per_iter) costs little, as long as the rollout still comfort-
        ably exceeds that window. A terminal reward reaches earlier states through
        the CRITIC bootstrap, not through the lambda sum.
        """
        steps = len(self.buf)
        last_value = last_value.to(self.sdev)
        gae = torch.zeros_like(last_value)
        adv = [None] * steps
        for t in reversed(range(steps)):
            nonterm = 1.0 - self.buf[t]["done"]
            nextv = last_value if t == steps - 1 else self.buf[t + 1]["value"]
            delta = self.buf[t]["reward"] + gamma * nextv * nonterm - self.buf[t]["value"]
            gae = delta + gamma * lam * nonterm * gae
            adv[t] = gae.clone()
        for t in range(steps):
            self.buf[t]["adv"] = adv[t]
            self.buf[t]["ret"] = adv[t] + self.buf[t]["value"]

    # ---- flatten ---------------------------------------------------------- #
    def flatten(self, drop=("reward", "done")):
        """Concatenate the buffer into [steps*B, ...] tensors and empty it.

        Key by key, dropping each key's per-step tensors as it goes: a dict
        comprehension would hold the whole buffer AND its copy at once, which on a
        GPU-resident buffer means peaking at twice the VRAM.
        """
        if not self.buf:
            raise RuntimeError("flatten() on an empty buffer")
        keys = [k for k in self.buf[0] if k not in drop]
        steps = len(self.buf)
        flat = {}
        for k in keys:
            flat[k] = torch.cat([self.buf[t][k] for t in range(steps)], dim=0)
            for t in range(steps):
                self.buf[t].pop(k, None)
        self.buf.clear()
        return flat


def whiten_(flat, key="adv", dist=None, device=None):
    """Normalize ``flat[key]`` in place using GLOBAL (all-rank) statistics.

    Whitening per rank would make an N-GPU run optimize a subtly different
    objective than a 1-GPU run on the same data; reducing the two moments keeps
    them identical.
    """
    a = flat[key]
    n = float(a.shape[0])
    dev = device or a.device
    s = torch.stack([a.sum().to(dev), (a * a).sum().to(dev),
                     torch.tensor(n, device=dev)])
    if dist is not None:
        dist.all_reduce_(s)
    mean = s[0] / s[2]
    std = (s[1] / s[2] - mean * mean).clamp(min=0).sqrt()
    flat[key] = (a - mean.to(a.device)) / (std.to(a.device) + 1e-8)
    return float(mean), float(std)


def minibatch_iter(flat, mb_size, device, keys=None, generator=None):
    """Shuffled minibatches of a flattened rollout, moved to ``device``.

    The permutation is drawn on the buffer's own device so indexing never crosses
    the host/device boundary twice.
    """
    keys = list(keys if keys is not None else flat.keys())
    n = flat[keys[0]].shape[0]
    idx_dev = flat[keys[0]].device
    perm = torch.randperm(n, device=idx_dev, generator=generator)
    for i in range(0, n, mb_size):
        idx = perm[i:i + mb_size]
        yield {k: flat[k][idx].to(device, non_blocking=True) for k in keys}


def explained_variance(flat, dist=None, device=None):
    """1 - Var(ret - value)/Var(ret) over the whole (all-rank) batch.

    The single most informative number in the log: if it is near 0 the critic is
    predicting nothing and every advantage is noise, no matter how healthy the
    policy loss looks.
    """
    with torch.no_grad():
        dev = device or flat["ret"].device
        ret = flat["ret"].to(dev)
        val = flat["value"].to(dev)
        n = float(ret.shape[0])
        s = torch.stack([ret.sum(), (ret * ret).sum(), (ret - val).sum(),
                         ((ret - val) ** 2).sum(), torch.tensor(n, device=dev)])
        if dist is not None:
            dist.all_reduce_(s)
        cnt = s[4]
        var_ret = s[1] / cnt - (s[0] / cnt) ** 2
        var_err = s[3] / cnt - (s[2] / cnt) ** 2
        return float(1.0 - var_err / (var_ret + 1e-8))

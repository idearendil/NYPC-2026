"""The PPO update. Game-independent: it only ever calls ``Policy.evaluate`` and
``Policy.evaluate_value`` on minibatches of a flattened rollout.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .buffer import explained_variance, minibatch_iter

_BASE_METRICS = ("ploss", "vloss", "entropy", "approx_kl", "clipfrac")


class PPO:
    """Clipped-surrogate PPO with an optional separate critic optimizer.

    Two parameter groups (``{'actor': ..., 'critic': ...}``) get one Adam each and
    one backward pass each. That is the right default when the actor and critic are
    separate networks: the value loss then needs no relative weight, and an early
    value-loss spike (they are large before the critic has learned anything) cannot
    blow up the policy's gradient through a shared trunk. The groups must be
    DISJOINT for this mode -- a shared trunk should declare a single group instead,
    and is then optimized as ``policy + vf_coef * value + extra``.
    """

    def __init__(self, policy, cfg, dist=None, device="cpu"):
        self.policy = policy
        self.cfg = cfg
        self.dist = dist
        self.device = device
        groups = policy.param_groups()
        if not groups:
            raise ValueError("Policy.param_groups() returned nothing to optimize")
        self.shared = len(groups) == 1
        if not self.shared and set(groups) != {"actor", "critic"}:
            raise ValueError(
                f"param_groups() must be either ONE group (shared trunk) or "
                f"exactly {{'actor', 'critic'}}; got {sorted(groups)}")
        # fused Adam is a single kernel over all parameters instead of one launch
        # per tensor -- worth having when the update runs hundreds of times an
        # iteration over many small tensors.
        adam_kw = dict(fused=True) if torch.device(device).type == "cuda" else {}
        self.opts = {name: torch.optim.Adam(params, lr=cfg.lr, **adam_kw)
                     for name, params in groups.items()}
        self.groups = groups
        self._names = None

    # ---- hyperparameters -------------------------------------------------- #
    def set_lr(self, lr):
        for opt in self.opts.values():
            for g in opt.param_groups:
                g["lr"] = lr

    # ---- the update ------------------------------------------------------- #
    def update(self, flat, cfg_it, *, mb_size, keys=None):
        """One iteration's worth of epochs over ``flat``. Returns a metric dict.

        ``cfg_it`` is the phase-resolved config (lr already applied), so epochs,
        clip, entropy coefficient and target_kl all come from it.
        """
        policy, dist, device = self.policy, self.dist, self.device
        keys = list(keys if keys is not None else flat.keys())
        acc, nb, epochs_run = None, 0, 0
        for _ in range(cfg_it.epochs):
            ep_kl = torch.zeros((), device=device)
            ep_nb = 0
            for mb in minibatch_iter(flat, mb_size, device, keys):
                # ---- actor ------------------------------------------------- #
                out = policy.evaluate(mb)
                logratio = out.logp - mb["old_logp"]
                ratio = torch.exp(logratio)
                with torch.no_grad():
                    # Schulman's positive estimator E[(r-1) - log r] ~ KL(old||new)
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfrac = ((ratio - 1).abs() > cfg_it.clip).float().mean()
                adv = mb["adv"]
                s1 = ratio * adv
                s2 = torch.clamp(ratio, 1 - cfg_it.clip, 1 + cfg_it.clip) * adv
                ploss = -torch.min(s1, s2).mean()
                actor_loss = (ploss - cfg_it.ent_coef * out.entropy.mean()
                              + out.extra_loss)

                # ---- critic ------------------------------------------------ #
                if not self.shared:
                    self._step("actor", actor_loss, cfg_it)
                cout = policy.evaluate_value(mb)
                vloss = F.mse_loss(cout.value, mb["ret"])
                critic_loss = vloss + cout.extra_loss
                if self.shared:
                    name = next(iter(self.opts))
                    self._step(name, actor_loss + cfg_it.vf_coef * critic_loss,
                               cfg_it)
                else:
                    self._step("critic", critic_loss, cfg_it)

                # ---- metrics (accumulated on-device; see utils.stack_metrics) #
                with torch.no_grad():
                    vals = {"ploss": ploss.detach(), "vloss": vloss.detach(),
                            "entropy": out.entropy.mean(), "approx_kl": approx_kl,
                            "clipfrac": clipfrac}
                    vals.update({k: torch.as_tensor(v, device=device).float()
                                 for k, v in out.metrics.items()})
                    vals.update({k: torch.as_tensor(v, device=device).float()
                                 for k, v in cout.metrics.items()})
                    if self._names is None:
                        self._names = list(_BASE_METRICS) + [
                            k for k in vals if k not in _BASE_METRICS]
                    row = torch.stack([vals[n].reshape(()).float()
                                       for n in self._names])
                    acc = row if acc is None else acc + row
                    ep_kl += approx_kl
                nb += 1
                ep_nb += 1
            epochs_run += 1
            # KL early stop: if this epoch's mean drift already exceeds the target,
            # stop refreshing on a now-stale batch instead of overshooting. The
            # (one per epoch) device sync only happens when the check is enabled.
            if cfg_it.target_kl is not None:
                kl_ep = (ep_kl / max(ep_nb, 1)).reshape(1)
                if dist is not None:
                    dist.all_reduce_(kl_ep, "mean")
                if float(kl_ep) > cfg_it.target_kl:
                    break
        m = acc / max(nb, 1)
        if dist is not None:
            dist.all_reduce_(m, "mean")
        out = dict(zip(self._names, m.tolist()))
        out["epochs_run"] = epochs_run
        out["minibatches"] = nb
        out["value_ev"] = explained_variance(flat, dist, device)
        return out

    def _step(self, name, loss, cfg_it):
        opt = self.opts[name]
        params = self.groups[name]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Average across ranks BEFORE clipping, so the clip acts on the gradient
        # the optimizer will actually apply (clipping first would give every rank a
        # different, individually-clipped gradient and then average those).
        if self.dist is not None:
            self.dist.all_reduce_grads(params)
        nn.utils.clip_grad_norm_(params, cfg_it.max_grad_norm)
        opt.step()

    # ---- checkpointing ---------------------------------------------------- #
    def state_dict(self):
        return {k: o.state_dict() for k, o in self.opts.items()}

    def load_state_dict(self, sd, verbose=True):
        """Restore optimizer moments, tolerating a changed parameter set.

        Adam's state is keyed positionally, so adding a head to a network makes the
        stored moments unmappable. That is not worth failing a resume over: a
        couple of hundred steps rebuilds them.
        """
        for k, o in self.opts.items():
            if k not in sd:
                continue
            try:
                o.load_state_dict(sd[k])
            except (ValueError, KeyError) as e:
                if verbose:
                    print(f"  optimizer '{k}' state incompatible ({e}); "
                          f"reinitializing it")

"""The opponent pool: who the agent plays, and how that set evolves.

This is the part of a self-play setup that most often decides whether training
works at all, and none of it is game-specific.

Design, and why:

* **Fixed scripted bots occupy the leading slots and are never evicted.** A pool
  of nothing but your own snapshots is homogeneous: the policy can climb by
  learning to beat its own habits and quietly lose to a committed strategy it has
  not seen in a while. One rusher and one greedy expander keep an absolute
  yardstick in the pool and in the win-rate curves.
* **Sampling is win-rate-INVERSE.** Each opponent's weight is
  ``(1 - ema_win_rate)`` floored at ``sample_floor``, so games concentrate on the
  opponents that are still beating us, while the floor keeps already-beaten ones
  in the mix (they are the ones a policy silently regresses against).
* **The pool grows when the MIN win rate crosses a threshold** -- i.e. only once
  the agent handles every current opponent, which is the signal that the pool has
  stopped being a curriculum.
* **Periodic PERMANENT snapshots.** Every ``snapshot_every`` iterations a snapshot
  is added that is never evicted, and the cap grows by one so the evictable
  capacity is unchanged. Without them a FIFO pool forgets its own history and
  cycles: opponents from 500 iterations ago are exactly the ones that catch a
  regression.
* **The EMA update is order-independent** (see ``apply_tally``), which is what
  lets several data-parallel ranks reach bit-identical win rates and therefore
  make identical add/evict decisions.
"""
from __future__ import annotations

import torch

from .utils import slice_rows, write_rows


class Tally:
    """On-device accumulator for one iteration's episode results.

    Kept on the GPU and read exactly once per iteration: tallying a finished game
    on the host would force a device sync on every step that ends an episode,
    which at a large batch size is every step.
    """

    def __init__(self, n_opp, device):
        self.ep = torch.zeros(2, device=device)        # [reward sum, episode count]
        self.wr_sum = torch.zeros(n_opp, device=device)
        self.wr_cnt = torch.zeros(n_opp, device=device)

    def update(self, reward, done, assign, mask=None):
        """Fold one step's finished games in.

        ``mask`` ([B] bool, optional) excludes games from the WIN-RATE tally while
        still counting them in the episode-reward average -- for games played under
        modified rules, which are useful training data but not a fair yardstick.
        """
        dn = done.float()
        self.ep[0] += (reward * dn).sum()
        self.ep[1] += dn.sum()
        res = torch.where(reward > 0, torch.ones_like(reward),
                          torch.where(reward < 0, torch.zeros_like(reward),
                                      torch.full_like(reward, 0.5)))
        cnt = dn if mask is None else dn * mask.float()
        self.wr_sum.scatter_add_(0, assign, res * cnt)
        self.wr_cnt.scatter_add_(0, assign, cnt)


class OpponentPool:
    """Scripted bots + frozen policy snapshots, with EMA win rates.

    Unified indexing: ``0 .. n_scripted-1`` are the scripted bots, and net
    snapshot ``j`` is unified index ``j + n_scripted``. ``ids`` gives every
    opponent a STABLE id that survives eviction index shifts, so a logged
    per-opponent win-rate curve keeps tracking the same opponent.
    """

    def __init__(self, scripted, policy, *, B, device, ema_alpha=0.02,
                 add_threshold=0.6, max_size=7, snapshot_every=0,
                 sample_floor=0.05, seed=0):
        self.scripted = list(scripted)
        self.n_scripted = len(self.scripted)
        self.device = device
        self.alpha = ema_alpha
        self.add_threshold = add_threshold
        self.max_size = max_size
        self.snapshot_every = snapshot_every
        self.floor = sample_floor
        # starts with one frozen copy of the initial policy: a fresh agent needs
        # an opponent of its own strength, not only the scripted extremes.
        self.nets = [policy.clone_frozen()]
        self.perm = [False]
        self.wr = torch.full((self.n_scripted + 1,), 0.5)      # CPU, EMA win rate
        self.ids = [s.name for s in self.scripted] + [0]
        self.next_id = 1
        self.gen = torch.Generator().manual_seed(seed)
        self.assign = self.sample(B).to(device)
        self._uniq = None

    # ---- shape ------------------------------------------------------------ #
    @property
    def size(self):
        return self.n_scripted + len(self.nets)

    @property
    def cap(self):
        """Total cap: the base size plus one slot per permanent snapshot, so the
        EVICTABLE capacity stays constant as permanents accumulate."""
        return self.max_size + sum(self.perm)

    # ---- matchmaking ------------------------------------------------------ #
    def sample(self, n):
        w = (1.0 - self.wr).clamp(min=self.floor)
        return torch.multinomial(w / w.sum(), n, replacement=True,
                                 generator=self.gen)

    def reassign(self, rows):
        """Draw fresh opponents for games that just finished."""
        self.assign[rows] = self.sample(rows.numel()).to(self.assign.device)
        self._uniq = None

    def unique(self):
        """The distinct opponents currently in play (cached).

        Cheap but not free: reading it off the GPU is a device sync, so it is
        recomputed only when the assignment actually changed -- on the steps where
        no episode ended, the cached list is exact.
        """
        if self._uniq is None:
            self._uniq = torch.unique(self.assign).tolist()
        return self._uniq

    def new_tally(self):
        return Tally(self.size, self.device)

    # ---- acting ----------------------------------------------------------- #
    def act(self, task, obs, **kw):
        """Sample every assigned opponent's action, grouped so each runs ONCE.

        Grouping is what keeps a mixed pool affordable: with 8 opponents in play
        the rollout costs 8 forward passes over disjoint slices of the batch, not
        8 passes over the whole batch.
        """
        uniq = self.unique()
        action, extra = task.empty_opponent_out(self.assign.shape[0])
        for p in uniq:
            rows = (self.assign == p).nonzero(as_tuple=True)[0]
            if p < self.n_scripted:
                bot = self.scripted[p]
                a, e = bot.act(task, obs, rows, **kw)
                write_rows(action, rows, a, full_batch=bot.full_batch)
                write_rows(extra, rows, e, full_batch=bot.full_batch)
            else:
                sub = slice_rows(obs, rows)
                sub_kw = {k: (v[rows] if torch.is_tensor(v) else v)
                          for k, v in kw.items()}
                a, _store, e = self.nets[p - self.n_scripted].act(sub, **sub_kw)
                write_rows(action, rows, a)
                write_rows(extra, rows, e)
        return action, extra

    def reset_rows(self, rows):
        for bot in self.scripted:
            bot.reset_rows(rows)

    # ---- win rates -------------------------------------------------------- #
    def apply_tally(self, tally, dist=None):
        """Advance the EMA win rates once per iteration; returns episode stats.

        ``n`` results with mean ``r`` for one opponent collapse to the closed form
        of ``n`` successive EMA steps that all saw the same value::

            wr <- (1-a)^n * wr + (1 - (1-a)^n) * r

        which is (a) equivalent to updating game-by-game up to the within-iteration
        ordering, and (b) ORDER-INDEPENDENT -- the property that lets several ranks
        all-reduce their tallies and end up with bit-identical win rates, and
        therefore identical snapshot and eviction decisions.
        """
        if dist is not None:
            dist.all_reduce_(tally.wr_sum)
            dist.all_reduce_(tally.wr_cnt)
            dist.all_reduce_(tally.ep)
        cnt, s = tally.wr_cnt.cpu(), tally.wr_sum.cpu()
        decay = (1.0 - self.alpha) ** cnt
        mean_res = s / cnt.clamp(min=1)
        self.wr = torch.where(cnt > 0, decay * self.wr + (1 - decay) * mean_res,
                              self.wr)
        return int(tally.ep[1]), float(tally.ep[0])

    # ---- growth ----------------------------------------------------------- #
    def _append(self, policy, permanent):
        self.nets.append(policy.clone_frozen())
        self.perm.append(permanent)
        self.wr = torch.cat([self.wr, torch.full((1,), 0.5)])
        self.ids.append(self.next_id)
        self.next_id += 1

    def maybe_snapshot(self, it, policy):
        """Periodic PERMANENT snapshot (never evicted; grows the cap by one)."""
        if self.snapshot_every and it > 0 and it % self.snapshot_every == 0:
            self._append(policy, permanent=True)
            return True
        return False

    def maybe_grow(self, policy):
        """Add the current policy once it beats even the hardest pool member.

        When the pool is at capacity the OLDEST evictable net goes; scripted bots
        and permanent snapshots are never touched.
        """
        if float(self.wr.min()) <= self.add_threshold:
            return False
        if self.size >= self.cap:
            evictable = [i for i, p in enumerate(self.perm) if not p]
            if evictable:
                self._evict(evictable[0])
        self._append(policy, permanent=False)
        return True

    def _evict(self, pos):
        idx = pos + self.n_scripted            # list position -> unified index
        self.nets.pop(pos)
        self.perm.pop(pos)
        self.ids.pop(idx)
        self.wr = torch.cat([self.wr[:idx], self.wr[idx + 1:]])
        # Games on the evicted opponent fall back to unified index 0; higher
        # indices shift down one. Reassign BEFORE shifting or the fallback rows
        # get shifted too.
        self.assign = torch.where(self.assign == idx,
                                  torch.zeros_like(self.assign), self.assign)
        self.assign = torch.where(self.assign > idx, self.assign - 1, self.assign)
        self._uniq = None

    # ---- logging ---------------------------------------------------------- #
    def metrics(self):
        m = {
            "pool_size": self.size,
            "pool_cap": self.cap,
            "pool_perm_count": self.n_scripted + sum(self.perm),
            "opp_winrate_min": float(self.wr.min()),
            "opp_winrate_mean": float(self.wr.mean()),
        }
        for k, oid in enumerate(self.ids):
            m[f"opp_winrate/{oid}"] = float(self.wr[k])
        for i, bot in enumerate(self.scripted):
            m[f"opp_winrate_{bot.name}"] = float(self.wr[i])
        return m

    # ---- checkpointing ---------------------------------------------------- #
    def state_dict(self):
        return {
            "nets": [n.state_dict() for n in self.nets],
            "perm": list(self.perm),
            "wr": self.wr,
            "ids": list(self.ids),
            "next_id": self.next_id,
            "scripted_names": [s.name for s in self.scripted],
            "assign": self.assign.cpu(),
            "gen": self.gen.get_state(),
        }

    def load_state_dict(self, sd, make_policy, *, B, rank=0, strict=False,
                        verbose=True):
        """Restore a pool, tolerating a changed set of scripted bots.

        Scripted slots are matched BY NAME: bots that were there keep their EMA win
        rate, bots you added since get 0.5 and slot in at their current position,
        and bots you removed are dropped -- with the assignment indices remapped
        accordingly. Without that, adding a second scripted opponent would silently
        reinterpret every stored index.
        """
        self.nets = []
        for s in sd["nets"]:
            p = make_policy()
            # strict=False by default: a snapshot only ever runs the policy, so a
            # head that has since changed shape can be left fresh rather than
            # making an old checkpoint unloadable.
            p.load_state_dict(s, strict=strict)
            self.nets.append(p.clone_frozen())
        self.perm = list(sd["perm"])
        self.ids = list(sd["ids"])
        self.next_id = sd["next_id"]
        old_names = list(sd.get("scripted_names", []))
        new_names = [s.name for s in self.scripted]
        old_wr = sd["wr"].cpu()
        assign = sd["assign"].clone()
        if old_names != new_names:
            # remap: old unified index -> new unified index (-1 = gone)
            remap = torch.full((old_wr.numel(),), -1, dtype=torch.long)
            wr = torch.full((len(new_names) + len(self.nets),), 0.5)
            for i, nm in enumerate(old_names):
                if nm in new_names:
                    j = new_names.index(nm)
                    remap[i] = j
                    wr[j] = old_wr[i]
            for j in range(len(self.nets)):       # nets keep their order
                remap[len(old_names) + j] = len(new_names) + j
                wr[len(new_names) + j] = old_wr[len(old_names) + j]
            self.wr = wr
            self.ids = new_names + [i for i in self.ids[len(old_names):]]
            assign = remap[assign.clamp(min=0)]
            assign = torch.where(assign < 0, torch.zeros_like(assign), assign)
            if verbose:
                print(f"  migrated pool: scripted {old_names} -> {new_names}")
        else:
            self.wr = old_wr
        self.gen.set_state(sd["gen"].cpu())
        # The stored assignment is rank 0's, for rank 0's B games. Any other rank
        # (or a changed B) just redraws: episodes never resume across runs, so
        # nothing downstream has to match it.
        if assign.numel() == B and rank == 0:
            self.assign = assign.to(self.device)
        else:
            self.assign = self.sample(B).to(self.device)
        self._uniq = None

    def reseed(self, seed, B=None):
        """Diverge this rank's matchmaking stream after a resume.

        Every rank restores the SAME generator state from the checkpoint, so
        without this the ranks would draw identical opponents for identical game
        slots and half the batch diversity would vanish.
        """
        self.gen.manual_seed(int(seed))
        self.assign = self.sample(B or self.assign.numel()).to(self.device)
        self._uniq = None

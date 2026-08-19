"""Data collection: drive the task for N turns, storing one transition per turn.

The loop itself is tiny -- the value is in what it takes off the task's hands
(opponent dispatch, episode restart, on-device statistics) and in the two things it
is careful about: never syncing the device except where unavoidable, and never
storing an alias of mutable env state.
"""
from __future__ import annotations

import torch


class RolloutContext:
    """What a task is allowed to reach during ``rollout_turn``.

    Deliberately narrow: the task can run the agent, run the critic, and run
    whichever opponents the pool assigned -- and knows nothing about the buffer,
    the optimizer, or how many ranks there are.
    """

    __slots__ = ("policy", "pool", "task", "device", "dist", "step", "steps")

    def __init__(self, policy, pool, task, device, dist=None):
        self.policy, self.pool, self.task = policy, pool, task
        self.device, self.dist = device, dist
        self.step, self.steps = 0, 0

    def agent_act(self, obs, **kw):
        """Sample the trainable policy -> ``(action, store, extra)``."""
        return self.policy.act(obs, **kw)

    def agent_value(self, obs):
        """Critic value for the transition being stored."""
        return self.policy.value(obs)

    def opponent_act(self, obs, **kw):
        """Run the assigned opponents -> ``(action, extra)``, merged full-batch."""
        return self.pool.act(self.task, obs, **kw)

    @property
    def assign(self):
        return self.pool.assign


def run_rollout(task, policy, pool, buffer, steps, *, device, dist=None,
                tally=None):
    """Collect ``steps`` turns into ``buffer``. Returns the tally.

    Everything runs under ``no_grad``: the stored transition is data, and the
    gradient is only ever taken later, in the update, against a re-evaluation of
    it. Autograd during collection would just build graphs to throw away.
    """
    tally = tally if tally is not None else pool.new_tally()
    ctx = RolloutContext(policy, pool, task, device, dist)
    ctx.steps = steps
    with torch.no_grad():
        for s in range(steps):
            ctx.step = s
            store = task.rollout_turn(ctx)
            reward, done = task.reward_done()
            if "value" not in store:
                raise KeyError("rollout_turn() must put 'value' in the stored "
                               "transition (ctx.agent_value(obs))")
            store["reward"] = reward
            store["done"] = done.float()
            buffer.add(store)
            tally.update(reward, done, pool.assign, mask=task.tally_mask())
            # `done.any()` is one device sync per step. It is worth it: at a large
            # batch size an episode ends nearly every step anyway, and skipping the
            # restart work when nothing finished is the common case at small B.
            if bool(done.any()):
                rows = done.nonzero(as_tuple=True)[0]
                # A finished game gets a FRESH opponent draw. Note that the pool's
                # win rates are the values from the START of this iteration (they
                # advance once per iteration, which is what keeps ranks in
                # lockstep); with a small EMA rate and hundreds of games per
                # iteration that is indistinguishable from updating as you go.
                pool.reassign(rows)
                pool.reset_rows(rows)
                task.reset_finished(done, rows)
    return tally


def bootstrap_value(task, policy):
    """Value of the state the rollout ended in, for the last GAE step.

    Uses the same observation the next iteration's first turn will see -- no env
    step happens in between, so any per-turn feature history the task carries is
    still the right one. A task that cannot express this as "observe the agent's
    side" overrides ``Task.bootstrap_value``.
    """
    with torch.no_grad():
        own = getattr(task, "bootstrap_value", None)
        if own is not None:
            return own(policy)
        return policy.value(task.observe(task.AGENT))

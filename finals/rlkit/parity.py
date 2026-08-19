"""Parity harness: prove a batched env matches the reference simulator.

When a contest hands you a reference implementation of the rules, the batched GPU
rewrite is the single riskiest thing you will write that day. It is riskier than
the network, because a wrong rule does not crash and does not show up in the loss
curves -- it trains a policy for a game nobody else is playing, and you find out
only when you lose the matches.

So: drive both implementations with the SAME actions, turn by turn, and compare
the full state after every turn. The first divergence is reported with the turn,
the game index and the field, which is usually enough to point at the rule.

Usage sketch::

    import rlkit
    refs = [ReferenceGame(seed=s) for s in range(B)]      # the organizers' code
    env  = MyBatchedEnv([g.map for g in refs], device="cuda")

    ok = rlkit.parity.run(
        refs, env, turns=200,
        sample_actions=lambda env, rng: my_random_actions(env, rng),
        step_ref=lambda ref, act: ref.play_turn(act),
        step_env=lambda env, acts: env.step(acts),
        snap_ref=lambda ref: dict(hp=ref.hp, gold=ref.gold, units=sorted(...)),
        snap_env=lambda env, b: dict(hp=env.hp[b].tolist(), gold=int(env.gold[b]),
                                     units=sorted(...)),
    )

Two rules for the snapshots:

* Compare EVERYTHING that the rules can touch, not just the obvious scalars. A
  mismatch you did not look at is a mismatch you ship.
* Make them ORDER-INDEPENDENT where the reference's order is arbitrary (sort the
  unit list by a stable key), or you will chase differences that do not matter --
  and stop trusting the harness, which is worse.
"""
from __future__ import annotations

import random


def _as_plain(x):
    """Tensors/arrays -> nested lists; anything else unchanged."""
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def compare(a, b, path="", atol=0.0):
    """First structural/numeric difference between two snapshots, or None.

    Handles nested dicts, sequences and scalars. ``atol`` allows a tolerance for
    floats; leave it at 0 when the reference is integer arithmetic, which it
    usually is -- an exact match is a much stronger statement, and cheap to get
    when both sides are doing integer game rules.
    """
    a, b = _as_plain(a), _as_plain(b)
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            return f"{path or '<root>'}: dict vs {type(b).__name__}"
        for k in sorted(set(a) | set(b)):
            if k not in a:
                return f"{path}.{k}: missing on the reference side"
            if k not in b:
                return f"{path}.{k}: missing on the batched side"
            m = compare(a[k], b[k], f"{path}.{k}" if path else str(k), atol)
            if m:
                return m
        return None
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))):
            return f"{path}: sequence vs {type(b).__name__}"
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            m = compare(x, y, f"{path}[{i}]", atol)
            if m:
                return m
        return None
    if isinstance(a, bool) or isinstance(b, bool):
        return None if bool(a) == bool(b) else f"{path}: {a} vs {b}"
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if atol and abs(float(a) - float(b)) <= atol:
            return None
        return None if a == b else f"{path}: {a} vs {b}"
    return None if a == b else f"{path}: {a!r} vs {b!r}"


def run(refs, env, *, sample_actions, step_ref, step_env, snap_ref, snap_env,
        turns=200, seed=0, atol=0.0, alive=None, verbose=True, stop_on_first=True):
    """Play ``turns`` turns in both implementations and compare after each one.

    ``sample_actions(env, rng)`` returns ``(batched_action, per_game_actions)`` --
    one object the batched env takes and a list of B objects the reference takes.
    Drawing the actions ONCE and feeding both sides is the whole point: if each
    side sampled its own, a divergence would be indistinguishable from noise.

    ``alive(ref) -> bool`` (optional) skips games the reference has finished, so a
    terminated game does not spam mismatches.

    Returns True when every turn of every game matched.
    """
    rng = random.Random(seed)
    B = len(refs)
    bad = []
    for t in range(turns):
        batched_action, per_game = sample_actions(env, rng)
        step_env(env, batched_action)
        for b in range(B):
            if alive is not None and not alive(refs[b]):
                continue
            step_ref(refs[b], per_game[b])
        for b in range(B):
            if alive is not None and not alive(refs[b]):
                continue
            msg = compare(snap_ref(refs[b]), snap_env(env, b), atol=atol)
            if msg:
                bad.append((t, b, msg))
                if verbose:
                    print(f"[MISMATCH] turn {t} game {b}: {msg}")
                if stop_on_first:
                    return False
    if verbose:
        print(f"parity: {turns} turns x {B} games matched"
              if not bad else f"parity: {len(bad)} mismatches")
    return not bad

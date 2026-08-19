"""Console + wandb logging. Rank 0 only; everywhere else every method is a no-op.

The console line is deliberately one dense line per iteration: over a run of
hundreds of iterations you read it as a column, and anything that wraps stops being
readable. wandb gets the full metric dict.
"""
from __future__ import annotations

import os


class Logger:
    def __init__(self, *, enabled=False, project="rlkit", config=None,
                 is_main=True, name=None, api_key_env="WANDB_API_KEY"):
        self.is_main = is_main
        self.run = None
        if not (enabled and is_main):
            return
        import wandb
        # The key comes from the environment (or a login done once on the box) --
        # never from a config file that ends up in git.
        if api_key_env and not os.environ.get(api_key_env):
            print(f"note: {api_key_env} is not set; wandb may prompt or run offline")
        self.run = wandb.init(project=project, config=config, name=name)

    def log(self, it, metrics, line=None):
        if line and self.is_main:
            print(line, flush=True)
        if self.run is not None:
            import wandb
            wandb.log(metrics, step=it)

    def close(self):
        if self.run is not None:
            self.run.finish()
            self.run = None


def console_line(it, phase, m, task_extra=""):
    """The one-line-per-iteration summary.

    Ordered by how often it tells you something: episode reward first (is it
    learning), then the losses, then explained variance (is the critic real), then
    the pool state (is the curriculum advancing), then throughput.
    """
    parts = [f"iter {it:4d}" + (f" p{phase}" if phase else "")]
    parts.append(f"eps {m.get('episodes', 0):5d}")
    parts.append(f"avg_ep_R {m.get('avg_ep_R', 0.0):+6.2f}")
    parts.append(f"ploss {m.get('ploss', 0.0):+.4f} vloss {m.get('vloss', 0.0):.3f} "
                 f"ent {m.get('entropy', 0.0):.3f} kl {m.get('approx_kl', 0.0):.4f}")
    parts.append(f"ev {m.get('value_ev', 0.0):+.3f}")
    parts.append(f"ep {m.get('epochs_run', 0)}/{m.get('epochs', 0)}")
    parts.append(f"pool {m.get('pool_size', 0)}/{m.get('pool_cap', 0)} "
                 f"wr_min {m.get('opp_winrate_min', 0.0):.2f}")
    if task_extra:
        parts.append(task_extra)
    parts.append(f"{m.get('steps_per_s', 0):,.0f} steps/s "
                 f"({m.get('iter_seconds', 0.0):.1f}s)")
    return " | ".join(parts)

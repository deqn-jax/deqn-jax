"""Mid-run optimizer switch: the switched optimizer keeps the configured
optimizer fields.

Before, the switch rebuilt the optimizer from only ``name``, ``switch_lr`` and
``grad_clip``, so every other configured field (L-BFGS memory, damping, betas,
weight decay) silently fell back to its default, while the checkpoint loader
rebuilds the switched optimizer from the full configured block. A post-switch
checkpoint trained with a non-default ``memory_size`` then could not be loaded.
"""

import os
import shutil

from deqn_jax.config import TrainConfig
from deqn_jax.training.checkpointing import load_policy_from_checkpoint
from deqn_jax.training.trainer import train_from_config


def test_post_switch_checkpoint_loads_with_configured_fields(tmp_path):
    ckpt_dir = str(tmp_path / "ck")
    cfg = TrainConfig.from_dict(
        {
            "model": "brock_mirman",
            "episodes": 3,
            "batch_size": 16,
            "seed": 0,
            "verbose": False,
            "network": {"type": "mlp", "hidden_sizes": [16]},
            "optimizer": {"name": "adam", "memory_size": 4},
            "switch_optimizer": "lbfgs",
            "switch_episode": 2,
            "checkpoint_dir": ckpt_dir,
            "checkpoint_every": 3,
        }
    )
    train_from_config(cfg)

    path = os.path.join(ckpt_dir, "checkpoint_000003.eqx")
    net, _ = load_policy_from_checkpoint(path)
    assert net is not None

    # checkpoint_best.eqx carries no episode in its name; the loader reads it
    # from the sidecar, or it would build the pre-switch optimizer's state.
    best = os.path.join(ckpt_dir, "checkpoint_best.eqx")
    shutil.copy(path, best)
    with open(os.path.join(ckpt_dir, "checkpoint_best.meta"), "w") as f:
        f.write("episode 3\nloss 1.000000e-03\n")
    net, _ = load_policy_from_checkpoint(best)
    assert net is not None

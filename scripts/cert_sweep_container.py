"""Disaster certification sweep — NGC-container launcher (GPU).

Same 4 arms x 3 seeds as scripts/disaster_cert_sweep.sh, but as a python
launcher for scripts/run_sweep_in_container.sh (in-process, no uv/console
scripts needed inside the container):

  LAUNCHER=scripts/cert_sweep_container.py ./scripts/run_sweep_in_container.sh

Resumable: a tag is skipped iff its DONE marker exists (checkpoint_best.eqx
alone is not enough — fallback saves write it mid-run).
"""

from pathlib import Path

from deqn_jax.config import TrainConfig
from deqn_jax.training.trainer import train_from_config

ARMS = [
    "disaster",
    "disaster_gated",
    "disaster_elbcov",
    "disaster_gated_elbcov",
    "disaster_gated_drift",
    "disaster_gated_rsob",
    "disaster_gated_drift_rsob",
    "disaster_gated_rsob25",  # escalation: rsob as leading term (w=25, 8 dirs)
    "disaster_gated_pcgrad",  # spec-let 6: per-equation conflict surgery
]
SEEDS = [0, 1, 2]


def main() -> None:
    for arm in ARMS:
        for seed in SEEDS:
            tag = f"{arm}_s{seed}"
            outdir = Path("runs/disaster_cert") / tag
            if (outdir / "DONE").exists():
                print(f"=== {tag} already done, skipping ===", flush=True)
                continue
            print(f"=== {tag} start ===", flush=True)
            cfg = TrainConfig.from_yaml(f"configs/{arm}.yaml").with_overrides(
                {
                    "seed": seed,
                    "checkpoint_dir": str(outdir),
                    "checkpoint_every": 1000,
                    "tensorboard_dir": f"runs/disaster_cert/{tag}_tb",
                    "verbose": True,
                }
            )
            train_from_config(cfg)
            (outdir / "DONE").write_text("ok\n")
            print(f"=== {tag} done ===", flush=True)
    print("CERT SWEEP COMPLETE", flush=True)


if __name__ == "__main__":
    main()

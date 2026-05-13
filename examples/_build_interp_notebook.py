"""Build ``examples/interp_brock_mirman.ipynb`` from chapter functions.

Each ``chapter_N`` function returns a list of nbformat cell dicts. ``main``
assembles them, sets notebook metadata, and writes the .ipynb to disk.
Re-run after editing chapters to regenerate; then execute with::

    uv run jupyter nbconvert --to notebook --execute \\
        examples/interp_brock_mirman.ipynb \\
        --output examples/interp_brock_mirman.ipynb

The companion design doc and plan are under ``docs/superpowers/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import nbformat as nbf

NB_PATH = Path(__file__).parent / "interp_brock_mirman.ipynb"


def md(text: str) -> Dict:
    return nbf.v4.new_markdown_cell(text)


def code(src: str) -> Dict:
    return nbf.v4.new_code_cell(src)


def chapter_intro() -> List[Dict]:
    return [
        md(
            """# Mech Interp on DEQNs: Brock-Mirman Walkthrough

A six-chapter narrated example. We train tiny `LinearPlusMLP` networks on
Brock-Mirman at three risk-aversion settings, then peel them open chapter
by chapter.

Each chapter introduces one mech-interp move on a substrate where every
claim is checkable against a known economic solution.

**Outline:**
- Ch 0 — Setup: train the networks, see what they compute
- Ch 1 — Output decomposition (BK linearization vs MLP correction)
- Ch 2 — Per-neuron contributions inside the MLP correction
- Ch 3 — Linear probes: what do live neurons encode?
- Ch 4 — Ablation: causation vs correlation
- Ch 5 — The intensity dial: γ ∈ {1, 2, 5}
- Ch 6 — Honest limits and pointers forward"""
        ),
        code(
            """import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from deqn_jax.config import NetworkConfig, OptimizerConfig, TrainConfig
from deqn_jax.interp import (
    ablate_neuron,
    branch_decompose,
    forward_with_activations,
    linear_probe,
    neuron_contributions,
)
from deqn_jax.training.trainer import train_from_config

FIGDIR = "../docs/dev/figures/interp"
GAMMAS = (1.0, 2.0, 5.0)
HIDDEN = (16, 16)
SEED = 0
EPISODES = 200"""
        ),
    ]


def chapter_0() -> List[Dict]:
    return [
        md(
            """## Chapter 0 — Setup

We train three Brock-Mirman policies, one per γ. Each is a
`LinearPlusMLP` with hidden sizes `(16, 16)` and the same seed. The linear
branch is the Blanchard-Kahn first-order solution of the model (fixed,
not trained). The MLP branch is initialized so that the network *is* the
BK linearization at init; training carves the nonlinear correction.

Below we train all three networks, then plot the γ=2 learned policy on a
(k, z) grid. Chapters 1–4 focus on the γ=2 network; γ=1 and γ=5 come back
in Chapter 5."""
        ),
        code(
            """def train_one(gamma: float):
    cfg = TrainConfig(
        model="brock_mirman",
        constants={"gamma": gamma},
        episodes=EPISODES,
        episode_length=128,
        batch_size=64,
        mc_samples=8,
        seed=SEED,
        network=NetworkConfig(
            type="linear_plus_mlp",
            hidden_sizes=HIDDEN,
            activation="tanh",
        ),
        optimizer=OptimizerConfig(name="adam", learning_rate=1e-3),
    )
    trained, history = train_from_config(cfg)
    final_loss = history["loss"][-1] if "loss" in history else float("nan")
    print(f"γ={gamma}  final loss={final_loss:.3e}")
    return trained


nets = {g: train_one(g) for g in GAMMAS}
net = nets[2.0]  # primary network for Ch 1–4"""
        ),
        code(
            """def state_grid(net, n=50):
    \"\"\"Grid over ±2σ of the ergodic support around the network's SS.\"\"\"
    k_ss = float(net.ss_state[0])
    z_ss = float(net.ss_state[1])
    sigma_z = 0.04
    rho = 0.9
    z_std = sigma_z / np.sqrt(1.0 - rho**2)
    ks = np.linspace(0.7 * k_ss, 1.3 * k_ss, n)
    zs = np.linspace(z_ss - 2 * z_std, z_ss + 2 * z_std, n)
    K, Z = np.meshgrid(ks, zs)
    states = jnp.stack([K.ravel(), Z.ravel()], axis=-1)
    return states, K, Z


states, K, Z = state_grid(net)
policy = np.asarray(net(states)).reshape(K.shape)

fig, ax = plt.subplots(figsize=(5, 4))
pcm = ax.pcolormesh(K, Z, policy, shading="auto", cmap="viridis")
ax.set_xlabel("capital k")
ax.set_ylabel("TFP z")
ax.set_title("Learned savings rate s(k, z) — γ=2")
fig.colorbar(pcm, ax=ax, label="s")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/ch0_policy_gamma2.png", dpi=150)
plt.show()"""
        ),
    ]


def main() -> None:
    chapters = [
        chapter_intro,
        chapter_0,
    ]
    cells: List[Dict] = []
    for chapter in chapters:
        cells.extend(chapter())

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    }

    NB_PATH.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"Wrote {NB_PATH}")


if __name__ == "__main__":
    main()

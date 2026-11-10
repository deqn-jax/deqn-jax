# Installation

DEQN-JAX requires Python 3.10+.

## From source (alpha)

```bash
git clone https://github.com/deqn-jax/deqn-jax.git
cd deqn-jax
uv sync
```

For editable installs (hacking on the framework):

```bash
uv pip install -e .
```

## CUDA

For GPU acceleration on Linux (x86_64 or aarch64):

```bash
uv pip install -U "jax[cuda13]"  # CUDA 13
# or
uv pip install -U "jax[cuda12]"  # CUDA 12
```

## Optional dependencies

TensorBoard logging (`--tensorboard`, or `tensorboard_dir` in a config such as
`configs/disaster.yaml`) works on the plain `uv sync` install: `tensorboardX`
is a core dependency.

| Group     | Purpose                                       | Install                        |
|-----------|-----------------------------------------------|--------------------------------|
| `wandb`   | Weights & Biases experiment tracking (`--wandb`) | `uv sync --extra wandb`     |
| `docs`    | mkdocs-material site (this site)              | `uv sync --extra docs`         |
| `all`     | Everything above                              | `uv sync --all-extras`         |

For an editable install outside `uv sync`, the same extras work as
`uv pip install -e ".[wandb]"`.

## Verify

```bash
uv run deqn-jax check
uv run deqn-jax list
```

`check` confirms JAX detects your devices; `list` shows available models.

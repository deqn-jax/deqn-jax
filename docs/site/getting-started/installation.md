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

| Group     | Purpose                                       | Install                        |
|-----------|-----------------------------------------------|--------------------------------|
| `logging` | TensorBoard / Orbax checkpoints               | `uv pip install -e ".[logging]"`|
| `wandb`   | Weights & Biases experiment tracking          | `uv pip install -e ".[wandb]"`  |
| `docs`    | mkdocs-material site (this site)              | `uv pip install -e ".[docs]"`   |
| `all`     | Everything                                    | `uv pip install -e ".[all]"`    |

## Verify

```bash
uv run deqn-jax check
uv run deqn-jax list
```

`check` confirms JAX detects your devices; `list` shows available models.

# DEQN-JAX

Pure JAX implementation of Deep Equilibrium Networks for solving economic models.

## Overview

DEQN-JAX trains neural networks to satisfy economic equilibrium conditions. It's a Physics-Informed Neural Network (PINN) approach for economics:

```
State (K, Z) → Policy Network → Policy (C, L) → Equilibrium Equations → Loss = Σ residuals²
```

## Installation

```bash
# CPU only
uv sync

# With CUDA 12 support
uv sync
uv pip install -U "jax[cuda12]"

# Or for older CUDA 11
uv pip install -U "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Verify CUDA
```python
import jax
print(jax.devices())  # Should show CudaDevice
```

## Quick Start

```python
from deqn_jax import train

# Train Brock-Mirman model
params, history = train("brock_mirman", episodes=1000)
```

Or via CLI:

```bash
# Basic training
deqn-jax train brock_mirman -n 1000

# With warm start from steady state
deqn-jax train brock_mirman -n 500 --warm-start

# Float64 precision (recommended for disaster model)
deqn-jax train disaster -n 1000 --fp64 --warm-start --lr 0.0001
```

## Features

- **Pure JAX**: Single JIT boundary for maximum performance
- **CUDA support**: Same code runs on CPU/GPU, ~10-50x speedup on GPU
- **Warm start**: L-BFGS initialization from steady state (~20 iters)
- **Named variables**: `s.k`, `p.sav_rate` instead of fragile index slicing
- **Equinox networks**: Clean, Pythonic neural network definitions
- **Optax optimizers**: Composable, well-maintained optimizer library
- **Monte Carlo expectations**: Antithetic variates for variance reduction
- **Gauss-Newton/LM**: Second-order optimizers with JAX autodiff Jacobians

## Models

- `brock_mirman`: Classic optimal growth model (Brock & Mirman, 1972)
- `disaster`: NK-DSGE with financial frictions (coming soon)

## Architecture

```
deqn-jax/
├── src/deqn_jax/
│   ├── types.py           # ModelSpec, TrainState
│   ├── networks/
│   │   ├── mlp.py         # Equinox MLP
│   │   └── lstm.py        # Equinox LSTM
│   ├── training/
│   │   ├── loss.py        # MC expectations, residual MSE
│   │   ├── episode.py     # lax.scan trajectory simulation
│   │   └── trainer.py     # Main training loop
│   ├── optimizers/
│   │   └── gauss_newton.py
│   └── models/
│       └── brock_mirman.py
```

## License

MIT

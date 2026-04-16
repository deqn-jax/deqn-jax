# Adding a network

1. Subclass `eqx.Module` in `src/deqn_jax/networks/your_net.py`.
2. Add a factory `create_your_net(...)` that returns a built instance.
3. Wire `network.type: "your_net"` in `training/trainer.py`'s network
   construction block (search for `create_mlp` or `create_linear_plus_mlp`).

```python
import equinox as eqx
import jax.numpy as jnp

class YourNet(eqx.Module):
    layers: list

    def __init__(self, in_dim, out_dim, key):
        ...

    def __call__(self, state):
        ...
        return policy
```

Make sure your network is compatible with `eqx.filter(model, eqx.is_array)`
for Optax compatibility.

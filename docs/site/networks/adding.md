# Adding a network

1. Subclass `eqx.Module` in `src/deqn_jax/networks/your_net.py`.
2. Add a factory `create_your_net(...)` that returns a built instance.
3. Add a `network.type: "your_net"` branch to `build_policy_net` in
   `src/deqn_jax/networks/factory.py`, next to the existing `lstm`,
   `transformer` and `linear_plus_mlp` branches.

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

The network must work with `eqx.filter(model, eqx.is_array)`, which Optax
relies on.

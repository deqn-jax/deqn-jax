"""Variable access helpers - named views over state/policy arrays.

Instead of fragile index slicing like `state[:, 0]`, use:

    s = unpack_state(state, MODEL)
    s.k  # capital
    s.z  # TFP

JAX traces through NamedTuples efficiently.
"""

from typing import Any, Dict, NamedTuple, Tuple, Type

import jax.numpy as jnp
from jax import Array


def make_state_type(names: Tuple[str, ...]) -> Type[NamedTuple]:
    """Dynamically create a NamedTuple type for state variables."""
    return NamedTuple("State", [(name, Array) for name in names])


def make_policy_type(names: Tuple[str, ...]) -> Type[NamedTuple]:
    """Dynamically create a NamedTuple type for policy variables."""
    return NamedTuple("Policy", [(name, Array) for name in names])


def unpack_array(arr: Array, names: Tuple[str, ...], nt_type: Type) -> NamedTuple:
    """Unpack array columns into named fields.

    Args:
        arr: Array of shape [batch, n_vars] or [n_vars]
        names: Tuple of variable names
        nt_type: NamedTuple type to create

    Returns:
        NamedTuple with named fields
    """
    if arr.ndim == 1:
        return nt_type(*[arr[i] for i in range(len(names))])
    return nt_type(*[arr[:, i] for i in range(len(names))])


def pack_array(nt: NamedTuple) -> Array:
    """Pack NamedTuple fields back into array.

    Args:
        nt: NamedTuple with array fields

    Returns:
        Array of shape [batch, n_vars] or [n_vars]
    """
    values = list(nt)
    if values[0].ndim == 0:
        return jnp.stack(values)
    return jnp.stack(values, axis=1)


class VariableSpec:
    """Specification for model variables with named access.

    Usage:
        spec = VariableSpec(
            state_names=("k", "z"),
            policy_names=("sav_rate",),
        )

        # Unpack
        s = spec.unpack_state(state_array)
        p = spec.unpack_policy(policy_array)

        # Access by name
        capital = s.k
        savings = p.sav_rate

        # Pack back
        new_state = spec.pack_state(s._replace(k=new_k))
    """

    def __init__(
        self,
        state_names: Tuple[str, ...],
        policy_names: Tuple[str, ...],
    ):
        self.state_names = state_names
        self.policy_names = policy_names
        self.n_states = len(state_names)
        self.n_policies = len(policy_names)

        # Create NamedTuple types
        self.StateType = make_state_type(state_names)
        self.PolicyType = make_policy_type(policy_names)

        # Create index lookup dicts
        self.state_idx = {name: i for i, name in enumerate(state_names)}
        self.policy_idx = {name: i for i, name in enumerate(policy_names)}

    def unpack_state(self, arr: Array) -> NamedTuple:
        """Unpack state array into named fields."""
        return unpack_array(arr, self.state_names, self.StateType)

    def unpack_policy(self, arr: Array) -> NamedTuple:
        """Unpack policy array into named fields."""
        return unpack_array(arr, self.policy_names, self.PolicyType)

    def pack_state(self, state: NamedTuple) -> Array:
        """Pack state NamedTuple back into array."""
        return pack_array(state)

    def pack_policy(self, policy: NamedTuple) -> Array:
        """Pack policy NamedTuple back into array."""
        return pack_array(policy)

    def get_state_idx(self, name: str) -> int:
        """Get index for state variable by name."""
        return self.state_idx[name]

    def get_policy_idx(self, name: str) -> int:
        """Get index for policy variable by name."""
        return self.policy_idx[name]


# ---------------------------------------------------------------------------
# Declarative per-variable initial-state distributions
# ---------------------------------------------------------------------------
#
# Matches DEQN-MAO's Variables.py convention where each state can carry
# an ``init: {distribution: ..., kwargs: ...}`` spec and the framework
# assembles an initial-state sampler from them. Using this is optional
# --- a model can still supply a monolithic ``init_state_fn`` directly.
#
# Example:
#
#     INIT_SPECS = {
#         "k": {"distribution": "uniform", "kwargs": {"minval": 0.1, "maxval": 1.0}},
#         "z": {"distribution": "normal",  "kwargs": {"mean": 0.0, "std": 0.04}},
#     }
#     init_state = make_init_state_fn(SPEC.state_names, INIT_SPECS)

import jax

_DISTRIBUTION_SAMPLERS: Dict[str, callable] = {
    # sampler(key, shape, **kwargs) -> Array
    "uniform": lambda key, shape, minval=0.0, maxval=1.0: jax.random.uniform(
        key, shape, minval=minval, maxval=maxval
    ),
    "normal": lambda key, shape, mean=0.0, std=1.0, stddev=None: (
        mean + (std if stddev is None else stddev) * jax.random.normal(key, shape)
    ),
    "lognormal": lambda key, shape, mean=0.0, std=1.0: jnp.exp(
        mean + std * jax.random.normal(key, shape)
    ),
    "truncated_normal": lambda key, shape, lower=-2.0, upper=2.0, mean=0.0, std=1.0: (
        mean + std * jax.random.truncated_normal(key, lower, upper, shape)
    ),
    "constant": lambda key, shape, value=0.0: jnp.full(shape, float(value)),
}


def make_init_state_fn(
    state_names: Tuple[str, ...],
    init_specs: Dict[str, Dict[str, Any]],
) -> callable:
    """Build an init_state_fn from per-variable distributional specs.

    Args:
        state_names: Ordered tuple of state variable names.
        init_specs: Dict mapping state name to a spec
            ``{"distribution": <name>, "kwargs": {...}}``. State names
            without an entry default to ``constant: 0``.

    Returns:
        A function with signature ``(key, batch_size, constants) -> Array``
        producing a ``[batch_size, len(state_names)]`` initial state.
        Constants are not consumed by default (distributions fix their
        kwargs at spec time) but the signature matches ModelSpec's
        ``init_state_fn`` contract so the framework can swap it in.

    Unknown distribution names raise ValueError at build time.
    """

    # Validate early so model-construction errors are obvious.
    for name, spec in init_specs.items():
        if name not in state_names:
            raise ValueError(
                f"init_specs contains unknown state '{name}'. "
                f"Known states: {state_names}"
            )
        dist = spec.get("distribution")
        if dist not in _DISTRIBUTION_SAMPLERS:
            raise ValueError(
                f"Unknown distribution '{dist}' for state '{name}'. "
                f"Available: {sorted(_DISTRIBUTION_SAMPLERS)}"
            )

    def init_state_fn(key, batch_size: int, constants: Dict[str, float]):
        keys = jax.random.split(key, len(state_names))
        columns = []
        for i, name in enumerate(state_names):
            if name in init_specs:
                spec = init_specs[name]
                sampler = _DISTRIBUTION_SAMPLERS[spec["distribution"]]
                kwargs = dict(spec.get("kwargs", {}))
                columns.append(sampler(keys[i], (batch_size,), **kwargs))
            else:
                columns.append(jnp.zeros((batch_size,)))
        return jnp.stack(columns, axis=1)

    return init_state_fn

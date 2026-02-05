"""Variable access helpers - named views over state/policy arrays.

Instead of fragile index slicing like `state[:, 0]`, use:

    s = unpack_state(state, MODEL)
    s.k  # capital
    s.z  # TFP

JAX traces through NamedTuples efficiently.
"""

from typing import Dict, NamedTuple, Tuple, Type
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


# Pre-built specs for models
BROCK_MIRMAN_SPEC = VariableSpec(
    state_names=("k", "z"),
    policy_names=("sav_rate",),
)

DISASTER_SPEC = VariableSpec(
    state_names=(
        "pi_lag", "k_lag", "c_lag", "q_lag", "i_lag", "R_lag",
        "w_tilda_lag", "L_lag", "eps", "mu_ups", "g", "mu_z", "m_p"
    ),
    policy_names=(
        "lambda_z", "i", "pi", "c", "w_tilda", "s", "omega_bar",
        "h", "F_w", "F_p", "q", "L"
    ),
)

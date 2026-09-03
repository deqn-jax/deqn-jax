"""Bit-identity net under the five train-step variants.

The five ``make_grad_step_*`` factories share ``optimizers/_step_common.py``.
Nothing about that sharing is allowed to move an operation or a random-key
split, so this module pins the actual numbers: two training episodes per
variant, compared against reference loss histories and parameter-leaf hashes
recorded from the pre-refactor tree (master 734dcdf, CPU, fp32).

The references live in ``tests/data/step_common_guard.npz``. Re-record them
only when a behavioural change to the variants is intended and stated:

    cd <checkout> && JAX_PLATFORMS=cpu uv run python \\
        tests/test_step_common_guard.py tests/data/step_common_guard.npz

Bit-identity is a property of one machine and one JAX build, not of the code.
The committed references were recorded on **macOS arm64** and the npz carries
that platform key. On the recording platform the test compares exactly — loss
histories bit-for-bit and parameter-leaf hashes. Anywhere else (Linux x86 CI)
the last bits legitimately differ, so the test falls back to a tolerant
comparison of the loss histories (``rtol=1e-5``, ``atol=1e-8``) and skips the
hash comparison, which cannot be made tolerant; the skip message names both
platform keys. The refactor net is therefore exact for the maintainer and a
trajectory check in CI.

Marked ``slow`` (~30 s on CPU): it trains, so it is deselected by the default
``-m "not slow"`` run and executed explicitly in CI's slow lane.
"""

import hashlib
import os
import platform
import sys

import numpy as np
import pytest

REFERENCE = os.path.join(os.path.dirname(__file__), "data", "step_common_guard.npz")


def _platform_key():
    """Identifies the float environment the reference numbers came from.

    Bit-identity is a property of one machine + one JAX build: the same code
    on Linux x86 produces different last bits. The key lets the test decide
    between exact and tolerant comparison.
    """
    import jax

    return "|".join(
        [
            platform.system(),
            platform.machine(),
            f"jax{jax.__version__}",
            f"x64={bool(jax.config.read('jax_enable_x64'))}",
        ]
    )


def _leaf_hash(params):
    import equinox as eqx
    import jax

    leaves = jax.tree.leaves(eqx.filter(params, eqx.is_array))
    h = hashlib.sha256()
    for leaf in leaves:
        a = np.asarray(leaf)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _bm(optimizer_name, gradient_surgery):
    from deqn_jax.config import NetworkConfig, OptimizerConfig, TrainConfig

    return TrainConfig(
        model="brock_mirman",
        episodes=2,
        batch_size=16,
        episode_length=8,
        mc_samples=2,
        seed=0,
        network=NetworkConfig(hidden_sizes=(16,)),
        optimizer=OptimizerConfig(name=optimizer_name, learning_rate=1e-3),
        gradient_surgery=gradient_surgery,
        verbose=False,
    )


def _disaster_composite_pcgrad():
    """The certified arm's recipe, shrunk: composite loss + PCGrad aux path."""
    from deqn_jax.config import TrainConfig

    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        "disaster_gated_pcgrad_bkpin.yaml",
    )
    cfg = TrainConfig.from_yaml(cfg_path)
    return cfg.model_copy(
        update={
            "episodes": 2,
            "batch_size": 16,
            "episode_length": 4,
            "mc_samples": 2,
            "seed": 0,
            "fp64": False,
            "curriculum_episodes": 2,
            "checkpoint_dir": None,
            "tensorboard_dir": None,
            "verbose": False,
            "network": cfg.network.model_copy(update={"hidden_sizes": (16,)}),
            "optimizer": cfg.optimizer.model_copy(update={"lr_warmup": 1}),
        }
    )


def build_cases():
    """Name -> TrainConfig. One case per train-step variant, plus the
    composite/PCGrad aux path which no brock_mirman case reaches."""
    return {
        "bm_standard": _bm("adam", "none"),
        "bm_pcgrad": _bm("adam", "pcgrad"),
        "bm_mao": _bm("mao", "none"),
        "bm_lbfgs": _bm("lbfgs", "none"),
        "bm_gn": _bm("gn", "none"),
        "disaster_composite_pcgrad": _disaster_composite_pcgrad(),
    }


def run_case(cfg):
    from deqn_jax.training.trainer import train_from_config

    params, history = train_from_config(cfg)
    return np.asarray(history["loss"], dtype=np.float64), _leaf_hash(params)


@pytest.mark.slow
@pytest.mark.parametrize("case_name", sorted(build_cases()))
def test_train_step_variant_matches_reference(case_name):
    ref = np.load(REFERENCE, allow_pickle=False)
    ref_platform = str(ref["platform"])
    here = _platform_key()
    exact = here == ref_platform

    losses, param_hash = run_case(build_cases()[case_name])
    ref_losses = ref[f"{case_name}__loss"]

    if exact:
        assert np.array_equal(losses, ref_losses), (
            f"{case_name}: loss history moved (EXACT mode, platform "
            f"{here})\n  got      {losses}\n  expected {ref_losses}"
        )
        assert param_hash == str(ref[f"{case_name}__hash"]), (
            f"{case_name}: trained parameters differ from the reference "
            f"tree (EXACT mode, platform {here})"
        )
        return

    # Different float environment: last bits legitimately differ, so the
    # parameter hash carries no signal here and only the trajectory is
    # checked, loosely.
    assert np.allclose(losses, ref_losses, rtol=1e-5, atol=1e-8), (
        f"{case_name}: loss history moved (TOLERANT mode, rtol=1e-5 "
        f"atol=1e-8; running on {here}, reference recorded on "
        f"{ref_platform})\n  got      {losses}\n  expected {ref_losses}"
    )
    pytest.skip(
        f"{case_name}: loss history matches within tolerance; parameter-hash "
        f"comparison skipped — running on {here}, reference recorded on "
        f"{ref_platform} (hashes cannot be compared with a tolerance)"
    )


def _record(out_path):
    results = {"platform": np.array(_platform_key())}
    for name, cfg in build_cases().items():
        losses, param_hash = run_case(cfg)
        results[f"{name}__loss"] = losses
        results[f"{name}__hash"] = np.array(param_hash)
        print(name, losses, param_hash, flush=True)
    np.savez(out_path, **results)
    print("wrote", out_path, "on", results["platform"])


if __name__ == "__main__":
    _record(sys.argv[1])

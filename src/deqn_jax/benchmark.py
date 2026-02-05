"""Benchmark DEQN-JAX training performance."""

import argparse
import time
from typing import Dict, Tuple

import jax
import jax.numpy as jnp


def benchmark_model(
    model_name: str,
    episodes: int = 500,
    hidden_sizes: Tuple[int, ...] = (64, 64),
    warmup_episodes: int = 10,
    batch_size: int = 64,
    verbose: bool = True,
) -> Dict[str, float]:
    """Benchmark training performance for a model.

    Args:
        model_name: Name of model to benchmark
        episodes: Number of training episodes
        hidden_sizes: MLP hidden layer sizes
        warmup_episodes: Episodes to run before timing (JIT compilation)
        batch_size: Training batch size
        verbose: Print progress

    Returns:
        Dict with timing statistics
    """
    from deqn_jax.training.trainer import train, create_train_state, make_train_step
    import optax

    # Load model
    if model_name == "brock_mirman":
        from deqn_jax.models.brock_mirman import MODEL
    elif model_name == "disaster":
        from deqn_jax.models.disaster import MODEL
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if verbose:
        print(f"Benchmarking {model_name}")
        print(f"  Episodes: {episodes} (+ {warmup_episodes} warmup)")
        print(f"  Hidden sizes: {hidden_sizes}")
        print(f"  Batch size: {batch_size}")
        print()

    # Initialize
    key = jax.random.PRNGKey(42)
    learning_rate = 1e-3

    state = create_train_state(
        MODEL, key,
        hidden_sizes=hidden_sizes,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )

    opt = optax.adam(learning_rate)
    train_step = make_train_step(MODEL, opt, 100, 5, batch_size)

    # Warmup (JIT compilation)
    if verbose:
        print("Warming up (JIT compilation)...")

    warmup_start = time.perf_counter()
    for _ in range(warmup_episodes):
        state, _ = train_step(state)
    jax.block_until_ready(state.params)
    warmup_time = time.perf_counter() - warmup_start

    if verbose:
        print(f"  Warmup time: {warmup_time:.2f}s ({warmup_time/warmup_episodes*1000:.1f}ms/episode)")
        print()

    # Timed run
    if verbose:
        print("Running benchmark...")

    losses = []
    start_time = time.perf_counter()

    for ep in range(episodes):
        state, metrics = train_step(state)
        losses.append(float(metrics.loss))

        if verbose and (ep + 1) % 100 == 0:
            elapsed = time.perf_counter() - start_time
            eps_per_sec = (ep + 1) / elapsed
            print(f"  Episode {ep + 1:4d} | Loss: {metrics.loss:.4e} | {eps_per_sec:.1f} ep/s")

    jax.block_until_ready(state.params)
    total_time = time.perf_counter() - start_time

    # Statistics
    results = {
        "model": model_name,
        "episodes": episodes,
        "warmup_episodes": warmup_episodes,
        "warmup_time_s": warmup_time,
        "total_time_s": total_time,
        "time_per_episode_ms": total_time / episodes * 1000,
        "episodes_per_second": episodes / total_time,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_reduction": losses[0] / losses[-1] if losses[-1] > 0 else float("inf"),
    }

    if verbose:
        print()
        print("Results:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Per episode: {results['time_per_episode_ms']:.2f}ms")
        print(f"  Throughput: {results['episodes_per_second']:.1f} episodes/sec")
        print(f"  Initial loss: {results['initial_loss']:.4e}")
        print(f"  Final loss: {results['final_loss']:.4e}")
        print(f"  Loss reduction: {results['loss_reduction']:.1f}x")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark DEQN-JAX training")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="brock_mirman",
        choices=["brock_mirman", "disaster"],
        help="Model to benchmark",
    )
    parser.add_argument(
        "--episodes", "-n",
        type=int,
        default=500,
        help="Number of episodes",
    )
    parser.add_argument(
        "--hidden",
        type=str,
        default="64,64",
        help="Hidden layer sizes",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup episodes",
    )
    parser.add_argument(
        "--fp64",
        action="store_true",
        help="Use float64 precision",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output",
    )

    args = parser.parse_args()

    if args.fp64:
        jax.config.update("jax_enable_x64", True)

    hidden_sizes = tuple(int(x) for x in args.hidden.split(","))

    results = benchmark_model(
        model_name=args.model,
        episodes=args.episodes,
        hidden_sizes=hidden_sizes,
        warmup_episodes=args.warmup,
        batch_size=args.batch_size,
        verbose=not args.quiet,
    )

    if args.quiet:
        print(f"{results['episodes_per_second']:.1f} ep/s, final loss: {results['final_loss']:.4e}")


if __name__ == "__main__":
    main()

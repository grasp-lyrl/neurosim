"""
Benchmark script for RotorpyDynamics performance.

Tests the step function performance under various conditions.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from neurosim.core.dynamics import create_dynamics, DynamicsType


def benchmark_step_function(
    dynamics_type: DynamicsType = DynamicsType.ROTORPY_MULTIROTOR_EULER,
    vehicle: str = "crazyflie",
    n_steps: int = 10000,
    dt: float = 0.001,
) -> dict:
    """
    Benchmark the step function performance.

    Args:
        dynamics_type: Type of dynamics model
        vehicle: Vehicle type
        n_steps: Number of steps to simulate
        dt: Time step size

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmarking {dynamics_type.value} with {vehicle}")
    print(f"{'=' * 60}")

    # Initialize dynamics
    initial_state = {
        "x": np.array([0.0, 0.0, 1.0]),
        "q": np.array([0.0, 0.0, 0.0, 1.0]),
        "v": np.zeros(3),
        "w": np.zeros(3),
    }

    dynamics = create_dynamics(
        model=dynamics_type, vehicle=vehicle, initial_state=initial_state
    )

    # Simple control input (hover thrust)
    control = {"cmd_motor_speeds": np.array([2500.0, 2500.0, 2500.0, 2500.0])}

    # Warmup
    print("Warming up...")
    for _ in range(100):
        dynamics.step(control, dt)

    # Benchmark
    print(f"Running {n_steps} steps...")
    latencies = []
    start_total = time.perf_counter()

    for i in range(n_steps):
        start = time.perf_counter()
        dynamics.step(control, dt)
        latencies.append(time.perf_counter() - start)

        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"  Step {i + 1}/{n_steps}", end="\r")

    total_time = time.perf_counter() - start_total
    print(f"\nCompleted {n_steps} steps")

    # Compute statistics
    latencies_us = np.array(latencies) * 1e6  # Convert to microseconds
    stats = {
        "dynamics_type": dynamics_type.value,
        "vehicle": vehicle,
        "n_steps": n_steps,
        "dt": dt,
        "total_time_s": total_time,
        "avg_latency_us": np.mean(latencies_us),
        "median_latency_us": np.median(latencies_us),
        "min_latency_us": np.min(latencies_us),
        "max_latency_us": np.max(latencies_us),
        "std_latency_us": np.std(latencies_us),
        "p95_latency_us": np.percentile(latencies_us, 95),
        "p99_latency_us": np.percentile(latencies_us, 99),
        "steps_per_second": n_steps / total_time,
        "max_real_time_factor": 1.0 / (np.mean(latencies_us) * 1e-6 / dt),
        "latencies_us": latencies_us,
    }

    return stats


def print_statistics(stats: dict) -> None:
    """Pretty print benchmark statistics."""
    print(f"\n{'=' * 60}")
    print("BENCHMARK RESULTS")
    print(f"{'=' * 60}")
    print(f"Model:                  {stats['dynamics_type']}")
    print(f"Vehicle:                {stats['vehicle']}")
    print(f"Steps:                  {stats['n_steps']}")
    print(f"Time step (dt):         {stats['dt'] * 1000:.3f} ms")
    print("\nPerformance:")
    print(f"  Total time:           {stats['total_time_s']:.3f} s")
    print(f"  Steps per second:     {stats['steps_per_second']:.1f} Hz")
    print(f"  Avg step time:        {stats['avg_latency_us']:.2f} μs")
    print(f"  Median step time:     {stats['median_latency_us']:.2f} μs")
    print(f"  Min step time:        {stats['min_latency_us']:.2f} μs")
    print(f"  Max step time:        {stats['max_latency_us']:.2f} μs")
    print(f"  Std deviation:        {stats['std_latency_us']:.2f} μs")
    print(f"  95th percentile:      {stats['p95_latency_us']:.2f} μs")
    print(f"  99th percentile:      {stats['p99_latency_us']:.2f} μs")
    print("\nReal-time capability:")
    print(f"  Max RT factor:        {stats['max_real_time_factor']:.1f}x")
    print(
        f"  (Can run up to {stats['max_real_time_factor']:.1f}x faster than real-time)"
    )
    print(f"{'=' * 60}")


def plot_latency_distribution(stats: dict, save_path: str = None) -> None:
    """Plot latency distribution histogram."""
    latencies = stats["latencies_us"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(latencies, bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(
        stats["avg_latency_us"], color="r", linestyle="--", label="Mean", linewidth=2
    )
    axes[0].axvline(
        stats["median_latency_us"],
        color="g",
        linestyle="--",
        label="Median",
        linewidth=2,
    )
    axes[0].set_xlabel("Latency (μs)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Step Latency Distribution\n{stats['dynamics_type']}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Time series (first 1000 steps)
    n_plot = min(1000, len(latencies))
    axes[1].plot(latencies[:n_plot], alpha=0.7, linewidth=0.5)
    axes[1].axhline(
        stats["avg_latency_us"], color="r", linestyle="--", label="Mean", linewidth=2
    )
    axes[1].set_xlabel("Step Number")
    axes[1].set_ylabel("Latency (μs)")
    axes[1].set_title(f"Step Latency Over Time (first {n_plot} steps)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {save_path}")

    plt.show()


def compare_dynamics_models() -> None:
    """Compare performance of different dynamics models."""
    print("\n" + "=" * 60)
    print("COMPARING DYNAMICS MODELS")
    print("=" * 60)

    results = []

    # Test different models
    models = [
        DynamicsType.ROTORPY_MULTIROTOR_EULER,
        DynamicsType.ROTORPY_MULTIROTOR,
    ]

    for model in models:
        try:
            stats = benchmark_step_function(
                dynamics_type=model, vehicle="crazyflie", n_steps=10000, dt=0.001
            )
            print_statistics(stats)
            results.append(stats)
        except Exception as e:
            print(f"Error testing {model.value}: {e}")

    # Comparison plot
    if len(results) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))

        models_names = [r["dynamics_type"] for r in results]
        avg_latencies = [r["avg_latency_us"] for r in results]
        std_latencies = [r["std_latency_us"] for r in results]

        x = np.arange(len(models_names))
        bars = ax.bar(x, avg_latencies, yerr=std_latencies, capsize=5, alpha=0.7)

        # Color bars by performance
        colors = plt.cm.RdYlGn_r(np.array(avg_latencies) / max(avg_latencies))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xlabel("Dynamics Model")
        ax.set_ylabel("Average Step Latency (μs)")
        ax.set_title("Dynamics Model Performance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(models_names, rotation=15, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, avg_latencies)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.1f} μs",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig("plots/dynamics_comparison.png", dpi=150, bbox_inches="tight")
        print("\nComparison plot saved to: plots/dynamics_comparison.png")
        plt.show()


def main():
    """Run benchmarks."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark RotorpyDynamics")
    parser.add_argument(
        "--model",
        type=str,
        default="rotorpy_multirotor_euler",
        choices=[
            "rotorpy_multirotor",
            "rotorpy_multirotor_euler",
            "rotorpy_px4_multirotor",
        ],
        help="Dynamics model to test",
    )
    parser.add_argument(
        "--vehicle",
        type=str,
        default="crazyflie",
        choices=["crazyflie", "hummingbird"],
        help="Vehicle type",
    )
    parser.add_argument(
        "--steps", type=int, default=10000, help="Number of steps to simulate"
    )
    parser.add_argument(
        "--dt", type=float, default=0.001, help="Time step size in seconds"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare different dynamics models",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate latency distribution plot"
    )

    args = parser.parse_args()

    # Create plots directory if needed
    Path("plots").mkdir(exist_ok=True)

    if args.compare:
        compare_dynamics_models()
    else:
        # Single model benchmark
        dynamics_type = DynamicsType(args.model.lower())
        stats = benchmark_step_function(
            dynamics_type=dynamics_type,
            vehicle=args.vehicle,
            n_steps=args.steps,
            dt=args.dt,
        )
        print_statistics(stats)

        if args.plot:
            plot_path = f"plots/latency_{args.model}_{args.vehicle}.png"
            plot_latency_distribution(stats, save_path=plot_path)


if __name__ == "__main__":
    main()

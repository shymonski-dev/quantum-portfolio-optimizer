"""CLI entry point for Phase II benchmarking."""

from quantum_portfolio_optimizer.benchmarks.phase2 import run_phase2_benchmark


def main() -> None:
    run_phase2_benchmark()


if __name__ == "__main__":
    main()

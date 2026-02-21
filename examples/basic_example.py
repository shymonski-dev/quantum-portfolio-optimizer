"""Minimal 3-asset optimisation example."""

from __future__ import annotations

import numpy as np
from qiskit.primitives import StatevectorEstimator, StatevectorSampler

from quantum_portfolio_optimizer.core.qubo_formulation import PortfolioQUBO
from quantum_portfolio_optimizer.core.vqe_solver import PortfolioVQESolver
from quantum_portfolio_optimizer.data.sample_datasets import generate_synthetic_dataset


def main() -> None:
    dataset = generate_synthetic_dataset(num_assets=3, num_points=20, seed=42)
    mu = dataset.returns.mean().values
    covariance = np.cov(dataset.returns.values.T)

    progress_snapshots: list[tuple[int, float, float]] = []

    def progress_callback(evaluation: int, energy: float, best_so_far: float) -> None:
        if evaluation % 100 == 0:
            progress_snapshots.append((evaluation, energy, best_so_far))

    builder = PortfolioQUBO(
        expected_returns=mu,
        covariance=covariance,
        budget=1.0,
        risk_aversion=0.5,
        time_steps=1,
        resolution_qubits=1,
        max_investment=1.0,
        penalty_strength=25.0,
    )
    qubo = builder.build()
    
    # Use Statevector primitives for simulation
    estimator = StatevectorEstimator(seed=123)
    sampler = StatevectorSampler(seed=123)
    
    solver = PortfolioVQESolver(
        estimator=estimator,
        sampler=sampler,
        seed=123,
        progress_callback=progress_callback
    )
    result = solver.solve(qubo)

    print("Optimal energy:", result.optimal_value)
    print("Number of evaluations:", result.num_evaluations)
    print("Converged:", result.converged, "-", result.optimizer_message)
    print("Ansatz report:", result.ansatz_report)
    print("Tracked progress (every 100 evaluations):", progress_snapshots[:5])

    best_bitstring = None
    if result.history:
        best_bitstring = np.argmin(result.history)
    print("Best iteration index:", best_bitstring)


if __name__ == "__main__":
    main()

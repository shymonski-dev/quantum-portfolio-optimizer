"""Lightweight validation routines for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core.qubo_formulation import PortfolioQUBO
from ..core.vqe_solver import PortfolioVQESolver
from ..data.sample_datasets import generate_synthetic_dataset
from ..simulation.provider import get_provider


@dataclass
class ValidationReport:
    optimal_value: float
    feasible: bool
    num_evaluations: int


def validate_small_instance(seed: Optional[int] = None) -> ValidationReport:
    dataset = generate_synthetic_dataset(num_assets=3, num_points=16, seed=seed)
    returns = dataset.returns.mean().values
    covariance = np.cov(dataset.returns.values.T)

    builder = PortfolioQUBO(
        expected_returns=returns,
        covariance=covariance,
        budget=1.0,
        risk_aversion=0.5,
        time_steps=1,
        resolution_qubits=1,
        max_investment=1.0,
    )
    qubo = builder.build()
    estimator, _ = get_provider({"name": "local_simulator", "shots": None, "seed": seed})
    solver = PortfolioVQESolver(estimator=estimator, seed=seed)
    result = solver.solve(qubo)

    feasible = result.optimal_value <= 10.0  # simple sanity bound for tests
    return ValidationReport(optimal_value=result.optimal_value, feasible=feasible, num_evaluations=result.num_evaluations)

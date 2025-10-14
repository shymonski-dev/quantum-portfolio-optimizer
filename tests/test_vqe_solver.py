import numpy as np

from quantum_portfolio_optimizer.core.optimizer_interface import DifferentialEvolutionConfig
from quantum_portfolio_optimizer.core.qubo_formulation import PortfolioQUBO
from quantum_portfolio_optimizer.core.vqe_solver import PortfolioVQESolver


def test_vqe_solver_runs_on_small_qubo():
    returns = np.array([0.02, 0.015])
    covariance = np.array([[0.1, 0.02], [0.02, 0.08]])
    qubo = PortfolioQUBO(
        expected_returns=returns,
        covariance=covariance,
        budget=1.0,
        risk_aversion=0.5,
        time_steps=1,
        resolution_qubits=1,
        max_investment=1.0,
    ).build()

    solver = PortfolioVQESolver(
        ansatz_options={"reps": 1},
        optimizer_config=DifferentialEvolutionConfig(bounds=[(-1.0, 1.0)], maxiter=5, popsize=6, seed=1),
        seed=1,
    )
    result = solver.solve(qubo)

    assert np.isfinite(result.optimal_value)
    assert result.num_evaluations > 0
    assert len(result.history) == result.num_evaluations

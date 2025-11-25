import numpy as np

from quantum_portfolio_optimizer.core.optimizer_interface import DifferentialEvolutionConfig
from quantum_portfolio_optimizer.core.qubo_formulation import PortfolioQUBO
from quantum_portfolio_optimizer.core.vqe_solver import PortfolioVQESolver
from quantum_portfolio_optimizer.simulation.provider import get_provider


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

    backend_config = {"name": "local_simulator", "shots": None, "seed": 1}
    estimator, _ = get_provider(backend_config)

    solver = PortfolioVQESolver(
        estimator=estimator,
        ansatz_options={"reps": 1},
        optimizer_config=DifferentialEvolutionConfig(bounds=[(-1.0, 1.0)], maxiter=5, popsize=6, seed=1),
        seed=1,
    )
    result = solver.solve(qubo)

    assert np.isfinite(result.optimal_value)
    assert result.num_evaluations > 0
    assert len(result.history) == result.num_evaluations
    assert len(result.best_history) == result.num_evaluations
    assert all(a <= b for a, b in zip(result.best_history, result.history))
    assert isinstance(result.converged, bool)


def test_progress_callback_receives_updates():
    returns = np.array([0.01, 0.02])
    covariance = np.eye(2) * 0.05
    qubo = PortfolioQUBO(
        expected_returns=returns,
        covariance=covariance,
        budget=1.0,
        risk_aversion=0.3,
        time_steps=1,
        resolution_qubits=1,
        max_investment=1.0,
    ).build()

    progress_updates = []

    def callback(evaluation: int, energy: float, best: float) -> None:
        progress_updates.append((evaluation, energy, best))

    backend_config = {"name": "local_simulator", "shots": None, "seed": 2}
    estimator, _ = get_provider(backend_config)

    solver = PortfolioVQESolver(
        estimator=estimator,
        ansatz_options={"reps": 1},
        optimizer_config=DifferentialEvolutionConfig(bounds=[(-1.0, 1.0)], maxiter=3, popsize=4, seed=2),
        seed=2,
        progress_callback=callback,
    )
    result = solver.solve(qubo)

    assert progress_updates, "Progress callback should receive updates"
    assert progress_updates[-1][0] == result.num_evaluations
    assert progress_updates[-1][2] == result.best_history[-1]

from quantum_portfolio_optimizer.core.optimizer_interface import DifferentialEvolutionConfig
from quantum_portfolio_optimizer.core.qubo_formulation import PortfolioQUBO
from quantum_portfolio_optimizer.core.vqe_solver import PortfolioVQESolver
from quantum_portfolio_optimizer.data.sample_datasets import generate_synthetic_dataset
from quantum_portfolio_optimizer.simulation.provider import get_provider


def test_end_to_end_pipeline():
    dataset = generate_synthetic_dataset(num_assets=3, num_points=12, seed=7)
    mu = dataset.returns.mean().values
    covariance = dataset.returns.cov().values

    qubo = PortfolioQUBO(
        expected_returns=mu,
        covariance=covariance,
        budget=1.0,
        risk_aversion=0.4,
        time_steps=1,
        resolution_qubits=1,
        max_investment=1.0,
        penalty_strength=15.0,
    ).build()

    backend_config = {"name": "local_simulator", "shots": None, "seed": 123}
    estimator, _ = get_provider(backend_config)

    solver = PortfolioVQESolver(
        estimator=estimator,
        ansatz_options={"reps": 1},
        optimizer_config=DifferentialEvolutionConfig(bounds=[(-1, 1)], maxiter=3, popsize=4, seed=123),
        seed=123,
    )
    result = solver.solve(qubo)

    assert result.optimal_value < 5.0
    assert len(result.best_history) == result.num_evaluations
    assert result.converged in {True, False}

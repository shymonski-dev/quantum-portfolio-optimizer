import numpy as np

from quantum_portfolio_optimizer.core.qubo_formulation import PortfolioQUBO


def test_qubo_small_instance():
    expected_returns = np.array([0.02, 0.015, 0.01])
    covariance = np.array(
        [
            [0.1, 0.02, 0.01],
            [0.02, 0.08, 0.015],
            [0.01, 0.015, 0.05],
        ]
    )
    builder = PortfolioQUBO(
        expected_returns=expected_returns,
        covariance=covariance,
        budget=1.0,
        risk_aversion=0.4,
        time_steps=1,
        resolution_qubits=1,
        max_investment=1.0,
    )
    qubo = builder.build()

    assert qubo.linear.shape[0] == 3
    assert qubo.quadratic.shape == (3, 3)

    h, j, offset = qubo.to_ising()
    assert h.shape[0] == 3
    assert j.shape == (3, 3)
    assert isinstance(offset, float)

    pauli = qubo.to_pauli()
    assert pauli.num_qubits == 3


def test_qubo_time_step_budget_penalty():
    builder = PortfolioQUBO(
        expected_returns=np.zeros(1),
        covariance=np.zeros((1, 1)),
        budget=1.0,
        risk_aversion=0.0,
        transaction_cost=0.0,
        time_steps=2,
        resolution_qubits=1,
        max_investment=1.0,
        penalty_strength=10.0,
        enforce_budget=False,
        time_step_budgets=[0.5, 0.5],
        time_budget_penalty=20.0,
    )
    qubo = builder.build()

    def energy(bits):
        x = np.array(bits, dtype=float)
        return float(qubo.offset + qubo.linear @ x + x @ qubo.quadratic @ x)

    match_budget = energy([1, 1])
    deviate = energy([0, 0])

    assert match_budget < deviate


def test_qubo_asset_limit_penalty():
    builder = PortfolioQUBO(
        expected_returns=np.zeros(2),
        covariance=np.zeros((2, 2)),
        budget=1.0,
        risk_aversion=0.0,
        time_steps=1,
        resolution_qubits=1,
        max_investment=0.5,
        penalty_strength=10.0,
        enforce_budget=False,
        asset_max_allocation=[0.5, 0.5],
        asset_penalty_strength=30.0,
    )
    qubo = builder.build()

    def energy(bits):
        x = np.array(bits, dtype=float)
        return float(qubo.offset + qubo.linear @ x + x @ qubo.quadratic @ x)

    match_limits = energy([1, 1])
    deviate = energy([0, 0])

    assert match_limits < deviate

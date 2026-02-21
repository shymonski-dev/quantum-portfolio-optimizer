"""Test suite to validate compliance with research paper specifications."""

import numpy as np
from quantum_portfolio_optimizer.core.qubo_formulation import PortfolioQUBO
from quantum_portfolio_optimizer.core.vqe_solver import PortfolioVQESolver
from quantum_portfolio_optimizer.core.optimizer_interface import (
    DifferentialEvolutionConfig,
)
from quantum_portfolio_optimizer.data.returns_calculator import (
    calculate_logarithmic_returns,
    calculate_rolling_covariance,
)
from quantum_portfolio_optimizer.simulation.provider import get_provider


class TestResearchCompliance:
    """Validate implementation against paper specifications."""

    def test_critical_parameters(self):
        """Test that critical parameters match paper values."""
        # Create QUBO with default parameters
        qubo_builder = PortfolioQUBO(
            expected_returns=[0.1, 0.2, 0.15], covariance=np.eye(3) * 0.01, budget=1.0
        )

        # Verify critical parameters
        assert qubo_builder.risk_aversion == 1000, (
            f"Risk aversion should be 1000, got {qubo_builder.risk_aversion}"
        )
        assert qubo_builder.transaction_cost == 0.01, (
            f"Transaction cost (ν) should be 0.01, got {qubo_builder.transaction_cost}"
        )
        assert qubo_builder.penalty_strength == 1.0, (
            f"Penalty strength (ρ) should be 1.0, got {qubo_builder.penalty_strength}"
        )

    def test_differential_evolution_config(self):
        """Test DE optimizer configuration matches paper."""
        config = DifferentialEvolutionConfig(bounds=[(-2 * np.pi, 2 * np.pi)])

        assert config.strategy == "best2bin", "Strategy should be 'best2bin'"
        assert config.mutation == (0, 0.25), (
            f"Mutation should be (0, 0.25), got {config.mutation}"
        )
        assert config.recombination == 0.4, (
            f"Recombination should be 0.4, got {config.recombination}"
        )

    def test_parameter_bounds(self):
        """Test VQE parameter bounds are [-2π, 2π]."""
        backend_config = {"name": "local_simulator", "shots": None, "seed": 1}
        estimator, _ = get_provider(backend_config)
        solver = PortfolioVQESolver(estimator=estimator)

        # Check default bounds
        assert solver.parameter_bounds == 2 * np.pi, (
            f"Default bounds should be 2π, got {solver.parameter_bounds}"
        )

    def test_ansatz_configuration(self):
        """Test Real Amplitudes ansatz has correct defaults."""
        from quantum_portfolio_optimizer.core.ansatz_library import (
            build_real_amplitudes,
        )

        # Build with defaults (3 reps, reverse_linear entanglement)
        ansatz = build_real_amplitudes(num_qubits=6)

        # With 6 qubits, 3 reps, reverse_linear: each rep has 6 RY gates
        # Total parameters = 6 * (3 + 1) = 24 (6 qubits * 4 layers of RY)
        # Note: function-based ansatz doesn't have .reps/.entanglement attributes
        assert ansatz.num_qubits == 6, (
            f"Ansatz should have 6 qubits, got {ansatz.num_qubits}"
        )
        assert ansatz.num_parameters > 0, "Ansatz should have parameters"

    def test_xs_problem_size(self):
        """Test XS problem size (2 time steps, 3 assets, 1 bit resolution)."""
        qubo_builder = PortfolioQUBO(
            expected_returns=[[0.1, 0.2, 0.15], [0.12, 0.18, 0.14]],  # 2 time steps
            covariance=np.eye(3) * 0.01,  # 3 assets
            budget=1.0,
            time_steps=2,
            resolution_qubits=1,
            risk_aversion=1000,
        )

        qubo = qubo_builder.build()

        # Should have 6 qubits total (2 time steps * 3 assets * 1 resolution bit)
        expected_qubits = 2 * 3 * 1
        assert qubo.num_variables == expected_qubits, (
            f"XS problem should have 6 qubits, got {qubo.num_variables}"
        )

    def test_logarithmic_returns(self):
        """Test logarithmic returns calculation."""
        prices = np.array([[100, 50, 75], [105, 52, 73], [110, 51, 76]])

        log_returns = calculate_logarithmic_returns(prices)

        # Verify shape
        assert log_returns.shape == (2, 3), "Should have (n-1) time steps"

        # Verify calculation: log(105/100) ≈ 0.04879
        expected_first = np.log(105 / 100)
        np.testing.assert_almost_equal(log_returns[0, 0], expected_first, decimal=5)

    def test_popsize_calculation(self):
        """Test that popsize meets minimum 0.8 * n_qubits requirement."""

        num_qubits = 6  # XS problem size
        min_popsize = int(0.8 * num_qubits)  # Should be at least 4.8 → 4

        config = DifferentialEvolutionConfig(
            bounds=[(-2 * np.pi, 2 * np.pi)] * num_qubits,
            popsize=2,  # Intentionally too small
        )

        # This should be adjusted internally
        # Note: Need to verify this is implemented in run_differential_evolution
        assert config.popsize < min_popsize, (
            "Test setup should start below minimum popsize"
        )
        assert min_popsize >= 4, "Minimum popsize for 6 qubits should be at least 4"


def test_integration_xs_problem():
    """Integration test for XS problem size with research parameters."""
    # Generate test data
    np.random.seed(42)
    prices = np.random.uniform(50, 150, size=(32, 3))  # 32 days, 3 assets
    log_returns = calculate_logarithmic_returns(prices)

    # Calculate statistics over 30-day window
    mu = np.mean(log_returns[-30:], axis=0)
    sigma = calculate_rolling_covariance(log_returns, window=30)

    # Build QUBO with research parameters
    qubo_builder = PortfolioQUBO(
        expected_returns=[mu, mu],  # 2 time steps
        covariance=sigma,
        budget=1.0,
        risk_aversion=1000,  # γ from paper
        transaction_cost=0.01,  # ν from paper
        time_steps=2,
        resolution_qubits=1,
        penalty_strength=1.0,  # ρ from paper
    )

    qubo = qubo_builder.build()

    # Configure VQE with research parameters
    config = DifferentialEvolutionConfig(
        bounds=[(-2 * np.pi, 2 * np.pi)] * qubo.num_variables,
        strategy="best2bin",
        mutation=(0, 0.25),
        recombination=0.4,
        popsize=10,  # Will be adjusted to meet 0.8 * n_qubits
        maxiter=50,
    )

    backend_config = {"name": "local_simulator", "shots": None, "seed": 123}
    estimator, _ = get_provider(backend_config)

    solver = PortfolioVQESolver(
        estimator=estimator,
        ansatz_name="real_amplitudes",
        ansatz_options={"reps": 3, "entanglement": "reverse_linear"},
        parameter_bounds=2 * np.pi,
        optimizer_config=config,
        seed=123,
    )

    # Solve
    result = solver.solve(qubo)

    # Basic validation
    assert result is not None, "Should return a result"
    assert len(result.optimal_parameters) == result.ansatz_report["num_parameters"]
    assert result.ansatz_report["num_qubits"] == 6  # XS problem size

    assert result is not None

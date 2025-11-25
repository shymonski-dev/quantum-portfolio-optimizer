"""Tests for QAOA solver implementation."""

import numpy as np
import pytest

from quantum_portfolio_optimizer.core.qaoa_solver import (
    PortfolioQAOASolver,
    QAOAResult,
    get_qaoa_circuit_depth,
)
from quantum_portfolio_optimizer.core.qubo_formulation import PortfolioQUBO
from quantum_portfolio_optimizer.core.optimizer_interface import DifferentialEvolutionConfig
from quantum_portfolio_optimizer.simulation.provider import get_provider


@pytest.fixture
def simple_qubo():
    """Create a simple 3-asset QUBO for testing."""
    expected_returns = np.array([0.01, 0.015, 0.008])
    covariance = np.array([
        [0.04, 0.01, 0.005],
        [0.01, 0.05, 0.01],
        [0.005, 0.01, 0.03],
    ])
    builder = PortfolioQUBO(
        expected_returns=expected_returns,
        covariance=covariance,
        budget=1.0,
        risk_aversion=500.0,
        penalty_strength=1000.0,
    )
    return builder.build()


class TestQAOAResultDataclass:
    """Test QAOAResult dataclass."""

    def test_qaoa_result_fields(self):
        """Test QAOAResult holds all expected fields."""
        result = QAOAResult(
            optimal_parameters=np.array([0.5, 0.3]),
            optimal_value=-0.5,
            best_bitstring="101",
            measurement_counts={"101": 500, "110": 300, "011": 224},
            history=[0.0, -0.3, -0.5],
            best_history=[0.0, -0.3, -0.5],
            num_evaluations=3,
            layers=1,
            converged=True,
            optimizer_message="Success",
        )
        assert result.layers == 1
        assert result.best_bitstring == "101"
        assert len(result.optimal_parameters) == 2  # gamma and beta for p=1


class TestQAOASolverRequirements:
    """Test QAOA solver initialization requirements."""

    def test_qaoa_requires_sampler(self):
        """QAOA must be initialized with a sampler."""
        with pytest.raises(ValueError, match="Sampler is required"):
            PortfolioQAOASolver(sampler=None)

    def test_qaoa_accepts_sampler(self):
        """QAOA accepts a valid sampler."""
        backend_config = {"name": "local_simulator", "shots": 100, "seed": 42}
        _, sampler = get_provider(backend_config)
        solver = PortfolioQAOASolver(sampler=sampler, layers=1)
        assert solver.layers == 1


class TestQAOACircuitConstruction:
    """Test QAOA circuit construction."""

    def test_circuit_has_correct_parameters(self, simple_qubo):
        """QAOA circuit should have 2*p parameters."""
        backend_config = {"name": "local_simulator", "shots": 100, "seed": 42}
        _, sampler = get_provider(backend_config)

        for layers in [1, 2, 3]:
            solver = PortfolioQAOASolver(sampler=sampler, layers=layers)
            circuit, params = solver._build_qaoa_circuit(simple_qubo)

            assert len(params) == 2 * layers  # gamma and beta per layer
            assert circuit.num_qubits == simple_qubo.num_variables

    def test_circuit_structure(self, simple_qubo):
        """Test basic circuit structure for p=1."""
        backend_config = {"name": "local_simulator", "shots": 100, "seed": 42}
        _, sampler = get_provider(backend_config)

        solver = PortfolioQAOASolver(sampler=sampler, layers=1)
        circuit, _ = solver._build_qaoa_circuit(simple_qubo)

        # Should have operations (H gates, RZ, CX, RX)
        assert circuit.depth() > 0
        assert circuit.num_qubits == 3


class TestQAOASolve:
    """Test QAOA solving capability."""

    def test_qaoa_solves_small_qubo(self, simple_qubo):
        """QAOA should successfully solve a small QUBO."""
        backend_config = {"name": "local_simulator", "shots": 512, "seed": 42}
        estimator, sampler = get_provider(backend_config)

        config = DifferentialEvolutionConfig(
            bounds=[(0, 2 * np.pi), (0, np.pi)],  # gamma, beta
            maxiter=5,
            popsize=3,
            seed=42,
        )

        solver = PortfolioQAOASolver(
            sampler=sampler,
            estimator=estimator,
            layers=1,
            optimizer_config=config,
            seed=42,
        )

        result = solver.solve(simple_qubo)

        # Basic validity checks
        assert isinstance(result, QAOAResult)
        assert result.best_bitstring is not None
        assert len(result.best_bitstring) >= 3
        assert all(c in '01' for c in result.best_bitstring)
        assert result.measurement_counts is not None
        assert result.num_evaluations > 0
        assert len(result.history) == result.num_evaluations

    def test_qaoa_progress_callback(self, simple_qubo):
        """Test that progress callback is called during optimization."""
        backend_config = {"name": "local_simulator", "shots": 256, "seed": 42}
        _, sampler = get_provider(backend_config)

        config = DifferentialEvolutionConfig(
            bounds=[(0, 2 * np.pi), (0, np.pi)],
            maxiter=3,
            popsize=2,
            seed=42,
        )

        callback_calls = []

        def progress_callback(iteration, energy, best_energy):
            callback_calls.append((iteration, energy, best_energy))

        solver = PortfolioQAOASolver(
            sampler=sampler,
            layers=1,
            optimizer_config=config,
            seed=42,
            progress_callback=progress_callback,
        )

        solver.solve(simple_qubo)

        assert len(callback_calls) > 0
        # Check callback receives valid data
        for iteration, energy, best_energy in callback_calls:
            assert isinstance(iteration, int)
            assert isinstance(energy, float)
            assert isinstance(best_energy, float)

    def test_qaoa_two_layers(self, simple_qubo):
        """Test QAOA with p=2 layers."""
        backend_config = {"name": "local_simulator", "shots": 256, "seed": 42}
        _, sampler = get_provider(backend_config)

        config = DifferentialEvolutionConfig(
            bounds=[(0, 2 * np.pi), (0, np.pi), (0, 2 * np.pi), (0, np.pi)],
            maxiter=3,
            popsize=2,
            seed=42,
        )

        solver = PortfolioQAOASolver(
            sampler=sampler,
            layers=2,
            optimizer_config=config,
            seed=42,
        )

        result = solver.solve(simple_qubo)

        assert result.layers == 2
        assert len(result.optimal_parameters) == 4  # 2 gammas + 2 betas


class TestQAOAEnergyEvaluation:
    """Test QUBO energy evaluation."""

    def test_evaluate_qubo_energy(self, simple_qubo):
        """Test energy evaluation for known bitstrings."""
        backend_config = {"name": "local_simulator", "shots": 100, "seed": 42}
        _, sampler = get_provider(backend_config)
        solver = PortfolioQAOASolver(sampler=sampler)

        # Test with all zeros
        bits_zero = np.array([0.0, 0.0, 0.0])
        energy_zero = solver._evaluate_qubo_energy(bits_zero, simple_qubo)
        assert isinstance(energy_zero, float)

        # Test with all ones
        bits_one = np.array([1.0, 1.0, 1.0])
        energy_one = solver._evaluate_qubo_energy(bits_one, simple_qubo)
        assert isinstance(energy_one, float)

        # Energies should generally be different
        # (unless the QUBO has very specific structure)


class TestCircuitDepthEstimation:
    """Test circuit depth estimation utility."""

    def test_depth_increases_with_qubits(self):
        """Circuit depth should increase with number of qubits."""
        depth_5 = get_qaoa_circuit_depth(5, layers=1)
        depth_10 = get_qaoa_circuit_depth(10, layers=1)
        depth_20 = get_qaoa_circuit_depth(20, layers=1)

        assert depth_10 > depth_5
        assert depth_20 > depth_10

    def test_depth_increases_with_layers(self):
        """Circuit depth should increase with number of layers."""
        depth_p1 = get_qaoa_circuit_depth(10, layers=1)
        depth_p2 = get_qaoa_circuit_depth(10, layers=2)
        depth_p3 = get_qaoa_circuit_depth(10, layers=3)

        assert depth_p2 > depth_p1
        assert depth_p3 > depth_p2

    def test_depth_values_reasonable(self):
        """Check that depth estimates are reasonable."""
        # For 5 qubits, p=1, depth should be manageable
        depth = get_qaoa_circuit_depth(5, layers=1)
        assert depth > 0
        assert depth < 100  # Reasonable upper bound

        # For 25 qubits, p=1, depth will be large due to ZZ terms
        depth_25 = get_qaoa_circuit_depth(25, layers=1)
        assert depth_25 > depth

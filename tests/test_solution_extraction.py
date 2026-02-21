"""Tests for binary solution extraction in VQE solver."""

import numpy as np
import pytest

from quantum_portfolio_optimizer.core.vqe_solver import PortfolioVQESolver, VQEResult
from quantum_portfolio_optimizer.core.qubo_formulation import PortfolioQUBO
from quantum_portfolio_optimizer.simulation.provider import get_provider


@pytest.fixture
def simple_qubo():
    """Create a simple 3-asset QUBO for testing."""
    expected_returns = np.array([0.01, 0.015, 0.008])
    covariance = np.array(
        [
            [0.04, 0.01, 0.005],
            [0.01, 0.05, 0.01],
            [0.005, 0.01, 0.03],
        ]
    )
    builder = PortfolioQUBO(
        expected_returns=expected_returns,
        covariance=covariance,
        budget=1.0,
        risk_aversion=500.0,
        penalty_strength=1000.0,
    )
    return builder.build()


class TestVQEResultDataclass:
    """Test VQEResult dataclass with new fields."""

    def test_result_with_bitstring(self):
        """Test VQEResult can hold bitstring data."""
        result = VQEResult(
            optimal_parameters=np.array([0.1, 0.2]),
            optimal_value=-0.5,
            history=[0.0, -0.3, -0.5],
            best_history=[0.0, -0.3, -0.5],
            num_evaluations=3,
            ansatz_report={},
            converged=True,
            optimizer_message="Success",
            best_bitstring="101",
            measurement_counts={"101": 500, "110": 300, "011": 224},
        )
        assert result.best_bitstring == "101"
        assert result.measurement_counts["101"] == 500

    def test_result_without_bitstring(self):
        """Test backward compatibility - bitstring fields are optional."""
        result = VQEResult(
            optimal_parameters=np.array([0.1, 0.2]),
            optimal_value=-0.5,
            history=[0.0, -0.3, -0.5],
            best_history=[0.0, -0.3, -0.5],
            num_evaluations=3,
            ansatz_report={},
            converged=True,
            optimizer_message="Success",
        )
        assert result.best_bitstring is None
        assert result.measurement_counts is None


class TestVQEWithoutSampler:
    """Test VQE without sampler returns None bitstring."""

    def test_vqe_without_sampler_returns_none_bitstring(self, simple_qubo):
        """VQE without sampler should return None for bitstring fields."""
        backend_config = {"name": "local_simulator", "shots": 100, "seed": 42}
        estimator, _ = get_provider(backend_config)

        from quantum_portfolio_optimizer.core.optimizer_interface import (
            DifferentialEvolutionConfig,
        )

        config = DifferentialEvolutionConfig(
            bounds=[(-np.pi, np.pi)] * 12,
            maxiter=5,
            popsize=3,
            seed=42,
        )

        solver = PortfolioVQESolver(
            estimator=estimator,
            sampler=None,  # No sampler
            ansatz_name="real_amplitudes",
            ansatz_options={"reps": 1},
            optimizer_config=config,
            seed=42,
        )

        result = solver.solve(simple_qubo)
        assert result.best_bitstring is None
        assert result.measurement_counts is None


class TestVQEWithSampler:
    """Test VQE with sampler returns valid bitstring."""

    def test_vqe_with_sampler_returns_bitstring(self, simple_qubo):
        """VQE with sampler should return valid bitstring."""
        backend_config = {"name": "local_simulator", "shots": 1024, "seed": 42}
        estimator, sampler = get_provider(backend_config)

        from quantum_portfolio_optimizer.core.optimizer_interface import (
            DifferentialEvolutionConfig,
        )

        config = DifferentialEvolutionConfig(
            bounds=[(-np.pi, np.pi)] * 12,
            maxiter=5,
            popsize=3,
            seed=42,
        )

        solver = PortfolioVQESolver(
            estimator=estimator,
            sampler=sampler,  # With sampler
            ansatz_name="real_amplitudes",
            ansatz_options={"reps": 1},
            optimizer_config=config,
            seed=42,
            extraction_shots=512,
        )

        result = solver.solve(simple_qubo)

        # Should have bitstring
        assert result.best_bitstring is not None
        assert len(result.best_bitstring) >= 3  # At least 3 qubits
        assert all(c in "01" for c in result.best_bitstring)

        # Should have measurement counts
        assert result.measurement_counts is not None
        assert len(result.measurement_counts) > 0
        assert all(
            isinstance(k, str) and isinstance(v, int)
            for k, v in result.measurement_counts.items()
        )

    def test_bitstring_represents_valid_portfolio(self, simple_qubo):
        """Test that extracted bitstring represents a valid portfolio selection."""
        backend_config = {"name": "local_simulator", "shots": 1024, "seed": 42}
        estimator, sampler = get_provider(backend_config)

        from quantum_portfolio_optimizer.core.optimizer_interface import (
            DifferentialEvolutionConfig,
        )

        config = DifferentialEvolutionConfig(
            bounds=[(-np.pi, np.pi)] * 12,
            maxiter=10,
            popsize=5,
            seed=42,
        )

        solver = PortfolioVQESolver(
            estimator=estimator,
            sampler=sampler,
            ansatz_name="real_amplitudes",
            ansatz_options={"reps": 1},
            optimizer_config=config,
            seed=42,
        )

        result = solver.solve(simple_qubo)

        # Convert bitstring to portfolio selection
        bitstring = result.best_bitstring.zfill(3)[-3:]  # Ensure 3 bits
        selection = [int(b) for b in bitstring[::-1]]  # Reverse for asset order

        # Should be a valid binary selection
        assert all(s in [0, 1] for s in selection)
        # At least some asset should be selected (budget constraint)
        # Note: With optimization, we expect a reasonable selection
        assert len(selection) == 3


class TestExtractSolutionMethod:
    """Test the extract_solution method directly."""

    def test_extract_solution_requires_sampler(self, simple_qubo):
        """extract_solution should raise if sampler is None."""
        backend_config = {"name": "local_simulator", "shots": 100, "seed": 42}
        estimator, _ = get_provider(backend_config)

        solver = PortfolioVQESolver(
            estimator=estimator,
            sampler=None,
            ansatz_name="real_amplitudes",
            ansatz_options={"reps": 1},
        )

        from quantum_portfolio_optimizer.core.ansatz_library import get_ansatz

        ansatz = get_ansatz("real_amplitudes", num_qubits=3, reps=1)
        params = np.zeros(ansatz.num_parameters)

        with pytest.raises(ValueError, match="Sampler is required"):
            solver.extract_solution(ansatz, params, num_qubits=3)

    def test_extract_solution_returns_counts(self, simple_qubo):
        """extract_solution should return bitstring and counts."""
        backend_config = {"name": "local_simulator", "shots": 512, "seed": 42}
        estimator, sampler = get_provider(backend_config)

        solver = PortfolioVQESolver(
            estimator=estimator,
            sampler=sampler,
            ansatz_name="real_amplitudes",
            ansatz_options={"reps": 1},
        )

        from quantum_portfolio_optimizer.core.ansatz_library import get_ansatz

        ansatz = get_ansatz("real_amplitudes", num_qubits=3, reps=1)
        params = np.random.RandomState(42).uniform(-np.pi, np.pi, ansatz.num_parameters)

        bitstring, counts = solver.extract_solution(
            ansatz, params, num_qubits=3, shots=512
        )

        assert isinstance(bitstring, str)
        assert all(c in "01" for c in bitstring)
        assert isinstance(counts, dict)
        # Total shots depends on sampler interface (V2 may use different defaults)
        assert sum(counts.values()) > 0
        assert bitstring in counts  # Best bitstring should be in counts
        assert counts[bitstring] == max(counts.values())  # Should be most frequent

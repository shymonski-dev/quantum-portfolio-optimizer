"""Tests for warm start utilities."""

import numpy as np
import pytest
from qiskit.circuit.library import RealAmplitudes

from quantum_portfolio_optimizer.core.warm_start import (
    WarmStartConfig,
    WarmStartResult,
    allocations_to_rotation_angles,
    warm_start_vqe,
    warm_start_qaoa,
    get_binary_initial_state,
    estimate_initial_energy,
)
from quantum_portfolio_optimizer.exceptions import WarmStartError


class TestAllocationsToRotationAngles:
    """Test allocation to rotation angle conversion."""

    def test_equal_allocations_produce_equal_angles(self):
        """Equal allocations should produce equal angles."""
        allocations = np.array([0.25, 0.25, 0.25, 0.25])
        angles = allocations_to_rotation_angles(allocations, add_noise=False)

        # All angles should be equal for equal allocations
        assert np.allclose(angles[0], angles[1])
        assert np.allclose(angles[1], angles[2])
        assert np.allclose(angles[2], angles[3])

    def test_zero_allocation_produces_zero_angle(self):
        """Zero allocation should produce angle near zero."""
        allocations = np.array([0.0, 1.0])
        angles = allocations_to_rotation_angles(allocations, add_noise=False)

        assert angles[0] == pytest.approx(0.0, abs=1e-10)

    def test_full_allocation_produces_pi_angle(self):
        """Full allocation should produce angle of pi."""
        allocations = np.array([1.0])
        angles = allocations_to_rotation_angles(allocations, add_noise=False)

        # arcsin(sqrt(1)) = pi/2, so 2*arcsin(sqrt(1)) = pi
        assert angles[0] == pytest.approx(np.pi, abs=1e-10)

    def test_angles_in_valid_range(self):
        """Angles should be in [0, pi] range."""
        allocations = np.array([0.1, 0.3, 0.4, 0.2])
        angles = allocations_to_rotation_angles(allocations, add_noise=False)

        assert all(0 <= a <= np.pi for a in angles)

    def test_noise_changes_angles(self):
        """Adding noise should change the angles."""
        allocations = np.array([0.5, 0.5])

        angles_no_noise = allocations_to_rotation_angles(allocations, add_noise=False)
        angles_with_noise = allocations_to_rotation_angles(
            allocations, add_noise=True, seed=42
        )

        assert not np.allclose(angles_no_noise, angles_with_noise)

    def test_seed_reproducibility(self):
        """Same seed should produce same angles."""
        allocations = np.array([0.3, 0.7])

        angles1 = allocations_to_rotation_angles(allocations, add_noise=True, seed=123)
        angles2 = allocations_to_rotation_angles(allocations, add_noise=True, seed=123)

        np.testing.assert_array_equal(angles1, angles2)

    def test_handles_unnormalized_allocations(self):
        """Should normalize allocations that don't sum to 1."""
        allocations = np.array([0.2, 0.2, 0.2])  # Sum = 0.6
        angles = allocations_to_rotation_angles(allocations, add_noise=False)

        # After normalization, each is 1/3
        expected = 2 * np.arcsin(np.sqrt(1 / 3))
        assert all(np.isclose(a, expected) for a in angles)


class TestWarmStartVQE:
    """Test VQE warm start functionality."""

    def test_produces_correct_number_of_parameters(self):
        """Should produce parameters matching ansatz."""
        expected_returns = [0.1, 0.15, 0.12, 0.08]
        covariance = np.eye(4) * 0.04

        ansatz = RealAmplitudes(num_qubits=4, reps=2)

        result = warm_start_vqe(
            expected_returns=expected_returns,
            covariance=covariance,
            ansatz=ansatz,
            config=WarmStartConfig(seed=42),
        )

        assert len(result.initial_parameters) == ansatz.num_parameters

    def test_classical_allocations_stored(self):
        """Should store the classical allocations used."""
        expected_returns = [0.1, 0.15, 0.12]
        covariance = np.eye(3) * 0.04
        ansatz = RealAmplitudes(num_qubits=3, reps=1)

        result = warm_start_vqe(
            expected_returns=expected_returns,
            covariance=covariance,
            ansatz=ansatz,
            config=WarmStartConfig(seed=42),
        )

        assert len(result.classical_allocations) == 3
        assert result.classical_allocations.sum() == pytest.approx(1.0, abs=0.01)

    def test_method_is_amplitude_encoding(self):
        """Should use amplitude encoding method."""
        expected_returns = [0.1, 0.2]
        covariance = np.eye(2) * 0.04
        ansatz = RealAmplitudes(num_qubits=2, reps=1)

        result = warm_start_vqe(
            expected_returns=expected_returns,
            covariance=covariance,
            ansatz=ansatz,
        )

        assert result.method == "amplitude_encoding"

    def test_raises_on_mismatched_dimensions(self):
        """Should raise error when dimensions don't match."""
        expected_returns = [0.1, 0.2, 0.3]  # 3 assets
        covariance = np.eye(3) * 0.04
        ansatz = RealAmplitudes(num_qubits=4, reps=1)  # 4 qubits - mismatch!

        with pytest.raises(WarmStartError):
            warm_start_vqe(
                expected_returns=expected_returns,
                covariance=covariance,
                ansatz=ansatz,
            )

    def test_estimated_improvement_is_positive(self):
        """Should estimate positive improvement."""
        expected_returns = [0.1, 0.15]
        covariance = np.eye(2) * 0.04
        ansatz = RealAmplitudes(num_qubits=2, reps=1)

        result = warm_start_vqe(
            expected_returns=expected_returns,
            covariance=covariance,
            ansatz=ansatz,
        )

        assert result.estimated_improvement > 1.0


class TestWarmStartQAOA:
    """Test QAOA warm start functionality."""

    def test_produces_correct_number_of_parameters(self):
        """Should produce 2p parameters (gamma, beta pairs)."""
        qubo_linear = np.array([0.1, 0.2, 0.15])
        qubo_quadratic = np.eye(3) * 0.05
        layers = 3

        result = warm_start_qaoa(
            qubo_linear=qubo_linear,
            qubo_quadratic=qubo_quadratic,
            layers=layers,
            config=WarmStartConfig(seed=42),
        )

        assert len(result.initial_parameters) == 2 * layers

    def test_gamma_in_valid_range(self):
        """Gamma parameters should be in [0, 2pi]."""
        qubo_linear = np.array([0.5, 0.3])
        qubo_quadratic = np.eye(2) * 0.1
        layers = 2

        result = warm_start_qaoa(
            qubo_linear=qubo_linear,
            qubo_quadratic=qubo_quadratic,
            layers=layers,
            config=WarmStartConfig(seed=42),
        )

        gammas = result.initial_parameters[::2]
        assert all(0 <= g <= 2 * np.pi for g in gammas)

    def test_beta_in_valid_range(self):
        """Beta parameters should be in [0, pi]."""
        qubo_linear = np.array([0.5, 0.3])
        qubo_quadratic = np.eye(2) * 0.1
        layers = 2

        result = warm_start_qaoa(
            qubo_linear=qubo_linear,
            qubo_quadratic=qubo_quadratic,
            layers=layers,
            config=WarmStartConfig(seed=42),
        )

        betas = result.initial_parameters[1::2]
        assert all(0 <= b <= np.pi for b in betas)

    def test_method_is_mean_field(self):
        """Should use mean field heuristic method."""
        qubo_linear = np.array([0.1])
        qubo_quadratic = np.array([[0.01]])
        layers = 1

        result = warm_start_qaoa(
            qubo_linear=qubo_linear,
            qubo_quadratic=qubo_quadratic,
            layers=layers,
        )

        assert result.method == "mean_field_heuristic"

    def test_includes_classical_allocations_when_provided(self):
        """Should compute classical allocations when returns/cov provided."""
        qubo_linear = np.array([0.1, 0.2])
        qubo_quadratic = np.eye(2) * 0.05
        expected_returns = [0.1, 0.15]
        covariance = np.eye(2) * 0.04

        result = warm_start_qaoa(
            qubo_linear=qubo_linear,
            qubo_quadratic=qubo_quadratic,
            layers=2,
            expected_returns=expected_returns,
            covariance=covariance,
            config=WarmStartConfig(seed=42),
        )

        # Should have computed classical allocations
        assert len(result.classical_allocations) == 2

    def test_handles_empty_quadratic(self):
        """Should handle empty quadratic terms."""
        qubo_linear = np.array([0.1, 0.2, 0.3])
        qubo_quadratic = np.array([]).reshape(0, 0)
        layers = 1

        result = warm_start_qaoa(
            qubo_linear=qubo_linear,
            qubo_quadratic=qubo_quadratic,
            layers=layers,
        )

        assert len(result.initial_parameters) == 2

    def test_xy_mixer_initialization_path(self):
        """Exchange mixer warm start should return valid parameters."""
        qubo_linear = np.array([0.1, 0.2, 0.3, 0.15])
        qubo_quadratic = np.eye(4) * 0.05
        layers = 2

        result = warm_start_qaoa(
            qubo_linear=qubo_linear,
            qubo_quadratic=qubo_quadratic,
            layers=layers,
            mixer_type="xy",
            config=WarmStartConfig(seed=42),
        )

        assert len(result.initial_parameters) == 2 * layers
        gammas = result.initial_parameters[::2]
        betas = result.initial_parameters[1::2]
        assert all(0 <= g <= 2 * np.pi for g in gammas)
        assert all(0 <= b <= np.pi for b in betas)


class TestGetBinaryInitialState:
    """Test binary state conversion."""

    def test_above_threshold_is_one(self):
        """Allocations above threshold should be 1."""
        allocations = np.array([0.5, 0.3, 0.1, 0.1])
        binary = get_binary_initial_state(allocations, threshold=0.15)

        assert binary == "1100"

    def test_below_threshold_is_zero(self):
        """Allocations below threshold should be 0."""
        allocations = np.array([0.01, 0.02, 0.97])
        binary = get_binary_initial_state(allocations, threshold=0.05)

        assert binary == "001"

    def test_equal_to_threshold_is_one(self):
        """Allocations equal to threshold should be 1."""
        allocations = np.array([0.1, 0.9])
        binary = get_binary_initial_state(allocations, threshold=0.1)

        assert binary == "11"

    def test_all_zeros_below_threshold(self):
        """All small allocations produce all zeros."""
        allocations = np.array([0.001, 0.002, 0.003])
        binary = get_binary_initial_state(allocations, threshold=0.01)

        assert binary == "000"


class TestEstimateInitialEnergy:
    """Test energy estimation."""

    def test_equal_weights_energy(self):
        """Energy for equal weights should be calculable."""
        allocations = np.array([0.5, 0.5])
        expected_returns = np.array([0.1, 0.2])
        covariance = np.array([[0.04, 0.01], [0.01, 0.04]])
        risk_aversion = 0.5

        energy = estimate_initial_energy(
            allocations, expected_returns, covariance, risk_aversion
        )

        # Manual calculation:
        # return = 0.5*0.1 + 0.5*0.2 = 0.15
        # variance = 0.5*0.5*0.04 + 2*0.5*0.5*0.01 + 0.5*0.5*0.04 = 0.025
        # energy = 0.5 * 0.025 - 0.15 = -0.1375
        assert energy == pytest.approx(-0.1375, abs=1e-6)

    def test_higher_risk_aversion_increases_energy(self):
        """Higher risk aversion should increase energy for same allocations."""
        allocations = np.array([0.5, 0.5])
        expected_returns = np.array([0.1, 0.2])
        covariance = np.array([[0.04, 0.01], [0.01, 0.04]])

        energy_low = estimate_initial_energy(
            allocations, expected_returns, covariance, 0.1
        )
        energy_high = estimate_initial_energy(
            allocations, expected_returns, covariance, 10.0
        )

        assert energy_high > energy_low


class TestWarmStartConfig:
    """Test configuration dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = WarmStartConfig()

        assert config.use_classical_solution is True
        assert config.allocation_threshold == 0.01
        assert config.noise_scale == 0.1
        assert config.seed is None

    def test_custom_values(self):
        """Should accept custom values."""
        config = WarmStartConfig(
            use_classical_solution=False,
            allocation_threshold=0.05,
            noise_scale=0.2,
            seed=123,
        )

        assert config.use_classical_solution is False
        assert config.allocation_threshold == 0.05
        assert config.noise_scale == 0.2
        assert config.seed == 123


class TestWarmStartResult:
    """Test result dataclass."""

    def test_stores_all_fields(self):
        """Should store all fields correctly."""
        result = WarmStartResult(
            initial_parameters=np.array([0.1, 0.2, 0.3]),
            classical_allocations=np.array([0.5, 0.5]),
            estimated_improvement=2.0,
            method="test_method",
        )

        assert len(result.initial_parameters) == 3
        assert len(result.classical_allocations) == 2
        assert result.estimated_improvement == 2.0
        assert result.method == "test_method"

import numpy as np
import pytest

from quantum_portfolio_optimizer.core.qubo_formulation import PortfolioQUBO, QUBOProblem
from quantum_portfolio_optimizer.exceptions import QUBOError


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


class TestDecodeBitstring:
    """Test multi-qubit resolution bitstring decoding."""

    def test_decode_single_qubit_resolution(self):
        """Test decoding with resolution_qubits=1 (binary allocation)."""
        builder = PortfolioQUBO(
            expected_returns=np.array([0.1, 0.2, 0.15]),
            covariance=np.eye(3) * 0.04,
            budget=1.0,
            time_steps=1,
            resolution_qubits=1,
            max_investment=1.0,
        )
        qubo = builder.build()

        # Bitstring "101" -> assets 0 and 2 selected (Qiskit little-endian)
        result = qubo.decode_bitstring("101")

        # With little-endian: rightmost bit is qubit 0
        # "101" -> qubit 0=1, qubit 1=0, qubit 2=1
        # So assets 0 and 2 are selected
        assert result["allocations"][0, 0] > 0  # Asset 0 selected
        assert result["allocations"][0, 1] == 0  # Asset 1 not selected
        assert result["allocations"][0, 2] > 0  # Asset 2 selected
        assert result["num_assets"] == 3
        assert result["time_steps"] == 1

    def test_decode_two_qubit_resolution(self):
        """Test decoding with resolution_qubits=2 (4 allocation levels)."""
        builder = PortfolioQUBO(
            expected_returns=np.array([0.1, 0.2]),
            covariance=np.eye(2) * 0.04,
            budget=1.0,
            time_steps=1,
            resolution_qubits=2,  # 4 levels: 0, 1/3, 2/3, 1
            max_investment=1.0,
        )
        qubo = builder.build()

        # With 2 assets and 2 resolution qubits: 4 qubits total
        # Variable order: (asset0, t0, bit0), (asset0, t0, bit1), (asset1, t0, bit0), (asset1, t0, bit1)
        assert qubo.num_variables == 4

        # Test all zeros: no allocation
        result_0 = qubo.decode_bitstring("0000")
        assert result_0["total_allocation"] == pytest.approx(0.0)

        # Test all ones: max allocation for both assets
        result_all = qubo.decode_bitstring("1111")
        assert result_all["total_allocation"] > 0
        assert result_all["binary_values"][(0, 0)] == 3  # 11 in binary = 3
        assert result_all["binary_values"][(1, 0)] == 3

    def test_decode_with_multiple_time_steps(self):
        """Test decoding with multiple time steps."""
        builder = PortfolioQUBO(
            expected_returns=np.array([0.1, 0.15]),
            covariance=np.eye(2) * 0.04,
            budget=1.0,
            time_steps=2,
            resolution_qubits=1,
            max_investment=1.0,
        )
        qubo = builder.build()

        # 2 assets * 2 time steps * 1 qubit = 4 qubits
        assert qubo.num_variables == 4

        # Decode bitstring
        result = qubo.decode_bitstring("1010")
        assert result["time_steps"] == 2
        assert len(result["allocation_per_time"]) == 2

    def test_decode_wrong_length_raises_error(self):
        """Bitstring with wrong length should raise QUBOError."""
        builder = PortfolioQUBO(
            expected_returns=np.array([0.1, 0.2]),
            covariance=np.eye(2) * 0.04,
            budget=1.0,
            time_steps=1,
            resolution_qubits=1,
        )
        qubo = builder.build()

        with pytest.raises(QUBOError, match="length"):
            qubo.decode_bitstring("1")  # Too short

        with pytest.raises(QUBOError, match="length"):
            qubo.decode_bitstring("1111")  # Too long (should be 2)

    def test_decode_returns_allocation_per_asset(self):
        """Decode should compute total allocation per asset."""
        builder = PortfolioQUBO(
            expected_returns=np.array([0.1, 0.2, 0.15]),
            covariance=np.eye(3) * 0.04,
            budget=1.0,
            time_steps=1,
            resolution_qubits=1,
            max_investment=1.0,
        )
        qubo = builder.build()

        result = qubo.decode_bitstring("111")
        assert len(result["allocation_per_asset"]) == 3
        assert all(a > 0 for a in result["allocation_per_asset"])

    def test_decode_binary_values_correct(self):
        """Test that binary values are correctly computed."""
        builder = PortfolioQUBO(
            expected_returns=np.array([0.1]),
            covariance=np.array([[0.04]]),
            budget=1.0,
            time_steps=1,
            resolution_qubits=3,  # 8 levels
            max_investment=1.0,
        )
        qubo = builder.build()

        # 1 asset * 1 time step * 3 qubits = 3 qubits
        assert qubo.num_variables == 3

        # Test binary value 5 = "101" in little-endian reversed to "101"
        # qubit 0=1 (2^0=1), qubit 1=0 (2^1=0), qubit 2=1 (2^2=4) -> total = 5
        result = qubo.decode_bitstring("101")
        assert result["binary_values"][(0, 0)] == 5

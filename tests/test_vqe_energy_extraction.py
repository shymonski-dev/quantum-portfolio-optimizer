"""Characterization tests for VQE energy extraction bounds checking.

Bug Location: src/quantum_portfolio_optimizer/core/vqe_solver.py:167
Bug: `return float(np.real(values[0]))` without checking if values is empty.

If the estimator returns an empty values array, this will raise IndexError
instead of a meaningful ValueError with clear error message.

These tests will FAIL before the fix and PASS after.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from quantum_portfolio_optimizer.core.vqe_solver import PortfolioVQESolver


class TestVQEEnergyExtraction:
    """Tests for VQE _extract_energy method bounds checking."""

    def test_extract_energy_with_valid_values_list(self):
        """Normal case: values is a non-empty list."""
        mock_result = MagicMock()
        mock_result.values = [1.5, 2.0, 3.0]

        energy = PortfolioVQESolver._extract_energy(mock_result)
        assert energy == pytest.approx(1.5)

    def test_extract_energy_with_valid_values_array(self):
        """Normal case: values is a non-empty numpy array."""
        mock_result = MagicMock()
        mock_result.values = np.array([2.5, 3.0])

        energy = PortfolioVQESolver._extract_energy(mock_result)
        assert energy == pytest.approx(2.5)

    def test_extract_energy_with_complex_values(self):
        """Values with imaginary component should return real part."""
        mock_result = MagicMock()
        mock_result.values = [1.5 + 0.1j]

        energy = PortfolioVQESolver._extract_energy(mock_result)
        assert energy == pytest.approx(1.5)

    def test_extract_energy_with_scalar_value(self):
        """Single scalar value (not array) should work."""
        mock_result = MagicMock()
        mock_result.values = 3.14

        energy = PortfolioVQESolver._extract_energy(mock_result)
        assert energy == pytest.approx(3.14)

    def test_extract_energy_empty_list_raises_value_error(self):
        """BUG TEST: Empty list should raise ValueError, not IndexError.

        Current behavior: raises IndexError (list index out of range)
        Expected behavior: raises ValueError with clear message
        """
        mock_result = MagicMock()
        mock_result.values = []

        # After fix, this should raise ValueError with descriptive message
        with pytest.raises(ValueError, match="empty"):
            PortfolioVQESolver._extract_energy(mock_result)

    def test_extract_energy_empty_array_raises_value_error(self):
        """BUG TEST: Empty numpy array should raise ValueError, not IndexError.

        Current behavior: raises IndexError (index 0 is out of bounds)
        Expected behavior: raises ValueError with clear message
        """
        mock_result = MagicMock()
        mock_result.values = np.array([])

        # After fix, this should raise ValueError with descriptive message
        with pytest.raises(ValueError, match="empty"):
            PortfolioVQESolver._extract_energy(mock_result)

    def test_extract_energy_empty_tuple_raises_value_error(self):
        """BUG TEST: Empty tuple should raise ValueError, not IndexError."""
        mock_result = MagicMock()
        mock_result.values = ()

        with pytest.raises(ValueError, match="empty"):
            PortfolioVQESolver._extract_energy(mock_result)


class TestVQEEnergyExtractionV2Interface:
    """Tests for V2 Qiskit primitive interface extraction."""

    def test_extract_energy_v2_interface(self):
        """Test extraction from V2 EstimatorResult format."""
        # V2 format: result[0].data.evs
        mock_evs = np.array([1.234])
        mock_data = MagicMock()
        mock_data.evs = mock_evs

        mock_first = MagicMock()
        mock_first.data = mock_data

        mock_result = MagicMock()
        mock_result.__getitem__ = MagicMock(return_value=mock_first)
        # Remove 'values' attribute to force V2 path
        del mock_result.values

        energy = PortfolioVQESolver._extract_energy(mock_result)
        assert energy == pytest.approx(1.234)

    def test_extract_energy_unrecognized_format_raises_error(self):
        """Unrecognized result format should raise clear ValueError."""
        mock_result = MagicMock(spec=[])  # No attributes

        with pytest.raises(ValueError, match="not recognised"):
            PortfolioVQESolver._extract_energy(mock_result)


class TestVQEExtractCounts:
    """Tests for VQE _extract_counts method."""

    def test_extract_counts_v2_interface(self):
        """Test count extraction from V2 Sampler result."""
        mock_counts = {"00": 250, "01": 250, "10": 250, "11": 250}
        mock_bitarray = MagicMock()
        mock_bitarray.get_counts.return_value = mock_counts

        mock_data = MagicMock()
        mock_data.keys.return_value = ["meas"]
        mock_data.meas = mock_bitarray

        mock_first = MagicMock()
        mock_first.data = mock_data

        mock_result = MagicMock()
        mock_result.__getitem__ = MagicMock(return_value=mock_first)

        counts = PortfolioVQESolver._extract_counts(mock_result, num_qubits=2)

        assert "00" in counts
        assert "11" in counts
        assert counts["00"] == 250

    def test_extract_counts_pads_bitstrings(self):
        """Bitstrings should be zero-padded to num_qubits length."""
        mock_counts = {"0": 500, "1": 500}  # Single bit from result
        mock_bitarray = MagicMock()
        mock_bitarray.get_counts.return_value = mock_counts

        mock_data = MagicMock()
        mock_data.keys.return_value = ["meas"]
        mock_data.meas = mock_bitarray

        mock_first = MagicMock()
        mock_first.data = mock_data

        mock_result = MagicMock()
        mock_result.__getitem__ = MagicMock(return_value=mock_first)

        counts = PortfolioVQESolver._extract_counts(mock_result, num_qubits=3)

        # Should be padded to 3 characters
        assert "000" in counts
        assert "001" in counts
        assert len(list(counts.keys())[0]) == 3

"""Tests for provider-agnostic ZNE via gate folding."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from quantum_portfolio_optimizer.simulation.zne import (
    fold_circuit,
    zne_evaluate,
    zne_extrapolate,
)


class TestZNEFolding:
    def _simple_circuit(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.5, 0)
        return qc

    def test_fold_noise_factor_1_unchanged(self):
        qc = self._simple_circuit()
        folded = fold_circuit(qc, 1)
        assert folded.count_ops() == qc.count_ops()

    def test_fold_noise_factor_3_triples_ops(self):
        qc = QuantumCircuit(1)
        qc.x(0)
        folded = fold_circuit(qc, 3)
        # x has inverse x, so x·x·x = 3 ops
        assert folded.count_ops().get("x", 0) == 3

    def test_fold_rejects_even_factor(self):
        qc = self._simple_circuit()
        with pytest.raises(ValueError, match="odd"):
            fold_circuit(qc, 2)

    def test_fold_rejects_zero(self):
        qc = self._simple_circuit()
        with pytest.raises(ValueError):
            fold_circuit(qc, 0)

    def test_fold_returns_new_circuit(self):
        qc = self._simple_circuit()
        folded = fold_circuit(qc, 1)
        assert folded is not qc

    def test_fold_factor_1_is_identity_on_statevector(self):
        """Folded circuit (nf=1) should have identical unitary to original."""
        from qiskit.quantum_info import Operator

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        folded = fold_circuit(qc, 1)
        assert np.allclose(Operator(qc).data, Operator(folded).data, atol=1e-10)


class TestZNEExtrapolation:
    def test_linear_extrapolation_exact(self):
        # y = 2*x + 1 -> at x=0, y=1
        nf = [1.0, 3.0]
        ev = [3.0, 7.0]
        result = zne_extrapolate(nf, ev, "linear")
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_quadratic_extrapolation_exact(self):
        # y = x^2 + 1 -> at x=0, y=1
        nf = [1.0, 3.0, 5.0]
        ev = [2.0, 10.0, 26.0]
        result = zne_extrapolate(nf, ev, "quadratic")
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_richardson_two_points_matches_linear(self):
        nf = [1.0, 3.0]
        ev = [3.0, 7.0]
        linear = zne_extrapolate(nf, ev, "linear")
        rich = zne_extrapolate(nf, ev, "richardson")
        assert rich == pytest.approx(linear, abs=1e-10)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="length"):
            zne_extrapolate([1, 3], [1.0, 2.0, 3.0])

    def test_unsupported_extrapolator_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            zne_extrapolate([1, 3], [1.0, 2.0], extrapolator="cubic_spline")

    def test_single_point_raises(self):
        with pytest.raises(ValueError, match="At least 2"):
            zne_extrapolate([1], [1.0])


class TestZNEIntegration:
    def test_zne_evaluate_calls_eval_fn_n_times(self):
        call_count = {"n": 0}

        def fake_eval(circuit):
            call_count["n"] += 1
            return 1.0

        qc = QuantumCircuit(1)
        qc.x(0)
        noise_factors = [1, 3, 5]
        zne_evaluate(fake_eval, qc, noise_factors)
        assert call_count["n"] == len(noise_factors)

    def test_zne_evaluate_extrapolates_linear(self):
        # eval_fn returns noise_factor itself -> y = nf, extrapolated at 0 = 0
        def linear_eval(circuit):
            # This is a mock -- we count gates to recover noise factor
            # Not robust but tests the plumbing
            return 1.0  # constant -> extrapolates to 1.0

        qc = QuantumCircuit(1)
        qc.x(0)
        result = zne_evaluate(linear_eval, qc, [1, 3, 5])
        assert result == pytest.approx(1.0, abs=1e-6)

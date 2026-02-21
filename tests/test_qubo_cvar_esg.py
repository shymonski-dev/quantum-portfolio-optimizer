import numpy as np
import pytest

from quantum_portfolio_optimizer.core.qubo_formulation import PortfolioQUBO


class TestCVaRRiskMetric:
    def _make_qubo(self, **kwargs):
        """Helper to make a simple 2-asset QUBO."""
        expected_returns = np.array([0.01, 0.015])
        covariance = np.array([[0.04, 0.02], [0.02, 0.09]])
        builder = PortfolioQUBO(
            expected_returns=expected_returns,
            covariance=covariance,
            budget=1.0,
            **kwargs,
        )
        return builder.build()

    def test_variance_mode_is_default(self):
        qubo_default = self._make_qubo()
        qubo_variance = self._make_qubo(risk_metric="variance")
        np.testing.assert_array_almost_equal(
            qubo_default.quadratic, qubo_variance.quadratic
        )

    def test_cvar_mode_changes_coefficients(self):
        qubo_var = self._make_qubo(risk_metric="variance")
        qubo_cvar = self._make_qubo(risk_metric="cvar")
        # CVaR (semivariance) should differ from full variance
        assert not np.allclose(qubo_var.quadratic, qubo_cvar.quadratic)

    def test_invalid_risk_metric_raises(self):
        with pytest.raises(ValueError, match="risk_metric"):
            self._make_qubo(risk_metric="invalid")

    def test_downside_cov_is_psd(self):
        expected_returns = np.array([0.01, 0.015, 0.008])
        covariance = np.array(
            [[0.04, 0.02, 0.01], [0.02, 0.09, 0.03], [0.01, 0.03, 0.06]]
        )
        builder = PortfolioQUBO(
            expected_returns=expected_returns, covariance=covariance, budget=1.0
        )
        dc = builder._compute_downside_covariance(covariance)
        eigenvalues = np.linalg.eigvalsh(dc)
        assert np.all(eigenvalues >= -1e-8)


class TestESGConstraints:
    def _make_qubo(self, **kwargs):
        expected_returns = np.array([0.01, 0.015, 0.008])
        covariance = np.array(
            [[0.04, 0.02, 0.01], [0.02, 0.09, 0.03], [0.01, 0.03, 0.06]]
        )
        builder = PortfolioQUBO(
            expected_returns=expected_returns,
            covariance=covariance,
            budget=1.0,
            **kwargs,
        )
        return builder.build()

    def test_zero_esg_weight_no_change(self):
        qubo_base = self._make_qubo()
        qubo_esg = self._make_qubo(esg_scores=np.array([0.8, 0.5, 0.9]), esg_weight=0.0)
        np.testing.assert_array_almost_equal(qubo_base.linear, qubo_esg.linear)

    def test_none_scores_no_change(self):
        qubo_base = self._make_qubo()
        qubo_esg = self._make_qubo(esg_scores=None, esg_weight=0.0)
        np.testing.assert_array_almost_equal(qubo_base.linear, qubo_esg.linear)

    def test_esg_modifies_linear_terms(self):
        qubo_base = self._make_qubo()
        esg = np.array([0.8, 0.5, 0.9])
        qubo_esg = self._make_qubo(esg_scores=esg, esg_weight=1.0)
        # Linear terms should differ
        assert not np.allclose(qubo_base.linear, qubo_esg.linear)

    def test_esg_nonzero_weight_none_scores_raises(self):
        with pytest.raises(ValueError, match="esg_scores"):
            self._make_qubo(esg_scores=None, esg_weight=1.0)

    def test_esg_wrong_length_raises(self):
        with pytest.raises(ValueError):
            self._make_qubo(esg_scores=np.array([0.8, 0.5]), esg_weight=1.0)
            # 3 assets but only 2 scores

    def test_high_esg_lowers_linear_cost(self):
        """Asset with high ESG score should have a lower (more negative) linear coefficient."""
        esg = np.array([0.1, 0.5, 1.0])
        qubo_base = self._make_qubo()
        qubo_esg = self._make_qubo(esg_scores=esg, esg_weight=1.0)
        # The change in linear[qubit for asset 2] should be more negative than for asset 0
        diff = qubo_esg.linear - qubo_base.linear
        # For single resolution_qubits=1, qubits 0,1,2 correspond to assets 0,1,2
        assert diff[2] < diff[0]  # higher ESG -> bigger reduction

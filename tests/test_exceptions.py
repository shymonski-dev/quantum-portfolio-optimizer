"""Tests for custom exception hierarchy."""

import pytest

from quantum_portfolio_optimizer.exceptions import (
    QuantumPortfolioError,
    DataError,
    InvalidTickerError,
    InvalidDateRangeError,
    InsufficientDataError,
    MarketDataError,
    QUBOError,
    InvalidCovarianceError,
    InvalidBudgetError,
    InvalidParameterError,
    OptimizationError,
    ConvergenceError,
    CircuitError,
    SolverError,
    BackendError,
    IBMAuthenticationError,
    IBMBackendNotFoundError,
    IBMSessionError,
    IBMJobError,
    LocalSimulatorError,
    WarmStartError,
    ScoringError,
    InvalidWeightsError,
)


class TestExceptionHierarchy:
    """Test that exception hierarchy is correct."""

    def test_data_errors_inherit_from_base(self):
        """Data errors should inherit from QuantumPortfolioError."""
        assert issubclass(DataError, QuantumPortfolioError)
        assert issubclass(InvalidTickerError, DataError)
        assert issubclass(InvalidDateRangeError, DataError)
        assert issubclass(InsufficientDataError, DataError)
        assert issubclass(MarketDataError, DataError)

    def test_qubo_errors_inherit_from_base(self):
        """QUBO errors should inherit from QuantumPortfolioError."""
        assert issubclass(QUBOError, QuantumPortfolioError)
        assert issubclass(InvalidCovarianceError, QUBOError)
        assert issubclass(InvalidBudgetError, QUBOError)
        assert issubclass(InvalidParameterError, QUBOError)

    def test_optimization_errors_inherit_from_base(self):
        """Optimization errors should inherit from QuantumPortfolioError."""
        assert issubclass(OptimizationError, QuantumPortfolioError)
        assert issubclass(ConvergenceError, OptimizationError)
        assert issubclass(CircuitError, OptimizationError)
        assert issubclass(SolverError, OptimizationError)

    def test_backend_errors_inherit_from_base(self):
        """Backend errors should inherit from QuantumPortfolioError."""
        assert issubclass(BackendError, QuantumPortfolioError)
        assert issubclass(IBMAuthenticationError, BackendError)
        assert issubclass(IBMBackendNotFoundError, BackendError)
        assert issubclass(IBMSessionError, BackendError)
        assert issubclass(IBMJobError, BackendError)
        assert issubclass(LocalSimulatorError, BackendError)

    def test_warm_start_error_inherits_from_base(self):
        """WarmStartError should inherit from QuantumPortfolioError."""
        assert issubclass(WarmStartError, QuantumPortfolioError)

    def test_scoring_errors_inherit_from_base(self):
        """Scoring errors should inherit from QuantumPortfolioError."""
        assert issubclass(ScoringError, QuantumPortfolioError)
        assert issubclass(InvalidWeightsError, ScoringError)


class TestExceptionAttributes:
    """Test that exceptions carry proper attributes."""

    def test_invalid_ticker_error_attributes(self):
        """InvalidTickerError should have ticker and reason attributes."""
        err = InvalidTickerError("AAPL!", "contains invalid character")
        assert err.ticker == "AAPL!"
        assert "AAPL!" in str(err)
        assert "invalid character" in str(err)
        assert err.details["ticker"] == "AAPL!"
        assert err.details["reason"] == "contains invalid character"

    def test_invalid_date_range_error_attributes(self):
        """InvalidDateRangeError should have date attributes."""
        err = InvalidDateRangeError("2024-06-01", "2024-01-01", "start after end")
        assert err.start_date == "2024-06-01"
        assert err.end_date == "2024-01-01"
        assert "start after end" in str(err)
        assert err.details["start_date"] == "2024-06-01"
        assert err.details["end_date"] == "2024-01-01"

    def test_insufficient_data_error_attributes(self):
        """InsufficientDataError should have data point details."""
        err = InsufficientDataError(required=20, actual=5, tickers=["AAPL", "MSFT"])
        assert err.required == 20
        assert err.actual == 5
        assert err.tickers == ["AAPL", "MSFT"]
        assert "20" in str(err)
        assert "5" in str(err)

    def test_invalid_covariance_error_attributes(self):
        """InvalidCovarianceError should have shape attribute."""
        err = InvalidCovarianceError("not symmetric", shape=(3, 4))
        assert err.shape == (3, 4)
        assert "not symmetric" in str(err)
        assert err.details["shape"] == (3, 4)

    def test_invalid_budget_error_attributes(self):
        """InvalidBudgetError should have budget attribute."""
        err = InvalidBudgetError(-1.0, "must be positive")
        assert err.budget == -1.0
        assert "-1.0" in str(err)
        assert "positive" in str(err)

    def test_convergence_error_attributes(self):
        """ConvergenceError should have iteration details."""
        err = ConvergenceError(iterations=100, final_value=0.5, tolerance=1e-6)
        assert err.iterations == 100
        assert err.final_value == 0.5
        assert err.tolerance == 1e-6
        assert "100" in str(err)

    def test_ibm_backend_not_found_attributes(self):
        """IBMBackendNotFoundError should have backend details."""
        err = IBMBackendNotFoundError(
            backend_name="ibm_fake",
            available_backends=["ibm_brisbane", "ibm_kyoto"]
        )
        assert err.backend_name == "ibm_fake"
        assert err.available_backends == ["ibm_brisbane", "ibm_kyoto"]
        assert "ibm_fake" in str(err)

    def test_ibm_job_error_attributes(self):
        """IBMJobError should have job details."""
        err = IBMJobError("job failed", job_id="abc123", status="ERROR")
        assert err.job_id == "abc123"
        assert err.status == "ERROR"
        assert "job failed" in str(err)


class TestExceptionToDict:
    """Test exception serialization to dict for API responses."""

    def test_base_exception_to_dict(self):
        """QuantumPortfolioError.to_dict() should return proper structure."""
        err = QuantumPortfolioError("test error", details={"key": "value"})
        d = err.to_dict()

        assert d["error_type"] == "QuantumPortfolioError"
        assert d["message"] == "test error"
        assert d["details"] == {"key": "value"}

    def test_invalid_ticker_to_dict(self):
        """InvalidTickerError.to_dict() should include ticker details."""
        err = InvalidTickerError("BAD!", "invalid format")
        d = err.to_dict()

        assert d["error_type"] == "InvalidTickerError"
        assert "BAD!" in d["message"]
        assert d["details"]["ticker"] == "BAD!"
        assert d["details"]["reason"] == "invalid format"

    def test_convergence_error_to_dict(self):
        """ConvergenceError.to_dict() should include iteration details."""
        err = ConvergenceError(iterations=50, final_value=1.23)
        d = err.to_dict()

        assert d["error_type"] == "ConvergenceError"
        assert d["details"]["iterations"] == 50
        assert d["details"]["final_value"] == 1.23

    def test_ibm_backend_not_found_to_dict(self):
        """IBMBackendNotFoundError.to_dict() should include available backends."""
        err = IBMBackendNotFoundError(
            backend_name="fake_backend",
            available_backends=["real_backend"]
        )
        d = err.to_dict()

        assert d["error_type"] == "IBMBackendNotFoundError"
        assert d["details"]["backend_name"] == "fake_backend"
        assert d["details"]["available_backends"] == ["real_backend"]


class TestExceptionCatching:
    """Test that exceptions can be caught at appropriate levels."""

    def test_catch_all_quantum_errors(self):
        """Should be able to catch all custom errors with base class."""
        errors = [
            InvalidTickerError("T", "reason"),
            InvalidDateRangeError("2024-01-01", "2024-06-01", "reason"),
            InvalidCovarianceError("reason"),
            ConvergenceError(10, 0.5),
            IBMAuthenticationError("reason"),
        ]

        for err in errors:
            with pytest.raises(QuantumPortfolioError):
                raise err

    def test_catch_data_errors(self):
        """Should be able to catch all data errors with DataError."""
        errors = [
            InvalidTickerError("T", "reason"),
            InvalidDateRangeError("2024-01-01", "2024-06-01", "reason"),
            InsufficientDataError(10, 5, ["T"]),
            MarketDataError("reason"),
        ]

        for err in errors:
            with pytest.raises(DataError):
                raise err

    def test_catch_qubo_errors(self):
        """Should be able to catch all QUBO errors with QUBOError."""
        errors = [
            InvalidCovarianceError("reason"),
            InvalidBudgetError(0, "reason"),
            InvalidParameterError("param", 0, "reason"),
        ]

        for err in errors:
            with pytest.raises(QUBOError):
                raise err

    def test_catch_backend_errors(self):
        """Should be able to catch all backend errors with BackendError."""
        errors = [
            IBMAuthenticationError("reason"),
            IBMBackendNotFoundError("backend"),
            IBMSessionError("reason"),
            IBMJobError("reason"),
            LocalSimulatorError("reason"),
        ]

        for err in errors:
            with pytest.raises(BackendError):
                raise err


class TestExceptionMessages:
    """Test that exception messages are clear and informative."""

    def test_market_data_error_includes_provider(self):
        """MarketDataError should mention the data provider."""
        err = MarketDataError("connection failed", provider="yfinance")
        assert "yfinance" in str(err)
        assert err.provider == "yfinance"

    def test_circuit_error_includes_type(self):
        """CircuitError should mention circuit type if provided."""
        err = CircuitError("too deep", circuit_type="ansatz")
        assert "ansatz" in err.details.get("circuit_type", "")
        assert "too deep" in str(err)

    def test_solver_error_includes_solver_type(self):
        """SolverError should mention solver type."""
        err = SolverError("VQE", "failed to converge")
        assert "VQE" in str(err)
        assert err.solver_type == "VQE"

    def test_warm_start_error_includes_solver_type(self):
        """WarmStartError should mention solver type if provided."""
        err = WarmStartError("invalid initial point", solver_type="QAOA")
        assert "QAOA" in err.details.get("solver_type", "")
        assert "invalid initial point" in str(err)

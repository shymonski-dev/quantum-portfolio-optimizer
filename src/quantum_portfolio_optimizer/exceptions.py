"""Custom exception hierarchy for Quantum Portfolio Optimizer.

This module provides a comprehensive exception hierarchy for the quantum
portfolio optimizer, enabling type-safe error handling and user-friendly
error messages throughout the application.
"""

from typing import Any, Optional


class QuantumPortfolioError(Exception):
    """Base exception for all quantum portfolio optimizer errors.

    All custom exceptions in this package inherit from this class,
    allowing for catch-all handling when needed.

    Attributes:
        message: Human-readable error description.
        details: Optional dictionary with additional error context.
    """

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


# =============================================================================
# Data Errors
# =============================================================================


class DataError(QuantumPortfolioError):
    """Base class for data-related errors."""

    pass


class InvalidTickerError(DataError):
    """Raised when ticker symbol validation fails.

    Examples:
        - Ticker contains invalid characters
        - Ticker is too long
        - Duplicate tickers provided
    """

    def __init__(self, ticker: str, reason: str):
        self.ticker = ticker
        super().__init__(
            f"Invalid ticker '{ticker}': {reason}",
            details={"ticker": ticker, "reason": reason},
        )


class InvalidDateRangeError(DataError):
    """Raised when date range validation fails.

    Examples:
        - Invalid date format
        - Start date after end date
        - Date range too short
        - End date in the future
    """

    def __init__(self, start_date: str, end_date: str, reason: str):
        self.start_date = start_date
        self.end_date = end_date
        super().__init__(
            f"Invalid date range '{start_date}' to '{end_date}': {reason}",
            details={"start_date": start_date, "end_date": end_date, "reason": reason},
        )


class InsufficientDataError(DataError):
    """Raised when fetched data has insufficient data points.

    This typically occurs when the date range is too short or
    when trading data is sparse.
    """

    def __init__(self, required: int, actual: int, tickers: list[str]):
        self.required = required
        self.actual = actual
        self.tickers = tickers
        super().__init__(
            f"Insufficient data: {actual} data points for {len(tickers)} tickers "
            f"(minimum {required} required)",
            details={
                "required": required,
                "actual": actual,
                "tickers": tickers,
            },
        )


class MarketDataError(DataError):
    """Raised when market data fetch fails.

    This wraps errors from the underlying data provider (yfinance).
    """

    def __init__(self, message: str, provider: str = "yfinance"):
        self.provider = provider
        super().__init__(
            f"Market data error ({provider}): {message}",
            details={"provider": provider},
        )


# =============================================================================
# QUBO Formulation Errors
# =============================================================================


class QUBOError(QuantumPortfolioError):
    """Base class for QUBO formulation errors."""

    pass


class InvalidCovarianceError(QUBOError):
    """Raised when covariance matrix is invalid.

    Examples:
        - Matrix is not square
        - Matrix is not symmetric
        - Matrix is not positive semi-definite
        - Dimensions don't match expected returns
    """

    def __init__(self, reason: str, shape: Optional[tuple] = None):
        self.shape = shape
        details = {"reason": reason}
        if shape:
            details["shape"] = shape
        super().__init__(f"Invalid covariance matrix: {reason}", details=details)


class InvalidBudgetError(QUBOError):
    """Raised when budget constraint is invalid.

    Examples:
        - Budget is non-positive
        - Budget exceeds maximum allowed
    """

    def __init__(self, budget: float, reason: str):
        self.budget = budget
        super().__init__(
            f"Invalid budget {budget}: {reason}",
            details={"budget": budget, "reason": reason},
        )


class InvalidParameterError(QUBOError):
    """Raised when QUBO parameters are invalid.

    Generic error for parameters not covered by specific exceptions.
    """

    def __init__(self, parameter: str, value: Any, reason: str):
        self.parameter = parameter
        self.value = value
        super().__init__(
            f"Invalid parameter '{parameter}' = {value}: {reason}",
            details={"parameter": parameter, "value": value, "reason": reason},
        )


# =============================================================================
# Optimization Errors
# =============================================================================


class OptimizationError(QuantumPortfolioError):
    """Base class for optimization-related errors."""

    pass


class ConvergenceError(OptimizationError):
    """Raised when optimization fails to converge.

    This may indicate that more iterations are needed or that
    the problem is poorly conditioned.
    """

    def __init__(
        self,
        iterations: int,
        final_value: float,
        tolerance: Optional[float] = None,
    ):
        self.iterations = iterations
        self.final_value = final_value
        self.tolerance = tolerance
        msg = f"Optimization did not converge after {iterations} iterations"
        if tolerance is not None:
            msg += f" (tolerance: {tolerance})"
        super().__init__(
            msg,
            details={
                "iterations": iterations,
                "final_value": final_value,
                "tolerance": tolerance,
            },
        )


class CircuitError(OptimizationError):
    """Raised when quantum circuit construction or execution fails.

    Examples:
        - Invalid ansatz configuration
        - Circuit too deep for backend
        - Measurement extraction failure
    """

    def __init__(self, message: str, circuit_type: Optional[str] = None):
        self.circuit_type = circuit_type
        details = {}
        if circuit_type:
            details["circuit_type"] = circuit_type
        super().__init__(f"Circuit error: {message}", details=details)


class SolverError(OptimizationError):
    """Raised when the VQE or QAOA solver encounters an error."""

    def __init__(self, solver_type: str, message: str):
        self.solver_type = solver_type
        super().__init__(
            f"{solver_type} solver error: {message}",
            details={"solver_type": solver_type},
        )


# =============================================================================
# Backend Errors
# =============================================================================


class BackendError(QuantumPortfolioError):
    """Base class for quantum backend errors."""

    pass


class IBMAuthenticationError(BackendError):
    """Raised when IBM Quantum authentication fails.

    Common causes:
        - Invalid API token
        - Invalid Cloud CRN
        - Expired credentials
    """

    def __init__(self, reason: str):
        super().__init__(
            f"IBM Quantum authentication failed: {reason}",
            details={"reason": reason},
        )


class IBMBackendNotFoundError(BackendError):
    """Raised when requested IBM Quantum backend is not available.

    This may occur if:
        - Backend name is misspelled
        - Backend is not accessible with current credentials
        - Backend is under maintenance
    """

    def __init__(self, backend_name: str, available_backends: Optional[list[str]] = None):
        self.backend_name = backend_name
        self.available_backends = available_backends
        details = {"backend_name": backend_name}
        if available_backends:
            details["available_backends"] = available_backends
        super().__init__(
            f"IBM Quantum backend '{backend_name}' not found",
            details=details,
        )


class IBMSessionError(BackendError):
    """Raised when IBM Quantum Session creation or management fails."""

    def __init__(self, message: str, session_id: Optional[str] = None):
        self.session_id = session_id
        details = {}
        if session_id:
            details["session_id"] = session_id
        super().__init__(f"IBM Quantum session error: {message}", details=details)


class IBMJobError(BackendError):
    """Raised when an IBM Quantum job fails.

    Includes details about the job for debugging.
    """

    def __init__(
        self,
        message: str,
        job_id: Optional[str] = None,
        status: Optional[str] = None,
    ):
        self.job_id = job_id
        self.status = status
        details = {}
        if job_id:
            details["job_id"] = job_id
        if status:
            details["status"] = status
        super().__init__(f"IBM Quantum job error: {message}", details=details)


class LocalSimulatorError(BackendError):
    """Raised when local simulator encounters an error."""

    def __init__(self, message: str):
        super().__init__(f"Local simulator error: {message}")


# =============================================================================
# Warm Start Errors
# =============================================================================


class WarmStartError(QuantumPortfolioError):
    """Raised when warm start initialization fails."""

    def __init__(self, message: str, solver_type: Optional[str] = None):
        self.solver_type = solver_type
        details = {}
        if solver_type:
            details["solver_type"] = solver_type
        super().__init__(f"Warm start error: {message}", details=details)


# =============================================================================
# Quality Scoring Errors
# =============================================================================


class ScoringError(QuantumPortfolioError):
    """Raised when solution quality scoring fails."""

    def __init__(self, message: str):
        super().__init__(f"Scoring error: {message}")


class InvalidWeightsError(ScoringError):
    """Raised when scoring weights are invalid."""

    def __init__(self, weights: dict[str, float], reason: str):
        self.weights = weights
        super().__init__(
            f"Invalid scoring weights: {reason}",
        )
        self.details["weights"] = weights
        self.details["reason"] = reason

"""Runtime checks for benchmark validation helpers."""

from quantum_portfolio_optimizer.benchmarks.validation_tests import (
    ValidationReport,
    validate_small_instance,
)


def test_validate_small_instance_runs():
    report = validate_small_instance(seed=7)

    assert isinstance(report, ValidationReport)
    assert report.num_evaluations > 0
    assert isinstance(report.feasible, bool)

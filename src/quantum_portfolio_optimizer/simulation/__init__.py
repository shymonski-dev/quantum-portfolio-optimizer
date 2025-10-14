"""Simulation backends and diagnostics."""

from .local_backend import get_default_estimator, get_default_sampler

__all__ = ["get_default_estimator", "get_default_sampler"]

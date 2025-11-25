# quantum_portfolio_optimizer/src/quantum_portfolio_optimizer/simulation/provider.py
"""Backend provider for local simulators and real quantum hardware."""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional, Tuple

# Qiskit primitives - use V2 base classes for type hints
try:
    from qiskit.primitives import BaseEstimatorV2 as BaseEstimator
    from qiskit.primitives import BaseSamplerV2 as BaseSampler
except ImportError:
    BaseEstimator = None
    BaseSampler = None

# Reference implementations (built into Qiskit, no qiskit-aer needed)
try:
    from qiskit.primitives import StatevectorEstimator as ReferenceEstimator
    from qiskit.primitives import StatevectorSampler as ReferenceSampler
except ImportError:
    ReferenceEstimator = None
    ReferenceSampler = None

# Qiskit Aer for simulation
try:
    from qiskit_aer.primitives import Estimator as AerEstimator
    from qiskit_aer.primitives import Sampler as AerSampler
except ImportError:
    AerEstimator = None
    AerSampler = None

from .ibm_provider import get_ibm_quantum_backend

logger = logging.getLogger(__name__)


def get_provider(config: Dict[str, Any]) -> Tuple[Optional[BaseEstimator], Optional[BaseSampler]]:
    """
    Returns an Estimator and a Sampler based on the provided configuration.

    Args:
        config (Dict[str, Any]): The backend configuration dictionary.

    Returns:
        A tuple containing an Estimator and a Sampler.
    """
    backend_name = config.get("name")
    logger.info(f"Initializing backend provider for: {backend_name}")

    if backend_name == "local_simulator":
        return _get_local_simulator(config)
    elif backend_name == "ibm_quantum":
        return get_ibm_quantum_backend(config)
    else:
        raise ValueError(f"Unsupported backend: {backend_name}")


def _get_local_simulator(config: Dict[str, Any]) -> Tuple[Optional[BaseEstimator], Optional[BaseSampler]]:
    """Returns a local simulator backend.

    Uses Qiskit's built-in StatevectorEstimator/StatevectorSampler for basic simulation.
    qiskit-aer is only required when noise_model is specified.
    """
    shots = config.get("shots")
    seed = config.get("seed")
    noise_model = config.get("noise_model")

    # For Estimator
    if noise_model is not None:
        # Noise model requires qiskit-aer
        if AerEstimator is None:
            raise RuntimeError(
                "qiskit-aer is required for noise model simulation. "
                "Install with: pip install qiskit-aer"
            )
        run_options = {"shots": shots, "seed_simulator": seed}
        estimator = AerEstimator(run_options=run_options)
        logger.debug(f"Using AerEstimator with noise model and options: {run_options}")
    elif ReferenceEstimator is not None:
        # Use built-in StatevectorEstimator (no qiskit-aer needed)
        estimator = ReferenceEstimator(seed=seed) if seed is not None else ReferenceEstimator()
        logger.debug("Using StatevectorEstimator for expectation values.")
    else:
        raise RuntimeError("Qiskit primitives not available. Please install qiskit>=1.0.0")

    # For Sampler
    if noise_model is not None:
        # Noise model requires qiskit-aer
        if AerSampler is None:
            raise RuntimeError(
                "qiskit-aer is required for noise model simulation. "
                "Install with: pip install qiskit-aer"
            )
        run_options = {"shots": shots, "seed": seed}
        sampler = AerSampler(run_options=run_options)
        logger.debug(f"Using AerSampler with noise model and options: {run_options}")
    elif ReferenceSampler is not None:
        # Use built-in StatevectorSampler (no qiskit-aer needed)
        sampler = ReferenceSampler(seed=seed) if seed is not None else ReferenceSampler()
        logger.debug("Using StatevectorSampler for sampling.")
    else:
        raise RuntimeError("Qiskit primitives not available. Please install qiskit>=1.0.0")

    return estimator, sampler

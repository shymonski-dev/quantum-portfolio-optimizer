"""Local simulator configuration helpers."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

try:
    from qiskit.primitives import StatevectorEstimator as ReferenceEstimator
    from qiskit.primitives import StatevectorSampler as ReferenceSampler
except ImportError:  # pragma: no cover - fallback for environments missing statevector primitives.
    ReferenceEstimator = None  # type: ignore
    ReferenceSampler = None  # type: ignore

try:
    from qiskit_aer.primitives import Estimator as AerEstimator
    from qiskit_aer.primitives import Sampler as AerSampler
except ImportError:  # pragma: no cover - Aer optional during tests
    AerEstimator = None
    AerSampler = None

logger = logging.getLogger(__name__)


def get_default_estimator(
    shots: Optional[int] = None,
    noise_model: Any = None,
    backend_options: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    max_parallel_threads: Optional[int] = None,
    max_parallel_experiments: Optional[int] = None,
    max_memory_mb: Optional[int] = None,
) -> object:
    """Return an estimator suitable for local simulation."""
    if ReferenceEstimator is not None and shots is None and noise_model is None:
        logger.debug("Using StatevectorEstimator for analytic expectation values.")
        return ReferenceEstimator()

    if AerEstimator is not None:
        run_options: Dict[str, Any] = {}
        if shots is not None:
            run_options["shots"] = shots
        if seed is not None:
            run_options["seed_simulator"] = seed
        backend_cfg: Dict[str, Any] = dict(backend_options or {})
        if max_parallel_threads is not None:
            backend_cfg.setdefault("max_parallel_threads", max_parallel_threads)
        if max_parallel_experiments is not None:
            backend_cfg.setdefault("max_parallel_experiments", max_parallel_experiments)
        if max_memory_mb is not None:
            backend_cfg.setdefault("max_memory_mb", max_memory_mb)
        extra: Dict[str, Any] = {}
        if noise_model is not None:
            extra["noise_model"] = noise_model
        try:
            estimator = AerEstimator(
                run_options=run_options or None,
                backend_options=backend_cfg or None,
                **extra,
            )
            logger.debug("Using AerEstimator with options: %s and backend options: %s", run_options, backend_cfg)
            if noise_model is not None:
                logger.info("AerEstimator configured with noise model: %s", noise_model)
            return estimator
        except TypeError:
            if noise_model is not None:
                logger.warning("AerEstimator does not support noise_model with current configuration; falling back to noiseless statevector.")
            else:
                logger.warning("Failed to initialise AerEstimator; falling back to statevector estimator.")

    if ReferenceEstimator is not None:
        logger.warning("qiskit_aer not available or unsupported configuration; using StatevectorEstimator.")
        return ReferenceEstimator()

    raise RuntimeError("No estimator backend available. Install qiskit-aer for AerEstimator support.")


def get_default_sampler(
    shots: int = 1024,
    noise_model: Any = None,
    backend_options: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    max_parallel_threads: Optional[int] = None,
    max_parallel_experiments: Optional[int] = None,
    max_memory_mb: Optional[int] = None,
) -> object:
    if ReferenceSampler is not None and noise_model is None:
        logger.debug("Using StatevectorSampler for noiseless sampling.")
        return ReferenceSampler()

    if AerSampler is not None:
        run_options: Dict[str, Any] = {"shots": shots}
        if seed is not None:
            run_options["seed"] = seed
        backend_cfg: Dict[str, Any] = dict(backend_options or {})
        if max_parallel_threads is not None:
            backend_cfg.setdefault("max_parallel_threads", max_parallel_threads)
        if max_parallel_experiments is not None:
            backend_cfg.setdefault("max_parallel_experiments", max_parallel_experiments)
        if max_memory_mb is not None:
            backend_cfg.setdefault("max_memory_mb", max_memory_mb)
        extra: Dict[str, Any] = {}
        if noise_model is not None:
            extra["noise_model"] = noise_model
        try:
            sampler = AerSampler(
                run_options=run_options,
                backend_options=backend_cfg or None,
                **extra,
            )
            logger.debug("Using AerSampler with shots=%s and backend options: %s", shots, backend_cfg)
            if noise_model is not None:
                logger.info("AerSampler configured with noise model.")
            return sampler
        except TypeError:
            if noise_model is not None:
                logger.warning("AerSampler does not support noise_model with current configuration; falling back to noiseless sampler.")
            else:
                logger.warning("Failed to initialise AerSampler; falling back to statevector sampler.")

    if ReferenceSampler is not None:
        logger.warning("qiskit_aer not available or unsupported configuration; using StatevectorSampler.")
        return ReferenceSampler()

    raise RuntimeError("No sampler backend available. Install qiskit-aer for AerSampler support.")

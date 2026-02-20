# quantum_portfolio_optimizer/src/quantum_portfolio_optimizer/simulation/ibm_provider.py
"""IBM Quantum backend provider using qiskit-ibm-runtime.

This module provides integration with IBM Quantum hardware using the modern
qiskit-ibm-runtime package, which includes:
- EstimatorV2 and SamplerV2 primitives
- Session support for iterative algorithms like VQE
- Error mitigation options (ZNE, dynamical decoupling, etc.)
"""

from __future__ import annotations
import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from quantum_portfolio_optimizer.exceptions import (
    BackendError,
    IBMAuthenticationError,
    IBMBackendNotFoundError,
)

logger = logging.getLogger(__name__)

# Import qiskit-ibm-runtime components
try:
    from qiskit_ibm_runtime import (
        QiskitRuntimeService,
        EstimatorV2,
        SamplerV2,
        Session,
        Batch,
    )
    from qiskit_ibm_runtime.options import (
        EstimatorOptions,
        SamplerOptions,
        ResilienceOptionsV2,
        ExecutionOptionsV2,
    )
    IBM_RUNTIME_AVAILABLE = True
except ImportError:
    IBM_RUNTIME_AVAILABLE = False
    QiskitRuntimeService = None
    EstimatorV2 = None
    SamplerV2 = None
    Session = None
    Batch = None
    EstimatorOptions = None
    SamplerOptions = None
    ResilienceOptionsV2 = None
    ExecutionOptionsV2 = None


@dataclass
class ErrorMitigationConfig:
    """Configuration for IBM Quantum error mitigation options.

    Attributes:
        zne_enabled: Enable Zero-Noise Extrapolation
        zne_noise_factors: Noise amplification factors for ZNE (e.g., [1, 3, 5])
        zne_extrapolator: Extrapolation method ('linear', 'exponential', 'polynomial')
        dynamical_decoupling: Enable dynamical decoupling sequences
        dd_sequence: DD sequence type ('XX', 'XY4', 'XpXm')
        twirling_enabled: Enable Pauli twirling for noise randomization
        resilience_level: Overall resilience level (0=none, 1=basic, 2=advanced)
    """
    zne_enabled: bool = False
    zne_noise_factors: Tuple[float, ...] = (1, 3, 5)
    zne_extrapolator: str = "exponential"
    dynamical_decoupling: bool = True
    dd_sequence: str = "XpXm"
    twirling_enabled: bool = True
    resilience_level: int = 1


@dataclass
class IBMQuantumConfig:
    """Configuration for IBM Quantum backend.

    Attributes:
        device: IBM Quantum backend name (e.g., 'ibm_brisbane', 'ibm_kyoto')
        channel: Channel type ('ibm_quantum' for open, 'ibm_cloud' for premium)
        instance: Instance in format 'hub/group/project' (optional)
        use_session: Whether to use Session mode for iterative algorithms
        session_max_time: Maximum session time in seconds (default: 8 hours)
        shots: Number of shots per circuit execution
        optimization_level: Transpilation optimization level (0-3)
        error_mitigation: Error mitigation configuration
    """
    device: str = "ibm_brisbane"
    channel: str = "ibm_quantum"
    instance: Optional[str] = None
    use_session: bool = True
    session_max_time: int = 28800  # 8 hours
    shots: int = 4096
    optimization_level: int = 3
    error_mitigation: ErrorMitigationConfig = field(default_factory=ErrorMitigationConfig)


# Global session manager for reusing sessions
_active_session: Optional[Session] = None
_session_backend: Optional[str] = None


def _get_runtime_service(token: Optional[str] = None, channel: str = "ibm_quantum", instance: Optional[str] = None) -> "QiskitRuntimeService":
    """Get or create a QiskitRuntimeService instance.

    Args:
        token: IBM Quantum API token. If None, uses QE_TOKEN env var or saved credentials.
        channel: Service channel ('ibm_quantum' or 'ibm_cloud')
        instance: Instance/CRN for IBM Cloud channel

    Returns:
        QiskitRuntimeService instance

    Raises:
        BackendError: If qiskit-ibm-runtime is not installed
        IBMAuthenticationError: If authentication fails
    """
    if not IBM_RUNTIME_AVAILABLE:
        raise BackendError(
            "qiskit-ibm-runtime is not installed. "
            "Install with: pip install qiskit-ibm-runtime"
        )

    # Try to get token from environment if not provided
    if token is None:
        token = os.environ.get("QE_TOKEN") or os.environ.get("IBM_QUANTUM_TOKEN")

    # Try to get instance/CRN from environment if not provided (for IBM Cloud)
    if instance is None and channel == "ibm_cloud":
        instance = os.environ.get("IBM_CLOUD_CRN") or os.environ.get("IBM_INSTANCE")

    try:
        if token:
            if channel == "ibm_cloud" and instance:
                service = QiskitRuntimeService(channel=channel, token=token, instance=instance)
                logger.info(f"Connected to IBM Cloud with instance: {instance[:50]}...")
            else:
                service = QiskitRuntimeService(channel=channel, token=token)
                logger.info(f"Connected to IBM Quantum via {channel} channel with provided token")
        else:
            # Try to use saved credentials
            service = QiskitRuntimeService(channel=channel)
            logger.info(f"Connected to IBM Quantum via {channel} channel with saved credentials")
        return service
    except Exception as e:
        error_str = str(e).lower()
        if "token" in error_str or "auth" in error_str or "credential" in error_str:
            raise IBMAuthenticationError(
                f"Invalid or missing API token. "
                f"Ensure QE_TOKEN environment variable is set correctly. Original error: {e}"
            )
        raise IBMAuthenticationError(
            f"Failed to connect to IBM Quantum: {e}"
        )


def _build_estimator_options(config: IBMQuantumConfig) -> "EstimatorOptions":
    """Build EstimatorOptions with error mitigation settings."""
    options = EstimatorOptions(default_shots=config.shots)

    # Configure resilience/error mitigation
    em = config.error_mitigation
    options.resilience_level = em.resilience_level

    # Dynamical decoupling
    if em.dynamical_decoupling:
        options.dynamical_decoupling.enable = True
        options.dynamical_decoupling.sequence_type = em.dd_sequence

    # Twirling
    if em.twirling_enabled:
        options.twirling.enable_gates = True
        options.twirling.enable_measure = True

    # ZNE (Zero-Noise Extrapolation) - requires resilience_level >= 2
    if em.zne_enabled and em.resilience_level >= 2:
        options.resilience.zne_mitigation = True
        options.resilience.zne.noise_factors = list(em.zne_noise_factors)
        options.resilience.zne.extrapolator = em.zne_extrapolator

    return options


def _build_sampler_options(config: IBMQuantumConfig) -> "SamplerOptions":
    """Build SamplerOptions with error mitigation settings."""
    options = SamplerOptions(default_shots=config.shots)

    # Dynamical decoupling
    em = config.error_mitigation
    if em.dynamical_decoupling:
        options.dynamical_decoupling.enable = True
        options.dynamical_decoupling.sequence_type = em.dd_sequence

    return options


def get_or_create_session(
    service: "QiskitRuntimeService",
    backend_name: str,
    max_time: int = 28800,
) -> "Session":
    """Get existing session or create a new one.

    Sessions keep your jobs prioritized on the quantum hardware,
    which is important for iterative algorithms like VQE.
    """
    global _active_session, _session_backend

    # Reuse existing session if it's for the same backend and still active
    if _active_session is not None and _session_backend == backend_name:
        try:
            # Check if session is still active
            if _active_session.status() in ["Active", "Running"]:
                logger.debug(f"Reusing existing session for {backend_name}")
                return _active_session
        except Exception:
            pass  # Session might be closed, create a new one

    # Create new session
    backend = service.backend(backend_name)
    _active_session = Session(backend=backend, max_time=max_time)
    _session_backend = backend_name
    logger.info(f"Created new session for {backend_name} (max_time={max_time}s)")

    return _active_session


def close_session():
    """Close the active session if one exists."""
    global _active_session, _session_backend

    if _active_session is not None:
        try:
            _active_session.close()
            logger.info(f"Closed session for {_session_backend}")
        except Exception as e:
            logger.warning(f"Error closing session: {e}")
        finally:
            _active_session = None
            _session_backend = None


def get_ibm_quantum_backend(config: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    Returns an EstimatorV2 and SamplerV2 from IBM Quantum Runtime.

    Args:
        config: Backend configuration dictionary with keys:
            - device: IBM Quantum backend name (required)
            - channel: 'ibm_quantum' or 'ibm_cloud' (default: 'ibm_quantum')
            - instance: 'hub/group/project' format (optional)
            - token: IBM Quantum API token (optional, falls back to environment)
            - use_session: Whether to use Session mode (default: True)
            - session_max_time: Max session time in seconds (default: 28800)
            - shots: Number of shots (default: 4096)
            - optimization_level: Transpilation level 0-3 (default: 3)
            - error_mitigation: Dict with error mitigation options

    Returns:
        Tuple of (EstimatorV2, SamplerV2) configured for IBM Quantum hardware.

    Raises:
        BackendError: If qiskit-ibm-runtime is not installed or device not specified
        IBMAuthenticationError: If authentication fails
        IBMBackendNotFoundError: If the requested backend is not available
        IBMSessionError: If session creation fails
    """
    if not IBM_RUNTIME_AVAILABLE:
        raise BackendError(
            "qiskit-ibm-runtime is not installed. "
            "Install with: pip install .[ibm] or pip install qiskit-ibm-runtime"
        )

    # Parse configuration
    device = config.get("device")
    if not device:
        raise BackendError("'device' must be specified in the backend config for IBM Quantum.")

    channel = config.get("channel", "ibm_quantum")
    instance = config.get("instance")
    use_session = config.get("use_session", True)
    session_max_time = config.get("session_max_time", 28800)
    shots = config.get("shots", 4096)
    optimization_level = config.get("optimization_level", 3)

    # Parse error mitigation config
    em_config = config.get("error_mitigation", {})
    error_mitigation = ErrorMitigationConfig(
        zne_enabled=em_config.get("zne_enabled", False),
        zne_noise_factors=tuple(em_config.get("zne_noise_factors", [1, 3, 5])),
        zne_extrapolator=em_config.get("zne_extrapolator", "exponential"),
        dynamical_decoupling=em_config.get("dynamical_decoupling", True),
        dd_sequence=em_config.get("dd_sequence", "XpXm"),
        twirling_enabled=em_config.get("twirling_enabled", True),
        resilience_level=em_config.get("resilience_level", 1),
    )

    # Build full config object
    ibm_config = IBMQuantumConfig(
        device=device,
        channel=channel,
        instance=instance,
        use_session=use_session,
        session_max_time=session_max_time,
        shots=shots,
        optimization_level=optimization_level,
        error_mitigation=error_mitigation,
    )

    # Get API token from config first, then environment
    token = config.get("token")
    if not token:
        token = os.environ.get("QE_TOKEN") or os.environ.get("IBM_QUANTUM_TOKEN")

    # Connect to IBM Quantum
    service = _get_runtime_service(token=token, channel=channel, instance=instance)

    # Verify backend exists
    try:
        backend = service.backend(device)
        logger.info(f"Connected to IBM Quantum backend: {device}")
        logger.info(f"  - Qubits: {backend.num_qubits}")
        logger.info(f"  - Status: {backend.status().status_msg}")
    except Exception as e:
        available = [b.name for b in service.backends()]
        raise IBMBackendNotFoundError(
            backend_name=device,
            available_backends=available[:10],
        ) from e

    # Build options
    estimator_options = _build_estimator_options(ibm_config)
    sampler_options = _build_sampler_options(ibm_config)

    # Create primitives with or without session
    if use_session:
        session = get_or_create_session(service, device, session_max_time)
        estimator = EstimatorV2(session=session, options=estimator_options)
        sampler = SamplerV2(session=session, options=sampler_options)
        logger.info("Created EstimatorV2 and SamplerV2 with Session mode")
    else:
        estimator = EstimatorV2(mode=backend, options=estimator_options)
        sampler = SamplerV2(mode=backend, options=sampler_options)
        logger.info("Created EstimatorV2 and SamplerV2 in job mode")

    # Log error mitigation settings
    logger.info("Error mitigation settings:")
    logger.info(f"  - Resilience level: {error_mitigation.resilience_level}")
    logger.info(f"  - Dynamical decoupling: {error_mitigation.dynamical_decoupling} ({error_mitigation.dd_sequence})")
    logger.info(f"  - Twirling: {error_mitigation.twirling_enabled}")
    logger.info(f"  - ZNE: {error_mitigation.zne_enabled}")

    return estimator, sampler


def list_available_backends(channel: str = "ibm_quantum") -> list:
    """List available IBM Quantum backends.

    Args:
        channel: Service channel ('ibm_quantum' or 'ibm_cloud')

    Returns:
        List of backend names with their qubit counts
    """
    service = _get_runtime_service(channel=channel)
    backends = []
    for backend in service.backends():
        try:
            status = backend.status()
            backends.append({
                "name": backend.name,
                "qubits": backend.num_qubits,
                "status": status.status_msg,
                "pending_jobs": status.pending_jobs,
            })
        except Exception:
            backends.append({
                "name": backend.name,
                "qubits": getattr(backend, "num_qubits", "?"),
                "status": "unknown",
            })
    return sorted(backends, key=lambda x: x.get("qubits", 0), reverse=True)

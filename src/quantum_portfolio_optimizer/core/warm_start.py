"""Warm starting utilities for VQE and QAOA from classical solutions.

This module provides functions to initialize quantum variational circuits
using classical Markowitz portfolio optimization solutions, potentially
improving convergence and solution quality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from qiskit import QuantumCircuit

from quantum_portfolio_optimizer.benchmarks.classical_baseline import (
    markowitz_baseline,
)
from quantum_portfolio_optimizer.exceptions import WarmStartError

logger = logging.getLogger(__name__)


@dataclass
class WarmStartConfig:
    """Configuration for warm starting quantum solvers.

    Attributes:
        use_classical_solution: Whether to use Markowitz solution for initialization
        allocation_threshold: Minimum allocation to consider an asset selected
        noise_scale: Scale of random noise added to break symmetry
        seed: Random seed for reproducibility
    """

    use_classical_solution: bool = True
    allocation_threshold: float = 0.01
    noise_scale: float = 0.1
    seed: Optional[int] = None


@dataclass
class WarmStartResult:
    """Result from warm start initialization.

    Attributes:
        initial_parameters: Parameter vector for the quantum circuit
        classical_allocations: The classical solution used for warm starting
        estimated_improvement: Estimated improvement factor vs random init
        method: The warm start method used
    """

    initial_parameters: np.ndarray
    classical_allocations: np.ndarray
    estimated_improvement: float
    method: str


def allocations_to_rotation_angles(
    allocations: np.ndarray,
    add_noise: bool = True,
    noise_scale: float = 0.1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Convert portfolio allocations to rotation angles for amplitude encoding.

    Uses the formula: θ = 2 * arcsin(sqrt(allocation))

    This maps allocations in [0, 1] to angles that, when applied as Ry rotations
    to the |0⟩ state, produce amplitudes proportional to sqrt(allocation).

    Args:
        allocations: Portfolio weights normalized to [0, 1]
        add_noise: Whether to add small noise to break symmetry
        noise_scale: Scale of random noise (default 0.1 radians)
        seed: Random seed for noise generation

    Returns:
        Array of rotation angles in radians
    """
    # Clip allocations to valid range [0, 1]
    allocations = np.clip(allocations, 0.0, 1.0)

    # Normalize if not already summing to 1
    total = np.sum(allocations)
    if total > 0:
        allocations = allocations / total

    # Convert to rotation angles: θ = 2 * arcsin(sqrt(w))
    # This encodes weights as amplitudes: |ψ⟩ = sum_i sqrt(w_i)|i⟩
    angles = 2 * np.arcsin(np.sqrt(allocations))

    # Add small noise to break symmetry and avoid local minima
    if add_noise:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, noise_scale, size=angles.shape)
        angles = angles + noise

    return angles


def warm_start_vqe(
    expected_returns: Sequence[float],
    covariance: Sequence[Sequence[float]],
    ansatz: QuantumCircuit,
    budget: float = 1.0,
    risk_aversion: float = 0.5,
    config: Optional[WarmStartConfig] = None,
) -> WarmStartResult:
    """Generate warm start parameters for VQE from classical Markowitz solution.

    The VQE warm start uses amplitude encoding to initialize the circuit
    such that the initial state approximates the classical optimal portfolio.

    For RealAmplitudes ansatz:
    - First layer Ry rotations: θ = 2 * arcsin(sqrt(allocation))
    - Subsequent layers: small random values near zero

    Args:
        expected_returns: Expected returns for each asset
        covariance: Covariance matrix
        ansatz: The VQE ansatz circuit
        budget: Total budget constraint
        risk_aversion: Risk aversion parameter for Markowitz optimization
        config: Warm start configuration

    Returns:
        WarmStartResult with initial parameters and metadata

    Raises:
        WarmStartError: If warm start initialization fails
    """
    if config is None:
        config = WarmStartConfig()

    num_qubits = ansatz.num_qubits
    num_params = ansatz.num_parameters

    try:
        # Get classical Markowitz solution
        classical_result = markowitz_baseline(
            expected_returns=expected_returns,
            covariance=covariance,
            budget=budget,
            risk_aversion=risk_aversion,
        )

        if not classical_result.success:
            logger.warning(
                f"Classical optimization did not converge: {classical_result.message}"
            )
            # Fall back to equal allocation
            classical_allocations = np.full(num_qubits, 1.0 / num_qubits)
        else:
            classical_allocations = classical_result.allocations

    except Exception as e:
        raise WarmStartError(
            f"Failed to compute classical baseline: {e}",
            solver_type="VQE",
        ) from e

    # Ensure allocations match the number of qubits (expand if needed for resolution encoding).
    if len(classical_allocations) != num_qubits:
        if num_qubits % len(classical_allocations) == 0:
            repeat = num_qubits // len(classical_allocations)
            classical_allocations = np.repeat(classical_allocations / repeat, repeat)
        else:
            raise WarmStartError(
                f"Classical allocations length ({len(classical_allocations)}) "
                f"does not match ansatz qubits ({num_qubits})",
                solver_type="VQE",
            )

    # Convert allocations to rotation angles for first layer
    first_layer_angles = allocations_to_rotation_angles(
        classical_allocations,
        add_noise=True,
        noise_scale=config.noise_scale,
        seed=config.seed,
    )

    # Build full parameter vector
    # Assume ansatz structure: (Ry layer + entangling) * reps
    # First Ry layer uses warm-started angles, rest use small random values
    rng = np.random.default_rng(config.seed)

    if num_params <= num_qubits:
        # Single layer - just use the angles directly
        initial_parameters = first_layer_angles[:num_params]
    else:
        # Multiple layers - use warm start for first layer, small random for rest
        initial_parameters = np.zeros(num_params)
        initial_parameters[:num_qubits] = first_layer_angles

        # Small random initialization for remaining parameters
        remaining = num_params - num_qubits
        initial_parameters[num_qubits:] = rng.normal(0, 0.1, size=remaining)

    # Estimate improvement factor (heuristic)
    # Warm starting typically provides 1.5-3x speedup in convergence
    estimated_improvement = 2.0 if config.use_classical_solution else 1.0

    logger.info(
        f"VQE warm start: {num_params} parameters initialized from "
        f"classical solution with allocations: {classical_allocations.round(3)}"
    )

    return WarmStartResult(
        initial_parameters=initial_parameters,
        classical_allocations=classical_allocations,
        estimated_improvement=estimated_improvement,
        method="amplitude_encoding",
    )


def warm_start_qaoa(
    qubo_linear: np.ndarray,
    qubo_quadratic: np.ndarray,
    layers: int,
    expected_returns: Optional[Sequence[float]] = None,
    covariance: Optional[Sequence[Sequence[float]]] = None,
    budget: float = 1.0,
    risk_aversion: float = 0.5,
    config: Optional[WarmStartConfig] = None,
    mixer_type: str = "x",
) -> WarmStartResult:
    """Generate warm start parameters for QAOA using mean-field heuristics.

    QAOA parameters (gamma, beta) can be initialized using:
    1. Energy scale from QUBO coefficients (for gamma)
    2. Mean-field approximation or classical solution (for beta)

    Args:
        qubo_linear: Linear QUBO coefficients (diagonal)
        qubo_quadratic: Quadratic QUBO coefficients (off-diagonal)
        layers: Number of QAOA layers (p)
        expected_returns: Expected returns for mean-field (optional)
        covariance: Covariance matrix for mean-field (optional)
        budget: Budget constraint
        risk_aversion: Risk aversion for optional classical solution
        config: Warm start configuration

    Returns:
        WarmStartResult with initial (gamma, beta) parameters

    Raises:
        WarmStartError: If warm start initialization fails
    """
    if config is None:
        config = WarmStartConfig()

    rng = np.random.default_rng(config.seed)

    try:
        # Estimate energy scale from QUBO coefficients
        linear_scale = np.abs(qubo_linear).mean() if len(qubo_linear) > 0 else 1.0
        quad_scale = (
            np.abs(qubo_quadratic).mean() if qubo_quadratic.size > 0 else linear_scale
        )
        energy_scale = max(linear_scale, quad_scale, 1e-6)

        if mixer_type == "xy":
            # Trotterized adiabatic schedule for XY mixer (Bartschi & Eidenbenz 2020)
            # gamma increases (cost grows), beta decreases (mixer fades) with depth
            gamma_scale = energy_scale if energy_scale > 0 else 1.0
            params = []
            for p in range(layers):
                t = (p + 1) / layers  # progress from 0 to 1
                gamma_p = (0.5 / gamma_scale) * (0.5 + 0.5 * t)
                beta_p = (np.pi / 4) * (1.0 - 0.5 * t)
                params.extend([gamma_p, beta_p])
            initial_parameters = np.clip(params, 0, 2 * np.pi)
            gamma_reference = (
                float(initial_parameters[0]) if len(initial_parameters) else 0.0
            )
        else:
            # Initialize gamma based on energy scale
            # Optimal gamma is typically O(1/energy_scale)
            gamma_initial = 0.5 / energy_scale

            # Initialize beta based on mean-field approximation
            # For transverse field mixer, optimal beta starts around pi/4
            beta_initial = np.pi / 4

            # Build parameter vector: [gamma_0, beta_0, gamma_1, beta_1, ...]
            initial_parameters = np.zeros(2 * layers)

            # Use interpolation strategy: gradually increase gamma, decrease beta
            for p in range(layers):
                # Linear interpolation from initial to scaled values
                t = (p + 1) / layers

                # Gamma increases with depth
                initial_parameters[2 * p] = gamma_initial * (0.5 + 0.5 * t)

                # Beta decreases with depth (annealing-like schedule)
                initial_parameters[2 * p + 1] = beta_initial * (1.0 - 0.3 * t)
            gamma_reference = gamma_initial

        # Add small noise for symmetry breaking
        if config.noise_scale > 0:
            noise = rng.normal(
                0, config.noise_scale * 0.1, size=initial_parameters.shape
            )
            initial_parameters = initial_parameters + noise

        # Clip to valid ranges
        initial_parameters[::2] = np.clip(
            initial_parameters[::2], 0, 2 * np.pi
        )  # gamma
        initial_parameters[1::2] = np.clip(initial_parameters[1::2], 0, np.pi)  # beta

        # Get classical allocations if available
        classical_allocations = np.zeros(len(qubo_linear))
        if expected_returns is not None and covariance is not None:
            try:
                classical_result = markowitz_baseline(
                    expected_returns=expected_returns,
                    covariance=covariance,
                    budget=budget,
                    risk_aversion=risk_aversion,
                )
                if classical_result.success:
                    classical_allocations = classical_result.allocations
            except Exception:
                pass  # Use zeros if classical optimization fails

        logger.info(
            f"QAOA warm start: {2 * layers} parameters initialized "
            f"(energy_scale={energy_scale:.4f}, gamma_init={gamma_reference:.4f})"
        )

        return WarmStartResult(
            initial_parameters=initial_parameters,
            classical_allocations=classical_allocations,
            estimated_improvement=1.5,
            method="mean_field_heuristic",
        )

    except Exception as e:
        raise WarmStartError(
            f"Failed to compute QAOA warm start: {e}",
            solver_type="QAOA",
        ) from e


def get_binary_initial_state(
    allocations: np.ndarray,
    threshold: float = 0.01,
) -> str:
    """Convert continuous allocations to binary selection string.

    Useful for QAOA warm starting where we want to bias toward
    the classical solution.

    Args:
        allocations: Continuous portfolio weights
        threshold: Minimum allocation to consider asset selected

    Returns:
        Binary string representing asset selection (e.g., "1010")
    """
    binary = ["1" if a >= threshold else "0" for a in allocations]
    return "".join(binary)


def estimate_initial_energy(
    allocations: np.ndarray,
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    risk_aversion: float,
) -> float:
    """Estimate the QUBO energy for a given allocation.

    This can be used to verify that the warm start solution is reasonable.

    Args:
        allocations: Portfolio weights
        expected_returns: Expected returns vector
        covariance: Covariance matrix
        risk_aversion: Risk aversion parameter

    Returns:
        Estimated QUBO objective value (lower is better)
    """
    weights = np.asarray(allocations)
    mu = np.asarray(expected_returns)
    sigma = np.asarray(covariance)

    portfolio_return = mu @ weights
    portfolio_variance = weights @ sigma @ weights

    # QUBO minimizes: risk_aversion * variance - return
    energy = risk_aversion * portfolio_variance - portfolio_return

    return float(energy)

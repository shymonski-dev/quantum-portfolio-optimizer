"""
Provider-agnostic Zero-Noise Extrapolation (ZNE) via gate folding.

Works on local simulators and real hardware alike. Proven to give 31.6%
improvement on IBM hardware for portfolio optimization problems (arXiv:2602.09047).
"""

from __future__ import annotations

import logging
from typing import Callable, List

import numpy as np
from qiskit import QuantumCircuit

logger = logging.getLogger(__name__)


def fold_circuit(circuit: QuantumCircuit, noise_factor: int) -> QuantumCircuit:
    """
    Amplify noise by gate folding: replace each gate G with G·G†·G (noise_factor=3),
    G·G†·G·G†·G (noise_factor=5), etc.

    Args:
        circuit: A BOUND QuantumCircuit (all parameters already assigned).
        noise_factor: Odd integer >= 1. noise_factor=1 returns a copy unchanged.

    Returns:
        New QuantumCircuit with folded gates.

    Raises:
        ValueError: If noise_factor is even or < 1.
    """
    if noise_factor < 1 or noise_factor % 2 == 0:
        raise ValueError(
            f"noise_factor must be an odd positive integer, got {noise_factor}"
        )
    if noise_factor == 1:
        return circuit.copy()

    # Number of extra G†·G repetitions per gate
    num_extra = (noise_factor - 1) // 2

    folded = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    folded.metadata = circuit.metadata

    for instruction in circuit.data:
        operation = instruction.operation
        qargs = instruction.qubits
        cargs = instruction.clbits

        # Apply the original gate
        folded.append(operation, qargs, cargs)

        # Append G†·G pairs
        for _ in range(num_extra):
            try:
                inv_op = operation.inverse()
            except Exception:
                # If inverse is not available, skip folding for this gate
                logger.warning(
                    "Could not invert gate %s -- skipping fold for this gate.",
                    operation.name,
                )
                break
            folded.append(inv_op, qargs, cargs)
            folded.append(operation, qargs, cargs)

    return folded


def zne_extrapolate(
    noise_factors: List[float],
    expectation_values: List[float],
    extrapolator: str = "linear",
) -> float:
    """
    Extrapolate expectation values to zero noise level.

    Args:
        noise_factors: List of noise scale factors (e.g. [1, 3, 5]).
        expectation_values: Corresponding expectation values at each noise level.
        extrapolator: "linear", "quadratic", or "richardson".
            "richardson" fits a polynomial of degree len-1 (Richardson extrapolation).

    Returns:
        Extrapolated value at noise_factor=0.

    Raises:
        ValueError: If inputs have mismatched lengths or extrapolator is unsupported.
    """
    if len(noise_factors) != len(expectation_values):
        raise ValueError(
            f"noise_factors length ({len(noise_factors)}) must match "
            f"expectation_values length ({len(expectation_values)})"
        )
    if len(noise_factors) < 2:
        raise ValueError("At least 2 (noise_factor, value) pairs are required.")

    supported = {"linear", "quadratic", "richardson"}
    if extrapolator not in supported:
        raise ValueError(
            f"Unsupported extrapolator '{extrapolator}'. Choose from {supported}."
        )

    nf = np.array(noise_factors, dtype=float)
    ev = np.array(expectation_values, dtype=float)

    if extrapolator == "linear":
        degree = 1
    elif extrapolator == "quadratic":
        degree = 2
    else:  # richardson
        degree = len(noise_factors) - 1

    coeffs = np.polyfit(nf, ev, deg=degree)
    # Evaluate polynomial at nf=0: only the constant term (last coefficient)
    return float(coeffs[-1])


def zne_evaluate(
    eval_fn: Callable[[QuantumCircuit], float],
    circuit: QuantumCircuit,
    noise_factors: List[int],
    extrapolator: str = "linear",
) -> float:
    """
    Run eval_fn at each noise level and extrapolate to zero noise.

    Args:
        eval_fn: Function that takes a QuantumCircuit and returns a float energy.
        circuit: Bound QuantumCircuit to evaluate.
        noise_factors: Odd integers (e.g. [1, 3, 5]).
        extrapolator: Passed to zne_extrapolate.

    Returns:
        ZNE-mitigated expectation value.
    """
    values = []
    for nf in noise_factors:
        folded = fold_circuit(circuit, nf)
        val = eval_fn(folded)
        logger.debug("ZNE noise_factor=%d -> value=%.6f", nf, val)
        values.append(val)

    result = zne_extrapolate(noise_factors, values, extrapolator)
    logger.debug("ZNE extrapolated value (nf->0): %.6f", result)
    return result

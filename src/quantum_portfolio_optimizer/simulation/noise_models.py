"""Noise model helpers for simulators."""

from __future__ import annotations

from typing import Optional

try:
    from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
except ImportError:  # pragma: no cover
    NoiseModel = None  # type: ignore

try:
    from qiskit_aer.noise import depolarizing_error
except ImportError:  # pragma: no cover
    depolarizing_error = None  # type: ignore


def simple_depolarising_noise(
    p_one: float = 0.001, p_two: float = 0.01
) -> Optional["NoiseModel"]:
    if NoiseModel is None or depolarizing_error is None:
        return None
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(p_one, 1), ["x", "y", "z", "h", "rx", "ry", "rz"]
    )
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p_two, 2), ["cx"])
    return noise_model


def simple_thermal_noise(
    t1: float = 50e3, t2: float = 70e3, gate_time: float = 50
) -> Optional["NoiseModel"]:
    if NoiseModel is None:
        return None
    noise_model = NoiseModel()
    tr_error = thermal_relaxation_error(t1, t2, gate_time)
    noise_model.add_all_qubit_quantum_error(tr_error, ["x", "rx", "ry", "rz"])
    noise_model.add_all_qubit_quantum_error(tr_error.tensor(tr_error), ["cx"])
    return noise_model

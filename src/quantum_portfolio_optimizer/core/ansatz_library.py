"""Collection of ansatz utilities used by the local VQE solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2, RealAmplitudes


def build_real_amplitudes(
    num_qubits: int,
    reps: int = 2,
    entanglement: str | Sequence[Sequence[int]] = "full",
    insert_barriers: bool = False,
) -> RealAmplitudes:
    return RealAmplitudes(num_qubits=num_qubits, reps=reps, entanglement=entanglement, insert_barriers=insert_barriers)


def build_cyclic_ansatz(num_qubits: int, reps: int = 1) -> QuantumCircuit:
    """Construct a light-weight cyclic entangling ansatz with parameter sharing."""
    qc = QuantumCircuit(num_qubits, name="CyclicAnsatz")
    from qiskit.circuit import ParameterVector

    for layer in range(reps):
        params = ParameterVector(f"theta_{layer}", length=2 * num_qubits)
        for qubit in range(num_qubits):
            qc.ry(params[2 * qubit], qubit)
            qc.rz(params[2 * qubit + 1], qubit)
        for qubit in range(num_qubits):
            qc.cx(qubit, (qubit + 1) % num_qubits)
    return qc


def build_efficient_su2(num_qubits: int, reps: int = 1, entanglement: str = "linear") -> EfficientSU2:
    return EfficientSU2(num_qubits=num_qubits, reps=reps, entanglement=entanglement)


def get_ansatz(name: str, num_qubits: int, **kwargs) -> QuantumCircuit:
    name_lower = name.lower()
    if name_lower in {"real", "real_amplitudes", "ra"}:
        return build_real_amplitudes(num_qubits=num_qubits, **kwargs)
    if name_lower in {"cyclic", "cycle"}:
        return build_cyclic_ansatz(num_qubits=num_qubits, **kwargs)
    if name_lower in {"efficient_su2", "esu2"}:
        return build_efficient_su2(num_qubits=num_qubits, **kwargs)
    raise ValueError(f"Unsupported ansatz name '{name}'.")


def initialise_parameters(
    circuit: QuantumCircuit,
    strategy: str = "zeros",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate an initial parameter vector for a given ansatz.

    Parameters
    ----------
    circuit:
        Ansatz circuit with parameter objects.
    strategy:
        'zeros' (default), 'uniform', or 'normal'.
    seed:
        Optional RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    num_params = circuit.num_parameters

    if strategy == "zeros":
        return np.zeros(num_params, dtype=float)
    if strategy == "uniform":
        return rng.uniform(-np.pi, np.pi, size=num_params)
    if strategy == "normal":
        return rng.normal(0.0, 0.2, size=num_params)
    raise ValueError(f"Unknown initialisation strategy '{strategy}'.")


@dataclass
class AnsatzReport:
    name: str
    num_qubits: int
    num_parameters: int
    depth: int
    size: int


def analyse_circuit(circuit: QuantumCircuit, name: Optional[str] = None) -> AnsatzReport:
    depth = circuit.depth()
    size = circuit.size()
    num_params = circuit.num_parameters
    return AnsatzReport(
        name=name or circuit.name,
        num_qubits=circuit.num_qubits,
        num_parameters=num_params,
        depth=depth,
        size=size,
    )


def compare_ansatze(ansatze: Iterable[QuantumCircuit]) -> List[AnsatzReport]:
    return [analyse_circuit(ansatz) for ansatz in ansatze]

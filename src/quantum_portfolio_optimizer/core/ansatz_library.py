"""Collection of ansatz utilities used by the local VQE solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2, real_amplitudes


def build_real_amplitudes(
    num_qubits: int,
    reps: int = 3,  # 3 layers as per paper
    entanglement: str | Sequence[Sequence[int]] = "reverse_linear",  # Reverse linear from paper
    insert_barriers: bool = False,
) -> QuantumCircuit:
    return real_amplitudes(num_qubits=num_qubits, reps=reps, entanglement=entanglement, insert_barriers=insert_barriers)


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


def build_efficient_su2(num_qubits: int, reps: int = 1, entanglement: str = "linear") -> QuantumCircuit:
    return efficient_su2(num_qubits=num_qubits, reps=reps, entanglement=entanglement)


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
    scale: float = 1.0,
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
    scale:
        Optional scaling factor applied to random initialisations.
    """
    rng = np.random.default_rng(seed)
    num_params = circuit.num_parameters

    if strategy == "zeros":
        return np.zeros(num_params, dtype=float)
    if strategy == "uniform":
        return rng.uniform(-np.pi, np.pi, size=num_params) * scale
    if strategy == "uniform_small":
        return rng.uniform(-0.2, 0.2, size=num_params) * scale
    if strategy == "normal":
        return rng.normal(0.0, 0.2, size=num_params) * scale
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


def generate_ansatz_family(
    num_qubits: int,
    include_real: bool = True,
    include_cyclic: bool = True,
    include_efficient: bool = False,
    real_configs: Optional[Sequence[Tuple[int, str]]] = None,
) -> List[QuantumCircuit]:
    circuits: List[QuantumCircuit] = []
    if include_real:
        configs = real_configs or [(2, "reverse_linear"), (3, "reverse_linear"), (2, "full")]
        for reps, ent in configs:
            circuits.append(build_real_amplitudes(num_qubits=num_qubits, reps=reps, entanglement=ent))
    if include_cyclic:
        circuits.append(build_cyclic_ansatz(num_qubits=num_qubits, reps=2))
    if include_efficient:
        circuits.append(build_efficient_su2(num_qubits=num_qubits, reps=2, entanglement="linear"))
    return circuits


def evaluate_initialisations(
    circuit: QuantumCircuit,
    strategies: Sequence[str],
    sample_count: int = 32,
    seed: Optional[int] = None,
    scale: float = 1.0,
) -> dict:
    """Evaluate summary statistics for different parameter initialisation strategies."""
    rng = np.random.default_rng(seed)
    results = {}
    for strategy in strategies:
        samples = []
        for _ in range(sample_count):
            strat_seed = int(rng.integers(0, 1 << 32))
            params = initialise_parameters(circuit, strategy=strategy, seed=strat_seed, scale=scale)
            samples.append(params)
        stacked = np.vstack(samples)
        results[strategy] = {
            "mean": np.mean(stacked, axis=0),
            "std": np.std(stacked, axis=0),
            "norm": np.linalg.norm(stacked, axis=1).mean(),
        }
    return results

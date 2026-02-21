"""Circuit partitioning and knitting utilities for large-scale portfolio optimization.

This module leverages qiskit-addon-cutting to split large portfolio circuits
across multiple quantum processor modules (e.g., IBM Kookaburra architecture).
"""

from __future__ import annotations

import logging

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, PauliList

try:
    from qiskit_addon_cutting import (
        partition_problem,
        generate_cutting_experiments,
        reconstruct_expectation_values,
    )

    CUTTING_AVAILABLE = True
except ImportError:
    CUTTING_AVAILABLE = False

logger = logging.getLogger(__name__)


def run_partitioned_vqe_step(
    sampler: object,
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    partitions: list[list[int]],
) -> float:
    """Run a single VQE energy evaluation using circuit knitting.

    Args:
        sampler: Qiskit SamplerV2 primitive.
        circuit: The bound quantum circuit.
        observable: The observable to evaluate.
        partitions: List of qubit index groups defining each partition.

    Returns:
        Expectation value reconstructed from sub-experiments.
    """
    if not CUTTING_AVAILABLE:
        raise ImportError("qiskit-addon-cutting is required for circuit partitioning.")

    # 1. Map qubit-to-partition labels for qiskit-addon-cutting
    num_qubits = circuit.num_qubits
    partition_labels = [None] * num_qubits
    for p_idx, qubit_indices in enumerate(partitions):
        for q_idx in qubit_indices:
            if q_idx < num_qubits:
                partition_labels[q_idx] = p_idx

    if None in partition_labels:
        next_p = len(partitions)
        for i in range(num_qubits):
            if partition_labels[i] is None:
                partition_labels[i] = next_p

    # 2. Convert observable to PauliList
    pauli_list = PauliList(observable.paulis)

    # 3. Partition the problem
    partitioned_problem = partition_problem(
        circuit=circuit, partition_labels=partition_labels, observables=pauli_list
    )

    # 4. Generate experiments
    subexperiments, coefficients = generate_cutting_experiments(
        partitioned_problem.subcircuits,
        partitioned_problem.subobservables,
        num_samples=np.inf,
    )

    # 5. Execute sub-experiments
    # subexperiments is a dict[partition_label, list[QuantumCircuit]]
    results = {}
    for label, circuits in subexperiments.items():
        # Run circuits for this partition
        job = sampler.run(circuits)
        results[label] = job.result()

    # 6. Knit results back together
    reconstructed_values = reconstruct_expectation_values(
        results, coefficients, partitioned_problem.subobservables
    )

    # Reconstruct gives <Pauli_i>, compute total energy: sum(coeff_i * <Pauli_i>)
    energy = np.dot(observable.coeffs, reconstructed_values)

    return float(np.real(energy))

import numpy as np

from quantum_portfolio_optimizer.core.ansatz_library import (
    analyse_circuit,
    build_cyclic_ansatz,
    build_real_amplitudes,
    compare_ansatze,
    initialise_parameters,
)


def test_real_amplitudes_build():
    ansatz = build_real_amplitudes(num_qubits=3, reps=1)
    params = initialise_parameters(ansatz, strategy="zeros")
    assert params.shape[0] == ansatz.num_parameters


def test_cyclic_ansatz_has_parameters():
    ansatz = build_cyclic_ansatz(num_qubits=3, reps=1)
    assert ansatz.num_parameters == 6  # 2 parameters per qubit


def test_ansatz_analysis():
    ansatz1 = build_real_amplitudes(2, reps=1)
    reports = compare_ansatze([ansatz1])
    assert reports[0].num_parameters == ansatz1.num_parameters
    report = analyse_circuit(ansatz1)
    assert report.depth >= 1

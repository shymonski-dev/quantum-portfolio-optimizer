from quantum_portfolio_optimizer.core.optimizer_interface import DifferentialEvolutionConfig
from quantum_portfolio_optimizer.benchmarks.phase2 import (
    BenchmarkResult,
    benchmark_ansatze,
    build_phase2_qubo,
    evaluate_noise_levels,
)


def small_optimizer_config(num_qubits: int) -> DifferentialEvolutionConfig:
    bounds = [(-1.0, 1.0)] * num_qubits
    return DifferentialEvolutionConfig(bounds=bounds, maxiter=3, popsize=4, seed=1)


def test_benchmark_ansatze_returns_results():
    qubo = build_phase2_qubo(num_assets=2, num_steps=1, seed=3)
    configs = [("real_amplitudes", {"reps": 1, "entanglement": "reverse_linear"})]
    results = benchmark_ansatze(qubo, configs, small_optimizer_config, shots=None)
    assert results, "Benchmark should return at least one result"
    first = results[0]
    assert isinstance(first, BenchmarkResult)
    assert first.evaluations > 0


def test_evaluate_noise_levels_runs():
    qubo = build_phase2_qubo(num_assets=2, num_steps=1, seed=4)
    problem = qubo.build()
    configs = [("real_amplitudes", {"reps": 1, "entanglement": "reverse_linear"})]
    result = benchmark_ansatze(qubo, configs, small_optimizer_config, shots=None)[0]
    noise_results = evaluate_noise_levels(
        problem,
        result.ansatz_name,
        result.ansatz_options,
        result.optimal_parameters,
        noise_levels=[0.0],
        shots=128,
        seed=5,
    )
    assert len(noise_results) == 1
    level, energy = noise_results[0]
    assert level == 0.0
    assert isinstance(energy, float)

from quantum_portfolio_optimizer.core.optimizer_interface import (
    DifferentialEvolutionConfig,
)
from quantum_portfolio_optimizer.benchmarks.phase2 import (
    BenchmarkResult,
    benchmark_ansatze,
    build_phase2_qubo,
    evaluate_noise_levels,
)
from quantum_portfolio_optimizer.simulation.provider import get_provider


def small_optimizer_config(num_qubits: int) -> DifferentialEvolutionConfig:
    bounds = [(-1.0, 1.0)] * num_qubits
    return DifferentialEvolutionConfig(bounds=bounds, maxiter=3, popsize=4, seed=1)


def test_benchmark_ansatze_returns_results(tmp_path):
    qubo = build_phase2_qubo(num_assets=2, num_steps=1, seed=3)
    configs = [("real_amplitudes", {"reps": 1, "entanglement": "reverse_linear"})]
    backend_config = {"name": "local_simulator", "shots": None, "seed": 1}
    estimator, _ = get_provider(backend_config)
    results = benchmark_ansatze(
        qubo, configs, small_optimizer_config, estimator, cache_dir=tmp_path
    )
    assert results, "Benchmark should return at least one result"
    first = results[0]
    assert isinstance(first, BenchmarkResult)
    assert first.evaluations > 0
    assert first.cache_metadata is not None


def test_cache_reuse(tmp_path):
    qubo = build_phase2_qubo(num_assets=2, num_steps=1, seed=5)
    configs = [("real_amplitudes", {"reps": 1, "entanglement": "reverse_linear"})]
    backend_config = {"name": "local_simulator", "shots": None, "seed": 1}
    estimator, _ = get_provider(backend_config)
    results_first = benchmark_ansatze(
        qubo, configs, small_optimizer_config, estimator, cache_dir=tmp_path
    )
    assert results_first
    assert results_first[0].cache_metadata is not None
    # Second run should hit cache and run quickly; just ensure result content matches and evaluations unchanged.
    results_second = benchmark_ansatze(
        qubo, configs, small_optimizer_config, estimator, cache_dir=tmp_path
    )
    assert results_second[0].evaluations == results_first[0].evaluations
    assert results_second[0].cache_metadata is not None


def test_evaluate_noise_levels_runs(tmp_path):
    qubo = build_phase2_qubo(num_assets=2, num_steps=1, seed=4)
    problem = qubo.build()
    configs = [("real_amplitudes", {"reps": 1, "entanglement": "reverse_linear"})]
    backend_config = {"name": "local_simulator", "shots": None, "seed": 1}
    estimator, _ = get_provider(backend_config)
    result = benchmark_ansatze(
        qubo, configs, small_optimizer_config, estimator, cache_dir=tmp_path
    )[0]
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

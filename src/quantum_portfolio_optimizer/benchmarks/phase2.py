"""Phase II benchmarking utilities for ansatz and optimiser evaluation."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

from ..core.ansatz_library import (
    analyse_circuit,
    evaluate_initialisations,
    generate_ansatz_family,
    get_ansatz,
)
from ..core.optimizer_interface import DifferentialEvolutionConfig
from ..core.qubo_formulation import PortfolioQUBO
from ..core.vqe_solver import PortfolioVQESolver
from ..data.sample_datasets import generate_synthetic_dataset
from ..simulation.local_backend import get_default_estimator, get_default_sampler
from ..simulation.noise_models import simple_depolarising_noise


@dataclass
class BenchmarkResult:
    ansatz_name: str
    reps: int
    entanglement: str
    optimal_value: float
    evaluations: int
    converged: bool
    history: List[float]
    optimal_parameters: np.ndarray
    ansatz_options: Dict


def build_phase2_qubo(num_assets: int = 3, num_steps: int = 2, seed: int = 42) -> PortfolioQUBO:
    dataset = generate_synthetic_dataset(num_assets=num_assets, num_points=64, seed=seed)
    mu = dataset.returns.mean().values
    covariance = np.cov(dataset.returns.values.T)
    return PortfolioQUBO(
        expected_returns=[mu] * num_steps,
        covariance=covariance,
        budget=1.0,
        risk_aversion=1000,
        transaction_cost=0.01,
        time_steps=num_steps,
        resolution_qubits=1,
        penalty_strength=1.0,
    )


def benchmark_ansatze(
    qubo: PortfolioQUBO,
    ansatz_configs: Iterable[Tuple[str, Dict]],
    optimizer_factory: Callable[[int], DifferentialEvolutionConfig],
    shots: int | None = None,
) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []
    problem = qubo.build()
    num_qubits = problem.num_variables
    estimator = get_default_estimator(shots=shots)

    for name, options in ansatz_configs:
        solver = PortfolioVQESolver(
            estimator=estimator,
            ansatz_name=name,
            ansatz_options=options,
            parameter_bounds=2 * np.pi,
            optimizer_config=optimizer_factory(num_qubits),
            seed=123,
        )
        vqe_result = solver.solve(problem)
        ansatz = get_ansatz(name, num_qubits=num_qubits, **options)
        report = analyse_circuit(ansatz)
        results.append(
            BenchmarkResult(
                ansatz_name=name,
                reps=options.get("reps", report.depth),
                entanglement=options.get("entanglement", "reverse_linear"),
                optimal_value=vqe_result.optimal_value,
                evaluations=vqe_result.num_evaluations,
                converged=vqe_result.converged,
                history=vqe_result.history,
                optimal_parameters=vqe_result.optimal_parameters,
                ansatz_options=dict(options),
            )
        )
    return results


def make_optimizer_config(num_qubits: int) -> DifferentialEvolutionConfig:
    bounds = [(-2 * np.pi, 2 * np.pi)] * num_qubits
    return DifferentialEvolutionConfig(
        bounds=bounds,
        popsize=10,
        maxiter=80,
        convergence_threshold=0.01,
        convergence_window=12,
    )


def summarize_results(results: List[BenchmarkResult]) -> None:
    print("Ansatz Benchmark Summary")
    for res in results:
        best = min(res.history) if res.history else float("nan")
        print(
            f"{res.ansatz_name:15s} reps={res.reps:<2} ent={res.entanglement:15s} "
            f"opt={res.optimal_value: .4f} best={best: .4f} evals={res.evaluations:<4} converged={res.converged}"
        )

    opt_values = [res.optimal_value for res in results]
    if opt_values:
        print("Overall best energy:", min(opt_values))
        print("Mean energy:", statistics.mean(opt_values))


def analyse_initialisations(num_qubits: int = 6, sample_count: int = 64, seed: int = 7) -> Dict[str, Dict[str, Dict[str, float]]]:
    family = generate_ansatz_family(num_qubits=num_qubits, include_cyclic=False, include_efficient=False)
    circuits = {f"real_{idx}": circuit for idx, circuit in enumerate(family)}
    strategies = ["zeros", "uniform_small", "uniform", "normal"]
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, circuit in circuits.items():
        stats = evaluate_initialisations(circuit, strategies=strategies, sample_count=sample_count, seed=seed)
        summary[name] = stats
    return summary


def energy_from_bitstring(bits: np.ndarray, qubo_problem) -> float:
    return float(qubo_problem.offset + qubo_problem.linear @ bits + bits @ qubo_problem.quadratic @ bits)


def evaluate_noise_levels(
    qubo_problem,
    ansatz_name: str,
    ansatz_options: Dict,
    parameters: np.ndarray,
    noise_levels: Iterable[float],
    shots: int = 4096,
    seed: int = 123,
) -> List[Tuple[float, float]]:
    ansatz = get_ansatz(ansatz_name, num_qubits=qubo_problem.num_variables, **ansatz_options)
    circuit = ansatz.assign_parameters(parameters)
    circuit_measure = circuit.copy()
    circuit_measure.measure_all()
    results = []
    for level in noise_levels:
        noise_model = simple_depolarising_noise(p_one=level, p_two=2 * level)
        sampler = get_default_sampler(shots=shots, noise_model=noise_model, seed=seed)
        sample_result = sampler.run([(circuit_measure, [])]).result()
        first = sample_result[0]
        key = next(iter(first.data.keys()))
        bitarray = first.data[key]
        counts = bitarray.get_counts()
        total_shots = bitarray.num_shots or shots
        energy = 0.0
        num_vars = qubo_problem.num_variables
        for bitstring, count in counts.items():
            padded = bitstring.zfill(num_vars)
            bits = np.array(list(padded[::-1]), dtype=int)
            probability = count / total_shots
            energy += probability * energy_from_bitstring(bits, qubo_problem)
        results.append((level, energy))
    return results


def run_phase2_benchmark() -> None:
    qubo_builder = build_phase2_qubo()
    configs = [
        ("real_amplitudes", {"reps": 2, "entanglement": "reverse_linear"}),
        ("real_amplitudes", {"reps": 3, "entanglement": "reverse_linear"}),
        ("real_amplitudes", {"reps": 2, "entanglement": "full"}),
        ("cyclic", {"reps": 2}),
    ]
    results = benchmark_ansatze(qubo_builder, configs, make_optimizer_config)
    summarize_results(results)

    init_summary = analyse_initialisations(num_qubits=qubo_builder.build().num_variables)
    for name, stats in init_summary.items():
        print(f"Initialisation stats for {name}:")
        for strategy, values in stats.items():
            print(
                f"  {strategy:12s} | mean-norm={np.linalg.norm(values['mean']):.3f} "
                f"| avg-param-norm={values['norm']:.3f} | std-mean={np.mean(values['std']):.3f}"
            )

    problem = qubo_builder.build()
    best = min(results, key=lambda r: r.optimal_value)
    noise_levels = [0.0, 0.001, 0.005]
    noise_results = evaluate_noise_levels(problem, best.ansatz_name, best.ansatz_options, best.optimal_parameters, noise_levels)
    print("Noise impact (depolarising p1, p2=2p1):")
    for level, energy in noise_results:
        print(f"  p={level:.4f} -> expected energy {energy:.4f}")

"""Phase II benchmarking utilities for ansatz and optimiser evaluation."""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

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
from ..simulation.provider import get_provider
from ..simulation.noise_models import simple_depolarising_noise
from ..utils.hashing import deterministic_hash
from ..utils.json_cache import read_json_cache, write_json_cache


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
    cache_metadata: Optional[Dict[str, str]] = None


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
    estimator: object,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []
    problem = qubo.build()
    num_qubits = problem.num_variables
    qubo_signature = deterministic_hash(
        {
            "linear": problem.linear.round(12).tolist(),
            "quadratic": problem.quadratic.round(12).tolist(),
            "offset": round(problem.offset, 12),
        }
    )

    for name, options in ansatz_configs:
        opt_config = optimizer_factory(num_qubits)
        cache_key = None
        cache_metadata: Optional[Dict[str, str]] = None
        if use_cache and cache_dir is not None:
            payload = {
                "qubo": qubo_signature,
                "ansatz_name": name,
                "ansatz_options": options,
                "optimizer": asdict(opt_config),
                "shots": getattr(estimator, 'shots', None),
            }
            cache_key = deterministic_hash(payload)
            cached = read_json_cache(cache_dir, cache_key)
            if cached is not None:
                cache_metadata = cached.get("metadata")
                results.append(
                    BenchmarkResult(
                        ansatz_name=cached["ansatz_name"],
                        reps=cached["reps"],
                        entanglement=cached["entanglement"],
                        optimal_value=cached["optimal_value"],
                        evaluations=cached["evaluations"],
                        converged=cached["converged"],
                        history=cached["history"],
                        optimal_parameters=np.asarray(cached["optimal_parameters"]),
                        ansatz_options=cached["ansatz_options"],
                        cache_metadata=cache_metadata,
                    )
                )
                if cache_metadata:
                    print(
                        f"[cache hit] {name} -> {cache_metadata.get('source', 'n/a')} "
                        f"@ {cache_metadata.get('timestamp', 'unknown')}"
                    )
                continue

        solver = PortfolioVQESolver(
            estimator=estimator,
            ansatz_name=name,
            ansatz_options=options,
            parameter_bounds=2 * np.pi,
            optimizer_config=opt_config,
            seed=123,
        )
        vqe_result = solver.solve(problem)
        ansatz = get_ansatz(name, num_qubits=num_qubits, **options)
        report = analyse_circuit(ansatz)
        benchmark_entry = BenchmarkResult(
            ansatz_name=name,
            reps=options.get("reps", report.depth),
            entanglement=options.get("entanglement", "reverse_linear"),
            optimal_value=vqe_result.optimal_value,
            evaluations=vqe_result.num_evaluations,
            converged=vqe_result.converged,
            history=vqe_result.history,
            optimal_parameters=vqe_result.optimal_parameters,
            ansatz_options=dict(options),
            cache_metadata=None,
        )
        results.append(benchmark_entry)
        if cache_dir is not None and cache_key is not None:
            metadata = write_json_cache(
                cache_dir,
                cache_key,
                {
                    "ansatz_name": benchmark_entry.ansatz_name,
                    "reps": benchmark_entry.reps,
                    "entanglement": benchmark_entry.entanglement,
                    "optimal_value": benchmark_entry.optimal_value,
                    "evaluations": benchmark_entry.evaluations,
                    "converged": benchmark_entry.converged,
                    "history": benchmark_entry.history,
                    "optimal_parameters": benchmark_entry.optimal_parameters.tolist(),
                    "ansatz_options": benchmark_entry.ansatz_options,
                },
            )
            benchmark_entry.cache_metadata = metadata
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
        cache_note = ""
        if res.cache_metadata:
            cache_note = (
                f" [cached {res.cache_metadata.get('timestamp', 'unknown')}"
                f" @ {res.cache_metadata.get('source', 'n/a')}]"
            )
        print(
            f"{res.ansatz_name:15s} reps={res.reps:<2} ent={res.entanglement:15s} "
            f"opt={res.optimal_value: .4f} best={best: .4f} evals={res.evaluations:<4} "
            f"converged={res.converged}{cache_note}"
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
        backend_config = {"name": "local_simulator", "shots": shots, "seed": seed, "noise_model": noise_model}
        _, sampler = get_provider(backend_config)
        
        # Handle V1/V2 primitive differences
        try:
            # V2 interface
            job_result = sampler.run([(circuit_measure, [])]).result()
            # Extract counts from V2 result
            first = job_result[0]
            key = next(iter(first.data.keys()))
            bitarray = first.data[key]
            counts = bitarray.get_counts()
            total_shots = bitarray.num_shots or shots
        except TypeError:
            # V1 interface (AerSampler)
            job_result = sampler.run([circuit_measure], shots=shots).result()
            # Extract counts from V1 result (quasi_dists)
            dist = job_result.quasi_dists[0]
            total_shots = shots or 1024
            counts = {format(k, f"0{qubo_problem.num_variables}b"): v * total_shots for k, v in dist.items()}

        energy = 0.0
        num_vars = qubo_problem.num_variables
        for bitstring, count in counts.items():
            padded = bitstring.zfill(num_vars)
            bits = np.array(list(padded[::-1]), dtype=int)
            probability = count / total_shots
            energy += probability * energy_from_bitstring(bits, qubo_problem)
        results.append((level, energy))
    return results


def run_phase2_benchmark(cache_dir: Optional[Path] = None, use_cache: bool = True) -> None:
    qubo_builder = build_phase2_qubo()
    configs = [
        ("real_amplitudes", {"reps": 2, "entanglement": "reverse_linear"}),
        ("real_amplitudes", {"reps": 3, "entanglement": "reverse_linear"}),
        ("real_amplitudes", {"reps": 2, "entanglement": "full"}),
        ("cyclic", {"reps": 2}),
    ]
    if cache_dir is None:
        cache_dir = Path(".benchmark_cache")
    
    backend_config = {"name": "local_simulator", "shots": 4096, "seed": 123}
    estimator, _ = get_provider(backend_config)

    results = benchmark_ansatze(qubo_builder, configs, make_optimizer_config, estimator, cache_dir=cache_dir, use_cache=use_cache)
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
    
    try:
        noise_levels = [0.0, 0.001, 0.005]
        noise_results = evaluate_noise_levels(
            problem, 
            best.ansatz_name, 
            best.ansatz_options, 
            best.optimal_parameters, 
            noise_levels
        )
        print("Noise impact (depolarising p1, p2=2p1):")
        for level, energy in noise_results:
            print(f"  p={level:.4f} -> expected energy {energy:.4f}")
    except RuntimeError as e:
        print(f"Skipping noise evaluation: {e}")

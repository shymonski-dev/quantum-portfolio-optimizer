"""Runtime checks for phase two benchmark helpers."""

import numpy as np

from quantum_portfolio_optimizer.benchmarks import phase2
from quantum_portfolio_optimizer.core.optimizer_interface import DifferentialEvolutionConfig


def test_make_optimizer_config_returns_differential_evolution_config():
    config = phase2.make_optimizer_config(4)

    assert isinstance(config, DifferentialEvolutionConfig)
    assert len(config.bounds) == 4


def test_run_phase2_benchmark_uses_default_cache_dir(monkeypatch):
    """Default cache path logic should run without import errors."""

    class DummyProblem:
        num_variables = 1
        linear = np.array([0.0])
        quadratic = np.array([[0.0]])
        offset = 0.0

    class DummyBuilder:
        def build(self):
            return DummyProblem()

    benchmark_result = phase2.BenchmarkResult(
        ansatz_name="real_amplitudes",
        reps=1,
        entanglement="reverse_linear",
        optimal_value=0.0,
        evaluations=1,
        converged=True,
        history=[0.0],
        optimal_parameters=np.array([0.0]),
        ansatz_options={"reps": 1},
    )

    monkeypatch.setattr(phase2, "build_phase2_qubo", lambda: DummyBuilder())
    monkeypatch.setattr(phase2, "get_provider", lambda _config: (object(), object()))
    monkeypatch.setattr(
        phase2,
        "benchmark_ansatze",
        lambda *_args, **_kwargs: [benchmark_result],
    )
    monkeypatch.setattr(phase2, "summarize_results", lambda _results: None)
    monkeypatch.setattr(phase2, "analyse_initialisations", lambda **_kwargs: {})
    monkeypatch.setattr(
        phase2,
        "evaluate_noise_levels",
        lambda *_args, **_kwargs: [],
    )

    phase2.run_phase2_benchmark(use_cache=False)

"""Variational Quantum Eigensolver tailored for the portfolio QUBO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from ..simulation.local_backend import get_default_estimator
from .ansatz_library import analyse_circuit, get_ansatz, initialise_parameters
from .optimizer_interface import DifferentialEvolutionConfig, run_differential_evolution
from .qubo_formulation import QUBOProblem


@dataclass
class VQEResult:
    optimal_parameters: np.ndarray
    optimal_value: float
    history: List[float]
    best_history: List[float]
    num_evaluations: int
    ansatz_report: dict
    converged: bool
    optimizer_message: str


class PortfolioVQESolver:
    """Simple VQE loop using Qiskit's Estimator primitive."""

    def __init__(
        self,
        estimator: Optional[object] = None,
        ansatz_name: str = "real_amplitudes",
        ansatz_options: Optional[dict] = None,
        init_strategy: str = "zeros",
        parameter_bounds: Optional[float] = 2 * np.pi,  # Extended range from paper
        optimizer_factory: Optional[
            Callable[[Callable[[np.ndarray], float]], DifferentialEvolutionConfig]
        ] = None,
        optimizer_config: Optional[DifferentialEvolutionConfig] = None,
        seed: Optional[int] = None,
        shots: Optional[int] = None,
        progress_callback: Optional[Callable[[int, float, float], None]] = None,
    ) -> None:
        self.estimator = estimator or get_default_estimator(shots=shots)
        self.ansatz_name = ansatz_name
        self.ansatz_options = ansatz_options or {}
        self.init_strategy = init_strategy
        self.parameter_bounds = parameter_bounds
        self.optimizer_factory = optimizer_factory
        self.optimizer_config = optimizer_config
        self.seed = seed
        self.progress_callback = progress_callback

    def solve(self, qubo: QUBOProblem) -> VQEResult:
        num_qubits = qubo.num_variables
        if num_qubits == 0:
            raise ValueError("QUBO must contain at least one variable.")

        ansatz = get_ansatz(self.ansatz_name, num_qubits=num_qubits, **self.ansatz_options)
        initial_point = initialise_parameters(ansatz, strategy=self.init_strategy, seed=self.seed)
        observable = qubo.to_pauli()
        history: List[float] = []
        best_history: List[float] = []
        best_so_far = np.inf

        def energy_evaluation(parameters: np.ndarray) -> float:
            nonlocal best_so_far
            circuit = ansatz.assign_parameters(parameters)
            try:
                job = self.estimator.run(circuits=[circuit], observables=[observable])
            except TypeError:
                job = self.estimator.run([(circuit, observable)])
            result = job.result()
            energy = self._extract_energy(result)
            history.append(energy)
            if energy < best_so_far:
                best_so_far = energy
            best_history.append(best_so_far)
            if self.progress_callback:
                self.progress_callback(len(history), energy, best_so_far)
            return energy

        if isinstance(self.parameter_bounds, tuple):
            bounds = [self.parameter_bounds] * ansatz.num_parameters
        elif isinstance(self.parameter_bounds, (int, float)):
            bounds = [(-abs(self.parameter_bounds), abs(self.parameter_bounds))] * ansatz.num_parameters
        elif self.parameter_bounds is None:
            bounds = [(-2*np.pi, 2*np.pi)] * ansatz.num_parameters  # Extended range from paper
        else:
            bounds = list(self.parameter_bounds)

        config = self.optimizer_config
        if config is None:
            config = DifferentialEvolutionConfig(bounds=bounds, seed=self.seed)
        else:
            config = DifferentialEvolutionConfig(
                bounds=bounds,
                strategy=config.strategy,
                maxiter=config.maxiter,
                popsize=config.popsize,
                tol=config.tol,
                mutation=config.mutation,
                recombination=config.recombination,
                seed=config.seed or self.seed,
                polish=config.polish,
                convergence_threshold=config.convergence_threshold,
                convergence_window=config.convergence_window,
                adaptive_mutation=config.adaptive_mutation,
                adaptive_recombination=config.adaptive_recombination,
            )

        _ = initial_point  # kept for future warm-start schemes

        result = run_differential_evolution(energy_evaluation, config=config, num_qubits=num_qubits)
        optimal_parameters = np.asarray(result.x, dtype=float)
        optimal_value = float(result.fun)

        report = analyse_circuit(ansatz, name=self.ansatz_name).__dict__
        return VQEResult(
            optimal_parameters=optimal_parameters,
            optimal_value=optimal_value,
            history=history,
            best_history=best_history,
            num_evaluations=len(history),
            ansatz_report=report,
            converged=bool(getattr(result, "success", False)),
            optimizer_message=str(getattr(result, "message", "")),
        )

    @staticmethod
    def _extract_energy(result: object) -> float:
        """Normalise estimator result objects across V1/V2 primitive interfaces."""
        if hasattr(result, "values"):
            values = getattr(result, "values")
            if isinstance(values, (list, tuple, np.ndarray)):
                return float(np.real(values[0]))
            return float(np.real(values))

        if hasattr(result, "__getitem__"):
            try:
                first = result[0]
                data = getattr(first, "data", None)
                if data is not None and hasattr(data, "evs"):
                    evs = getattr(data, "evs")
                    return float(np.real(np.asarray(evs).item()))
            except Exception as exc:  # pragma: no cover - defensive fallback.
                raise ValueError("Unsupported estimator result format") from exc

        raise ValueError("Estimator result object is not recognised.")

"""Classical optimiser integration layer."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import OptimizeResult, differential_evolution


@dataclass
class DifferentialEvolutionConfig:
    bounds: Sequence[Tuple[float, float]]
    strategy: str = 'best2bin'  # Strategy from paper
    maxiter: int = 20
    popsize: int = 10
    tol: float = 1e-6
    mutation: Tuple[float, float] = (0, 0.25)  # From paper
    recombination: float = 0.4  # From paper
    seed: Optional[int] = None
    polish: bool = False
    convergence_threshold: float = 0.025  # 2.5% threshold from paper
    convergence_window: int = 10  # Check over 10 generations
    adaptive_mutation: bool = True
    adaptive_recombination: bool = True
    x0: Optional[Sequence[float]] = None  # Initial point for warm start


def run_differential_evolution(
    objective: Callable[[np.ndarray], float],
    config: DifferentialEvolutionConfig,
    num_qubits: Optional[int] = None,
) -> OptimizeResult:
    """Execute SciPy's differential evolution with research-aligned configuration."""

    # Adjust popsize based on number of qubits if provided
    popsize = config.popsize
    if num_qubits is not None and config.strategy == 'best2bin':
        min_popsize = int(0.8 * num_qubits)
        popsize = max(popsize, min_popsize)

    best_value = [np.inf]
    history = deque(maxlen=config.convergence_window if config.convergence_window > 0 else None)

    def wrapped_objective(params: np.ndarray) -> float:
        value = objective(params)
        if value < best_value[0]:
            best_value[0] = value
        return value

    def callback(xk: np.ndarray, convergence_metric: float) -> bool:
        if history.maxlen is None or config.convergence_threshold <= 0:
            return False
        history.append(best_value[0])
        if len(history) == history.maxlen:
            baseline = history[0]
            latest = history[-1]
            denominator = max(abs(baseline), 1e-12)
            relative_change = abs(latest - baseline) / denominator
            if relative_change < config.convergence_threshold:
                return True
        return False

    result = differential_evolution(
        wrapped_objective,
        bounds=config.bounds,
        strategy=config.strategy,  # Use strategy from config
        maxiter=config.maxiter,
        popsize=popsize,  # Use adjusted popsize
        tol=config.tol,
        mutation=config.mutation,
        recombination=config.recombination,
        seed=config.seed,
        polish=config.polish,
        updating="immediate",
        callback=callback if history.maxlen else None,
        x0=config.x0,  # Warm start initial point
    )
    return result

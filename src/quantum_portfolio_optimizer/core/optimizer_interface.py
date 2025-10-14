"""Classical optimiser integration layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import OptimizeResult, differential_evolution


@dataclass
class DifferentialEvolutionConfig:
    bounds: Sequence[Tuple[float, float]]
    maxiter: int = 20
    popsize: int = 10
    tol: float = 1e-6
    mutation: Tuple[float, float] = (0.5, 1.0)
    recombination: float = 0.7
    seed: Optional[int] = None
    polish: bool = False


def run_differential_evolution(
    objective: Callable[[np.ndarray], float],
    config: DifferentialEvolutionConfig,
) -> OptimizeResult:
    """Execute SciPy's differential evolution with a consistent configuration."""
    result = differential_evolution(
        objective,
        bounds=config.bounds,
        maxiter=config.maxiter,
        popsize=config.popsize,
        tol=config.tol,
        mutation=config.mutation,
        recombination=config.recombination,
        seed=config.seed,
        polish=config.polish,
        updating="immediate",
    )
    return result

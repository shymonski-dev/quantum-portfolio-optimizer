"""Variational Quantum Eigensolver tailored for the portfolio QUBO."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..simulation.zne import fold_circuit, zne_extrapolate
from .ansatz_library import analyse_circuit, get_ansatz, initialise_parameters
from .optimizer_interface import DifferentialEvolutionConfig, run_differential_evolution
from .qubo_formulation import QUBOProblem

logger = logging.getLogger(__name__)


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
    # New fields for binary solution extraction (optional for backward compatibility)
    best_bitstring: Optional[str] = None
    measurement_counts: Optional[Dict[str, int]] = None


class PortfolioVQESolver:
    """Simple VQE loop using Qiskit's Estimator primitive.

    Optionally accepts a Sampler primitive for extracting binary solutions
    from the optimized circuit via measurement sampling.
    """

    def __init__(
        self,
        estimator: object,
        sampler: Optional[object] = None,  # NEW: for binary solution extraction
        ansatz_name: str = "real_amplitudes",
        ansatz_options: Optional[dict] = None,
        init_strategy: str = "zeros",
        parameter_bounds: Optional[float] = 2 * np.pi,  # Extended range from paper
        optimizer_factory: Optional[
            Callable[[Callable[[np.ndarray], float]], DifferentialEvolutionConfig]
        ] = None,
        optimizer_config: Optional[DifferentialEvolutionConfig] = None,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int, float, float], None]] = None,
        extraction_shots: int = 1024,  # NEW: shots for solution extraction
        zne_config: Optional[dict] = None,
    ) -> None:
        if estimator is None:
            raise ValueError("Estimator cannot be None.")
        self.estimator = estimator
        self.sampler = sampler  # NEW
        self.ansatz_name = ansatz_name
        self.ansatz_options = ansatz_options or {}
        self.init_strategy = init_strategy
        self.parameter_bounds = parameter_bounds
        self.optimizer_factory = optimizer_factory
        self.optimizer_config = optimizer_config
        self.seed = seed
        self.progress_callback = progress_callback
        self.extraction_shots = extraction_shots  # NEW
        self.zne_config = zne_config or {}

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

            if self.zne_config.get("zne_gate_folding", False):
                noise_factors = self.zne_config.get("zne_noise_factors", [1, 3, 5])
                extrapolator = self.zne_config.get("zne_extrapolator", "linear")
                zne_values = []
                for nf in noise_factors:
                    bound_for_zne = ansatz.assign_parameters(parameters)
                    folded = fold_circuit(bound_for_zne, nf)
                    try:
                        zne_job = self.estimator.run([(folded, observable)])
                        zne_result = zne_job.result()
                        zne_values.append(self._extract_energy(zne_result))
                    except Exception as e:
                        logger.warning("ZNE evaluation at nf=%d failed: %s", nf, e)
                        zne_values.append(float("inf"))
                energy = zne_extrapolate(noise_factors, zne_values, extrapolator)
            else:
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
                x0=config.x0,
            )

        _ = initial_point  # kept for future warm-start schemes

        result = run_differential_evolution(energy_evaluation, config=config, num_qubits=num_qubits)
        optimal_parameters = np.asarray(result.x, dtype=float)
        optimal_value = float(result.fun)

        # Extract binary solution if sampler is available
        best_bitstring = None
        measurement_counts = None
        if self.sampler is not None:
            try:
                best_bitstring, measurement_counts = self.extract_solution(
                    ansatz, optimal_parameters, num_qubits, shots=self.extraction_shots
                )
                logger.info(f"Extracted best bitstring: {best_bitstring}")
            except Exception as e:
                logger.warning(f"Solution extraction failed: {e}")

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
            best_bitstring=best_bitstring,
            measurement_counts=measurement_counts,
        )

    @staticmethod
    def _extract_energy(result: object) -> float:
        """Normalise estimator result objects across V1/V2 primitive interfaces."""
        if hasattr(result, "values"):
            values = getattr(result, "values")
            if isinstance(values, (list, tuple, np.ndarray)):
                if len(values) == 0:
                    raise ValueError("Estimator returned empty values array")
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

    def extract_solution(
        self,
        ansatz,
        parameters: np.ndarray,
        num_qubits: int,
        shots: int = 1024,
    ) -> Tuple[str, Dict[str, int]]:
        """Sample the optimized circuit to extract binary solution bitstrings.

        Args:
            ansatz: The parameterized quantum circuit (ansatz).
            parameters: Optimal parameters from VQE optimization.
            num_qubits: Number of qubits (for padding bitstrings).
            shots: Number of measurement shots.

        Returns:
            Tuple of (best_bitstring, counts_dict).
        """
        if self.sampler is None:
            raise ValueError("Sampler is required for solution extraction.")

        # Bind parameters and add measurements
        bound_circuit = ansatz.assign_parameters(parameters)
        measured_circuit = bound_circuit.copy()
        measured_circuit.measure_all()

        # Run sampler - handle both V1 and V2 interfaces
        try:
            # V2 interface: sampler.run([(circuit, params)])
            job = self.sampler.run([(measured_circuit, [])])
        except TypeError:
            # V1 interface: sampler.run(circuits, shots=shots)
            job = self.sampler.run([measured_circuit], shots=shots)

        result = job.result()
        counts = self._extract_counts(result, num_qubits, shots=shots)

        # Find the most frequent bitstring
        best_bitstring = max(counts, key=counts.get)
        return best_bitstring, counts

    @staticmethod
    def _extract_counts(
        result: object,
        num_qubits: int,
        shots: Optional[int] = None,
    ) -> Dict[str, int]:
        """Extract measurement counts from sampler result, handling V1/V2 interfaces.

        Args:
            result: Sampler result object.
            num_qubits: Number of qubits for padding bitstrings.

        Returns:
            Dictionary mapping bitstrings to counts.
        """
        # V2 interface: result[0].data.<key>.get_counts()
        if hasattr(result, "__getitem__"):
            try:
                first = result[0]
                if hasattr(first, "data"):
                    data = first.data
                    # Get the first data key (measurement register name)
                    key = next(iter(data.keys())) if hasattr(data, "keys") else None
                    if key is not None:
                        bitarray = getattr(data, key)
                        if hasattr(bitarray, "get_counts"):
                            raw_counts = bitarray.get_counts()
                            # Pad bitstrings to num_qubits length
                            counts = {}
                            for bitstring, count in raw_counts.items():
                                padded = bitstring.zfill(num_qubits)
                                counts[padded] = count
                            return counts
            except (IndexError, StopIteration, AttributeError):
                pass

        # V1 interface: result.quasi_dists or result.metadata
        if hasattr(result, "quasi_dists"):
            quasi_dists = result.quasi_dists
            if quasi_dists and len(quasi_dists) > 0:
                dist = quasi_dists[0]
                total_shots = shots
                if total_shots is None:
                    metadata = getattr(result, "metadata", None)
                    if isinstance(metadata, (list, tuple)) and metadata:
                        first_meta = metadata[0]
                        if isinstance(first_meta, dict):
                            total_shots = first_meta.get("shots")
                    elif isinstance(metadata, dict):
                        total_shots = metadata.get("shots")
                if total_shots is None:
                    total_shots = 1024
                total_shots = max(int(total_shots), 1)
                counts = {}
                for state, prob in dist.items():
                    bitstring = format(state, f"0{num_qubits}b")
                    counts[bitstring] = int(prob * total_shots)
                return counts

        raise ValueError("Unsupported sampler result format")

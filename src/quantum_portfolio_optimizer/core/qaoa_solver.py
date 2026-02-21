"""Quantum Approximate Optimization Algorithm (QAOA) for portfolio optimization."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from ..simulation.zne import fold_circuit, zne_extrapolate
from ..simulation.partitioning import (
    run_partitioned_vqe_step,
)  # Can be used for QAOA energy too
from .optimizer_interface import DifferentialEvolutionConfig, run_differential_evolution
from .qubo_formulation import QUBOProblem

logger = logging.getLogger(__name__)


@dataclass
class QAOAResult:
    """Result from QAOA optimization."""

    optimal_parameters: np.ndarray
    optimal_value: float
    best_bitstring: str
    measurement_counts: Dict[str, int]
    history: List[float]
    best_history: List[float]
    num_evaluations: int
    layers: int
    converged: bool
    optimizer_message: str
    circuit_report: Dict[str, float]


class PortfolioQAOASolver:
    """QAOA solver for portfolio optimization problems.

    QAOA alternates between cost and mixer layers to find approximate
    solutions to combinatorial optimization problems.

    Note: QAOA circuits are generally deeper than VQE circuits, making
    VQE more suitable for NISQ devices with limited coherence times.
    """

    def __init__(
        self,
        sampler: object,
        estimator: Optional[object] = None,
        layers: int = 1,
        optimizer_config: Optional[DifferentialEvolutionConfig] = None,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int, float, float], None]] = None,
        shots: int = 1024,
        zne_config: Optional[dict] = None,
        cvar_alpha: float = 1.0,
        mixer_type: str = "x",
        num_assets: Optional[int] = None,
        use_partitioning: bool = False,  # 2026 Modular Hardware support
    ) -> None:
        """Initialize QAOA solver.

        Args:
            sampler: Qiskit Sampler primitive for measurements (required).
            estimator: Optional Estimator primitive for energy evaluation.
            layers: Number of QAOA layers (p). Default 1.
            optimizer_config: Configuration for the classical optimizer.
            seed: Random seed for reproducibility.
            progress_callback: Optional callback(iteration, energy, best_energy).
            shots: Number of measurement shots.
            zne_config: Optional ZNE configuration dict with keys:
                zne_gate_folding (bool), zne_noise_factors (list), zne_extrapolator (str).
            cvar_alpha: CVaR tail fraction in (0, 1]. 1.0 = standard expectation value.
            mixer_type: Mixer type, 'x' for standard X-mixer or 'xy' for XY-mixer.
            num_assets: Number of assets to select (required when mixer_type='xy').
            use_partitioning: Enable circuit partitioning for modular hardware.
        """
        if sampler is None:
            raise ValueError("Sampler is required for QAOA.")
        if estimator is not None:
            warnings.warn(
                "estimator parameter is deprecated and will be removed in a future version. "
                "QAOA uses Sampler primitive only.",
                DeprecationWarning,
                stacklevel=2,
            )
        if not (0 < cvar_alpha <= 1.0):
            raise ValueError(f"cvar_alpha must be in (0, 1], got {cvar_alpha}")
        if mixer_type not in ("x", "xy"):
            raise ValueError(f"mixer_type must be 'x' or 'xy', got '{mixer_type}'")
        if mixer_type == "xy" and num_assets is None:
            raise ValueError("num_assets must be set when mixer_type='xy'")
        self.sampler = sampler
        self.estimator = estimator
        self.layers = layers
        self.optimizer_config = optimizer_config
        self.seed = seed
        self.progress_callback = progress_callback
        self.shots = shots
        self.zne_config = zne_config or {}
        self.cvar_alpha = cvar_alpha
        self.mixer_type = mixer_type
        self.num_assets = num_assets
        self.use_partitioning = use_partitioning

    def _build_qaoa_circuit(
        self, qubo: QUBOProblem
    ) -> Tuple[QuantumCircuit, List[Parameter]]:
        """Build parameterized QAOA circuit for the given QUBO.

        QAOA circuit structure:
        1. Initial state: |+>^n (H gates on all qubits)
        2. For each layer p:
           a. Cost layer: exp(-i * gamma_p * H_C)
           b. Mixer layer: exp(-i * beta_p * H_M)

        Args:
            qubo: The QUBO problem to solve.

        Returns:
            Tuple of (circuit, parameters list).
        """
        num_qubits = qubo.num_variables
        qc = QuantumCircuit(num_qubits)

        # Create parameters: gamma and beta for each layer
        gammas = [Parameter(f"gamma_{p}") for p in range(self.layers)]
        betas = [Parameter(f"beta_{p}") for p in range(self.layers)]

        # Validate num_assets for XY mixer now that n_qubits is known
        if self.mixer_type == "xy":
            if self.num_assets < 1 or self.num_assets >= num_qubits:
                raise ValueError(
                    f"num_assets ({self.num_assets}) must satisfy 1 <= num_assets < n_qubits ({num_qubits})"
                )

        # Initial state
        if self.mixer_type == "xy":
            self._prepare_dicke_state(qc, num_qubits, self.num_assets)
        else:
            qc.h(range(num_qubits))

        # Convert QUBO to Ising form once and reuse across layers.
        h, j_matrix, _ = qubo.to_ising()

        # QAOA layers
        for p in range(self.layers):
            # Cost layer: exp(-i * gamma * H_C)
            self._apply_cost_layer(qc, h, j_matrix, gammas[p])
            # Mixer layer: exp(-i * beta * H_M)
            if self.mixer_type == "xy":
                self._apply_xy_mixer_layer(qc, betas[p], num_qubits)
            else:
                self._apply_mixer_layer(qc, betas[p])

        # Collect all parameters in order [gamma_0, beta_0, gamma_1, beta_1, ...]
        parameters = []
        for p in range(self.layers):
            parameters.append(gammas[p])
            parameters.append(betas[p])

        return qc, parameters

    def _apply_cost_layer(
        self, qc: QuantumCircuit, h: np.ndarray, j_matrix: np.ndarray, gamma: Parameter
    ) -> None:
        """Apply the cost layer exp(-i * gamma * H_C).

        For Ising form: H_C = sum_i h_i * Z_i + sum_{i<j} J_{ij} * Z_i * Z_j

        Args:
            qc: Quantum circuit to modify.
            h: Ising linear coefficients (from qubo.to_ising()).
            j_matrix: Ising quadratic coefficients (from qubo.to_ising()).
            gamma: Parameter for this layer.
        """
        num_qubits = qc.num_qubits

        # Linear terms: h_i * Z_i -> RZ(2 * gamma * h_i) on qubit i
        for i in range(num_qubits):
            coeff = h[i]
            if abs(coeff) > 1e-10:
                qc.rz(2 * gamma * coeff, i)

        # Quadratic terms: J_{ij} * Z_i * Z_j -> CNOT-RZ-CNOT sequence
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                coeff = j_matrix[i, j]
                if abs(coeff) > 1e-10:
                    # ZZ interaction: exp(-i * gamma * J * Z_i * Z_j)
                    # Implemented as: CNOT(i,j) - RZ(2*gamma*J, j) - CNOT(i,j)
                    qc.cx(i, j)
                    qc.rz(2 * gamma * coeff, j)
                    qc.cx(i, j)

    def _apply_mixer_layer(self, qc: QuantumCircuit, beta: Parameter) -> None:
        """Apply the mixer layer exp(-i * beta * H_M).

        Standard mixer: H_M = sum_i X_i -> RX(2*beta) on each qubit.

        Args:
            qc: Quantum circuit to modify.
            beta: Parameter for this layer.
        """
        for i in range(qc.num_qubits):
            qc.rx(2 * beta, i)

    def _compute_objective(self, energies: np.ndarray, counts: np.ndarray) -> float:
        """Compute the QAOA objective over measured bitstring energies.

        If cvar_alpha == 1.0: standard weighted expectation value (backward-compatible).
        If cvar_alpha < 1.0: CVaR — average over the worst (highest-energy) alpha-fraction
        of measured shots. This gives 4.5x faster convergence (Barkoutsos et al. 2020).

        For a minimization QUBO, "worst" = highest energy (largest cost).

        Args:
            energies: Array of QUBO energies for each unique bitstring.
            counts:   Array of shot counts corresponding to each bitstring.

        Returns:
            Scalar objective value.
        """
        if self.cvar_alpha == 1.0:
            return float(np.average(energies, weights=counts))

        # Sort descending by energy (worst/highest first)
        sort_idx = np.argsort(energies)[::-1]
        sorted_e = energies[sort_idx]
        sorted_c = counts[sort_idx]

        total_shots = int(counts.sum())
        cutoff = int(np.ceil(self.cvar_alpha * total_shots))

        # Accumulate from the worst end until we have 'cutoff' shots
        cum = np.cumsum(sorted_c)
        mask = cum <= cutoff
        if not mask.any():
            mask[0] = True  # always include the single worst bitstring

        tail_energies = sorted_e[mask]
        tail_counts = sorted_c[mask]
        return float(np.average(tail_energies, weights=tail_counts))

    def _prepare_dicke_state(self, qc: QuantumCircuit, n_qubits: int, k: int) -> None:
        """Prepare the Dicke state |D_n^k>: uniform superposition of all n-qubit
        computational basis states with exactly k ones.

        This is the correct initial state for XY-mixer QAOA, ensuring only
        valid k-asset portfolios are explored (constraint-preserving).

        For n <= 8: uses StatePreparation with explicit statevector (exact).
        For n > 8:  TODO: implement recursive Bartschi-Eidenbenz construction
                          (O(n*k) gates) for hardware efficiency.

        Args:
            qc: QuantumCircuit to apply preparation to (in-place).
            n_qubits: Total number of qubits n.
            k: Hamming weight (number of selected assets).
        """
        from qiskit.circuit.library import StatePreparation

        # Build statevector: 1/sqrt(C(n,k)) for all weight-k basis states, 0 elsewhere
        n_states = 2**n_qubits
        sv = np.zeros(n_states, dtype=complex)

        amplitude = 1.0 / np.sqrt(comb(n_qubits, k))
        for bits in combinations(range(n_qubits), k):
            # Convert qubit indices to integer index (LSB = qubit 0)
            idx = sum(1 << b for b in bits)
            sv[idx] = amplitude

        # StatePreparation is Qiskit 2.x preferred API (replaces deprecated initialize)
        prep = StatePreparation(sv)
        qc.append(prep, range(n_qubits))

    def _apply_xy_mixer_layer(
        self, qc: QuantumCircuit, beta: "Parameter", n_qubits: int
    ) -> None:
        """Apply XY mixer: H_B = sum_{i<j} (X_i X_j + Y_i Y_j).

        The XY mixer preserves Hamming weight, ensuring only valid k-asset
        portfolios are explored after Dicke state initialisation.

        Uses XXPlusYYGate(theta=2*beta) which implements exp(-i*beta*(XX+YY)/2)
        on each qubit pair.

        Args:
            qc: QuantumCircuit (modified in-place).
            beta: Qiskit Parameter for the mixer angle.
            n_qubits: Number of qubits.
        """
        try:
            from qiskit.circuit.library import XXPlusYYGate

            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    # XXPlusYYGate(theta, beta_phase)
                    # theta=2*beta follows the XY-QAOA literature convention
                    qc.append(XXPlusYYGate(2 * beta, 0), [i, j])
        except ImportError:
            # Fallback: manual CNOT+RY+RZ decomposition of XX+YY rotation
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    # XX+YY rotation by angle 2*beta on qubits (i, j)
                    qc.cx(j, i)
                    qc.ry(-beta, i)
                    qc.rz(-np.pi / 2, j)
                    qc.cx(i, j)
                    qc.ry(beta, i)
                    qc.cx(j, i)
                    qc.rz(np.pi / 2, j)

    def _run_circuit_and_get_objective(
        self, bound_circuit: QuantumCircuit, qubo: QUBOProblem
    ) -> float:
        """Run sampler on bound_circuit, extract counts, compute objective (CVaR or expectation)."""
        measured = bound_circuit.copy()
        measured.measure_all()

        # ISA Transpilation for hardware (2026 Requirement)
        if hasattr(self.sampler, "backend"):
            try:
                from qiskit.transpiler.preset_passmanagers import (
                    generate_preset_pass_manager,
                )

                backend = self.sampler.backend
                pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
                isa_measured = pm.run(measured)
                job = self.sampler.run([(isa_measured, [])])
            except Exception as e:
                logger.warning("ISA transpilation failed for QAOA: %s", e)
                job = self.sampler.run([(measured, [])])
        else:
            try:
                job = self.sampler.run([(measured, [])])
            except TypeError:
                job = self.sampler.run([measured], shots=self.shots)

        result = job.result()
        counts = self._extract_counts(result, bound_circuit.num_qubits, self.shots)

        if not counts:
            return float("inf")

        # Build energy and count arrays for _compute_objective (supports CVaR)
        bitstrings = list(counts.keys())
        energies = np.array(
            [
                self._evaluate_qubo_energy(
                    np.array([int(b) for b in bs[::-1]], dtype=float), qubo
                )
                for bs in bitstrings
            ]
        )
        counts_array = np.array([counts[bs] for bs in bitstrings], dtype=float)

        return self._compute_objective(energies, counts_array)

    def solve(self, qubo: QUBOProblem) -> QAOAResult:
        """Solve the QUBO problem using QAOA.

        Args:
            qubo: The QUBO problem to solve.

        Returns:
            QAOAResult with optimal parameters, bitstring, and counts.
        """
        num_qubits = qubo.num_variables
        if num_qubits == 0:
            raise ValueError("QUBO must contain at least one variable.")

        # Build QAOA circuit
        qaoa_circuit, parameters = self._build_qaoa_circuit(qubo)
        num_params = len(parameters)  # 2 * layers (gamma and beta per layer)
        circuit_report = {
            "name": "QAOA",
            "num_qubits": num_qubits,
            "num_parameters": num_params,
            "depth": qaoa_circuit.depth(),
            "size": qaoa_circuit.size(),
            "layers": self.layers,
        }

        logger.info(
            f"QAOA circuit: {num_qubits} qubits, {self.layers} layers, "
            f"{num_params} parameters, depth={qaoa_circuit.depth()}"
        )

        history: List[float] = []
        best_history: List[float] = []
        best_so_far = np.inf

        def objective(param_values: np.ndarray) -> float:
            """Evaluate expected energy for given parameters."""
            nonlocal best_so_far

            # Bind parameters and measure
            bound_circuit = qaoa_circuit.assign_parameters(
                dict(zip(parameters, param_values))
            )

            # 2026 Logic: Skip local folding if hardware-native ZNE is active
            use_local_zne = self.zne_config.get("zne_gate_folding", False)
            has_native_resilience = False
            try:
                # Check for IBM SamplerV2 resilience level
                options = getattr(self.sampler, "options", None)
                if (
                    options and getattr(options, "resilience_level", 0) >= 1
                ):  # Sampler V2 has resilience
                    has_native_resilience = True
            except Exception:
                pass

            if self.use_partitioning:
                partitions = qubo.metadata.get("partitions", [])
                if partitions:
                    try:
                        # For QAOA, we can use the same partitioned energy evaluation logic
                        energy = run_partitioned_vqe_step(
                            self.sampler, bound_circuit, qubo.to_pauli(), partitions
                        )
                    except Exception as e:
                        logger.error("Partitioned QAOA evaluation failed: %s", e)
                        energy = float("inf")
                else:
                    logger.warning("No partitions for QAOA. Falling back.")
                    energy = self._run_circuit_and_get_objective(bound_circuit, qubo)
            elif use_local_zne and not has_native_resilience:
                noise_factors = self.zne_config.get("zne_noise_factors", [1, 3, 5])
                extrapolator = self.zne_config.get("zne_extrapolator", "linear")
                values = [
                    self._run_circuit_and_get_objective(
                        fold_circuit(bound_circuit, nf), qubo
                    )
                    for nf in noise_factors
                ]
                energy = zne_extrapolate(noise_factors, values, extrapolator)
            else:
                energy = self._run_circuit_and_get_objective(bound_circuit, qubo)

            history.append(energy)
            if energy < best_so_far:
                best_so_far = energy
            best_history.append(best_so_far)

            if self.progress_callback:
                self.progress_callback(len(history), energy, best_so_far)

            return energy

        # Set up optimizer bounds
        # gamma typically in [0, 2*pi], beta in [0, pi]
        bounds = []
        for _ in range(self.layers):
            bounds.append((0, 2 * np.pi))  # gamma
            bounds.append((0, np.pi))  # beta

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
                x0=config.x0,
            )

        # Run optimization
        result = run_differential_evolution(
            objective, config=config, num_qubits=num_qubits
        )
        optimal_parameters = np.asarray(result.x, dtype=float)
        optimal_value = float(result.fun)

        # Final measurement with optimal parameters
        bound_circuit = qaoa_circuit.assign_parameters(
            dict(zip(parameters, optimal_parameters))
        )
        measured_circuit = bound_circuit.copy()
        measured_circuit.measure_all()

        try:
            job = self.sampler.run([(measured_circuit, [])])
        except TypeError:
            job = self.sampler.run([measured_circuit], shots=self.shots)

        final_result = job.result()
        measurement_counts = self._extract_counts(
            final_result, num_qubits, shots=self.shots
        )
        best_bitstring = max(measurement_counts, key=measurement_counts.get)

        logger.info(f"QAOA optimization complete. Best bitstring: {best_bitstring}")

        return QAOAResult(
            optimal_parameters=optimal_parameters,
            optimal_value=optimal_value,
            best_bitstring=best_bitstring,
            measurement_counts=measurement_counts,
            history=history,
            best_history=best_history,
            num_evaluations=len(history),
            layers=self.layers,
            converged=bool(getattr(result, "success", False)),
            optimizer_message=str(getattr(result, "message", "")),
            circuit_report=circuit_report,
        )

    @staticmethod
    def _evaluate_qubo_energy(bits: np.ndarray, qubo: QUBOProblem) -> float:
        """Evaluate the QUBO energy for a given bitstring.

        E(x) = offset + x^T * linear + x^T * quadratic * x

        Args:
            bits: Binary array representing the solution.
            qubo: The QUBO problem.

        Returns:
            Energy value for this bitstring.
        """
        return float(
            qubo.offset
            + np.dot(qubo.linear, bits)
            + np.dot(bits, np.dot(qubo.quadratic, bits))
        )

    @staticmethod
    def _extract_counts(
        result: object,
        num_qubits: int,
        shots: Optional[int] = None,
    ) -> Dict[str, int]:
        """Extract measurement counts from sampler result.

        Handles both V1 and V2 Qiskit primitive interfaces.

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
                    key = next(iter(data.keys())) if hasattr(data, "keys") else None
                    if key is not None:
                        bitarray = getattr(data, key)
                        if hasattr(bitarray, "get_counts"):
                            raw_counts = bitarray.get_counts()
                            counts = {}
                            for bitstring, count in raw_counts.items():
                                padded = bitstring.zfill(num_qubits)
                                counts[padded] = count
                            return counts
            except (IndexError, StopIteration, AttributeError):
                pass

        # V1 interface: result.quasi_dists
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


def get_qaoa_circuit_depth(num_qubits: int, layers: int) -> int:
    """Estimate QAOA circuit depth for given problem size.

    This helps users understand the hardware requirements.

    Args:
        num_qubits: Number of qubits (assets).
        layers: Number of QAOA layers (p).

    Returns:
        Estimated circuit depth.
    """
    # H layer: 1
    # Per QAOA layer:
    #   Cost layer: ~n*(n-1)/2 * 3 for ZZ terms (CNOT-RZ-CNOT) + n for Z terms
    #   Mixer layer: n RX gates
    # Very rough estimate
    cost_layer_depth = num_qubits + 3 * (num_qubits * (num_qubits - 1) // 2)
    mixer_layer_depth = 1  # RX gates are parallel
    qaoa_layer_depth = cost_layer_depth + mixer_layer_depth
    return 1 + layers * qaoa_layer_depth

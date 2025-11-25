"""Quantum Approximate Optimization Algorithm (QAOA) for portfolio optimization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from ..simulation.provider import get_provider
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
        """
        if sampler is None:
            raise ValueError("Sampler is required for QAOA.")
        self.sampler = sampler
        self.estimator = estimator
        self.layers = layers
        self.optimizer_config = optimizer_config
        self.seed = seed
        self.progress_callback = progress_callback
        self.shots = shots

    def _build_qaoa_circuit(self, qubo: QUBOProblem) -> Tuple[QuantumCircuit, List[Parameter]]:
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

        # Initial state: uniform superposition
        qc.h(range(num_qubits))

        # QAOA layers
        for p in range(self.layers):
            # Cost layer: exp(-i * gamma * H_C)
            self._apply_cost_layer(qc, qubo, gammas[p])
            # Mixer layer: exp(-i * beta * H_M)
            self._apply_mixer_layer(qc, betas[p])

        # Collect all parameters in order [gamma_0, beta_0, gamma_1, beta_1, ...]
        parameters = []
        for p in range(self.layers):
            parameters.append(gammas[p])
            parameters.append(betas[p])

        return qc, parameters

    def _apply_cost_layer(
        self, qc: QuantumCircuit, qubo: QUBOProblem, gamma: Parameter
    ) -> None:
        """Apply the cost layer exp(-i * gamma * H_C).

        For QUBO: H_C = sum_i h_i * Z_i + sum_{i<j} J_{ij} * Z_i * Z_j

        The transformation from binary x_i to Pauli Z_i is: x_i = (1 - Z_i) / 2

        Args:
            qc: Quantum circuit to modify.
            qubo: QUBO problem with linear and quadratic terms.
            gamma: Parameter for this layer.
        """
        num_qubits = qubo.num_variables

        # Linear terms: h_i * Z_i -> RZ(2 * gamma * h_i) on qubit i
        for i in range(num_qubits):
            coeff = qubo.linear[i]
            if abs(coeff) > 1e-10:
                # Convert from QUBO coefficient to Ising coefficient
                # Z_i = 1 - 2*x_i, so x_i = (1-Z_i)/2
                # Linear term: h_i * x_i -> h_i * (1-Z_i)/2 = h_i/2 - (h_i/2)*Z_i
                qc.rz(gamma * coeff, i)

        # Quadratic terms: J_{ij} * Z_i * Z_j -> CNOT-RZ-CNOT sequence
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                coeff = qubo.quadratic[i, j] + qubo.quadratic[j, i]
                if abs(coeff) > 1e-10:
                    # ZZ interaction: exp(-i * gamma * J * Z_i * Z_j)
                    # Implemented as: CNOT(i,j) - RZ(2*gamma*J, j) - CNOT(i,j)
                    qc.cx(i, j)
                    qc.rz(gamma * coeff, j)
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
            measured_circuit = bound_circuit.copy()
            measured_circuit.measure_all()

            # Get measurement counts
            try:
                job = self.sampler.run([(measured_circuit, [])])
            except TypeError:
                job = self.sampler.run([measured_circuit], shots=self.shots)

            result = job.result()
            counts = self._extract_counts(result, num_qubits)

            # Calculate expected energy from measurement counts
            total_shots = sum(counts.values())
            energy = 0.0
            for bitstring, count in counts.items():
                # Convert bitstring to binary array (LSB first)
                bits = np.array([int(b) for b in bitstring[::-1]], dtype=float)
                # Evaluate QUBO energy for this bitstring
                bit_energy = self._evaluate_qubo_energy(bits, qubo)
                energy += (count / total_shots) * bit_energy

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
            bounds.append((0, np.pi))      # beta

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
            )

        # Run optimization
        result = run_differential_evolution(objective, config=config, num_qubits=num_qubits)
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
        measurement_counts = self._extract_counts(final_result, num_qubits)
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
    def _extract_counts(result: object, num_qubits: int) -> Dict[str, int]:
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
                counts = {}
                for state, prob in dist.items():
                    bitstring = format(state, f"0{num_qubits}b")
                    counts[bitstring] = int(prob * 1024)
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

"""QUBO construction utilities for the local quantum portfolio optimiser.

Phase 1 targets small problems (2–3 assets, up to 2 time steps) while keeping
the API extensible for later phases.  The formulation discretises allocation
levels using binary resolution qubits and maps the resulting QUBO onto an
Ising Hamiltonian compatible with Qiskit primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from qiskit.quantum_info import SparsePauliOp


@dataclass
class QUBOProblem:
    """Container for a binary quadratic optimisation problem."""

    linear: np.ndarray
    quadratic: np.ndarray
    offset: float
    variable_order: List[Tuple[int, int, int]]  # (asset, time_step, bit)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.linear = np.asarray(self.linear, dtype=float)
        self.quadratic = np.asarray(self.quadratic, dtype=float)
        if self.linear.ndim != 1:
            raise ValueError("linear coefficients must be a 1-D array")
        if self.quadratic.shape != (self.linear.size, self.linear.size):
            raise ValueError("quadratic matrix must be square with size matching linear terms")
        if not np.allclose(self.quadratic, self.quadratic.T, atol=1e-9):
            raise ValueError("quadratic matrix must be symmetric")

    @property
    def num_variables(self) -> int:
        return self.linear.size

    def to_ising(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return (h, J, offset) for the equivalent Ising Hamiltonian."""
        num_vars = self.num_variables
        h = np.zeros(num_vars, dtype=float)
        j = np.zeros((num_vars, num_vars), dtype=float)
        constant = float(self.offset)

        # Convert QUBO of the form E(x) = offset + sum a_i x_i + sum_{i<j} b_ij x_i x_j
        # to Ising variables z_i \in {-1, 1} via x_i = (1 - z_i) / 2.
        for i in range(num_vars):
            constant += 0.5 * self.linear[i]
            h[i] += -0.5 * self.linear[i]

        for i in range(num_vars):
            for j_idx in range(i + 1, num_vars):
                coeff = self.quadratic[i, j_idx]
                if abs(coeff) < 1e-12:
                    continue
                constant += 0.25 * coeff
                h[i] += -0.25 * coeff
                h[j_idx] += -0.25 * coeff
                j[i, j_idx] += 0.25 * coeff
                j[j_idx, i] += 0.25 * coeff

            # Diagonal terms can exist if users inject them manually; treat as linear.
            diag = self.quadratic[i, i]
            if abs(diag) > 1e-12:
                constant += 0.25 * diag
                h[i] += -0.25 * diag

        return h, j, constant

    def to_pauli(self, atol: float = 1e-12) -> SparsePauliOp:
        """Convert the Ising Hamiltonian into a Qiskit SparsePauliOp."""
        h, j, constant = self.to_ising()
        num_qubits = self.num_variables
        labels: List[str] = []
        coeffs: List[complex] = []

        if abs(constant) > atol:
            labels.append("I" * num_qubits)
            coeffs.append(complex(constant))

        for idx, coeff in enumerate(h):
            if abs(coeff) <= atol:
                continue
            label = ["I"] * num_qubits
            label[num_qubits - idx - 1] = "Z"  # Qiskit uses little-endian ordering.
            labels.append("".join(label))
            coeffs.append(complex(coeff))

        for i in range(num_qubits):
            for j_idx in range(i + 1, num_qubits):
                coeff = j[i, j_idx]
                if abs(coeff) <= atol:
                    continue
                label = ["I"] * num_qubits
                label[num_qubits - i - 1] = "Z"
                label[num_qubits - j_idx - 1] = "Z"
                labels.append("".join(label))
                coeffs.append(complex(coeff))

        if not labels:
            labels = ["I" * num_qubits]
            coeffs = [0.0]

        return SparsePauliOp(labels, coeffs)


class PortfolioQUBO:
    """Builds a QUBO representation for discretised portfolio optimisation."""

    def __init__(
        self,
        expected_returns: Iterable[Iterable[float]] | Iterable[float],
        covariance: Iterable[Iterable[float]],
        budget: float,
        risk_aversion: float = 1000,  # γ parameter from paper
        transaction_cost: float = 0.01,  # ν parameter from paper
        time_steps: int = 1,
        resolution_qubits: int = 1,
        max_investment: float = 1.0,
        penalty_strength: float = 1.0,  # ρ parameter from paper
        enforce_budget: bool = True,
        time_step_budgets: Optional[Iterable[float]] = None,
        time_budget_penalty: Optional[float] = None,
        asset_max_allocation: Optional[Iterable[float]] = None,
        asset_penalty_strength: Optional[float] = None,
    ) -> None:
        self.risk_aversion = float(risk_aversion)
        self.transaction_cost = float(transaction_cost)
        self.time_steps = int(time_steps)
        self.resolution_qubits = int(resolution_qubits)
        self.max_investment = float(max_investment)
        self.penalty_strength = float(penalty_strength)
        self.time_budget_penalty = float(time_budget_penalty) if time_budget_penalty is not None else self.penalty_strength
        self.asset_penalty_strength = (
            float(asset_penalty_strength) if asset_penalty_strength is not None else self.penalty_strength
        )
        self.enforce_budget = bool(enforce_budget)
        self.budget = float(budget)

        if self.time_steps < 1:
            raise ValueError("time_steps must be >= 1")
        if self.resolution_qubits < 1:
            raise ValueError("resolution_qubits must be >= 1 in Phase 1")
        if self.max_investment <= 0:
            raise ValueError("max_investment must be positive")
        if self.budget <= 0:
            raise ValueError("budget must be positive")

        self.expected_returns = self._normalise_returns(expected_returns, self.time_steps)
        self.covariance = np.asarray(covariance, dtype=float)
        if self.covariance.ndim != 2 or self.covariance.shape[0] != self.covariance.shape[1]:
            raise ValueError("covariance must be a square matrix")

        self.num_assets = self.covariance.shape[0]
        if self.expected_returns.shape[1] != self.num_assets:
            raise ValueError("expected_returns second dimension must match covariance")

        if not np.allclose(self.covariance, self.covariance.T, atol=1e-8):
            raise ValueError("covariance matrix must be symmetric")

        self._bit_weights = np.array([2**b for b in range(self.resolution_qubits)], dtype=float)
        self._levels = 2**self.resolution_qubits
        self._allocation_scale = self.max_investment / max(1, self.time_steps)
        self._normalisation = (
            self._allocation_scale if self._levels == 1 else self._allocation_scale / (self._levels - 1)
        )

        self._variable_order: List[Tuple[int, int, int]] = []
        for asset in range(self.num_assets):
            for t_step in range(self.time_steps):
                for bit in range(self.resolution_qubits):
                    self._variable_order.append((asset, t_step, bit))

        # Pre-compute index groupings for constraints and energy terms.
        self._asset_indices: Dict[int, List[int]] = {asset: [] for asset in range(self.num_assets)}
        self._time_indices: Dict[int, List[int]] = {t_step: [] for t_step in range(self.time_steps)}
        self._asset_time_indices: Dict[Tuple[int, int], List[int]] = {}
        self._variable_weights = np.zeros(len(self._variable_order), dtype=float)
        for idx, (asset, t_step, bit) in enumerate(self._variable_order):
            self._asset_indices[asset].append(idx)
            self._time_indices[t_step].append(idx)
            self._asset_time_indices.setdefault((asset, t_step), []).append(idx)
            self._variable_weights[idx] = self._normalisation * self._bit_weights[bit]

        self.time_step_budgets: Optional[np.ndarray]
        if time_step_budgets is not None:
            time_budget_array = np.asarray(list(time_step_budgets), dtype=float)
            if time_budget_array.shape != (self.time_steps,):
                raise ValueError("time_step_budgets must have length equal to time_steps")
            if np.any(time_budget_array <= 0):
                raise ValueError("time_step_budgets entries must be positive")
            if np.any(time_budget_array > self.max_investment):
                raise ValueError("time_step_budgets cannot exceed max_investment")
            self.time_step_budgets = time_budget_array
        else:
            self.time_step_budgets = None

        self.asset_max_allocation: Optional[np.ndarray]
        if asset_max_allocation is not None:
            asset_array = np.asarray(list(asset_max_allocation), dtype=float)
            if asset_array.shape != (self.num_assets,):
                raise ValueError("asset_max_allocation must match number of assets")
            if np.any(asset_array <= 0):
                raise ValueError("asset_max_allocation entries must be positive")
            if np.any(asset_array > self.max_investment):
                raise ValueError("asset_max_allocation cannot exceed max_investment")
            self.asset_max_allocation = asset_array
        else:
            self.asset_max_allocation = None

    @staticmethod
    def _normalise_returns(
        expected_returns: Iterable[Iterable[float]] | Iterable[float], time_steps: int
    ) -> np.ndarray:
        returns = np.asarray(expected_returns, dtype=float)
        if returns.ndim == 1:
            returns = np.tile(returns, (time_steps, 1))
        elif returns.ndim == 2:
            if returns.shape[0] != time_steps:
                raise ValueError("When providing a matrix, expected_returns.shape[0] must equal time_steps")
        else:
            raise ValueError("expected_returns must be 1-D or 2-D")
        return returns

    @staticmethod
    def calculate_lambda_parameter(K: int, K_prime: int) -> float:
        """Calculate λ for transaction cost approximation as per paper Eq. 14.

        Parameters:
            K: Number of discretization levels (2^Nr)
            K_prime: Modified discretization parameter

        Returns:
            λ parameter for transaction cost calculation
        """
        import math
        return math.sqrt(3) * K / (2 * K_prime)

    def _weights_for_asset(self, asset: int) -> List[Tuple[int, float]]:
        """Allocation contributions for all variables that belong to a given asset."""
        return [(idx, self._variable_weights[idx]) for idx in self._asset_indices.get(asset, [])]

    def _apply_penalty(
        self,
        linear: np.ndarray,
        quadratic: np.ndarray,
        offset: float,
        indices: List[int],
        limit: float,
        penalty: float,
    ) -> float:
        if not indices:
            return offset
        weights = self._variable_weights[indices]
        offset += penalty * limit**2
        for i, idx_i in enumerate(indices):
            weight_i = weights[i]
            linear[idx_i] += penalty * weight_i**2 - 2 * penalty * limit * weight_i
            for j in range(i + 1, len(indices)):
                idx_j = indices[j]
                weight_j = weights[j]
                coeff = 2 * penalty * weight_i * weight_j
                quadratic[idx_i, idx_j] += coeff
                quadratic[idx_j, idx_i] += coeff
        return offset

    def build(self) -> QUBOProblem:
        """Construct the QUBO problem."""
        num_vars = len(self._variable_order)
        linear = np.zeros(num_vars, dtype=float)
        quadratic = np.zeros((num_vars, num_vars), dtype=float)
        offset = 0.0

        # Expected return term (maximisation -> negative contribution).
        for idx, (asset, t_step, _bit) in enumerate(self._variable_order):
            mu = self.expected_returns[t_step, asset]
            linear[idx] += -mu * self._variable_weights[idx]

        # Risk term aggregated per asset (x^T Sigma x).
        for asset_i in range(self.num_assets):
            weights_i = self._weights_for_asset(asset_i)
            for asset_j in range(asset_i, self.num_assets):
                weights_j = self._weights_for_asset(asset_j)
                cov_coeff = self.covariance[asset_i, asset_j] * self.risk_aversion
                if abs(cov_coeff) <= 1e-12:
                    continue
                for idx_i, weight_i in weights_i:
                    pair_iter = weights_j
                    if asset_i == asset_j:
                        pair_iter = [(idx_j, weight_j) for idx_j, weight_j in weights_j if idx_j >= idx_i]
                    for idx_j, weight_j in pair_iter:
                        coeff = cov_coeff * weight_i * weight_j
                        if idx_i == idx_j:
                            linear[idx_i] += coeff
                        else:
                            quadratic[idx_i, idx_j] += coeff
                            quadratic[idx_j, idx_i] += coeff

        # Budget constraint enforcement via quadratic penalty.
        if self.enforce_budget:
            offset = self._apply_penalty(
                linear,
                quadratic,
                offset,
                list(range(num_vars)),
                self.budget,
                self.penalty_strength,
            )

        if self.time_step_budgets is not None:
            for t_step, limit in enumerate(self.time_step_budgets):
                indices = self._time_indices[t_step]
                offset = self._apply_penalty(linear, quadratic, offset, indices, limit, self.time_budget_penalty)

        if self.asset_max_allocation is not None:
            for asset, limit in enumerate(self.asset_max_allocation):
                indices = self._asset_indices[asset]
                offset = self._apply_penalty(linear, quadratic, offset, indices, limit, self.asset_penalty_strength)

        # Transaction cost penalty encourages smooth allocation across time.
        if self.transaction_cost > 0 and self.time_steps > 1:
            lambda_factor = self.calculate_lambda_parameter(self._levels, max(1, self._levels - 1))
            lambda_tc = self.transaction_cost * lambda_factor
            for asset in range(self.num_assets):
                prev_vars: List[Tuple[int, float]] = []
                for t_step in range(self.time_steps):
                    indices = self._asset_time_indices.get((asset, t_step), [])
                    curr_vars = [(idx, self._variable_weights[idx]) for idx in indices]
                    if prev_vars:
                        # A^2 term for previous allocations.
                        for idx_a, (var_a, weight_a) in enumerate(prev_vars):
                            linear[var_a] += lambda_tc * weight_a**2
                            for idx_b in range(idx_a + 1, len(prev_vars)):
                                var_b, weight_b = prev_vars[idx_b]
                                coeff = 2 * lambda_tc * weight_a * weight_b
                                quadratic[var_a, var_b] += coeff
                                quadratic[var_b, var_a] += coeff

                        # B^2 term for current allocations.
                        for idx_a, (var_a, weight_a) in enumerate(curr_vars):
                            linear[var_a] += lambda_tc * weight_a**2
                            for idx_b in range(idx_a + 1, len(curr_vars)):
                                var_b, weight_b = curr_vars[idx_b]
                                coeff = 2 * lambda_tc * weight_a * weight_b
                                quadratic[var_a, var_b] += coeff
                                quadratic[var_b, var_a] += coeff

                        # -2AB cross term.
                        for var_prev, weight_prev in prev_vars:
                            for var_curr, weight_curr in curr_vars:
                                coeff = -2 * lambda_tc * weight_prev * weight_curr
                                quadratic[var_prev, var_curr] += coeff
                                quadratic[var_curr, var_prev] += coeff
                    prev_vars = curr_vars

        quadratic = (quadratic + quadratic.T) / 2.0  # Ensure symmetry.

        metadata: Dict[str, Any] = {
            "num_assets": self.num_assets,
            "time_steps": self.time_steps,
            "resolution_qubits": self.resolution_qubits,
            "max_investment": self.max_investment,
            "budget": self.budget,
            "risk_aversion": self.risk_aversion,
            "transaction_cost": self.transaction_cost,
            "allocation_scale": self._allocation_scale,
            "normalisation": self._normalisation,
            "bit_weights": self._bit_weights.tolist(),
            "budget_penalty": self.penalty_strength,
            "time_budget_penalty": self.time_budget_penalty,
            "asset_penalty_strength": self.asset_penalty_strength,
        }
        if self.time_step_budgets is not None:
            metadata["time_step_budgets"] = self.time_step_budgets.tolist()
        if self.asset_max_allocation is not None:
            metadata["asset_max_allocation"] = self.asset_max_allocation.tolist()

        return QUBOProblem(
            linear=linear,
            quadratic=quadratic,
            offset=offset,
            variable_order=self._variable_order,
            metadata=metadata,
        )

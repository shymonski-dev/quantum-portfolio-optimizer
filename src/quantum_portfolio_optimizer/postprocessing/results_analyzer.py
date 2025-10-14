"""Utilities for interpreting VQE measurement outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ..core.qubo_formulation import QUBOProblem


@dataclass
class CandidateSolution:
    bitstring: str
    energy: float
    allocations: np.ndarray


class ResultAnalyzer:
    def __init__(self, qubo: QUBOProblem) -> None:
        self.qubo = qubo

    def bitstring_to_allocations(self, bitstring: str) -> np.ndarray:
        if len(bitstring) != self.qubo.num_variables:
            raise ValueError("Bitstring length must match QUBO variable count.")
        bits = np.array(list(bitstring[::-1]), dtype=int)
        resolution = int(self.qubo.metadata.get("resolution_qubits", 1))
        time_steps = int(self.qubo.metadata.get("time_steps", 1))

        num_assets = int(self.qubo.metadata.get("num_assets", 0))
        if num_assets <= 0:
            num_assets = max((asset for asset, _, _ in self.qubo.variable_order), default=-1) + 1

        normalisation = float(self.qubo.metadata.get("normalisation", 1.0))
        bit_weights = self.qubo.metadata.get("bit_weights")
        if bit_weights is None:
            bit_weights = [2**b for b in range(resolution)]
        bit_weights = np.asarray(bit_weights, dtype=float)

        allocations = np.zeros((time_steps, num_assets), dtype=float)
        for idx, (asset, t_step, bit) in enumerate(self.qubo.variable_order):
            if asset >= num_assets or t_step >= time_steps:
                continue
            weight = normalisation * bit_weights[bit]
            allocations[t_step, asset] += bits[idx] * weight

        if time_steps == 1:
            return allocations[0]
        return allocations

    def evaluate_bitstrings(self, bitstrings: List[str]) -> List[CandidateSolution]:
        hamiltonian = self.qubo.to_pauli()
        energies = []
        for bitstring in bitstrings:
            bits = np.array(list(bitstring[::-1]), dtype=int)
            energy = float(bits @ self.qubo.quadratic @ bits + self.qubo.linear @ bits + self.qubo.offset)
            allocations = self.bitstring_to_allocations(bitstring)
            energies.append(CandidateSolution(bitstring=bitstring, energy=energy, allocations=allocations))
        return sorted(energies, key=lambda c: c.energy)

    def top_k(self, results: List[Tuple[str, float]], k: int = 3) -> List[Tuple[str, float]]:
        return sorted(results, key=lambda kv: kv[1])[:k]

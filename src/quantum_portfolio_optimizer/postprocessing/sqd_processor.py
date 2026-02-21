"""Simplified stochastic quasi-decoherence (SQD) post-processing."""

from __future__ import annotations

import numpy as np


class SimpleSQDProcessor:
    """Minimal noise-aware aggregation for measurement counts."""

    def __init__(self, smoothing: float = 1e-3) -> None:
        self.smoothing = smoothing

    def process_counts(self, counts: dict[str, int]) -> dict[str, float]:
        total = sum(counts.values()) + self.smoothing * len(counts)
        return {
            bitstring: (count + self.smoothing) / total
            for bitstring, count in counts.items()
        }

    def expectation_from_counts(
        self, counts: dict[str, int], objective: np.ndarray
    ) -> float:
        processed = self.process_counts(counts)
        value = 0.0
        for bitstring, probability in processed.items():
            bit_array = np.array(list(bitstring[::-1]), dtype=int)
            energy = float(bit_array @ objective @ bit_array)
            value += probability * energy
        return value

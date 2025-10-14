"""Performance tracking utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict


@dataclass
class PerformanceTracker:
    metrics: Dict[str, float] = field(default_factory=dict)

    def timed(self, name: str):
        """Context manager for timing code blocks."""

        class _Timer:
            def __enter__(_self):
                _self.start = perf_counter()
                return _self

            def __exit__(_self, exc_type, exc_val, exc_tb):
                duration = perf_counter() - _self.start
                self.metrics[name] = duration

        return _Timer()

    def record(self, name: str, value: float) -> None:
        self.metrics[name] = value

    def get(self, name: str, default: float = 0.0) -> float:
        return self.metrics.get(name, default)

"""Deterministic hashing utilities for JSON-serialisable payloads."""

from __future__ import annotations

import hashlib
import json
from typing import Any


class _CustomEncoder(json.JSONEncoder):
    def default(self, obj: Any):  # type: ignore[override]
        try:
            import numpy as np

            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        if isinstance(obj, set):
            return sorted(obj)
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)


def deterministic_hash(payload: Any) -> str:
    """Return a stable SHA256 hash for arbitrary nested payloads."""

    normalised = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), cls=_CustomEncoder
    )
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()

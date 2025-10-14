"""Simple JSON-based caching helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def read_json_cache(cache_dir: Path, key: str) -> Optional[Any]:
    path = _cache_path(cache_dir, key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def write_json_cache(cache_dir: Path, key: str, data: Any) -> Dict[str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, key)
    payload = dict(data)
    metadata = dict(payload.get("metadata", {}))
    metadata.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    metadata["source"] = str(path.resolve())
    payload["metadata"] = metadata
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return metadata

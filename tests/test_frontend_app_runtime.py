"""Runtime checks for the web route wiring."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd


def _load_frontend_module():
    root = Path(__file__).resolve().parents[1]
    app_path = root / "frontend" / "app.py"
    module_name = f"frontend_app_runtime_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, app_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sample_prices() -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=20, freq="D")
    return pd.DataFrame(
        {
            "AAPL": 100.0 + np.arange(20, dtype=float),
            "MSFT": 200.0 + np.arange(20, dtype=float),
        },
        index=idx,
    )


def test_optimize_rejects_non_json_body():
    module = _load_frontend_module()
    client = module.app.test_client()

    response = client.post("/optimize", data="invalid", content_type="text/plain")

    assert response.status_code == 400
    assert "Invalid request body" in response.get_json()["error"]


def test_optimize_ibm_request_does_not_mutate_environment(monkeypatch):
    module = _load_frontend_module()
    client = module.app.test_client()

    monkeypatch.setattr(
        module, "fetch_stock_data", lambda *_args, **_kwargs: _sample_prices()
    )

    captured = {}

    def fake_get_provider(config):
        captured["config"] = config
        raise RuntimeError("stop after provider config capture")

    monkeypatch.setattr(module, "get_provider", fake_get_provider)
    monkeypatch.setenv("QE_TOKEN", "original-token")
    monkeypatch.setenv("IBM_INSTANCE", "original-instance")
    monkeypatch.setenv("IBM_CLOUD_CRN", "original-crn")

    payload = {
        "tickers": "AAPL,MSFT",
        "start_date": "2023-01-01",
        "end_date": "2023-03-31",
        "backend": "ibm_quantum",
        "ibm_device": "ibm_test",
        "ibm_api_key": "request-token",
        "ibm_channel": "ibm_quantum",
        "algorithm": "vqe",
        "warm_start": False,
    }

    response = client.post("/optimize", json=payload)

    assert response.status_code == 500
    assert captured["config"]["token"] == "request-token"
    assert os.environ.get("QE_TOKEN") == "original-token"
    assert os.environ.get("IBM_INSTANCE") == "original-instance"
    assert os.environ.get("IBM_CLOUD_CRN") == "original-crn"


def test_optimize_passes_gate_folding_config_to_variational_solver(monkeypatch):
    module = _load_frontend_module()
    client = module.app.test_client()

    monkeypatch.setattr(
        module, "fetch_stock_data", lambda *_args, **_kwargs: _sample_prices()
    )
    monkeypatch.setattr(module, "get_provider", lambda _config: (object(), object()))

    captured = {}

    class FakeVQESolver:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs
            raise RuntimeError("stop after solver init capture")

    monkeypatch.setattr(module, "PortfolioVQESolver", FakeVQESolver)

    payload = {
        "tickers": "AAPL,MSFT",
        "start_date": "2023-01-01",
        "end_date": "2023-03-31",
        "backend": "local_simulator",
        "algorithm": "vqe",
        "warm_start": False,
        "zne_gate_folding": True,
        "zne_noise_factors": "1,3,5",
        "zne_extrapolator": "linear",
    }

    response = client.post("/optimize", json=payload)

    assert response.status_code == 500
    zne_config = captured["kwargs"]["zne_config"]
    assert zne_config["zne_gate_folding"] is True
    assert zne_config["zne_noise_factors"] == [1.0, 3.0, 5.0]
    assert zne_config["zne_extrapolator"] == "linear"


def test_template_script_avoids_inner_html_usage():
    root = Path(__file__).resolve().parents[1]
    template_path = root / "frontend" / "templates" / "index.html"
    content = template_path.read_text(encoding="utf-8")

    assert "innerHTML" not in content

"""Tests for command line execution flow."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from click.testing import CliRunner

import quantum_portfolio_optimizer.cli as cli_module


class _DummyQUBO:
    metadata = {"num_assets": 2}
    num_variables = 2

    def decode_bitstring(self, _bitstring: str):
        return {"allocation_per_asset": [0.7, 0.3]}


class _DummyQUBOBuilder:
    def __init__(self, **_kwargs):
        pass

    def build(self):
        return _DummyQUBO()


class _DummyVQEResult:
    best_bitstring = "10"
    optimal_value = -1.234
    num_evaluations = 7
    converged = True


class _DummyQAOAResult:
    best_bitstring = "01"
    optimal_value = -0.456
    num_evaluations = 5
    converged = False


class _DummyVQESolver:
    def __init__(self, **_kwargs):
        pass

    def solve(self, _qubo):
        return _DummyVQEResult()


class _DummyQAOASolver:
    def __init__(self, **_kwargs):
        pass

    def solve(self, _qubo):
        return _DummyQAOAResult()


def _sample_prices() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    return pd.DataFrame(
        {
            "AAPL": 100.0 + np.arange(20, dtype=float),
            "MSFT": 200.0 + np.arange(20, dtype=float),
        },
        index=idx,
    )


def _markowitz_result():
    return SimpleNamespace(
        expected_return=0.01,
        variance=0.0004,
        sharpe_ratio=0.5,
        success=True,
    )


def test_cli_run_vqe_json_output(monkeypatch):
    config = {
        "portfolio": {
            "tickers": ["AAPL", "MSFT"],
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
        },
        "algorithm": {
            "name": "vqe",
            "settings": {"ansatz": "real_amplitudes", "reps": 1},
        },
        "backend": {"name": "local_simulator", "shots": 128, "seed": 7},
        "optimizer": {"maxiter": 3, "popsize": 4, "seed": 7},
        "risk_model": {"parameters": {"risk_factor": 0.5}},
    }

    monkeypatch.setattr(cli_module, "load_config", lambda _path: config)
    monkeypatch.setattr(
        cli_module, "fetch_stock_data", lambda *_args, **_kwargs: _sample_prices()
    )
    monkeypatch.setattr(cli_module, "get_provider", lambda _cfg: (object(), object()))
    monkeypatch.setattr(cli_module, "PortfolioQUBO", _DummyQUBOBuilder)
    monkeypatch.setattr(cli_module, "PortfolioVQESolver", _DummyVQESolver)
    monkeypatch.setattr(
        cli_module,
        "get_ansatz",
        lambda *_args, **_kwargs: SimpleNamespace(num_parameters=3),
    )
    monkeypatch.setattr(
        cli_module, "markowitz_baseline", lambda **_kwargs: _markowitz_result()
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli, ["run", "--config", "dummy.yaml", "--json-output"]
    )

    assert result.exit_code == 0
    assert '"algorithm": "vqe"' in result.output
    assert '"objective_value": -1.234' in result.output
    assert '"best_bitstring": "10"' in result.output


def test_cli_run_qaoa_text_output(monkeypatch):
    config = {
        "portfolio": {
            "tickers": ["AAPL", "MSFT"],
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
        },
        "algorithm": {
            "name": "qaoa",
            "settings": {"layers": 1, "cvar_alpha": 0.5, "mixer_type": "x"},
        },
        "backend": {"name": "local_simulator", "shots": 128, "seed": 11},
        "optimizer": {"maxiter": 2, "popsize": 3, "seed": 11},
        "risk_model": {"parameters": {"risk_factor": 0.4}},
    }

    monkeypatch.setattr(cli_module, "load_config", lambda _path: config)
    monkeypatch.setattr(
        cli_module, "fetch_stock_data", lambda *_args, **_kwargs: _sample_prices()
    )
    monkeypatch.setattr(cli_module, "get_provider", lambda _cfg: (object(), object()))
    monkeypatch.setattr(cli_module, "PortfolioQUBO", _DummyQUBOBuilder)
    monkeypatch.setattr(cli_module, "PortfolioQAOASolver", _DummyQAOASolver)
    monkeypatch.setattr(
        cli_module, "markowitz_baseline", lambda **_kwargs: _markowitz_result()
    )

    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["run", "--config", "dummy.yaml"])

    assert result.exit_code == 0
    assert "Optimization complete." in result.output
    assert "Algorithm: qaoa" in result.output
    assert "Objective value: -0.456000" in result.output


def test_cli_run_writes_json_file(monkeypatch):
    config = {
        "portfolio": {
            "tickers": ["AAPL", "MSFT"],
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
        },
        "algorithm": {
            "name": "vqe",
            "settings": {"ansatz": "real_amplitudes", "reps": 1},
        },
        "backend": {"name": "local_simulator", "shots": 128, "seed": 7},
        "optimizer": {"maxiter": 3, "popsize": 4, "seed": 7},
        "risk_model": {"parameters": {"risk_factor": 0.5}},
    }

    monkeypatch.setattr(cli_module, "load_config", lambda _path: config)
    monkeypatch.setattr(
        cli_module, "fetch_stock_data", lambda *_args, **_kwargs: _sample_prices()
    )
    monkeypatch.setattr(cli_module, "get_provider", lambda _cfg: (object(), object()))
    monkeypatch.setattr(cli_module, "PortfolioQUBO", _DummyQUBOBuilder)
    monkeypatch.setattr(cli_module, "PortfolioVQESolver", _DummyVQESolver)
    monkeypatch.setattr(
        cli_module,
        "get_ansatz",
        lambda *_args, **_kwargs: SimpleNamespace(num_parameters=3),
    )
    monkeypatch.setattr(
        cli_module, "markowitz_baseline", lambda **_kwargs: _markowitz_result()
    )

    runner = CliRunner()
    with runner.isolated_filesystem():
        output_file = "result.json"
        result = runner.invoke(
            cli_module.cli,
            ["run", "--config", "dummy.yaml", "--json-file", output_file],
        )

        assert result.exit_code == 0
        payload = json.loads(Path(output_file).read_text(encoding="utf-8"))
        assert payload["algorithm"] == "vqe"
        assert payload["best_bitstring"] == "10"
        assert "Saved JSON result to:" in result.output

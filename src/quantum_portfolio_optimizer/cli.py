"""Command line entry points for portfolio optimization runs."""

from __future__ import annotations

import json
import pprint
from typing import Any, Dict

import click
import numpy as np

from .benchmarks.classical_baseline import markowitz_baseline
from .core import PortfolioQAOASolver, PortfolioQUBO, PortfolioVQESolver
from .core.ansatz_library import get_ansatz
from .core.optimizer_interface import DifferentialEvolutionConfig
from .data import fetch_stock_data
from .data.returns_calculator import calculate_logarithmic_returns
from .simulation import get_provider
from .utils import load_config


@click.group()
def cli() -> None:
    """Quantum portfolio optimizer command line interface."""


@cli.command()
@click.option("--config", "-c", default="config.yaml", help="Path to the configuration file.")
@click.option("--json-output", is_flag=True, help="Print full result as JSON.")
@click.option(
    "--json-file",
    type=click.Path(dir_okay=False, writable=True, path_type=str),
    default=None,
    help="Write full result payload to a JSON file.",
)
def run(config: str, json_output: bool, json_file: str | None) -> None:
    """Run a full portfolio optimization experiment from config."""
    click.echo(f"Loading configuration from: {config}")
    try:
        config_data = load_config(config)
        click.echo("Configuration loaded successfully:")
        pprint.pprint(config_data)

        click.echo("\nFetching stock data...")
        portfolio_config = config_data.get("portfolio", {})
        tickers = portfolio_config.get("tickers")
        start_date = portfolio_config.get("start_date")
        end_date = portfolio_config.get("end_date")

        if not all([tickers, start_date, end_date]):
            raise ValueError(
                "Tickers, start_date, and end_date must be defined in the portfolio config."
            )

        stock_data = fetch_stock_data(tickers, start_date, end_date)
        click.echo("Stock data fetched successfully:")
        click.echo(stock_data.head())

        click.echo("\nInitializing backend...")
        backend_config: Dict[str, Any] = dict(config_data.get("backend", {}))
        backend_name = backend_config.get("name", "local_simulator")
        if backend_name == "ibm_quantum":
            ibm_defaults = dict(config_data.get("ibm_quantum", {}))
            backend_config = {**ibm_defaults, **backend_config}
            backend_config["name"] = "ibm_quantum"

        estimator, sampler = get_provider(backend_config)
        click.echo(f"Backend '{backend_config.get('name')}' initialized successfully.")
        click.echo(f"Estimator: {estimator}")
        click.echo(f"Sampler: {sampler}")

        click.echo("\nPreparing optimization inputs...")
        log_returns = calculate_logarithmic_returns(stock_data)
        if log_returns.ndim == 1:
            log_returns = log_returns[:, None]
        mean_returns = np.mean(log_returns, axis=0)
        covariance = np.cov(log_returns, rowvar=False)
        if np.ndim(covariance) == 0:
            covariance = np.array([[float(covariance)]], dtype=float)
        covariance = np.asarray(covariance, dtype=float)

        algo_config = config_data.get("algorithm", {})
        algo_name = str(algo_config.get("name", "vqe")).lower()
        algo_settings = algo_config.get("settings", {}) or {}

        risk_config = config_data.get("risk_model", {}).get("parameters", {}) or {}
        risk_factor = float(risk_config.get("risk_factor", 0.5))
        risk_aversion = float(risk_config.get("risk_aversion", 100 + (1 - risk_factor) * 9900))
        risk_metric = str(risk_config.get("risk_metric", "variance"))
        cvar_confidence = float(risk_config.get("cvar_confidence", 0.95))
        esg_scores = risk_config.get("esg_scores") or None
        esg_weight = float(risk_config.get("esg_weight", 0.0))
        if esg_scores is not None and len(esg_scores) == 0:
            esg_scores = None

        resolution_qubits = int(algo_settings.get("resolution_qubits", 1))
        time_steps = int(algo_settings.get("time_steps", 1))
        max_investment = float(algo_settings.get("max_investment", 1.0))
        budget = float(algo_settings.get("budget", 1.0))
        penalty_strength = float(algo_settings.get("penalty_strength", 1000.0))
        transaction_cost = float(algo_settings.get("transaction_cost", 0.0))

        qubo_builder = PortfolioQUBO(
            expected_returns=mean_returns,
            covariance=covariance,
            budget=budget,
            risk_aversion=risk_aversion,
            transaction_cost=transaction_cost,
            time_steps=time_steps,
            resolution_qubits=resolution_qubits,
            max_investment=max_investment,
            penalty_strength=penalty_strength,
            risk_metric=risk_metric,
            cvar_confidence=cvar_confidence,
            esg_scores=np.array(esg_scores, dtype=float) if esg_scores is not None else None,
            esg_weight=esg_weight if esg_scores is not None else 0.0,
        )
        qubo = qubo_builder.build()
        click.echo(
            "QUBO ready: "
            f"{qubo.metadata.get('num_assets', len(tickers))} assets, "
            f"{qubo.num_variables} variables."
        )

        optimizer_raw = config_data.get("optimizer", {}) or {}
        maxiter = int(optimizer_raw.get("maxiter", 50))
        popsize = int(optimizer_raw.get("popsize", 10))
        seed = backend_config.get("seed")
        if seed is None:
            seed = optimizer_raw.get("seed")
        if seed is not None:
            seed = int(seed)

        mitigation_cfg = config_data.get("error_mitigation", {}) or {}
        solver_zne_config = None
        if mitigation_cfg.get("zne_gate_folding", False):
            solver_zne_config = {
                "zne_gate_folding": True,
                "zne_noise_factors": mitigation_cfg.get("zne_noise_factors", [1, 3, 5]),
                "zne_extrapolator": mitigation_cfg.get("zne_extrapolator", "linear"),
            }

        click.echo(f"\nRunning solver: {algo_name}")
        if algo_name == "qaoa":
            layers = int(algo_settings.get("layers", 1))
            cvar_alpha = float(algo_settings.get("cvar_alpha", 1.0))
            mixer_type = str(algo_settings.get("mixer_type", "x")).lower()
            num_assets_select = algo_settings.get("num_assets")
            if num_assets_select is not None:
                num_assets_select = int(num_assets_select)

            bounds = []
            for _ in range(layers):
                bounds.append((0, 2 * np.pi))
                bounds.append((0, np.pi))
            optimizer_config = DifferentialEvolutionConfig(
                bounds=bounds,
                maxiter=maxiter,
                popsize=popsize,
                seed=seed,
            )
            solver = PortfolioQAOASolver(
                sampler=sampler,
                layers=layers,
                optimizer_config=optimizer_config,
                seed=seed,
                shots=int(backend_config.get("shots", 1024) or 1024),
                zne_config=solver_zne_config,
                cvar_alpha=cvar_alpha,
                mixer_type=mixer_type,
                num_assets=num_assets_select,
            )
        else:
            ansatz_name = str(algo_settings.get("ansatz", "real_amplitudes"))
            ansatz_options: Dict[str, Any] = {}
            if "reps" in algo_settings:
                ansatz_options["reps"] = int(algo_settings["reps"])
            if "entanglement" in algo_settings:
                ansatz_options["entanglement"] = algo_settings["entanglement"]
            if "insert_barriers" in algo_settings:
                ansatz_options["insert_barriers"] = bool(algo_settings["insert_barriers"])

            ansatz_for_bounds = get_ansatz(
                ansatz_name,
                num_qubits=qubo.num_variables,
                **ansatz_options,
            )
            bounds = [(-2 * np.pi, 2 * np.pi)] * ansatz_for_bounds.num_parameters
            optimizer_config = DifferentialEvolutionConfig(
                bounds=bounds,
                maxiter=maxiter,
                popsize=popsize,
                seed=seed,
            )
            solver = PortfolioVQESolver(
                estimator=estimator,
                sampler=sampler,
                ansatz_name=ansatz_name,
                ansatz_options=ansatz_options,
                optimizer_config=optimizer_config,
                seed=seed,
                extraction_shots=int(backend_config.get("shots", 1024) or 1024),
                zne_config=solver_zne_config,
            )

        result = solver.solve(qubo)

        best_bitstring = getattr(result, "best_bitstring", None)
        if best_bitstring:
            decoded = qubo.decode_bitstring(best_bitstring)
            weights = np.array(decoded["allocation_per_asset"], dtype=float)
        else:
            weights = np.ones(len(tickers), dtype=float)
        weights = np.clip(weights, 0, None)
        if weights.sum() <= 0:
            weights = np.ones(len(tickers), dtype=float)
        weights = weights / weights.sum()

        expected_return = float(weights @ mean_returns) * 252
        variance = float(weights @ covariance @ weights) * 252
        volatility = float(np.sqrt(max(variance, 1e-12)))
        sharpe = expected_return / volatility if volatility > 0 else 0.0

        classical_risk = risk_aversion / 10000 if risk_aversion > 1 else risk_aversion
        baseline = markowitz_baseline(
            expected_returns=mean_returns,
            covariance=covariance,
            budget=budget,
            risk_aversion=classical_risk,
        )

        payload = {
            "algorithm": algo_name,
            "backend": backend_config.get("name"),
            "tickers": tickers,
            "weights_pct": {
                ticker: round(float(weight) * 100, 2) for ticker, weight in zip(tickers, weights)
            },
            "best_bitstring": best_bitstring,
            "objective_value": float(result.optimal_value),
            "evaluations": int(result.num_evaluations),
            "converged": bool(result.converged),
            "expected_return_pct": round(expected_return * 100, 2),
            "portfolio_risk_pct": round(volatility * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "baseline": {
                "expected_return_pct": round(baseline.expected_return * 252 * 100, 2),
                "portfolio_risk_pct": round(float(np.sqrt(max(baseline.variance * 252, 0))) * 100, 2),
                "sharpe_ratio": round(float(baseline.sharpe_ratio), 3),
                "success": bool(baseline.success),
            },
        }

        click.echo("\nOptimization complete.")
        if json_file:
            with open(json_file, "w", encoding="utf-8") as file_handle:
                json.dump(payload, file_handle, indent=2, sort_keys=True)
                file_handle.write("\n")
            click.echo(f"Saved JSON result to: {json_file}")

        if json_output:
            click.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            click.echo(f"Algorithm: {payload['algorithm']}")
            click.echo(f"Backend: {payload['backend']}")
            click.echo(f"Objective value: {payload['objective_value']:.6f}")
            click.echo(f"Evaluations: {payload['evaluations']}")
            click.echo(f"Converged: {payload['converged']}")
            click.echo(f"Expected return: {payload['expected_return_pct']:.2f}%")
            click.echo(f"Portfolio risk: {payload['portfolio_risk_pct']:.2f}%")
            click.echo(f"Sharpe ratio: {payload['sharpe_ratio']:.3f}")
            if best_bitstring:
                click.echo(f"Best bitstring: {best_bitstring}")
            click.echo("Weights:")
            for ticker, weight_pct in payload["weights_pct"].items():
                click.echo(f"  {ticker}: {weight_pct:.2f}%")

    except (FileNotFoundError, ValueError, NotImplementedError) as e:
        click.secho(str(e), fg="red")
    except Exception as e:  # pragma: no cover - user execution path
        click.secho(f"An error occurred: {e}", fg="red")


if __name__ == "__main__":
    cli()

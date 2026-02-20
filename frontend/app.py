"""Flask web frontend for Quantum Portfolio Optimizer."""

import os
import logging
from flask import Flask, render_template, request, jsonify
import numpy as np
from typing import Any

from quantum_portfolio_optimizer.data import fetch_stock_data
from quantum_portfolio_optimizer.data.returns_calculator import calculate_logarithmic_returns
from quantum_portfolio_optimizer.core.qubo_formulation import PortfolioQUBO
from quantum_portfolio_optimizer.core.vqe_solver import PortfolioVQESolver
from quantum_portfolio_optimizer.core.qaoa_solver import PortfolioQAOASolver
from quantum_portfolio_optimizer.core.optimizer_interface import DifferentialEvolutionConfig
from quantum_portfolio_optimizer.simulation.provider import get_provider
from quantum_portfolio_optimizer.benchmarks.classical_baseline import markowitz_baseline, mip_baseline
from quantum_portfolio_optimizer.postprocessing.quality_scorer import score_solution
from quantum_portfolio_optimizer.core.warm_start import (
    warm_start_vqe, warm_start_qaoa, WarmStartConfig
)
from quantum_portfolio_optimizer.core.ansatz_library import get_ansatz
from quantum_portfolio_optimizer.exceptions import (
    QuantumPortfolioError,
    DataError,
    QUBOError,
    OptimizationError,
    BackendError,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Error message guidance for common issues
ERROR_GUIDANCE = {
    'No data': "Could not retrieve stock data. Verify ticker symbols are correct and the date range contains trading days.",
    'yfinance': "Market data service error. Check your internet connection and try again.",
    'Invalid ticker': "One or more ticker symbols are invalid. Use standard formats like AAPL, MSFT, BRK.B.",
    'date': "Invalid date range. Ensure start date is before end date and end date is not in the future.",
    'QUBO': "Problem formulation failed. Try reducing the number of assets or adjusting risk parameters.",
    'IBM': "IBM Quantum connection issue. Verify your API key and CRN are correct.",
    'authentication': "Authentication failed. Check your IBM Quantum credentials.",
    'converge': "Optimization may need more iterations. Try increasing the Max Iterations setting.",
    'timeout': "Request timed out. Try using the local simulator or reducing problem size.",
    'Insufficient data': "Not enough historical data. Try extending the date range or using fewer assets.",
}


def classify_error(error_msg: str) -> dict:
    """Classify error and return user-friendly guidance."""
    error_str = str(error_msg).lower()
    for key, guidance in ERROR_GUIDANCE.items():
        if key.lower() in error_str:
            return {
                'message': guidance,
                'technical': str(error_msg),
                'type': key
            }
    return {
        'message': str(error_msg),
        'technical': str(error_msg),
        'type': 'unknown'
    }


@app.route('/')
def index():
    """Main page with input form."""
    return render_template('index.html')


@app.route('/optimize', methods=['POST'])
def optimize():
    """Run portfolio optimization."""
    try:
        # Get form data
        data = request.json
        tickers = [t.strip().upper() for t in data.get('tickers', 'AAPL,GOOG,MSFT,AMZN').split(',')]
        start_date = data.get('start_date', '2023-01-01')
        end_date = data.get('end_date', '2024-01-01')
        risk_factor = float(data.get('risk_factor', 0.5))
        ansatz_type = data.get('ansatz', 'real_amplitudes').lower().replace('realamplitudes', 'real_amplitudes').replace('efficientsu2', 'efficient_su2')
        reps = int(data.get('reps', 2))
        backend_type = data.get('backend', 'local_simulator')
        maxiter = int(data.get('maxiter', 50))
        algorithm = data.get('algorithm', 'vqe')
        qaoa_layers = int(data.get('qaoa_layers', 1))
        cvar_alpha = float(data.get('cvar_alpha', 1.0))
        cvar_alpha = max(0.05, min(1.0, cvar_alpha))
        mixer_type = data.get('mixer_type', 'x').lower()
        if mixer_type not in ('x', 'xy'):
            mixer_type = 'x'
        use_warm_start = data.get('warm_start', True)  # Default enabled
        resolution_qubits = int(data.get('resolution_qubits', 1))  # Default binary

        # ESG parameters
        esg_scores_raw = data.get('esg_scores', '')
        esg_scores: list | None = None
        if esg_scores_raw and str(esg_scores_raw).strip():
            try:
                parsed = [float(v.strip()) for v in str(esg_scores_raw).split(',') if v.strip()]
                if parsed:
                    esg_scores = parsed
            except ValueError:
                esg_scores = None
        esg_weight = float(data.get('esg_weight', 0.0))

        # MIP: number of assets to select (for integer baseline)
        num_assets_select_raw = data.get('num_assets_select', None)
        num_assets_select: int | None = None
        if num_assets_select_raw is not None and str(num_assets_select_raw).strip():
            try:
                num_assets_select = int(num_assets_select_raw)
            except (ValueError, TypeError):
                num_assets_select = None

        logger.info(f"Starting optimization: {tickers}, {start_date} to {end_date}")
        logger.info(f"Settings: algorithm={algorithm}, risk={risk_factor}, backend={backend_type}")
        if algorithm == 'vqe':
            logger.info(f"VQE Settings: ansatz={ansatz_type}, reps={reps}")
        else:
            logger.info(f"QAOA Settings: layers={qaoa_layers}")

        # Fetch market data
        stock_data = fetch_stock_data(tickers, start_date, end_date)

        if stock_data.empty:
            return jsonify({'error': 'Failed to fetch stock data. Check tickers and dates.'}), 400

        # Calculate returns using logarithmic returns (as per research paper)
        log_returns = calculate_logarithmic_returns(stock_data)
        if log_returns.ndim == 1:
            log_returns = log_returns[:, None]
        mean_returns = np.mean(log_returns, axis=0)
        cov_matrix = np.cov(log_returns, rowvar=False)
        num_assets = len(tickers)

        # Scale risk aversion based on slider (0-1) -> (100-10000)
        # Lower slider = lower risk tolerance = higher risk_aversion (penalize risk more)
        risk_aversion = 100 + (1 - risk_factor) * 9900

        # Validate ESG scores length if provided
        if esg_scores is not None and len(esg_scores) != num_assets:
            logger.warning(
                f"ESG scores length ({len(esg_scores)}) != num_assets ({num_assets}); ignoring ESG"
            )
            esg_scores = None
            esg_weight = 0.0

        # Formulate QUBO using PortfolioQUBO
        qubo_builder = PortfolioQUBO(
            expected_returns=mean_returns,
            covariance=cov_matrix,
            budget=1.0,  # Normalized budget
            risk_aversion=risk_aversion,
            transaction_cost=0.0,  # Single period, no transaction cost
            time_steps=1,
            resolution_qubits=resolution_qubits,  # Configurable allocation precision
            max_investment=1.0,
            penalty_strength=1000.0,
            enforce_budget=True,
            esg_scores=np.array(esg_scores) if esg_scores else None,
            esg_weight=esg_weight if esg_scores else 0.0,
        )
        qubo = qubo_builder.build()
        total_qubits = num_assets * resolution_qubits
        logger.info(f"QUBO built: {num_assets} assets x {resolution_qubits} resolution = {total_qubits} qubits")

        # Get backend
        if backend_type == "ibm_quantum":
            ibm_device = data.get('ibm_device', 'ibm_strasbourg')
            ibm_api_key = data.get('ibm_api_key', '')
            ibm_channel = data.get('ibm_channel', 'ibm_quantum')
            ibm_instance = (data.get('ibm_instance') or '').strip()
            ibm_crn = (data.get('ibm_crn') or '').strip()

            if not ibm_api_key:
                return jsonify({'error': 'IBM Quantum API Key is required'}), 400

            def parse_bool(value: Any, default: bool = False) -> bool:
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value.strip().lower() in {"true", "1", "yes", "on"}
                return default

            if ibm_channel == 'ibm_cloud':
                if not ibm_crn:
                    return jsonify({'error': 'Cloud Resource Name (CRN) is required for IBM Cloud access'}), 400
                instance_value = ibm_crn
            else:
                instance_value = ibm_instance or None

            # Set credentials as environment variables for the provider
            os.environ['QE_TOKEN'] = ibm_api_key
            if instance_value:
                os.environ['IBM_INSTANCE'] = instance_value
            else:
                os.environ.pop('IBM_INSTANCE', None)

            if ibm_channel == 'ibm_cloud':
                os.environ['IBM_CLOUD_CRN'] = instance_value or ''
            else:
                os.environ.pop('IBM_CLOUD_CRN', None)

            raw_noise = data.get('zne_noise_factors', '1,3,5')
            if isinstance(raw_noise, str):
                try:
                    zne_noise_factors = [float(val.strip()) for val in raw_noise.split(',') if val.strip()]
                except ValueError:
                    zne_noise_factors = [1.0, 3.0, 5.0]
            else:
                zne_noise_factors = [float(val) for val in raw_noise] or [1.0, 3.0, 5.0]
            if not zne_noise_factors:
                zne_noise_factors = [1.0, 3.0, 5.0]

            resilience_level = int(data.get('resilience_level', 1))
            resilience_level = max(0, min(resilience_level, 2))

            backend_config = {
                "name": "ibm_quantum",
                "device": ibm_device,
                "channel": ibm_channel,
                "instance": instance_value,
                "shots": 4096,
                "optimization_level": 3,
                "use_session": True,
                "error_mitigation": {
                    "resilience_level": resilience_level,
                    "dynamical_decoupling": parse_bool(data.get('dd_enabled', True), True),
                    "dd_sequence": data.get('dd_sequence', 'XpXm') or 'XpXm',
                    "twirling_enabled": parse_bool(data.get('twirling_enabled', True), True),
                    "zne_enabled": parse_bool(data.get('zne_enabled', False), False),
                    "zne_noise_factors": zne_noise_factors,
                    "zne_extrapolator": data.get('zne_extrapolator', 'exponential') or 'exponential',
                }
            }
            logger.info(f"Using IBM Quantum backend: {ibm_device} via {ibm_channel}")
        else:
            # Local simulator: support gate-folding ZNE when requested
            raw_noise = data.get('zne_noise_factors', '')
            local_zne_enabled = bool(data.get('zne_gate_folding', False))
            if isinstance(raw_noise, str) and raw_noise.strip():
                try:
                    local_zne_factors = [float(v.strip()) for v in raw_noise.split(',') if v.strip()]
                except ValueError:
                    local_zne_factors = [1.0, 3.0, 5.0]
            else:
                local_zne_factors = [1.0, 3.0, 5.0]
            backend_config = {
                "name": "local_simulator",
                "shots": 1024,
                "seed": 42,
                "zne_gate_folding": local_zne_enabled,
                "zne_noise_factors": local_zne_factors,
                "zne_extrapolator": data.get('zne_extrapolator', 'linear') or 'linear',
            }

        estimator, sampler = get_provider(backend_config)

        # Track progress
        progress_data = {'iteration': 0, 'best_energy': float('inf')}
        def progress_callback(iteration, energy, best_energy):
            progress_data['iteration'] = iteration
            progress_data['best_energy'] = best_energy

        # Prepare ansatz for bounds and warm start (VQE only)
        ansatz_for_solver = None
        if algorithm == 'vqe':
            ansatz_for_solver = get_ansatz(ansatz_type, num_qubits=total_qubits, reps=reps)

        # Compute warm start initial parameters if enabled
        warm_start_result = None
        initial_point = None
        if use_warm_start:
            try:
                warm_config = WarmStartConfig(seed=42)
                if algorithm == 'qaoa':
                    warm_start_result = warm_start_qaoa(
                        qubo_linear=qubo.linear,
                        qubo_quadratic=qubo.quadratic,
                        layers=qaoa_layers,
                        expected_returns=mean_returns,
                        covariance=cov_matrix,
                        config=warm_config,
                    )
                    initial_point = warm_start_result.initial_parameters.tolist()
                    logger.info(f"Warm start (QAOA): estimated {warm_start_result.estimated_improvement:.1f}x improvement")
                else:  # VQE
                    warm_start_result = warm_start_vqe(
                        expected_returns=mean_returns,
                        covariance=cov_matrix,
                        ansatz=ansatz_for_solver,
                        config=warm_config,
                    )
                    initial_point = warm_start_result.initial_parameters.tolist()
                    logger.info(f"Warm start (VQE): estimated {warm_start_result.estimated_improvement:.1f}x improvement")
            except Exception as e:
                logger.warning(f"Warm start failed, using random initialization: {e}")
                warm_start_result = None
                initial_point = None

        # Build ZNE config for solver (used by QAOA; VQE uses estimator-level ZNE)
        solver_zne_config: dict | None = None
        if backend_type == 'local_simulator' and backend_config.get('zne_gate_folding'):
            solver_zne_config = {
                'zne_gate_folding': True,
                'zne_noise_factors': backend_config['zne_noise_factors'],
                'zne_extrapolator': backend_config['zne_extrapolator'],
            }
        elif backend_type == 'ibm_quantum' and backend_config.get('error_mitigation', {}).get('zne_enabled'):
            em = backend_config['error_mitigation']
            solver_zne_config = {
                'zne_gate_folding': True,
                'zne_noise_factors': em.get('zne_noise_factors', [1.0, 3.0, 5.0]),
                'zne_extrapolator': em.get('zne_extrapolator', 'exponential'),
            }

        # Select and run the appropriate solver
        if algorithm == 'qaoa':
            # QAOA: Configure bounds for gamma and beta parameters
            bounds = []
            for _ in range(qaoa_layers):
                bounds.append((0, 2 * np.pi))  # gamma
                bounds.append((0, np.pi))       # beta

            optimizer_config = DifferentialEvolutionConfig(
                bounds=bounds,
                maxiter=maxiter,
                seed=42,
                x0=initial_point,  # Warm start initial point
            )

            # XY-mixer requires knowing how many assets to select (cardinality)
            xy_num_assets: int | None = None
            if mixer_type == 'xy':
                xy_num_assets = num_assets_select if num_assets_select else max(1, num_assets // 2)
                logger.info(f"XY-mixer: selecting {xy_num_assets} assets")

            solver = PortfolioQAOASolver(
                sampler=sampler,
                layers=qaoa_layers,
                optimizer_config=optimizer_config,
                seed=42,
                progress_callback=progress_callback,
                shots=1024,
                zne_config=solver_zne_config,
                cvar_alpha=cvar_alpha,
                mixer_type=mixer_type,
                num_assets=xy_num_assets,
            )
            logger.info(
                f"Running QAOA with {qaoa_layers} layers, "
                f"CVaR alpha={cvar_alpha}, mixer={mixer_type}"
            )
        else:
            # VQE: Configure optimizer bounds for ansatz parameters
            bounds = [(-2 * np.pi, 2 * np.pi)] * ansatz_for_solver.num_parameters
            optimizer_config = DifferentialEvolutionConfig(
                bounds=bounds,
                maxiter=maxiter,
                seed=42,
                x0=initial_point,  # Warm start initial point
            )

            solver = PortfolioVQESolver(
                estimator=estimator,
                sampler=sampler,
                ansatz_name=ansatz_type,
                ansatz_options={'reps': reps},
                optimizer_config=optimizer_config,
                seed=42,
                progress_callback=progress_callback,
                extraction_shots=1024,
            )
            logger.info(f"Running VQE with {ansatz_type} ansatz, reps={reps}")

        result = solver.solve(qubo)

        # Interpret solution using decode_bitstring for multi-qubit resolution
        if result.best_bitstring:
            decoded = qubo.decode_bitstring(result.best_bitstring)
            weights = decoded['allocation_per_asset']  # Already a list
            # Normalize to sum to 1
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
            else:
                weights = [1.0 / num_assets] * num_assets
        else:
            # Fallback to equal weight if no bitstring
            weights = [1.0 / num_assets] * num_assets

        # Identify selected assets (non-zero allocation)
        selected_assets = [tickers[i] for i, w in enumerate(weights) if w > 0.001]
        if not selected_assets:
            selected_assets = tickers
            weights = [1.0 / num_assets] * num_assets

        expected_return = sum(w * r for w, r in zip(weights, mean_returns)) * 252  # Annualized
        portfolio_variance = sum(
            weights[i] * weights[j] * cov_matrix[i][j]
            for i in range(len(weights))
            for j in range(len(weights))
        ) * 252
        portfolio_risk = portfolio_variance ** 0.5
        sharpe_ratio = expected_return / portfolio_risk if portfolio_risk > 0 else 0

        # Run classical Markowitz baseline for comparison
        classical_result = markowitz_baseline(
            expected_returns=mean_returns,
            covariance=cov_matrix,
            budget=1.0,
            risk_aversion=risk_aversion / 10000,  # Scale down to match classical range (0-1)
        )

        # Calculate annualized classical metrics
        classical_return = classical_result.expected_return * 252
        classical_variance = classical_result.variance * 252
        classical_risk = np.sqrt(classical_variance)
        classical_sharpe = classical_return / classical_risk if classical_risk > 0 else 0

        # Run MIP (integer-constrained) baseline alongside Markowitz
        mip_k = num_assets_select if num_assets_select else max(1, num_assets // 2)
        mip_k = min(mip_k, num_assets)  # clamp
        mip_result = None
        try:
            mip_result = mip_baseline(
                expected_returns=mean_returns,
                covariance=cov_matrix,
                budget=1.0,
                num_assets=mip_k,
                risk_aversion=risk_aversion / 10000,
            )
            mip_return = mip_result.expected_return * 252
            mip_variance = mip_result.variance * 252
            mip_risk = np.sqrt(mip_variance)
            mip_sharpe = mip_return / mip_risk if mip_risk > 0 else 0
        except Exception as e:
            logger.warning(f"MIP baseline failed: {e}")
            mip_result = None

        # Score solution quality
        quality = score_solution(
            allocations=np.array(weights),
            expected_return=expected_return,
            portfolio_risk=portfolio_risk,
            budget=1.0,
            classical_return=classical_return if classical_result.success else None,
            classical_risk=classical_risk if classical_result.success else None,
        )

        response = {
            'success': True,
            'tickers': tickers,
            'selected_assets': selected_assets,
            'weights': {t: round(w * 100, 1) for t, w in zip(tickers, weights)},
            'optimal_value': round(result.optimal_value, 6),
            'resolution': {
                'qubits_per_asset': resolution_qubits,
                'allocation_levels': 2 ** resolution_qubits,
                'total_qubits': total_qubits,
            },
            'expected_return': round(expected_return * 100, 2),
            'portfolio_risk': round(portfolio_risk * 100, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'iterations': result.num_evaluations,
            'settings': {
                'algorithm': algorithm,
                'risk_factor': risk_factor,
                'ansatz': ansatz_type if algorithm == 'vqe' else None,
                'reps': reps if algorithm == 'vqe' else None,
                'qaoa_layers': qaoa_layers if algorithm == 'qaoa' else None,
                'backend': backend_type
            },
            'warm_start': {
                'enabled': use_warm_start,
                'method': warm_start_result.method if warm_start_result else None,
                'estimated_improvement': round(warm_start_result.estimated_improvement, 1) if warm_start_result else None,
            },
            'classical_baseline': {
                'expected_return': round(classical_return * 100, 2),
                'portfolio_risk': round(classical_risk * 100, 2),
                'sharpe_ratio': round(classical_sharpe, 2),
                'allocations': {t: round(w * 100, 1) for t, w in zip(tickers, classical_result.allocations)},
                'success': classical_result.success,
            },
            'mip_baseline': {
                'expected_return': round(mip_return * 100, 2) if mip_result and mip_result.success else None,
                'portfolio_risk': round(mip_risk * 100, 2) if mip_result and mip_result.success else None,
                'sharpe_ratio': round(mip_sharpe, 2) if mip_result and mip_result.success else None,
                'allocations': (
                    {t: round(w * 100, 1) for t, w in zip(tickers, mip_result.allocations)}
                    if mip_result and mip_result.success else None
                ),
                'num_assets_selected': mip_k,
                'success': mip_result.success if mip_result else False,
                'message': mip_result.message if mip_result else 'MIP baseline not computed',
            },
            'qaoa_settings': {
                'cvar_alpha': cvar_alpha if algorithm == 'qaoa' else None,
                'mixer_type': mixer_type if algorithm == 'qaoa' else None,
            },
            'esg': {
                'enabled': esg_scores is not None and esg_weight > 0,
                'weight': esg_weight,
                'scores': {t: round(s, 2) for t, s in zip(tickers, esg_scores)} if esg_scores else None,
            },
            'convergence': {
                'history': result.history[:100] if len(result.history) > 100 else result.history,
                'best_history': result.best_history[:100] if len(result.best_history) > 100 else result.best_history,
                'converged': result.converged,
            },
            'quality_score': {
                'total_score': round(quality.total_score, 1),
                'grade': quality.grade,
                'components': {k: round(v, 1) for k, v in quality.component_scores.items()},
                'summary': quality.summary,
            },
        }

        if algorithm == 'vqe':
            circuit_report = dict(result.ansatz_report or {})
            circuit_report.setdefault('name', ansatz_type)
            circuit_report['ansatz'] = ansatz_type
        else:
            circuit_report = dict(getattr(result, 'circuit_report', {}) or {})
            circuit_report.setdefault('name', 'QAOA')
        circuit_report.update({
            'algorithm': algorithm.upper(),
            'backend': backend_type,
            'shots': backend_config.get('shots'),
        })
        response['circuit_report'] = circuit_report

        # Add measurement statistics if available
        if result.measurement_counts:
            total = sum(result.measurement_counts.values())
            sorted_counts = sorted(result.measurement_counts.items(),
                                   key=lambda x: x[1], reverse=True)[:5]
            response['measurements'] = {
                'total_shots': total,
                'top_solutions': [
                    {'bitstring': bs, 'count': count, 'probability': round(count/total*100, 1)}
                    for bs, count in sorted_counts
                ]
            }

        logger.info(f"Optimization complete: {selected_assets}")
        return jsonify(response)

    except QuantumPortfolioError as e:
        logger.exception("Optimization failed with typed exception")
        error_dict = e.to_dict()
        return jsonify({
            'error': error_dict['message'],
            'technical_details': str(e),
            'error_type': error_dict['error_type'],
            'details': error_dict.get('details', {})
        }), 500
    except Exception as e:
        logger.exception("Optimization failed")
        error_info = classify_error(str(e))
        return jsonify({
            'error': error_info['message'],
            'technical_details': error_info['technical'],
            'error_type': error_info['type']
        }), 500


@app.route('/backends')
def list_backends():
    """List available backends."""
    backends = [
        {'id': 'local_simulator', 'name': 'Local Simulator', 'description': 'Fast, runs on your computer'},
    ]

    # Check if IBM Quantum is available
    if os.environ.get('QE_TOKEN'):
        backends.append({
            'id': 'ibm_quantum',
            'name': 'IBM Quantum',
            'description': 'Real quantum hardware (requires API key)'
        })

    return jsonify(backends)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  Quantum Portfolio Optimizer")
    print("  Open http://localhost:8080 in your browser")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=8080)

"""Flask web frontend for Quantum Portfolio Optimizer."""

import os
import logging
from flask import Flask, render_template, request, jsonify
import numpy as np

from quantum_portfolio_optimizer.data import fetch_stock_data
from quantum_portfolio_optimizer.data.returns_calculator import calculate_logarithmic_returns
from quantum_portfolio_optimizer.core.qubo_formulation import PortfolioQUBO
from quantum_portfolio_optimizer.core.vqe_solver import PortfolioVQESolver
from quantum_portfolio_optimizer.core.qaoa_solver import PortfolioQAOASolver
from quantum_portfolio_optimizer.core.optimizer_interface import DifferentialEvolutionConfig
from quantum_portfolio_optimizer.simulation.provider import get_provider
from quantum_portfolio_optimizer.benchmarks.classical_baseline import markowitz_baseline
from quantum_portfolio_optimizer.postprocessing.quality_scorer import score_solution
from quantum_portfolio_optimizer.core.warm_start import (
    warm_start_vqe, warm_start_qaoa, WarmStartConfig
)
from qiskit.circuit.library import RealAmplitudes, EfficientSU2
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
        use_warm_start = data.get('warm_start', True)  # Default enabled
        resolution_qubits = int(data.get('resolution_qubits', 1))  # Default binary

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
        mean_returns = np.mean(log_returns, axis=0)
        cov_matrix = np.cov(log_returns, rowvar=False)
        num_assets = len(tickers)

        # Scale risk aversion based on slider (0-1) -> (100-10000)
        # Lower slider = lower risk tolerance = higher risk_aversion (penalize risk more)
        risk_aversion = 100 + (1 - risk_factor) * 9900

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
        )
        qubo = qubo_builder.build()
        total_qubits = num_assets * resolution_qubits
        logger.info(f"QUBO built: {num_assets} assets x {resolution_qubits} resolution = {total_qubits} qubits")

        # Get backend
        if backend_type == "ibm_quantum":
            ibm_device = data.get('ibm_device', 'ibm_strasbourg')
            ibm_api_key = data.get('ibm_api_key', '')
            ibm_crn = data.get('ibm_crn', '')

            if not ibm_api_key or not ibm_crn:
                return jsonify({'error': 'IBM Quantum API Key and CRN are required'}), 400

            # Set credentials as environment variables for the provider
            os.environ['QE_TOKEN'] = ibm_api_key
            os.environ['IBM_CLOUD_CRN'] = ibm_crn

            backend_config = {
                "name": "ibm_quantum",
                "device": ibm_device,
                "channel": "ibm_cloud",
                "instance": ibm_crn,
                "shots": 4096,
                "optimization_level": 3,
                "use_session": True,
                "error_mitigation": {
                    "resilience_level": 1,
                    "dynamical_decoupling": True,
                    "dd_sequence": "XpXm",
                    "twirling_enabled": True,
                }
            }
            logger.info(f"Using IBM Quantum backend: {ibm_device}")
        else:
            backend_config = {"name": "local_simulator", "shots": 1024, "seed": 42}

        estimator, sampler = get_provider(backend_config)

        # Track progress
        progress_data = {'iteration': 0, 'best_energy': float('inf')}
        def progress_callback(iteration, energy, best_energy):
            progress_data['iteration'] = iteration
            progress_data['best_energy'] = best_energy

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
                    # Build ansatz for warm start
                    ansatz_class = RealAmplitudes if ansatz_type == 'real_amplitudes' else EfficientSU2
                    temp_ansatz = ansatz_class(num_assets, reps=reps)
                    warm_start_result = warm_start_vqe(
                        expected_returns=mean_returns,
                        covariance=cov_matrix,
                        ansatz=temp_ansatz,
                        config=warm_config,
                    )
                    initial_point = warm_start_result.initial_parameters.tolist()
                    logger.info(f"Warm start (VQE): estimated {warm_start_result.estimated_improvement:.1f}x improvement")
            except Exception as e:
                logger.warning(f"Warm start failed, using random initialization: {e}")
                warm_start_result = None
                initial_point = None

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

            solver = PortfolioQAOASolver(
                sampler=sampler,
                estimator=estimator,
                layers=qaoa_layers,
                optimizer_config=optimizer_config,
                seed=42,
                progress_callback=progress_callback,
                shots=1024,
            )
            logger.info(f"Running QAOA with {qaoa_layers} layers")
        else:
            # VQE: Configure optimizer bounds for ansatz parameters
            bounds = [(-2 * np.pi, 2 * np.pi)] * (num_assets * (reps + 1) * 2)
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

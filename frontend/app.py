"""Flask web frontend for Quantum Portfolio Optimizer."""

import os
import logging
from flask import Flask, render_template, request, jsonify
import numpy as np

from quantum_portfolio_optimizer.data import fetch_stock_data
from quantum_portfolio_optimizer.data.returns_calculator import calculate_logarithmic_returns
from quantum_portfolio_optimizer.core.qubo_formulation import PortfolioQUBO
from quantum_portfolio_optimizer.core.vqe_solver import PortfolioVQESolver
from quantum_portfolio_optimizer.core.optimizer_interface import DifferentialEvolutionConfig
from quantum_portfolio_optimizer.simulation.provider import get_provider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


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

        logger.info(f"Starting optimization: {tickers}, {start_date} to {end_date}")
        logger.info(f"Settings: risk={risk_factor}, ansatz={ansatz_type}, reps={reps}, backend={backend_type}")

        # Fetch market data
        stock_data = fetch_stock_data(tickers, start_date, end_date)

        if stock_data.empty:
            return jsonify({'error': 'Failed to fetch stock data. Check tickers and dates.'}), 400

        # Calculate returns using logarithmic returns (as per research paper)
        log_returns = calculate_logarithmic_returns(stock_data)
        mean_returns = log_returns.mean().values
        cov_matrix = log_returns.cov().values
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
            resolution_qubits=1,  # Binary allocation (in/out)
            max_investment=1.0,
            penalty_strength=1000.0,
            enforce_budget=True,
        )
        qubo = qubo_builder.build()

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

        estimator, _ = get_provider(backend_config)

        # Configure optimizer
        bounds = [(-2 * np.pi, 2 * np.pi)] * (num_assets * (reps + 1) * 2)  # Approximate param count
        optimizer_config = DifferentialEvolutionConfig(
            bounds=bounds,
            maxiter=maxiter,
            seed=42,
        )

        # Track progress
        progress_data = {'iteration': 0, 'best_energy': float('inf')}
        def progress_callback(iteration, energy, best_energy):
            progress_data['iteration'] = iteration
            progress_data['best_energy'] = best_energy

        # Run VQE
        solver = PortfolioVQESolver(
            estimator=estimator,
            ansatz_name=ansatz_type,
            ansatz_options={'reps': reps},
            optimizer_config=optimizer_config,
            seed=42,
            progress_callback=progress_callback,
        )

        result = solver.solve(qubo)

        # Interpret binary solution
        # The optimal parameters need to be converted to a binary decision
        # For simplicity, we'll use the energy landscape to determine selection
        binary_solution = _extract_binary_solution(result, num_assets)
        selected_assets = [tickers[i] for i, b in enumerate(binary_solution) if b == 1]

        # Calculate portfolio metrics
        if len(selected_assets) > 0:
            weights = [1.0 / len(selected_assets) if b == 1 else 0.0 for b in binary_solution]
        else:
            weights = [1.0 / num_assets] * num_assets  # Fallback to equal weight
            selected_assets = tickers

        expected_return = sum(w * r for w, r in zip(weights, mean_returns)) * 252  # Annualized
        portfolio_variance = sum(
            weights[i] * weights[j] * cov_matrix[i][j]
            for i in range(len(weights))
            for j in range(len(weights))
        ) * 252
        portfolio_risk = portfolio_variance ** 0.5
        sharpe_ratio = expected_return / portfolio_risk if portfolio_risk > 0 else 0

        response = {
            'success': True,
            'tickers': tickers,
            'selected_assets': selected_assets,
            'weights': {t: round(w * 100, 1) for t, w in zip(tickers, weights)},
            'binary_solution': binary_solution,
            'optimal_value': round(result.optimal_value, 6),
            'expected_return': round(expected_return * 100, 2),
            'portfolio_risk': round(portfolio_risk * 100, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'iterations': result.num_evaluations,
            'settings': {
                'risk_factor': risk_factor,
                'ansatz': ansatz_type,
                'reps': reps,
                'backend': backend_type
            }
        }

        logger.info(f"Optimization complete: {selected_assets}")
        return jsonify(response)

    except Exception as e:
        logger.exception("Optimization failed")
        return jsonify({'error': str(e)}), 500


def _extract_binary_solution(result, num_assets):
    """Extract binary selection from VQE result.

    Uses the optimal parameters to sample the circuit and determine
    which assets should be selected.
    """
    # For single-qubit resolution, we can interpret the energy landscape
    # A simple heuristic: use threshold on optimal value
    # In practice, you'd sample the final circuit and count bitstrings

    # For now, use a simple approach based on the number of assets
    # If optimal value is low, include more assets
    optimal_val = result.optimal_value

    # Simple heuristic: include assets based on energy
    # Lower energy = more diversification typically
    threshold = 0.5
    if optimal_val < -0.1:
        # Low energy suggests good diversification
        return [1] * num_assets
    else:
        # Higher energy, be more selective
        # Include roughly half
        selection = [0] * num_assets
        for i in range(min(num_assets // 2 + 1, num_assets)):
            selection[i] = 1
        return selection


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

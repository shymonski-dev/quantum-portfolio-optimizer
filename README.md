# Quantum Portfolio Optimizer

A quantum computing application for portfolio optimization using VQE and QAOA algorithms. This project enables users to optimize investment portfolios using both local quantum simulators and real IBM Quantum hardware, with support for up to 25 assets.

## Overview

The Quantum Portfolio Optimizer uses quantum computing to solve the Markowitz portfolio optimization problem. It formulates the asset allocation as a Quadratic Unconstrained Binary Optimization (QUBO) problem and solves it using hybrid quantum-classical algorithms.

### Supported Algorithms

- **VQE (Variational Quantum Eigensolver)**: Recommended for most use cases. Uses shallow parameterized circuits with configurable ansätze (RealAmplitudes, EfficientSU2). Best for 10+ assets due to lower circuit depth.
- **QAOA (Quantum Approximate Optimization Algorithm)**: Experimental alternative. Uses problem-specific cost and mixer layers. Configurable depth (p=1 or p=2).

### How It Works

1. **Data Fetching**: Retrieves historical stock prices from Yahoo Finance
2. **Returns Calculation**: Computes logarithmic returns and covariance matrices
3. **QUBO Formulation**: Converts the portfolio optimization into a quantum-compatible format
4. **Quantum Optimization**: Uses VQE or QAOA to find optimal asset allocations
5. **Binary Solution Extraction**: Samples the optimized circuit to get actual portfolio selections
6. **Results Interpretation**: Translates quantum results into portfolio weights and metrics

## Features

- **Multiple Algorithms**: Choose between VQE (recommended) and QAOA (experimental) via web UI
- **Binary Solution Extraction**: Actual bitstring sampling from optimized circuits
- **Web Interface**: User-friendly Flask-based frontend for easy interaction
- **Dual Backend Support**: Run on local simulators or IBM Quantum hardware
- **Real-time Data**: Fetches live market data via Yahoo Finance
- **Configurable Risk**: Adjustable risk tolerance from conservative to aggressive
- **Multiple Ansätze**: Support for RealAmplitudes and EfficientSU2 quantum circuits (VQE)
- **Error Mitigation**: Built-in support for dynamical decoupling and Pauli twirling
- **Qiskit 2.x Compatible**: Updated for the latest Qiskit quantum computing framework
- **Scalable**: Supports up to 25 assets on IBM Quantum hardware (156 qubits)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd quantum_portfolio_optimizer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# For IBM Quantum hardware support
pip install qiskit-ibm-runtime
```

## Quick Start

### Web Interface (Recommended)

```bash
cd frontend
python app.py
```

Open http://localhost:8080 in your browser.

### Command Line

```bash
# Run with default configuration
python -m quantum_portfolio_optimizer

# Run tests
pytest
```

## Web Interface Usage

1. **Enter Stock Tickers**: Comma-separated symbols (e.g., AAPL, GOOG, MSFT)
2. **Set Date Range**: Historical period for analysis
3. **Adjust Risk Tolerance**: Slider from conservative (left) to aggressive (right)
4. **Select Backend**:
   - **Local Simulator**: Fast, runs on your computer
   - **IBM Quantum Hardware**: Real quantum computer (requires credentials)
5. **Configure Algorithm** (Advanced Settings):
   - **VQE**: Choose ansatz type (RealAmplitudes/EfficientSU2) and circuit depth
   - **QAOA**: Choose number of layers (p=1 or p=2)
6. **Run Optimization**: Click "Optimize Portfolio" and view results

### Algorithm Selection Guide

| Scenario | Recommended Algorithm |
|----------|----------------------|
| 10+ assets | VQE with RealAmplitudes |
| Quick exploration | VQE with reps=1 |
| Small problems (<8 assets) | Either VQE or QAOA |
| Research/comparison | Try both |

### IBM Quantum Hardware Setup

To use real quantum hardware:

1. Create an account at [IBM Quantum](https://quantum.ibm.com)
2. Get your API key and Cloud Resource Name (CRN) from the dashboard
3. Select "IBM Quantum Hardware" in the web interface
4. Enter your API key and CRN
5. Choose a device (e.g., ibm_aachen, ibm_strasbourg)

## Project Structure

```
quantum_portfolio_optimizer/
├── frontend/                    # Web interface
│   ├── app.py                  # Flask application
│   └── templates/              # HTML templates
├── src/quantum_portfolio_optimizer/
│   ├── core/                   # Core algorithms
│   │   ├── qubo_formulation.py # QUBO builder
│   │   ├── vqe_solver.py       # VQE implementation
│   │   ├── qaoa_solver.py      # QAOA implementation
│   │   └── ansatz_library.py   # Quantum circuits
│   ├── data/                   # Data handling
│   │   ├── data_fetcher.py     # Yahoo Finance integration
│   │   └── returns_calculator.py
│   └── simulation/             # Backend providers
│       ├── provider.py         # Backend factory
│       └── ibm_provider.py     # IBM Quantum integration
├── tests/                      # Test suite
├── config.yaml                 # Configuration file
└── requirements.txt
```

## Configuration

Edit `config.yaml` to customize defaults:

```yaml
portfolio:
  tickers: ["AAPL", "GOOG", "MSFT", "AMZN"]
  start_date: "2022-01-01"
  end_date: "2023-01-01"

algorithm:
  name: "vqe"
  settings:
    ansatz: "RealAmplitudes"
    reps: 2

backend:
  name: "local_simulator"  # or "ibm_quantum"
  shots: 1024
```

## IBM Quantum Error Mitigation

When using IBM Quantum hardware, the following error mitigation techniques are available:

- **Dynamical Decoupling**: Suppresses decoherence during idle times
- **Pauli Twirling**: Randomizes noise for better mitigation
- **Zero-Noise Extrapolation (ZNE)**: Extrapolates to zero-noise limit (advanced)

Configure in the web interface or `config.yaml`:

```yaml
ibm_quantum:
  device: "ibm_brisbane"
  error_mitigation:
    resilience_level: 1
    dynamical_decoupling: true
    dd_sequence: "XpXm"
    twirling_enabled: true
```

## Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_vqe_solver.py
```

All 39 tests pass, including integration tests, QAOA tests, and IBM Quantum compatibility tests.

## Requirements

- Python 3.10+
- Qiskit >= 1.0.0
- Flask (for web interface)
- NumPy, Pandas, SciPy
- yfinance (for market data)
- qiskit-ibm-runtime (optional, for IBM Quantum)

## License

MIT License

## Acknowledgments

- Built with [Qiskit](https://qiskit.org/)
- Market data from [Yahoo Finance](https://finance.yahoo.com/)
- Inspired by quantum finance research on portfolio optimization

# Quantum Portfolio Optimizer

[![Tests](https://github.com/shymonski-dev/quantum-portfolio-optimizer/actions/workflows/tests.yml/badge.svg)](https://github.com/shymonski-dev/quantum-portfolio-optimizer/actions/workflows/tests.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit 2.x](https://img.shields.io/badge/Qiskit-2.x-6929c4.svg)](https://qiskit.org)

A quantum-classical hybrid application for portfolio optimization using Variational Quantum Eigensolver (VQE) and Quantum Approximate Optimization Algorithm (QAOA). Optimized for **2026 IBM Quantum Hardware**, this software supports ISA-compliant execution, AI-enhanced transpilation, ESG constraints, CVaR risk objectives, and integer-constrained classical baselines.

## Overview

The Quantum Portfolio Optimizer solves the Markowitz portfolio optimization problem by formulating asset allocation as a Quadratic Unconstrained Binary Optimization (QUBO) problem and solving it with hybrid quantum-classical algorithms.

### Machine Readable Summary
```
project_name: quantum_portfolio_optimizer
version: 0.2.1
primary_goal: markowitz_portfolio_optimization
problem_formulation: quadratic_unconstrained_binary_optimization
algorithm_value_vqe: vqe
algorithm_value_qaoa: qaoa
backend_value_local_simulator: local_simulator
backend_value_ibm_cloud: ibm_cloud
isa_compliance: fully_supported
ai_transpilation: supported
ai_sector_clustering: supported
ansatz_functional_builders: supported
circuit_knitting_partitioning: supported
cvar_qaoa: supported
xy_mixer: supported
esg_constraints: supported
mip_baseline: supported
zne_local: supported
max_supported_assets: 100
test_count: 336
```

### Supported Algorithms

**VQE (Variational Quantum Eigensolver)** — Recommended for most use cases. Uses shallow parameterized circuits (`real_amplitudes` or `efficient_su2` functional builders) with differential evolution optimization. Best for 10+ assets due to lower circuit depth. Supports warm-start initialization from the Markowitz solution.

**QAOA (Quantum Approximate Optimization Algorithm)** — Problem-specific cost and mixer layers. Configurable depth (p=1, 2, …). Two variants:
- **X-Mixer** (standard): Transverse field, energy-weighted bitstring evaluation
- **XY-Mixer + Dicke state init**: Constraint-preserving mixer that maintains exact portfolio cardinality without penalty tuning (Bartschi & Eidenbenz 2019)

Both QAOA variants support the **CVaR objective** (`cvar_alpha` ∈ (0,1]): evaluating only the lowest-energy tail of the shot distribution gives up to 4.5× faster convergence (Barkoutsos et al. 2020).

### How It Works

1. **Data Fetching**: Institutional-grade market data via **OpenBB Platform SDK** (supporting Tiingo, yfinance, FMP, etc.)
2. **Returns Calculation**: Logarithmic returns and covariance matrices
3. **QUBO Formulation**: Portfolio optimization converted to a quantum-compatible form, with optional ESG scoring and CVaR risk objective
4. **Quantum Optimization**: VQE or QAOA finds optimal asset allocations; **ISA-compliant transpilation** and **AI-enhanced optimization** used for IBM hardware.
5. **Solution Extraction**: Bitstring sampling from the optimized circuit
6. **Classical Baselines**: Markowitz continuous optimization and MIP integer-constrained exhaustive baseline for honest comparison
7. **Results Interpretation**: Portfolio weights, risk metrics, quality score

## Features

- **2026 IBM Hardware Ready**: Full support for `ibm_cloud` channels, ISA-compliant circuits, and native Resilience V2 (Cloud-side ZNE).
- **Dynamic Circuit Partitioning**: Supports "Circuit Knitting" via `qiskit-addon-cutting` to split large portfolios across multiple quantum processor modules (e.g., IBM Kookaburra architecture).
- **AI-Enhanced Transpilation**: Leverages 2026-era AI passes to minimize gate counts and noise impact on chips like "Nighthawk" and "Flamingo."
- **Functional Circuit Builders**: Migrated to modern Qiskit functional builders (`real_amplitudes`, `efficient_su2`) for reduced memory overhead and future-proofing.
- **Multiple Algorithms**: VQE (recommended) and QAOA with X or XY mixer via the web UI
- **CVaR-QAOA**: Conditional Value-at-Risk objective for faster, noise-robust convergence
- **ESG Constraints**: Per-asset ESG scores and weighting term in the QUBO objective
- **MIP Baseline**: Exhaustive integer-constrained classical solver alongside Markowitz for honest benchmarking
- **ZNE Gate Folding**: Provider-agnostic Zero Noise Extrapolation on local simulators and managed ZNE on IBM hardware.
- **Warm Start**: VQE and QAOA initialized from classical Markowitz/heuristic solutions
- **Dual Backend**: Local StatevectorSampler or IBM Quantum hardware (Heron r1/r2)
- **Multi-resolution Encoding**: Configurable qubits-per-asset for finer allocation granularity
- **336 Tests**: Full unit and integration coverage; 100% passing on Python 3.10–3.14

## Installation

```bash
git clone https://github.com/shymonski-dev/quantum-portfolio-optimizer.git
cd quantum-portfolio-optimizer

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .

# Optional: IBM Quantum hardware support
pip install -e ".[ibm]"

# Optional: noise simulation via Qiskit Aer
pip install -e ".[noise]"

# Everything
pip install -e ".[all]"
```

## Setup & Security

For IBM Quantum hardware, this software utilizes environment variables to ensure API keys are never hardcoded or committed to source control.

```bash
# For IBM Quantum Platform (Free Tier)
export IBM_QUANTUM_TOKEN="your_api_key"

# For IBM Cloud (Enterprise/U.S. East)
export IBM_QUANTUM_TOKEN="your_iam_api_key"
export IBM_CLOUD_INSTANCE="your_crn_instance"
```

# Institutional Data Setup (OpenBB)
This software uses the **OpenBB Platform SDK** for high-fidelity data. While `yfinance` is available as a free fallback, professional users should configure **Tiingo** for adjusted data quality.

```bash
# Set your OpenBB credentials as environment variables
export OPENBB_TIINGO_TOKEN="your_tiingo_key"
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
qpo          # uses config.yaml defaults
pytest       # run all tests
```

## GitHub Marketplace Action

This repository now includes a root `action.yml` so it can be published as a GitHub Marketplace Action.

### Quick Workflow Example

```yaml
name: Portfolio Optimization

on:
  workflow_dispatch:

jobs:
  optimize:
    runs-on: ubuntu-latest
    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Run quantum portfolio optimizer
        id: optimizer
        uses: shymonski-dev/quantum-portfolio-optimizer@main
        with:
          config-path: config.yaml
          python-version: "3.11"

      - name: Print summary
        run: |
          echo "Algorithm: ${{ steps.optimizer.outputs.algorithm }}"
          echo "Backend: ${{ steps.optimizer.outputs.backend }}"
          echo "Objective value: ${{ steps.optimizer.outputs.objective-value }}"
          echo "Expected return (%): ${{ steps.optimizer.outputs.expected-return-pct }}"
          echo "Portfolio risk (%): ${{ steps.optimizer.outputs.portfolio-risk-pct }}"
          echo "Sharpe ratio: ${{ steps.optimizer.outputs.sharpe-ratio }}"
```

Use `@main` for initial testing, then move to a stable release tag such as `@v0` for production workflows.

### Action Inputs

| Input | Default | Description |
|-------|---------|-------------|
| `config-path` | `config.yaml` | Path to configuration file for the optimizer command |
| `python-version` | `3.11` | Python version used on the runner |
| `install-extras` | empty | Package extras to install (empty for base package only) |
| `working-directory` | `.` | Directory that contains project and config files |
| `print-json` | `false` | Print full JSON payload to workflow logs |

### Action Outputs

| Output | Description |
|--------|-------------|
| `result-path` | Path to JSON result file on runner |
| `result-json` | Full JSON payload on one line |
| `algorithm` | Solver algorithm used |
| `backend` | Backend used |
| `objective-value` | Objective value from optimizer |
| `expected-return-pct` | Expected annual return percent |
| `portfolio-risk-pct` | Portfolio volatility percent |
| `sharpe-ratio` | Sharpe ratio |
| `converged` | Convergence flag |
| `best-bitstring` | Best measured bitstring when available |

### Marketplace Publish Checklist

1. Push changes to the default branch in the public repository.
2. Create a release tag (for example `v0.3.0`) and a major tag (`v0`).
3. Open the repository on GitHub, then choose **Releases** and publish the release.
4. Open **Actions** listing flow from the repository and submit metadata for review.
5. After approval, update workflow examples to use the stable major tag (`@v0`).

## Web Interface Usage

1. **Enter Stock Tickers** — comma-separated (e.g., `AAPL, GOOG, MSFT, AMZN`)
2. **Set Date Range** — historical period for return/covariance estimation
3. **Adjust Risk Tolerance** — slider: conservative (left) → aggressive (right)
4. **Select Backend** — Local Simulator or IBM Quantum (requires credentials)
5. **Algorithm** (Advanced Settings):
   - **VQE**: ansatz type (real_amplitudes / efficient_su2), circuit depth (reps)
   - **QAOA**: layers (p), CVaR alpha slider, mixer type (X or XY)
6. **ESG & Integer Constraints**:
   - ESG scores (one per ticker), ESG weight (0 = disabled)
   - Assets to select (K) — drives the MIP baseline and XY-mixer cardinality
7. **Run Optimization** and compare quantum vs Markowitz vs MIP results

### Algorithm Selection Guide

| Scenario | Recommendation |
|----------|---------------|
| 10+ assets, hardware | VQE with real_amplitudes, reps=2 |
| Fast exploration | VQE with reps=1 |
| Noise-robust small problems | CVaR-QAOA (alpha=0.2–0.5), p=1 |
| Strict cardinality constraint | XY-mixer QAOA + Dicke state init |
| ESG mandate | Enable ESG scores + weight ≥ 0.1 |
| Honest benchmarking | MIP baseline with same K as quantum |

### IBM Quantum Setup

1. Create an account at [IBM Quantum](https://quantum.ibm.com/)
2. Obtain your API key plus either a hub/group/project instance (ibm_quantum channel) or a Cloud Resource Name (ibm_cloud channel)
3. In the web UI: select IBM Quantum backend, choose channel and device, enter credentials
4. Recommended settings for 2026 Heron hardware:
   - Resilience Level 1 (DD + Twirling)
   - ZNE gate folding: noise factors `1,3,5`, extrapolator `exponential`

## Project Structure

```
quantum_portfolio_optimizer/
├── frontend/
│   ├── app.py                       # Flask application (Phase 4 complete)
│   └── templates/index.html         # Web UI
├── src/quantum_portfolio_optimizer/
│   ├── core/
│   │   ├── qubo_formulation.py      # QUBO builder (ESG, CVaR risk, multi-resolution)
│   │   ├── vqe_solver.py            # VQE with warm start + bitstring extraction
│   │   ├── qaoa_solver.py           # QAOA: X/XY mixer, CVaR objective, ZNE
│   │   ├── warm_start.py            # Classical → quantum parameter initialization
│   │   ├── constraints.py           # Unified constraint system
│   │   └── ansatz_library.py        # RealAmplitudes, EfficientSU2
│   ├── data/
│   │   ├── data_fetcher.py          # OpenBB Platform integration
│   │   └── returns_calculator.py    # Log returns + covariance
│   ├── simulation/
│   │   ├── provider.py              # Backend factory (local / IBM)
│   │   ├── ibm_provider.py          # IBM Quantum integration (Qiskit 2.x)
│   │   └── zne.py                   # Provider-agnostic ZNE gate folding
│   ├── benchmarks/
│   │   └── classical_baseline.py    # Markowitz + MIP integer baseline
│   └── postprocessing/
│       └── quality_scorer.py        # A–F composite quality score
├── tests/                           # 331 tests, 100% passing
├── .github/workflows/               # CI: Python 3.10–3.12 matrix
├── config.yaml
├── CHANGELOG.md
├── CONTRIBUTING.md
└── CODE_OF_CONDUCT.md
```

## Configuration

Edit `config.yaml` to set defaults used by the CLI:

```yaml
portfolio:
  tickers: ["AAPL", "GOOG", "MSFT", "AMZN"]
  start_date: "2022-01-01"
  end_date: "2023-01-01"

algorithm:
  name: "vqe"
  settings:
    ansatz: "real_amplitudes"
    reps: 2

backend:
  name: "local_simulator"
  shots: 1024
```

## Error Mitigation

### Local Simulator — ZNE Gate Folding

Enable via the Advanced Settings or pass `zne_gate_folding: true` in the API request. Noise factors `[1, 3, 5]` with linear or Richardson extrapolation.

### IBM Quantum

| Resilience Level | Techniques |
|-----------------|-----------|
| 0 | None |
| 1 | Dynamical Decoupling + Pauli Twirling |
| 2 | Level 1 + managed ZNE |

ZNE via gate folding (provider-agnostic, `simulation/zne.py`) demonstrated a **31.6% energy improvement** on Heron r1 hardware (February 2026 benchmarks).

## Classical Baselines

Two baselines run automatically alongside every quantum optimization:

| Baseline | Method | Constraint |
|----------|--------|-----------|
| Markowitz | Continuous QP (scipy SLSQP) | Budget = 1.0 |
| MIP | Exhaustive subset QP (n ≤ 15) or greedy (n > 15) | Exactly K assets selected |

The MIP baseline gives an honest integer-constrained upper bound for comparison with QAOA's combinatorial objective.

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/

# Specific module
pytest tests/test_qaoa_solver.py -v
```

Current status: **331 passed, 1 skipped** on Python 3.12 / macOS (February 2026).

## Future Research & v0.3.0 Roadmap

While the current version provides a state-of-the-art implementation of Markowitz portfolio optimization, the v0.3.0 roadmap targets problems in the "Hard" complexity class where true Quantum Advantage resides.

### 1. Non-Gaussian Path Integrals
Standard financial models assume Gaussian (Normal) return distributions. v0.3.0 will explore **non-Hermitian quantum mechanics (PT-Symmetry)** to model leptokurtic "fat-tailed" markets. Using path integral kernels that naturally account for extreme events, the optimizer will minimize risk against **"Path-Based Ruin"** rather than simple variance.

### 2. Crash-Manifold Prediction (Quantum TDA)
Market crashes are often topological collapses of high-dimensional data structures. We are researching the integration of **Quantum Topological Data Analysis (TDA)** to monitor the "shape" of the market manifold. 
- **Persistent Homology**: Using Betti numbers to detect structural brittleness.
- **Early Warning System**: Monitoring spikes in the **Persistence Landscape Norm** to predict when a portfolio is approaching a "cliff edge" or manifold singularity.

### 3. Scaling to 4,000+ Qubits
Leveraging the **IBM Kookaburra** architecture, we will extend the **Modular Partitioning** logic to support 100+ assets with high-resolution discretization, utilizing **Quantum Serverless** for real-time circuit knitting across multi-chip systems.

## Requirements

- Python 3.10+
- Qiskit >= 2.3.0
- Flask, NumPy, Pandas, SciPy, scikit-learn
- scikit-learn>=1.3.0
- openbb-core>=1.6.0
- openbb-equity>=1.6.0
- openbb-yfinance>=1.4.0
- openbb-tiingo>=1.4.0
- yfinance==0.2.55
- qiskit-ibm-runtime *(optional — IBM Quantum hardware)*
- qiskit-aer *(optional — noise simulation)*

## License

This software is dual-licensed under the **GNU Affero General Public License v3 (AGPLv3)** for open-source use and a **Commercial License** for proprietary enterprise use. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Built with [Qiskit](https://qiskit.org/)
- Institutional market data via [OpenBB Platform](https://openbb.co/)
- CVaR-QAOA: Barkoutsos et al. (2020), *Improving Variational Quantum Optimization using CVaR*
- XY-Mixer: Bartschi & Eidenbenz (2019), *Grover Mixers for QAOA*
- ZNE: Temme et al. (2017), *Error Mitigation for Short-Depth Quantum Circuits*

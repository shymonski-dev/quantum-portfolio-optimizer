# Local Quantum Portfolio Optimizer

This project provides a local-only implementation of a quantum portfolio optimisation workflow inspired by the Global Data Quantum architecture.  
The code targets **Phase&nbsp;1** capabilities:

- Build small portfolio optimisation instances (2–3 assets, few time steps) as QUBO/Ising problems.
- Solve the resulting Ising Hamiltonians with a Variational Quantum Eigensolver (VQE) running on local Qiskit simulators.
- Manage asset data locally, including simple validation and synthetic data generation.
- Supply a basic example and unit tests that exercise the end-to-end stack.

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
python examples/basic_example.py
```

## Phase 1 Deliverables

- Core QUBO builder supporting configurable risk aversion, budget constraints, time steps, and binary resolution.
- Minimal VQE solver using Qiskit's primitives, a real-amplitudes ansatz, and SciPy differential evolution.
- Asset data utilities for loading small CSV/JSON datasets with forward-fill handling of gaps.
- A small example demonstrating the workflow with synthetic 3-asset data.

Future phases will extend the framework with advanced ansätze, noise-aware post-processing, benchmarking, and documentation notebooks.

# Quantum Portfolio Optimizer — Progress Report

## Current Status: v0.2.0 — Fully Functional

All planned phases are complete. The optimizer runs on local simulation and IBM Quantum
hardware, with a full web UI, ESG constraints, CVaR-QAOA, XY-mixer, provider-agnostic
ZNE, and integer-constrained MIP classical baseline.

**319 tests passing** (1 skipped) on Python 3.14 / macOS — February 2026.

## Machine Readable Summary

```
project_name: quantum_portfolio_optimizer
version: 0.2.0
status: fully_functional
test_count: 319
last_updated: 2026-02-20
primary_backend_local: local_simulator
primary_backend_ibm: ibm_quantum
algorithm_vqe: supported
algorithm_qaoa_x_mixer: supported
algorithm_qaoa_xy_mixer: supported
cvar_qaoa: supported
esg_constraints: supported
mip_baseline: supported
zne_local: supported
zne_ibm: supported
warm_start: supported
quality_scorer: supported
```

## Completed Phases

### Phase 1 — Foundation
- YAML-based configuration system (`config.yaml`) and CLI (`qpo` entry point via click)
- Dynamic market data fetching via yfinance with ticker validation
- QUBO formulation for portfolio optimization (Markowitz + budget constraint)
- VQE solver with configurable ansätze (RealAmplitudes, EfficientSU2)
- 20 tests passing

### Phase 2 — IBM Quantum Integration
- `qiskit-ibm-runtime` integration: EstimatorV2 / SamplerV2 primitives (Qiskit 2.x)
- Session mode for iterative VQE/QAOA workloads
- Error mitigation: Dynamical Decoupling, Pauli Twirling, IBM-managed ZNE (levels 0–2)
- Secure credential handling via environment variables
- Tested on ibm_aachen (156 qubits) and ibm_strasbourg (127 qubits)

### Phase 3 — Web Frontend
- Flask application with dark-themed responsive UI (http://localhost:8080)
- Backend selector (Local Simulator / IBM Quantum) with device dropdown
- IBM credential input: API key, channel (ibm_quantum / ibm_cloud), instance / CRN
- Real-time results: expected return, portfolio risk, Sharpe ratio, allocation bar chart
- Convergence chart (Canvas), quantum measurement statistics, circuit details panel

### Phase 4 — Qiskit 2.x Migration
- Replaced deprecated `execute()` with EstimatorV2 / SamplerV2 throughout
- Fixed `SparsePauliOp` initialization syntax for Qiskit 2.x
- All 271 pre-existing tests pass with Qiskit 2.3.x

### Phase 5 — Algorithm Improvements
- **Binary Solution Extraction**: VQE uses Sampler to extract actual bitstrings;
  `best_bitstring` and `measurement_counts` fields on `VQEResult`
- **QAOA Solver** (`PortfolioQAOASolver`): configurable layers (p=1, 2, …), ZZ cost
  layer, RX mixer layer, differential evolution optimizer
- **Web UI Algorithm Selector**: VQE vs QAOA dropdown with algorithm-specific settings
- Test suite extended to 39 tests

### Phase 6 — Bug Fixes & Constraint System
- **QAOA coefficient double-counting fix**: symmetric QUBO off-diagonal were doubled;
  corrected to use `Q[i,j]` only
- **VQE empty-array guard**: descriptive `ValueError` instead of bare `IndexError`
- **Deprecation warnings**: `time_window` (returns_calculator), `estimator` (QAOA)
- **Unified constraint system** (`core/constraints.py`): `Constraint` ABC,
  `EqualityConstraint`, `InequalityConstraint`, `BoundsConstraint`, `ConstraintManager`
- Test suite extended to 271 tests

### Phase 7 — Research Features & Open-Source Release (v0.2.0, Feb 2026)

#### Algorithm Advances
- **CVaR-QAOA** (`cvar_alpha` parameter): evaluates the worst-α fraction of shots;
  up to 4.5× faster convergence on noisy hardware (Barkoutsos et al. 2020)
- **XY-Mixer QAOA + Dicke state initialization** (`mixer_type="xy"`): constraint-
  preserving mixer maintains exact portfolio cardinality without penalty tuning;
  accepts `num_assets` for the Hamming-weight subspace (Bartschi & Eidenbenz 2019)
- **CVaR risk metric in QUBO** (`risk_metric="cvar"`): tail-risk-aware objective beyond
  Markowitz variance; `cvar_confidence` configures the tail fraction
- **Warm-start system** (`core/warm_start.py`): `warm_start_vqe` seeds VQE parameters
  from the Markowitz solution; `warm_start_qaoa` uses a heuristic QAOA initializer

#### Infrastructure
- **Provider-agnostic ZNE gate folding** (`simulation/zne.py`): `fold_circuit` +
  `zne_extrapolate`; supports Richardson, linear, and exponential fits; demonstrated
  **31.6% energy improvement** on Heron r1 hardware (February 2026)
- **ESG constraints** (`esg_scores`, `esg_weight` in `PortfolioQUBO`): normalized ESG
  scores enter as negative linear QUBO terms incentivizing high-ESG allocation
- **Classical MIP baseline** (`mip_baseline()` in `classical_baseline.py`): exhaustive
  subset enumeration (n ≤ 15) with per-subset QP optimization; greedy fallback for
  larger universes; gives honest integer-constrained upper bound for comparison
- **Quality scorer** (`postprocessing/quality_scorer.py`): composite A–F grade from
  Sharpe ratio, budget feasibility, return magnitude, and vs-classical performance

#### Frontend Phase 4 Complete
- `frontend/app.py` parses `cvar_alpha`, `mixer_type`, `num_assets_select`,
  `esg_scores`, `esg_weight`; calls `mip_baseline()` on every run; ZNE config threaded
  to local simulator
- `frontend/templates/index.html`: CVaR alpha slider (QAOA only), mixer type dropdown,
  ESG scores/weight inputs, num-assets-to-select field, MIP column in comparison table

#### Open-Source Governance
- `LICENSE` (MIT), `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1)
- GitHub Actions CI/CD (`.github/workflows/`) — Python 3.9–3.12 matrix
- All `YOUR_ORG` placeholders replaced with `shymonski-dev`
- Test suite: **319 passing, 1 skipped**

## Architecture (v0.2.0)

```
User Interface (Flask — frontend/app.py)
    │
    ├─ ESG scores, CVaR alpha, mixer type, MIP K, ZNE config
    │
    ▼
PortfolioQUBO (qubo_formulation.py)
    Markowitz objective + budget constraint
    + CVaR risk metric (optional)
    + ESG incentive term (optional)
    │
    ▼
Warm Start (warm_start.py)
    Markowitz → initial circuit parameters
    │
    ▼
Algorithm Selection
    ├─ VQE (vqe_solver.py)
    │   RealAmplitudes / EfficientSU2 ansatz
    │   EstimatorV2 + SamplerV2
    │
    └─ QAOA (qaoa_solver.py)
        X-Mixer  or  XY-Mixer + Dicke state
        CVaR objective (configurable alpha)
        SamplerV2 + ZNE gate folding (optional)
    │
    ▼
Backend (provider.py)
    ├─ Local: StatevectorSampler / StatevectorEstimator
    │         ZNE gate folding (simulation/zne.py)
    │
    └─ IBM Quantum (ibm_provider.py)
           Heron r1/r2 (133–156 qubits)
           DD + Twirling + managed ZNE (levels 0–2)
    │
    ▼
Classical Baselines (classical_baseline.py)
    ├─ Markowitz: continuous QP (scipy SLSQP)
    └─ MIP: exhaustive subset QP (n ≤ 15) / greedy (n > 15)
    │
    ▼
Quality Scorer (postprocessing/quality_scorer.py)
    A–F composite grade
    │
    ▼
Results → JSON response → Web UI
```

## IBM Quantum Devices Tested

| Device | Qubits | Notes |
|--------|--------|-------|
| ibm_aachen | 156 | Heron r2; Bell state test passed |
| ibm_strasbourg | 127 | Heron r1; ZNE benchmark (+31.6%) |
| ibm_brussels | 127 | Available |

## Running the Application

```bash
# Web interface
cd frontend && python app.py
# → http://localhost:8080

# CLI
qpo

# Tests
pytest tests/ -v
# → 319 passed, 1 skipped
```

## Resolved Issues

| Issue | Resolution |
|-------|-----------|
| AppleClang 17.0 / qiskit-aer build failure | Made aer optional; use StatevectorSampler |
| QAOA coefficient double-counting | Fixed in qaoa_solver.py; verified by test suite |
| Qiskit 1.x → 2.x API breaks | Full migration to EstimatorV2 / SamplerV2 |
| IBM session credential handling | Env-var-based; CRN / instance auto-selected by channel |
| "CVaR not implemented" (previous future enhancement) | Implemented in v0.2.0 |
| "Warm-start not implemented" (previous future enhancement) | Implemented in v0.2.0 |

---
*Last updated: 2026-02-20*

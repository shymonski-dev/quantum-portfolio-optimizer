# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-02-21

### Added
- **Institutional Data Migration**: Successfully transitioned primary data fetching engine to **OpenBB Platform SDK**. Resolved `curl_cffi` session conflicts by pinning `yfinance==0.2.55`. 
- **AI Sector Clustering**: Implemented `AI Orchestrator` utilizing Spectral Min-Cut graph partitioning. Automatically identifies optimal hardware cut-points based on asset correlation manifolds to minimize modular execution overhead.
- **Dynamic Circuit Partitioning**: Integration with `qiskit-addon-cutting` to support modular quantum hardware (IBM Kookaburra). Enables "cutting" of large portfolio circuits into smaller sectors for parallel execution.
- **2026 Web UI Upgrades**: Added "Modular Partitioning" toggles, AI Auto-Clustering options, and JSON sector configuration fields to the advanced frontend settings.
- **ISA Compliance**: Automatic transpilation to Instruction Set Architecture (ISA) for IBM Quantum hardware. Ensures circuits submitted to `EstimatorV2` and `SamplerV2` are optimized for the physical target chip.
- **AI-Enhanced Transpilation**: Integrated 2026-era AI-driven circuit optimization to minimize gate depth and noise impact.
- **Hardware-Native Mitigation**: Solvers now automatically leverage cloud-side Resilience Level 2 (ZNE) and skip slow client-side gate folding when running on hardware.
- **IBM Cloud Support**: Explicit support for `ibm_cloud` channels and modern CRN-based authentication for U.S. East region hardware.
- **Job Management Utility**: Added `scripts/manage_jobs.py` for monitoring and cancelling active hardware jobs.

### Changed
- **Optimized Observable Construction**: Refactored `QUBOProblem.to_pauli` to use the 2026-standard `from_sparse_list` method, reducing classical memory overhead by up to 2x for large portfolios.
- **Functional Ansatz Builders**: Migrated the core ansatz library from deprecated classes (`RealAmplitudes`, `EfficientSU2`) to modern functional builders (`real_amplitudes()`, `efficient_su2()`) for future-proofing against Qiskit 3.0.
- **Improved V2 API Handling**: Updated IBM provider to use `mode` instead of the deprecated `session` keyword for primitive initialization.

### Fixed
- **Qiskit 1.0/2.0 Compatibility**: Resolved `ImportError` issues caused by removal of lowercase ansatz functions in recent Qiskit versions.
- **Hardware Connection Faults**: Fixed channel name validation errors (`ibm_quantum_platform`) for latest Runtime instances.
- **Security Scrub**: Removed all hardcoded API tokens and CRNs from the repository, transitioning to environment-variable-based authentication.

## [0.2.0] - 2026-02-20

### Added
- **CVaR-QAOA**: Conditional Value-at-Risk objective (`cvar_alpha` parameter, default 1.0).
  Evaluating only the lowest-energy tail fraction of shot measurements yields up to 4.5×
  faster convergence on noisy hardware (Barkoutsos et al. 2020).
- **XY-Mixer QAOA + Dicke state initialization**: Constraint-preserving mixer that
  strictly maintains portfolio cardinality without penalty tuning. Accepts `mixer_type="xy"`
  and `num_assets` (Bartschi & Eidenbenz 2019).
- **CVaR risk metric in QUBO**: `risk_metric="cvar"` flag in `PortfolioQUBO` for
  tail-risk-aware portfolio construction beyond Markowitz variance.
- **ESG constraints**: `esg_scores` and `esg_weight` parameters in `PortfolioQUBO`.
  Normalized ESG scores enter as negative linear terms, incentivizing high-ESG allocations.
- **Classical MIP baseline**: `mip_baseline()` in `classical_baseline.py`. Exhaustive
  subset enumeration (n ≤ 15) with QP weight optimization for each subset; greedy
  fallback for larger universes. Returns the globally optimal integer-constrained solution.
- **Provider-agnostic ZNE gate folding**: `simulation/zne.py` implements gate-level noise
  amplification and Richardson/linear/exponential extrapolation. Works on both local
  StatevectorSampler and IBM hardware. Demonstrated **31.6% energy improvement** on
  Heron r1 (February 2026).
- **Frontend Phase 4** (`frontend/app.py`):
  - Parses `cvar_alpha`, `mixer_type`, `num_assets_select`, `esg_scores`, `esg_weight`
  - Calls `mip_baseline()` on every run; result exposed in API response
  - ZNE gate-folding config threaded to local simulator backend
  - `mip_baseline`, `qaoa_settings`, and `esg` keys added to JSON response
- **Frontend UI Phase 4** (`frontend/templates/index.html`):
  - CVaR alpha slider (visible for QAOA only)
  - Mixer type dropdown (X-Mixer / XY-Mixer + Dicke state)
  - ESG scores text input + ESG weight slider
  - Assets-to-select (K) input for MIP baseline and XY-mixer cardinality
  - MIP column added to the Quantum vs Classical comparison table
- **GitHub Actions CI/CD**: `.github/workflows/` — automated testing on Python 3.9–3.12
- **Open-source governance**: `LICENSE` (MIT), `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`
- **Warm-start system** (`core/warm_start.py`): `WarmStartConfig`, `warm_start_vqe`,
  `warm_start_qaoa` — initializes circuit parameters from the Markowitz solution.
- **Quality scorer** (`postprocessing/quality_scorer.py`): Composite A–F grade based on
  Sharpe ratio, budget feasibility, return, and vs-classical performance.
- **Ansatz library** (`core/ansatz_library.py`): Named factory for RealAmplitudes and
  EfficientSU2 with configurable reps.

### Changed
- GitHub org placeholder `YOUR_ORG` replaced with `shymonski-dev` across README badges,
  `setup.py` project URLs, and `CONTRIBUTING.md`.
- Comparison table in the web UI now shows Markowitz, MIP, and vs-Markowitz diff columns.

### Fixed
- QAOA coefficient double-counting: symmetric QUBO off-diagonal terms were being doubled.
  Now uses `Q[i,j]` directly (not `Q[i,j] + Q[j,i]`).
- VQE empty-array bounds: `IndexError` when Estimator returned empty values replaced with
  descriptive `ValueError`.
- QUBO `__post_init__` enforces full matrix symmetry; asymmetric inputs are symmetrised.

## [0.1.0] - 2024-11-01

### Added
- QAOA solver with configurable layers (p=1, 2, …) and differential evolution optimizer
- VQE solver with RealAmplitudes/EfficientSU2 ansätze and differential evolution
- Binary QUBO formulation: Markowitz objective + budget constraint + transaction costs +
  multi-period support
- IBM Quantum hardware integration via Qiskit 2.x V2 primitives (EstimatorV2, SamplerV2)
- Error mitigation on IBM hardware: Dynamical Decoupling, Pauli Twirling,
  IBM-managed ZNE (Resilience Levels 0–2)
- Flask web frontend with dark-themed responsive UI
- CLI interface via click (`qpo` entry point)
- Classical Markowitz baseline (`scipy` SLSQP) for benchmarking
- Phase I and II benchmarking suites with result caching
- Yahoo Finance data integration with ticker validation
- Comprehensive test suite (271 tests)
- Multi-period portfolio optimization
- Multi-resolution qubit encoding (configurable allocation granularity)
- Unified constraint system (`core/constraints.py`):
  `EqualityConstraint`, `InequalityConstraint`, `BoundsConstraint`, `ConstraintManager`

[Unreleased]: https://github.com/shymonski-dev/quantum-portfolio-optimizer/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/shymonski-dev/quantum-portfolio-optimizer/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/shymonski-dev/quantum-portfolio-optimizer/releases/tag/v0.1.0

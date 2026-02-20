# Contributing to Quantum Portfolio Optimizer

Thank you for your interest in contributing to the Quantum Portfolio Optimizer! This project is a quantum-classical hybrid optimization research platform for portfolio construction, and we welcome contributions from quantum computing researchers, financial engineers, and software developers alike.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/shymonski-dev/quantum-portfolio-optimizer.git
cd quantum-portfolio-optimizer

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Optional: install noise simulation and IBM Quantum support
pip install -e ".[all]"
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src/

# Run a specific test file
pytest tests/test_vqe_solver.py -v

# Run a specific test
pytest tests/test_classical_baseline.py::TestMIPBaseline::test_mip_exactly_k_assets_selected -v
```

All tests must pass before submitting a pull request.

## Code Style

- **PEP 8**: All code must comply with PEP 8 style guidelines
- **Linting**: We use [ruff](https://docs.astral.sh/ruff/) for fast linting
  ```bash
  ruff check src/ tests/
  ```
- **Formatting**: We use [black](https://black.readthedocs.io/) for code formatting
  ```bash
  black src/ tests/
  ```
- Keep line length to 100 characters where practical
- Use type hints for function signatures

## Branch Naming

Use descriptive branch names with the following prefixes:

- `feature/short-description` -- New features (e.g., `feature/cvar-risk-metric`)
- `fix/issue-description` -- Bug fixes (e.g., `fix/qaoa-coefficient-scaling`)
- `docs/topic` -- Documentation changes (e.g., `docs/ibm-hardware-guide`)
- `test/description` -- Test additions or improvements
- `refactor/description` -- Code refactoring without behavior changes

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear, focused commits
3. Ensure all tests pass (`pytest tests/ -v`)
4. Add tests for any new features or bug fixes
5. Update `CHANGELOG.md` under the `[Unreleased]` section
6. Submit a pull request with a clear description of changes
7. Address any review feedback

### PR Checklist

- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Code follows project style guidelines (ruff, black)
- [ ] CHANGELOG.md updated
- [ ] Documentation updated if needed

## Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <description>

[optional body]
```

Types:
- `feat:` -- New feature (e.g., `feat: add CVaR risk metric to QUBO formulation`)
- `fix:` -- Bug fix (e.g., `fix: correct QAOA coefficient scaling for multi-asset`)
- `docs:` -- Documentation only (e.g., `docs: update IBM hardware setup guide`)
- `test:` -- Adding or updating tests
- `refactor:` -- Code change that neither fixes a bug nor adds a feature
- `perf:` -- Performance improvement
- `ci:` -- CI/CD configuration changes

## Issue Reporting

- Use the GitHub issue templates for bug reports and feature requests
- For bugs, include: steps to reproduce, expected behavior, actual behavior, and environment details
- For features, include: problem statement, proposed solution, and relevant references

## Adding New Algorithms

To add a new quantum or classical solver:

1. **Implement the solver**: Create a new file in `src/quantum_portfolio_optimizer/core/` following the pattern of `vqe_solver.py` or `qaoa_solver.py`. Your solver should accept a QUBO matrix and return optimization results.

2. **Add to provider factory**: Register the new algorithm in the provider/selector so it can be chosen via configuration.

3. **Add tests**: Create a corresponding test file in `tests/` with unit tests covering:
   - Basic functionality with small problems (3-5 assets)
   - Edge cases (single asset, maximum assets)
   - Parameter validation
   - Result format correctness

4. **Update documentation**: Add the algorithm to README.md and note it in CHANGELOG.md.

5. **Benchmark**: If applicable, add benchmark comparisons against existing solvers.

## Questions?

Open a GitHub issue or start a GitHub Discussion for any questions about contributing.

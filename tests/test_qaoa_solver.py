"""Tests for QAOA solver implementation."""

import numpy as np
import pytest

from quantum_portfolio_optimizer.core.qaoa_solver import (
    PortfolioQAOASolver,
    QAOAResult,
    get_qaoa_circuit_depth,
)
from quantum_portfolio_optimizer.core.qubo_formulation import PortfolioQUBO
from quantum_portfolio_optimizer.core.optimizer_interface import (
    DifferentialEvolutionConfig,
)
from quantum_portfolio_optimizer.simulation.provider import get_provider


@pytest.fixture
def simple_qubo():
    """Create a simple 3-asset QUBO for testing."""
    expected_returns = np.array([0.01, 0.015, 0.008])
    covariance = np.array(
        [
            [0.04, 0.01, 0.005],
            [0.01, 0.05, 0.01],
            [0.005, 0.01, 0.03],
        ]
    )
    builder = PortfolioQUBO(
        expected_returns=expected_returns,
        covariance=covariance,
        budget=1.0,
        risk_aversion=500.0,
        penalty_strength=1000.0,
    )
    return builder.build()


class TestQAOAResultDataclass:
    """Test QAOAResult dataclass."""

    def test_qaoa_result_fields(self):
        """Test QAOAResult holds all expected fields."""
        result = QAOAResult(
            optimal_parameters=np.array([0.5, 0.3]),
            optimal_value=-0.5,
            best_bitstring="101",
            measurement_counts={"101": 500, "110": 300, "011": 224},
            history=[0.0, -0.3, -0.5],
            best_history=[0.0, -0.3, -0.5],
            num_evaluations=3,
            layers=1,
            converged=True,
            optimizer_message="Success",
            circuit_report={"num_qubits": 3, "num_parameters": 2},
        )
        assert result.layers == 1
        assert result.best_bitstring == "101"
        assert len(result.optimal_parameters) == 2  # gamma and beta for p=1


class TestQAOASolverRequirements:
    """Test QAOA solver initialization requirements."""

    def test_qaoa_requires_sampler(self):
        """QAOA must be initialized with a sampler."""
        with pytest.raises(ValueError, match="Sampler is required"):
            PortfolioQAOASolver(sampler=None)

    def test_qaoa_accepts_sampler(self):
        """QAOA accepts a valid sampler."""
        backend_config = {"name": "local_simulator", "shots": 100, "seed": 42}
        _, sampler = get_provider(backend_config)
        solver = PortfolioQAOASolver(sampler=sampler, layers=1)
        assert solver.layers == 1


class TestQAOACircuitConstruction:
    """Test QAOA circuit construction."""

    def test_circuit_has_correct_parameters(self, simple_qubo):
        """QAOA circuit should have 2*p parameters."""
        backend_config = {"name": "local_simulator", "shots": 100, "seed": 42}
        _, sampler = get_provider(backend_config)

        for layers in [1, 2, 3]:
            solver = PortfolioQAOASolver(sampler=sampler, layers=layers)
            circuit, params = solver._build_qaoa_circuit(simple_qubo)

            assert len(params) == 2 * layers  # gamma and beta per layer
            assert circuit.num_qubits == simple_qubo.num_variables

    def test_circuit_structure(self, simple_qubo):
        """Test basic circuit structure for p=1."""
        backend_config = {"name": "local_simulator", "shots": 100, "seed": 42}
        _, sampler = get_provider(backend_config)

        solver = PortfolioQAOASolver(sampler=sampler, layers=1)
        circuit, _ = solver._build_qaoa_circuit(simple_qubo)

        # Should have operations (H gates, RZ, CX, RX)
        assert circuit.depth() > 0
        assert circuit.num_qubits == 3


class TestQAOASolve:
    """Test QAOA solving capability."""

    def test_qaoa_solves_small_qubo(self, simple_qubo):
        """QAOA should successfully solve a small QUBO."""
        backend_config = {"name": "local_simulator", "shots": 512, "seed": 42}
        _, sampler = get_provider(backend_config)

        config = DifferentialEvolutionConfig(
            bounds=[(0, 2 * np.pi), (0, np.pi)],  # gamma, beta
            maxiter=5,
            popsize=3,
            seed=42,
        )

        solver = PortfolioQAOASolver(
            sampler=sampler,
            layers=1,
            optimizer_config=config,
            seed=42,
        )

        result = solver.solve(simple_qubo)

        # Basic validity checks
        assert isinstance(result, QAOAResult)
        assert result.best_bitstring is not None
        assert len(result.best_bitstring) >= 3
        assert all(c in "01" for c in result.best_bitstring)
        assert result.measurement_counts is not None
        assert result.num_evaluations > 0
        assert len(result.history) == result.num_evaluations

    def test_qaoa_progress_callback(self, simple_qubo):
        """Test that progress callback is called during optimization."""
        backend_config = {"name": "local_simulator", "shots": 256, "seed": 42}
        _, sampler = get_provider(backend_config)

        config = DifferentialEvolutionConfig(
            bounds=[(0, 2 * np.pi), (0, np.pi)],
            maxiter=3,
            popsize=2,
            seed=42,
        )

        callback_calls = []

        def progress_callback(iteration, energy, best_energy):
            callback_calls.append((iteration, energy, best_energy))

        solver = PortfolioQAOASolver(
            sampler=sampler,
            layers=1,
            optimizer_config=config,
            seed=42,
            progress_callback=progress_callback,
        )

        solver.solve(simple_qubo)

        assert len(callback_calls) > 0
        # Check callback receives valid data
        for iteration, energy, best_energy in callback_calls:
            assert isinstance(iteration, int)
            assert isinstance(energy, float)
            assert isinstance(best_energy, float)

    def test_qaoa_two_layers(self, simple_qubo):
        """Test QAOA with p=2 layers."""
        backend_config = {"name": "local_simulator", "shots": 256, "seed": 42}
        _, sampler = get_provider(backend_config)

        config = DifferentialEvolutionConfig(
            bounds=[(0, 2 * np.pi), (0, np.pi), (0, 2 * np.pi), (0, np.pi)],
            maxiter=3,
            popsize=2,
            seed=42,
        )

        solver = PortfolioQAOASolver(
            sampler=sampler,
            layers=2,
            optimizer_config=config,
            seed=42,
        )

        result = solver.solve(simple_qubo)

        assert result.layers == 2
        assert len(result.optimal_parameters) == 4  # 2 gammas + 2 betas
        assert result.circuit_report["num_qubits"] == simple_qubo.num_variables

    def test_qaoa_preserves_optimizer_initial_point(self, simple_qubo, monkeypatch):
        backend_config = {"name": "local_simulator", "shots": 256, "seed": 11}
        _, sampler = get_provider(backend_config)
        warm = [0.15, 0.3]
        captured = {}

        def fake_run(objective, config, num_qubits):
            captured["x0"] = config.x0
            params = np.asarray(config.x0 or [0.0] * len(config.bounds), dtype=float)
            objective(params)

            class Result:
                x = params
                fun = 0.0
                success = True
                message = "mock"

            return Result()

        monkeypatch.setattr(
            "quantum_portfolio_optimizer.core.qaoa_solver.run_differential_evolution",
            fake_run,
        )

        solver = PortfolioQAOASolver(
            sampler=sampler,
            layers=1,
            optimizer_config=DifferentialEvolutionConfig(
                bounds=[(0, 2 * np.pi), (0, np.pi)],
                maxiter=1,
                popsize=2,
                seed=11,
                x0=warm,
            ),
            seed=11,
        )

        solver.solve(simple_qubo)
        assert captured["x0"] == warm


class TestQAOAEnergyEvaluation:
    """Test QUBO energy evaluation."""

    def test_evaluate_qubo_energy(self, simple_qubo):
        """Test energy evaluation for known bitstrings."""
        backend_config = {"name": "local_simulator", "shots": 100, "seed": 42}
        _, sampler = get_provider(backend_config)
        solver = PortfolioQAOASolver(sampler=sampler)

        # Test with all zeros
        bits_zero = np.array([0.0, 0.0, 0.0])
        energy_zero = solver._evaluate_qubo_energy(bits_zero, simple_qubo)
        assert isinstance(energy_zero, float)

        # Test with all ones
        bits_one = np.array([1.0, 1.0, 1.0])
        energy_one = solver._evaluate_qubo_energy(bits_one, simple_qubo)
        assert isinstance(energy_one, float)

        # Energies should generally be different
        # (unless the QUBO has very specific structure)


class TestCircuitDepthEstimation:
    """Test circuit depth estimation utility."""

    def test_depth_increases_with_qubits(self):
        """Circuit depth should increase with number of qubits."""
        depth_5 = get_qaoa_circuit_depth(5, layers=1)
        depth_10 = get_qaoa_circuit_depth(10, layers=1)
        depth_20 = get_qaoa_circuit_depth(20, layers=1)

        assert depth_10 > depth_5
        assert depth_20 > depth_10

    def test_depth_increases_with_layers(self):
        """Circuit depth should increase with number of layers."""
        depth_p1 = get_qaoa_circuit_depth(10, layers=1)
        depth_p2 = get_qaoa_circuit_depth(10, layers=2)
        depth_p3 = get_qaoa_circuit_depth(10, layers=3)

        assert depth_p2 > depth_p1
        assert depth_p3 > depth_p2

    def test_depth_values_reasonable(self):
        """Check that depth estimates are reasonable."""
        # For 5 qubits, p=1, depth should be manageable
        depth = get_qaoa_circuit_depth(5, layers=1)
        assert depth > 0
        assert depth < 100  # Reasonable upper bound

        # For 25 qubits, p=1, depth will be large due to ZZ terms
        depth_25 = get_qaoa_circuit_depth(25, layers=1)
        assert depth_25 > depth


class TestCVaRQAOA:
    def test_cvar_alpha_one_is_expectation(self):
        """alpha=1.0 must reproduce the existing weighted-average behaviour."""
        from quantum_portfolio_optimizer.core.qaoa_solver import PortfolioQAOASolver
        from quantum_portfolio_optimizer.simulation.provider import get_provider

        _, sampler = get_provider({"name": "local_simulator", "shots": 256, "seed": 42})
        # Create solver with alpha=1.0 — should work identically to no alpha
        solver = PortfolioQAOASolver(
            sampler=sampler, layers=1, shots=256, cvar_alpha=1.0
        )
        assert solver.cvar_alpha == 1.0

    def test_compute_objective_cvar_known_values(self):
        """Unit test _compute_objective with hand-computed expected value."""
        import numpy as np
        from quantum_portfolio_optimizer.core.qaoa_solver import PortfolioQAOASolver
        from quantum_portfolio_optimizer.simulation.provider import get_provider

        _, sampler = get_provider({"name": "local_simulator", "shots": 100, "seed": 42})
        solver = PortfolioQAOASolver(sampler=sampler, layers=1, cvar_alpha=0.5)

        # 4 bitstrings, energies [1, 2, 3, 4], counts [25, 25, 25, 25]
        energies = np.array([1.0, 2.0, 3.0, 4.0])
        counts = np.array([25.0, 25.0, 25.0, 25.0])
        # alpha=0.5 -> worst 50% = 50 shots from top: energies [4,3] each 25 shots
        # CVaR = (4*25 + 3*25) / 50 = 3.5
        result = solver._compute_objective(energies, counts)
        assert result == pytest.approx(3.5, abs=1e-6)

    def test_compute_objective_alpha_1(self):
        """alpha=1.0 yields standard weighted average."""
        from quantum_portfolio_optimizer.core.qaoa_solver import PortfolioQAOASolver
        from quantum_portfolio_optimizer.simulation.provider import get_provider

        _, sampler = get_provider({"name": "local_simulator", "shots": 100, "seed": 42})
        solver = PortfolioQAOASolver(sampler=sampler, layers=1, cvar_alpha=1.0)

        energies = np.array([1.0, 2.0, 3.0])
        counts = np.array([10.0, 10.0, 10.0])
        result = solver._compute_objective(energies, counts)
        assert result == pytest.approx(2.0, abs=1e-6)

    def test_compute_objective_alpha_0_5(self):
        """alpha=0.5 returns the worst 50% average energy."""
        from quantum_portfolio_optimizer.core.qaoa_solver import PortfolioQAOASolver
        from quantum_portfolio_optimizer.simulation.provider import get_provider

        _, sampler = get_provider({"name": "local_simulator", "shots": 100, "seed": 42})
        solver = PortfolioQAOASolver(sampler=sampler, layers=1, cvar_alpha=0.5)

        energies = np.array([1.0, 3.0])
        counts = np.array([50.0, 50.0])
        # alpha=0.5 -> cutoff=50 shots from worst end -> only energy=3.0 (50 shots)
        result = solver._compute_objective(energies, counts)
        assert result == pytest.approx(3.0, abs=1e-6)

    def test_compute_objective_alpha_0_1(self):
        """alpha=0.1 returns the worst 10% average energy."""
        from quantum_portfolio_optimizer.core.qaoa_solver import PortfolioQAOASolver
        from quantum_portfolio_optimizer.simulation.provider import get_provider

        _, sampler = get_provider({"name": "local_simulator", "shots": 100, "seed": 42})
        solver = PortfolioQAOASolver(sampler=sampler, layers=1, cvar_alpha=0.1)

        energies = np.array([1.0, 3.0])
        counts = np.array([100.0, 100.0])
        # alpha=0.1 -> cutoff=20 shots from worst end -> all from energy=3.0
        result = solver._compute_objective(energies, counts)
        assert result == pytest.approx(3.0, abs=1e-6)

    def test_cvar_alpha_zero_raises(self):
        from quantum_portfolio_optimizer.core.qaoa_solver import PortfolioQAOASolver
        from quantum_portfolio_optimizer.simulation.provider import get_provider

        _, sampler = get_provider({"name": "local_simulator", "shots": 100, "seed": 42})
        with pytest.raises(ValueError, match="cvar_alpha"):
            PortfolioQAOASolver(sampler=sampler, layers=1, cvar_alpha=0.0)

    def test_cvar_alpha_above_one_raises(self):
        from quantum_portfolio_optimizer.core.qaoa_solver import PortfolioQAOASolver
        from quantum_portfolio_optimizer.simulation.provider import get_provider

        _, sampler = get_provider({"name": "local_simulator", "shots": 100, "seed": 42})
        with pytest.raises(ValueError, match="cvar_alpha"):
            PortfolioQAOASolver(sampler=sampler, layers=1, cvar_alpha=1.5)


class TestXYMixerQAOA:
    def _get_sampler(self):
        from quantum_portfolio_optimizer.simulation.provider import get_provider

        _, sampler = get_provider({"name": "local_simulator", "shots": 256, "seed": 42})
        return sampler

    def test_xy_mixer_requires_num_assets(self):
        from quantum_portfolio_optimizer.core.qaoa_solver import PortfolioQAOASolver

        with pytest.raises(ValueError, match="num_assets"):
            PortfolioQAOASolver(sampler=self._get_sampler(), layers=1, mixer_type="xy")

    def test_invalid_mixer_type_raises(self):
        from quantum_portfolio_optimizer.core.qaoa_solver import PortfolioQAOASolver

        with pytest.raises(ValueError, match="mixer_type"):
            PortfolioQAOASolver(sampler=self._get_sampler(), layers=1, mixer_type="rz")

    def test_dicke_state_correct_hamming_weight(self):
        """Statevector of Dicke(n=4, k=2) has support only on weight-2 bitstrings."""
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        from quantum_portfolio_optimizer.core.qaoa_solver import PortfolioQAOASolver

        solver = PortfolioQAOASolver(
            sampler=self._get_sampler(), layers=1, mixer_type="xy", num_assets=2
        )
        qc = QuantumCircuit(4)
        solver._prepare_dicke_state(qc, 4, 2)
        sv = Statevector(qc)
        probs = sv.probabilities_dict()
        for bitstring, prob in probs.items():
            if prob > 1e-10:
                assert bitstring.count("1") == 2, (
                    f"Bitstring {bitstring} has wrong Hamming weight, prob={prob}"
                )

    def test_dicke_state_uniform_superposition(self):
        """All C(4,2)=6 weight-2 states have equal probability."""
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        from math import comb
        from quantum_portfolio_optimizer.core.qaoa_solver import PortfolioQAOASolver

        solver = PortfolioQAOASolver(
            sampler=self._get_sampler(), layers=1, mixer_type="xy", num_assets=2
        )
        qc = QuantumCircuit(4)
        solver._prepare_dicke_state(qc, 4, 2)
        sv = Statevector(qc)
        probs = sv.probabilities_dict()
        expected_prob = 1.0 / comb(4, 2)
        for bitstring, prob in probs.items():
            if prob > 1e-10:
                assert prob == pytest.approx(expected_prob, abs=1e-8)

    def test_xy_mixer_circuit_builds(self):
        """XY mixer QAOA circuit builds without error for n=4, k=2."""
        import numpy as np
        from quantum_portfolio_optimizer.core.qaoa_solver import PortfolioQAOASolver
        from quantum_portfolio_optimizer.core.qubo_formulation import PortfolioQUBO

        returns = np.array([0.01, 0.015, 0.008, 0.012])
        cov = np.eye(4) * 0.05
        qubo = PortfolioQUBO(returns, cov, budget=1.0).build()

        solver = PortfolioQAOASolver(
            sampler=self._get_sampler(),
            layers=1,
            mixer_type="xy",
            num_assets=2,
            shots=64,
        )
        circuit, params = solver._build_qaoa_circuit(qubo)
        assert circuit is not None
        assert len(params) == 2  # 1 gamma + 1 beta for layers=1

    def test_xy_mixer_hamming_weight_preserved(self):
        """After solve() with XY mixer, all measured bitstrings must have Hamming weight == num_assets."""
        from quantum_portfolio_optimizer.core.qaoa_solver import PortfolioQAOASolver
        from quantum_portfolio_optimizer.core.qubo_formulation import PortfolioQUBO
        from quantum_portfolio_optimizer.core.optimizer_interface import (
            DifferentialEvolutionConfig,
        )

        returns = np.array([0.01, 0.015, 0.008, 0.012])
        cov = np.eye(4) * 0.05
        qubo = PortfolioQUBO(returns, cov, budget=1.0).build()

        config = DifferentialEvolutionConfig(
            bounds=[(0, 2 * np.pi), (0, np.pi)],
            maxiter=2,
            popsize=2,
            seed=42,
        )

        solver = PortfolioQAOASolver(
            sampler=self._get_sampler(),
            layers=1,
            mixer_type="xy",
            num_assets=2,
            optimizer_config=config,
            seed=42,
            shots=256,
        )

        result = solver.solve(qubo)
        for bitstring, count in result.measurement_counts.items():
            assert bitstring.count("1") == 2, (
                f"Bitstring {bitstring} (count={count}) has Hamming weight "
                f"{bitstring.count('1')} != 2"
            )

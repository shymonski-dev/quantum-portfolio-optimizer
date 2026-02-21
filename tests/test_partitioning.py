"""Tests for circuit partitioning and knitting (2026 Modular Hardware support)."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorSampler, StatevectorEstimator

from quantum_portfolio_optimizer.core.qubo_formulation import PortfolioQUBO
from quantum_portfolio_optimizer.simulation.partitioning import (
    run_partitioned_vqe_step,
    CUTTING_AVAILABLE,
)
from quantum_portfolio_optimizer.core.vqe_solver import PortfolioVQESolver


@pytest.mark.skipif(not CUTTING_AVAILABLE, reason="qiskit-addon-cutting not installed")
class TestCircuitPartitioning:
    """Test suite for modular hardware partitioning logic."""

    def test_qubo_sector_to_partition_mapping(self):
        """Verify sectors are correctly mapped to qubit indices."""
        # 4 assets, 2 sectors, 1 bit per asset
        sectors = {"tech": [0, 1], "energy": [2, 3]}
        mu = np.array([0.1, 0.12, 0.08, 0.09])
        cov = np.eye(4) * 0.05

        builder = PortfolioQUBO(
            expected_returns=mu, covariance=cov, budget=1.0, sectors=sectors
        )
        qubo = builder.build()

        partitions = qubo.metadata.get("partitions")
        assert len(partitions) == 2
        # Tech qubits: 0, 1
        # Energy qubits: 2, 3
        assert set(partitions[0]) == {0, 1}
        assert set(partitions[1]) == {2, 3}

    def test_partition_logic_identifies_cuts(self):
        """Check if gates spanning across sectors are identified for cutting."""
        from qiskit_addon_cutting import partition_problem

        qc = QuantumCircuit(4)
        qc.cx(0, 1)  # Internal to partition 0
        qc.cx(2, 3)  # Internal to partition 1
        qc.cx(1, 2)  # CROSS-PARTITION gate (cut needed)

        partition_labels = [0, 0, 1, 1]
        problem = partition_problem(qc, partition_labels)

        # Verify that subcircuits are created
        assert len(problem.subcircuits) == 2
        # Check that the subcircuits have the expected number of qubits
        assert problem.subcircuits[0].num_qubits == 2
        assert problem.subcircuits[1].num_qubits == 2

    def test_partitioned_vqe_step_equivalence(self):
        """Verify that partitioned evaluation matches standard evaluation on a simulator."""
        from qiskit.primitives import StatevectorEstimator
        from qiskit_aer.primitives import SamplerV2 as AerSampler

        # Small 2-qubit problem
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)  # Entangled state

        observable = SparsePauliOp.from_list([("ZZ", 1.0)])
        partitions = [[0], [1]]  # Cut the CX gate

        sampler = AerSampler()
        estimator = StatevectorEstimator()

        # Standard energy
        standard_energy = estimator.run([(qc, observable)]).result()[0].data.evs

        # Partitioned energy
        # Note: we use the sampler for sub-experiments in the partitioning module
        partitioned_energy = run_partitioned_vqe_step(
            sampler, qc, observable, partitions
        )

        # Should be mathematically equivalent (within sampling noise tolerance)
        assert np.isclose(standard_energy, partitioned_energy, atol=0.1)

    def test_solver_integration(self):
        """Ensure PortfolioVQESolver utilizes partitioning logic when enabled."""
        mu = np.array([0.1, 0.15])
        cov = np.eye(2) * 0.04
        sectors = {"A": [0], "B": [1]}

        builder = PortfolioQUBO(
            expected_returns=mu, covariance=cov, budget=1.0, sectors=sectors
        )
        qubo = builder.build()

        estimator = StatevectorEstimator()
        sampler = StatevectorSampler()
        solver = PortfolioVQESolver(
            estimator=estimator,
            sampler=sampler,
            use_partitioning=True,
            ansatz_name="ra",
            ansatz_options={"reps": 1},
        )

        # Mock the partitioning call to verify it's hit
        from unittest.mock import patch

        with patch(
            "quantum_portfolio_optimizer.core.vqe_solver.run_partitioned_vqe_step"
        ) as mock_part:
            mock_part.return_value = -0.5

            # Limit optimizer to force a quick call
            from quantum_portfolio_optimizer.core.optimizer_interface import (
                DifferentialEvolutionConfig,
            )

            solver.optimizer_config = DifferentialEvolutionConfig(
                bounds=[(0, 1)] * 4, maxiter=1, popsize=1
            )

            solver.solve(qubo)
            assert mock_part.called

"""Tests for IBM Quantum provider module with mocked qiskit-ibm-runtime."""

from unittest.mock import MagicMock, patch, PropertyMock
import pytest

from quantum_portfolio_optimizer.exceptions import (
    BackendError,
    IBMAuthenticationError,
    IBMBackendNotFoundError,
)


class TestIBMProviderWithoutRuntime:
    """Test IBM provider behavior when qiskit-ibm-runtime is not available."""

    def test_get_runtime_service_raises_backend_error(self):
        """Should raise BackendError when qiskit-ibm-runtime not installed."""
        with patch.dict("sys.modules", {"qiskit_ibm_runtime": None}):
            # Need to reload the module to pick up the mocked import
            import importlib
            from quantum_portfolio_optimizer.simulation import ibm_provider

            # Force IBM_RUNTIME_AVAILABLE to False
            original_available = ibm_provider.IBM_RUNTIME_AVAILABLE
            ibm_provider.IBM_RUNTIME_AVAILABLE = False

            try:
                with pytest.raises(BackendError, match="qiskit-ibm-runtime"):
                    ibm_provider._get_runtime_service()
            finally:
                ibm_provider.IBM_RUNTIME_AVAILABLE = original_available

    def test_get_ibm_quantum_backend_raises_backend_error(self):
        """Should raise BackendError when qiskit-ibm-runtime not installed."""
        from quantum_portfolio_optimizer.simulation import ibm_provider

        original_available = ibm_provider.IBM_RUNTIME_AVAILABLE
        ibm_provider.IBM_RUNTIME_AVAILABLE = False

        try:
            with pytest.raises(BackendError, match="qiskit-ibm-runtime"):
                ibm_provider.get_ibm_quantum_backend({"device": "ibm_test"})
        finally:
            ibm_provider.IBM_RUNTIME_AVAILABLE = original_available


class TestIBMProviderConfigValidation:
    """Test IBM provider configuration validation."""

    def test_missing_device_raises_error(self):
        """Should raise BackendError when device not specified."""
        from quantum_portfolio_optimizer.simulation import ibm_provider

        # Only test if runtime is available
        if not ibm_provider.IBM_RUNTIME_AVAILABLE:
            pytest.skip("qiskit-ibm-runtime not installed")

        with pytest.raises(BackendError, match="device"):
            ibm_provider.get_ibm_quantum_backend({})

    def test_empty_device_raises_error(self):
        """Should raise BackendError when device is empty string."""
        from quantum_portfolio_optimizer.simulation import ibm_provider

        if not ibm_provider.IBM_RUNTIME_AVAILABLE:
            pytest.skip("qiskit-ibm-runtime not installed")

        with pytest.raises(BackendError, match="device"):
            ibm_provider.get_ibm_quantum_backend({"device": ""})


class TestIBMProviderWithMockedRuntime:
    """Test IBM provider with mocked qiskit-ibm-runtime."""

    @pytest.fixture
    def mock_runtime(self):
        """Create mocked qiskit-ibm-runtime components."""
        with patch("quantum_portfolio_optimizer.simulation.ibm_provider.QiskitRuntimeService") as mock_service, \
             patch("quantum_portfolio_optimizer.simulation.ibm_provider.EstimatorV2") as mock_estimator, \
             patch("quantum_portfolio_optimizer.simulation.ibm_provider.SamplerV2") as mock_sampler, \
             patch("quantum_portfolio_optimizer.simulation.ibm_provider.Session") as mock_session, \
             patch("quantum_portfolio_optimizer.simulation.ibm_provider.EstimatorOptions") as mock_est_opts, \
             patch("quantum_portfolio_optimizer.simulation.ibm_provider.SamplerOptions") as mock_samp_opts, \
             patch("quantum_portfolio_optimizer.simulation.ibm_provider.IBM_RUNTIME_AVAILABLE", True):

            # Configure mock service
            mock_backend = MagicMock()
            mock_backend.name = "ibm_test"
            mock_backend.num_qubits = 127
            mock_backend.status.return_value.status_msg = "online"

            mock_service_instance = MagicMock()
            mock_service_instance.backend.return_value = mock_backend
            mock_service_instance.backends.return_value = [mock_backend]
            mock_service.return_value = mock_service_instance

            # Configure mock session
            mock_session_instance = MagicMock()
            mock_session_instance.status.return_value = "Active"
            mock_session.return_value = mock_session_instance

            # Configure mock options
            mock_est_opts_instance = MagicMock()
            mock_est_opts.return_value = mock_est_opts_instance

            mock_samp_opts_instance = MagicMock()
            mock_samp_opts.return_value = mock_samp_opts_instance

            yield {
                "service": mock_service,
                "service_instance": mock_service_instance,
                "estimator": mock_estimator,
                "sampler": mock_sampler,
                "session": mock_session,
                "session_instance": mock_session_instance,
                "backend": mock_backend,
                "est_opts": mock_est_opts,
                "samp_opts": mock_samp_opts,
            }

    def test_successful_connection(self, mock_runtime):
        """Should successfully create estimator and sampler."""
        from quantum_portfolio_optimizer.simulation import ibm_provider

        # Reset session state
        ibm_provider._active_session = None
        ibm_provider._session_backend = None

        config = {
            "device": "ibm_test",
            "channel": "ibm_quantum",
            "shots": 1024,
        }

        with patch.object(ibm_provider, "IBM_RUNTIME_AVAILABLE", True):
            estimator, sampler = ibm_provider.get_ibm_quantum_backend(config)

        assert estimator is not None
        assert sampler is not None
        mock_runtime["service"].assert_called()

    def test_authentication_error(self, mock_runtime):
        """Should raise IBMAuthenticationError on auth failure."""
        from quantum_portfolio_optimizer.simulation import ibm_provider

        mock_runtime["service"].side_effect = Exception("Invalid token")

        with patch.object(ibm_provider, "IBM_RUNTIME_AVAILABLE", True):
            with pytest.raises(IBMAuthenticationError, match="token"):
                ibm_provider._get_runtime_service(token="bad_token")

    def test_backend_not_found_error(self, mock_runtime):
        """Should raise IBMBackendNotFoundError when backend doesn't exist."""
        from quantum_portfolio_optimizer.simulation import ibm_provider

        # Reset session state
        ibm_provider._active_session = None
        ibm_provider._session_backend = None

        # Configure backend lookup to fail
        mock_runtime["service_instance"].backend.side_effect = Exception("Backend not found")

        # Configure available backends
        mock_available = MagicMock()
        mock_available.name = "ibm_brisbane"
        mock_runtime["service_instance"].backends.return_value = [mock_available]

        config = {
            "device": "ibm_nonexistent",
            "channel": "ibm_quantum",
        }

        with patch.object(ibm_provider, "IBM_RUNTIME_AVAILABLE", True):
            with pytest.raises(IBMBackendNotFoundError) as exc_info:
                ibm_provider.get_ibm_quantum_backend(config)

            assert exc_info.value.backend_name == "ibm_nonexistent"
            assert "ibm_brisbane" in exc_info.value.available_backends

    def test_session_reuse(self, mock_runtime):
        """Should reuse existing session for same backend."""
        from quantum_portfolio_optimizer.simulation import ibm_provider

        # Reset session state
        ibm_provider._active_session = None
        ibm_provider._session_backend = None

        config = {
            "device": "ibm_test",
            "use_session": True,
        }

        with patch.object(ibm_provider, "IBM_RUNTIME_AVAILABLE", True):
            # First call creates session
            ibm_provider.get_ibm_quantum_backend(config)
            first_call_count = mock_runtime["session"].call_count

            # Simulate active session
            ibm_provider._active_session = mock_runtime["session_instance"]
            ibm_provider._session_backend = "ibm_test"

            # Second call should reuse session (if status is Active)
            ibm_provider.get_ibm_quantum_backend(config)
            second_call_count = mock_runtime["session"].call_count

            # Session constructor should not be called again
            assert second_call_count == first_call_count

    def test_session_close(self, mock_runtime):
        """Should close session properly."""
        from quantum_portfolio_optimizer.simulation import ibm_provider

        # Set up active session
        ibm_provider._active_session = mock_runtime["session_instance"]
        ibm_provider._session_backend = "ibm_test"

        ibm_provider.close_session()

        mock_runtime["session_instance"].close.assert_called_once()
        assert ibm_provider._active_session is None
        assert ibm_provider._session_backend is None


class TestErrorMitigationConfig:
    """Test error mitigation configuration."""

    def test_default_config_values(self):
        """ErrorMitigationConfig should have sensible defaults."""
        from quantum_portfolio_optimizer.simulation.ibm_provider import ErrorMitigationConfig

        config = ErrorMitigationConfig()

        assert config.zne_enabled is False
        assert config.dynamical_decoupling is True
        assert config.twirling_enabled is True
        assert config.resilience_level == 1

    def test_custom_config_values(self):
        """ErrorMitigationConfig should accept custom values."""
        from quantum_portfolio_optimizer.simulation.ibm_provider import ErrorMitigationConfig

        config = ErrorMitigationConfig(
            zne_enabled=True,
            zne_noise_factors=(1, 2, 3),
            zne_extrapolator="linear",
            dynamical_decoupling=False,
            dd_sequence="XY4",
            twirling_enabled=False,
            resilience_level=2,
        )

        assert config.zne_enabled is True
        assert config.zne_noise_factors == (1, 2, 3)
        assert config.zne_extrapolator == "linear"
        assert config.dynamical_decoupling is False
        assert config.dd_sequence == "XY4"
        assert config.twirling_enabled is False
        assert config.resilience_level == 2


class TestIBMQuantumConfig:
    """Test IBM Quantum configuration."""

    def test_default_config_values(self):
        """IBMQuantumConfig should have sensible defaults."""
        from quantum_portfolio_optimizer.simulation.ibm_provider import IBMQuantumConfig

        config = IBMQuantumConfig()

        assert config.device == "ibm_brisbane"
        assert config.channel == "ibm_quantum"
        assert config.use_session is True
        assert config.shots == 4096
        assert config.optimization_level == 3

    def test_custom_config_values(self):
        """IBMQuantumConfig should accept custom values."""
        from quantum_portfolio_optimizer.simulation.ibm_provider import (
            IBMQuantumConfig,
            ErrorMitigationConfig,
        )

        em_config = ErrorMitigationConfig(resilience_level=2)
        config = IBMQuantumConfig(
            device="ibm_kyoto",
            channel="ibm_cloud",
            instance="hub/group/project",
            use_session=False,
            shots=8192,
            optimization_level=2,
            error_mitigation=em_config,
        )

        assert config.device == "ibm_kyoto"
        assert config.channel == "ibm_cloud"
        assert config.instance == "hub/group/project"
        assert config.use_session is False
        assert config.shots == 8192
        assert config.optimization_level == 2
        assert config.error_mitigation.resilience_level == 2

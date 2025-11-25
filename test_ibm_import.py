# quantum_portfolio_optimizer/test_ibm_import.py
try:
    from qiskit_ibm_provider import IBMProvider
    print("IBMProvider imported successfully.")
except ImportError as e:
    print(f"Failed to import IBMProvider: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

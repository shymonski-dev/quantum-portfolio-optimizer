from qiskit_ibm_runtime import QiskitRuntimeService
import os

def list_pending_jobs():
    # Use environment variables for security
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    instance = os.environ.get("IBM_CLOUD_INSTANCE")
    
    if not token:
        print("Error: IBM_QUANTUM_TOKEN environment variable not set.")
        return
    
    # Instance is optional if default is saved, but recommended for cloud channel
    service = QiskitRuntimeService(channel='ibm_cloud', token=token, instance=instance)
    
    jobs = service.jobs(limit=5)
    if not jobs:
        print("No jobs found.")
        return

    for job in jobs:
        print(f"ID: {job.job_id()} | Backend: {job.backend().name} | Status: {job.status()}")

if __name__ == "__main__":
    list_pending_jobs()

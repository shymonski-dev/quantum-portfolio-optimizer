from qiskit_ibm_runtime import QiskitRuntimeService
import os

def list_pending_jobs():
    token = "qX6dC_xUh6g7RDTHgcagMZi4bauvfYobJsXPc0LuKfcD"
    instance = "crn:v1:bluemix:public:quantum-computing:us-east:a/2e9879e40b034feaa9dd3a6c000673f5:bb92b7f6-559f-4453-b7fd-0f21652093ae::"
    
    service = QiskitRuntimeService(channel='ibm_cloud', token=token, instance=instance)
    
    jobs = service.jobs(limit=5)
    if not jobs:
        print("No jobs found.")
        return

    for job in jobs:
        print(f"ID: {job.job_id()} | Backend: {job.backend().name} | Status: {job.status()}")

if __name__ == "__main__":
    list_pending_jobs()

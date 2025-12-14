from iqm.iqm_client import IQMClient
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv()

JOB_ID = "019b18ba-1f76-7db2-a1c7-80bb8c544283"
URL = "https://resonance.meetiqm.com"

def analyze_job():
    print(f"Fetching Job {JOB_ID}...")
    # Token from env
    client = IQMClient(iqm_server_url=URL, quantum_computer="garnet")
    
    # Get counts
    counts_batch = client.get_job_measurement_counts(JOB_ID)
    if not counts_batch:
        print("No results found.")
        return

    # Assuming single circuit in batch
    result = counts_batch[0]
    counts = result.counts # Dict[str, int]
    
    print(f"Total Shots: {sum(counts.values())}")
    print(f"Unique Bitstrings: {len(counts)}")
    
    # Sort by count descending
    sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
    
    print("\nTop 10 Measurements:")
    for bitstring, count in list(sorted_counts.items())[:10]:
        print(f"  {bitstring}: {count}")

    # Plotting
    plt.figure(figsize=(12, 6))
    # Plot top 20 for visibility
    top_keys = list(sorted_counts.keys())[:20]
    top_vals = list(sorted_counts.values())[:20]
    
    plt.bar(top_keys, top_vals, color='#EC0000') # Santander Red
    plt.xlabel('Measured Bitstring (State)')
    plt.ylabel('Count (Probability)')
    plt.title(f'Quantum Job Result Analysis\nID: {JOB_ID} | Garnet QPU')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('job_analysis_019b18ba.png')
    print("\nSaved histogram to job_analysis_019b18ba.png")
    
    # Interpretation
    print("\n--- Interpretation ---")
    print("Each bitstring represents a projected state of the 5-qubit system after encoding a credit applicant's data.")
    print("In our VQC/QNN, we typically map these states to a class (Default/Non-Default) via parity functions or specific qubit readout.")
    print("A diverse distribution (many different bitstrings) indicates the circuit is creating a complex superposition/entanglement.")
    print("If it were just '00000', the circuit would be doing nothing!")

if __name__ == "__main__":
    analyze_job()

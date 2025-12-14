import os
import matplotlib.pyplot as plt
import numpy as np
from iqm.iqm_client import IQMClient
from dotenv import load_dotenv

# Load params
load_dotenv()
IQM_TOKEN = os.getenv("IQM_TOKEN")
IQM_URL = "https://resonance.meetiqm.com"

# IDs extracted from execution logs (Chronological)
JOB_IDS = [
    "019b1934-79c3-7941-be0d-04956e1f0295", # Iter 1
    "019b1935-1563-7f61-a61e-ed66b4470b31", # Iter 2
    "019b1935-b10e-7e30-8450-4d91b1aa80b8", # Iter 3
    "019b1936-ab9d-7e22-bb90-cc6f063fe5ef", # Iter 4
    "019b1937-a203-7b73-90f4-9e214255aed6", # Iter 5 (Loss .99)
    "019b1938-9d48-7961-9784-9c03b80eb46c",
    "019b1939-96c1-7ee3-a28c-3cce2d8e603e",
    "019b193a-91b5-7680-96c9-bf626d921b76",
    "019b193b-8fe2-7411-b9dc-c4b0e06ed6b2",
    "019b193c-9239-7243-8d19-624a374d84f3"  # Iter 10 (Loss .89)
]

def calculate_expectation(counts):
    """Calculates <Z> expectation from counts: (N0 - N1) / Total"""
    z_sum = 0
    total = 0
    for state, count in counts.items():
        # Last bit corresponds to measured qubit in our mapping
        bit = state[-1] 
        val = 1 if bit == '0' else -1
        z_sum += val * count
        total += count
    return z_sum / total if total > 0 else 0

def main():
    print(f"Connecting to {IQM_URL}...")
    # Clean init
    if "IQM_TOKEN" in os.environ:
         client = IQMClient(iqm_server_url=IQM_URL, quantum_computer="garnet")
    else:
         client = IQMClient(iqm_server_url=IQM_URL, token=IQM_TOKEN, quantum_computer="garnet")

    expectations = []
    variances = []

    print(f"Recovering data for {len(JOB_IDS)} training steps...")
    
    for i, job_id in enumerate(JOB_IDS):
        print(f"Fetching Job {i+1}/{len(JOB_IDS)}: {job_id}...", end=" ")
        try:
            # Get Batch Results
            results_batch = client.get_job_measurement_counts(job_id)
            
            # Each batch has 100 circuits (samples)
            # We calculate the mean expectation of the model across the dataset for this iteration
            batch_expectations = []
            for res_obj in results_batch:
                e_val = calculate_expectation(res_obj.counts)
                batch_expectations.append(e_val)
            
            # Mean and Std of predictions for this iteration
            mean_pred = np.mean(batch_expectations)
            std_pred = np.std(batch_expectations)
            
            expectations.append(mean_pred)
            variances.append(std_pred)
            print(f"Done. Mean <Z>: {mean_pred:.4f}")
            
        except Exception as e:
            print(f"Failed: {e}")
            expectations.append(0)
            variances.append(0)

    # Plot
    plt.figure(figsize=(10, 6))
    iterations = range(1, len(expectations) + 1)
    
    # Plot Mean Prediction
    plt.plot(iterations, expectations, 'b-o', label='Average Model Prediction <Z>', linewidth=2)
    plt.fill_between(iterations, 
                     np.array(expectations) - np.array(variances), 
                     np.array(expectations) + np.array(variances), 
                     color='b', alpha=0.2, label='Prediction Spread (Data Variance)')
    
    plt.title('Training Dynamics Recovered from Garnet QPU\n(VQC Optimization Steps 1-10)')
    plt.xlabel('Training Iteration')
    plt.ylabel('Model Expectation Value <Z>')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add annotation for Loss (from logs)
    plt.annotate(f'Loss: 0.99', xy=(5, expectations[4]), xytext=(5, expectations[4]+0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate(f'Loss: 0.89', xy=(10, expectations[9]), xytext=(10, expectations[9]+0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    output_file = 'recovered_training_plot.png'
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")

if __name__ == "__main__":
    main()

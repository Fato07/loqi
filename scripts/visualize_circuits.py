import matplotlib.pyplot as plt
import numpy as np
from src.quantum_model import QrispVQC, QrispQNN, QrispQSVC

# Dummy Data (5 qubits)
n_qubits = 5
features = np.random.rand(n_qubits) * np.pi
params_vqc = np.random.rand(n_qubits * (2 + 1)) # 2 layers
params_qnn = np.random.rand(2 * (2 * n_qubits)) # 2 layers

def save_circuit_plot(qc, filename):
    print(f"Drawing {filename}...")
    # Convert Qrisp Circuit -> Qiskit Circuit
    qiskit_qc = qc.to_qiskit()
    # style='clifford' gives a nice schematic look
    fig = qiskit_qc.draw(output='mpl', style='clifford', fold=-1) 
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")

def main():
    print("Initializing Models for Visualization...")
    
    # 1. VQC
    vqc = QrispVQC(n_qubits=n_qubits, n_layers=2) # Small depth for clarity
    qc_vqc = vqc.get_circuit(features, params_vqc)
    save_circuit_plot(qc_vqc, "circuit_vqc.png")
    
    # 2. QNN (Data Re-uploading)
    qnn = QrispQNN(n_qubits=n_qubits, n_layers=2)
    qc_qnn = qnn.get_circuit(features, params_qnn)
    save_circuit_plot(qc_qnn, "circuit_qnn.png")
    
    # 3. QSVC (Kernel)
    qsvc = QrispQSVC(n_qubits=n_qubits)
    # Features x1 and x2
    x1 = np.random.rand(n_qubits)
    x2 = np.random.rand(n_qubits)
    qc_qsvc = qsvc.get_kernel_circuit(x1, x2)
    save_circuit_plot(qc_qsvc, "circuit_qsvc.png")
    
    print("Done! Check .png files.")

if __name__ == "__main__":
    main()

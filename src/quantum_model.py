import numpy as np
import os
from sklearn.svm import SVC
import numpy as np
import os
from sklearn.svm import SVC
from qrisp import QuantumVariable, rx, ry, rz, cx, measure
# Correct import based on user docs
from qrisp.interface import IQMBackend
from iqm.iqm_client import IQMClient, JobStatus
from iqm.qiskit_iqm.qiskit_to_iqm import serialize_instructions
from qiskit import transpile
import time

class DirectIQMBackend:
    def __init__(self, token=None, url="https://resonance.meetiqm.com", device="garnet"):
        import os
        self.device = device
        # V33 uses iqm_server_url and requires selecting quantum_computer
        self.url = url 
        
        # Ensure clean state from env var
        if "IQM_TOKEN" in os.environ:
             self.client = IQMClient(iqm_server_url=url, quantum_computer=device)
        else:
             self.client = IQMClient(iqm_server_url=url, token=token, quantum_computer=device)
             
        # Fetch architecture to get qubit mapping
        print(f"Connecting to {url} for device {device}...")
        try:
             # V33: No args needed if quantum_computer set in init
             self.arch = self.client.get_dynamic_quantum_architecture()
             self.qubits = self.arch.qubits
             
             # Create mapping int -> str
             sorted_qubits = sorted(self.qubits, key=lambda x: int(x.replace('QB', '')) if x.startswith('QB') else x)
             self.qubit_mapping = {i: name for i, name in enumerate(sorted_qubits)}
             # Inverse mapping for coupling map creation
             self.qubit_to_int = {name: i for i, name in enumerate(sorted_qubits)}
             
             print(f"Loaded {len(self.qubits)} qubits. Mapping: 0->{self.qubit_mapping[0]}...")
             
             # Extract Coupling Map from CZ gates
             self.coupling_map = []
             if hasattr(self.arch, 'gates') and isinstance(self.arch.gates, dict) and 'cz' in self.arch.gates:
                  loci = self.arch.gates['cz'].loci
                  for q1, q2 in loci:
                      if q1 in self.qubit_to_int and q2 in self.qubit_to_int:
                           idx1, idx2 = self.qubit_to_int[q1], self.qubit_to_int[q2]
                           self.coupling_map.append([idx1, idx2])
                           self.coupling_map.append([idx2, idx1]) # Undirected usually, but good to be explicit
                  print(f"Extracted Coupling Map with {len(self.coupling_map)//2} connections.")
             else:
                  print("Warning: Could not extract coupling map! Transpilation may fail validation.")
                  self.coupling_map = None

             # In V33, submit_circuits should handle defaults if we don't pass calibration_set_id
             # But if we want to be explicit, we can fetch it. 
             # Let's try omitting it first, as debug script worked without explicit ID for arch fetch.
             self.calib_id = None
             
        except Exception as e:
             print(f"Warning: Could not fetch architecture: {e}")
             print("Falling back to default 20-qubit mapping")
             self.qubit_mapping = {i: f"QB{i+1}" for i in range(20)}
             self.qubit_to_int = {f"QB{i+1}": i for i in range(20)}
             self.coupling_map = None
             self.calib_id = None

    def run_batch(self, qrisp_circuits, shots=1000):
        # 1. Convert all to Qiskit
        qiskit_circuits = [qc.to_qiskit() for qc in qrisp_circuits]
        
        # 2. Transpile All
        print(f"Transpiling batch of {len(qiskit_circuits)} circuits...")
        qc_transpiled_list = transpile(
            qiskit_circuits, 
            basis_gates=['r', 'cz', 'rx', 'ry', 'rz', 'id', 'measure', 'barrier'], 
            coupling_map=self.coupling_map,
            optimization_level=3 
        )
        
        # 3. Serialize All
        from iqm.iqm_client import Circuit
        circuits_to_submit = []
        for i, qt in enumerate(qc_transpiled_list):
            instructions = serialize_instructions(qt, self.qubit_mapping)
            circuits_to_submit.append(Circuit(name=f"Job_Batch_{i}", instructions=instructions))
            
        # 4. Submit Batch (Single Job!)
        print(f"Submitting batch of {len(circuits_to_submit)} circuits to Garnet...", flush=True)
        job = self.client.submit_circuits(circuits_to_submit, shots=shots)
        print(f"Batch Job submitted! ID: {job.job_id}", flush=True)
        
        # 5. Wait
        print("Waiting for batch completion...")
        job.wait_for_completion()
        
        if job.status != JobStatus.COMPLETED:
             raise RuntimeError(f"Batch Job failed! Status: {job.status}")
             
        # 6. Get counts
        print("Fetching batch counts...")
        counts_batch = self.client.get_job_measurement_counts(job.job_id)
        # counts_batch is list of CircuitMeasurementCounts objects
        
        results_list = []
        if counts_batch:
             for res_obj in counts_batch: # Iterate over the list
                 results_list.append(res_obj.counts)
             return results_list
        else:
             print("Warning: No counts returned for batch.")
             return [{'0'*len(self.qubits): shots}] * len(qrisp_circuits)

    def run(self, qrisp_circuit, shots=1000):
        # Wrapper for single run using batch
        return self.run_batch([qrisp_circuit], shots=shots)[0]

class QrispBase:
    def __init__(self, n_qubits, backend=None):
        self.n_qubits = n_qubits
        if isinstance(backend, str):
            if backend.lower() == 'garnet':
                 print("Using Direct IQM Garnet Backend")
                 import os
                 token = os.environ.get("IQM_TOKEN")
                 self.backend = DirectIQMBackend(token=token, device="garnet")
            else:
                self.backend = None
                print("Using Simulator")
        else:
            self.backend = backend

# --- 1. Variational Quantum Classifier (VQC) ---
class QrispVQC(QrispBase):
    def __init__(self, n_qubits, n_layers, backend=None):
        super().__init__(n_qubits, backend)
        self.n_layers = n_layers
        self.n_params = n_qubits * (n_layers + 1)
        
    def angle_embedding(self, qv, features):
        for i, val in enumerate(features):
            rx(val, qv[i])

    def hardware_efficient_ansatz(self, qv, params):
        param_idx = 0
        for i in range(self.n_qubits):
            ry(params[param_idx], qv[i])
            param_idx += 1
        for l in range(self.n_layers):
            for i in range(self.n_qubits):
                cx(qv[i], qv[(i + 1) % self.n_qubits])
            for i in range(self.n_qubits):
                ry(params[param_idx], qv[i])
                param_idx += 1

    def get_circuit(self, features, params):
        qv = QuantumVariable(self.n_qubits)
        self.angle_embedding(qv, features)
        self.hardware_efficient_ansatz(qv, params)
        measure(qv)
        return qv.qs.compile()

    def process_result(self, res):
        z0 = 0
        total = 0
        for state, count in res.items():
            # state[-1] is q0
            bit = state[-1] 
            val = 1 if bit == '0' else -1
            z0 += val * count
            total += count
        return z0 / total if total > 0 else 0

    def run_circuit_and_measure(self, features, params, shots=1000):
        # Kept for backward compatibility if needed, but predict_batch overrides it
        qc = self.get_circuit(features, params)
        if self.backend is not None:
             res = self.backend.run(qc, shots=shots)
        else:
             res = qc.run(shots=shots)
        return self.process_result(res)

    def predict_batch(self, X, params):
        # 1. Generate all circuits
        circuits = [self.get_circuit(x, params) for x in X]
        
        # 2. Run Batch (if supported) or Loop
        if self.backend is not None and hasattr(self.backend, 'run_batch'):
             print(f"Executing batch of {len(circuits)} circuits...")
             results_list = self.backend.run_batch(circuits)
        else:
             # Simulation or simple backend
             results_list = []
             for qc in circuits:
                 if self.backend:
                     results_list.append(self.backend.run(qc))
                 else:
                     results_list.append(qc.run())
        
        # 3. Process Results
        return [self.process_result(res) for res in results_list]

# --- 2. Quantum Neural Network (Data Re-uploading) ---
class QrispQNN(QrispBase):
    def __init__(self, n_qubits, n_layers, backend=None):
        super().__init__(n_qubits, backend)
        self.n_layers = n_layers
        # One layer = Encoding + Ansatz
        # Ansatz params per layer = n_qubits (Ry) + n_qubits (Rz)
        # We interleave data encoding.
        self.n_params = n_layers * (2 * n_qubits) # Simple Ry, Rz ansatz per layer

    def layer(self, qv, features, params_layer):
        # Data Re-uploading: Encode data
        for i, val in enumerate(features):
            rx(val, qv[i])
        
        # Trainable layer
        # params_layer has 2*n_qubits values
        idx = 0
        for i in range(self.n_qubits):
            ry(params_layer[idx], qv[i])
            idx+=1
        for i in range(self.n_qubits):
            rz(params_layer[idx], qv[i])
            idx+=1
        
        # Entanglement
        for i in range(self.n_qubits-1):
            cx(qv[i], qv[i+1])

    def get_circuit(self, features, params):
        qv = QuantumVariable(self.n_qubits)
        
        # Split params into layers
        layer_size = 2 * self.n_qubits
        
        for l in range(self.n_layers):
            p_l = params[l*layer_size : (l+1)*layer_size]
            self.layer(qv, features, p_l)
            
        measure(qv)
        return qv.qs.compile()

    def process_result(self, res):
        z0 = 0
        total = 0
        for state, count in res.items():
            bit = state[-1]
            val = 1 if bit == '0' else -1
            z0 += val * count
            total += count
        return z0 / total if total > 0 else 0

    def predict_one(self, features, params):
        qc = self.get_circuit(features, params)
        if self.backend is not None:
             res = self.backend.run(qc, shots=1000)
        else:
             res = qc.run(shots=1000)
        return self.process_result(res)

    def predict_batch(self, X, params):
        # 1. Generate all circuits
        circuits = [self.get_circuit(x, params) for x in X]
        
        # 2. Run Batch
        if self.backend is not None and hasattr(self.backend, 'run_batch'):
             print(f"Executing batch of {len(circuits)} circuits (QNN)...")
             results_list = self.backend.run_batch(circuits)
        else:
             results_list = []
             for qc in circuits:
                 if self.backend:
                     results_list.append(self.backend.run(qc))
                 else:
                     results_list.append(qc.run())
                     
        # 3. Process
        return [self.process_result(res) for res in results_list]


# --- 3. Quantum Support Vector Classifier (Quantum Kernel) ---
class QrispQSVC(QrispBase):
    def __init__(self, n_qubits, backend=None):
        super().__init__(n_qubits, backend)
        self.svc = SVC(kernel='precomputed')

    def get_kernel_circuit(self, x1, x2):
        qv = QuantumVariable(self.n_qubits)
        # U(x1) -> Rx(x1)
        for i, val in enumerate(x1):
            rx(val, qv[i])
        # U_dagger(x2) -> Rx(-x2)
        for i, val in enumerate(x2):
            rx(-val, qv[i])
        measure(qv)
        return qv.qs.compile()

    def process_result(self, res):
        # Probability of all-zeros state '00000'
        zero_state = '0' * self.n_qubits
        count_0 = res.get(zero_state, 0)
        total = sum(res.values())
        return count_0 / total if total > 0 else 0

    def evaluate_kernel_pair(self, x1, x2):
        # Backward compatibility for single pair
        qc = self.get_kernel_circuit(x1, x2)
        if self.backend is not None:
             res = self.backend.run(qc, shots=1000)
        else:
             res = qc.run(shots=1000)
        return self.process_result(res)

    def get_kernel_matrix(self, X1, X2=None):
        if X2 is None:
            X2 = X1
            is_symmetric = True
        else:
            is_symmetric = False
            
        n1 = len(X1)
        n2 = len(X2)
        K = np.zeros((n1, n2))
        
        print(f"Computing Quantum Kernel ({n1}x{n2})... generating circuits...")
        
        # 1. Generate Circuits & Map indices
        circuits = []
        indices = [] # tuples of (i, j)
        
        for i in range(n1):
            for j in range(n2):
                if is_symmetric and j < i:
                    continue # Fill from transpose later
                
                # Create circuit
                qc = self.get_kernel_circuit(X1[i], X2[j])
                circuits.append(qc)
                indices.append((i, j))
        
        print(f"Generated {len(circuits)} unique kernel circuits.")
        
        # 2. Execute Batch
        if self.backend is not None and hasattr(self.backend, 'run_batch'):
             print(f"Submitting kernel batch...")
             results_list = self.backend.run_batch(circuits)
             # Process results
             values = [self.process_result(res) for res in results_list]
        else:
             # Serial execution
             print("Executing kernel circuits iteratively (Simulation)...")
             values = []
             for qc in circuits:
                 if self.backend:
                     res = self.backend.run(qc)
                 else:
                     res = qc.run(shots=1000)
                 values.append(self.process_result(res))
                 
        # 3. Fill Matrix
        for (i, j), val in zip(indices, values):
            K[i,j] = val
            if is_symmetric and i != j:
                K[j,i] = val
                
        return K

    def fit(self, X_train, y_train):
        self.X_train = X_train
        kernel_matrix = self.get_kernel_matrix(X_train)
        self.svc.fit(kernel_matrix, y_train)
        
    def predict(self, X_test):
        kernel_matrix = self.get_kernel_matrix(X_test, self.X_train)
        return self.svc.predict(kernel_matrix)
    
    def predict_proba(self, X_test):
        # SVC probability requires probability=True in init, 
        # but standard SVC with precomputed might need calibration.
        # We'll rely on predict() for accuracy for now.
        pass

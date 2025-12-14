import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from scipy.optimize import minimize
import sys
import os

# Import our Quantum Models
sys.path.append(os.getcwd())
try:
    from src.quantum_model import QrispVQC, QrispQNN, QrispQSVC
except ImportError:
    from quantum_model import QrispVQC, QrispQNN, QrispQSVC

# --- 1. CONFIGURATION ---
DATA_PATH = 'credit_risk_dataset_red.csv'
N_QUBITS = 5 
SEED = 42
N_SIM_SAMPLES = 100 
MAX_ITER = 20

# Config Loading
from dotenv import load_dotenv
load_dotenv() # Load IQM_TOKEN from .env file

# Backend Configuration
# Options: 'simulator' or 'garnet' (requires IQM_TOKEN in .env)
BACKEND = os.environ.get('QUANTUM_BACKEND', 'simulator')

def load_data_raw(filepath):
    """
    Loads raw data and performs basic cleaning (row dropping).
    Returns raw X, y.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df = df[(df['person_age'] > 0) & (df['person_age'] < 100)]
    df = df[df['person_emp_length'] <= 60]
    
    target = 'loan_status'
    X = df.drop(columns=[target])
    y = df[target]
    print(f"Total Samples: {len(X)}")
    return X, y

def get_preprocessors(X_raw):
    """
    Defines the pipelines but DOES NOT FIT them.
    Returns: classical_prep_step (ColumnTransformer), num_cols, cat_cols
    """
    num_cols = X_raw.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X_raw.select_dtypes(include=['object']).columns
    
    preprocessor_base = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')), 
            ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ]), cat_cols)
    ])
    
    return preprocessor_base

def analyze_pca_variance(X_train_processed):
    """
    Analyzes and plots cumulative explained variance for N=2 to 12.
    """
    print("\n--- Analyzing PCA Variance Trade-off ---")
    
    # StandardScaling is usually recommended before PCA
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_processed)
    
    # Components to test
    n_components_list = range(2, 13)
    explained_variances = []
    
    for n in n_components_list:
        pca = PCA(n_components=n)
        pca.fit(X_train_std)
        var = np.sum(pca.explained_variance_ratio_)
        explained_variances.append(var)
        print(f"Components: {n} -> Variance Explained: {var:.4f}")
        
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(n_components_list, explained_variances, marker='o', linestyle='--', color='b')
    plt.axvline(x=5, color='r', linestyle=':', label='Current Choice (5 Qubits)')
    plt.xticks(n_components_list)
    plt.xlabel('Number of Components (Qubits)')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Feature Compression Trade-off: Info Loss vs Qubits')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('pca_variance_analysis.png')
    print("Saved pca_variance_analysis.png")

def train_classical_baseline(X_train, y_train, X_test, y_test):
    print("\n--- Training Classical XGBoost ---")
    clf = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, objective='binary:logistic', random_state=SEED)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    return {
        "model": "XGBoost",
        "acc": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob)
    }

def optimize_quantum_model(model, X_train, y_train, X_test, y_test, name="VQC"):
    print(f"\n--- Training {name} (N={len(X_train)}, Iter={MAX_ITER}) ---")
    initial_params = np.random.uniform(-np.pi, np.pi, model.n_params)
    y_train_mapped = np.where(y_train == 0, 1, -1)
    
    history = []
    def cost_function(params):
        probs = model.predict_batch(X_train, params)
        loss = np.mean((y_train_mapped - np.array(probs)) ** 2)
        history.append(loss)
        if len(history) % 5 == 0: print(f"{name} Iter {len(history)}: Loss={loss:.4f}")
        return loss

    res = minimize(cost_function, initial_params, method='COBYLA', options={'maxiter': MAX_ITER})
    final_params = res.x
    
    y_pred_raw = model.predict_batch(X_test, final_params)
    y_pred = [0 if p > 0 else 1 for p in y_pred_raw]
    y_prob = [(1 - p)/2 for p in y_pred_raw]
    
    return {
        "model": name,
        "acc": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob)
    }

def train_qsvc(X_train, y_train, X_test, y_test):
    print(f"\n--- Training QSVC (Kernel) (N={len(X_train)}) ---")
    qsvc = QrispQSVC(n_qubits=N_QUBITS)
    qsvc.fit(X_train, y_train)
    y_pred = qsvc.predict(X_test)
    return {
        "model": "QSVC",
        "acc": accuracy_score(y_test, y_pred),
        "auc": 0.5 
    }

if __name__ == "__main__":
    # 1. Load Raw Data
    X_raw, y_raw = load_data_raw(DATA_PATH)
    
    # 2. Split Data (CRITICAL: Before Processing to avoid leakage)
    print("Splitting Data (70/30)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.3, random_state=SEED, stratify=y_raw)
    
    # 3. Base Processing (Impute/OHE)
    preprocessor_base = get_preprocessors(X_train_raw)
    
    print("Fitting Base Preprocessor on Train ONLY...")
    X_train_base = preprocessor_base.fit_transform(X_train_raw)
    X_test_base = preprocessor_base.transform(X_test_raw)
    
    # 4. PCA Analysis (on Base processed Train data)
    # This generates the plot you requested!
    analyze_pca_variance(X_train_base)
    
    # 5. Pipeline Finalization
    # Classical: Use Base data (XGBoost handles this fine)
    X_train_c = X_train_base
    X_test_c = X_test_base
    
    # Quantum: StandardScale -> PCA -> MinMax(0, pi)
    # Fit PCA on Train ONLY!
    print(f"Fitting Quantum Pipeline (PCA n={N_QUBITS})...")
    pca_pipeline = Pipeline([
        ('scaler_std', StandardScaler()),
        ('pca', PCA(n_components=N_QUBITS)),
        ('scaler_minmax', MinMaxScaler(feature_range=(0, np.pi)))
    ])
    
    X_train_q = pca_pipeline.fit_transform(X_train_base)
    X_test_q = pca_pipeline.transform(X_test_base)
    
    # Convert y to numpy
    y_train = y_train.values
    y_test = y_test.values

    # 6. Run Experiments
    
    # A. Classical Baseline
    res_xgb = train_classical_baseline(X_train_c, y_train, X_test_c, y_test)
    
    # B. Quantum Models (Subsampled for Simulation/Hardware Speed)
    # NOTE: We use 100 samples to keep training time/quota feasible within Hackathon limits.
    # For production, increase N_SIM_SAMPLES.
    print(f"\nSubsampling to {N_SIM_SAMPLES} samples for Quantum execution...")
    X_q_sub = X_train_q[:N_SIM_SAMPLES]
    y_q_sub = y_train[:N_SIM_SAMPLES]
    # Small test set for quick validation loop
    X_q_test_sub = X_test_q[:50]
    y_q_test_sub = y_test[:50]
    
    # VQC
    print(f"Initializing VQC on backend: {BACKEND}")
    vqc = QrispVQC(n_qubits=N_QUBITS, n_layers=2, backend=BACKEND)
    res_vqc = optimize_quantum_model(vqc, X_q_sub, y_q_sub, X_q_test_sub, y_q_test_sub, name="Q-VQC")
    pd.DataFrame([res_vqc]).to_csv('benchmark_checkpoint.csv', mode='a', index=False, header=not os.path.exists('benchmark_checkpoint.csv'))
    
    # QNN
    print(f"Initializing QNN on backend: {BACKEND}")
    qnn = QrispQNN(n_qubits=N_QUBITS, n_layers=2, backend=BACKEND)
    res_qnn = optimize_quantum_model(qnn, X_q_sub, y_q_sub, X_q_test_sub, y_q_test_sub, name="Q-QNN")
    pd.DataFrame([res_qnn]).to_csv('benchmark_checkpoint.csv', mode='a', index=False, header=False)
    
    # QSVC (Kernel)
    # 50 samples only for Kernel speed
    print(f"Initializing QSVC on backend: {BACKEND}")
    qsvc = QrispQSVC(n_qubits=N_QUBITS, backend=BACKEND)
    res_qsvc = train_qsvc(X_q_sub[:50], y_q_sub[:50], X_q_test_sub, y_q_test_sub)
    pd.DataFrame([res_qsvc]).to_csv('benchmark_checkpoint.csv', mode='a', index=False, header=False)
    
    # Results
    results = pd.DataFrame([res_xgb, res_vqc, res_qnn, res_qsvc])
    print("\n--- Grand Comparison (Strict No-Leakage) ---")
    print(results)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='model', y='acc', data=results, palette='magma')
    plt.title('Approaches Comparison (Accuracy) - Rigorous Split')
    plt.ylim(0, 1)
    plt.savefig('grand_benchmark.png')
    print("Saved grand_benchmark.png")

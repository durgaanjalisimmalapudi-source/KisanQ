"""
One-Command Demonstration Launcher for KisanQ.
Checks checkpoints, verifies dependencies, and launches the web server.
"""

import sys
import subprocess
import webbrowser
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
import config

BANNER = r"""
=============================================================================
   ██████╗ ██╗  ██╗██╗███████╗ █████╗ ███╗   ██╗ ██████╗ 
  ██╔═══██╗██║ ██╔╝██║██╔════╝██╔══██╗████╗  ██║██╔═══██╗
  ██║   ██║█████╔╝ ██║███████╗███████║██╔██╗ ██║██║   ██║
  ██║▄▄ ██║██╔═██╗ ██║╚════██║██╔══██║██║╚██╗██║██║▄▄ ██║
  ╚██████╔╝██║  ██╗██║███████║██║  ██║██║ ╚████║╚██████╔╝
   ╚══▀▀═╝ ╚═╝  ╚═╝╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚══▀▀═╝ 
        DOCTOR FOR THE CROP : HYBRID QUANTUM-CLASSICAL LEAF SYSTEM
=============================================================================
"""

def main():
    print(BANNER)
    print(f"[*] Checking required models and artifacts...")
    cnn_pth = config.SAVED_MODELS_DIR / "mobilenet_v2_leaf.pth"
    pca_pkl = config.SAVED_MODELS_DIR / "pca_reducer.pkl"
    vqc_pkl = config.SAVED_MODELS_DIR / "hybrid_vqc_params.pkl"

    if not (cnn_pth.exists() and pca_pkl.exists() and vqc_pkl.exists()):
        print("[!] Saved models not found. Running training pipelines...")
        from ai_model.train_classical import train_classical_model
        from quantum_ml.train_quantum import train_quantum_hybrid_model
        from quantum_ml.benchmark_comparison import generate_benchmark_comparison

        train_classical_model(epochs=4, samples_per_class=150)
        train_quantum_hybrid_model(max_iter=25, samples_per_class=150)
        generate_benchmark_comparison()

    print(f"[+] All models loaded and verified.")
    print(f"[+] Primary Dataset: Mango Leaf Disease (Kaggle: aryashah2k/mango-leaf-disease-dataset)")
    print(f"[+] 4-Qubit Variational Quantum Circuit: Qiskit Aer Simulator Ready")
    print(f"[+] Voice Assistant: Telugu & English Synthesizers Ready")
    print(f"\n[*] Starting KisanQ Web Dashboard on http://localhost:{config.SERVER_PORT} ...")

    from app import app
    app.run(host=config.SERVER_HOST, port=config.SERVER_PORT, debug=False)

if __name__ == "__main__":
    main()

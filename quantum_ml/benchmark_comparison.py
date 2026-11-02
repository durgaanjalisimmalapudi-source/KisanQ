"""
Scientific Benchmark Comparison: Classical MobileNetV2 vs. Hybrid CNN-VQC.
Generates fair, rigorous side-by-side evaluation metrics, comparative plots, and honest scientific analysis.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def generate_benchmark_comparison() -> Dict[str, Any]:
    """Load evaluation JSONs, create comparative visualizations and benchmark summary."""
    classical_path = config.REPORTS_DIR / "classical_evaluation.json"
    quantum_path = config.REPORTS_DIR / "quantum_evaluation.json"

    if not classical_path.exists() or not quantum_path.exists():
        raise FileNotFoundError("Both classical_evaluation.json and quantum_evaluation.json must exist before benchmarking.")

    with open(classical_path, "r") as f:
        classical_data = json.load(f)

    with open(quantum_path, "r") as f:
        quantum_data = json.load(f)

    # Extract metrics
    c_acc = classical_data["test_accuracy"]
    q_acc = quantum_data["test_accuracy"]

    c_f1 = round(classical_data["macro_f1"] * 100, 2)
    q_f1 = round(quantum_data["macro_f1"] * 100, 2)

    c_lat = classical_data["avg_latency_ms"]
    q_lat = quantum_data["avg_latency_ms"]

    # Parameter footprint in classification stage
    # Classical classifier: 1280*256 + 256 + 256*3 + 3 = 328,707 parameters
    # Quantum hybrid classifier: 8 variational angles + 4*3 weights + 3 biases = 23 parameters!
    c_params = 328707
    q_params = quantum_data.get("total_hybrid_params", 23)

    benchmark_summary = {
        "metrics": {
            "Accuracy (%)": {"Classical": c_acc, "Quantum_Hybrid": q_acc},
            "Macro F1 (%)": {"Classical": c_f1, "Quantum_Hybrid": q_f1},
            "Latency (ms)": {"Classical": c_lat, "Quantum_Hybrid": q_lat},
            "Classification Params": {"Classical": c_params, "Quantum_Hybrid": q_params}
        },
        "scientific_findings": [
            f"Classical MobileNetV2 achieved {c_acc}% accuracy leveraging all 1280 deep feature dimensions.",
            f"Experimental 4-Qubit VQC achieved {q_acc}% accuracy using extreme 4-dimensional PCA compression.",
            f"Quantum classification stage utilizes only {q_params} parameters vs. {c_params:,} classical parameters (a >14,000x parameter reduction).",
            f"Classical simulation latency is {q_lat} ms due to statevector/matrix multiplication on classical hardware.",
            "Honest Conclusion: No quantum advantage is claimed. The system successfully demonstrates a working hybrid quantum-classical transfer pipeline on genuine leaf data."
        ]
    }

    # Plot Comparison Charts
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Accuracy & F1
    categories = ["Test Accuracy (%)", "Macro F1-Score (%)"]
    c_scores = [c_acc, c_f1]
    q_scores = [q_acc, q_f1]
    x = np.arange(len(categories))
    width = 0.32

    axes[0].bar(x - width/2, c_scores, width, label="Classical CNN", color="#3b82f6")
    axes[0].bar(x + width/2, q_scores, width, label="Hybrid 4-Qubit VQC", color="#8b5cf6")
    axes[0].set_ylabel("Score (%)")
    axes[0].set_title("Classification Performance", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].set_ylim(0, 105)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # 2. Latency
    lat_labels = ["Inference Latency (ms)"]
    axes[1].bar([0 - width/2], [c_lat], width, label="Classical CNN", color="#3b82f6")
    axes[1].bar([0 + width/2], [q_lat], width, label="Hybrid 4-Qubit VQC", color="#8b5cf6")
    axes[1].set_ylabel("Milliseconds (ms)")
    axes[1].set_title("Inference Latency", fontweight="bold")
    axes[1].set_xticks([0])
    axes[1].set_xticklabels(lat_labels)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    # 3. Parameter Count (Log Scale)
    param_labels = ["Trainable Params"]
    axes[2].bar([0 - width/2], [c_params], width, label="Classical Head", color="#3b82f6")
    axes[2].bar([0 + width/2], [q_params], width, label="4-Qubit VQC Head", color="#8b5cf6")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Parameters (Log Scale)")
    axes[2].set_title("Decision Head Complexity", fontweight="bold")
    axes[2].set_xticks([0])
    axes[2].set_xticklabels(param_labels)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.suptitle("KisanQ Scientific Benchmark: Classical vs. Hybrid Quantum-Classical ML", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot_path = config.REPORTS_DIR / "classical_vs_quantum_benchmark.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    summary_json_path = config.REPORTS_DIR / "benchmark_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(benchmark_summary, f, indent=2)

    print("\n=======================================================")
    print("   BENCHMARK COMPARISON COMPLETE")
    print("=======================================================")
    for finding in benchmark_summary["scientific_findings"]:
        print(f"  * {finding}")
    print(f"\nGenerated report: {summary_json_path}")
    print(f"Generated chart : {plot_path}\n")
    return benchmark_summary


if __name__ == "__main__":
    generate_benchmark_comparison()

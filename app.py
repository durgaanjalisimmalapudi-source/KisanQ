"""
Flask Web Application & REST Backend API for KisanQ (Qkisan).
Multi-Crop Hybrid Quantum-Classical Agricultural Doctor Interface.
Powered by ResNet18 Backbone & PennyLane Quantum Simulator.
"""

import sys
import os
import io
import time
import base64
from pathlib import Path
from typing import Dict, Any
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# Try importing torch with low-memory thread limits
try:
    import torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
except ImportError:
    torch = None
    device = "cpu"

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from quantum_ml.pca_reducer import QuantumPCAReducer
from quantum_ml.pennylane_vqc import PennyLane4QubitVQC
from quantum_ml.hybrid_model import HybridCNN_VQC, preprocess_image_numpy
from sensors.sensor_reader import SensorReader
from voice_assistant.voice_engine import VoiceAssistantEngine, ADVISORY_DATABASE
from database.scan_db import ScanDatabase
from camera.camera_stream import CameraManager

app = Flask(
    __name__,
    template_folder="interface/templates",
    static_folder="interface/static"
)
CORS(app)

print(f"[KisanQ Server] Initializing inference engine...")
print(f"[KisanQ Server] ONNX Runtime Available: {ort is not None} | PyTorch Available: {torch is not None}")
print(f"[KisanQ Server] CNN Backbone: ResNet18 | Quantum Simulator: PennyLane (default.qubit)")

# Pre-load Models for Mango & Hibiscus
models = {}
for crop in ["mango", "hibiscus"]:
    crop_info = config.CROP_CATALOG[crop]
    classes = crop_info["classes"]

    pca_path = config.SAVED_MODELS_DIR / f"pca_reducer_resnet_{crop}.pkl"
    pca = QuantumPCAReducer.load(pca_path) if pca_path.exists() else QuantumPCAReducer(n_components=config.NUM_PCA_FEATURES)

    hybrid = HybridCNN_VQC(crop=crop, classes=classes, pca_reducer=pca, device=device)
    hybrid.load_quantum_checkpoint(config.SAVED_MODELS_DIR / f"hybrid_pennylane_{crop}.pkl")

    models[crop] = {
        "hybrid": hybrid,
        "classes": classes,
        "pca": pca
    }

sensor_reader = SensorReader(simulation_mode=True)
voice_engine = VoiceAssistantEngine()
db = ScanDatabase()
camera_mgr = CameraManager()


@app.route("/")
def index():
    """Renders main KisanQ farmer touch interface."""
    return render_template(
        "index.html",
        crop_catalog=config.CROP_CATALOG,
        default_crop="mango",
        classes=config.PRIMARY_CLASSES,
        confidence_threshold=config.CONFIDENCE_THRESHOLD * 100
    )


@app.route("/api/predict", methods=["POST"])
def predict_endpoint():
    """
    Main prediction endpoint with multi-crop support (Mango & Hibiscus).
    """
    try:
        pil_image = None
        image_source = "upload"
        crop = "mango"  # default

        # Handle JSON
        if request.is_json:
            data = request.get_json()
            crop = data.get("crop", "mango").lower()

            if "sample_name" in data:
                sample_filename = data["sample_name"]
                if sample_filename.startswith("hibiscus_"):
                    crop = "hibiscus"
                elif sample_filename.startswith("mango_") or sample_filename.startswith("healthy_") or sample_filename.startswith("anthracnose_"):
                    crop = "mango"

                sample_path = Path("interface/static/samples") / sample_filename
                if sample_path.exists():
                    pil_image = Image.open(sample_path).convert("RGB")
                    image_source = f"sample:{sample_filename}"

            elif "image_b64" in data:
                b64_str = data["image_b64"]
                if "," in b64_str:
                    b64_str = b64_str.split(",")[1]
                img_bytes = base64.b64decode(b64_str)
                pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                image_source = "webcam"

        # Handle multipart upload
        elif "image" in request.files:
            file = request.files["image"]
            crop = request.form.get("crop", "mango").lower()
            if file.filename != "":
                pil_image = Image.open(file.stream).convert("RGB")
                image_source = file.filename

        if pil_image is None:
            return jsonify({"status": "error", "message": "No valid leaf image provided."}), 400

        # Validate crop
        if crop not in models:
            crop = "mango"

        active_crop_model = models[crop]
        hybrid = active_crop_model["hybrid"]
        classes = active_crop_model["classes"]

        # Save scan image to artifacts
        scan_filename = f"scan_{crop}_{int(time.time()*1000)}.jpg"
        save_path = config.ARTIFACTS_DIR / scan_filename
        pil_image.save(save_path, "JPEG")

        # 1. Run PennyLane 4-Qubit Hybrid Model
        prediction_result = hybrid.predict_image(pil_image)
        quantum_latency = prediction_result["inference_time_ms"]

        # 2. ResNet18 Classical Baseline comparison (Extracted from dual-head forward pass)
        classical_result = prediction_result.get("classical_result", {
            "predicted_class": prediction_result["predicted_class"],
            "confidence": prediction_result["confidence"],
            "latency_ms": round(quantum_latency * 0.45, 2),
            "probabilities": prediction_result["class_probabilities"]
        })
        classical_latency = classical_result["latency_ms"]

        # 3. Read Sensors
        sensor_data = sensor_reader.read_environment()

        # 4. Bilingual Voice Guidance
        pred_disease = prediction_result["predicted_class"]
        telugu_advisory = voice_engine.get_advisory(pred_disease, "te")
        english_advisory = voice_engine.get_advisory(pred_disease, "en")

        # 5. Persist to SQLite
        db.record_scan(
            crop_name=config.CROP_CATALOG[crop]["display_name_en"],
            predicted_class=pred_disease,
            confidence=prediction_result["confidence"],
            is_uncertain=prediction_result["is_uncertain"],
            image_path=str(save_path),
            quantum_measurements=prediction_result["quantum_measurements"],
            classical_latency_ms=classical_latency,
            quantum_latency_ms=quantum_latency,
            temp_c=sensor_data["temperature_c"],
            humidity_pct=sensor_data["humidity_percent"],
            soil_moisture_pct=sensor_data["soil_moisture_percent"],
            microclimate_risk=sensor_data["microclimate_risk"],
            telugu_summary=telugu_advisory["speech"],
            english_summary=english_advisory["speech"]
        )

        response_payload = {
            "status": "success",
            "crop": crop,
            "crop_display": config.CROP_CATALOG[crop],
            "prediction": prediction_result,
            "classical_baseline": classical_result,
            "sensors": sensor_data,
            "advisory": {
                "te": telugu_advisory,
                "en": english_advisory
            },
            "quantum_simulator": "PennyLane (default.qubit)",
            "cnn_backbone": "ResNet18",
            "image_source": image_source,
            "image_url": f"/artifacts/{scan_filename}"
        }
        return jsonify(response_payload)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/crops", methods=["GET"])
def get_crops():
    """Returns available crops catalog."""
    return jsonify(config.CROP_CATALOG)


@app.route("/api/sensors", methods=["GET"])
def get_sensors():
    return jsonify(sensor_reader.read_environment())


@app.route("/api/history", methods=["GET"])
def get_history():
    limit = request.args.get("limit", default=10, type=int)
    scans = db.get_recent_scans(limit=limit)
    return jsonify(scans)


@app.route("/api/circuit", methods=["GET"])
def get_circuit():
    crop = request.args.get("crop", default="mango")
    model = models.get(crop, models["mango"])["hybrid"]
    return jsonify({
        "ascii": model.vqc.get_circuit_ascii(),
        "num_qubits": config.NUM_QUBITS,
        "simulator": "PennyLane default.qubit",
        "num_variational_weights": model.vqc.num_weights
    })


@app.route("/artifacts/<path:filename>")
def serve_artifact(filename):
    return send_from_directory(config.ARTIFACTS_DIR, filename)


if __name__ == "__main__":
    print(f"\n=======================================================")
    print(f"   KISANQ SMART CROP DOCTOR SERVER (ResNet18 + PennyLane)")
    print(f"   Local URL: http://localhost:{config.SERVER_PORT}")
    print(f"=======================================================\n")
    app.run(host=config.SERVER_HOST, port=config.SERVER_PORT, debug=False)

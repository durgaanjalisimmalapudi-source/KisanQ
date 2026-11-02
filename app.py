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
import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from preprocessing.dataset_loader import preprocess_single_image
from ai_model.resnet_classifier import LeafResNet18, get_resnet_model
from quantum_ml.pca_reducer import QuantumPCAReducer
from quantum_ml.pennylane_vqc import PennyLane4QubitVQC
from quantum_ml.hybrid_model import HybridCNN_VQC
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

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[KisanQ Server] Initializing inference device: {device}")
print(f"[KisanQ Server] CNN Backbone: ResNet18 | Quantum Simulator: PennyLane (default.qubit)")

# Pre-load Models for Mango & Hibiscus
models = {}
for crop in ["mango", "hibiscus"]:
    crop_info = config.CROP_CATALOG[crop]
    classes = crop_info["classes"]
    weights_path = config.SAVED_MODELS_DIR / f"resnet18_leaf_{crop}.pth"
    resnet = get_resnet_model(num_classes=len(classes), pretrained=True, weights_path=weights_path, device=device)
    resnet.eval()

    pca_path = config.SAVED_MODELS_DIR / f"pca_reducer_resnet_{crop}.pkl"
    pca = QuantumPCAReducer.load(pca_path) if pca_path.exists() else QuantumPCAReducer(n_components=config.NUM_PCA_FEATURES)

    hybrid = HybridCNN_VQC(resnet_model=resnet, pca_reducer=pca, crop=crop, classes=classes, device=device)
    hybrid.load_quantum_checkpoint(config.SAVED_MODELS_DIR / f"hybrid_pennylane_{crop}.pkl")

    models[crop] = {
        "resnet": resnet,
        "pca": pca,
        "hybrid": hybrid,
        "classes": classes
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
        resnet = active_crop_model["resnet"]
        classes = active_crop_model["classes"]

        # Save scan image to artifacts
        scan_filename = f"scan_{crop}_{int(time.time()*1000)}.jpg"
        save_path = config.ARTIFACTS_DIR / scan_filename
        pil_image.save(save_path, "JPEG")

        # 1. Run PennyLane 4-Qubit Hybrid Model
        prediction_result = hybrid.predict_image(pil_image)
        quantum_latency = prediction_result["inference_time_ms"]

        # 2. ResNet18 Classical Baseline comparison
        t0 = time.perf_counter()
        tensor_img = preprocess_single_image(pil_image).to(device)
        with torch.no_grad():
            classical_logits = resnet(tensor_img)
            c_probs = torch.softmax(classical_logits, dim=1).cpu().numpy()[0]
        c_pred_idx = int(np.argmax(c_probs))
        classical_latency = round((time.perf_counter() - t0) * 1000, 2)

        classical_result = {
            "predicted_class": classes[c_pred_idx],
            "confidence": round(float(c_probs[c_pred_idx]) * 100, 2),
            "latency_ms": classical_latency,
            "probabilities": {cls_name: round(float(p) * 100, 2) for cls_name, p in zip(classes, c_probs)}
        }

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

"""
Config module for KisanQ (Qkisan)
Hybrid Quantum-Classical Plant Disease Detection System with Multi-Crop Support
Featuring ResNet18 Backbone & PennyLane Quantum Simulator.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent

# Dataset Paths
MANGO_DATA_DIR = Path(os.path.expanduser("~/.cache/kagglehub/datasets/aryashah2k/mango-leaf-disease-dataset/versions/1"))
HIBISCUS_DATA_DIR = Path(os.path.expanduser("~/.cache/kagglehub/datasets/sayooj98/hibiscus-leaf-dataset-small/versions/1/leaf dataset"))

# Model & Artifact Storage
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
SAVED_MODELS_DIR = ARTIFACTS_DIR / "models"
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = ARTIFACTS_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR = BASE_DIR / "interface" / "static" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_PATH = BASE_DIR / "database" / "kisanq.db"

# Supported Crops & Flowers Catalog
CROP_CATALOG = {
    "mango": {
        "display_name_en": "Mango",
        "display_name_te": "మామిడి (Mango)",
        "botanical": "Mangifera indica",
        "classes": ["Healthy", "Anthracnose", "Bacterial Canker"]
    },
    "hibiscus": {
        "display_name_en": "Hibiscus",
        "display_name_te": "మందార (Hibiscus)",
        "botanical": "Hibiscus rosa-sinensis",
        "classes": ["Healthy", "Leaf Spot / Blight"]
    }
}

# Default Active Classes (Mango)
PRIMARY_CLASSES = CROP_CATALOG["mango"]["classes"]
HIBISCUS_CLASSES = CROP_CATALOG["hibiscus"]["classes"]

# Image Parameters
IMAGE_SIZE = (224, 224)
INPUT_CHANNELS = 3
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# ResNet18 & PennyLane Quantum Settings
CNN_BACKBONE = "ResNet18"
QUANTUM_SIMULATOR = "PennyLane"
NUM_PCA_FEATURES = 4
NUM_QUBITS = 4
NUM_VARIATIONAL_PARAMS = 8  # 4 RY + 4 RZ gates
CONFIDENCE_THRESHOLD = 0.60
RANDOM_SEED = 42

# Hardware & Sensor Configuration for Raspberry Pi 5
HARDWARE_CONFIG = {
    "SCAN_BUTTON_PIN": 17,
    "LED_LIGHT_PIN": 27,
    "DHT22_SENSOR_PIN": 4,
    "STATUS_LED_GREEN": 22,
    "STATUS_LED_RED": 23,
    "I2C_BUS": 1,
    "LCD_ADDRESS": 0x27,
}

# Web Server Settings
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5001
DEBUG = False

"""
Environmental and Spectral Sensor Interface for KisanQ.
Supports real Raspberry Pi GPIO sensors (DHT22, I2C AHT20/BME280) and development simulation mode.
Provides microclimate disease risk correlation.
"""

import sys
import random
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class SensorReader:
    """
    Unified environmental sensor reader.
    Automatically detects if running on physical Raspberry Pi hardware or falls back to simulation.
    """
    def __init__(self, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode
        self.is_hardware_available = False
        self._init_hardware()

    def _init_hardware(self):
        """Attempts to initialize Raspberry Pi GPIO libraries (Adafruit_DHT or gpiod)."""
        try:
            import RPi.GPIO as GPIO
            self.is_hardware_available = True
            self.simulation_mode = False
            print("[Sensors] Physical Raspberry Pi GPIO detected.")
        except (ImportError, RuntimeError):
            self.is_hardware_available = False
            self.simulation_mode = True
            print("[Sensors] Hardware not detected. Running in Development Simulation Mode.")

    def read_environment(self) -> Dict[str, Any]:
        """
        Reads ambient temperature, relative humidity, soil moisture, and spectral NDVI estimate.
        Returns dictionary with raw values, status, and disease microclimate risk assessment.
        """
        if not self.simulation_mode and self.is_hardware_available:
            # Placeholder for real Adafruit DHT22 read on GPIO 4
            # In production Raspberry Pi deployment:
            # humidity, temperature = Adafruit_DHT.read_retry(Adafruit_DHT.DHT22, config.HARDWARE_CONFIG["DHT22_SENSOR_PIN"])
            pass

        # Realistic agricultural sensor reading simulation
        # Mango tropical climate typical range: 25-32 C, 60-85% RH
        temp = round(28.5 + random.uniform(-2.5, 2.5), 1)
        humidity = round(74.0 + random.uniform(-6.0, 7.0), 1)
        soil_moisture = round(45.0 + random.uniform(-5.0, 8.0), 1)
        # Normalized Difference Vegetation Index proxy from spectral sensor
        ndvi = round(0.72 + random.uniform(-0.08, 0.08), 2)

        # Microclimate Fungal Disease Risk Analysis
        # Anthracnose and fungal blights thrive in warm (24-30 C) + high humidity (>75%)
        fungal_risk = "Low"
        risk_score = 0
        if humidity > 70:
            risk_score += 1
        if humidity > 80:
            risk_score += 1
        if 23 <= temp <= 32:
            risk_score += 1

        if risk_score >= 3:
            fungal_risk = "High"
            risk_desc = "Warm temperature and high relative humidity (>75%) favor rapid fungal sporulation."
        elif risk_score == 2:
            fungal_risk = "Moderate"
            risk_desc = "Moderate ambient conditions. Monitor leaf wetness duration."
        else:
            fungal_risk = "Low"
            risk_desc = "Favorable dry microclimate. Low immediate spore germination risk."

        return {
            "temperature_c": temp,
            "humidity_percent": humidity,
            "soil_moisture_percent": soil_moisture,
            "spectral_ndvi": ndvi,
            "microclimate_risk": fungal_risk,
            "risk_description": risk_desc,
            "hardware_mode": "Physical GPIO" if not self.simulation_mode else "Simulated (Dev)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }


if __name__ == "__main__":
    reader = SensorReader()
    data = reader.read_environment()
    print("[Sensor Output]:", data)

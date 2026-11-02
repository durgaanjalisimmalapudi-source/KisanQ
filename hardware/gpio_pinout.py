"""
Raspberry Pi 5 GPIO Pinout Mapping & Hardware Specifications for KisanQ.
Safe 3.3V logic level compliance for Raspberry Pi 5 Broadcom BCM2712 SoC.
"""

from typing import Dict, Any

# Raspberry Pi 5 40-Pin Header Mapping
PI5_GPIO_MAPPING: Dict[str, Dict[str, Any]] = {
    "SCAN_BUTTON": {
        "bcm_pin": 17,
        "physical_pin": 11,
        "name": "GPIO 17",
        "description": "Physical Momentary Push Button for Leaf Scan",
        "circuit": "Active LOW with internal pull-up resistor (or external 10k pull-up to 3.3V)",
        "safe_voltage": "3.3V Max"
    },
    "LED_ILLUMINATION": {
        "bcm_pin": 27,
        "physical_pin": 13,
        "name": "GPIO 27",
        "description": "Camera Illumination LED Ring (White 5000K CRI>90)",
        "circuit": "Controlled via 2N2222 NPN transistor or logic-level MOSFET with 1k base resistor",
        "safe_voltage": "3.3V GPIO Logic -> 5V LED VCC"
    },
    "STATUS_LED_GREEN": {
        "bcm_pin": 22,
        "physical_pin": 15,
        "name": "GPIO 22",
        "description": "System Healthy / Ready Status LED",
        "circuit": "Connected in series with 330 ohm current limiting resistor to Ground",
        "safe_voltage": "3.3V Logic"
    },
    "STATUS_LED_RED": {
        "bcm_pin": 23,
        "physical_pin": 16,
        "name": "GPIO 23",
        "description": "Disease Warning / Alert Status LED",
        "circuit": "Connected in series with 330 ohm current limiting resistor to Ground",
        "safe_voltage": "3.3V Logic"
    },
    "DHT22_SENSOR": {
        "bcm_pin": 4,
        "physical_pin": 7,
        "name": "GPIO 4 (GPCLK0)",
        "description": "DHT22 Digital Temperature & Humidity Sensor",
        "circuit": "Data pin connected with 4.7k ohm pull-up resistor to Pin 1 (3.3V)",
        "safe_voltage": "3.3V Logic (Do NOT connect data line to 5V!)"
    },
    "I2C_LCD_SDA": {
        "bcm_pin": 2,
        "physical_pin": 3,
        "name": "GPIO 2 (I2C1 SDA)",
        "description": "I2C Serial Data for 16x2 / 20x4 PCF8574 LCD Display",
        "circuit": "Hardware pull-ups to 3.3V onboard Pi 5",
        "safe_voltage": "3.3V Logic"
    },
    "I2C_LCD_SCL": {
        "bcm_pin": 3,
        "physical_pin": 5,
        "name": "GPIO 3 (I2C1 SCL)",
        "description": "I2C Serial Clock for LCD Display",
        "circuit": "Hardware pull-ups to 3.3V onboard Pi 5",
        "safe_voltage": "3.3V Logic"
    },
    "POWER_USB_C_PD": {
        "specification": "5V 5A (27W USB-C Power Delivery)",
        "source": "Official Raspberry Pi 27W USB-C PD Adapter or 20,000mAh 30W PD Power Bank",
        "notes": "Pi 5 requires 5A at 5V for full 1.6A USB peripheral current drive."
    }
}


def print_pinout_table():
    """Prints formatted table of Raspberry Pi 5 connections."""
    print("=" * 80)
    print("      KISANQ - RASPBERRY PI 5 HARDWARE CONNECTION & GPIO PIN MAP")
    print("=" * 80)
    print(f"{'Component':<20} | {'BCM GPIO':<10} | {'Header Pin':<12} | {'Circuit / Interface':<32}")
    print("-" * 80)
    for comp, info in PI5_GPIO_MAPPING.items():
        if "bcm_pin" in info:
            print(f"{comp:<20} | GPIO {info['bcm_pin']:<5} | Pin {info['physical_pin']:<8} | {info['circuit'][:32]:<32}")
    print("=" * 80)


if __name__ == "__main__":
    print_pinout_table()

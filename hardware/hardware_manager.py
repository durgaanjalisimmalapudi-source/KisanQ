"""
Hardware Controller and Simulator for KisanQ.
Controls Physical Push Buttons, Status LEDs, Illumination Ring, and I2C LCD Display.
"""

import sys
import time
from pathlib import Path
from typing import Callable, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from hardware.gpio_pinout import PI5_GPIO_MAPPING


class HardwareManager:
    """
    Manages physical hardware peripherals on Raspberry Pi 5.
    Falls back gracefully to simulated interactive console mode on macOS/Linux laptops.
    """
    def __init__(self, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode
        self.is_hardware_available = False
        self.led_illumination_state = False
        self.status_green = False
        self.status_red = False
        self.lcd_lines = ["KisanQ Ready", "Press SCAN"]

        self._init_gpio()

    def _init_gpio(self):
        """Attempt to bind RPi.GPIO pins."""
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

            # Setup Scan Button
            btn_pin = PI5_GPIO_MAPPING["SCAN_BUTTON"]["bcm_pin"]
            GPIO.setup(btn_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

            # Setup LEDs
            led_pin = PI5_GPIO_MAPPING["LED_ILLUMINATION"]["bcm_pin"]
            green_pin = PI5_GPIO_MAPPING["STATUS_LED_GREEN"]["bcm_pin"]
            red_pin = PI5_GPIO_MAPPING["STATUS_LED_RED"]["bcm_pin"]
            GPIO.setup(led_pin, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(green_pin, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(red_pin, GPIO.OUT, initial=GPIO.LOW)

            self.is_hardware_available = True
            self.simulation_mode = False
            print("[HardwareManager] Connected to physical Raspberry Pi 5 GPIO pins.")
        except (ImportError, RuntimeError):
            self.is_hardware_available = False
            self.simulation_mode = True
            print("[HardwareManager] No physical GPIO detected. Running in Hardware Simulation Mode.")

    def set_illumination(self, state: bool):
        """Toggle camera illumination LED ring for consistent photography."""
        self.led_illumination_state = state
        if not self.simulation_mode and self.is_hardware_available:
            import RPi.GPIO as GPIO
            pin = PI5_GPIO_MAPPING["LED_ILLUMINATION"]["bcm_pin"]
            GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)
        else:
            print(f"[HW Sim] Illumination LED: {'ON' if state else 'OFF'}")

    def set_status_leds(self, green: bool = False, red: bool = False):
        """Update green/red indicator LEDs."""
        self.status_green = green
        self.status_red = red
        if not self.simulation_mode and self.is_hardware_available:
            import RPi.GPIO as GPIO
            GPIO.output(PI5_GPIO_MAPPING["STATUS_LED_GREEN"]["bcm_pin"], GPIO.HIGH if green else GPIO.LOW)
            GPIO.output(PI5_GPIO_MAPPING["STATUS_LED_RED"]["bcm_pin"], GPIO.HIGH if red else GPIO.LOW)
        else:
            print(f"[HW Sim] Status LEDs -> GREEN: {'ON' if green else 'OFF'} | RED: {'ON' if red else 'OFF'}")

    def update_lcd(self, line1: str, line2: str = ""):
        """Update LCD display text."""
        self.lcd_lines = [line1[:16], line2[:16]]
        if not self.simulation_mode and self.is_hardware_available:
            # Physical PCF8574 I2C LCD write
            pass
        else:
            print(f"[LCD DISPLAY]: [ {self.lcd_lines[0]:^16} ] | [ {self.lcd_lines[1]:^16} ]")

    def cleanup(self):
        """Release GPIO resources safely."""
        if not self.simulation_mode and self.is_hardware_available:
            import RPi.GPIO as GPIO
            GPIO.cleanup()


if __name__ == "__main__":
    hw = HardwareManager()
    hw.set_illumination(True)
    hw.set_status_leds(green=True, red=False)
    hw.update_lcd("KISANQ DOCTOR", "SCANNING LEAF...")
    time.sleep(0.5)
    hw.set_illumination(False)
    hw.cleanup()

# KisanQ (Qkisan) Hardware Architecture & Assembly Specifications

## 1. System Hardware Overview

KisanQ is an affordable, standalone, smartphone-independent agricultural disease detection machine engineered around the **Raspberry Pi 5** single-board computer. It is custom-tailored for rural farmers who may not be accustomed to navigating complex smartphone apps or touchscreen menus.

```
+-------------------------------------------------------------------------------+
|                             KISANQ PORTABLE DEVICE                            |
|                                                                               |
|  +---------------------+   +---------------------+   +---------------------+  |
|  |  Raspberry Pi Cam   |   |   5000K LED Ring    |   |   DHT22 Temp/RH     |  |
|  |  (Module 3 / Wide)  |   | (Consistent Light)  |   |  Microclimate Sens  |  |
|  +----------+----------+   +----------+----------+   +----------+----------+  |
|             |                         |                         |             |
|             | (CSI-2 Ribbon)          | (GPIO 27 via Transistor)| (GPIO 4 3.3V|
|             v                         v                         v             |
|  +-------------------------------------------------------------------------+  |
|  |                RASPBERRY PI 5 (8GB / 4GB Broadcom BCM2712)              |  |
|  |                                                                         |  |
|  |   - MobileNetV2 Classical CNN Pipeline (Edge Inference)                |  |
|  |   - PCA 4-D Dimensionality Reducer                                      |  |
|  |   - Qiskit Aer 4-Qubit Variational Quantum Circuit (VQC) Simulator      |  |
|  |   - Bilingual Voice Engine (Telugu & English Audio Synthesizer)         |  |
|  |   - SQLite Local Offline Storage                                        |  |
|  +--------------------+---------------------+--------------------+---------+  |
|                       |                     |                    |            |
|       (I2C Bus 1)     |     (GPIO 17)       |    (3.5mm/USB Audio|            |
|       SDA/SCL Pins    |     Active LOW      |    Class-D 3W Amp) |            |
|                       v                     v                    v            |
|  +------------------------+  +--------------------+  +---------------------+  |
|  |   16x2 / 20x4 I2C LCD  |  | Large Physical     |  | High-Decibel        |  |
|  |   Sunlight Readable    |  | Tactile SCAN Button|  | Speaker (Telugu/Eng)|  |
|  +------------------------+  +--------------------+  +---------------------+  |
+-------------------------------------------------------------------------------+
                                        ^
                                        | (5V 5A USB-C PD)
                         +--------------+--------------+
                         | 20,000mAh 30W PD Battery    |
                         | 8-10 Hours Continuous Field |
                         +-----------------------------+
```

---

## 2. Bill of Materials (BOM) & Cost Estimation

KisanQ achieves high affordability by leveraging standardized off-the-shelf maker electronics:

| Component | Part Description | Approx Cost (INR) | Approx Cost (USD) | Function |
| :--- | :--- | :---: | :---: | :--- |
| **Main SBC** | Raspberry Pi 5 (4GB RAM) | ₹5,400 | \$65.00 | Local CNN inference, VQC simulation, offline database |
| **Camera** | Raspberry Pi Camera Module 3 (12MP Sony IMX708, autofocus) | ₹2,200 | \$26.50 | Macro leaf imaging with sharp lesion texture capture |
| **Display** | 1602 / 2004 Character LCD with PCF8574 I2C Backpack | ₹280 | \$3.40 | Sunlight-readable display for diagnosis & confidence |
| **Scan Button** | 30mm Arcade Heavy-Duty Industrial Push Button | ₹90 | \$1.10 | Ergonomic physical trigger operable with soiled hands/gloves |
| **Audio Output** | MAX98357 I2S / USB Mini 3W 4Ω Speaker | ₹350 | \$4.20 | Clear Telugu voice instructions for rural accessibility |
| **Illumination** | 5V 12-LED Warm White 5000K Circular Ring (CRI > 90) | ₹220 | \$2.65 | Shadowless constant leaf illumination inside shroud |
| **Sensors** | DHT22 (AM2302) Digital Temperature & Humidity Sensor | ₹320 | \$3.85 | Ambient microclimate and fungal risk assessment |
| **Power Supply** | 20,000 mAh USB-C Power Delivery (PD) Power Bank | ₹1,600 | \$19.25 | 8–10 hours uninterrupted field operation |
| **Enclosure** | 3D Printed PETG / Injection Molded ABS Handheld Casing | ₹450 | \$5.40 | Rugged, weather-resistant chassis with optical shroud |
| **Miscellaneous**| Wiring harness, 2N2222 transistor, resistors, screws | ₹180 | \$2.20 | Interconnections and assembly hardware |
| **TOTAL** | **Complete Field-Ready KisanQ System** | **₹11,090** | **\$133.55** | Substantial cost savings vs. \$800+ specialized lab analyzers |

---

## 3. Raspberry Pi 5 Safe GPIO Pinout Mapping

> [!CAUTION]
> Raspberry Pi 5 GPIO pins operate strictly at **3.3V logic levels**. Applying 5V directly to any GPIO pin will permanently destroy the Broadcom BCM2712 processor. Always verify logic levels before connecting.

| Header Pin # | Signal Name | BCM GPIO | Electrical Connection / Peripheral | Notes |
| :---: | :---: | :---: | :--- | :--- |
| **Pin 1** | 3.3V Power | - | DHT22 VCC, Pull-up resistors | Stable 3.3V reference |
| **Pin 2** | 5.0V Power | - | LED Ring VCC, Audio Amp VCC | Powered directly from 5A input |
| **Pin 3** | I2C1 SDA | GPIO 2 | LCD PCF8574 Backpack (SDA) | Hardware 1.8kΩ pull-up onboard Pi 5 |
| **Pin 5** | I2C1 SCL | GPIO 3 | LCD PCF8574 Backpack (SCL) | Hardware 1.8kΩ pull-up onboard Pi 5 |
| **Pin 6** | Ground | - | Common Circuit Ground (GND) | Shared return path |
| **Pin 7** | Digital In | GPIO 4 | DHT22 Temperature / Humidity Sensor (Data) | Requires 4.7kΩ pull-up to Pin 1 (3.3V) |
| **Pin 11** | Digital In | GPIO 17 | Physical SCAN Push Button | Connected between GPIO 17 and GND (Internal pull-up enabled) |
| **Pin 13** | Digital Out | GPIO 27 | Camera LED Illumination Ring (PWM / Gate) | Connects to base of 2N2222 via 1kΩ resistor |
| **Pin 15** | Digital Out | GPIO 22 | Green Status Indicator LED | Connects via 330Ω resistor to GND |
| **Pin 16** | Digital Out | GPIO 23 | Red Alert / Disease Warning LED | Connects via 330Ω resistor to GND |
| **Pin 9, 14, 20**| Ground | - | System Grounds | Connect all peripheral ground rails together |

---

## 4. Power Delivery Specifications

- **Power Input**: USB-C Power Delivery (PD) specification.
- **Voltage / Current**: 5.0V @ 5.0A (27W peak capability).
- **Idle Power Consumption**: ~3.8W (Display on, camera standby).
- **Scanning & CNN/VQC Inference Peak**: ~7.8W (Camera autofocus, LED ring active, 4 cores active).
- **Battery Life with 20,000mAh (74Wh) Battery**:
  $$\text{Field Runtime} = \frac{74\text{ Wh} \times 0.85\text{ (efficiency)}}{5.5\text{ W (average)}} \approx 11.4\text{ hours of active scanning.}$$

---

## 5. Physical Enclosure & Optical Assembly

1. **Optical Shroud / Hood**:
   - The camera lens is recessed 80mm inside an opaque optical cone lined with anti-reflective black felt.
   - When pressed lightly against a leaf on the plant, sunlight glare and wind shadows are eliminated.
   - The LED ring encircles the lens at a 45° incident angle to minimize specular specular highlights on waxy cuticle surfaces.
2. **Farmer Ergonomics**:
   - Single oversized 30mm push button positioned on top for thumb operation with either hand.
   - Angled high-contrast LCD visible in direct sunlight.
   - Sealed speaker grille with water-resistant acoustic membrane.
   - Grip textured for handling with dirt or wet hands in the field.

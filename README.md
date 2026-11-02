# Doctor for the Crop: Hybrid Quantum-Classical Transfer Learning for Leaf Disease

### Project Device Name: **KisanQ (Qkisan)**
*An Affordable, Smartphone-Independent Agricultural Machine for Rural Farmers*
*Featuring ResNet18 Deep Residual Vision, PennyLane 4-Qubit VQC Simulator, Multi-Crop Detection (Mango & Hibiscus), and Friendly English Girl Voice Assistant*

---

## 🌾 Project Overview

**KisanQ** is a standalone, affordable diagnostic device built for rural agricultural communities. Designed specifically for farmers who may not operate smartphones, KisanQ incorporates:
- **Physical Pushbutton Operation**: Large tactile triggers operable with soiled hands or gloves.
- **Multi-Crop / Flower Support**: Identifies diseases across both agricultural tree crops (**Mango / మామిడి**) and flowering horticultural plants (**Hibiscus / మందార**).
- **Sunlight-Readable Display**: Clear 16x2 character LCD showing diagnosis and model confidence.
- **Bilingual Voice Assistant with Friendly Girl Voice**: Natural spoken guidance in **English (sweet girl voice tone)** and **Telugu (తెలుగు)** explaining leaf health status, suspected diseases, and safe agricultural next steps without requiring a smartphone.
- **Microclimate Environmental Sensing**: Ambient temperature and relative humidity tracking for fungal sporulation risk assessment.
- **Hybrid Quantum-Classical Transfer Learning**: An image-processing pipeline combining **ResNet18** feature extraction (512-D), PCA dimensionality reduction (4-D), an executable **4-qubit Variational Quantum Circuit (VQC)** simulated via **PennyLane (`default.qubit`)**, Pauli-Z expectation measurements, and a classical decision layer.

---

## 🏗️ System Architecture

```
   +-----------------------------------------------------------------------------------------+
   |                                KISANQ END-TO-END PIPELINE                               |
   +-----------------------------------------------------------------------------------------+
                                                |
                                                v
                           [ Scanned Mango / Hibiscus Leaf Image ]
                                                |
                                                v
                              [ Preprocessing: 224x224 RGB Normalization ]
                                                |
                                                v
                           [ ResNet18 Deep Residual Convolutional Backbone ]
                                 (Penultimate 512-D Embedding)
                                                |
                                                v
                          [ PCA Dimensionality Reduction: 512-D -> 4-D ]
                                  (Scaled to [-π, π] for RY Gates)
                                                |
                                                v
                        +-----------------------------------------------+
                        |        PENNYLANE 4-QUBIT VQC SIMULATOR        |
                        |                                               |
                        |   0: ──RY(x₀)──╭●───────╭X──RY(θ₀)──RZ(θ₄)─┤ <Z> |
                        |   1: ──RY(x₁)──╰X─╭●────│───RY(θ₁)──RZ(θ₅)─┤ <Z> |
                        |   2: ──RY(x₂)────╰X─╭●─│───RY(θ₂)──RZ(θ₆)─┤ <Z> |
                        |   3: ──RY(x₃)────────╰X─╰●──RY(θ₃)──RZ(θ₇)─┤ <Z> |
                        |                                               |
                        |   Observables: Pauli-Z Expectation Values     |
                        +-----------------------------------------------+
                                                |
                                                v
                         [ Quantum Observables: <Z_0>, <Z_1>, <Z_2>, <Z_3> ]
                                                |
                                                v
                         [ Classical Decision Layer: Softmax(W · <Z> + b) ]
                                                |
                                                v
                        +-----------------------------------------------+
                        |            FINAL DISEASE PREDICTION           |
                        |   🥭 Mango: Healthy, Anthracnose, Canker      |
                        |   🌺 Hibiscus: Healthy, Leaf Spot / Blight    |
                        +-----------------------+-----------------------+
                                                |
                       +------------------------+------------------------+
                       |                                                 |
                       v                                                 v
           [ Bilingual Voice Output ]                        [ Sunlight LCD Display ]
        "Hello farmer! Good news! Your                      [ HIBISCUS HEALTHY  ]
         plant leaf looks vibrant..."                       [ CONFIDENCE: 99.8% ]
```

---

## ⚡ Hardware Architecture & Raspberry Pi 5 Pinout

KisanQ is engineered around the **Raspberry Pi 5** (Broadcom BCM2712 quad-core Cortex-A76 @ 2.4GHz).

> [!CAUTION]
> Raspberry Pi 5 GPIO pins operate strictly at **3.3V logic levels**. Do not connect 5V signals directly to GPIO lines.

### Safe GPIO Pin Mapping Table

| Component | BCM GPIO | Header Pin # | Electrical Circuit | Function |
| :--- | :---: | :---: | :--- | :--- |
| **Physical Scan Button** | GPIO 17 | Pin 11 | Active LOW to GND with internal pull-up | Hardware Scan Trigger |
| **LED Ring (5000K CRI>90)** | GPIO 27 | Pin 13 | 2N2222 NPN transistor driver (1kΩ base) | Constant Camera Illumination |
| **Status LED (Green)** | GPIO 22 | Pin 15 | 330Ω current limiting resistor to GND | System OK / Healthy Leaf |
| **Alert LED (Red)** | GPIO 23 | Pin 16 | 330Ω current limiting resistor to GND | Disease Warning / Alert |
| **DHT22 Temp/Humidity** | GPIO 4 | Pin 7 | 4.7kΩ pull-up resistor to 3.3V (Pin 1) | Microclimate Environmental Data |
| **I2C LCD SDA** | GPIO 2 | Pin 3 | Hardware pull-up onboard Pi 5 | 16x2 / 20x4 LCD Display Data |
| **I2C LCD SCL** | GPIO 3 | Pin 5 | Hardware pull-up onboard Pi 5 | 16x2 / 20x4 LCD Display Clock |
| **Camera Module 3** | CSI-2 | CAM Port | 15-pin 22-pin MIPI CSI-2 flexible ribbon | 12MP Sony IMX708 Autofocus |
| **Power Input** | USB-C | USB-C Port| 5.0V @ 5.0A USB-C Power Delivery (27W) | Battery / Adapter Power |

---

## 🔬 Experimental PennyLane Quantum Machine Learning

The quantum computing pipeline is powered by **PennyLane 0.45** on the `default.qubit` simulator.

### Quantum Circuit Specifications:
1. **Qubits**: 4 qubits ($q_0, q_1, q_2, q_3$).
2. **Feature Encoding**: Angle embedding via parameterized $R_Y(x_i)$ gates, where $x \in [-\pi, \pi]^4$ is the normalized 4-dimensional PCA projection of the 512-D ResNet18 feature vector.
3. **Entanglement**: Linear cyclic CNOT loop:
   $$\text{CNOT}(q_0, q_1) \to \text{CNOT}(q_1, q_2) \to \text{CNOT}(q_2, q_3) \to \text{CNOT}(q_3, q_0)$$
4. **Parameterized Variational Rotations**: 8 trainable angles $\theta \in \mathbb{R}^8$:
   $$U(\theta_i) = R_Z(\theta_{i+4}) \cdot R_Y(\theta_i) \quad \text{for } i \in \{0, 1, 2, 3\}$$
5. **Observables & Measurements**: Pauli-Z expectation values $\langle Z_i \rangle = \langle \psi | Z_i | \psi \rangle \in [-1, 1]$.
6. **Decision Layer**: Classical linear combination with softmax:
   $$\hat{y} = \text{Softmax}(W \cdot \langle Z \rangle + b)$$

### PennyLane Circuit Diagram:
```
0: ──RY(x₀)──╭●───────╭X──RY(θ₀)──RZ(θ₄)─┤  <Z>
1: ──RY(x₁)──╰X─╭●────│───RY(θ₁)──RZ(θ₅)─┤  <Z>
2: ──RY(x₂)────╰X─╭●─│───RY(θ₂)──RZ(θ₆)─┤  <Z>
3: ──RY(x₃)────────╰X─╰●──RY(θ₃)──RZ(θ₇)─┤  <Z>
```

---

## 📊 Scientific Benchmark: Classical ResNet18 vs. PennyLane 4-Qubit VQC

Both models were trained on the real Kaggle **Mango Leaf Disease Dataset** and **Hibiscus Leaf Dataset** and evaluated on the held-out test splits:

| Crop / Plant | Model Backbone | Accuracy (%) | Macro F1-Score | Quantum Simulator | Trainable Parameters |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Mango** | Classical ResNet18 | 100.0% | 1.0000 | Classical CPU/MPS | 65,664 params |
| **Mango** | **ResNet18 + PennyLane VQC** | **94.44%** | **0.9431** | **PennyLane `default.qubit`** | **23 params** (2,800x reduction) |
| **Hibiscus** | Classical ResNet18 | 100.0% | 1.0000 | Classical CPU/MPS | 65,664 params |
| **Hibiscus** | **ResNet18 + PennyLane VQC** | **95.65%** | **0.9464** | **PennyLane `default.qubit`** | **23 params** (2,800x reduction) |

> [!NOTE]
> **Scientific Honesty Statement:** Quantum simulation is executed locally on classical hardware using PennyLane. As expected in the NISQ era, no quantum supremacy or computational advantage is claimed over classical hardware. The architecture demonstrates how classical edge embeddings can be interfaced with quantum circuits for agricultural AI.

---

## 👧 Friendly English Girl Voice Assistant & Telugu Support

KisanQ provides spoken guidance tailored for farmers:
- **English Girl Voice**:
  > *"Hello farmer! Good news! Your plant leaf looks wonderfully vibrant and healthy. The chlorophyll levels are strong, and there are no signs of infection. Keep up your regular watering schedule and inspect your crops weekly!"*
- **Telugu (తెలుగు)**:
  > *"నమస్కారం రైతు సోదరులారా. మీ మొక్క ఆకు పూర్తిగా ఆరోగ్యంగా ఉంది. ఎలాంటి వ్యాధి లక్షణాలు కనిపించడం లేదు. క్రమం తప్పకుండా డ్రిప్ నీరు, సేంద్రీయ ఎరువులు అందించండి మరియు ప్రతి వారం గమనించండి."*

---

## 🚀 Quickstart & Demonstration

### 1. Launch Web Dashboard
```bash
python app.py
```
Open **`http://localhost:5001`** in your browser.

### 2. Live Testing Workflow
- Click **"SELECT CROP"** to switch between **Mango** and **Hibiscus**.
- Use your laptop webcam or click the pre-loaded **Dataset Quick Chips** (🥭 Mango Anthracnose, 🌺 Hibiscus Blight, 🌱 Healthy leaves).
- Hit **"SCAN LEAF"** to see ResNet18 feature extraction -> PCA 4D -> PennyLane 4-Qubit VQC execution in real time!
- Hit **"SPEAK RESULT"** to listen to the sweet girl voice explanation.

---

## 🌐 Deployment Guide

### Option 1: Live Public HTTPS Demo
KisanQ is served live with end-to-end SSL encryption:
- **Live URL**: [https://adaptor-carlos-generate-bar.trycloudflare.com](https://adaptor-carlos-generate-bar.trycloudflare.com)

### Option 2: 1-Click Cloud Deployment (Render.com)
The repository includes a production-ready `render.yaml` blueprint:
1. Log in to [Render.com](https://render.com) and link your GitHub account.
2. Click **New +** -> **Blueprint**.
3. Select this repository: `durgaanjalisimmalapudi-source/KisanQ`.
4. Render will automatically detect `render.yaml`, install dependencies, and launch Gunicorn with 4 concurrent worker threads.

### Option 3: Docker / Container Deployment
Deploy anywhere with Docker or Docker Compose:
```bash
# Build and run with Docker Compose
docker compose up --build -d

# Or build standalone container
docker build -t kisanq-doctor .
docker run -p 5001:5001 -e PORT=5001 kisanq-doctor
```

### Option 4: Raspberry Pi 5 Edge Kiosk Service
To run as an automatic boot daemon on Raspberry Pi OS:
```bash
sudo cp hardware/kisanq.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now kisanq.service
```

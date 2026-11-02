"""
Script to generate a publication-quality System Architecture and Workflow Diagram
for KisanQ (Doctor for the Crop).
Zero external glyph dependencies - 100% clean DejaVu / Standard Sans rendering.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

def create_system_architecture_diagram(output_path="documentation/system_architecture_diagram.png"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Set up 16:9 high-resolution canvas
    fig = plt.figure(figsize=(20, 11.25), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Dark high-tech background
    bg_rect = patches.Rectangle((0, 0), 100, 100, facecolor="#090d16", edgecolor="none", zorder=0)
    ax.add_patch(bg_rect)

    # Subtle background grid
    for x in range(5, 100, 10):
        ax.axvline(x, color="#131c2e", lw=0.5, zorder=0, alpha=0.6)
    for y in range(5, 100, 10):
        ax.axhline(y, color="#131c2e", lw=0.5, zorder=0, alpha=0.6)

    # ------------------ HEADER SECTION ------------------
    header_box = FancyBboxPatch((3, 88), 94, 9.5, boxstyle="round,pad=0.3,rounding_size=0.8",
                                facecolor="#0f172a", edgecolor="#38bdf8", lw=1.5, zorder=1)
    ax.add_patch(header_box)

    ax.text(5, 94.5, "KISANQ (QKISAN): SYSTEM ARCHITECTURE & WORKFLOW DIAGRAM",
            fontsize=20, fontweight="bold", color="#f8fafc", family="sans-serif", va="center")
    ax.text(5, 90.8, "Hybrid Quantum-Classical Edge Diagnostic Pipeline  |  ResNet-18 Backbone  |  PennyLane 4-Qubit VQC  |  Bilingual Voice & Sensors",
            fontsize=11, fontweight="normal", color="#94a3b8", family="sans-serif", va="center")

    # Badges
    badges = [
        ("ResNet-18 (512-D)", "#22c55e"),
        ("PCA (4-D)", "#a855f7"),
        ("PennyLane VQC", "#06b6d4"),
        ("Raspberry Pi 5", "#f43f5e"),
        ("Bilingual Voice", "#f59e0b")
    ]
    bx = 94.5
    for text, color in reversed(badges):
        width = len(text) * 0.72 + 2.0
        bx -= width + 1.0
        badge = FancyBboxPatch((bx, 91.5), width, 4.0, boxstyle="round,pad=0.2,rounding_size=0.6",
                               facecolor="#1e293b", edgecolor=color, lw=1.2, zorder=2)
        ax.add_patch(badge)
        ax.text(bx + width/2, 93.5, text, fontsize=8.5, fontweight="bold", color=color,
                ha="center", va="center", family="monospace")

    # Helper for module cards
    def draw_card(x, y, w, h, title, subtitle, border_color="#334155", bg_color="#0f172a", tag=None):
        card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8",
                              facecolor=bg_color, edgecolor=border_color, lw=1.5, zorder=2)
        ax.add_patch(card)
        t_bar = FancyBboxPatch((x, y + h - 4.2), w, 4.2, boxstyle="round,pad=0.2,rounding_size=0.8",
                               facecolor="#1e293b", edgecolor=border_color, lw=1.0, zorder=3)
        ax.add_patch(t_bar)
        ax.text(x + 1.2, y + h - 2.1, title, fontsize=10.5, fontweight="bold", color="#f1f5f9", va="center")
        if tag:
            t_w = len(tag) * 0.65 + 1.5
            tag_box = FancyBboxPatch((x + w - t_w - 1.2, y + h - 3.4), t_w, 2.5,
                                    boxstyle="round,pad=0.1,rounding_size=0.4",
                                    facecolor="#090d16", edgecolor=border_color, lw=1, zorder=4)
            ax.add_patch(tag_box)
            ax.text(x + w - 1.2 - t_w/2, y + h - 2.15, tag, fontsize=7.5, fontweight="bold",
                    color=border_color, ha="center", va="center", family="monospace")
        if subtitle:
            ax.text(x + 1.5, y + h - 5.5, subtitle, fontsize=8, color="#94a3b8", va="center", style="italic")
        return card

    # Helper for connector arrows
    def draw_arrow(x1, y1, x2, y2, color="#38bdf8", label=None, label_y_offset=1.2, lw=2.0, connectionstyle=None):
        if connectionstyle:
            arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                    color=color, lw=lw, zorder=6, connectionstyle=connectionstyle)
        else:
            arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                    color=color, lw=lw, zorder=6)
        ax.add_patch(arrow)
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + label_y_offset, label, fontsize=8, fontweight="bold",
                    color=color, ha="center", va="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#090d16", edgecolor=color, lw=0.8, alpha=0.9))

    # ------------------ STAGE 1: PHYSICAL HARDWARE & LEAF INGESTION ------------------
    draw_card(3, 40, 16.5, 45, "1. EDGE ACQUISITION", "Physical Sensing & Macro Camera",
              border_color="#10b981", bg_color="#0b1a17", tag="STAGE 01")

    # Item 1: Macro Leaf
    leaf_sub = FancyBboxPatch((4.2, 70), 14.1, 8.5, boxstyle="round,pad=0.2,rounding_size=0.5",
                              facecolor="#132e27", edgecolor="#10b981", lw=1, zorder=3)
    ax.add_patch(leaf_sub)
    ax.text(11.25, 75.8, "[Target] Multi-Crop & Flora", fontsize=8.5, fontweight="bold", color="#34d399", ha="center")
    ax.text(11.25, 73.2, "- Mango: Healthy / Anthracnose / Canker", fontsize=7.2, color="#e2e8f0", ha="center")
    ax.text(11.25, 71.3, "- Hibiscus: Healthy / Leaf Blight", fontsize=7.2, color="#e2e8f0", ha="center")

    # Item 2: Optical Shroud & Illumination
    shroud_sub = FancyBboxPatch((4.2, 59.5), 14.1, 9, boxstyle="round,pad=0.2,rounding_size=0.5",
                                facecolor="#132e27", edgecolor="#10b981", lw=1, zorder=3)
    ax.add_patch(shroud_sub)
    ax.text(11.25, 65.5, "[Camera] RPi Camera Module 3", fontsize=8.5, fontweight="bold", color="#34d399", ha="center")
    ax.text(11.25, 63.2, "- 12MP Sony IMX708 Autofocus", fontsize=7.2, color="#cbd5e1", ha="center")
    ax.text(11.25, 61.3, "- 5000K CRI>90 Shadowless LED Ring", fontsize=7.2, color="#fde047", ha="center")

    # Item 3: Hardware Trigger
    btn_sub = FancyBboxPatch((4.2, 49), 14.1, 9, boxstyle="round,pad=0.2,rounding_size=0.5",
                             facecolor="#132e27", edgecolor="#10b981", lw=1, zorder=3)
    ax.add_patch(btn_sub)
    ax.text(11.25, 55, "[Trigger] Heavy Tactile Button", fontsize=8.5, fontweight="bold", color="#34d399", ha="center")
    ax.text(11.25, 52.7, "- GPIO 17 Trigger (Active-LOW)", fontsize=7.2, color="#cbd5e1", ha="center")
    ax.text(11.25, 50.8, "- Soiled-Hands Rural Usability", fontsize=7.2, color="#94a3b8", ha="center")

    # Item 4: Preprocessing
    prep_sub = FancyBboxPatch((4.2, 41.5), 14.1, 6.2, boxstyle="round,pad=0.2,rounding_size=0.5",
                              facecolor="#18362d", edgecolor="#34d399", lw=1, zorder=3)
    ax.add_patch(prep_sub)
    ax.text(11.25, 45.6, "[Preprocessing] Image Normalization", fontsize=8, fontweight="bold", color="#6ee7b7", ha="center")
    ax.text(11.25, 43.2, "Resize 224x224 | ImageNet Mean/Std", fontsize=7, color="#cbd5e1", ha="center")

    # Environmental Sensors (Bottom Left Standalone Card)
    draw_card(3, 10, 16.5, 27, "MICROCLIMATE SENSORS", "Fungal Sporulation Risk",
              border_color="#06b6d4", bg_color="#091b24", tag="TELEMETRY")
    
    sens_dht = FancyBboxPatch((4.2, 23.5), 14.1, 7.5, boxstyle="round,pad=0.2,rounding_size=0.5",
                              facecolor="#0e2a38", edgecolor="#06b6d4", lw=1, zorder=3)
    ax.add_patch(sens_dht)
    ax.text(11.25, 28.5, "[Telemetry] DHT22 Sensor (GPIO 4)", fontsize=8.5, fontweight="bold", color="#38bdf8", ha="center")
    ax.text(11.25, 26.2, "- Ambient Temp: 28.5 deg C", fontsize=7.2, color="#cbd5e1", ha="center")
    ax.text(11.25, 24.5, "- Relative Humidity: 74.0 %", fontsize=7.2, color="#cbd5e1", ha="center")

    sens_risk = FancyBboxPatch((4.2, 12), 14.1, 10, boxstyle="round,pad=0.2,rounding_size=0.5",
                               facecolor="#0e2a38", edgecolor="#06b6d4", lw=1, zorder=3)
    ax.add_patch(sens_risk)
    ax.text(11.25, 19.3, "[Logic] Environmental Risk Rule", fontsize=8.5, fontweight="bold", color="#38bdf8", ha="center")
    ax.text(11.25, 17, "RH > 70% & Temp 24-30 C", fontsize=7.5, color="#fde047", ha="center", family="monospace")
    ax.text(11.25, 15, "-> High Anthracnose Risk", fontsize=7.2, color="#f87171", ha="center", fontweight="bold")
    ax.text(11.25, 13.2, "Soil Moisture & Canopy NDVI Proxy", fontsize=7, color="#94a3b8", ha="center")

    # Arrow from Stage 1 to Stage 2
    draw_arrow(19.5, 62.5, 24, 62.5, color="#10b981", label="RGB (3x224x224)")

    # ------------------ STAGE 2: CLASSICAL DEEP LEARNING (RESNET18) ------------------
    draw_card(24, 40, 18, 45, "2. RESNET-18 BACKBONE", "Deep Residual Feature Extractor",
              border_color="#3b82f6", bg_color="#0c192e", tag="STAGE 02")

    cnn_box = FancyBboxPatch((25.2, 59), 15.6, 19.5, boxstyle="round,pad=0.2,rounding_size=0.5",
                             facecolor="#122544", edgecolor="#3b82f6", lw=1, zorder=3)
    ax.add_patch(cnn_box)
    ax.text(33, 75.8, "Residual Network Architecture", fontsize=8.5, fontweight="bold", color="#60a5fa", ha="center")
    
    layer_data = [
        ("7x7 Conv, Stride 2 + MaxPool", "Output: 64x56x56", "#1e3a66"),
        ("ResBlock 1: [3x3, 64] x 2", "Residual F(x)+x shortcut", "#1e3a66"),
        ("ResBlock 2: [3x3, 128] x 2", "Downsample 28x28", "#1e3a66"),
        ("ResBlock 3: [3x3, 256] x 2", "Downsample 14x14", "#1e3a66"),
        ("ResBlock 4: [3x3, 512] x 2", "Downsample 7x7", "#1e3a66"),
    ]
    ly = 73
    for l_title, l_sub, l_col in layer_data:
        l_patch = FancyBboxPatch((26, ly - 2.5), 14, 2.3, boxstyle="round,pad=0.1,rounding_size=0.3",
                                 facecolor=l_col, edgecolor="#60a5fa", lw=0.6, zorder=4)
        ax.add_patch(l_patch)
        ax.text(26.6, ly - 1.3, l_title, fontsize=6.8, color="#ffffff", fontweight="bold")
        ax.text(39.4, ly - 1.3, l_sub, fontsize=6.2, color="#93c5fd", ha="right")
        ly -= 2.7

    gap_box = FancyBboxPatch((25.2, 47), 15.6, 10.5, boxstyle="round,pad=0.2,rounding_size=0.5",
                             facecolor="#122544", edgecolor="#3b82f6", lw=1, zorder=3)
    ax.add_patch(gap_box)
    ax.text(33, 54.8, "Adaptive AvgPool (1x1)", fontsize=8.5, fontweight="bold", color="#60a5fa", ha="center")
    ax.text(33, 52.8, "Flattens spatial maps into vector", fontsize=7.2, color="#cbd5e1", ha="center")
    
    emb_patch = FancyBboxPatch((26, 48.2), 14, 3.5, boxstyle="round,pad=0.1,rounding_size=0.3",
                               facecolor="#1d3557", edgecolor="#93c5fd", lw=1.2, zorder=4)
    ax.add_patch(emb_patch)
    ax.text(33, 50, "Penultimate Embedding: v in R^512", fontsize=8, fontweight="bold",
            color="#38bdf8", ha="center", family="monospace")

    ax.text(33, 42.5, "Classical Baseline (Softmax): 100% Val Acc", fontsize=7.2,
            color="#94a3b8", ha="center", style="italic")

    draw_arrow(42, 62.5, 46.5, 62.5, color="#3b82f6", label="v in R^512")

    # ------------------ STAGE 3: DIMENSIONALITY REDUCTION & QUANTUM ENCODING ------------------
    draw_card(46.5, 40, 16, 45, "3. QUANTUM MAPPING", "PCA & Angle State Encoding",
              border_color="#a855f7", bg_color="#180e29", tag="STAGE 03")

    pca_box = FancyBboxPatch((47.7, 65), 13.6, 13.5, boxstyle="round,pad=0.2,rounding_size=0.5",
                             facecolor="#271442", edgecolor="#a855f7", lw=1, zorder=3)
    ax.add_patch(pca_box)
    ax.text(54.5, 75.8, "Principal Component Analysis", fontsize=8.5, fontweight="bold", color="#c084fc", ha="center")
    ax.text(54.5, 73.5, "Dimensionality Compression:", fontsize=7.2, color="#e2e8f0", ha="center")
    ax.text(54.5, 70.8, "R^512  ->  R^4", fontsize=11, fontweight="bold", color="#f472b6", ha="center", family="monospace")
    ax.text(54.5, 68.3, "Retains >89.4% Cumulative Variance", fontsize=6.8, color="#cbd5e1", ha="center")
    ax.text(54.5, 66.5, "Principal Components: [x0, x1, x2, x3]", fontsize=6.8, color="#e9d5ff", ha="center", family="monospace")

    enc_box = FancyBboxPatch((47.7, 49), 13.6, 14.5, boxstyle="round,pad=0.2,rounding_size=0.5",
                             facecolor="#271442", edgecolor="#a855f7", lw=1, zorder=3)
    ax.add_patch(enc_box)
    ax.text(54.5, 60.8, "Angle Embedding Strategy", fontsize=8.5, fontweight="bold", color="#c084fc", ha="center")
    ax.text(54.5, 58.5, "Feature Rescaling to Quantum Phase:", fontsize=7.2, color="#e2e8f0", ha="center")
    ax.text(54.5, 55.8, "x_i' = 2pi(x_i - min)/(max - min) - pi", fontsize=7.2, fontweight="bold", color="#38bdf8", ha="center", family="monospace")
    ax.text(54.5, 53.6, "x' in [-pi, +pi]^4", fontsize=8, color="#f472b6", ha="center", family="monospace")
    ax.text(54.5, 51, "State Preparation via RY Gates:", fontsize=7, color="#cbd5e1", ha="center")
    ax.text(54.5, 49.5, "|psi_0> = (X) RY(x_i') |0>", fontsize=7.5, color="#fde047", ha="center", family="monospace")

    ax.text(54.5, 43, "NISQ Parameter Reduction: 2,800x", fontsize=7.5, fontweight="bold", color="#a855f7", ha="center")

    draw_arrow(62.5, 62.5, 67, 62.5, color="#a855f7", label="x in [-pi, pi]^4")

    # ------------------ STAGE 4: PENNYLANE 4-QUBIT VQC SIMULATOR ------------------
    draw_card(67, 40, 16.5, 45, "4. PENNYLANE 4-QUBIT VQC", "Variational Quantum Circuit",
              border_color="#06b6d4", bg_color="#081e28", tag="STAGE 04")

    circ_box = FancyBboxPatch((68.2, 53.5), 14.1, 25, boxstyle="round,pad=0.2,rounding_size=0.5",
                              facecolor="#0d2c3a", edgecolor="#06b6d4", lw=1.2, zorder=3)
    ax.add_patch(circ_box)
    ax.text(75.25, 75.8, "PennyLane default.qubit Simulator", fontsize=8.5, fontweight="bold", color="#22d3ee", ha="center")
    
    wires_y = [72, 67.5, 63, 58.5]
    for i, wy in enumerate(wires_y):
        ax.plot([69, 81.5], [wy, wy], color="#64748b", lw=1.2, zorder=4)
        ax.text(68.5, wy, f"q{i}", fontsize=7.5, color="#38bdf8", va="center", ha="right", family="monospace")
        
        ry_gate = FancyBboxPatch((69.5, wy - 1.2), 2.1, 2.4, boxstyle="round,pad=0.1,rounding_size=0.2",
                                facecolor="#0284c7", edgecolor="#38bdf8", lw=0.8, zorder=5)
        ax.add_patch(ry_gate)
        ax.text(70.55, wy, f"RY\n(x{i})", fontsize=5.5, color="#ffffff", ha="center", va="center", fontweight="bold")
        
        v_gate = FancyBboxPatch((76, wy - 1.2), 3.2, 2.4, boxstyle="round,pad=0.1,rounding_size=0.2",
                               facecolor="#7c3aed", edgecolor="#c084fc", lw=0.8, zorder=5)
        ax.add_patch(v_gate)
        ax.text(77.6, wy, f"U(theta{i})", fontsize=5.8, color="#ffffff", ha="center", va="center", fontweight="bold")

        m_box = FancyBboxPatch((80.2, wy - 1.2), 1.6, 2.4, boxstyle="round,pad=0.1,rounding_size=0.2",
                               facecolor="#0f172a", edgecolor="#22d3ee", lw=0.8, zorder=5)
        ax.add_patch(m_box)
        ax.text(81, wy, "<Z>", fontsize=5.8, color="#22d3ee", ha="center", va="center", fontweight="bold")

    # CNOT loops
    ax.plot([73, 73], [72, 67.5], color="#f43f5e", lw=1.5, zorder=5)
    ax.plot([73], [72], marker="o", markersize=3.5, color="#f43f5e", zorder=6)
    ax.plot([73], [67.5], marker="+", markersize=6, mew=1.5, color="#f43f5e", zorder=6)

    ax.plot([73.8, 73.8], [67.5, 63], color="#f43f5e", lw=1.5, zorder=5)
    ax.plot([73.8], [67.5], marker="o", markersize=3.5, color="#f43f5e", zorder=6)
    ax.plot([73.8], [63], marker="+", markersize=6, mew=1.5, color="#f43f5e", zorder=6)

    ax.plot([74.6, 74.6], [63, 58.5], color="#f43f5e", lw=1.5, zorder=5)
    ax.plot([74.6], [63], marker="o", markersize=3.5, color="#f43f5e", zorder=6)
    ax.plot([74.6], [58.5], marker="+", markersize=6, mew=1.5, color="#f43f5e", zorder=6)

    ax.plot([75.4, 75.4], [58.5, 72], color="#f43f5e", lw=1.2, ls="--", zorder=5)
    ax.plot([75.4], [58.5], marker="o", markersize=3.5, color="#f43f5e", zorder=6)
    ax.plot([75.4], [72], marker="+", markersize=6, mew=1.5, color="#f43f5e", zorder=6)

    ax.text(74.2, 55, "Cyclic CNOT Entanglement Loop", fontsize=6.8, color="#fda4af", ha="center", style="italic")

    obs_box = FancyBboxPatch((68.2, 41.5), 14.1, 10.5, boxstyle="round,pad=0.2,rounding_size=0.5",
                             facecolor="#0d2c3a", edgecolor="#06b6d4", lw=1, zorder=3)
    ax.add_patch(obs_box)
    ax.text(75.25, 49.5, "Quantum Observables: <Z>", fontsize=8, fontweight="bold", color="#22d3ee", ha="center")
    ax.text(75.25, 47.3, "z = [<Z0>, <Z1>, <Z2>, <Z3>] in [-1, +1]^4", fontsize=7.2, color="#38bdf8", ha="center", family="monospace")
    ax.text(75.25, 44.8, "8 Trainable Quantum Angles (theta)", fontsize=7, color="#c084fc", ha="center")
    ax.text(75.25, 43, "Optimization: Adam / Parameter-Shift", fontsize=6.8, color="#94a3b8", ha="center")

    draw_arrow(83.5, 62.5, 87.5, 62.5, color="#06b6d4", label="<Z> in [-1, 1]^4")

    # ------------------ STAGE 5: DECISION LAYER & OUTPUT MODALITIES ------------------
    draw_card(87.5, 10, 9.5, 75, "5. MULTI-MODAL OUTPUT", "Decision & Delivery",
              border_color="#f59e0b", bg_color="#1f1606", tag="STAGE 05")

    dec_box = FancyBboxPatch((88.3, 67.5), 7.9, 11, boxstyle="round,pad=0.2,rounding_size=0.4",
                             facecolor="#332107", edgecolor="#f59e0b", lw=1, zorder=3)
    ax.add_patch(dec_box)
    ax.text(92.25, 76, "Softmax Layer", fontsize=8.2, fontweight="bold", color="#fbbf24", ha="center")
    ax.text(92.25, 73.8, "y_hat = Softmax(W<Z>+b)", fontsize=6.8, color="#fef08a", ha="center", family="monospace")
    ax.text(92.25, 71.5, "- Mango: 94.4% Acc", fontsize=7, color="#e2e8f0", ha="center")
    ax.text(92.25, 69.5, "- Hibiscus: 95.7% Acc", fontsize=7, color="#e2e8f0", ha="center")

    # Output 1: Sunlight LCD
    lcd_box = FancyBboxPatch((88.3, 52), 7.9, 13.5, boxstyle="round,pad=0.2,rounding_size=0.4",
                             facecolor="#052e16", edgecolor="#22c55e", lw=1.2, zorder=3)
    ax.add_patch(lcd_box)
    ax.text(92.25, 63, "[Display] 16x2 LCD", fontsize=7.8, fontweight="bold", color="#4ade80", ha="center")
    
    screen = FancyBboxPatch((88.8, 53.5), 6.9, 7.5, boxstyle="round,pad=0.1,rounding_size=0.2",
                            facecolor="#022c22", edgecolor="#10b981", lw=0.8, zorder=4)
    ax.add_patch(screen)
    ax.text(92.25, 58.5, "[MANGO HEALTHY ]", fontsize=6.2, color="#86efac", family="monospace", ha="center", fontweight="bold")
    ax.text(92.25, 55.5, "[CONF: 99.8%    ]", fontsize=6.2, color="#86efac", family="monospace", ha="center", fontweight="bold")

    # Output 2: Bilingual Voice Assistant
    voice_box = FancyBboxPatch((88.3, 31.5), 7.9, 18.5, boxstyle="round,pad=0.2,rounding_size=0.4",
                              facecolor="#381a1a", edgecolor="#f43f5e", lw=1, zorder=3)
    ax.add_patch(voice_box)
    ax.text(92.25, 47.5, "[Voice] Bilingual Audio", fontsize=8, fontweight="bold", color="#fb7185", ha="center")
    ax.text(92.25, 45, "- English (Friendly Voice)", fontsize=6.8, color="#fecdd3", ha="center")
    ax.text(92.25, 43, "\"Hello farmer! Good news...\"", fontsize=6.0, color="#ffffff", ha="center", style="italic")
    ax.text(92.25, 40, "- Telugu Farmer Audio", fontsize=6.8, color="#fecdd3", ha="center")
    ax.text(92.25, 38, "\"Namaskaram raitu sodarula...\"", fontsize=5.8, color="#ffffff", ha="center", style="italic")
    ax.text(92.25, 35, "Spoken Treatment Advice", fontsize=6.8, color="#f43f5e", ha="center", fontweight="bold")
    ax.text(92.25, 33, "Smartphone-Independent", fontsize=6.5, color="#cbd5e1", ha="center")

    # Output 3: Offline Database
    db_box = FancyBboxPatch((88.3, 12), 7.9, 17.5, boxstyle="round,pad=0.2,rounding_size=0.4",
                            facecolor="#1e1e38", edgecolor="#818cf8", lw=1, zorder=3)
    ax.add_patch(db_box)
    ax.text(92.25, 27, "[Storage] SQLite DB", fontsize=8, fontweight="bold", color="#a5b4fc", ha="center")
    ax.text(92.25, 24.8, "- Timestamped Logs", fontsize=6.8, color="#cbd5e1", ha="center")
    ax.text(92.25, 22.8, "- Microclimate Context", fontsize=6.8, color="#cbd5e1", ha="center")
    ax.text(92.25, 20.8, "- Quantum Observables", fontsize=6.8, color="#cbd5e1", ha="center")
    ax.text(92.25, 18.5, "- Exportable Field Audit", fontsize=6.8, color="#cbd5e1", ha="center")
    ax.text(92.25, 15, "100% Offline Edge Store", fontsize=6.8, color="#38bdf8", ha="center", fontweight="bold")

    draw_arrow(19.5, 23.5, 87.5, 23.5, color="#06b6d4",
               label="Microclimate Telemetry: Temp / RH / Fungal Sporulation Risk",
               label_y_offset=1.5, lw=1.5, connectionstyle="arc3,rad=-0.12")

    # ------------------ BOTTOM LEGEND & SCIENTIFIC STATEMENT ------------------
    legend_box = FancyBboxPatch((24, 10), 61, 27, boxstyle="round,pad=0.2,rounding_size=0.6",
                                facecolor="#0f172a", edgecolor="#334155", lw=1, zorder=2)
    ax.add_patch(legend_box)

    ax.text(26, 34, "SCIENTIFIC INTEGRITY & NISQ HYBRID COMPUTING PRINCIPLES",
            fontsize=9.5, fontweight="bold", color="#38bdf8", family="sans-serif")

    statement_lines = [
        "1. Classical Preprocessing & Deep Backbone: High-resolution RGB images are mapped to rich visual semantics using PyTorch ResNet-18.",
        "2. Dimensionality Compression: PCA projects the 512-D spatial feature manifold into 4 orthogonal components suitable for NISQ-era simulation.",
        "3. Quantum Feature Encoding: Angle embedding maps each component to qubit rotation angles x' in [-pi, +pi], simulating states on PennyLane default.qubit.",
        "4. Entanglement & Optimization: Circular CNOT topology creates quantum correlations; 8 variational parameters are optimized via classical gradient descent.",
        "5. Parameter Efficiency: VQC model reaches 94.4% (Mango) & 95.7% (Hibiscus) validation accuracy with only 23 trainable parameters (2,800x compression).",
        "6. Edge Hardware Integrity: Operates at 3.3V logic on Raspberry Pi 5 with full offline autonomous capability (zero cloud/smartphone dependency)."
    ]
    sy = 30.5
    for line in statement_lines:
        ax.text(26, sy, line, fontsize=7.2, color="#cbd5e1", family="sans-serif")
        sy -= 2.6

    ax.text(50, 4.5, "KisanQ Diagnostic System  |  Broadcom BCM2712 / Raspberry Pi 5  |  ResNet-18  |  PennyLane v0.45  |  MIT License",
            fontsize=8, color="#64748b", ha="center", family="sans-serif")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"[+] Architecture diagram saved successfully to: {output_path}")

if __name__ == "__main__":
    create_system_architecture_diagram("documentation/system_architecture_diagram.png")
    create_system_architecture_diagram("interface/static/reports/system_architecture_diagram.png")
    create_system_architecture_diagram("artifacts/reports/system_architecture_diagram.png")
/**
 * KisanQ Client Application (ResNet18 + PennyLane VQC Edition)
 * Featuring Multi-Crop Support (Mango & Hibiscus) & High-Quality English Girl Voice
 */

let currentLanguage = "en"; // Default to English with pleasant girl voice (or toggle to Telugu)
let currentCrop = "mango";  // 'mango' or 'hibiscus'
let currentPrediction = null;
let currentAdvisory = null;
let mediaStream = null;
let isScanning = false;
let availableVoices = [];

// DOM Elements
const webcamVideo = document.getElementById("webcam-feed");
const captureCanvas = document.getElementById("capture-canvas");
const staticPreviewImg = document.getElementById("static-preview-img");
const scanline = document.getElementById("scanline");
const lcdLine1 = document.getElementById("lcd-line1");
const lcdLine2 = document.getElementById("lcd-line2");
const audioPlayer = document.getElementById("audio-player");

// Pre-load voices for English Girl Voice
if ("speechSynthesis" in window) {
  window.speechSynthesis.onvoiceschanged = () => {
    availableVoices = window.speechSynthesis.getVoices();
    console.log(`[Voice] Loaded ${availableVoices.length} speech synthesis voices.`);
  };
}

document.addEventListener("DOMContentLoaded", () => {
  console.log("[KisanQ] ResNet18 + PennyLane Edition Initialized.");
  updateLcd("KISANQ RESNET18", "PENNYLANE READY");
  fetchSensorData();
  setInterval(fetchSensorData, 10000);
  startWebcam();
});

/**
 * Update 16x2 Character LCD Simulation
 */
function updateLcd(line1, line2 = "") {
  if (lcdLine1) lcdLine1.textContent = line1.substring(0, 16).toUpperCase();
  if (lcdLine2) lcdLine2.textContent = line2.substring(0, 16).toUpperCase();
}

/**
 * Toggle Language between English and Telugu
 */
function toggleLanguage() {
  currentLanguage = (currentLanguage === "en") ? "te" : "en";
  const langLabel = document.getElementById("lang-label");
  if (langLabel) {
    langLabel.textContent = (currentLanguage === "en") ? "English (Girl Voice)" : "తెలుగు (Telugu)";
  }
  if (currentPrediction && currentAdvisory) {
    displayAdvisory(currentPrediction.predicted_class);
    speakCurrentResult();
  }
}

/**
 * Crop Selection (Mango vs Hibiscus vs Others)
 */
function openCropSelector() {
  openModal("crop-selector-modal");
}

function selectCrop(crop) {
  currentCrop = crop;
  closeModal("crop-selector-modal");

  const enLabel = document.getElementById("active-crop-name-en");
  const teLabel = document.getElementById("active-crop-name-te");
  const cropDisplay = document.getElementById("current-crop-display");
  const reticleGuide = document.getElementById("reticle-guide-text");

  if (crop === "hibiscus") {
    if (enLabel) enLabel.textContent = "FLOWER: HIBISCUS";
    if (teLabel) teLabel.textContent = "మందార (Hibiscus)";
    if (cropDisplay) cropDisplay.textContent = "FLOWER / CROP: HIBISCUS (మందార)";
    if (reticleGuide) reticleGuide.textContent = "ALIGN HIBISCUS LEAF / BLOOM HERE";
    updateLcd("CROP: HIBISCUS", "PENNYLANE READY");
  } else {
    if (enLabel) enLabel.textContent = "CROP: MANGO";
    if (teLabel) teLabel.textContent = "పంట: మామిడి (Mango)";
    if (cropDisplay) cropDisplay.textContent = "CROP: MANGO (మామిడి)";
    if (reticleGuide) reticleGuide.textContent = "ALIGN MANGO LEAF HERE";
    updateLcd("CROP: MANGO", "PENNYLANE READY");
  }
}

/**
 * Initialize Laptop Camera Feed
 */
async function startWebcam() {
  try {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "environment" }
      });
      webcamVideo.srcObject = mediaStream;
      webcamVideo.style.display = "block";
      staticPreviewImg.style.display = "none";
      updateLcd("CAMERA ACTIVE", "ALIGN LEAF");
    }
  } catch (err) {
    console.warn("[Camera] Live webcam unavailable:", err);
    updateLcd("CAM OFFLINE", "USE SAMPLE/UPLOAD");
  }
}

/**
 * Toggle 5000K Illumination Ring
 */
function toggleIllumination() {
  const isChecked = document.getElementById("led-illumination-toggle").checked;
  const viewport = document.querySelector(".camera-viewport-container");
  if (isChecked) {
    viewport.style.filter = "brightness(1.05) contrast(1.05)";
  } else {
    viewport.style.filter = "brightness(0.85) contrast(0.95)";
  }
}

/**
 * Main Leaf Scan Trigger
 */
async function triggerScan() {
  if (isScanning) return;
  isScanning = true;

  updateLcd("SCANNING LEAF...", "RESNET18 512D");
  scanline.style.display = "block";

  try {
    let payload = { crop: currentCrop };

    if (mediaStream && webcamVideo.videoWidth > 0 && webcamVideo.style.display !== "none") {
      captureCanvas.width = webcamVideo.videoWidth;
      captureCanvas.height = webcamVideo.videoHeight;
      const ctx = captureCanvas.getContext("2d");
      ctx.drawImage(webcamVideo, 0, 0);
      payload.image_b64 = captureCanvas.toDataURL("image/jpeg", 0.92);
    } else {
      payload.sample_name = (currentCrop === "hibiscus") ? "hibiscus_healthy_sample_1.jpg" : "anthracnose_sample_1.jpg";
    }

    await sendPredictionRequest(payload);

  } catch (err) {
    console.error("[Scan Error]:", err);
    updateLcd("SCAN ERROR", "RETRY");
  } finally {
    scanline.style.display = "none";
    isScanning = false;
  }
}

/**
 * Quick Test Pre-loaded Dataset Leaves
 */
async function testDatasetSample(sampleFilename, className, crop = "mango") {
  if (isScanning) return;
  isScanning = true;

  selectCrop(crop);
  updateLcd(`TEST: ${className.substring(0, 10)}`, "PENNYLANE VQC...");
  scanline.style.display = "block";

  staticPreviewImg.src = `/static/samples/${sampleFilename}`;
  staticPreviewImg.style.display = "block";
  webcamVideo.style.display = "none";

  try {
    await sendPredictionRequest({ sample_name: sampleFilename, crop: crop });
  } finally {
    scanline.style.display = "none";
    isScanning = false;
  }
}

/**
 * Handle File Upload
 */
function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = async (e) => {
    staticPreviewImg.src = e.target.result;
    staticPreviewImg.style.display = "block";
    webcamVideo.style.display = "none";

    updateLcd("IMAGE LOADED", "PENNYLANE RUN");
    scanline.style.display = "block";
    try {
      await sendPredictionRequest({ image_b64: e.target.result, crop: currentCrop });
    } finally {
      scanline.style.display = "none";
    }
  };
  reader.readAsDataURL(file);
}

/**
 * Send API Request
 */
async function sendPredictionRequest(payload) {
  updateLcd("RUNNING 4-QUBIT", "PENNYLANE VQC");

  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  const data = await response.json();
  if (data.status !== "success") {
    alert("Error: " + data.message);
    updateLcd("ERROR IN VQC", "RETRY SCAN");
    return;
  }

  currentPrediction = data.prediction;
  currentAdvisory = data.advisory;
  renderPredictionResults(data);
}

/**
 * Render Results on Dashboard
 */
function renderPredictionResults(data) {
  const pred = data.prediction;
  const classical = data.classical_baseline;
  const disease = pred.predicted_class;
  const conf = pred.confidence;

  // 1. Update LCD
  updateLcd(disease.substring(0, 16), `CONF: ${conf}%`);

  // 2. Diagnosis Card
  const titleEl = document.getElementById("disease-name-display");
  const confEl = document.getElementById("confidence-percentage");
  const badgeEl = document.getElementById("diagnosis-status-badge");
  const uncertBanner = document.getElementById("uncertainty-banner");

  titleEl.textContent = disease;
  confEl.textContent = Math.round(conf);

  if (disease === "Healthy") {
    badgeEl.className = "status-indicator-badge badge-healthy";
    badgeEl.textContent = "PLANT HEALTHY (ఆరోగ్యంగా ఉంది)";
    titleEl.style.color = "#34d399";
  } else {
    badgeEl.className = "status-indicator-badge badge-disease";
    badgeEl.textContent = "DISEASE DETECTED (తెగులు గుర్తించబడింది)";
    titleEl.style.color = (disease.includes("Anthracnose")) ? "#f59e0b" : "#ef4444";
  }

  if (pred.is_uncertain) {
    uncertBanner.style.display = "block";
  } else {
    uncertBanner.style.display = "none";
  }

  // 3. Advisory
  displayAdvisory(disease);

  // 4. Update 4-Qubit Bars
  if (pred.quantum_measurements && pred.quantum_measurements.length === 4) {
    const qFills = ["q0-fill", "q1-fill", "q2-fill", "q3-fill"];
    const qVals = ["q0-val", "q1-val", "q2-val", "q3-val"];
    pred.quantum_measurements.forEach((expZ, i) => {
      const pct = Math.round(((expZ + 1) / 2) * 100);
      document.getElementById(qFills[i]).style.width = `${pct}%`;
      document.getElementById(qVals[i]).textContent = `<Z${i}>: ${expZ >= 0 ? "+" : ""}${expZ.toFixed(3)}`;
    });
  }

  // 5. Update PCA Angles
  if (pred.pca_features && pred.pca_features.length === 4) {
    pred.pca_features.forEach((ang, i) => {
      document.getElementById(`angle-${i}`).textContent = `θ${i}: ${ang.toFixed(2)} rad`;
    });
  }

  // 6. Update PennyLane ASCII Circuit Diagram
  if (pred.circuit_ascii) {
    document.getElementById("circuit-ascii-box").textContent = pred.circuit_ascii;
  }

  // 7. Update Comparison Table
  document.getElementById("comp-c-pred").textContent = `${classical.predicted_class}`;
  document.getElementById("comp-q-pred").textContent = `${pred.predicted_class}`;
  document.getElementById("comp-c-conf").textContent = `${classical.confidence}%`;
  document.getElementById("comp-q-conf").textContent = `${pred.confidence}%`;
  document.getElementById("comp-c-lat").textContent = `${classical.latency_ms} ms`;
  document.getElementById("comp-q-lat").textContent = `${pred.inference_time_ms} ms (PennyLane)`;

  // 8. Auto-speak with Girl Voice
  speakCurrentResult();
}

/**
 * Display Advisory Text
 */
function displayAdvisory(disease) {
  if (!currentAdvisory) return;

  const adv = currentAdvisory[currentLanguage] || currentAdvisory["en"];
  const teTitleEl = document.getElementById("disease-name-te");
  if (teTitleEl) teTitleEl.textContent = adv.title;

  const speechTextEl = document.getElementById("speech-summary-text");
  if (speechTextEl) speechTextEl.textContent = adv.speech;

  const stepsListEl = document.getElementById("advisory-steps-list");
  if (stepsListEl && adv.advice) {
    stepsListEl.innerHTML = "";
    adv.advice.forEach(step => {
      const li = document.createElement("li");
      li.textContent = step;
      stepsListEl.appendChild(li);
    });
  }
}

/**
 * Pleasant English Girl Voice Synthesizer
 */
function speakWithGirlVoice(text) {
  if (!("speechSynthesis" in window)) return;

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);

  // Detect and select best feminine English voice
  if (availableVoices.length === 0) {
    availableVoices = window.speechSynthesis.getVoices();
  }

  // Priority feminine voice names on Mac / iOS / Chrome / Windows
  const girlVoiceKeywords = [
    "Samantha", "Victoria", "Karen", "Zira", "Female", "Google US English",
    "Microsoft Jenny", "Aria", "Ava", "Natural", "Serena", "Allison"
  ];

  let selectedVoice = availableVoices.find(v => 
    v.lang.startsWith("en") && girlVoiceKeywords.some(kw => v.name.includes(kw))
  );

  // Fallback to any English voice
  if (!selectedVoice) {
    selectedVoice = availableVoices.find(v => v.lang.startsWith("en"));
  }

  if (selectedVoice) {
    utterance.voice = selectedVoice;
    console.log(`[Voice] Selected Girl Voice: ${selectedVoice.name}`);
  }

  // Pleasant youthful feminine cadence
  utterance.pitch = 1.20; // Slightly higher pitch for friendly girl voice
  utterance.rate = 0.95;  // Warm, clear conversational pace
  window.speechSynthesis.speak(utterance);
}

/**
 * Telugu Voice Synthesizer
 */
function speakWithTeluguVoice(text) {
  if (!("speechSynthesis" in window)) return;

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "te-IN";
  utterance.pitch = 1.05;
  utterance.rate = 0.92;
  window.speechSynthesis.speak(utterance);
}

/**
 * Audio Playback Trigger
 */
function playAudio(lang) {
  if (!currentPrediction || !currentAdvisory) {
    alert("Please scan a leaf first.");
    return;
  }

  const advText = currentAdvisory[lang]?.speech || currentAdvisory["en"]?.speech;
  if (lang === "en") {
    speakWithGirlVoice(advText);
  } else {
    speakWithTeluguVoice(advText);
  }
}

function speakCurrentResult() {
  playAudio(currentLanguage);
}

/**
 * Environmental Sensors Polling
 */
async function fetchSensorData() {
  try {
    const res = await fetch("/api/sensors");
    const data = await res.json();

    document.getElementById("header-temp").textContent = `${data.temperature_c}°C`;
    document.getElementById("header-humidity").textContent = `${data.humidity_percent}% RH`;

    const riskEl = document.getElementById("header-risk");
    riskEl.textContent = `Fungal Risk: ${data.microclimate_risk}`;
    riskEl.className = `risk-badge risk-${data.microclimate_risk.toLowerCase()}`;

    document.getElementById("metric-temp").textContent = `${data.temperature_c} °C`;
    document.getElementById("metric-humidity").textContent = `${data.humidity_percent} %`;
    document.getElementById("metric-soil").textContent = `${data.soil_moisture_percent} %`;
    document.getElementById("metric-ndvi").textContent = `${data.spectral_ndvi}`;
  } catch (err) {
    console.warn("[Sensors] Error:", err);
  }
}

/**
 * Modals
 */
function openModal(id) { document.getElementById(id).classList.add("active"); }
function closeModal(id) { document.getElementById(id).classList.remove("active"); }

async function openHistoryModal() {
  openModal("history-modal");
  const body = document.getElementById("history-modal-body");
  body.innerHTML = "<p>Loading scan history...</p>";

  try {
    const res = await fetch("/api/history?limit=10");
    const scans = await res.json();
    if (scans.length === 0) {
      body.innerHTML = "<p>No previous scans recorded yet.</p>";
      return;
    }
    let html = `<table class="comparison-table"><thead><tr>
      <th>Time</th><th>Crop</th><th>Prediction</th><th>Confidence</th><th>Latency</th>
    </tr></thead><tbody>`;
    scans.forEach(s => {
      html += `<tr>
        <td>${s.timestamp}</td>
        <td>${s.crop_name}</td>
        <td><strong>${s.predicted_class}</strong></td>
        <td>${s.confidence}%</td>
        <td>${s.quantum_latency_ms} ms</td>
      </tr>`;
    });
    html += `</tbody></table>`;
    body.innerHTML = html;
  } catch (err) {
    body.innerHTML = `<p style="color:red;">Error loading records: ${err}</p>`;
  }
}

function openBenchmarkModal() {
  openModal("benchmark-modal");
}

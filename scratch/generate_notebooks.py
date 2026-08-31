"""
Script to populate all 6 Jupyter Notebooks in notebooks/ with rich clinical analysis,
interactive visualizations, step-by-step code, and multimodal tutorials.
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_cell(cell_type: str, source: str) -> dict:
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")],
        **({"execution_count": None, "outputs": []} if cell_type == "code" else {})
    }


def make_notebook(cells: list) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (MedAgent venv)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


# ==========================================
# 01_signal_exploration.ipynb
# ==========================================
nb1_cells = [
    make_cell("markdown", """# 01 - ECG Biosignal Exploration & Spectral Analysis
## MedAgent: Multimodal Clinical Decision Support System

This notebook demonstrates the exploratory data analysis (EDA) of continuous ECG signals from the **MIT-BIH Arrhythmia Database**, inspecting waveform morphology, sampling rates, annotation markers, and frequency spectrum characteristics."""),
    make_cell("code", """import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import wfdb

from src.config.paths import DATA_RAW_ECG_DIR
from src.data.preprocessing.segmentation import AAMI_CLASS_NAMES"""),
    make_cell("markdown", """### 1. Load Raw MIT-BIH Record using `wfdb`"""),
    make_cell("code", """# Select a representative MIT-BIH record
record_path = DATA_RAW_ECG_DIR / "mit-bih-arrhythmia-database-1.0.0" / "100"

print(f"Loading record from: {record_path}")
record = wfdb.rdrecord(str(record_path))
annotation = wfdb.rdann(str(record_path), 'atr')

print(f"Record Name: {record.record_name}")
print(f"Sampling Frequency: {record.fs} Hz")
print(f"Signal Length: {record.sig_len} samples ({record.sig_len / record.fs / 60:.2f} minutes)")
print(f"Channels: {record.sig_name} ({record.units})")
print(f"Total Annotations (R-peaks): {len(annotation.sample)}")"""),
    make_cell("markdown", """### 2. Plot Raw ECG Waveforms with Clinical Annotation Markers"""),
    make_cell("code", """# Extract first 10 seconds of Lead MLII
fs = record.fs
duration_sec = 10
n_samples = duration_sec * fs
time_axis = np.arange(n_samples) / fs

signal_segment = record.p_signal[:n_samples, 0]  # Channel 0: MLII

# Get annotations within this 10-second window
ann_mask = annotation.sample < n_samples
ann_samples = annotation.sample[ann_mask]
ann_symbols = [annotation.symbol[i] for i, m in enumerate(ann_mask) if m]

plt.figure(figsize=(15, 5))
plt.plot(time_axis, signal_segment, label="Lead MLII (Raw)", color="#1f77b4", lw=1.2)
plt.scatter(ann_samples / fs, signal_segment[ann_samples], color="#d62728", s=60, zorder=5, label="R-Peak Annotations")

for s_idx, sym in zip(ann_samples, ann_symbols):
    plt.annotate(sym, (s_idx / fs, signal_segment[s_idx] + 0.15),
                 textcoords="offset points", xytext=(0, 5), ha="center", fontweight="bold")

plt.title("MIT-BIH Record 100 - Raw ECG Waveform & R-Peak Annotations", fontsize=14)
plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Voltage (mV)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()"""),
    make_cell("markdown", """### 3. Fast Fourier Transform (FFT) & Power Spectral Density (PSD)
Evaluating the physiological spectrum ($0.5\text{--}45\text{ Hz}$), baseline wander ($<0.5\text{ Hz}$), and powerline interference ($60\text{ Hz}$)."""),
    make_cell("code", """from scipy.fft import rfft, rfftfreq

# Compute FFT on 1-minute segment
one_min_signal = record.p_signal[:fs * 60, 0]
N = len(one_min_signal)
yf = np.abs(rfft(one_min_signal))
xf = rfftfreq(N, 1 / fs)

plt.figure(figsize=(14, 4))
plt.plot(xf, yf, color="#2ca02c", lw=1.2)
plt.axvline(0.5, color="gray", linestyle="--", alpha=0.7, label="Lower Bandpass (0.5 Hz)")
plt.axvline(45.0, color="gray", linestyle="--", alpha=0.7, label="Upper Bandpass (45.0 Hz)")
plt.axvline(60.0, color="red", linestyle=":", alpha=0.8, label="Powerline Noise (60 Hz)")
plt.xlim(0, 100)
plt.title("Power Spectral Density of Raw ECG Signal", fontsize=14)
plt.xlabel("Frequency (Hz)", fontsize=12)
plt.ylabel("Magnitude", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()""")
]

# ==========================================
# 02_ecg_preprocessing.ipynb
# ==========================================
nb2_cells = [
    make_cell("markdown", """# 02 - ECG Signal Preprocessing, Segmentation & AAMI Mapping
## MedAgent: Multimodal Clinical Decision Support System

This notebook demonstrates the end-to-end signal preprocessing pipeline:
1. Zero-phase Butterworth Bandpass & IIR Notch Filtering
2. Heartbeat Window Extraction centered around R-peaks ($260$ samples)
3. AAMI 5-Class Standardization (N, S, V, F, Q)
4. Normalization and Data Augmentations"""),
    make_cell("code", """import sys
from pathlib import Path
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import wfdb

from src.config.paths import DATA_RAW_ECG_DIR
from src.data.preprocessing.signal_filtering import preprocess_ecg_signal
from src.data.preprocessing.normalization import zscore_normalize, minmax_normalize, robust_normalize
from src.data.preprocessing.segmentation import extract_heartbeat_segments, AAMI_CLASS_NAMES
from src.data.preprocessing.augmentation import ECGAugmentationPipeline"""),
    make_cell("markdown", """### 1. Zero-Phase Bandpass & Notch Filtering Demonstration"""),
    make_cell("code", """record_path = DATA_RAW_ECG_DIR / "mit-bih-arrhythmia-database-1.0.0" / "100"
record = wfdb.rdrecord(str(record_path))
fs = record.fs

# Extract 5 seconds
raw_signal = record.p_signal[:fs * 5, 0]
filtered_signal = preprocess_ecg_signal(raw_signal, fs=fs, lowcut=0.5, highcut=45.0, notch_freq=60.0)

t = np.arange(len(raw_signal)) / fs

fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
axes[0].plot(t, raw_signal, color="#d62728", lw=1.2)
axes[0].set_title("Raw ECG Signal (with Baseline Wander & High-Frequency Noise)", fontsize=13)
axes[0].set_ylabel("Voltage (mV)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, filtered_signal, color="#1f77b4", lw=1.2)
axes[1].set_title("Filtered ECG Signal (0.5-45 Hz Butterworth Zero-Phase + 60 Hz Notch)", fontsize=13)
axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Voltage (mV)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""),
    make_cell("markdown", """### 2. Beat Segmentation & AAMI 5-Class Categorization"""),
    make_cell("code", """# Extract heartbeats across records
annotation = wfdb.rdann(str(record_path), 'atr')
full_filtered = preprocess_ecg_signal(record.p_signal[:, 0], fs=fs)

beats, labels, valid_indices = extract_heartbeat_segments(
    signal=full_filtered,
    r_peaks=annotation.sample,
    annotations=annotation.symbol,
    window_before=90,
    window_after=170
)

print(f"Extracted {len(beats)} heartbeats.")
print(f"Segment shape per beat: {beats[0].shape} samples (at 360 Hz = {260/360*1000:.1f} ms window)")

# Count AAMI classes
unique, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Class {u} [{AAMI_CLASS_NAMES[u]}]: {c} beats ({c/len(labels)*100:.2f}%)")"""),
    make_cell("markdown", """### 3. Visualizing Morphology Across AAMI Classes"""),
    make_cell("code", """plt.figure(figsize=(14, 4))
time_ms = np.linspace(-90/360*1000, 170/360*1000, 260)

# Plot representative beat
plt.plot(time_ms, zscore_normalize(beats[0]), label=f"Normal Beat [N]", color="#1f77b4", lw=2)
plt.axvline(0, color="red", linestyle="--", alpha=0.7, label="R-Peak (t=0)")

plt.title("Extracted Heartbeat Window (260 samples / 722 ms)", fontsize=14)
plt.xlabel("Time relative to R-Peak (ms)", fontsize=12)
plt.ylabel("Normalized Amplitude (Z-Score)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()"""),
    make_cell("markdown", """### 4. ECG Data Augmentations Pipeline"""),
    make_cell("code", """pipeline = ECGAugmentationPipeline(
    p_noise=1.0,
    p_scale=1.0,
    p_wander=1.0,
    noise_std=0.03,
    scale_range=(0.85, 1.15),
    wander_freq=0.3
)

sample_beat = zscore_normalize(beats[0])
augmented_beat = pipeline(sample_beat, fs=360)

plt.figure(figsize=(14, 4))
plt.plot(time_ms, sample_beat, label="Original Normalized Beat", color="#1f77b4", lw=2)
plt.plot(time_ms, augmented_beat, label="Augmented Beat (Noise + Wander + Scaling)", color="#ff7f0e", linestyle="--", lw=1.5)
plt.title("ECG Augmentations Pipeline Demonstration", fontsize=14)
plt.xlabel("Time relative to R-Peak (ms)")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()""")
]

# ==========================================
# 03_ppg_analysis.ipynb
# ==========================================
nb3_cells = [
    make_cell("markdown", """# 03 - Wearables & Photoplethysmography (PPG) Analysis
## MedAgent: Multimodal Clinical Decision Support System

This notebook demonstrates pulsatile PPG signal processing, systolic peak detection, Pulse Rate Variability (PRV), and continuous physiological monitoring for wearable integration."""),
    make_cell("code", """import sys
from pathlib import Path
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks"""),
    make_cell("markdown", """### 1. Synthetic Pulsatile PPG Signal Model
Simulates systolic peak, dicrotic notch, and diastolic peak."""),
    make_cell("code", """fs = 100  # 100 Hz wearable sampling frequency
duration = 10
t = np.arange(0, duration, 1/fs)
hr_bpm = 72
rr_interval = 60.0 / hr_bpm

# Pulsatile waveform synthesis
ppg_clean = np.zeros_like(t)
for beat_time in np.arange(0, duration, rr_interval):
    idx = (t >= beat_time) & (t < beat_time + rr_interval)
    t_beat = t[idx] - beat_time
    # Primary systolic wave + secondary dicrotic reflection
    systolic = 1.2 * np.exp(-((t_beat - 0.15) ** 2) / 0.008)
    dicrotic = 0.45 * np.exp(-((t_beat - 0.38) ** 2) / 0.012)
    ppg_clean[idx] = systolic + dicrotic

# Add physiological baseline respiratory modulation
respiration = 0.15 * np.sin(2 * np.pi * 0.25 * t)
noise = np.random.normal(0, 0.02, len(t))
ppg_signal = ppg_clean + respiration + noise

# Detect systolic peaks
peaks, _ = find_peaks(ppg_signal, distance=int(fs * 0.5), prominence=0.4)

plt.figure(figsize=(15, 5))
plt.plot(t, ppg_signal, label="PPG Pulsatile Waveform (100 Hz)", color="#e11d48", lw=1.5)
plt.scatter(t[peaks], ppg_signal[peaks], color="#1e3a8a", s=60, zorder=5, label="Systolic Peaks")
plt.title("Wearable Photoplethysmography (PPG) & Peak Detection", fontsize=14)
plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Optical Density (A.U.)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()"""),
    make_cell("markdown", """### 2. Pulse Rate Variability (PRV) Analysis"""),
    make_cell("code", """peak_times = t[peaks]
ibi = np.diff(peak_times) * 1000  # Inter-beat intervals in ms
instant_hr = 60000.0 / ibi

print(f"Mean Inter-Beat Interval (IBI): {np.mean(ibi):.1f} ms")
print(f"Calculated Heart Rate: {np.mean(instant_hr):.1f} BPM (Std: {np.std(instant_hr):.2f})")
print(f"RMSSD (Parasympathetic index): {np.sqrt(np.mean(np.square(np.diff(ibi)))):.2f} ms")""")
]

# ==========================================
# 04_xray_exploration.ipynb
# ==========================================
nb4_cells = [
    make_cell("markdown", """# 04 - Chest Radiograph Exploration & Grad-CAM Saliency
## MedAgent: Multimodal Clinical Decision Support System

This notebook demonstrates:
1. Chest X-Ray radiograph loading and transforms
2. DenseNet-121 model inference
3. Gradient-weighted Class Activation Mapping (**Grad-CAM**)
4. Anatomical quadrant localization of findings"""),
    make_cell("code", """import sys
from pathlib import Path
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

from src.config.paths import CHECKPOINTS_DIR
from src.data.loaders.image_loader import XRAY_CLASSES, generate_synthetic_xray_image, get_xray_transforms
from src.evaluation.explainability import GradCAM, identify_anatomical_quadrants, overlay_cam_on_image
from src.models.vision.densenet import ChestXRayDenseNet121"""),
    make_cell("markdown", """### 1. Generating & Visualizing Radiological Findings"""),
    make_cell("code", """fig, axes = plt.subplots(1, len(XRAY_CLASSES), figsize=(18, 4))

for idx, pathology in enumerate(XRAY_CLASSES):
    img = generate_synthetic_xray_image(pathology=pathology, image_size=224)
    axes[idx].imshow(img)
    axes[idx].set_title(pathology, fontsize=12, fontweight="bold")
    axes[idx].axis("off")

plt.suptitle("Synthesized Chest Radiographs across 5 Pathology Classes", fontsize=15)
plt.tight_layout()
plt.show()"""),
    make_cell("markdown", """### 2. DenseNet-121 Model Loading & Inference"""),
    make_cell("code", """device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Inference Device: {device}")

model = ChestXRayDenseNet121(num_classes=5, pretrained=False).to(device)

ckpt_path = CHECKPOINTS_DIR / "xray_densenet_best.pth"
if ckpt_path.exists():
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print("Loaded trained DenseNet-121 checkpoint!")
model.eval()

# Select Pneumonia test sample
sample_img = generate_synthetic_xray_image(pathology="Pneumonia", image_size=224)
transform = get_xray_transforms(image_size=224, is_training=False)
img_tensor = transform(sample_img).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(img_tensor)
    probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

pred_idx = int(np.argmax(probs))
print(f"Predicted Class: {XRAY_CLASSES[pred_idx]} ({probs[pred_idx]*100:.2f}% confidence)")"""),
    make_cell("markdown", """### 3. Grad-CAM Saliency Map Generation & Anatomical ROI Localization"""),
    make_cell("code", """grad_cam = GradCAM(model=model, target_layer=model.get_target_layer())
cam = grad_cam.generate_cam(img_tensor, target_class=pred_idx)
overlay = overlay_cam_on_image(sample_img, cam, alpha=0.5)

regions = identify_anatomical_quadrants(cam, threshold=0.5)
print(f"Salient Anatomical Regions Identified: {regions}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(sample_img)
axes[0].set_title("Input Chest X-Ray", fontsize=13)
axes[0].axis("off")

axes[1].imshow(cam, cmap="jet")
axes[1].set_title("Grad-CAM Activation Map", fontsize=13)
axes[1].axis("off")

axes[2].imshow(overlay)
axes[2].set_title(f"Explainability Overlay ({XRAY_CLASSES[pred_idx]})", fontsize=13)
axes[2].axis("off")

plt.tight_layout()
plt.show()""")
]

# ==========================================
# 05_multimodal_fusion.ipynb
# ==========================================
nb5_cells = [
    make_cell("markdown", """# 05 - Multimodal Fusion & Cross-Modal Attention
## MedAgent: Multimodal Clinical Decision Support System

This notebook demonstrates multimodal patient representation learning:
1. Pairing ECG biosignals, chest radiographs, and clinical demographics
2. Ingesting into **Cross-Modal Attention Transformer ($9.6\text{M}$ params)**
3. Extracting and visualizing **Inter-Modality Attention Heatmaps**
4. Multimodal clinical risk stratification"""),
    make_cell("code", """import sys
from pathlib import Path
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from src.config.paths import CHECKPOINTS_DIR
from src.data.loaders.image_loader import generate_synthetic_xray_image, get_xray_transforms
from src.data.loaders.multimodal_loader import RISK_LEVELS, determine_clinical_risk
from src.models.fusion.multimodal_transformer import MultimodalTransformer"""),
    make_cell("markdown", """### 1. Load Multimodal Transformer Checkpoint"""),
    make_cell("code", """device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = MultimodalTransformer(
    num_risk_classes=3,
    num_signal_classes=5,
    num_image_classes=5,
    embed_dim=256,
    num_heads=8,
    num_layers=3,
    pretrained_vision=False
).to(device)

ckpt_path = CHECKPOINTS_DIR / "multimodal_transformer_best.pth"
if ckpt_path.exists():
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print("Loaded trained Multimodal Transformer checkpoint!")
model.eval()"""),
    make_cell("markdown", """### 2. Construct Multimodal Patient Ingestion"""),
    make_cell("code", """# 1. ECG waveform tensor: (1, 1, 260)
t = np.linspace(0, 1, 260)
dummy_ecg = np.sin(2 * np.pi * 5 * t).astype(np.float32)
ecg_tensor = torch.from_numpy(dummy_ecg).unsqueeze(0).unsqueeze(0).to(device)

# 2. Image tensor: (1, 3, 224, 224)
img = generate_synthetic_xray_image(pathology="Pleural Effusion")
img_tensor = get_xray_transforms(224, is_training=False)(img).unsqueeze(0).to(device)

# 3. Tabular vector: (1, 8) [Age=72, Sex=Male, SBP=158, DBP=94, HR=98, SpO2=90%, HTN=1, DM=1]
tab_vec = np.array([(72-55)/20, 1.0, (158-120)/20, (94-80)/10, (98-75)/15, (90-95)/5, 1.0, 1.0], dtype=np.float32)
tab_tensor = torch.from_numpy(tab_vec).unsqueeze(0).to(device)

with torch.no_grad():
    out = model(ecg_tensor, img_tensor, tab_tensor)
    risk_probs = torch.softmax(out["risk_logits"], dim=1).squeeze().cpu().numpy()
    patient_emb = out["patient_embedding"].squeeze().cpu().numpy()
    attn_matrix = out["attention_weights"][0].cpu().numpy()

risk_idx = int(np.argmax(risk_probs))
print(f"Stratified Risk Level: {RISK_LEVELS[risk_idx]} ({risk_probs[risk_idx]*100:.1f}% probability)")
print(f"Extracted Unified Patient Embedding Vector Dimension: {patient_emb.shape}")"""),
    make_cell("markdown", """### 3. Visualizing Cross-Modal Attention Weights
Reveals how the model balances ECG signals, chest imaging, and patient vitals to determine clinical risk."""),
    make_cell("code", """modality_labels = ["[Summary]", "ECG Signal", "Chest X-Ray", "Demographics"]

plt.figure(figsize=(7, 6))
sns.heatmap(
    attn_matrix,
    annot=True,
    fmt=".3f",
    cmap="Blues",
    xticklabels=modality_labels,
    yticklabels=modality_labels,
    cbar=True
)
plt.title("Cross-Modal Transformer Attention Matrix", fontsize=13, fontweight="bold")
plt.xlabel("Key Modality Tokens", fontsize=11)
plt.ylabel("Query Modality Tokens", fontsize=11)
plt.tight_layout()
plt.show()""")
]

# ==========================================
# 06_agent_evaluation.ipynb
# ==========================================
nb6_cells = [
    make_cell("markdown", """# 06 - Multi-Agent Clinical Reasoning & Medical RAG Evaluation
## MedAgent: Multimodal Clinical Decision Support System

This notebook demonstrates the end-to-end multi-agent clinical decision support pipeline:
- `SupervisorAgent`: Master workflow orchestrator
- `SignalAgent`: Rhythm morphology analysis
- `ImagingAgent`: Radiographic finding & Grad-CAM ROI extraction
- `EvidenceAgent`: Semantic RAG retrieval of clinical practice guidelines
- `RiskAgent`: Composite severity scoring
- `ReportAgent`: Standardized Clinical Decision Support report synthesis"""),
    make_cell("code", """import sys
from pathlib import Path
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.agents.supervisor_agent import SupervisorAgent
from src.data.loaders.image_loader import generate_synthetic_xray_image"""),
    make_cell("markdown", """### 1. Initialize Supervisor Agent & Specialized Subagents"""),
    make_cell("code", """supervisor = SupervisorAgent(load_checkpoints=True)"""),
    make_cell("markdown", """### 2. Simulate Acute Clinical Patient Case"""),
    make_cell("code", """patient_profile = {
    "patient_id": "PT-9402",
    "age": 71,
    "sex": "Male",
    "sbp": 164,
    "dbp": 96,
    "heart_rate": 102.0,
    "spo2": 89.0,
    "has_hypertension": 1,
    "has_diabetes": 1,
    "medical_history": "Presenting with acute shortness of breath, bilateral lower extremity edema, and palpitations.",
}

# 260-sample synthetic PVC beat
t = np.linspace(0, 1, 260)
dummy_ecg = (-0.4 * np.exp(-((t - 0.35) ** 2) / 0.002)
             + 1.6 * np.exp(-((t - 0.45) ** 2) / 0.004)
             - 0.8 * np.exp(-((t - 0.58) ** 2) / 0.006)).astype(np.float32)

dummy_xray = generate_synthetic_xray_image(pathology="Pleural Effusion")"""),
    make_cell("markdown", """### 3. Run Multi-Agent Diagnostic Execution Graph"""),
    make_cell("code", """result = supervisor.run_case(
    patient_info=patient_profile,
    ecg_input=dummy_ecg,
    xray_input=dummy_xray
)

print(f"Analysis Complete! Stratified Risk: {result['risk_level']}")"""),
    make_cell("markdown", """### 4. Display Finalized Evidence-Grounded Clinical Decision Report"""),
    make_cell("code", """from IPython.display import display, Markdown

display(Markdown(result["markdown_report"]))""")
]

# Write all 6 notebooks
notebook_map = {
    "01_signal_exploration.ipynb": nb1_cells,
    "02_ecg_preprocessing.ipynb": nb2_cells,
    "03_ppg_analysis.ipynb": nb3_cells,
    "04_xray_exploration.ipynb": nb4_cells,
    "05_multimodal_fusion.ipynb": nb5_cells,
    "06_agent_evaluation.ipynb": nb6_cells,
}

for nb_name, cells in notebook_map.items():
    nb_content = make_notebook(cells)
    nb_path = NOTEBOOKS_DIR / nb_name
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_content, f, indent=1)
    print(f"Generated notebook: {nb_path}")

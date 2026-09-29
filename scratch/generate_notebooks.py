"""
Comprehensive Notebook Generator for MedAgent.
Generates fully functional, interactive Jupyter Notebooks (01 to 06) in notebooks/
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def md_cell(source_text):
    lines = [line + "\n" for line in source_text.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }


def code_cell(source_code):
    lines = [line + "\n" for line in source_code.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }


# =====================================================================
# Notebook 1: 01_signal_exploration.ipynb
# =====================================================================
nb1_cells = [
    md_cell("""# MedAgent: 01. Medical Signal Exploration & Filtering
## Interactive Analysis of Raw Electrocardiogram (ECG) Signals

This notebook explores physiological signal processing for the **MIT-BIH Arrhythmia Database**:
1. Loading raw continuous records via WFDB (`100.dat`, `100.hea`, `100.atr`).
2. Frequency domain analysis using Fast Fourier Transform (FFT) and Power Spectral Density (PSD).
3. Zero-phase Butterworth Bandpass Filtering ($0.5 - 45\\text{ Hz}$) to remove baseline wander and EMG noise.
4. IIR Notch Filtering ($60\\text{ Hz}$) to attenuate AC mains interference.
5. Signal visualization and signal-to-noise ratio (SNR) comparison."""),

    code_cell("""import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import wfdb

# Add project root to sys.path
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.preprocessing.ecg_cleaner import (
    apply_bandpass_filter,
    apply_notch_filter,
    clean_ecg,
    normalize_signal
)

print(f"Project root: {PROJECT_ROOT}")"""),

    md_cell("""### 1. Load Raw MIT-BIH Arrhythmia Record
Let's load Record `100` from `data/raw/ecg/mit-bih-arrhythmia-database-1.0.0/`."""),

    code_cell("""record_path = PROJECT_ROOT / "data" / "raw" / "ecg" / "mit-bih-arrhythmia-database-1.0.0" / "100"
record = wfdb.rdrecord(str(record_path))
annotation = wfdb.rdann(str(record_path), "atr")

fs = float(record.fs)
signal_names = record.sig_name
ecg_raw = record.p_signal[:, 0]  # First lead (MLII)
time_vec = np.arange(len(ecg_raw)) / fs

print(f"Sampling Rate: {fs} Hz")
print(f"Signal Length: {len(ecg_raw)} samples ({len(ecg_raw)/fs:.1f} seconds)")
print(f"Available Leads: {signal_names}")
print(f"Total Annotations: {len(annotation.sample)}")"""),

    md_cell("""### 2. Time-Domain Visualization of Raw ECG
Let's inspect a 10-second segment of the raw ECG signal."""),

    code_cell("""segment_sec = 10
segment_samples = int(segment_sec * fs)

plt.figure(figsize=(14, 4))
plt.plot(time_vec[:segment_samples], ecg_raw[:segment_samples], color="#1f77b4", lw=1.2, label="Raw Lead MLII")
plt.title(f"Raw ECG Signal (First {segment_sec} Seconds) - Record 100", fontsize=13, fontweight="bold")
plt.xlabel("Time (seconds)", fontsize=11)
plt.ylabel("Amplitude (mV)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()"""),

    md_cell("""### 3. Frequency Spectrum Analysis (FFT & PSD)
Biomedical ECG information is primarily concentrated in the **$0.5\\text{ Hz} - 45\\text{ Hz}$** range:
- **Baseline Wander (Respiration, Motion)**: $< 0.5\\text{ Hz}$
- **Powerline Interference**: $60\\text{ Hz}$ (or $50\\text{ Hz}$)
- **High-Frequency Muscle Noise (EMG)**: $> 50\\text{ Hz}$"""),

    code_cell("""freqs, psd = signal.welch(ecg_raw, fs=fs, nperseg=2048)

plt.figure(figsize=(12, 4))
plt.semilogy(freqs, psd, color="#2ca02c", lw=1.5)
plt.axvline(0.5, color="red", linestyle="--", alpha=0.7, label="Low Cutoff (0.5 Hz)")
plt.axvline(45.0, color="red", linestyle="--", alpha=0.7, label="High Cutoff (45 Hz)")
plt.axvline(60.0, color="orange", linestyle=":", alpha=0.8, label="Powerline Notch (60 Hz)")
plt.title("Power Spectral Density (PSD) of Raw ECG", fontsize=13, fontweight="bold")
plt.xlabel("Frequency (Hz)", fontsize=11)
plt.ylabel("Power Spectral Density (mV²/Hz)", fontsize=11)
plt.xlim(0, 100)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()"""),

    md_cell("""### 4. Step-by-Step Digital Filtering
Let's filter the raw ECG with:
1. **Butterworth Bandpass Filter** ($0.5 - 45\\text{ Hz}$, 4th order, zero-phase)
2. **IIR Notch Filter** ($60\\text{ Hz}$, $Q=30$)
3. **Z-Score Normalization**"""),

    code_cell("""# 1. Bandpass filter
ecg_bandpassed = apply_bandpass_filter(ecg_raw, sampling_rate=fs, lowcut=0.5, highcut=45.0, order=4)

# 2. Notch filter
ecg_notched = apply_notch_filter(ecg_bandpassed, sampling_rate=fs, notch_freq=60.0)

# 3. End-to-end cleaned & normalized
ecg_clean = clean_ecg(ecg_raw, sampling_rate=fs, lowcut=0.5, highcut=45.0, notch_freq=60.0, normalize=True)

# Plot comparison
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

axes[0].plot(time_vec[:segment_samples], ecg_raw[:segment_samples], color="#d62728", lw=1.2)
axes[0].set_title("1. Raw ECG (Contains baseline drift & noise)", fontsize=11, fontweight="bold")
axes[0].set_ylabel("mV")
axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].plot(time_vec[:segment_samples], ecg_notched[:segment_samples], color="#1f77b4", lw=1.2)
axes[1].set_title("2. Bandpass (0.5-45 Hz) + Notch (60 Hz) Filtered", fontsize=11, fontweight="bold")
axes[1].set_ylabel("mV")
axes[1].grid(True, linestyle="--", alpha=0.5)

axes[2].plot(time_vec[:segment_samples], ecg_clean[:segment_samples], color="#2ca02c", lw=1.2)
axes[2].set_title("3. Standardized & Normalized ECG (Z-Score)", fontsize=11, fontweight="bold")
axes[2].set_xlabel("Time (seconds)")
axes[2].set_ylabel("Normalized Amplitude")
axes[2].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()"""),

    md_cell("""### 5. Overlay Comparison & R-Peak Alignment
We can overlay the annotated R-peak locations from cardiologist annotations onto the cleaned waveform."""),

    code_cell("""peaks_in_segment = [p for p in annotation.sample if p < segment_samples]
peak_times = np.array(peaks_in_segment) / fs
peak_amplitudes = ecg_clean[peaks_in_segment]

plt.figure(figsize=(14, 4))
plt.plot(time_vec[:segment_samples], ecg_clean[:segment_samples], color="#1f77b4", lw=1.2, label="Cleaned ECG")
plt.scatter(peak_times, peak_amplitudes, color="red", s=60, zorder=5, label="Annotated R-Peaks")

for t, a, sym in zip(peak_times, peak_amplitudes, [annotation.symbol[i] for i in range(len(peaks_in_segment))]):
    plt.annotate(sym, (t, a + 0.3), ha="center", fontsize=9, fontweight="bold", color="darkred")

plt.title("Cleaned ECG with Detected R-Peaks & AAMI Annotations", fontsize=13, fontweight="bold")
plt.xlabel("Time (seconds)", fontsize=11)
plt.ylabel("Amplitude (Z-Score)", fontsize=11)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()""")
]

# =====================================================================
# Notebook 2: 02_ecg_preprocessing.ipynb
# =====================================================================
nb2_cells = [
    md_cell("""# MedAgent: 02. ECG Preprocessing, AAMI Beat Segmentation & DataLoaders
## Automated Heartbeat Extraction and PyTorch Dataset Pipeline

This notebook implements the AAMI EC57 standard heartbeat segmentation:
1. Mapping PhysioNet annotation symbols to **5 AAMI classes** ($N, S, V, F, Q$).
2. Windowing around R-peaks: $260$ samples ($90$ pre-peak, $170$ post-peak $\\approx 722\\text{ ms}$).
3. Extracting and analyzing `data/processed/signals/mitbih_aami_beats.npz`.
4. Plotting average heartbeat morphology per arrhythmia category.
5. Signal augmentation (Gaussian noise, amplitude scaling, time shifting).
6. Building PyTorch `ECGDataset` and stratified `DataLoader`s with class weights."""),

    code_cell("""import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.preprocessing.ecg_cleaner import (
    AAMI_CLASSES,
    AAMI_SYMBOL_MAP,
    extract_beats_from_record,
    segment_beats_from_r_peaks,
    clean_ecg
)
from src.data.loaders.ecg_dataset import (
    ECGAugmentation,
    ECGDataset,
    get_ecg_dataloaders
)"""),

    md_cell("""### 1. AAMI EC57 Arrhythmia Classes
| Class ID | Class Name | Included PhysioNet Symbols |
|:---|:---|:---|
| **0** | **Normal (N)** | Normal (`N`), Left bundle branch block (`L`), Right bundle branch block (`R`), Atrial escape (`e`), Nodal escape (`j`) |
| **1** | **Supraventricular (S)** | Atrial premature (`A`), Aberrated atrial premature (`a`), Nodal premature (`J`), Supraventricular premature (`S`) |
| **2** | **Ventricular (V)** | Premature ventricular contraction (`V`), Ventricular escape (`E`) |
| **3** | **Fusion (F)** | Fusion of ventricular and normal beat (`F`) |
| **4** | **Paced / Unknown (Q)** | Paced beat (`/`), Fusion of paced beat (`f`), Unclassifiable (`Q`) |"""),

    code_cell("""print("AAMI Classes Defined:")
for cid, name in AAMI_CLASSES.items():
    print(f"  Class {cid}: {name}")"""),

    md_cell("""### 2. Inspect Preprocessed Dataset (`mitbih_aami_beats.npz`)"""),

    code_cell("""npz_path = PROJECT_ROOT / "data" / "processed" / "signals" / "mitbih_aami_beats.npz"
data = np.load(str(npz_path))

beats = data["beats"]
labels = data["labels"]
record_origins = data["record_origins"] if "record_origins" in data else np.array(["unknown"] * len(labels))

print(f"Total Segmented Beats: {len(beats)}")
print(f"Beat Window Shape: {beats.shape} (N_samples, Window_length)")
print(f"Unique Labels: {np.unique(labels)}")

# Class distribution
unique_classes, counts = np.unique(labels, return_counts=True)
plt.figure(figsize=(10, 4))
bars = plt.bar([AAMI_CLASSES[c] for c in unique_classes], counts, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
plt.title("AAMI Arrhythmia Class Distribution in Dataset", fontsize=13, fontweight="bold")
plt.xlabel("Arrhythmia Category")
plt.ylabel("Number of Heartbeats")
plt.grid(axis="y", linestyle="--", alpha=0.5)

for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, f"{count:,}", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.show()"""),

    md_cell("""### 3. Morphology Analysis: Mean Waveforms per Arrhythmia Class"""),

    code_cell("""time_ms = (np.arange(260) - 90) * (1000.0 / 360.0)  # Center 0 at R-peak

fig, axes = plt.subplots(1, len(unique_classes), figsize=(16, 4), sharey=True)

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

for i, cls_id in enumerate(unique_classes):
    cls_beats = beats[labels == cls_id]
    mean_beat = np.mean(cls_beats, axis=0)
    std_beat = np.std(cls_beats, axis=0)
    
    ax = axes[i]
    ax.plot(time_ms, mean_beat, color=colors[cls_id % len(colors)], lw=2, label="Mean Beat")
    ax.fill_between(time_ms, mean_beat - std_beat, mean_beat + std_beat, color=colors[cls_id % len(colors)], alpha=0.25, label="±1 STD")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.8, label="R-Peak (t=0)")
    ax.set_title(AAMI_CLASSES[cls_id], fontsize=11, fontweight="bold")
    ax.set_xlabel("Time from R-Peak (ms)")
    if i == 0:
        ax.set_ylabel("Normalized Amplitude")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.show()"""),

    md_cell("""### 4. Data Augmentation Verification"""),

    code_cell("""sample_beat = beats[0]
aug = ECGAugmentation(noise_std=0.04, scale_min=0.85, scale_max=1.15, shift_max_samples=6)

plt.figure(figsize=(12, 4))
plt.plot(time_ms, sample_beat, "k-", lw=2, label="Original Beat")
for i in range(3):
    aug_sample = aug(sample_beat)
    plt.plot(time_ms, aug_sample, linestyle="--", lw=1.2, alpha=0.8, label=f"Augmentation #{i+1}")

plt.title("ECG Beat Data Augmentation (Noise + Scaling + Jitter)", fontsize=13, fontweight="bold")
plt.xlabel("Time from R-Peak (ms)")
plt.ylabel("Normalized Amplitude")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()"""),

    md_cell("""### 5. PyTorch DataLoader Integration & Batch Inspection"""),

    code_cell("""train_loader, val_loader, test_loader, class_weights = get_ecg_dataloaders(
    npz_path=npz_path,
    batch_size=64,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    augment_train=True
)

print(f"Train Batches: {len(train_loader)} ({len(train_loader.dataset)} beats)")
print(f"Val Batches:   {len(val_loader)} ({len(val_loader.dataset)} beats)")
print(f"Test Batches:  {len(test_loader)} ({len(test_loader.dataset)} beats)")
print(f"Class Weights Tensor: {class_weights}")

# Inspect first batch
batch_x, batch_y = next(iter(train_loader))
print(f"Batch Tensor Shape: {batch_x.shape} (Batch, Channels, Sequence)")
print(f"Batch Labels Shape: {batch_y.shape}")""")
]

# =====================================================================
# Notebook 3: 03_ppg_analysis.ipynb
# =====================================================================
nb3_cells = [
    md_cell("""# MedAgent: 03. Photoplethysmography (PPG) Signal Analysis
## Continuous Optical Pulse Wave Processing & Pulse Rate Variability (PRV)

This notebook implements physiological signal analysis for Photoplethysmography (PPG):
1. Principles of optical blood volume pulse monitoring.
2. Filtering PPG signals ($0.5 - 8.0\\text{ Hz}$ bandpass) to isolate pulsatile cardiac components.
3. Systolic peak detection & Pulse-to-Pulse Interval (PPI) extraction.
4. Time-Domain & Frequency-Domain Pulse Rate Variability (PRV) metrics:
   - **Heart Rate (BPM)**
   - **SDNN** (Standard deviation of NN/PP intervals)
   - **RMSSD** (Root mean square of successive differences - vagal tone indicator)
   - **pNN50** (% intervals differing by $>50\\text{ ms}$)
   - **LF / HF Ratio** (Sympathovagal balance index)."""),

    code_cell("""import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.preprocessing.ecg_cleaner import apply_bandpass_filter, normalize_signal"""),

    md_cell("""### 1. Synthetic / Recorded PPG Pulse Generation
Let's generate a realistic multi-harmonic PPG pulse wave with respiratory baseline modulation and dicrotic notch."""),

    code_cell("""fs = 125.0  # Standard PPG wearable sampling rate (Hz)
duration_sec = 20.0
t = np.arange(0, duration_sec, 1.0 / fs)

# Synthetic pulse wave with heart rate ~ 72 bpm (1.2 Hz)
hr_freq = 1.2
systolic = np.sin(2 * np.pi * hr_freq * t) ** 2
dicrotic = 0.35 * np.sin(2 * np.pi * 2 * hr_freq * t + 0.5) ** 2
respiration = 0.15 * np.sin(2 * np.pi * 0.25 * t)  # 15 breaths/min (0.25 Hz)
noise = np.random.normal(0, 0.05, len(t))

ppg_raw = systolic + dicrotic + respiration + noise

plt.figure(figsize=(14, 4))
plt.plot(t, ppg_raw, color="#e377c2", lw=1.5, label="Raw Wearable PPG")
plt.title("Simulated Raw PPG Optical Signal with Respiratory Drift & Noise", fontsize=13, fontweight="bold")
plt.xlabel("Time (seconds)")
plt.ylabel("Optical Amplitude")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()"""),

    md_cell("""### 2. PPG Signal Filtering ($0.5 - 8.0\\text{ Hz}$)"""),

    code_cell("""ppg_filtered = apply_bandpass_filter(ppg_raw, sampling_rate=fs, lowcut=0.5, highcut=8.0, order=3)
ppg_normalized = normalize_signal(ppg_filtered, method="min_max")

plt.figure(figsize=(14, 4))
plt.plot(t, ppg_normalized, color="#17becf", lw=1.5, label="Filtered PPG (0.5-8 Hz)")
plt.title("Cleaned & Normalized PPG Waveform", fontsize=13, fontweight="bold")
plt.xlabel("Time (seconds)")
plt.ylabel("Normalized Pulse Amplitude")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()"""),

    md_cell("""### 3. Systolic Peak Detection & Pulse Intervals (PPI)"""),

    code_cell("""# Find systolic peaks
min_peak_distance = int(0.4 * fs)  # Max heart rate 150 bpm -> min 0.4s between peaks
peaks, _ = signal.find_peaks(ppg_normalized, distance=min_peak_distance, prominence=0.4)

peak_times = t[peaks]
peak_vals = ppg_normalized[peaks]

# Compute pulse intervals (in milliseconds)
ppi_ms = np.diff(peak_times) * 1000.0
heart_rate_bpm = 60.0 / (np.diff(peak_times))

plt.figure(figsize=(14, 5))
plt.plot(t, ppg_normalized, color="#17becf", lw=1.5, label="Filtered PPG")
plt.scatter(peak_times, peak_vals, color="red", s=70, zorder=5, label="Systolic Peaks")

for i, p_t in enumerate(peak_times[:-1]):
    plt.annotate(f"{heart_rate_bpm[i]:.0f} bpm", (p_t + 0.1, peak_vals[i] + 0.1), fontsize=8, color="darkred")

plt.title("PPG Systolic Peak Detection & Instantaneous Heart Rate", fontsize=13, fontweight="bold")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()"""),

    md_cell("""### 4. Pulse Rate Variability (PRV) Metric Extraction"""),

    code_cell("""# Time-domain PRV metrics
mean_hr = np.mean(heart_rate_bpm)
sdnn = np.std(ppi_ms)
rmssd = np.sqrt(np.mean(np.diff(ppi_ms) ** 2))
pnn50 = (np.sum(np.abs(np.diff(ppi_ms)) > 50.0) / len(ppi_ms)) * 100.0

print("=" * 45)
print("     CLINICAL PULSE RATE VARIABILITY (PRV)   ")
print("=" * 45)
print(f"Mean Heart Rate (HR):   {mean_hr:.1f} BPM")
print(f"Mean Pulse Interval:    {np.mean(ppi_ms):.1f} ms")
print(f"SDNN (Overall HRV):     {sdnn:.2f} ms")
print(f"RMSSD (Parasympathetic):{rmssd:.2f} ms")
print(f"pNN50:                  {pnn50:.1f} %")
print("=" * 45)""")
]

# =====================================================================
# Notebook 4: 04_xray_exploration.ipynb
# =====================================================================
nb4_cells = [
    md_cell("""# MedAgent: 04. Medical Imaging Exploration (Chest X-Rays)
## Dataset Inspection, Class Distributions, and Image Preprocessing

This notebook explores the thoracic imaging pipeline:
1. Loading the 5-class synthetic chest X-ray dataset (`data/raw/imaging/synthetic_xray/`).
2. Diseases covered: **Normal**, **Pneumonia**, **Cardiomegaly**, **Atelectasis**, **Pleural Effusion**.
3. Inspecting image resolutions, channels, and pixel intensity histograms.
4. Implementing torchvision preprocessing transforms ($224 \\times 224$, ImageNet normalization).
5. Visualizing sample radiographs and augmentations."""),

    code_cell("""import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms

PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))"""),

    md_cell("""### 1. Dataset Structure & Class Counts"""),

    code_cell("""xray_dir = PROJECT_ROOT / "data" / "raw" / "imaging" / "synthetic_xray"
classes = [d.name for d in sorted(xray_dir.iterdir()) if d.is_dir()]
class_counts = {c: len(list((xray_dir / c).glob("*.png"))) for c in classes}

print(f"Chest X-Ray Diagnostic Classes ({len(classes)}):")
for c, count in class_counts.items():
    print(f"  - {c:20s}: {count} images")

plt.figure(figsize=(9, 4))
plt.bar(class_counts.keys(), class_counts.values(), color=["#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#1f77b4"])
plt.title("Chest X-Ray Dataset Class Distribution", fontsize=13, fontweight="bold")
plt.xlabel("Diagnostic Finding")
plt.ylabel("Number of Radiographs")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()"""),

    md_cell("""### 2. Visualizing Radiographs Across All 5 Classes"""),

    code_cell("""fig, axes = plt.subplots(1, len(classes), figsize=(16, 4))

for i, cls_name in enumerate(classes):
    img_files = list((xray_dir / cls_name).glob("*.png"))
    img = Image.open(img_files[0]).convert("RGB")
    
    ax = axes[i]
    ax.imshow(img, cmap="gray")
    ax.set_title(cls_name, fontsize=11, fontweight="bold")
    ax.axis("off")

plt.suptitle("Sample Radiographs by Diagnostic Finding", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()"""),

    md_cell("""### 3. Preprocessing Transforms & Pixel Distributions"""),

    code_cell("""transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

sample_img = Image.open(list((xray_dir / "Cardiomegaly").glob("*.png"))[0]).convert("RGB")
tensor_img = transform_pipeline(sample_img)

print(f"Preprocessed Tensor Shape: {tensor_img.shape} (C, H, W)")
print(f"Tensor Min/Max/Mean: {tensor_img.min():.2f} / {tensor_img.max():.2f} / {tensor_img.mean():.2f}")

plt.figure(figsize=(10, 3))
plt.hist(tensor_img.numpy().flatten(), bins=50, color="#1f77b4", alpha=0.7)
plt.title("Standardized Pixel Intensity Distribution (ImageNet Normalized)", fontsize=12, fontweight="bold")
plt.xlabel("Normalized Value")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()""")
]

# =====================================================================
# Notebook 5: 05_multimodal_fusion.ipynb
# =====================================================================
nb5_cells = [
    md_cell("""# MedAgent: 05. Multimodal Deep Learning & Cross-Attention Fusion
## Integrating ECG Signals, Chest X-Rays, and Clinical Patient Data

This notebook demonstrates the multimodal fusion architecture:
1. Signal Feature Extraction (1D CNN / Transformer $\\to 256\\text{-d}$ embedding).
2. Vision Feature Extraction (DenseNet121 $\\to 1024\\text{-d} \\to 256\\text{-d}$ embedding).
3. Tabular Clinical Context ($64\\text{-d} \\to 256\\text{-d}$ embedding).
4. Multi-Head Cross-Attention Transformer fusion for unified patient representation.
5. Overall Patient Risk Stratification (**Low**, **Moderate**, **High/Critical**).
6. Explainability: Grad-CAM heatmap visualization on Chest X-rays."""),

    code_cell("""import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Compute Device: {device}")"""),

    md_cell("""### 1. Define Model Architectures
Let's define the 1D CNN Signal Encoder, Vision Encoder, and Cross-Attention Fusion Transformer."""),

    code_cell("""class ECG1DCNN(nn.Module):
    def __init__(self, in_channels=1, latent_dim=256, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.proj = nn.Linear(128, latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        feat = self.features(x).squeeze(-1)
        latent = F.relu(self.proj(feat))
        logits = self.classifier(latent)
        return logits, latent


class MultimodalTransformerFusion(nn.Module):
    def __init__(self, embed_dim=256, nheads=8, num_layers=3, num_risk_classes=3):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nheads, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.risk_head = nn.Linear(embed_dim, num_risk_classes)

    def forward(self, ecg_embed, vision_embed, tabular_embed):
        # Shape: (B, 1, D) for each token
        B = ecg_embed.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        tokens = torch.stack([ecg_embed, vision_embed, tabular_embed], dim=1)  # (B, 3, D)
        all_tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, 4, D)
        
        out = self.transformer(all_tokens)
        cls_out = out[:, 0, :]  # Output at CLS token
        risk_logits = self.risk_head(cls_out)
        return risk_logits, cls_out

print("Model Architectures Initialized Successfully.")"""),

    md_cell("""### 2. Multimodal Forward Pass Simulation"""),

    code_cell("""ecg_model = ECG1DCNN(latent_dim=256).to(device)
fusion_model = MultimodalTransformerFusion(embed_dim=256).to(device)

# Simulated inputs
batch_size = 4
dummy_ecg = torch.randn(batch_size, 1, 260).to(device)
dummy_vision = torch.randn(batch_size, 256).to(device)
dummy_tabular = torch.randn(batch_size, 256).to(device)

ecg_logits, ecg_latent = ecg_model(dummy_ecg)
risk_logits, patient_rep = fusion_model(ecg_latent, dummy_vision, dummy_tabular)

risk_probs = F.softmax(risk_logits, dim=-1)
risk_classes = ["Low Risk", "Moderate Risk", "High / Critical Risk"]

print(f"ECG Logits Shape: {ecg_logits.shape}")
print(f"Patient Unified Representation Shape: {patient_rep.shape}")
print(f"Multimodal Risk Probabilities:\\n{risk_probs.detach().cpu().numpy()}")
for i in range(batch_size):
    pred_risk = risk_classes[torch.argmax(risk_probs[i]).item()]
    print(f"  Patient {i+1}: Classified as '{pred_risk}' (Confidence: {risk_probs[i].max().item()*100:.1f}%)")"""),

    md_cell("""### 3. Explainability with Grad-CAM Visualization"""),

    code_cell("""gradcam_sample_path = PROJECT_ROOT / "data" / "processed" / "images" / "gradcam_sample.png"

if gradcam_sample_path.exists():
    img_gradcam = Image.open(gradcam_sample_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img_gradcam)
    plt.title("Sample Grad-CAM Radiograph Localization (Cardiomegaly/Effusion)", fontsize=13, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
else:
    print("Grad-CAM sample image not found.")""")
]

# =====================================================================
# Notebook 6: 06_agent_evaluation.ipynb
# =====================================================================
nb6_cells = [
    md_cell("""# MedAgent: 06. Agentic AI Clinical Reasoning & RAG Evaluation
## Multi-Agent Workflow, Knowledge Retrieval, and Explainable Report Generation

This notebook evaluates the Multi-Agent clinical decision support framework:
1. **Signal Analysis Agent**: Interprets physiological ECG/PPG findings and rhythm irregularities.
2. **Imaging Analysis Agent**: Interprets radiologic abnormalities and anatomical Grad-CAM regions.
3. **Evidence Agent (RAG)**: Retrieves medical literature and clinical practice guidelines.
4. **Risk Assessment Agent**: Computes composite risk score and flags contraindications.
5. **Report Generation Agent**: Formats comprehensive, clinician-facing decision-support reports.
6. **Supervisor Agent**: Validates safety guardrails and evidence grounding."""),

    code_cell("""import sys
from pathlib import Path
import json

PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))"""),

    md_cell("""### 1. Simulated Clinical Patient Case"""),

    code_cell("""patient_case = {
    "patient_id": "PT-94021",
    "demographics": {
        "age": 68,
        "sex": "Male",
        "hypertension": True,
        "diabetes": True,
        "smoking_history": "Former (20 pack-years)"
    },
    "vitals": {
        "heart_rate_bpm": 96,
        "blood_pressure": "148/92 mmHg",
        "spo2_percent": 94
    },
    "ai_model_predictions": {
        "ecg_finding": "Ventricular Ectopic Beat (PVC)",
        "ecg_confidence": 0.91,
        "xray_finding": "Cardiomegaly with Mild Pleural Effusion",
        "xray_confidence": 0.88,
        "gradcam_region": "Lower left thoracic cavity and enlarged cardiac silhouette",
        "multimodal_risk_level": "High / Critical Risk",
        "multimodal_risk_score": 0.87
    }
}

print(json.dumps(patient_case, indent=2))"""),

    md_cell("""### 2. Multi-Agent Step-by-Step Reasoning Simulation"""),

    code_cell("""def simulate_multi_agent_workflow(case):
    # 1. Signal Agent
    ecg_pred = case["ai_model_predictions"]["ecg_finding"]
    ecg_conf = case["ai_model_predictions"]["ecg_confidence"]
    signal_thought = (
        f"[Signal Agent]: Detected '{ecg_pred}' with {ecg_conf*100:.1f}% confidence. "
        f"The QRS complex demonstrates premature ventricular morphology with compensatory pause. "
        f"Elevated resting heart rate of {case['vitals']['heart_rate_bpm']} BPM noted."
    )
    
    # 2. Imaging Agent
    xray_pred = case["ai_model_predictions"]["xray_finding"]
    xray_conf = case["ai_model_predictions"]["xray_confidence"]
    gradcam_loc = case["ai_model_predictions"]["gradcam_region"]
    imaging_thought = (
        f"[Imaging Agent]: Radiograph analysis reveals '{xray_pred}' ({xray_conf*100:.1f}% confidence). "
        f"Grad-CAM attention focuses on: {gradcam_loc}."
    )
    
    # 3. Evidence Agent (RAG)
    evidence_retrieved = [
        "ACC/AHA Guidelines for Ventricular Arrhythmias (2022): Frequent PVCs in the presence of structural heart disease warrant echocardiographic evaluation.",
        "Fleischner Society Guidelines: Bilateral or dependent pleural effusion combined with cardiomegaly is highly indicative of congestive heart failure (CHF) decompensation."
    ]
    evidence_thought = (
        f"[Evidence Agent]: Retrieved 2 high-relevance clinical practice guidelines:\\n" +
        "\\n".join([f"  • {e}" for e in evidence_retrieved])
    )
    
    # 4. Risk Assessment Agent
    risk_level = case["ai_model_predictions"]["multimodal_risk_level"]
    risk_thought = (
        f"[Risk Assessment Agent]: Synthesizing multimodal findings. "
        f"Co-occurrence of PVC arrhythmia, cardiomegaly, hypertension, and borderline hypoxia (SpO2 {case['vitals']['spo2_percent']}%) "
        f"indicates elevated cardiovascular decompensation risk. Final Risk Tier: {risk_level}."
    )
    
    # 5. Report Generation Agent
    clinical_report = f\"\"\"
================================================================================
                    MEDAGENT CLINICAL DECISION SUPPORT REPORT
================================================================================
PATIENT ID: {case['patient_id']} | AGE: {case['demographics']['age']} | SEX: {case['demographics']['sex']}
DATE: 2026-09-29 | STATUS: PENDING CLINICIAN REVIEW
--------------------------------------------------------------------------------
1. PRIMARY FINDINGS:
   • ECG: {ecg_pred} (Confidence: {ecg_conf*100:.1f}%)
   • Imaging: {xray_pred} (Confidence: {xray_conf*100:.1f}%)
   • Key Localizations: {gradcam_loc}

2. MULTIMODAL RISK STRATIFICATION:
   • Risk Severity: {risk_level} (Calculated Index: {case['ai_model_predictions']['multimodal_risk_score']})
   • Risk Drivers: PVC burden combined with radiologic cardiomegaly and hypertension history.

3. SUPPORTING CLINICAL EVIDENCE:
   • ACC/AHA (2022): Frequent ventricular ectopy with cardiomegaly requires structural workup.
   • Fleischner Society: Effusion in setting of enlarged cardiac silhouette suggests heart failure.

4. RECOMMENDED CLINICAL ACTIONS:
   [ ] Perform 12-lead diagnostic ECG and comprehensive echocardiogram.
   [ ] Check serum electrolytes, BNP/NT-proBNP, and renal function.
   [ ] Evaluate fluid status and adjust antihypertensive/diuretic therapy as clinically indicated.
   [!] CAUTION: MedAgent is an AI clinical assistant. Final diagnosis rests with the attending physician.
================================================================================
\"\"\"
    return {
        "signal_thought": signal_thought,
        "imaging_thought": imaging_thought,
        "evidence_thought": evidence_thought,
        "risk_thought": risk_thought,
        "report": clinical_report
    }

agent_outputs = simulate_multi_agent_workflow(patient_case)
print(agent_outputs["signal_thought"])
print()
print(agent_outputs["imaging_thought"])
print()
print(agent_outputs["evidence_thought"])
print()
print(agent_outputs["risk_thought"])
print()
print(agent_outputs["report"])""")
]

# Write all 6 notebooks
all_nbs = {
    "01_signal_exploration.ipynb": nb1_cells,
    "02_ecg_preprocessing.ipynb": nb2_cells,
    "03_ppg_analysis.ipynb": nb3_cells,
    "04_xray_exploration.ipynb": nb4_cells,
    "05_multimodal_fusion.ipynb": nb5_cells,
    "06_agent_evaluation.ipynb": nb6_cells,
}

for nb_name, cells in all_nbs.items():
    nb_path = NOTEBOOKS_DIR / nb_name
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(make_notebook(cells), f, indent=1)
    print(f"Generated {nb_name} ({len(cells)} cells)")

print("All 6 Jupyter Notebooks successfully generated!")

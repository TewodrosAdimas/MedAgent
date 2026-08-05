# MedAgent Data Dictionary

## 1. Introduction

This document defines the data structure, metadata, and interpretation of all information used in the **MedAgent** system.

MedAgent uses multimodal healthcare data:

* Physiological signals
* Medical images
* Patient information
* Clinical observations
* AI-generated predictions
* Agent reasoning outputs

The purpose of this dictionary is to ensure consistency across:

* Data preprocessing
* Model training
* Evaluation
* Deployment
* Clinical interpretation

---

# 2. Data Architecture Overview

```text
MedAgent Data


├── Patient Information
│
├── Medical Signals
│   ├── ECG
│   ├── PPG
│   ├── EDA
│   ├── Accelerometer
│   └── SpO2
│
├── Medical Images
│   ├── X-ray
│   ├── CT
│   └── MRI
│
├── Clinical Context
│
├── Model Outputs
│
└── Agent Reports

```

---

# 3. Patient Information Dataset

## Purpose

Stores demographic and clinical context required for personalized analysis.

---

## Patient Schema

| Field           | Type        | Description               |
| --------------- | ----------- | ------------------------- |
| patient_id      | string      | Unique patient identifier |
| age             | integer     | Patient age               |
| sex             | categorical | Biological sex            |
| height_cm       | float       | Height                    |
| weight_kg       | float       | Weight                    |
| medical_history | text        | Previous conditions       |
| medications     | text        | Current medications       |
| created_at      | datetime    | Record creation time      |

---

Example:

```json
{
 "patient_id": "001",
 "age": 65,
 "medical_history": [
    "hypertension"
 ],
 "medications": [
    "beta_blocker"
 ]
}
```

---

# 4. Medical Signal Data

## 4.1 ECG Dataset

## Purpose

Electrocardiogram data represents electrical activity of the heart.

Applications:

* Arrhythmia detection
* Heart rhythm analysis
* Cardiac risk estimation

---

## ECG Schema

| Field            | Type        | Description          |
| ---------------- | ----------- | -------------------- |
| patient_id       | string      | Patient identifier   |
| timestamp        | datetime    | Measurement time     |
| sampling_rate    | integer     | Samples per second   |
| signal           | array       | ECG waveform values  |
| duration_seconds | float       | Recording duration   |
| lead_type        | categorical | ECG lead information |
| quality_score    | float       | Signal quality score |

---

Example:

```json
{
 "patient_id": "001",
 "sampling_rate": 250,
 "duration_seconds": 10,
 "lead_type": "II",
 "signal": [
    0.12,
    0.14,
    0.18
 ]
}
```

---

# 4.2 PPG Dataset

## Purpose

Photoplethysmography measures blood volume changes through optical sensing.

Applications:

* Heart rate estimation
* Blood circulation analysis
* Stress monitoring

---

## PPG Schema

| Field             | Type     | Description        |
| ----------------- | -------- | ------------------ |
| patient_id        | string   | Patient identifier |
| timestamp         | datetime | Measurement time   |
| sampling_rate     | integer  | Frequency          |
| signal            | array    | PPG waveform       |
| heart_rate        | float    | Estimated BPM      |
| oxygen_saturation | float    | SpO2 value         |

---

Example:

```json
{
 "patient_id": "001",
 "heart_rate": 78,
 "oxygen_saturation": 97
}
```

---

# 4.3 EDA Dataset

## Purpose

Electrodermal activity represents changes in skin conductance.

Applications:

* Stress detection
* Emotional state analysis
* Autonomic nervous system monitoring

---

## EDA Schema

| Field            | Type     | Description             |
| ---------------- | -------- | ----------------------- |
| patient_id       | string   | Patient identifier      |
| timestamp        | datetime | Measurement time        |
| sampling_rate    | integer  | Frequency               |
| signal           | array    | Skin conductance values |
| tonic_component  | float    | Baseline activity       |
| phasic_component | float    | Rapid response          |

---

# 4.4 Accelerometer Dataset

## Purpose

Captures movement and physical activity.

Applications:

* Activity recognition
* Fall detection
* Exercise monitoring

---

## Accelerometer Schema

| Field         | Type     | Description        |
| ------------- | -------- | ------------------ |
| patient_id    | string   | Patient identifier |
| timestamp     | datetime | Measurement time   |
| x_axis        | float    | X acceleration     |
| y_axis        | float    | Y acceleration     |
| z_axis        | float    | Z acceleration     |
| sampling_rate | integer  | Frequency          |

---

Example:

```json
{
 "x_axis":0.12,
 "y_axis":0.04,
 "z_axis":9.81
}
```

---

# 5. Medical Imaging Data

## 5.1 Image Metadata

| Field            | Type        | Description             |
| ---------------- | ----------- | ----------------------- |
| image_id         | string      | Unique image identifier |
| patient_id       | string      | Patient identifier      |
| modality         | categorical | X-ray, CT, MRI          |
| image_path       | string      | Storage location        |
| width            | integer     | Image width             |
| height           | integer     | Image height            |
| acquisition_date | datetime    | Acquisition time        |
| diagnosis_label  | string      | Ground truth label      |

---

Example:

```json
{
 "image_id":"xray_001",
 "modality":"chest_xray",
 "diagnosis_label":"pneumonia"
}
```

---

# 6. Clinical Context Data

## Purpose

Additional information used by reasoning agents.

---

## Clinical Context Schema

| Field              | Type   | Description        |
| ------------------ | ------ | ------------------ |
| patient_id         | string | Patient identifier |
| symptoms           | text   | Reported symptoms  |
| diagnosis_history  | text   | Previous diagnosis |
| laboratory_results | JSON   | Lab values         |
| physician_notes    | text   | Clinical notes     |

---

Example:

```json
{
 "symptoms":[
    "shortness_of_breath"
 ],
 "lab_results":{
    "oxygen":"95%"
 }
}
```

---

# 7. Preprocessed Data

Processed data is stored separately from raw data.

Structure:

```text
data/processed/

├── signals/
│   ├── ecg/
│   ├── ppg/
│   └── eda/
│
├── images/
│
└── multimodal/

```

---

## Signal Processing Output

| Field             | Description        |
| ----------------- | ------------------ |
| normalized_signal | Scaled waveform    |
| segment_id        | Window identifier  |
| features          | Extracted features |
| label             | Training target    |

---

# 8. Model Input Data

## Signal Model Input

Format:

```text
(batch_size, channels, sequence_length)
```

Example:

```text
(32,1,2500)
```

Meaning:

```text
32 samples
1 ECG channel
2500 time points
```

---

## Vision Model Input

Format:

```text
(batch_size,height,width,channels)
```

Example:

```text
(32,224,224,3)
```

---

## Multimodal Input

Example:

```json
{
 "signal_embedding":"vector",
 "image_embedding":"vector",
 "patient_features":"vector"
}
```

---

# 9. Model Prediction Data

## Prediction Schema

| Field         | Type     | Description        |
| ------------- | -------- | ------------------ |
| prediction_id | string   | Unique prediction  |
| patient_id    | string   | Patient identifier |
| model_name    | string   | Model used         |
| prediction    | string   | Output class       |
| confidence    | float    | Probability        |
| timestamp     | datetime | Prediction time    |

---

Example:

```json
{
 "model_name":"ECGTransformer",
 "prediction":"arrhythmia_risk",
 "confidence":0.87
}
```

---

# 10. Agent Output Data

## Agent Report Schema

| Field           | Type        | Description        |
| --------------- | ----------- | ------------------ |
| report_id       | string      | Report identifier  |
| patient_id      | string      | Patient            |
| findings        | JSON        | Extracted findings |
| risk_level      | categorical | Low/Medium/High    |
| explanation     | text        | Reasoning          |
| recommendations | text        | Suggested actions  |

---

Example:

```json
{
 "risk_level":"moderate",
 "findings":[
    "abnormal ECG",
    "lung opacity"
 ],
 "recommendation":
 "clinical review recommended"
}
```

---

# 11. Data Quality Requirements

## Signal Quality

Required:

* Minimal missing values
* Correct sampling rate
* Artifact detection
* Valid physiological ranges

---

## Image Quality

Required:

* Correct modality
* Sufficient resolution
* Valid metadata
* No corrupted files

---

# 12. Privacy and Security Considerations

Medical data requires strict protection.

Requirements:

* Patient anonymization
* Removal of personally identifiable information
* Secure storage
* Access control
* Encryption

---

# 13. Dataset Organization

Final repository structure:

```text
data/

├── raw/

│   ├── ecg/
│   ├── ppg/
│   ├── eda/
│   ├── accelerometer/
│   └── imaging/


├── processed/

│   ├── signals/
│   ├── images/
│   └── multimodal/


└── external/

    └── public_datasets/

```

---

# 14. Summary

The MedAgent data architecture supports multimodal healthcare AI by organizing:

1. Patient information
2. Physiological signals
3. Medical images
4. Clinical context
5. Model predictions
6. Agent-generated reports

A consistent data structure enables reproducible experiments, reliable model training, and safe deployment.

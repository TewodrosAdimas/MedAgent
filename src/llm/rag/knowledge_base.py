"""
Curated Medical Knowledge Base for Evidence-Grounded Clinical Decision Support.

Contains structured clinical practice guidelines:
- AHA / ACC / HRS Clinical Guidelines for Cardiac Arrhythmias & ECG Analysis
- Fleischner Society Thoracic Imaging & Chest Radiograph Diagnostic Guidelines
- ACC / AHA Multimodal Cardiovascular Risk Stratification & Decompensation Protocols
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ClinicalGuideline:
    id: str
    modality: str  # 'ecg', 'xray', 'multimodal', 'cardiovascular'
    topic: str
    title: str
    content: str
    citation: str
    strength_of_recommendation: str  # 'Class I (Strong)', 'Class IIa (Moderate)', etc.


# Curated Evidence Corpus
GUIDELINE_CORPUS: List[ClinicalGuideline] = [
    ClinicalGuideline(
        id="AHA-ECG-001",
        modality="ecg",
        topic="Ventricular Arrhythmias",
        title="ACC/AHA/HRS Guidelines for Management of Patients With Ventricular Arrhythmias",
        content=(
            "Premature ventricular complexes (PVCs) and non-sustained ventricular tachycardia in patients "
            "with known or suspected structural heart disease warrant comprehensive diagnostic evaluation, "
            "including 12-lead ECG, 24-48h Holter monitoring, and transthoracic echocardiography to rule out "
            "ischemic or non-ischemic cardiomyopathy."
        ),
        citation="Al-Khatib SM et al., Circulation. 2018;138(13):e272-e391.",
        strength_of_recommendation="Class I (Strong)",
    ),
    ClinicalGuideline(
        id="AHA-ECG-002",
        modality="ecg",
        topic="Supraventricular Tachycardia & Ectopy",
        title="ACC/AHA/HRS Guideline for the Management of Adult Patients With Supraventricular Tachycardia",
        content=(
            "Frequent supraventricular premature beats (PACs) are associated with increased risk of incident "
            "atrial fibrillation and stroke. Evaluation of underlying triggers (hypertension, sleep apnea, "
            "electrolyte imbalance) and clinical monitoring is indicated."
        ),
        citation="Page RL et al., J Am Coll Cardiol. 2016;67(13):e27-e115.",
        strength_of_recommendation="Class IIa (Moderate)",
    ),
    ClinicalGuideline(
        id="FLEISCHNER-XRAY-001",
        modality="xray",
        topic="Cardiomegaly & Heart Failure",
        title="Fleischner Society Recommendations: Radiographic Patterns of Cardiothoracic Disease",
        content=(
            "Enlargement of the cardiac silhouette (cardiothoracic ratio > 0.50 on PA chest radiograph) "
            "associated with upper zone vascular redistribution, interstitial edema, or bilateral pleural effusions "
            "is strongly suggestive of elevated left ventricular filling pressures and congestive heart failure decompensation."
        ),
        citation="Hansell DM et al., Radiology. 2008;246(3):697-722.",
        strength_of_recommendation="Class I (Strong)",
    ),
    ClinicalGuideline(
        id="FLEISCHNER-XRAY-002",
        modality="xray",
        topic="Pneumonia & Pulmonary Consolidation",
        title="ATS/IDSA Guidelines on the Diagnosis and Treatment of Community-Acquired Pneumonia",
        content=(
            "A demonstrable radiographic infiltrate, consolidation, or opacity on chest radiograph or CT, "
            "in the presence of clinical features such as fever, cough, dyspnea, or leukocytosis, is required "
            "for the definitive diagnosis of community-acquired pneumonia."
        ),
        citation="Metlay JP et al., Am J Respir Crit Care Med. 2019;200(7):e45-e67.",
        strength_of_recommendation="Class I (Strong)",
    ),
    ClinicalGuideline(
        id="FLEISCHNER-XRAY-003",
        modality="xray",
        topic="Pleural Effusion & Atelectasis",
        title="BTS Guidelines for the Investigation of Unilateral and Bilateral Pleural Effusions",
        content=(
            "Dependent pleural effusion with blunting of the costophrenic angle and compressive atelectasis "
            "warrants diagnostic thoracentesis if of unknown etiology, or clinical optimization with diuretic therapy "
            "if clinically consistent with heart failure."
        ),
        citation="Hooper C et al., Thorax. 2010;65(Suppl 2):ii4-ii17.",
        strength_of_recommendation="Class I (Strong)",
    ),
    ClinicalGuideline(
        id="ACC-MULTI-001",
        modality="multimodal",
        topic="Multimodal Cardiovascular Risk Stratification",
        title="2019 ACC/AHA Guideline on the Primary Prevention of Cardiovascular Disease",
        content=(
            "Integration of physiological biosignal anomalies (e.g., resting tachycardia, ectopy), radiographic "
            "cardiomegaly, and traditional risk factors (hypertension, diabetes, age, smoking) significantly improves "
            "10-year atherosclerotic cardiovascular disease (ASCVD) risk categorization over isolated single-modality scoring."
        ),
        citation="Arnett DK et al., Circulation. 2019;140(11):e596-e646.",
        strength_of_recommendation="Class I (Strong)",
    ),
    ClinicalGuideline(
        id="AHA-SAFETY-001",
        modality="multimodal",
        topic="AI Clinical Decision Support Safety",
        title="Principles for Clinical AI Evaluation and Human-in-the-Loop Governance",
        content=(
            "AI-derived diagnostic observations and risk assessments must serve as auxiliary decision support. "
            "All findings, automated calibrations, and therapeutic recommendations require validation by a licensed "
            "attending clinician before modifying patient management."
        ),
        citation="Sendak MP et al., NEJM Catalyst Innovations in Care Delivery. 2020;1(6).",
        strength_of_recommendation="Standard of Care",
    ),
]

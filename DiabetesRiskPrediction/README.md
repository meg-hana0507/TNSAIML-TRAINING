# 🩺 GlycoGuard AI — Clinical Diabetes Risk Prediction System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65%2B-FF4B4B.svg?logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Model](https://img.shields.io/badge/Model-Random%20Forest%20Classifier-success.svg)](#model-architecture--performance)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.812-brightgreen.svg)](#model-architecture--performance)

**GlycoGuard AI** is a medical-grade web application and machine learning decision-support system designed for early risk assessment and biomarker stratification of **Type 2 Diabetes Mellitus**.

Trained on the clinical **Pima Indians Diabetes Dataset**, the system combines robust data preprocessing, median imputation for physiological non-measurements, and an ensemble **Random Forest Classifier** wrapped in a modern, high-contrast, interactive Streamlit dashboard.

---

## 🌟 Key Features

- **🧪 Single Patient Clinical Assessment**:
  - Interactive sliders and number inputs for all 8 vital biomarkers.
  - One-click clinical archetype presets (*Low Risk*, *Borderline / Pre-Diabetic*, *High Risk*, *Cohort Medians*).
  - ADA (American Diabetes Association) guideline-aligned biomarker status indicators.
- **🎯 Live Semicircular Risk Gauge**:
  - Real-time polar gauge with visual risk zones (🟢 **Low <35%**, 🟡 **Moderate 35-60%**, 🔴 **High >60%**) and an indicator needle updating dynamically.
- **⚡ Interactive "What-If" Lifestyle Intervention Simulator**:
  - Experiment with target reductions in glucose, BMI, or blood pressure to calculate immediate absolute risk reductions in real time.
- **📐 Built-In Dual-Unit BMI Calculator**:
  - Convert Height & Weight (Metric `cm/kg` or Imperial `in/lbs`) directly to BMI and apply it to the assessment with one click.
- **📁 Cohort Batch Screening (CSV Upload)**:
  - Upload patient rosters in bulk to screen entire clinics or registries simultaneously.
  - Interactive table with progress bars, risk category tagging, and one-click downloadable annotated reports (`.csv`).
  - Built-in downloadable sample CSV template for instant testing.
- **🧠 Model Intelligence & Explainability**:
  - Gini feature importance rankings (Glucose & BMI as primary risk factors).
  - Confusion matrix and full classification report.
- **📈 Population Cohort Analytics**:
  - Interactive biomarker distribution charts (histograms, scatter correlations, age densities).
- **🛡️ ADA Standards & Medical Disclaimer**:
  - Contextualized against American Diabetes Association diagnostic cutoffs.

---

## 📊 Model Architecture & Performance

The model pipeline employs a **Random Forest Classifier** with balanced class weighting and a training-only **median imputation strategy** to account for physiological zeros in unrecorded tests (e.g., zero glucose or insulin).

| Metric | Score | Description |
| :--- | :---: | :--- |
| **ROC-AUC** | **0.8123** | Area under the Receiver Operating Characteristic curve |
| **Accuracy** | **74.03%** | Overall correct classification rate on held-out test data |
| **Precision** | **65.22%** | Positive predictive value for diabetic outcomes |
| **Recall (Sensitivity)** | **55.56%** | Ability to detect true diabetic cases |
| **F1-Score** | **0.6000** | Harmonic mean of precision and recall |

### 📌 Biomarker Feature Importance (Gini Score)
1. **Glucose**: 26.1%
2. **BMI (Body Mass Index)**: 16.7%
3. **Age**: 12.6%
4. **Diabetes Pedigree Function (DPF)**: 11.8%
5. **Insulin**: 9.9%
6. **Blood Pressure**: 8.2%
7. **Pregnancies**: 7.5%
8. **Skin Thickness**: 7.3%

---

## 📁 Repository Structure

```text
DiabetesRiskPrediction/
├── .streamlit/
│   └── config.toml          # Enforces dark medical theme with high contrast
├── dataset/
│   ├── diabetes.csv         # Pima Indians Diabetes Dataset
│   └── train.py             # Preprocessing, model training & export script
├── .gitignore               # Excludes virtual environments, pycache, temporary files
├── app.py                   # Main Streamlit web application
├── metadata.joblib          # Exported feature names, metrics & cohort statistics
├── model.joblib             # Serialized trained scikit-learn Pipeline
├── README.md                # Project documentation and setup guide
└── requirements.txt         # Production Python dependencies
```

---

## 🚀 Getting Started Locally

### 1. Prerequisites
Ensure you have **Python 3.10+** installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/DiabetesRiskPrediction.git
cd DiabetesRiskPrediction
```

### 3. Set Up a Virtual Environment (Recommended)
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. (Optional) Retrain the Model
If you ever modify the dataset or wish to retrain the pipeline:
```bash
python dataset/train.py
```
This will train the pipeline and update `model.joblib` and `metadata.joblib`.

### 6. Launch the Web Application
```bash
streamlit run app.py
```
The application will launch automatically in your default browser at **`http://localhost:8501`**.

---

## 🩺 Biomarker Clinical References (ADA Guidelines)

| Biomarker | Standard Normal Range | Pre-Diabetes / Elevated | High Risk / Diabetic |
| :--- | :---: | :---: | :---: |
| **Fasting Glucose** | 70 – 99 mg/dL | 100 – 125 mg/dL | ≥ 126 mg/dL |
| **2-Hour OGTT Glucose** | < 140 mg/dL | 140 – 199 mg/dL | ≥ 200 mg/dL |
| **Body Mass Index (BMI)** | 18.5 – 24.9 kg/m² | 25.0 – 29.9 kg/m² | ≥ 30.0 kg/m² |
| **Diastolic Blood Pressure** | < 80 mm Hg | 80 – 89 mm Hg | ≥ 90 mm Hg |
| **Fasting Insulin** | 16 – 166 μU/mL | Elevated | Hyperinsulinemia |
| **Diabetes Pedigree (DPF)** | < 0.50 | 0.50 – 0.99 | ≥ 1.00 |

---

## ⚠️ Regulatory & Clinical Disclaimer

*GlycoGuard AI is intended solely for educational, research, and preliminary population screening demonstration purposes. It is **not** a certified medical diagnostic device and must never substitute for professional clinical judgment, direct physician consultation, or official laboratory diagnostic tests (such as Venous Plasma Glucose or HbA1c testing).*

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).

# Heart Health Risk Analysis: Supervised Modeling, Decision Calibration & Patient Phenotyping

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Library-Pandas-150458.svg)](https://pandas.pydata.org/)

An end-to-end clinical machine learning investigation using the historical **Framingham Heart Study** teaching cohort ($n = 4,240$). This project demonstrates physiological baseline prediction, calibrated binary coronary risk classification, clinical threshold optimization to address the **accuracy paradox**, unsupervised biomarker phenotyping, and surrogate decision-tree interpretability.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Clinical & Methodological Disclaimer](#clinical--methodological-disclaimer)
- [Dataset Characteristics](#dataset-characteristics)
- [Five-Stage Analytical Pipeline](#five-stage-analytical-pipeline)
  - [Stage 1: Linear Regression (Systolic Blood Pressure)](#stage-1-linear-regression-systolic-blood-pressure)
  - [Stage 2: Standardized Logistic Regression (10-Year CHD Risk)](#stage-2-standardized-logistic-regression-10-year-chd-risk)
  - [Stage 3: Decision Threshold Calibration & The Accuracy Paradox](#stage-3-decision-threshold-calibration--the-accuracy-paradox)
  - [Stage 4: Unsupervised Patient Phenotyping (K-Means & Hierarchical)](#stage-4-unsupervised-patient-phenotyping-k-means--hierarchical)
  - [Stage 5: Surrogate Rule-Based Tree Extraction](#stage-5-surrogate-rule-based-tree-extraction)
- [Key Quantitative Results](#key-quantitative-results)
- [Repository Structure](#repository-structure)
- [Installation & Getting Started](#installation--getting-started)
- [Reproducibility](#reproducibility)
- [License](#license)

---

## Project Overview

Cardiovascular diseases remain a leading cause of mortality worldwide. Standard default machine learning heuristics often fail in clinical environments due to class imbalance, skewed cost matrices (where a false negative is substantially more dangerous than a false positive), and "black-box" decision surfaces.

This study implements a multi-paradigm approach:
1. **Continuous Risk Modeling:** Quantifies independent hemodynamic associations with systolic blood pressure (`sysBP`).
2. **Binary Risk Multipliers:** Derives standardized odds ratios for 10-year coronary heart disease events (`TenYearCHD`).
3. **Threshold Calibration:** Solves the clinical accuracy paradox by tuning operating classification thresholds to optimize sensitivity.
4. **Unsupervised Phenotyping:** Identifies physiological patient archetypes using K-Means and Ward's hierarchical agglomeration.
5. **Surrogate Explainability:** Distills complex multidimensional clusters into human-auditable decision rules with $>83\%$ fidelity.

---

## Clinical & Methodological Disclaimer

> **IMPORTANT CLINICAL NOTICE**  
> This project is conducted strictly for educational, statistical, and machine learning research purposes using a historical observational teaching dataset.  
> - **Not a Medical Device:** The models, thresholds, and cluster classifications contained herein must **not** be used to diagnose, triage, or prescribe clinical treatment for patients.
> - **Correlation vs. Causation:** Regression coefficients and odds ratios describe observational statistical associations and do not prove causal biochemical pathways.
> - **Population Drift & External Validity:** The cohort is based on historical demographics from Framingham, Massachusetts. Direct application to external populations (e.g., modern diverse cohorts or healthcare systems in Sub-Saharan Africa) will cause significant calibration drift. Local validation and recalibration are mandatory before deploying any clinical decision support software.

---

## Dataset Characteristics

- **Cohort Size:** 4,240 patient records
- **Feature Space:** 16 demographic, behavioral, and clinical biomarkers
- **Target Outcomes:**
  - `sysBP` (Continuous, mmHg): Baseline range $83.5$ – $295.0\text{ mmHg}$ ($\mu = 132.35, \sigma = 22.03$)
  - `TenYearCHD` (Binary): $15.19\%$ overall positive event rate ($644$ positive cases, $3,596$ negative cases)

### Feature Architecture & Missingness

| Feature | Type | Description | Missing Count |
| :--- | :--- | :--- | :--- |
| `glucose` | Float | Fasting blood glucose level (mg/dL) | **388** |
| `education` | Float | Categorical education level (1–4) | **105** |
| `BPMeds` | Binary | Blood pressure medication usage (0/1) | **53** |
| `totChol` | Float | Total serum cholesterol level (mg/dL) | **50** |
| `cigsPerDay` | Float | Average daily cigarettes smoked | **29** |
| `BMI` | Float | Body Mass Index ($\text{kg/m}^2$) | **19** |
| `heartRate` | Float | Resting heart rate (bpm) | **1** |
| `male` | Binary | Biological sex (0 = Female, 1 = Male) | 0 |
| `age` | Integer | Patient age in years ($32$–$70$) | 0 |
| `currentSmoker` | Binary | Current smoking status (0/1) | 0 |
| `prevalentStroke`| Binary | History of stroke (0/1) | 0 |
| `prevalentHyp` | Binary | Prevalent hypertension diagnosis (0/1) | 0 |
| `diabetes` | Binary | Diagnosed diabetes (0/1) | 0 |
| `sysBP` | Float | Systolic blood pressure (mmHg) | 0 |
| `diaBP` | Float | Diastolic blood pressure (mmHg) | 0 |
| `TenYearCHD` | Binary | 10-year incident coronary heart disease event | 0 |

### Preprocessing & Data Hygiene
- **Data Leakage Safeguard:** All imputations (`SimpleImputer(strategy='median')`) and feature scalers (`StandardScaler()`) are fitted strictly within `sklearn.pipeline.Pipeline` on training partitions only.
- **Stratified Partitioning:** An $80/20$ train-test split (`random_state=42`) is executed with target stratification to preserve the $15.2\%$ minority class distribution across folds.

---

## Five-Stage Analytical Pipeline

### Stage 1: Linear Regression (Systolic Blood Pressure)
Predicts `sysBP` using upstream metabolic and demographic predictors: `age`, `BMI`, `cigsPerDay`, `totChol`, `glucose`, and `male`. Closely coupled downstream variables (`diaBP`, `BPMeds`, `prevalentHyp`) are deliberately excluded to assess risk factors available prior to formal hypertension management.

- **Model Performance:**
  - **Test $R^2$:** $0.230$ (vs. Mean Baseline $R^2 = -0.001$)
  - **Test RMSE:** $18.470\text{ mmHg}$ (vs. Mean Baseline $\text{RMSE} = 21.050\text{ mmHg}$)
- **Key Predictor Contributions:**
  - $\text{BMI}$: $+1.497\text{ mmHg}$ per unit increase
  - $\text{Age}$: $+0.871\text{ mmHg}$ per year
  - $\text{Glucose}$: $+0.068\text{ mmHg}$ per mg/dL
  - $\text{Male Sex}$: $-3.000\text{ mmHg}$ baseline adjustment (holding other factors constant)

### Stage 2: Standardized Logistic Regression (10-Year CHD Risk)
Models the log-odds of developing coronary heart disease within 10 years. Standardized coefficients $\beta_j$ are transformed into Odds Ratios ($\text{OR} = e^{\beta_j}$), reflecting risk multiplier per standard deviation shift.

$$\log\left(\frac{p}{1-p}\right) = \beta_0 + \sum_{j=1}^{p} \beta_j X_j$$

- **Top Risk Multipliers:**
  - **Age:** $\text{OR} = 1.781$ ($\beta = +0.577$) — Primary long-term driver
  - **Systolic BP:** $\text{OR} = 1.353$ ($\beta = +0.302$) — Hemodynamic vascular load
  - **Cigarettes Per Day:** $\text{OR} = 1.311$ ($\beta = +0.271$) — Dose-dependent endothelial damage
  - **Male Sex:** $\text{OR} = 1.245$ ($\beta = +0.219$) — Increased coronary prevalence
  - **Glucose:** $\text{OR} = 1.179$ ($\beta = +0.165$) — Metabolic vascular impairment

### Stage 3: Decision Threshold Calibration & The Accuracy Paradox
In imbalanced clinical tasks, standard accuracy is dangerously deceptive:

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{Total}}$$

At the default threshold ($\tau = 0.50$), the model achieves **84.4% accuracy**, yet an all-negative dummy classifier achieves **84.8%** by predicting zero events for everyone. At $\tau = 0.50$, the model misses $122$ out of $129$ actual cardiac cases (Sensitivity: $5.4\%$).

Calibrating the threshold down to the screening level ($\tau = 0.15$) aligns decision boundaries with population risk tolerance:

| Decision Threshold ($\tau$) | True Positives (TP) | False Negatives (FN) | False Positives (FP) | True Negatives (TN) | Sensitivity (Recall) | Specificity | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.50 (Default)** | 7 | 122 | 10 | 709 | 5.4% | 98.6% | **84.4%** |
| **0.30 (Balanced)** | 34 | 95 | 65 | 654 | 26.4% | 91.0% | **81.1%** |
| **0.15 (Screening)** | **80** | **49** | 231 | 488 | **62.0%** | 67.9% | **67.0%** |

*Clinical Trade-Off:* Lowering the cutoff to $0.15$ catches **73 additional cardiac patients**, avoiding unmanaged, fatal events at the manageable expense of diagnostic follow-up for 231 non-cardiac patients.

- **Test ROC AUC:** **0.702** (verifying robust discrimination across operating points above random 0.50 ranking).

### Stage 4: Unsupervised Patient Phenotyping (K-Means & Hierarchical)
Using 7 standardized continuous physiological features (`age`, `BMI`, `sysBP`, `diaBP`, `totChol`, `glucose`, `heartRate`), cohort segmentation was performed without outcome labels.

- **Clustering Diagnostics:**
  - Evaluated $k \in [2, 8]$ using Scree Elbow Plots and Silhouette Analysis.
  - Selected $k = 4$ ($\text{Silhouette} = 0.162$, vs. Ward Hierarchical $\text{Silhouette} = 0.096$).

| Cluster Archetype | Patients ($n$) | Mean Age | Mean BMI | Mean SysBP | Mean Glucose | Observed 10-Yr CHD Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cluster 1: Optimal / Young** | 1,700 | 42.6 | 24.2 | 118.6 mmHg | 78.1 mg/dL | **6.5%** |
| **Cluster 2: Aging / Dyslipidemic** | 1,488 | 55.0 | 25.6 | 129.1 mmHg | 80.6 mg/dL | **17.6%** |
| **Cluster 0: Severe Hypertensive** | 1,009 | 53.2 | 28.8 | 159.6 mmHg | 81.8 mg/dL | **24.7%** |
| **Cluster 3: Extreme Metabolic Outlier**| 43 | 54.2 | 28.2 | 151.4 mmHg | 253.1 mg/dL | **51.2%** |

### Stage 5: Surrogate Rule-Based Tree Extraction
To make the 7-dimensional clustering transparent for hospital triage, a depth-4 decision tree surrogate model was trained on the cluster labels:
- **Surrogate Fidelity:** **83.3% accuracy** in reconstructing cluster assignments using simple scalar rules.
- **Primary Diagnostic Cutoffs:**
  1. **Root Split (`Age`):** Separates the young normotensive cohort ($\text{Age} \le 48.5$ years).
  2. **Hemodynamic Split (`sysBP` & `diaBP`):** Segregates vascular hypertension ($\text{sysBP} > 141.5\text{ mmHg}$).
  3. **Metabolic Split (`glucose`):** Isolates extreme diabetic outliers ($\text{glucose} > 170.5\text{ mg/dL}$) into high-risk Cluster 3.

---

## Key Quantitative Results

```text
================================================================================
                    CLINICAL MACHINE LEARNING BENCHMARK SUMMARY
================================================================================
Dataset Cohort Size                : 4,240 records (16 features)
Minority Event Rate (TenYearCHD)   : 15.19%

Stage 1: Systolic BP Regression
  - Test R²                        : 0.230 (Baseline: -0.001)
  - Test RMSE                      : 18.47 mmHg (Baseline: 21.05 mmHg)
  - Primary Predictors             : BMI (+1.50 mmHg), Age (+0.87 mmHg)

Stage 2: Standardized CHD Logistic Regression
  - Model Test ROC AUC             : 0.702
  - Top Odds Ratios (per std dev)  : Age (1.78), sysBP (1.35), cigsPerDay (1.31)

Stage 3: Decision Threshold Calibration
  - Threshold 0.50 (Default)       : Sensitivity 5.4%,  Accuracy 84.4% (Caught: 7/129)
  - Threshold 0.15 (Screening)      : Sensitivity 62.0%, Accuracy 67.0% (Caught: 80/129)

Stage 4 & 5: Unsupervised Phenotypes & Surrogates
  - Selected K-Means Archetypes    : 4 distinct clusters (CHD risk: 6.5% to 51.2%)
  - Decision Tree Fidelity         : 83.3% reconstruction accuracy
================================================================================
```

---

## Repository Structure

```bash
├── README.md                                    # Comprehensive project documentation
├── Heart_Health_Risk_Analysis.ipynb             # Jupyter Notebook with full analysis & code
├── presentation/
│   ├── heart_health_risk_presentation.html      # 9-Slide deck HTML artifact
│   └── presentation_script.md                   # Slide-by-slide executive presentation script
└── requirements.txt                             # Python dependencies
```

---

## Installation & Getting Started

### Prerequisites
- Python 3.9, 3.10, or 3.11
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/heart-health-risk-analysis.git
cd heart-health-risk-analysis
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

*(If `requirements.txt` is not yet created, run:)*
```bash
pip install numpy pandas matplotlib scikit-learn jupyterlab
```

### 4. Launch the Notebook
```bash
jupyter lab Heart_Health_Risk_Analysis.ipynb
```

The dataset is ingested automatically from the public raw repository via `pandas.read_csv`, requiring no manual downloads.

---

## Reproducibility

Every experimental step incorporates deterministic pseudorandom seed pinning (`RANDOM_STATE = 42`):
- Deterministic median imputation order inside Scikit-Learn pipelines
- Consistent stratified train-test splits ($80/20$)
- Reproducible K-Means centroid initializations (`n_init=10`, `random_state=42`)
- Deterministic decision-tree surrogate partitioning

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
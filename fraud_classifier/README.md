<div align="center">

# Enterprise Fraud Risk Intelligence System

### End-to-end fraud detection pipeline structured to SR 11-7 model risk management standards

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat)](https://xgboost.ai)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-IEEE--CIS%20Fraud-purple?style=flat&logo=kaggle)](https://www.kaggle.com/competitions/ieee-fraud-detection)

</div>

---

## The Problem

A financial institution processes hundreds of thousands of transactions daily. 3.5% are fraudulent. A simple rule flags everything over $500 — it misses 60% of fraud and blocks legitimate customers. A black-box model scores better but can't explain its decisions to a regulator, a compliance officer, or the customer it just declined.

The solution requires three things working together: **detection** that finds the right transactions, **explainability** that traces why each one was flagged, and **governance** that proves the system is working — and alerts when it isn't.

This project builds all three.

---

## Pipeline

```
Raw Data                    Detection               Explainability      Governance
────────────────────────────────────────────────────────────────────────────────

590,540 transactions   →   Champion vs         →   SHAP attribution  →   PSI drift
+ identity signals         Challenger model        per transaction       monitoring

IEEE-CIS dataset           Decision Tree           Top risk drivers      Population
(Vesta Corporation)        vs XGBoost              in plain English      stability
                                                                          tracking
                                                        ↓
                                               LLM Triage Agent
                                               SAR-style narrative
                                               for compliance analyst
```

---

## Results

> Validated on an **out-of-time holdout** (last 20% by TransactionDT).
> Random splits are not used — they cause data leakage on time-series transaction data.

| Metric | Champion: Decision Tree | Challenger: XGBoost | SR 11-7 Threshold |
|--------|------------------------|---------------------|-------------------|
| **KS Statistic** | 0.5067 | **0.6670** | > 0.40 ✓ |
| **Gini Coefficient** | 0.6461 | **0.8217** | > 0.60 ✓ |
| AUC-ROC | 0.8231 | **0.9108** | — |
| AUC-PR | 0.3601 | **0.4970** | — |

> **Why KS and Gini — not accuracy?**
> With a 3.5% fraud rate, a model predicting "legitimate" every time achieves 96.5% accuracy.
> KS and Gini measure *discrimination* — the separation between fraud and legitimate score distributions.
> These are the metrics that appear in bank model validation reports.

### Population Stability

| Comparison | PSI | Status |
|------------|-----|--------|
| Train → Validation | 0.00105 | ✅ Stable |
| Train → Test | 0.00046 | ✅ Stable |

PSI < 0.10 = Stable · PSI 0.10–0.25 = Monitor · PSI > 0.25 = Model review required

---

## Dataset

**Source:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) — Vesta Corporation

| Stat | Value |
|------|-------|
| Total transactions | 590,540 |
| Fraud cases | 20,663 |
| Fraud rate | 3.50% |
| Features (after engineering) | 63 |
| Transaction table | 55 columns |
| Identity table | 41 columns |

---

## Five Components

### 1 · Data Pipeline & EDA
590,540 transactions merged with 144,233 identity records. Null governance: columns with >90% missing values dropped. Class imbalance documented: 20,663 fraud cases (3.5%) against 569,877 legitimate transactions.

### 2 · Feature Engineering
10 behavioural risk features engineered from domain logic — not statistical selection alone:

| Feature | What it captures | Fraud signal |
|---------|-----------------|--------------|
| `amt_zscore` | How unusual is this amount for *this specific card's* history | High z-score = risk |
| `card1_txn_count` | Transaction velocity — how many charges in this window | High count = risk |
| `email_mismatch` | Purchaser and recipient email domains differ | 1 = risk |
| `P_email_risky` | Purchaser using free / anonymous email provider | 1 = risk |
| `is_night` | Transaction between 00:00–06:00 UTC | 1 = elevated risk |
| `addr_mismatch` | Billing and shipping address inconsistency | 1 = risk |
| `is_large_txn` | Transaction exceeds $500 USD | 1 = risk |
| `amt_log` | Log-transform correcting right-skewed amount distribution | Normalisation |
| `C_sum` | Aggregate of historical address and device count signals | Low = risk |
| `is_mobile` | Device type context | Context feature |

**Top features by SHAP importance:** `C5` · `C14` · `C1` · `M4` · `TransactionAmt`

### 3 · Champion-Challenger Modelling

**Champion — Decision Tree**
`max_depth=8` · `min_samples_leaf=50` · `class_weight='balanced'`
Fully interpretable. Every decision path traceable. Regulatory baseline.

**Challenger — XGBoost**
300 estimators · `learning_rate=0.05` · `scale_pos_weight=27.6` (imbalance ratio)
Early stopping on AUC-PR — the correct metric for imbalanced fraud data.
`tree_method='hist'` for memory efficiency at 590k rows.

The champion-challenger framework is how banks compare models before production deployment. The challenger doesn't replace the champion until it demonstrably outperforms on validated metrics.

### 4 · SHAP Explainability

Every prediction has a traceable reason. SHAP (SHapley Additive exPlanations) decomposes the model's output into individual feature contributions — showing not just *what* was flagged but *why*.

This satisfies:
- SR 11-7 conceptual soundness requirement
- GDPR Article 22 right to explanation
- EU AI Act transparency requirements
- Internal audit and compliance review requirements

### 5 · PSI Drift Monitoring + LLM Triage Agent

**PSI Monitor**
Compares score distributions between training population and live population across 10 score bins. Detects model degradation before it becomes a business problem. Standard MRM requirement for models in production.

**LLM Triage Agent**
Flagged transactions are passed to Claude with their SHAP attribution. The agent generates a SAR-style investigation narrative for the compliance analyst:

```
Transaction flagged at 94.3% fraud probability — CRITICAL risk.

Primary drivers: elevated C5 count (14x above card average) and
TransactionAmt of $847 which is 3.2 standard deviations above this
card's historical mean. Email domain mismatch detected between
purchaser and recipient — a secondary velocity signal further
elevates the risk profile.

RECOMMENDATION: Immediate block. Escalate to compliance officer.
Case sent for further investigation — HIGH PRIORITY.
```

What previously required 15 minutes of manual analyst write-up per flagged case is generated automatically, with full SHAP-backed reasoning.

---

## Regulatory Alignment

| SR 11-7 Requirement | Implementation |
|--------------------|----------------|
| Model development documentation | Feature rationale, assumption documentation, data lineage in each notebook |
| Conceptual soundness | SHAP explainability, decision tree interpretability |
| Out-of-time validation | TransactionDT-ordered split — no random sampling |
| Ongoing monitoring | PSI drift monitoring across 10 score bins |
| Champion-challenger | Structured comparison: Decision Tree baseline vs XGBoost challenger |
| Sensitivity analysis | Class imbalance handling, threshold analysis |

---

## Key BFSI Concepts Demonstrated

| Concept | Where used |
|---------|-----------|
| KS Statistic | Model validation (Notebook 2) |
| Gini Coefficient | Model validation (Notebook 2) |
| PSI Drift Monitoring | Production monitoring (Notebook 3) |
| Champion-Challenger | Model comparison (Notebook 2) |
| SHAP / XAI | Explainability (Notebooks 2 & 3) |
| SAR Narrative | LLM triage layer (Notebook 3) |
| Class Imbalance Handling | `scale_pos_weight`, `class_weight='balanced'` |
| Time-Aware Validation | Out-of-time split on TransactionDT |

---

## Repository Structure

```
├── notebooks/
│   ├── 01_eda_feature_engineering.py      # Data pipeline, null governance, feature engineering
│   ├── 02_model_training_validation.py    # Champion-challenger, KS/Gini/SHAP, validation plots
│   └── 03_risk_pipeline_llm_triage.py     # PSI monitor + LLM SAR triage agent
│
├── outputs/
│   ├── metrics.json                       # KS, Gini, AUC-ROC, AUC-PR
│   ├── psi_results.json                   # Drift monitoring results
│   ├── shap_importance.csv                # Feature-level SHAP values
│   ├── triage_demo.csv                    # LLM triage narratives (5 highest-risk cases)
│   └── submission.csv                     # Kaggle leaderboard submission
│
├── models/
│   ├── champion_decision_tree.pkl
│   └── challenger_xgboost.pkl
│
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
# 1. Clone and install
git clone https://github.com/hiralsarkar/Fraud-Risk-Classifier
cd Fraud-Risk-Classifier
pip install -r requirements.txt

# 2. Download data from Kaggle
# kaggle competitions download -c ieee-fraud-detection
# Place CSVs in ./data/ieee-fraud-detection/

# 3. Run notebooks in order
python notebooks/01_eda_feature_engineering.py
python notebooks/02_model_training_validation.py

# 4. LLM triage layer — requires Anthropic API key
# Get yours at console.anthropic.com
# Windows:   setx ANTHROPIC_API_KEY "sk-ant-..."
# Mac/Linux: export ANTHROPIC_API_KEY="sk-ant-..."
python notebooks/03_risk_pipeline_llm_triage.py
```

---

## Stack

`Python 3.11` · `XGBoost` · `scikit-learn` · `SHAP` · `pandas` · `numpy` · `matplotlib` · `Anthropic API` · `joblib` · `pyarrow`

---

## Skills Demonstrated

`Python` `XGBoost` `scikit-learn` `SHAP` `pandas` `Anthropic API` `Model Risk Management` `SR 11-7` `AML` `Fraud Detection` `Feature Engineering` `Model Validation` `PSI Monitoring` `Champion-Challenger` `GDPR XAI`

---

<div align="center">

*Built June 2026 · Hiral Sarkar*
*[LinkedIn](https://linkedin.com/in/hiralsarkar) · [GitHub](https://github.com/hiralsarkar)*

</div>

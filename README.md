# BFSI Fraud Risk Classifier

**A production-grade fraud detection pipeline built on the IEEE-CIS dataset, structured to SR 11-7 model risk management standards.**

> Built as a portfolio project demonstrating end-to-end AI risk model development for BFSI (Banking, Financial Services & Insurance).

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   FRAUD RISK PIPELINE                       │
├──────────┬──────────┬──────────┬──────────┬────────────────┤
│ Step 1   │ Step 2   │ Step 3   │ Step 4   │ Step 5         │
│ Ingest & │ Feature  │ Classify │ Alert &  │ Validate &     │
│ Inventory│ Engineer │ (Model)  │ Triage   │ Monitor        │
├──────────┼──────────┼──────────┼──────────┼────────────────┤
│ 590,540  │ 63 risk  │ Decision │ LLM SAR  │ PSI drift      │
│ txns     │ features │ Tree +   │ narrative│ monitor        │
│ merged   │ engineered│ XGBoost │ via API  │ KS / Gini      │
└──────────┴──────────┴──────────┴──────────┴────────────────┘
```

---

## Validation Results

Validated on a time-aware holdout (last 20% of data by TransactionDT — never random split on fraud data).

| Metric           | Champion: Decision Tree | Challenger: XGBoost | SR 11-7 Benchmark |
|------------------|------------------------|---------------------|-------------------|
| **KS Statistic** | 0.5067                 | **0.6670**          | > 0.40 ✓          |
| **Gini**         | 0.6461                 | **0.8217**          | > 0.60 ✓          |
| AUC-ROC          | 0.8231                 | **0.9108**          | —                 |
| AUC-PR           | 0.3601                 | **0.4970**          | —                 |

> **Why KS and Gini?** These are the standard metrics in bank model validation reports (SR 11-7 / SS1-23). Accuracy is not used — the 3.5% fraud rate makes it meaningless.

### PSI Drift Monitoring

| Comparison         | PSI Value | Assessment |
|--------------------|-----------|------------|
| Train → Validation | 0.00105   | ✅ Stable  |
| Train → Test data  | 0.00046   | ✅ Stable  |

PSI thresholds: `< 0.10` Stable | `0.10–0.25` Minor drift | `> 0.25` Model review required

---

## Dataset

**Source:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data) — Vesta Corporation

| Stat | Value |
|------|-------|
| Total transactions | 590,540 |
| Fraud cases | 20,663 |
| Fraud rate | 3.50% |
| Features (after engineering) | 63 |
| Transaction table | 55 columns |
| Identity table | 41 columns |

---

## Feature Engineering

10 domain-relevant features engineered:

| Feature | Description | Fraud signal |
|---------|-------------|--------------|
| `amt_log` | Log-transform of transaction amount | Right-skew correction |
| `amt_zscore` | Amount z-score per card (unusual spend) | High zscore = risk |
| `email_mismatch` | Purchaser vs recipient email domain differs | 1 = risk |
| `P_email_risky` | Purchaser uses free/anonymous email provider | 1 = risk |
| `is_night` | Transaction between 00:00–06:00 UTC | 1 = elevated risk |
| `card1_txn_count` | Transaction velocity for this card | High = risk |
| `addr_mismatch` | Billing vs shipping address mismatch | 1 = risk |
| `is_large_txn` | Transaction > $500 USD | 1 = risk |
| `C_sum` | Sum of C-columns (previous address/device counts) | Low = risk |
| `is_mobile` | Device type is mobile | Context feature |

**Top 5 features by SHAP importance:** `C5`, `C14`, `C1`, `M4`, `TransactionAmt`

---

## Models

### Champion: Decision Tree (interpretable baseline)
- `max_depth=8`, `min_samples_leaf=50`, `class_weight='balanced'`
- Fully interpretable — every decision path is explainable to a regulator
- Baseline against which XGBoost challenger is compared

### Challenger: XGBoost
- 300 estimators, `learning_rate=0.05`, `scale_pos_weight=27.6` (handles class imbalance)
- `tree_method='hist'` for memory efficiency on 590k rows
- Early stopping on AUC-PR (the right metric for imbalanced fraud data)

---

## LLM Triage Agent (Notebook 3)

Flagged transactions are routed to an LLM-powered triage agent that:
1. Computes SHAP values for the specific transaction
2. Identifies the top risk-increasing and risk-reducing factors
3. Generates a SAR-style narrative for compliance analysts

**Example output:**
```
Transaction flagged at 94.3% fraud probability — CRITICAL risk.
Primary drivers: elevated C5 count (14x above card average) and
TransactionAmt of $847 which is 3.2 standard deviations above this
card's historical mean. Email domain mismatch detected.
RECOMMENDATION: Immediate block. Notify compliance officer.
```

This replaces manual write-up time for each flagged case (~15 min/case → automated).

---

## Repository Structure

```
Fraud-Risk-Classifier/
├── fraud_classifier/
│   ├── notebooks/
│   │   ├── 01_eda_feature_engineering.py      # EDA, null handling, feature engineering
│   │   ├── 02_model_training_validation.py    # Champion-challenger, KS/Gini, SHAP
│   │   └── 03_risk_pipeline_llm_triage.py     # PSI monitor + LLM triage agent
│   ├── outputs/
│   │   ├── metrics.json                       # Validation metrics
│   │   ├── psi_results.json                   # Drift monitoring results
│   │   └── *.png                              # Charts (EDA, KS, SHAP, PSI)
│   ├── models/                                # Trained model binaries (not committed)
│   ├── data/                                  # Raw IEEE-CIS CSVs (not committed)
│   └── requirements.txt
└── README.md
```

---

## How to Run

```bash
# 1. Clone and install
git clone https://github.com/hiralsarkar/Fraud-Risk-Classifier.git
cd Fraud-Risk-Classifier/fraud_classifier
pip install -r requirements.txt

# 2. Download data from Kaggle
# kaggle competitions download -c ieee-fraud-detection
# Place CSVs in ./data/ieee-fraud-detection/

# 3. Run notebooks in order
python notebooks/01_eda_feature_engineering.py
python notebooks/02_model_training_validation.py

# 4. For LLM triage layer, set your OpenRouter API key
# Get a free key at: openrouter.ai/keys
export OPENROUTER_API_KEY="your-key-here"
python notebooks/03_risk_pipeline_llm_triage.py
```

---

## Regulatory Alignment

This project follows SR 11-7 (Federal Reserve Model Risk Management Guidance) principles:

- **Model development**: documented assumptions, training/validation split rationale
- **Model validation**: KS, Gini, AUC-PR on out-of-time holdout
- **Ongoing monitoring**: PSI drift detection between training and live populations
- **Explainability**: SHAP values for every prediction (XAI requirement under GDPR Art. 22 / EU AI Act)
- **Champion-challenger**: structured comparison before deployment

---

## Key BFSI Concepts Demonstrated

| Concept | Where used |
|---------|-----------|
| KS Statistic | Model validation (Notebook 2) |
| Gini coefficient | Model validation (Notebook 2) |
| PSI drift monitoring | Production monitoring (Notebook 3) |
| Champion-challenger | Model comparison (Notebook 2) |
| SHAP / XAI | Explainability (Notebooks 2 & 3) |
| SAR narrative | LLM triage layer (Notebook 3) |
| Class imbalance handling | `scale_pos_weight`, `class_weight='balanced'` |
| Time-aware validation | Out-of-time split on TransactionDT |

---

## Skills Demonstrated

`Python` `XGBoost` `scikit-learn` `SHAP` `pandas` `LLM API Integration` `Model Risk Management` `SR 11-7` `AML` `Fraud Detection` `Feature Engineering` `Model Validation` `PSI Monitoring`

---

*Built May 2026 — IEEE-CIS Fraud Detection (Kaggle)*

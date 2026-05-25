"""


╔══════════════════════════════════════════════════════════════╗
║   BFSI FRAUD RISK CLASSIFIER — NOTEBOOK 2                   ║
║   Model Training, Validation & Explainability               ║
╚══════════════════════════════════════════════════════════════╝

Run AFTER 01_eda_feature_engineering.py
Outputs: trained models, validation metrics, SHAP plots
"""

# ── CELL 1: Imports ───────────────────────────────────────────
import os
os.environ.setdefault('NUMBA_DISABLE_CUDA', '1')  # shap/numba: no CUDA on this machine

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json, joblib, warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, roc_curve,
                             classification_report, confusion_matrix)
from sklearn.preprocessing import label_binarize
import xgboost as xgb
import shap

_HERE     = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(_HERE, '..', 'outputs') + os.sep
MODEL_DIR = os.path.join(_HERE, '..', 'models') + os.sep
os.makedirs(MODEL_DIR, exist_ok=True)

print("✓ Imports OK")


# ── CELL 2: Load prepared data ────────────────────────────────
X_train = pd.read_parquet(OUT_DIR + 'X_train.parquet')
y_train = pd.read_parquet(OUT_DIR + 'y_train.parquet')['isFraud']
X_test  = pd.read_parquet(OUT_DIR + 'X_test.parquet')

with open(OUT_DIR + 'features.json') as f:
    FEATURES = json.load(f)

print(f"X_train: {X_train.shape}, fraud rate: {y_train.mean()*100:.2f}%")


# ── CELL 3: Train/validation split (time-aware) ───────────────
# NOTE: IEEE-CIS is time-ordered. Use last 20% as validation — 
# never use random split on time-series fraud data (leakage risk)
split_idx = int(len(X_train) * 0.8)
X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

print(f"\nTime-aware split:")
print(f"  Train : {len(X_tr):,} rows  | fraud rate: {y_tr.mean()*100:.2f}%")
print(f"  Valid : {len(X_val):,} rows  | fraud rate: {y_val.mean()*100:.2f}%")


# ── CELL 4: Helper — compute BFSI validation metrics ─────────
def bfsi_metrics(y_true, y_prob, label='Model'):
    """
    Compute the metrics that appear in bank model validation reports:
    - KS Statistic  : max separation between fraud/legit score distributions
    - Gini          : 2*AUC - 1 (standard credit risk measure)
    - AUC-ROC       : area under ROC curve
    - AUC-PR        : area under Precision-Recall curve (better for imbalanced)
    """
    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr  = average_precision_score(y_true, y_prob)
    gini    = 2 * auc_roc - 1

    # KS statistic
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks = round(float(np.max(tpr - fpr)), 4)

    print(f"\n{'═'*45}")
    print(f"  {label}")
    print(f"{'═'*45}")
    print(f"  KS Statistic  : {ks:.4f}  {'✓ Good (>0.4)' if ks>0.4 else '⚠ Weak (<0.4)'}")
    print(f"  Gini          : {gini:.4f}  {'✓ Good (>0.6)' if gini>0.6 else '⚠ Weak (<0.6)'}")
    print(f"  AUC-ROC       : {auc_roc:.4f}")
    print(f"  AUC-PR        : {auc_pr:.4f}  (key metric for imbalanced data)")
    print(f"{'═'*45}")

    return {'ks': ks, 'gini': gini, 'auc_roc': auc_roc, 'auc_pr': auc_pr}


# ── CELL 5: CHAMPION — Decision Tree (interpretable baseline) ─
print("\nTraining Champion: Decision Tree...")

dt = DecisionTreeClassifier(
    max_depth=8,
    min_samples_leaf=50,
    class_weight='balanced',   # handles class imbalance
    random_state=42
)
dt.fit(X_tr, y_tr)
dt_prob = dt.predict_proba(X_val)[:, 1]
dt_metrics = bfsi_metrics(y_val, dt_prob, 'CHAMPION: Decision Tree')

joblib.dump(dt, MODEL_DIR + 'champion_decision_tree.pkl')
print("✓ Decision Tree saved")


# ── CELL 6: CHALLENGER — XGBoost ─────────────────────────────
print("\nTraining Challenger: XGBoost...")

# scale_pos_weight = ratio of negatives to positives (handles imbalance)
neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
scale_pw  = neg / pos

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pw,
    eval_metric='aucpr',
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

xgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False
)

xgb_prob = xgb_model.predict_proba(X_val)[:, 1]
xgb_metrics = bfsi_metrics(y_val, xgb_prob, 'CHALLENGER: XGBoost')

joblib.dump(xgb_model, MODEL_DIR + 'challenger_xgboost.pkl')
print("✓ XGBoost saved")


# ── CELL 7: Champion-Challenger comparison plot ───────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Champion vs Challenger — Model Validation', fontsize=13)

# ROC curves
for prob, label, color in [
    (dt_prob,  'Champion (Decision Tree)', '#534AB7'),
    (xgb_prob, 'Challenger (XGBoost)',     '#1D9E75')
]:
    fpr, tpr, _ = roc_curve(y_val, prob)
    auc = roc_auc_score(y_val, prob)
    axes[0].plot(fpr, tpr, label=f'{label} (AUC={auc:.3f})', linewidth=2, color=color)

axes[0].plot([0,1],[0,1],'--', color='gray', linewidth=1, label='Random')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend(fontsize=9)

# Precision-Recall curves
for prob, label, color in [
    (dt_prob,  'Champion (Decision Tree)', '#534AB7'),
    (xgb_prob, 'Challenger (XGBoost)',     '#1D9E75')
]:
    prec, rec, _ = precision_recall_curve(y_val, prob)
    ap = average_precision_score(y_val, prob)
    axes[1].plot(rec, prec, label=f'{label} (AP={ap:.3f})', linewidth=2, color=color)

baseline = y_val.mean()
axes[1].axhline(y=baseline, color='gray', linestyle='--', linewidth=1,
                label=f'Random ({baseline:.3f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve (key for imbalanced data)')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR + '02_champion_challenger.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n✓ Validation plot saved → {OUT_DIR}02_champion_challenger.png")


# ── CELL 8: KS score distribution plot ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle('KS Score Distributions (fraud vs legitimate)', fontsize=12)

for ax, prob, label in [
    (axes[0], dt_prob,  'Champion: Decision Tree'),
    (axes[1], xgb_prob, 'Challenger: XGBoost')
]:
    legit_scores = prob[y_val == 0]
    fraud_scores = prob[y_val == 1]
    ax.hist(legit_scores, bins=50, alpha=0.6, color='#1D9E75', label='Legitimate', density=True)
    ax.hist(fraud_scores, bins=50, alpha=0.6, color='#D85A30', label='Fraud', density=True)
    fpr, tpr, _ = roc_curve(y_val, prob)
    ks = float(np.max(tpr - fpr))
    ax.set_title(f'{label}\nKS = {ks:.4f}', fontsize=10)
    ax.set_xlabel('Fraud probability score')
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR + '03_ks_distributions.png', dpi=150, bbox_inches='tight')
plt.show()


# ── CELL 9: SHAP explainability (XGBoost) ────────────────────
print("\nComputing SHAP values (using 2000 validation samples)...")

# Use a sample for speed — SHAP is O(n)
sample_idx = np.random.choice(len(X_val), size=min(2000, len(X_val)), replace=False)
X_shap = X_val.iloc[sample_idx]

explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_shap)

# SHAP summary plot (most important to least)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_shap, plot_type='bar',
                  max_display=20, show=False)
plt.title('SHAP Feature Importance — XGBoost Challenger', fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR + '04_shap_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ SHAP importance plot saved → {OUT_DIR}04_shap_importance.png")

# SHAP beeswarm (direction of effect)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_shap, max_display=15, show=False)
plt.title('SHAP Beeswarm — Feature Impact Direction', fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR + '05_shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ SHAP beeswarm saved → {OUT_DIR}05_shap_beeswarm.png")


# ── CELL 10: Save metrics summary ────────────────────────────
metrics_summary = {
    'champion_decision_tree': dt_metrics,
    'challenger_xgboost': xgb_metrics
}

with open(OUT_DIR + 'metrics.json', 'w') as f:
    json.dump(metrics_summary, f, indent=2)

print(f"\n✓ Metrics saved → {OUT_DIR}metrics.json")


# ── CELL 11: Top features by SHAP ────────────────────────────
shap_df = pd.DataFrame({
    'feature': X_shap.columns,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

print("\n TOP 15 FEATURES BY SHAP IMPORTANCE")
print("─" * 45)
for _, row in shap_df.head(15).iterrows():
    bar = '█' * int(row['mean_abs_shap'] * 200)
    print(f"  {row['feature']:<28} {row['mean_abs_shap']:.4f}  {bar}")

# Save SHAP importance
shap_df.to_csv(OUT_DIR + 'shap_importance.csv', index=False)

# Save test predictions for submission
test_prob = xgb_model.predict_proba(X_test)[:, 1]
import pandas as pd
sub = pd.read_csv(os.path.join(_HERE, '..', 'data', 'ieee-fraud-detection', 'sample_submission.csv'))
sub['isFraud'] = test_prob
sub.to_csv(OUT_DIR + 'submission.csv', index=False)

print(f"\n✓ Kaggle submission file saved → {OUT_DIR}submission.csv")
print("✓ Notebook 2 complete — run 03_risk_pipeline.py next")

# %% [markdown]
# # Model Training, Calibration & Validation
# **Champion-challenger | KS / Gini / AUC-PR | calibration | operating threshold | SHAP**
#
# Three time-ordered periods (defined in notebook 1):
#
# | Period | Used for |
# |---|---|
# | **fit** (first 70%) | fitting both models |
# | **valid** (next 15%) | XGBoost early stopping, probability calibration, threshold choice |
# | **oot** (last 15%) | the reported metrics only — touched once, at the end |
#
# Run after `01_eda_feature_engineering.py`.

# %%
import os, json, warnings
os.environ.setdefault('NUMBA_DISABLE_CUDA', '1')
import numpy as np
import pandas as pd
import matplotlib
if os.environ.get('MPLBACKEND') is None and not os.environ.get('DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, brier_score_loss)
import xgboost as xgb
import shap
warnings.filterwarnings('ignore')

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
OUT_DIR   = os.path.join(_HERE, '..', 'outputs') + os.sep
MODEL_DIR = os.path.join(_HERE, '..', 'models') + os.sep
DATA_DIR  = os.environ.get('FRAUD_DATA_DIR', os.path.join(_HERE, '..', 'data', 'ieee-fraud-detection')) + os.sep
os.makedirs(MODEL_DIR, exist_ok=True)

# Operating-point assumptions. Change these to match the business, then re-run.
ALERT_CAPACITY_PER_DAY = 300     # alerts the review team can work per day
COST_MISSED_FRAUD      = None    # None = use the fraud transaction's own amount as the loss
COST_REVIEW            = 5.0     # cost of reviewing one alert (USD)
SEED = 42

# %% [markdown]
# ## Load data and periods

# %%
X_all = pd.read_parquet(OUT_DIR + 'X_train.parquet')
y_all = pd.read_parquet(OUT_DIR + 'y_train.parquet')['isFraud']
meta  = pd.read_parquet(OUT_DIR + 'train_meta.parquet')
X_test = pd.read_parquet(OUT_DIR + 'X_test.parquet')

idx = {p: (meta['period'] == p).values for p in ('fit', 'valid', 'oot')}
X_fit, y_fit = X_all[idx['fit']], y_all[idx['fit']]
X_val, y_val = X_all[idx['valid']], y_all[idx['valid']]
X_oot, y_oot = X_all[idx['oot']], y_all[idx['oot']]
amt_val, amt_oot = X_val['TransactionAmt'].values, X_oot['TransactionAmt'].values
days = {p: max((meta.loc[idx[p], 'TransactionDT'].max() - meta.loc[idx[p], 'TransactionDT'].min()) / 86400, 1.0)
        for p in idx}
for p, (X, y) in {'fit': (X_fit, y_fit), 'valid': (X_val, y_val), 'oot': (X_oot, y_oot)}.items():
    print(f"{p:<5} {len(X):>8,} rows | {days[p]:5.1f} days | fraud rate {y.mean()*100:.2f}%")

# %% [markdown]
# ## Metrics
# KS and Gini are the usual discrimination measures in bank model validation; AUC-PR is the most
# informative single number at a 3.5% base rate. Brier score measures calibration.

# %%
def risk_metrics(y_true, p):
    fpr, tpr, _ = roc_curve(y_true, p)
    auc = roc_auc_score(y_true, p)
    return {'ks': round(float(np.max(tpr - fpr)), 4), 'gini': round(float(2 * auc - 1), 4),
            'auc_roc': round(float(auc), 4), 'auc_pr': round(float(average_precision_score(y_true, p)), 4),
            'brier': round(float(brier_score_loss(y_true, p)), 5)}

def show(name, m):
    print(f"{name:<34} KS {m['ks']:.4f} | Gini {m['gini']:.4f} | AUC {m['auc_roc']:.4f} | AUC-PR {m['auc_pr']:.4f} | Brier {m['brier']:.5f}")

# %% [markdown]
# ## Champion: decision tree
# Depth-8 tree with balanced class weights: every path can be printed for a reviewer.

# %%
dt = DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, class_weight='balanced', random_state=SEED)
dt.fit(X_fit, y_fit)
dt_val, dt_oot = dt.predict_proba(X_val)[:, 1], dt.predict_proba(X_oot)[:, 1]

# %% [markdown]
# ## Challenger: XGBoost
# Early stopping watches the **valid** period. The OOT period is not seen here.

# %%
scale_pw = float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))
xgb_model = xgb.XGBClassifier(
    n_estimators=1000, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pw, eval_metric='aucpr', early_stopping_rounds=50,
    tree_method='hist', random_state=SEED, n_jobs=-1, verbosity=0)
xgb_model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
xgb_val, xgb_oot = xgb_model.predict_proba(X_val)[:, 1], xgb_model.predict_proba(X_oot)[:, 1]
print(f"XGBoost best iteration: {xgb_model.best_iteration}")

# %% [markdown]
# ## Calibration
# Class weighting inflates raw scores, so a raw 0.9 is not a 90% fraud probability. An isotonic map
# fitted on the **valid** period turns scores into probabilities. It is monotonic, so KS/Gini/AUC
# are unchanged; only the Brier score and the meaning of the number change.

# %%
calibrators = {}
for name, s_val in (('decision_tree', dt_val), ('xgboost', xgb_val)):
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(s_val, y_val)
    calibrators[name] = iso
p_dt_val,  p_dt_oot  = calibrators['decision_tree'].predict(dt_val),  calibrators['decision_tree'].predict(dt_oot)
p_xgb_val, p_xgb_oot = calibrators['xgboost'].predict(xgb_val), calibrators['xgboost'].predict(xgb_oot)

results = {
    'champion_decision_tree': {'oot': risk_metrics(y_oot, p_dt_oot), 'oot_raw_score_brier': risk_metrics(y_oot, dt_oot)['brier']},
    'challenger_xgboost':     {'oot': risk_metrics(y_oot, p_xgb_oot), 'oot_raw_score_brier': risk_metrics(y_oot, xgb_oot)['brier'],
                               'best_iteration': int(xgb_model.best_iteration)},
}
print("\nOUT-OF-TIME RESULTS (calibrated probabilities)")
show('Champion: decision tree', results['champion_decision_tree']['oot'])
show('Challenger: XGBoost', results['challenger_xgboost']['oot'])
print(f"Brier raw -> calibrated (XGBoost): {results['challenger_xgboost']['oot_raw_score_brier']:.5f} -> "
      f"{results['challenger_xgboost']['oot']['brier']:.5f}")

# %% [markdown]
# ## Operating threshold
# Chosen on the **valid** period: the threshold with the lowest expected cost
# (missed fraud amount + review cost per alert) among those that keep the alert volume within
# `ALERT_CAPACITY_PER_DAY`. Then reported, unchanged, on OOT.

# %%
def at_threshold(y, p, amt, t, n_days):
    flag = p >= t
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    miss_loss = float(amt[(~flag) & (y == 1)].sum()) if COST_MISSED_FRAUD is None else fn * COST_MISSED_FRAUD
    return {'threshold': round(float(t), 4), 'alerts_per_day': round(float(flag.sum() / n_days), 1),
            'precision': round(tp / max(tp + fp, 1), 4), 'recall': round(tp / max(tp + fn, 1), 4),
            'fraud_amount_caught_pct': round(float(amt[flag & (y == 1)].sum() / max(amt[y == 1].sum(), 1e-9)) * 100, 2),
            'expected_cost': round(miss_loss + COST_REVIEW * int(flag.sum()), 2)}

y_val_a, y_oot_a = y_val.values, y_oot.values
candidates = np.unique(np.quantile(p_xgb_val, np.linspace(0.80, 0.9995, 400)))
grid = [at_threshold(y_val_a, p_xgb_val, amt_val, t, days['valid']) for t in candidates]
feasible = [g for g in grid if g['alerts_per_day'] <= ALERT_CAPACITY_PER_DAY] or [max(grid, key=lambda g: g['threshold'])]
chosen = min(feasible, key=lambda g: g['expected_cost'])
THRESHOLD = chosen['threshold']
oot_op = at_threshold(y_oot_a, p_xgb_oot, amt_oot, THRESHOLD, days['oot'])

operating_point = {'model': 'challenger_xgboost', 'threshold_probability': THRESHOLD,
                   'assumptions': {'alert_capacity_per_day': ALERT_CAPACITY_PER_DAY, 'cost_review': COST_REVIEW,
                                   'cost_missed_fraud': 'transaction amount' if COST_MISSED_FRAUD is None else COST_MISSED_FRAUD},
                   'valid': chosen, 'oot': oot_op}
results['challenger_xgboost']['operating_point_oot'] = oot_op
print(f"\nThreshold {THRESHOLD:.4f} -> OOT: {oot_op['alerts_per_day']} alerts/day, precision {oot_op['precision']:.3f}, "
      f"recall {oot_op['recall']:.3f}, {oot_op['fraud_amount_caught_pct']}% of fraud $ caught")

# %% [markdown]
# ## Validation plots (OOT)

# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle('Champion vs challenger — out-of-time period', fontsize=13)
for p, label, color in [(p_dt_oot, 'Decision tree', '#534AB7'), (p_xgb_oot, 'XGBoost', '#1D9E75')]:
    fpr, tpr, _ = roc_curve(y_oot, p)
    axes[0].plot(fpr, tpr, label=f'{label} (AUC={roc_auc_score(y_oot, p):.3f})', lw=2, color=color)
    prec, rec, _ = precision_recall_curve(y_oot, p)
    axes[1].plot(rec, prec, label=f'{label} (AP={average_precision_score(y_oot, p):.3f})', lw=2, color=color)
axes[0].plot([0, 1], [0, 1], '--', color='gray', lw=1)
axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='ROC'); axes[0].legend(fontsize=9)
axes[1].axhline(y_oot.mean(), color='gray', ls='--', lw=1, label=f'Base rate ({y_oot.mean():.3f})')
axes[1].axvline(oot_op['recall'], color='#D85A30', ls=':', lw=1.5, label='Chosen threshold')
axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision-recall'); axes[1].legend(fontsize=9)
# Reliability diagram
bins = np.linspace(0, 1, 11)
for p, label, color in [(xgb_oot, 'XGBoost raw score', '#9CA3AF'), (p_xgb_oot, 'XGBoost calibrated', '#1D9E75')]:
    b = np.clip(np.digitize(p, bins) - 1, 0, 9)
    df = pd.DataFrame({'b': b, 'p': p, 'y': y_oot_a}).groupby('b').mean()
    axes[2].plot(df['p'], df['y'], 'o-', label=label, color=color)
axes[2].plot([0, 1], [0, 1], '--', color='gray', lw=1)
axes[2].set(xlabel='Predicted', ylabel='Observed fraud rate', title='Reliability (OOT)'); axes[2].legend(fontsize=9)
plt.tight_layout(); plt.savefig(OUT_DIR + '02_champion_challenger.png', dpi=150, bbox_inches='tight'); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle('KS score distributions (OOT)', fontsize=12)
for ax, p, label in [(axes[0], p_dt_oot, 'Decision tree'), (axes[1], p_xgb_oot, 'XGBoost')]:
    ax.hist(p[y_oot_a == 0], bins=50, alpha=0.6, color='#1D9E75', label='Legitimate', density=True)
    ax.hist(p[y_oot_a == 1], bins=50, alpha=0.6, color='#D85A30', label='Fraud', density=True)
    fpr, tpr, _ = roc_curve(y_oot, p)
    ax.set_title(f'{label} — KS = {np.max(tpr - fpr):.4f}', fontsize=10); ax.set_xlabel('Calibrated probability'); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(OUT_DIR + '03_ks_distributions.png', dpi=150, bbox_inches='tight'); plt.close(fig)

# %% [markdown]
# ## SHAP explainability (OOT sample)

# %%
rng = np.random.default_rng(SEED)
X_shap = X_oot.iloc[rng.choice(len(X_oot), size=min(2000, len(X_oot)), replace=False)]
shap_values = shap.TreeExplainer(xgb_model).shap_values(X_shap)
for kind, fname, title in [('bar', '04_shap_importance.png', 'SHAP feature importance — XGBoost'),
                           (None, '05_shap_beeswarm.png', 'SHAP beeswarm — direction of effect')]:
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, plot_type=kind, max_display=20 if kind else 15, show=False)
    plt.title(title, fontsize=12); plt.tight_layout(); plt.savefig(OUT_DIR + fname, dpi=150, bbox_inches='tight'); plt.close('all')

shap_df = pd.DataFrame({'feature': X_shap.columns, 'mean_abs_shap': np.abs(shap_values).mean(axis=0)}) \
            .sort_values('mean_abs_shap', ascending=False)
shap_df.to_csv(OUT_DIR + 'shap_importance.csv', index=False)
shap_df.head(15).to_json(OUT_DIR + 'top_features.json', orient='records', indent=2)

# %% [markdown]
# ## Save models and outputs

# %%
joblib.dump(dt, MODEL_DIR + 'champion_decision_tree.pkl')
joblib.dump(xgb_model, MODEL_DIR + 'challenger_xgboost.pkl')
joblib.dump(calibrators, MODEL_DIR + 'calibrators.pkl')

results['split'] = {p: {'rows': int(idx[p].sum()), 'days': round(float(days[p]), 1),
                        'fraud_rate': round(float(y_all[idx[p]].mean()), 4)} for p in idx}
results['note'] = 'Metrics are on the out-of-time period (last 15% by TransactionDT), calibrated probabilities.'
with open(OUT_DIR + 'metrics.json', 'w') as f:
    json.dump(results, f, indent=2)
with open(OUT_DIR + 'operating_point.json', 'w') as f:
    json.dump(operating_point, f, indent=2)

sub_path = DATA_DIR + 'sample_submission.csv'
if os.path.exists(sub_path):
    test_meta = pd.read_parquet(OUT_DIR + 'test_meta.parquet')
    sub = pd.DataFrame({'TransactionID': test_meta['TransactionID'],
                        'isFraud': calibrators['xgboost'].predict(xgb_model.predict_proba(X_test)[:, 1])})
    sub.to_csv(OUT_DIR + 'submission.csv', index=False)

print("Notebook 2 complete — run 03_risk_pipeline_llm_triage.py next")

# %% [markdown]
# # Monitoring & Triage
# **Part A — PSI drift monitor** on scores and on the top model features.
# **Part B — triage narratives** for alerts: SHAP drivers → plain-English note for an analyst.
#
# Run after `02_model_training_validation.py`.
#
# Optional LLM narratives: `pip install anthropic`, set `ANTHROPIC_API_KEY`
# (and optionally `TRIAGE_MODEL`). Without a key, a deterministic template is used.
# Every narrative is checked so it cannot state a number that is not in its input, and
# every prompt/response is logged to `outputs/triage_log.jsonl` with the model id.

# %%
import os, re, json, hashlib, warnings
from datetime import datetime, timezone
os.environ.setdefault('NUMBA_DISABLE_CUDA', '1')
import numpy as np
import pandas as pd
import matplotlib
if os.environ.get('MPLBACKEND') is None and not os.environ.get('DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import shap
warnings.filterwarnings('ignore')

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_HERE, '..', '..', '.env'))
except ImportError:
    pass

OUT_DIR   = os.path.join(_HERE, '..', 'outputs') + os.sep
MODEL_DIR = os.path.join(_HERE, '..', 'models') + os.sep
TRIAGE_MODEL = os.environ.get('TRIAGE_MODEL', 'claude-haiku-5-5')
N_DEMO_CASES = 5          # alerts that get an LLM narrative (when a key is set)
N_CONSOLE_CASES = 25      # top out-of-time alerts exported for the analyst console

# %%
X_all  = pd.read_parquet(OUT_DIR + 'X_train.parquet')
y_all  = pd.read_parquet(OUT_DIR + 'y_train.parquet')['isFraud']
meta   = pd.read_parquet(OUT_DIR + 'train_meta.parquet')
X_test = pd.read_parquet(OUT_DIR + 'X_test.parquet')
xgb_model   = joblib.load(MODEL_DIR + 'challenger_xgboost.pkl')
calibrator  = joblib.load(MODEL_DIR + 'calibrators.pkl')['xgboost']
with open(OUT_DIR + 'operating_point.json') as f:
    THRESHOLD = json.load(f)['threshold_probability']
with open(OUT_DIR + 'features.json') as f:
    FEATURES = json.load(f)

X_val = X_all[(meta['period'] == 'valid').values]
X_oot, y_oot = X_all[(meta['period'] == 'oot').values], y_all[(meta['period'] == 'oot').values]
prob = lambda X: calibrator.predict(xgb_model.predict_proba(X)[:, 1])

# %% [markdown]
# ## Part A — PSI drift monitor
# Reference = the **valid** period (scored out-of-sample; in-sample fit scores would look artificially
# confident). Bins are **deciles of the reference**, so each holds ~10% of it. Special values
# (-999 = missing) get their own bin.
#
# PSI < 0.10 stable · 0.10–0.25 monitor · > 0.25 review. These are industry rules of thumb; SR 11-7
# asks for ongoing monitoring but sets no numeric thresholds.

# %%
def compute_psi(expected, actual, n_bins=10, special=(-999,)):
    e, a = np.asarray(expected, float), np.asarray(actual, float)
    e_sp, a_sp = np.isin(e, special), np.isin(a, special)
    edges = np.unique(np.quantile(e[~e_sp], np.linspace(0, 1, n_bins + 1))) if (~e_sp).any() else np.array([0.0, 1.0])
    edges[0], edges[-1] = -np.inf, np.inf
    e_cnt = np.append(np.histogram(e[~e_sp], edges)[0], e_sp.sum())
    a_cnt = np.append(np.histogram(a[~a_sp], edges)[0], a_sp.sum())
    e_pct = np.maximum(e_cnt / max(len(e), 1), 1e-4)
    a_pct = np.maximum(a_cnt / max(len(a), 1), 1e-4)
    contrib = (a_pct - e_pct) * np.log(a_pct / e_pct)
    return round(float(contrib.sum()), 5), contrib

def assess(psi):
    return 'stable' if psi < 0.10 else 'monitor' if psi < 0.25 else 'review'

ref_scores = prob(X_val)
oot_scores, test_scores = prob(X_oot), prob(X_test)
psi_score_oot, _ = compute_psi(ref_scores, oot_scores)
psi_score_test, contrib_test = compute_psi(ref_scores, test_scores)

top_feats = pd.read_csv(OUT_DIR + 'shap_importance.csv')['feature'].head(20).tolist()
feature_psi = []
for f in top_feats:
    p_oot, _ = compute_psi(X_val[f], X_oot[f])
    p_test, _ = compute_psi(X_val[f], X_test[f])
    feature_psi.append({'feature': f, 'psi_oot': p_oot, 'psi_kaggle_test': p_test, 'status': assess(max(p_oot, p_test))})
feature_psi.sort(key=lambda r: -r['psi_kaggle_test'])

psi_results = {'reference': 'valid period (calibrated scores)',
               'score': {'psi_valid_to_oot': psi_score_oot, 'psi_valid_to_kaggle_test': psi_score_test,
                         'status': assess(max(psi_score_oot, psi_score_test))},
               'features': feature_psi}
with open(OUT_DIR + 'psi_results.json', 'w') as f:
    json.dump(psi_results, f, indent=2)
print(f"Score PSI  valid→OOT {psi_score_oot:.4f} | valid→Kaggle test {psi_score_test:.4f}")
print(pd.DataFrame(feature_psi).head(10).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].hist(ref_scores, bins=50, alpha=0.6, density=True, color='#1D9E75', label='Valid (reference)')
axes[0].hist(test_scores, bins=50, alpha=0.6, density=True, color='#534AB7', label='Kaggle test')
axes[0].set(xlabel='Calibrated probability', ylabel='Density', title=f'Score PSI valid→test = {psi_score_test:.4f}')
axes[0].set_yscale('log'); axes[0].legend()
fp = pd.DataFrame(feature_psi).head(15).iloc[::-1]
axes[1].barh(fp['feature'], fp['psi_kaggle_test'], color=['#D85A30' if v > 0.25 else '#BA7517' if v > 0.10 else '#1D9E75' for v in fp['psi_kaggle_test']])
for x in (0.10, 0.25):
    axes[1].axvline(x, color='gray', ls='--', lw=1)
axes[1].set(xlabel='PSI (valid → Kaggle test)', title='Feature drift, top SHAP features')
plt.tight_layout(); plt.savefig(OUT_DIR + '06_psi_monitor.png', dpi=150, bbox_inches='tight'); plt.close(fig)

# %% [markdown]
# ## Part B — triage narratives
# Risk tiers are on the **calibrated probability**, anchored to the operating threshold from notebook 2:
# below the threshold no alert is raised.
#
# Many IEEE-CIS columns are masked by Vesta (C1–C14, D1–D15, M1–M9, V…), so the dictionary below says so
# rather than inventing a meaning. The LLM is told to use only the numbers it is given; any narrative that
# states another number is rejected and replaced by the template.

# %%
FEATURE_DICTIONARY = {
    'TransactionAmt': 'transaction amount in USD',
    'ProductCD': 'product code (frequency-encoded)',
    'card1': 'card identifier (masked)', 'card6': 'card type, debit or credit (frequency-encoded)',
    'P_emaildomain': 'purchaser email domain (frequency-encoded)', 'R_emaildomain': 'recipient email domain (frequency-encoded)',
    'dist1': 'distance between addresses (masked meaning)',
    'amt_log': 'log of the amount', 'amt_zscore': "amount compared with this card's earlier amounts, in standard deviations",
    'card1_txn_count_7d': 'transactions on this card in the previous 7 days',
    'card1_prior_amt_mean': 'average amount of earlier transactions on this card (USD)',
    'secs_since_prev_card_txn': 'seconds since the previous transaction on this card',
    'email_mismatch': '1 if purchaser and recipient email domains differ',
    'P_email_anon': '1 if purchaser email is missing or anonymous',
    'is_large_txn': '1 if amount is over $500', 'C_sum': 'sum of the masked C count features',
    'is_mobile': '1 if the device is mobile', 'dt_hour': 'hour of day relative to the dataset reference time',
}
def describe(feature):
    if feature in FEATURE_DICTIONARY:
        return FEATURE_DICTIONARY[feature]
    for prefix, text in (('C', 'masked count feature'), ('D', 'masked time-delta feature (days)'),
                         ('M', 'masked match flag'), ('V', 'masked Vesta engineered feature'), ('id_', 'masked identity feature')):
        if feature.startswith(prefix) and feature[len(prefix):].isdigit():
            return text
    return 'masked or encoded feature'

def risk_tier(p):
    if p < THRESHOLD:      return 'LOW'
    if p < 0.50:           return 'MEDIUM'
    if p < 0.80:           return 'HIGH'
    return 'CRITICAL'

ACTIONS = {'LOW': 'No alert; approve.', 'MEDIUM': 'Manual review within 4 hours.',
           'HIGH': 'Escalate to a senior analyst; consider a temporary hold.',
           'CRITICAL': 'Block and notify the compliance officer.'}

explainer = shap.TreeExplainer(xgb_model)

def explain_transaction(row, top_n=5):
    row_df = row.to_frame().T.astype(float)
    p = float(calibrator.predict(xgb_model.predict_proba(row_df)[:, 1])[0])
    sv = explainer.shap_values(row_df)[0]
    contrib = pd.DataFrame({'feature': row_df.columns, 'value': row_df.iloc[0].values, 'shap': sv})
    shown = lambda v: 'missing' if v == -999 else round(float(v), 2)
    rows = lambda d: [{'feature': r.feature, 'meaning': describe(r.feature), 'value': shown(r.value), 'shap': round(float(r.shap), 3)}
                      for r in d.itertuples()]
    return p, rows(contrib.nlargest(top_n, 'shap')), rows(contrib.nsmallest(top_n, 'shap'))

def template_narrative(p, tier, risk):
    a, b = risk[0], risk[1]
    return (f"Fraud probability {p:.1%} ({tier}). Largest contributions toward fraud: {a['feature']} = {a['value']} "
            f"({a['meaning']}) and {b['feature']} = {b['value']} ({b['meaning']}). {ACTIONS[tier]}")

def numbers_grounded(text, allowed):
    """Every number in the narrative must appear in the prompt input (as given, or as a % of a probability)."""
    allowed_vals = set()
    for v in allowed:
        allowed_vals.update({round(v, 2), round(v, 1), round(v, 0), round(v * 100, 1), round(v * 100, 0), round(abs(v), 2)})
    for tok in re.findall(r'\d+(?:\.\d+)?', text.replace(',', '')):
        x = float(tok)
        if x <= 10 and x == int(x):      # small integers ("top 2 factors", "4 hours") are fine
            continue
        if round(x, 2) not in allowed_vals and round(x, 1) not in allowed_vals and round(x, 0) not in allowed_vals:
            return False
    return True

def llm_narrative(prompt):
    if not os.environ.get('ANTHROPIC_API_KEY'):
        return None
    try:
        import anthropic
        msg = anthropic.Anthropic().messages.create(model=TRIAGE_MODEL, max_tokens=300,
                                                    messages=[{'role': 'user', 'content': prompt}])
        return ''.join(b.text for b in msg.content if getattr(b, 'type', '') == 'text').strip() or None
    except Exception as e:                     # network/auth errors fall back to the template
        print(f"LLM call failed ({type(e).__name__}); using template")
        return None

def triage(row, txn_id, use_llm=True):
    p, risk, protect = explain_transaction(row)
    tier = risk_tier(p)
    payload = {'fraud_probability': round(p, 4), 'risk_tier': tier, 'recommended_action': ACTIONS[tier],
               'risk_increasing': risk, 'risk_reducing': protect}
    prompt = ("You are a fraud analyst writing a triage note for a compliance officer.\n"
              "Use ONLY the facts in the JSON below. Do not state any number that is not in it. "
              "Masked features have no known business meaning: name them, do not guess what they measure.\n"
              "Write 3-4 plain sentences: the probability and tier, the top 2-3 drivers, and the recommended action as given.\n\n"
              + json.dumps(payload, indent=2))
    allowed = [p] + [float(r[k]) for r in risk + protect for k in ('value', 'shap') if isinstance(r[k], (int, float))]
    text, source = (llm_narrative(prompt) if use_llm else None), f'LLM:{TRIAGE_MODEL}'
    if text and not numbers_grounded(text, allowed):
        source, text = 'template (LLM output rejected: ungrounded number)', None
    if not text:
        source = source if source.startswith('template') else 'template'
        text = template_narrative(p, tier, risk)
    with open(OUT_DIR + 'triage_log.jsonl', 'a') as f:
        f.write(json.dumps({'ts': datetime.now(timezone.utc).isoformat(), 'transaction_id': int(txn_id),
                            'model_version': 'challenger_xgboost', 'narrative_source': source,
                            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'prompt': prompt,
                            'narrative': text}) + '\n')
    return {'transaction_id': int(txn_id), 'fraud_probability': round(p, 4), 'risk_tier': tier,
            'narrative': text, 'source': source, 'top_risk_factors': [r['feature'] for r in risk],
            'drivers': [[r['feature'], str(r['value']), r['shap']] for r in risk[:4] + protect[:3]]}

# %% [markdown]
# ## Demo — the highest-probability OOT alerts

# %%
oot_meta = meta.loc[(meta['period'] == 'oot').values].reset_index(drop=True)
order = np.argsort(-oot_scores)[:max(N_DEMO_CASES, N_CONSOLE_CASES)]
results = []
for rank, i in enumerate(order, 1):
    r = triage(X_oot.iloc[i], oot_meta.loc[i, 'TransactionID'], use_llm=rank <= N_DEMO_CASES)
    r['actual'] = 'FRAUD' if y_oot.iloc[i] == 1 else 'LEGIT'
    r['amount'] = round(float(X_oot.iloc[i]['TransactionAmt']), 2)
    for col in ('ProductCD', 'card4', 'card6', 'P_emaildomain', 'DeviceType'):
        r[col] = None if col not in oot_meta or pd.isna(oot_meta.loc[i, col]) else str(oot_meta.loc[i, col])
    results.append(r)
    if rank <= N_DEMO_CASES:
        print(f"\n#{rank} {r['transaction_id']} | {r['fraud_probability']:.1%} {r['risk_tier']} | actual {r['actual']} | {r['source']}\n  {r['narrative']}")
pd.DataFrame(results[:N_DEMO_CASES]).drop(columns='drivers').to_csv(OUT_DIR + 'triage_demo.csv', index=False)
with open(OUT_DIR + 'triage_cases.json', 'w') as f:
    json.dump(results[:N_CONSOLE_CASES], f, indent=1)

# %% [markdown]
# ## Results summary → `outputs/RESULTS.md`
# README numbers are copied from this file, so the README never drifts from what the code produced.

# %%
with open(OUT_DIR + 'metrics.json') as f:
    m = json.load(f)
with open(OUT_DIR + 'operating_point.json') as f:
    op = json.load(f)
dtm, xgm = m['champion_decision_tree']['oot'], m['challenger_xgboost']['oot']
lines = ['# Results (generated by notebook 3)', '',
         f"Out-of-time period: {m['split']['oot']['rows']:,} transactions over {m['split']['oot']['days']} days, "
         f"fraud rate {m['split']['oot']['fraud_rate']*100:.2f}%. Calibrated probabilities.", '',
         '| Metric | Champion: decision tree | Challenger: XGBoost |', '|---|---|---|']
lines += [f"| {k} | {dtm[key]:.4f} | {xgm[key]:.4f} |" for k, key in
          [('KS', 'ks'), ('Gini', 'gini'), ('AUC-ROC', 'auc_roc'), ('AUC-PR', 'auc_pr'), ('Brier', 'brier')]]
o = op['oot']
lines += ['', f"**Operating point** (chosen on the valid period, capacity {op['assumptions']['alert_capacity_per_day']} alerts/day): "
          f"threshold {op['threshold_probability']:.4f} → on OOT {o['alerts_per_day']} alerts/day, precision {o['precision']:.3f}, "
          f"recall {o['recall']:.3f}, {o['fraud_amount_caught_pct']}% of fraud dollars caught.", '',
          f"**Score PSI**: valid→OOT {psi_score_oot:.4f}, valid→Kaggle test {psi_score_test:.4f} ({assess(max(psi_score_oot, psi_score_test))}).",
          f"Features needing review (PSI > 0.25): {', '.join(r['feature'] for r in feature_psi if r['status'] == 'review') or 'none'}."]
with open(OUT_DIR + 'RESULTS.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('\n'.join(lines))

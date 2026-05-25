"""
╔══════════════════════════════════════════════════════════════╗
║   BFSI FRAUD RISK CLASSIFIER — NOTEBOOK 3                   ║
║   Production Risk Pipeline:                                  ║
║   PSI Drift Monitor + LLM Triage Agent                      ║
╚══════════════════════════════════════════════════════════════╝

Run AFTER 02_model_training_validation.py
This is the differentiating notebook — the AI layer that turns 
your project from 'ML assignment' to 'production risk system'

REQUIREMENTS:
  pip install requests
  Set env var: export OPENROUTER_API_KEY="your-key-here"
  Get a free key at: openrouter.ai/keys
"""

# ── CELL 1: Imports ───────────────────────────────────────────
import os
os.environ.setdefault('NUMBA_DISABLE_CUDA', '1')

# Load .env from project root (two levels up from this file)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_HERE := os.path.dirname(os.path.abspath(__file__)),
                             '..', '..', '.env'))
except ImportError:
    pass  # dotenv not installed; rely on system env vars

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, joblib, warnings
warnings.filterwarnings('ignore')

import shap

_HERE     = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(_HERE, '..', 'outputs') + os.sep
MODEL_DIR = os.path.join(_HERE, '..', 'models') + os.sep
DATA_DIR  = os.path.join(_HERE, '..', 'data', 'ieee-fraud-detection') + os.sep

print("✓ Imports OK")


# ── CELL 2: Load model and data ───────────────────────────────
X_train = pd.read_parquet(OUT_DIR + 'X_train.parquet')
y_train = pd.read_parquet(OUT_DIR + 'y_train.parquet')['isFraud']
X_test  = pd.read_parquet(OUT_DIR + 'X_test.parquet')

xgb_model = joblib.load(MODEL_DIR + 'challenger_xgboost.pkl')

with open(OUT_DIR + 'features.json') as f:
    FEATURES = json.load(f)

print(f"Model loaded. Features: {len(FEATURES)}")


# ══════════════════════════════════════════════════════════════
# PART A: PSI DRIFT MONITOR
# ══════════════════════════════════════════════════════════════
#
# PSI (Population Stability Index) is THE standard metric in 
# bank model risk management (SR 11-7) for detecting when
# live data has drifted from training data.
#
# PSI < 0.10  → Stable. No action needed.
# PSI 0.10–0.25 → Minor drift. Monitor closely.
# PSI > 0.25  → Major drift. Trigger model review.
#
# In production this runs nightly comparing training distribution
# vs yesterday's transaction scores.
# ══════════════════════════════════════════════════════════════

# ── CELL 3: PSI function ─────────────────────────────────────
def compute_psi(expected, actual, n_bins=10):
    """
    Compute Population Stability Index between two score distributions.
    
    Parameters
    ----------
    expected : array-like — scores from training data (reference)
    actual   : array-like — scores from new/test data
    n_bins   : int — number of equal-width bins
    
    Returns
    -------
    psi_total : float
    psi_df    : DataFrame with per-bin breakdown
    """
    expected = np.array(expected)
    actual   = np.array(actual)
    
    # Build bins on expected distribution
    breakpoints = np.linspace(0, 1, n_bins + 1)
    
    expected_pcts, actual_pcts = [], []
    for i in range(n_bins):
        lo, hi = breakpoints[i], breakpoints[i+1]
        exp_pct = np.mean((expected >= lo) & (expected < hi))
        act_pct = np.mean((actual   >= lo) & (actual   < hi))
        expected_pcts.append(max(exp_pct, 1e-4))  # avoid log(0)
        actual_pcts.append(max(act_pct, 1e-4))
    
    expected_pcts = np.array(expected_pcts)
    actual_pcts   = np.array(actual_pcts)
    
    psi_bins  = (actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts)
    psi_total = psi_bins.sum()
    
    psi_df = pd.DataFrame({
        'bin': [f'{breakpoints[i]:.1f}–{breakpoints[i+1]:.1f}' for i in range(n_bins)],
        'expected_pct': (expected_pcts * 100).round(2),
        'actual_pct':   (actual_pcts   * 100).round(2),
        'psi_contribution': psi_bins.round(5)
    })
    
    return round(psi_total, 5), psi_df


def interpret_psi(psi):
    if psi < 0.10:
        return "STABLE — No action needed"
    elif psi < 0.25:
        return "MINOR DRIFT — Monitor closely, schedule review"
    else:
        return "MAJOR DRIFT — Model review required (SR 11-7 trigger)"


# ── CELL 4: Run PSI — train vs test ──────────────────────────
# Use train (reference) vs test (live population) scores
train_scores = xgb_model.predict_proba(X_train)[:, 1]
test_scores  = xgb_model.predict_proba(X_test)[:, 1]

psi_value, psi_table = compute_psi(train_scores, test_scores)

print("\n" + "═"*50)
print("  PSI DRIFT MONITORING REPORT")
print("  Reference: Training data | Actual: Test data")
print("═"*50)
print(f"  PSI = {psi_value:.5f}")
print(f"  Assessment: {interpret_psi(psi_value)}")
print("═"*50)
print("\nPer-bin breakdown:")
print(psi_table.to_string(index=False))


# ── CELL 5: PSI visualisation ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'PSI Drift Monitor — PSI = {psi_value:.4f} ({interpret_psi(psi_value)})',
             fontsize=11)

# Score distributions
axes[0].hist(train_scores, bins=50, alpha=0.6, color='#1D9E75',
             label='Training (reference)', density=True)
axes[0].hist(test_scores,  bins=50, alpha=0.6, color='#534AB7',
             label='Test (actual)', density=True)
axes[0].set_xlabel('Fraud probability score')
axes[0].set_ylabel('Density')
axes[0].set_title('Score distributions')
axes[0].legend()

# PSI per bin
colors = ['#D85A30' if v > 0.025 else '#1D9E75' for v in psi_table['psi_contribution']]
axes[1].bar(psi_table['bin'], psi_table['psi_contribution'], color=colors)
axes[1].axhline(y=0.01, color='orange', linestyle='--', linewidth=1,
                label='Contribution threshold (0.01)')
axes[1].set_xlabel('Score bin')
axes[1].set_ylabel('PSI contribution')
axes[1].set_title('PSI per score bin')
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR + '06_psi_monitor.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n✓ PSI plot saved → {OUT_DIR}06_psi_monitor.png")


# ══════════════════════════════════════════════════════════════
# PART B: LLM TRIAGE AGENT
# ══════════════════════════════════════════════════════════════
#
# This is the differentiating layer of the project.
#
# In a real bank:
# - The model scores 500,000 transactions per day
# - ~3,000 are flagged (score > threshold)
# - Investigators can review ~200 per analyst per day
# - The LLM triage layer auto-generates a SAR-style narrative
#   for each flagged transaction — telling the analyst exactly
#   WHY the model flagged it, in plain English
#
# This replaces hours of manual write-up per case.
# ══════════════════════════════════════════════════════════════

# ── CELL 6: Set up SHAP explainer ────────────────────────────
print("\nSetting up SHAP explainer for transaction-level explanations...")
explainer = shap.TreeExplainer(xgb_model)
print("✓ SHAP explainer ready")


# ── CELL 7: Transaction explanation function ─────────────────
def explain_transaction(transaction_row, model, explainer, feature_names, top_n=5):
    """
    Generate SHAP-based explanation for a single transaction.
    Returns the top N features driving the fraud score.
    """
    row_df = pd.DataFrame([transaction_row], columns=feature_names)
    score  = model.predict_proba(row_df)[0, 1]
    sv     = explainer.shap_values(row_df)[0]
    
    contrib = pd.DataFrame({
        'feature': feature_names,
        'value':   transaction_row,
        'shap':    sv
    }).sort_values('shap', ascending=False)
    
    top_risk    = contrib.head(top_n)
    top_protect = contrib.tail(top_n).sort_values('shap')
    
    return score, top_risk, top_protect


# ── CELL 8: LLM triage function ──────────────────────────────
def generate_triage_narrative(transaction_row, model, explainer,
                               feature_names, api_key=None):
    """
    Generate a SAR-style risk narrative for a flagged transaction
    using Claude via the Anthropic API.
    
    If no API key is provided, returns a template-based narrative
    (still useful for portfolio demo purposes).
    """
    score, top_risk, top_protect = explain_transaction(
        transaction_row, model, explainer, feature_names
    )
    
    risk_factors = "\n".join([
        f"  - {row['feature']}: value={row['value']:.3f}, risk_contribution={row['shap']:.4f}"
        for _, row in top_risk.iterrows()
    ])
    protect_factors = "\n".join([
        f"  - {row['feature']}: value={row['value']:.3f}, contribution={row['shap']:.4f}"
        for _, row in top_protect.iterrows()
    ])
    
    prompt = f"""You are a financial crime risk analyst at a bank.
A transaction has been flagged by our fraud detection model with a score of {score:.1%}.

Top risk-increasing factors (from SHAP analysis):
{risk_factors}

Top risk-reducing factors:
{protect_factors}

Write a concise SAR-style triage narrative (3-4 sentences) that:
1. States the fraud probability and overall risk level (Low/Medium/High/Critical)
2. Explains the top 2-3 risk factors in plain English
3. Recommends an action: Auto-approve, Manual review, Immediate block

Be specific and professional. Do not use jargon. Write for a compliance officer.
Do not read out loud your thinking process — just give the final narrative."""

    # Try OpenRouter API if key available — tries models in order until one responds
    key = api_key or os.environ.get('OPENROUTER_API_KEY')
    _FREE_MODELS = [
        ("deepseek/deepseek-chat:free",              "DeepSeek-Chat"),
        ("meta-llama/llama-3.3-70b-instruct:free",   "Llama-3.3-70B"),
        ("google/gemma-4-31b-it:free",               "Gemma-4-31B"),
        ("mistralai/mistral-7b-instruct:free",        "Mistral-7B"),
        ("qwen/qwen3-235b-a22b:free",                "Qwen3-235B"),
        ("nvidia/nemotron-3-super-120b-a12b:free",   "Nemotron-120B"),
        ("microsoft/phi-4-reasoning-plus:free",      "Phi-4-Reasoning"),
        ("openrouter/free",                         "OpenRouter-Free")  # fallback to any available free model
    ]
    if key:
        import requests as _requests
        import time as _time
        narrative = source = None
        for model_id, model_name in _FREE_MODELS:
            try:
                resp = _requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": model_id, "max_tokens": 300,
                          "messages": [{"role": "user", "content": prompt}]},
                    timeout=30,
                )
                if resp.status_code == 429:
                    _time.sleep(2)
                    continue  # rate-limited, try next model
                resp.raise_for_status()
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if content:
                    narrative = content
                    source = f"{model_name} via OpenRouter"
                    break
            except Exception:
                continue
        if not narrative:
            narrative = _template_narrative(score, top_risk)
            source = "Template (all free models rate-limited)"
    else:
        narrative = _template_narrative(score, top_risk)
        source = "Template (set OPENROUTER_API_KEY for LLM narratives)"
    
    return {
        'fraud_score': round(score, 4),
        'risk_level': _risk_level(score),
        'narrative': narrative,
        'source': source,
        'top_risk_factors': top_risk[['feature','shap']].to_dict('records')
    }


def _risk_level(score):
    if score < 0.3:  return 'LOW'
    if score < 0.5:  return 'MEDIUM'
    if score < 0.75: return 'HIGH'
    return 'CRITICAL'


def _template_narrative(score, top_risk):
    """Fallback when no API key — still useful for demo."""
    level = _risk_level(score)
    factor1 = top_risk.iloc[0]['feature'] if len(top_risk) > 0 else 'unknown'
    factor2 = top_risk.iloc[1]['feature'] if len(top_risk) > 1 else 'unknown'
    
    actions = {
        'LOW': 'Auto-approve transaction.',
        'MEDIUM': 'Route to manual review queue within 4 hours.',
        'HIGH': 'Escalate to senior analyst. Consider temporary hold.',
        'CRITICAL': 'Immediate block recommended. Notify compliance officer.'
    }
    
    return (
        f"Transaction flagged at {score:.1%} fraud probability — {level} risk. "
        f"Primary risk drivers are elevated '{factor1}' and '{factor2}' signals "
        f"relative to this cardholder's historical profile. "
        f"{actions[level]}"
    )


# ── CELL 9: Demo — triage 5 high-risk transactions ───────────
print("\n" + "═"*60)
print("  FRAUD TRIAGE AGENT — LIVE DEMO")
print("  (Set OPENROUTER_API_KEY for LLM-generated narratives)")
print("═"*60)

# Score validation set
split_idx   = int(len(X_train) * 0.8)
X_val       = X_train.iloc[split_idx:]
y_val       = y_train.iloc[split_idx:]
val_scores  = xgb_model.predict_proba(X_val)[:, 1]

# Pick 5 highest-scoring transactions for demo
top5_idx = np.argsort(val_scores)[-5:][::-1]

triage_results = []
for rank, idx in enumerate(top5_idx, 1):
    txn_row = X_val.iloc[idx].values
    result  = generate_triage_narrative(
        txn_row, xgb_model, explainer, FEATURES
    )
    actual_label = y_val.iloc[idx]
    
    print(f"\n{'─'*60}")
    print(f"  CASE #{rank}  |  Actual label: {'FRAUD' if actual_label else 'LEGIT'}")
    print(f"  Fraud score : {result['fraud_score']:.1%}")
    print(f"  Risk level  : {result['risk_level']}")
    print(f"  Source      : {result['source']}")
    print(f"\n  NARRATIVE:")
    print(f"  {result['narrative']}")
    print(f"\n  TOP RISK FACTORS:")
    for f in result['top_risk_factors']:
        print(f"    • {f['feature']}: SHAP = {f['shap']:.4f}")
    
    triage_results.append({
        'rank': rank,
        'actual_fraud': int(actual_label),
        **result
    })

# Save triage demo output
triage_df = pd.DataFrame(triage_results)
triage_df.to_csv(OUT_DIR + 'triage_demo.csv', index=False)
print(f"\n✓ Triage demo saved → {OUT_DIR}triage_demo.csv")


# ── CELL 10: Final summary ────────────────────────────────────
with open(OUT_DIR + 'metrics.json') as f:
    metrics = json.load(f)

print("\n" + "═"*60)
print("  PROJECT COMPLETE — FINAL RESULTS SUMMARY")
print("═"*60)

for model_name, m in metrics.items():
    label = "CHAMPION" if "decision_tree" in model_name else "CHALLENGER"
    name  = "Decision Tree" if "decision_tree" in model_name else "XGBoost"
    print(f"\n  {label}: {name}")
    print(f"    KS Statistic  : {m['ks']:.4f}")
    print(f"    Gini          : {m['gini']:.4f}")
    print(f"    AUC-ROC       : {m['auc_roc']:.4f}")
    print(f"    AUC-PR        : {m['auc_pr']:.4f}")

print(f"\n  PSI (train vs test) : {psi_value:.5f} — {interpret_psi(psi_value)}")
print(f"\n  Outputs saved to    : {OUT_DIR}")
print("═"*60)
print("\n✓ All 3 notebooks complete. Push to GitHub.")

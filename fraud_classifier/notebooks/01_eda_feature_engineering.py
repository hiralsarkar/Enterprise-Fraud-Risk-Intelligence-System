"""
╔══════════════════════════════════════════════════════════════╗
║   BFSI FRAUD RISK CLASSIFIER — NOTEBOOK 1                   ║
║   EDA & Feature Engineering                                  ║
║   IEEE-CIS Fraud Detection Dataset                          ║
╚══════════════════════════════════════════════════════════════╝

Run this file as a script or copy cells into Jupyter.
Outputs are saved to ../outputs/
"""

# ── CELL 1: Imports & config ──────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', '{:.3f}'.format)

import os
_HERE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, '..', 'data', 'ieee-fraud-detection') + os.sep
OUT_DIR  = os.path.join(_HERE, '..', 'outputs') + os.sep
os.makedirs(OUT_DIR, exist_ok=True)

print("✓ Imports OK")


# ── CELL 2: Load data ─────────────────────────────────────────
print("\nLoading data (this takes ~30s on the full dataset)...")

train_txn  = pd.read_csv(DATA_DIR + 'train_transaction.csv')
train_id   = pd.read_csv(DATA_DIR + 'train_identity.csv')
test_txn   = pd.read_csv(DATA_DIR + 'test_transaction.csv')
test_id    = pd.read_csv(DATA_DIR + 'test_identity.csv')

print(f"Train transactions : {train_txn.shape}")
print(f"Train identity     : {train_id.shape}")
print(f"Test  transactions : {test_txn.shape}")
print(f"Test  identity     : {test_id.shape}")


# ── CELL 3: Merge identity tables ────────────────────────────
train = train_txn.merge(train_id, on='TransactionID', how='left')
test  = test_txn.merge(test_id,   on='TransactionID', how='left')

print(f"\nMerged train : {train.shape}")
print(f"Merged test  : {test.shape}")


# ── CELL 4: Target overview ───────────────────────────────────
fraud_rate = train['isFraud'].mean()
fraud_n    = train['isFraud'].sum()

print(f"\n{'═'*45}")
print(f"  FRAUD OVERVIEW")
print(f"{'═'*45}")
print(f"  Total transactions : {len(train):,}")
print(f"  Fraud cases        : {fraud_n:,}")
print(f"  Legitimate cases   : {len(train)-fraud_n:,}")
print(f"  Fraud rate         : {fraud_rate*100:.2f}%")
print(f"  Class imbalance    : 1 : {round((1-fraud_rate)/fraud_rate)}")
print(f"{'═'*45}")

# ── CELL 5: EDA plots ─────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle('IEEE-CIS Fraud Detection — Exploratory Analysis', fontsize=14, y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# 5a: Class imbalance
ax1 = fig.add_subplot(gs[0, 0])
counts = train['isFraud'].value_counts()
bars = ax1.bar(['Legitimate', 'Fraud'], counts.values,
               color=['#1D9E75', '#D85A30'], width=0.5)
ax1.set_title('Class distribution', fontsize=11)
ax1.set_ylabel('Count')
for bar, count in zip(bars, counts.values):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
             f'{count:,}', ha='center', fontsize=9)

# 5b: Transaction amount by fraud
ax2 = fig.add_subplot(gs[0, 1])
legit_amt = train[train['isFraud']==0]['TransactionAmt'].clip(upper=1000)
fraud_amt = train[train['isFraud']==1]['TransactionAmt'].clip(upper=1000)
ax2.hist(legit_amt, bins=50, alpha=0.6, color='#1D9E75', label='Legit',  density=True)
ax2.hist(fraud_amt, bins=50, alpha=0.6, color='#D85A30', label='Fraud', density=True)
ax2.set_title('Transaction amount distribution', fontsize=11)
ax2.set_xlabel('Amount (USD, clipped at $1000)')
ax2.legend(fontsize=9)

# 5c: Fraud rate by ProductCD
ax3 = fig.add_subplot(gs[0, 2])
prod = train.groupby('ProductCD')['isFraud'].mean().sort_values(ascending=False)
bars3 = ax3.bar(prod.index, prod.values*100, color='#534AB7', width=0.5)
ax3.set_title('Fraud rate by product code', fontsize=11)
ax3.set_ylabel('Fraud rate (%)')
for bar, val in zip(bars3, prod.values*100):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
             f'{val:.1f}%', ha='center', fontsize=8)

# 5d: Fraud rate by card type
ax4 = fig.add_subplot(gs[1, 0])
card = train.groupby('card4')['isFraud'].mean().dropna().sort_values(ascending=False)
ax4.barh(card.index, card.values*100, color='#534AB7')
ax4.set_title('Fraud rate by card network', fontsize=11)
ax4.set_xlabel('Fraud rate (%)')

# 5e: Fraud rate by card6 (debit/credit)
ax5 = fig.add_subplot(gs[1, 1])
c6 = train.groupby('card6')['isFraud'].mean().dropna().sort_values(ascending=False)
ax5.bar(c6.index, c6.values*100, color=['#D85A30','#1D9E75','#534AB7','#BA7517'][:len(c6)], width=0.4)
ax5.set_title('Fraud rate by card type', fontsize=11)
ax5.set_ylabel('Fraud rate (%)')

# 5f: Transaction time pattern (hour of day proxy)
ax6 = fig.add_subplot(gs[1, 2])
# TransactionDT is seconds from reference point; derive hour-of-day
train['hour'] = (train['TransactionDT'] // 3600) % 24
hour_fraud = train.groupby('hour')['isFraud'].mean()
ax6.plot(hour_fraud.index, hour_fraud.values*100, color='#D85A30', linewidth=2)
ax6.fill_between(hour_fraud.index, hour_fraud.values*100, alpha=0.15, color='#D85A30')
ax6.set_title('Fraud rate by hour of day', fontsize=11)
ax6.set_xlabel('Hour (UTC)')
ax6.set_ylabel('Fraud rate (%)')

plt.savefig(OUT_DIR + '01_eda_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n✓ EDA plot saved → {OUT_DIR}01_eda_overview.png")


# ── CELL 6: Null analysis ─────────────────────────────────────
null_pct = (train.isnull().sum() / len(train) * 100).sort_values(ascending=False)
# Drop columns with >90% nulls — not enough signal
drop_cols = null_pct[null_pct > 90].index.tolist()
print(f"\nColumns with >90% nulls (will drop): {len(drop_cols)}")
print(f"First 10: {drop_cols[:10]}")

train.drop(columns=drop_cols, inplace=True)
test.drop(columns=[c for c in drop_cols if c in test.columns], inplace=True)
print(f"\nDataset after null-drop: {train.shape}")


# ── CELL 7: Feature engineering ──────────────────────────────
print("\nEngineering features...")

def engineer_features(df):
    df = df.copy()

    # 1. Log-transform transaction amount (right-skewed)
    df['amt_log'] = np.log1p(df['TransactionAmt'])

    # 2. Amount z-score per card1 (unusual spend for that card)
    card_stats = df.groupby('card1')['TransactionAmt'].agg(['mean','std']).reset_index()
    card_stats.columns = ['card1','card1_amt_mean','card1_amt_std']
    df = df.merge(card_stats, on='card1', how='left')
    df['amt_zscore'] = (
        (df['TransactionAmt'] - df['card1_amt_mean'])
        / df['card1_amt_std'].replace(0, np.nan)
    ).fillna(0)

    # 3. Email domain mismatch (purchaser vs recipient)
    df['email_mismatch'] = (
        df['P_emaildomain'].fillna('') != df['R_emaildomain'].fillna('')
    ).astype(int)

    # 4. High-risk email domain flag
    risky_domains = ['gmail.com','yahoo.com','hotmail.com','anonymous.com','']
    df['P_email_risky'] = df['P_emaildomain'].fillna('').isin(risky_domains).astype(int)

    # 5. Hour of day (from TransactionDT)
    df['hour'] = (df['TransactionDT'] // 3600) % 24
    df['is_night'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)

    # 6. Transaction count per card (velocity proxy)
    card_txn_count = df.groupby('card1')['TransactionID'].transform('count')
    df['card1_txn_count'] = card_txn_count

    # 7. Addr mismatch (billing vs shipping country proxy)
    df['addr_mismatch'] = (
        df['addr1'].fillna(-1) != df['addr2'].fillna(-1)
    ).astype(int)

    # 8. Large transaction flag (>500 USD)
    df['is_large_txn'] = (df['TransactionAmt'] > 500).astype(int)

    # 9. C-column sum (count features — number of previous addresses etc)
    c_cols = [c for c in df.columns if c.startswith('C') and c[1:].isdigit()]
    df['C_sum'] = df[c_cols].fillna(0).sum(axis=1)

    # 10. Device type binary
    df['is_mobile'] = (df.get('DeviceType','').fillna('') == 'mobile').astype(int)

    return df

train = engineer_features(train)
test  = engineer_features(test)

new_features = ['amt_log','amt_zscore','email_mismatch','P_email_risky',
                'hour','is_night','card1_txn_count','addr_mismatch',
                'is_large_txn','C_sum','is_mobile']

print(f"✓ Engineered {len(new_features)} new features:")
for f in new_features:
    fraud_mean = train.groupby('isFraud')[f].mean()
    print(f"  {f:<22} legit={fraud_mean.get(0,0):.3f}  fraud={fraud_mean.get(1,0):.3f}")


# ── CELL 8: Encode categoricals ───────────────────────────────
cat_cols = train.select_dtypes(include='object').columns.tolist()
cat_cols = [c for c in cat_cols if c not in ['TransactionID'] and c in test.columns]

print(f"\nEncoding {len(cat_cols)} categorical columns...")
for col in cat_cols:
    # Label-encode as frequency rank (works well for tree models)
    freq = train[col].value_counts()
    train[col] = train[col].map(freq).fillna(0)
    test[col]  = test[col].map(freq).fillna(0)

print("✓ Categorical encoding done")


# ── CELL 9: Prepare final feature set ────────────────────────
DROP = ['TransactionID', 'isFraud', 'hour',   # hour already captured in is_night
        'card1_amt_mean', 'card1_amt_std']     # intermediate cols

FEATURES = [c for c in train.columns if c not in DROP and c in test.columns]

X_train = train[FEATURES].fillna(-999)
y_train = train['isFraud']
X_test  = test[FEATURES].fillna(-999)

print(f"\nFinal feature count : {len(FEATURES)}")
print(f"X_train shape       : {X_train.shape}")
print(f"X_test  shape       : {X_test.shape}")
print(f"Fraud rate in train : {y_train.mean()*100:.2f}%")

# Save for notebook 2
X_train.to_parquet(OUT_DIR + 'X_train.parquet', index=False)
y_train.to_frame().to_parquet(OUT_DIR + 'y_train.parquet', index=False)
X_test.to_parquet(OUT_DIR + 'X_test.parquet', index=False)

import json
with open(OUT_DIR + 'features.json', 'w') as f:
    json.dump(FEATURES, f)

print(f"\n✓ Saved X_train, y_train, X_test → {OUT_DIR}")
print("✓ Notebook 1 complete — run 02_model_training.py next")

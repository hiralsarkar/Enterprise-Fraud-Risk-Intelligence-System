# %% [markdown]
# # EDA & Feature Engineering
# **IEEE-CIS Fraud Detection (Vesta Corporation)**
#
# The dataset ships as a transaction table and an identity table. This notebook merges them,
# defines the time-ordered **fit / validation / out-of-time (OOT)** periods used everywhere downstream,
# and builds features that are **point-in-time**: every feature for a transaction uses only that
# transaction and the ones before it, and every learned encoding is fitted on the fit period only.
#
# Run as a script (`python 01_eda_feature_engineering.py`) or open the `.ipynb` export.
# Outputs go to `../outputs/`.

# %%
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
if os.environ.get('MPLBACKEND') is None and not os.environ.get('DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:            # running inside Jupyter
    _HERE = os.getcwd()
DATA_DIR = os.environ.get('FRAUD_DATA_DIR', os.path.join(_HERE, '..', 'data', 'ieee-fraud-detection')) + os.sep
OUT_DIR  = os.path.join(_HERE, '..', 'outputs') + os.sep
os.makedirs(OUT_DIR, exist_ok=True)

# Time-ordered split of the labelled data. The OOT slice is never used for fitting,
# early stopping, calibration or threshold choice — only for the final report.
FIT_FRAC, VALID_FRAC = 0.70, 0.15          # OOT = remaining 0.15

# %% [markdown]
# ## Load & merge
# Left join on `TransactionID`: not every transaction has an identity record.

# %%
train_txn = pd.read_csv(DATA_DIR + 'train_transaction.csv')
train_id  = pd.read_csv(DATA_DIR + 'train_identity.csv')
test_txn  = pd.read_csv(DATA_DIR + 'test_transaction.csv')
test_id   = pd.read_csv(DATA_DIR + 'test_identity.csv')
# The Kaggle test identity file uses "id-01" instead of "id_01"
test_id.columns = [c.replace('-', '_') for c in test_id.columns]

train = train_txn.merge(train_id, on='TransactionID', how='left').sort_values('TransactionDT').reset_index(drop=True)
test  = test_txn.merge(test_id,  on='TransactionID', how='left').sort_values('TransactionDT').reset_index(drop=True)
print(f"Merged train : {train.shape}   Merged test : {test.shape}")

# %% [markdown]
# ## Time-ordered periods
# `TransactionDT` is a seconds offset from an undisclosed reference date, so splits are by position
# in time, never random.

# %%
n = len(train)
fit_end, valid_end = int(n * FIT_FRAC), int(n * (FIT_FRAC + VALID_FRAC))
train['period'] = np.select([train.index < fit_end, train.index < valid_end], ['fit', 'valid'], 'oot')
fit_mask = train['period'] == 'fit'

fraud_rate = train['isFraud'].mean()
print(f"Total {n:,} | fraud {int(train['isFraud'].sum()):,} ({fraud_rate*100:.2f}%)")
print(train.groupby('period')['isFraud'].agg(['size', 'mean']).rename(columns={'size': 'rows', 'mean': 'fraud_rate'}))

# %% [markdown]
# ## EDA plots

# %%
fig = plt.figure(figsize=(16, 12))
fig.suptitle('IEEE-CIS Fraud Detection — Exploratory Analysis', fontsize=14, y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
counts = train['isFraud'].value_counts().sort_index()
bars = ax1.bar(['Legitimate', 'Fraud'], counts.values, color=['#1D9E75', '#D85A30'], width=0.5)
ax1.set_title('Class distribution', fontsize=11); ax1.set_ylabel('Count')
for bar, c in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01, f'{c:,}', ha='center', fontsize=9)

ax2 = fig.add_subplot(gs[0, 1])
for lab, col, name in [(0, '#1D9E75', 'Legit'), (1, '#D85A30', 'Fraud')]:
    ax2.hist(train.loc[train['isFraud'] == lab, 'TransactionAmt'].clip(upper=1000), bins=50,
             alpha=0.6, color=col, label=name, density=True)
ax2.set_title('Transaction amount distribution', fontsize=11)
ax2.set_xlabel('Amount (USD, clipped at $1000)'); ax2.legend(fontsize=9)

ax3 = fig.add_subplot(gs[0, 2])
prod = train.groupby('ProductCD')['isFraud'].mean().sort_values(ascending=False)
ax3.bar(prod.index, prod.values * 100, color='#534AB7', width=0.5)
ax3.set_title('Fraud rate by product code', fontsize=11); ax3.set_ylabel('Fraud rate (%)')

ax4 = fig.add_subplot(gs[1, 0])
card = train.groupby('card4')['isFraud'].mean().dropna().sort_values(ascending=False)
ax4.barh(card.index, card.values * 100, color='#534AB7')
ax4.set_title('Fraud rate by card network', fontsize=11); ax4.set_xlabel('Fraud rate (%)')

ax5 = fig.add_subplot(gs[1, 1])
c6 = train.groupby('card6')['isFraud'].mean().dropna().sort_values(ascending=False)
ax5.bar(c6.index, c6.values * 100, color='#534AB7', width=0.4)
ax5.set_title('Fraud rate by card type', fontsize=11); ax5.set_ylabel('Fraud rate (%)')

ax6 = fig.add_subplot(gs[1, 2])
# Hour relative to the undisclosed reference time, NOT a clock hour in any known timezone
hour_fraud = train.groupby((train['TransactionDT'] // 3600) % 24)['isFraud'].mean()
ax6.plot(hour_fraud.index, hour_fraud.values * 100, color='#D85A30', linewidth=2)
ax6.set_title('Fraud rate by relative hour (TransactionDT offset)', fontsize=11)
ax6.set_xlabel('Hour of day, relative'); ax6.set_ylabel('Fraud rate (%)')

plt.savefig(OUT_DIR + '01_eda_overview.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# %% [markdown]
# ## Drop near-empty columns
# Decided on the **fit period only**, then applied to every period and to the Kaggle test set.

# %%
null_pct = train.loc[fit_mask].isnull().mean() * 100
drop_cols = [c for c in null_pct[null_pct > 90].index if c not in ('isFraud', 'period')]
train.drop(columns=drop_cols, inplace=True)
test.drop(columns=[c for c in drop_cols if c in test.columns], inplace=True)
print(f"Dropped {len(drop_cols)} columns with >90% nulls in the fit period")

# %% [markdown]
# ## Point-in-time feature engineering
# Card history features use only **earlier** transactions on the same `card1` (cumulative sums shifted
# by one row), so no transaction sees its own future. Train and Kaggle test are processed as one
# chronological stream: test transactions happen after train and may use train history, which is what
# a live system would see.
#
# | Feature | Definition |
# |---|---|
# | `amt_log` | log(1 + amount) |
# | `card1_txn_count_7d` | transactions on this card in the previous 7 days |
# | `card1_prior_amt_mean` | mean amount of earlier transactions on this card |
# | `amt_zscore` | amount vs. this card's earlier amounts (0 when fewer than 2 earlier txns) |
# | `secs_since_prev_card_txn` | seconds since the previous transaction on this card |
# | `email_mismatch` | purchaser and recipient email domains both present and different |
# | `P_email_anon` | purchaser email missing or `anonymous.com` |
# | `dt_hour` | hour of day relative to the undisclosed reference (not a known timezone) |
# | `is_large_txn` | amount > $500 |
# | `C_sum` | sum of the C count columns |
# | `is_mobile` | DeviceType is mobile |

# %%
def engineer_features(df):
    """df must be sorted by TransactionDT. Uses only current and earlier rows."""
    df = df.copy()
    amt = df['TransactionAmt']
    df['amt_log'] = np.log1p(amt)

    card = df['card1'].fillna(-1)
    g = amt.groupby(card)
    n_prior = g.cumcount()
    sum_prior = g.cumsum() - amt
    sq_prior = (amt ** 2).groupby(card).cumsum() - amt ** 2
    mean_prior = sum_prior / n_prior.replace(0, np.nan)
    var_prior = (sq_prior - n_prior * mean_prior ** 2) / (n_prior - 1).replace(0, np.nan)
    std_prior = np.sqrt(var_prior.clip(lower=0))
    # Velocity over a fixed 7-day window (an all-time count would grow with time and drift by construction)
    week = 7 * 86400
    df['card1_txn_count_7d'] = df['TransactionDT'].groupby(card).transform(
        lambda t: np.arange(len(t)) - np.searchsorted(t.values, t.values - week, side='left'))
    df['card1_prior_amt_mean'] = mean_prior
    df['amt_zscore'] = ((amt - mean_prior) / std_prior.replace(0, np.nan)).fillna(0).clip(-20, 20)
    df['secs_since_prev_card_txn'] = df['TransactionDT'].groupby(card).diff()

    p, r = df['P_emaildomain'], df['R_emaildomain']
    df['email_mismatch'] = (p.notna() & r.notna() & (p != r)).astype(int)
    df['P_email_anon'] = (p.isna() | (p == 'anonymous.com')).astype(int)

    df['dt_hour'] = (df['TransactionDT'] // 3600) % 24
    df['is_large_txn'] = (amt > 500).astype(int)
    c_cols = [c for c in df.columns if c.startswith('C') and c[1:].isdigit()]
    df['C_sum'] = df[c_cols].fillna(0).sum(axis=1)
    dev = df['DeviceType'] if 'DeviceType' in df.columns else pd.Series('', index=df.index)
    df['is_mobile'] = (dev.fillna('') == 'mobile').astype(int)
    return df

stream = pd.concat([train.assign(_src='train'), test.assign(_src='test')], ignore_index=True)
stream = engineer_features(stream.sort_values('TransactionDT', kind='stable').reset_index(drop=True))
train = stream[stream['_src'] == 'train'].drop(columns='_src').reset_index(drop=True)
test  = stream[stream['_src'] == 'test'].drop(columns=['_src', 'isFraud', 'period'], errors='ignore').reset_index(drop=True)
fit_mask = train['period'] == 'fit'

NEW_FEATURES = ['amt_log', 'card1_txn_count_7d', 'card1_prior_amt_mean', 'amt_zscore',
                'secs_since_prev_card_txn', 'email_mismatch', 'P_email_anon', 'dt_hour',
                'is_large_txn', 'C_sum', 'is_mobile']
print(train.loc[fit_mask].groupby('isFraud')[NEW_FEATURES].mean().T.rename(columns={0: 'legit', 1: 'fraud'}))

# %%
# Keep human-readable values for the analyst console before encoding
DISPLAY_COLS = [c for c in ('ProductCD', 'card4', 'card6', 'P_emaildomain', 'DeviceType') if c in train.columns]
train_display = train[DISPLAY_COLS].copy()

# %% [markdown]
# ## Encode categoricals
# Frequency encoding with counts learned on the **fit period only**; unseen categories get 0.

# %%
cat_cols = [c for c in train.select_dtypes(include='object').columns if c not in ('period',) and c in test.columns]
for col in cat_cols:
    freq = train.loc[fit_mask, col].value_counts()
    train[col] = train[col].map(freq).fillna(0)
    test[col]  = test[col].map(freq).fillna(0)
print(f"Frequency-encoded {len(cat_cols)} categorical columns")

# %% [markdown]
# ## Final feature set
# `TransactionDT` is excluded: it is a clock, so the model would learn time trends that do not
# carry into the future. Raw `card1` stays (it is an identifier the trees can bucket) but its
# history comes only through the point-in-time features above.

# %%
DROP = ['TransactionID', 'isFraud', 'period', 'TransactionDT']
FEATURES = [c for c in train.columns if c not in DROP and c in test.columns]

X_train = train[FEATURES].fillna(-999)
y_train = train['isFraud'].astype(int)
X_test  = test[FEATURES].fillna(-999)

X_train.to_parquet(OUT_DIR + 'X_train.parquet', index=False)
y_train.to_frame().to_parquet(OUT_DIR + 'y_train.parquet', index=False)
X_test.to_parquet(OUT_DIR + 'X_test.parquet', index=False)
pd.concat([train[['TransactionID', 'TransactionDT', 'period']], train_display], axis=1) \
    .to_parquet(OUT_DIR + 'train_meta.parquet', index=False)
test[['TransactionID', 'TransactionDT']].to_parquet(OUT_DIR + 'test_meta.parquet', index=False)

with open(OUT_DIR + 'features.json', 'w') as f:
    json.dump(FEATURES, f)
with open(OUT_DIR + 'eda_stats.json', 'w') as f:
    json.dump({'total_transactions': int(n), 'fraud_count': int(y_train.sum()),
               'fraud_rate_pct': round(float(fraud_rate) * 100, 2), 'feature_count': len(FEATURES),
               'periods': {p: int((train['period'] == p).sum()) for p in ('fit', 'valid', 'oot')}}, f, indent=2)

print(f"Features: {len(FEATURES)} | X_train {X_train.shape} | X_test {X_test.shape}")
print("Notebook 1 complete — run 02_model_training_validation.py next")

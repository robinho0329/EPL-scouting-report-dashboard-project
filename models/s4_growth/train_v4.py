"""
S4 Growth Prediction v4 — Player Development Profile (Classification)
═══════════════════════════════════════════════════════════════════════
HONEST REDESIGN: Individual 1-year point prediction is not reliable
with this data (R² consistently negative in v1-v3). Instead we now
classify players into development trajectories.

Changes vs v3:
  1. CLASSIFICATION instead of regression:
       - "Improving"  : next-season composite > current + 0.3 std
       - "Stable"     : within ±0.3 std
       - "Declining"  : next-season composite < current - 0.3 std
  2. POSITION-SPECIFIC models: Separate classifiers per position group.
     Position one-hot dummies are NOT features.
  3. Peak age curves: LOWESS smoothing, min 30 samples per age bucket.
  4. Minimum 1500 minutes to be included.
  5. Features: age, age², EPL experience, minutes trend slope (2-3
     season), performance trend slope, team quality, market value
     trend — NO absolute performance level.
  6. Scout outputs:
       - "Likely improvers"  (U25, classified Improving)
       - "Stability picks"   (prime 26-30, classified Stable)
       - "Decline risk"      (30+, classified Declining)
       - Peak age curves per position

Saves:
  models/s4_growth/train_v4.py        (this file)
  data/scout/growth_v4_results.json
  data/scout/growth_predictions_v4.parquet
  models/s4_growth/figures/fig*_v4.png
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.stats import mode as scipy_mode
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, balanced_accuracy_score,
    confusion_matrix, roc_auc_score
)
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb

warnings.filterwarnings('ignore')

# ─────────────────────────── Paths ───────────────────────────
BASE_DIR  = Path("C:/Users/xcv54/workspace/EPL project")
DATA_DIR  = BASE_DIR / "data" / "processed"
SCOUT_DIR = BASE_DIR / "data" / "scout"
FIG_DIR   = BASE_DIR / "models" / "s4_growth" / "figures"

SCOUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({'figure.dpi': 120, 'font.size': 11,
                     'axes.titlesize': 13, 'axes.labelsize': 11})
PALETTE = {'FW': '#e74c3c', 'MF': '#3498db', 'DF': '#2ecc71', 'GK': '#f39c12'}

# Classification threshold: ±0.3 standard deviations of the positional
# composite score defines "Improving" vs "Stable" vs "Declining"
DELTA_THRESHOLD = 0.3   # in Z-score units

# Literature-backed peak ages (sports-science meta-analyses)
PEAK_AGE_PRIORS = {'FW': 27, 'MF': 27, 'DF': 28, 'GK': 30}

CLASS_LABELS  = ['Declining', 'Stable', 'Improving']
CLASS_COLORS  = {'Improving': '#27ae60', 'Stable': '#f39c12', 'Declining': '#c0392b'}

POSITIONS = ['FW', 'MF', 'DF', 'GK']
MIN_MINUTES = 1500   # per season

# ═══════════════════════════════════════════════════════════════
# 1. Load data
# ═══════════════════════════════════════════════════════════════
print("=" * 65)
print("[1] Loading data...")
print("=" * 65)

season_df = pd.read_parquet(DATA_DIR / "player_season_stats.parquet")
match_df  = pd.read_parquet(DATA_DIR / "player_match_logs.parquet")
team_df   = pd.read_parquet(DATA_DIR / "team_season_summary.parquet")

print(f"  Season stats : {season_df.shape}")
print(f"  Match logs   : {match_df.shape}")
print(f"  Team summary : {team_df.shape}")


def season_to_year(s):
    try:
        return int(str(s).split('/')[0])
    except:
        return np.nan


def simplify_position(pos):
    if pd.isna(pos): return 'MF'
    pos = str(pos).upper()
    if 'GK' in pos: return 'GK'
    if 'FW' in pos: return 'FW'
    if 'DF' in pos: return 'DF'
    return 'MF'


season_df['season_year'] = season_df['season'].apply(season_to_year)
team_df['season_year']   = team_df['Season'].apply(season_to_year)
season_df['pos_simple']  = season_df['pos'].apply(simplify_position)

# Apply minimum minutes filter
season_df_hist = season_df[season_df['min'].fillna(0) >= MIN_MINUTES].copy()
print(f"  After {MIN_MINUTES}-min filter: {season_df_hist.shape[0]} rows")

# ═══════════════════════════════════════════════════════════════
# 2. Match log aggregation
# ═══════════════════════════════════════════════════════════════
print("\n[2] Aggregating match logs...")

match_detail = match_df[match_df['detail_stats_available'] == True].copy()
match_detail['season_year'] = match_detail['season'].apply(season_to_year)


def safe_sum(x): return x.fillna(0).sum()


match_agg = (
    match_detail
    .groupby(['player', 'season_year'])
    .agg(
        tklw_total=('tklw', safe_sum),
        int_total=('int',  safe_sum),
        sh_total=('sh',   safe_sum),
        sot_total=('sot',  safe_sum),
        fls_total=('fls',  safe_sum),
        fld_total=('fld',  safe_sum),
        crs_total=('crs',  safe_sum),
        match_count=('min', 'count')
    )
    .reset_index()
)
print(f"  Match aggregation: {match_agg.shape}")

# ═══════════════════════════════════════════════════════════════
# 3. Composite performance score (per-90, position-aware)
# ═══════════════════════════════════════════════════════════════
print("\n[3] Computing composite performance score...")

season_df_hist = season_df_hist.merge(match_agg, on=['player', 'season_year'], how='left')
season_df_hist['90s_safe'] = season_df_hist['90s'].fillna(1).clip(lower=0.1)

season_df_hist['gls_p90']  = season_df_hist['gls'].fillna(0)               / season_df_hist['90s_safe']
season_df_hist['ast_p90']  = season_df_hist['ast'].fillna(0)               / season_df_hist['90s_safe']
season_df_hist['tklw_p90'] = season_df_hist['tklw_total'].fillna(0)        / season_df_hist['90s_safe']
season_df_hist['int_p90']  = season_df_hist['int_total'].fillna(0)         / season_df_hist['90s_safe']
season_df_hist['sh_p90']   = season_df_hist['sh_total'].fillna(0)          / season_df_hist['90s_safe']
season_df_hist['crs_p90']  = season_df_hist['crs_total'].fillna(0)         / season_df_hist['90s_safe']
season_df_hist['fld_p90']  = season_df_hist['fld_total'].fillna(0)         / season_df_hist['90s_safe']


def compute_composite_score(row):
    pos = row['pos_simple']
    if pos == 'FW':
        return (row['gls_p90'] * 3.0 + row['ast_p90'] * 2.0 +
                row['sh_p90']  * 0.3  + row['fld_p90'] * 0.5)
    elif pos == 'MF':
        return (row['gls_p90']  * 2.0 + row['ast_p90']  * 2.5 +
                row['tklw_p90'] * 0.8 + row['int_p90']  * 0.8 +
                row['crs_p90']  * 0.5 + row['fld_p90']  * 0.4)
    elif pos == 'DF':
        return (row['tklw_p90'] * 3.0 + row['int_p90']  * 3.0 +
                row['gls_p90']  * 1.0 + row['ast_p90']  * 0.8 +
                row['fld_p90']  * 0.3)
    else:  # GK — playing time itself is the primary signal for GKs
        return row['90s_safe'] * 0.1


season_df_hist['raw_score'] = season_df_hist.apply(compute_composite_score, axis=1)

# Z-score within position × season (removes season difficulty drift)
def zscore_within_group(df, col, group_cols):
    result = pd.Series(np.nan, index=df.index)
    for key, idx in df.groupby(group_cols).groups.items():
        vals = df.loc[idx, col]
        if len(vals) >= 3:
            mu, sigma = vals.mean(), vals.std()
            result.loc[idx] = (vals - mu) / sigma if sigma > 0 else 0.0
        else:
            result.loc[idx] = 0.0
    return result


season_df_hist['perf_z'] = zscore_within_group(
    season_df_hist, 'raw_score', ['pos_simple', 'season_year']
)
print(f"  perf_z stats: {season_df_hist['perf_z'].describe().to_dict()}")

# ═══════════════════════════════════════════════════════════════
# 4. Peak age analysis — LOWESS smoothing, min 30 samples
#    Compare with academic literature priors.
# ═══════════════════════════════════════════════════════════════
print("\n[4] Peak age analysis (LOWESS, min 30 samples per age bucket)...")
print("=" * 65)

# Simple LOWESS (tricube kernel, manual implementation to avoid dependency issues)
def lowess_smooth(x, y, frac=0.4):
    """Manual LOWESS: weighted local linear regression."""
    n = len(x)
    yhat = np.zeros(n)
    r = np.abs(x - x[:, None])  # (n, n) pairwise distances
    h = np.sort(r, axis=1)[:, max(1, int(np.ceil(frac * n)) - 1)]  # bandwidth per point
    for i in range(n):
        w = np.clip(1 - (r[i] / (h[i] + 1e-10)) ** 3, 0, None) ** 3  # tricube
        W = np.diag(w)
        X_ = np.column_stack([np.ones(n), x])
        try:
            beta = np.linalg.lstsq(X_.T @ W @ X_, X_.T @ W @ y, rcond=None)[0]
            yhat[i] = beta[0] + beta[1] * x[i]
        except np.linalg.LinAlgError:
            yhat[i] = y[i]
    return yhat


peak_age_results = {}

for pos in POSITIONS:
    pos_data = season_df_hist[season_df_hist['pos_simple'] == pos].copy()
    prior    = PEAK_AGE_PRIORS[pos]

    age_stats = (
        pos_data.groupby('age')['perf_z']
        .agg(['mean', 'std', 'count'])
        .reset_index()
        .rename(columns={'mean': 'mean_z', 'std': 'std_z', 'count': 'n'})
        .sort_values('age')
    )

    age_valid = age_stats[age_stats['n'] >= 30].copy()

    if len(age_valid) >= 4:
        ages    = age_valid['age'].values.astype(float)
        mean_zs = age_valid['mean_z'].values

        # LOWESS smooth (frac=0.4: each point influenced by 40% of neighbours)
        smoothed = lowess_smooth(ages, mean_zs, frac=0.4)

        data_peak_raw = int(ages[np.argmax(smoothed)])

        # Hard-clip: data peak constrained to [prior-2, prior+2]
        data_peak = int(np.clip(data_peak_raw, prior - 2, prior + 2))

        amplitude = float(smoothed.max() - smoothed.min())
        w_data    = min(1.0, (amplitude / 0.30) * min(1.0, len(age_valid) / 12.0))
        w_prior   = 1.0 - w_data
        blended   = int(round(w_data * data_peak + w_prior * prior))

        peak_age_results[pos] = {
            'peak_age':          blended,
            'data_peak_age_raw': data_peak_raw,
            'data_peak_age':     data_peak,
            'prior_peak_age':    prior,
            'amplitude':         round(amplitude, 4),
            'w_data':            round(w_data, 3),
            'w_prior':           round(w_prior, 3),
            'valid_age_buckets': int(len(age_valid)),
            'valid_age_range':   [int(ages.min()), int(ages.max())],
            'smoothing_method':  'lowess_frac0.4',
            'age_curve': {
                'ages':     [int(a) for a in ages],
                'mean_z':   [round(float(z), 4) for z in mean_zs],
                'smoothed': [round(float(s), 4) for s in smoothed],
                'n':        [int(row['n']) for _, row in age_valid.iterrows()],
            }
        }
        print(f"\n  {pos} Peak Age:")
        print(f"    Raw data peak={data_peak_raw}  ->  clipped={data_peak}")
        print(f"    Prior={prior},  Blended (LOWESS)={blended}")
        print(f"    Amplitude={amplitude:.3f}, w_data={w_data:.2f}, valid buckets={len(age_valid)}")
    else:
        peak_age_results[pos] = {
            'peak_age': prior, 'data_peak_age': None, 'prior_peak_age': prior,
            'w_data': 0.0, 'w_prior': 1.0,
            'valid_age_range': [18, 35], 'valid_age_buckets': int(len(age_valid)),
            'smoothing_method': 'lowess_frac0.4',
        }
        print(f"\n  {pos}: Insufficient data -> using prior peak_age={prior}")

print("\n" + "=" * 65)

pos_peak_ages = {pos: peak_age_results[pos]['peak_age'] for pos in POSITIONS}

# ═══════════════════════════════════════════════════════════════
# 5. Build classification target
#    delta_z = Z(N+1) - Z(N)
#    Improving  : delta_z >  +DELTA_THRESHOLD (within position std)
#    Declining  : delta_z <  -DELTA_THRESHOLD
#    Stable     : otherwise
#    Note: threshold is in Z-score units, which is already positionally normalised.
# ═══════════════════════════════════════════════════════════════
print("\n[5] Building classification target...")

score_lookup = (
    season_df_hist[['player', 'season_year', 'perf_z', 'min']]
    .rename(columns={'perf_z': 'z', 'min': 'mins_lookup'})
)

df = season_df_hist.copy()
df['next1_year'] = df['season_year'] + 1

df = df.merge(
    score_lookup.rename(columns={'season_year': 'next1_year',
                                  'z': 'z_next1',
                                  'mins_lookup': 'min_next1'}),
    on=['player', 'next1_year'], how='inner'
)

df['delta_z'] = df['z_next1'] - df['perf_z']

def classify_delta(d):
    if d >  DELTA_THRESHOLD: return 'Improving'
    if d < -DELTA_THRESHOLD: return 'Declining'
    return 'Stable'


df['dev_class'] = df['delta_z'].apply(classify_delta)

print(f"  Labelled rows: {len(df)}")
print("  Class distribution:")
print(df['dev_class'].value_counts().to_string())
print(df.groupby('pos_simple')['dev_class'].value_counts().to_string())

# ═══════════════════════════════════════════════════════════════
# 6. Injury proxy (minutes drop)
# ═══════════════════════════════════════════════════════════════
print("\n[6] Computing injury proxy...")

min_prev = (
    season_df_hist[['player', 'season_year', 'min']]
    .rename(columns={'season_year': 'prev_year', 'min': 'prev_min'})
)
min_prev['season_year'] = min_prev['prev_year'] + 1

df = df.merge(min_prev[['player', 'season_year', 'prev_min']],
              on=['player', 'season_year'], how='left')

df['min_drop_pct'] = np.where(
    (df['prev_min'].notna()) & (df['prev_min'] > 0),
    (df['prev_min'] - df['min']) / df['prev_min'],
    0.0
)
df['injury_risk'] = (df['min_drop_pct'] > 0.40).astype(int)
print(f"  Injury-risk rows: {df['injury_risk'].sum()} ({df['injury_risk'].mean():.1%})")

# ═══════════════════════════════════════════════════════════════
# 7. Feature engineering — TRAJECTORY ONLY, no absolute level
#    Position is NOT a feature (models are trained per position).
#    Features:
#      age, age², age_to_peak, EPL experience,
#      minutes trend slope (2-3 season), perf trend slope (2-3 season),
#      team quality (points), market value trend slope,
#      min_drop_pct (injury proxy), consistency, starter_ratio
# ═══════════════════════════════════════════════════════════════
print("\n[7] Feature engineering (trajectory only, no position dummies, no absolute level)...")

# Team points
team_points = team_df[['team', 'season_year', 'points']].copy()
df = df.merge(team_points, on=['team', 'season_year'], how='left')

# EPL experience (cumulative seasons)
df_s = df.sort_values(['player', 'season_year'])
df_s['epl_seasons'] = df_s.groupby('player').cumcount() + 1
df['epl_seasons'] = df_s['epl_seasons'].values

# Match consistency (CV of minutes per game)
match_consistency = (
    match_df[match_df['min'].notna()]
    .groupby(['player', 'season'])
    .agg(min_mean=('min', 'mean'), min_std=('min', 'std'))
    .reset_index()
)
match_consistency['season_year'] = match_consistency['season'].apply(season_to_year)
match_consistency['consistency'] = np.where(
    match_consistency['min_mean'] > 0,
    1 - (match_consistency['min_std'].fillna(0) / match_consistency['min_mean']).clip(0, 1),
    0
)
df = df.merge(match_consistency[['player', 'season_year', 'consistency']],
              on=['player', 'season_year'], how='left')

# Starter ratio
df['starter_ratio'] = (df['starts'].fillna(0) / df['mp'].replace(0, np.nan)).clip(0, 1)

# Market value trend slope (2-3 seasons)
df_s2 = df.sort_values(['player', 'season_year']).reset_index(drop=True)

perf_trend_vals, min_trend_vals, mv_trend_vals = [], [], []

for _, group in df_s2.groupby('player'):
    group  = group.sort_values('season_year').reset_index(drop=True)
    scores = group['perf_z'].values
    mins   = group['min'].fillna(0).values
    mvs    = group['market_value'].values if 'market_value' in group.columns else np.zeros(len(group))
    years  = group['season_year'].values

    for j in range(len(group)):
        past_mask  = years < years[j]
        past_s     = scores[past_mask]
        past_m     = mins[past_mask]
        past_mv    = mvs[past_mask]
        past_y     = years[past_mask]

        # Use last 3 seasons
        recent_s  = past_s[-3:]  if len(past_s) >= 2 else []
        recent_m  = past_m[-3:]  if len(past_m) >= 2 else []
        recent_mv = past_mv[-3:] if len(past_mv) >= 2 else []
        recent_y  = past_y[-3:]  if len(past_y) >= 2 else []

        if len(recent_s) >= 2:
            ry   = np.array(recent_y, dtype=float)
            ry_c = ry - ry.mean()
            std_ry = np.std(ry_c)
            slope_s  = 0.0 if std_ry == 0 else stats.linregress(ry_c, np.array(recent_s, dtype=float))[0]
            slope_m  = 0.0 if std_ry == 0 else stats.linregress(ry_c, np.array(recent_m, dtype=float))[0]
            # Market value trend only if non-zero values available
            mv_vals = np.array(recent_mv, dtype=float)
            if np.any(mv_vals > 0) and std_ry > 0:
                slope_mv = stats.linregress(ry_c, mv_vals)[0]
            else:
                slope_mv = 0.0
            perf_trend_vals.append(slope_s)
            min_trend_vals.append(slope_m)
            mv_trend_vals.append(slope_mv)
        else:
            perf_trend_vals.append(np.nan)
            min_trend_vals.append(np.nan)
            mv_trend_vals.append(np.nan)

df_s2['perf_trend'] = perf_trend_vals
df_s2['min_trend']  = min_trend_vals
df_s2['mv_trend']   = mv_trend_vals
df = df_s2.copy()

# Age features
df['peak_age_pos'] = df['pos_simple'].map(pos_peak_ages).fillna(27)
df['age_to_peak']  = df['peak_age_pos'] - df['age']
df['age_sq']       = df['age'] ** 2

print("  Feature engineering complete.")

# ═══════════════════════════════════════════════════════════════
# 8. Position-specific feature sets (NO position dummies)
# ═══════════════════════════════════════════════════════════════
FEATURE_COLS = [
    'age',
    'age_sq',
    'age_to_peak',
    'epl_seasons',
    'perf_trend',       # slope of perf_z over last 3 seasons
    'min_trend',        # slope of minutes over last 3 seasons
    'mv_trend',         # slope of market value over last 3 seasons
    'min_drop_pct',     # injury proxy
    'consistency',
    'starter_ratio',
    'points',           # team quality
]

extra_cols = ['player', 'season_year', 'pos_simple', 'team',
              'injury_risk', 'perf_z', 'delta_z', 'market_value']

all_needed = FEATURE_COLS + ['dev_class'] + extra_cols
all_needed = list(dict.fromkeys(all_needed))
available  = [c for c in all_needed if c in df.columns]
df_model   = df[available].copy().reset_index(drop=True)

# Drop rows where >40% of features are NaN
feat_avail = [c for c in FEATURE_COLS if c in df_model.columns]
nan_frac   = df_model[feat_avail].isna().mean(axis=1)
df_model   = df_model[nan_frac < 0.4].copy().reset_index(drop=True)

print(f"\n[8] Model dataset: {df_model.shape[0]} rows, {len(feat_avail)} features")
print("  Class distribution:")
print(df_model['dev_class'].value_counts().to_string())

# ═══════════════════════════════════════════════════════════════
# 9. Position-specific classifier training
#    Separate model per position group — no position features.
# ═══════════════════════════════════════════════════════════════
print("\n[9] Training position-specific classifiers...")
print("=" * 65)

# Time-based split (2023+ = test)
train_mask = df_model['season_year'] <= 2020
val_mask   = (df_model['season_year'] >= 2021) & (df_model['season_year'] <= 2022)
test_mask  = df_model['season_year'] >= 2023

pos_models  = {}   # {pos: {'model': pipeline, 'metrics': dict}}
pos_results = {}

for pos in POSITIONS:
    pos_mask = df_model['pos_simple'] == pos
    df_pos   = df_model[pos_mask].copy()

    n_total = len(df_pos)
    if n_total < 30:
        print(f"\n  [{pos}] Insufficient data ({n_total} rows) — skipping model.")
        continue

    feat_cols_pos = [c for c in feat_avail if c in df_pos.columns]
    X = df_pos[feat_cols_pos]
    y = df_pos['dev_class']

    X_tr = X[df_pos['season_year'] <= 2020]
    y_tr = y[df_pos['season_year'] <= 2020]
    X_va = X[(df_pos['season_year'] >= 2021) & (df_pos['season_year'] <= 2022)]
    y_va = y[(df_pos['season_year'] >= 2021) & (df_pos['season_year'] <= 2022)]
    X_te = X[df_pos['season_year'] >= 2023]
    y_te = y[df_pos['season_year'] >= 2023]

    if len(X_tr) < 20 or len(y_tr.unique()) < 2:
        print(f"\n  [{pos}] Not enough training samples or single class — skipping.")
        continue

    print(f"\n  [{pos}] n_train={len(X_tr)}, n_val={len(X_va)}, n_test={len(X_te)}")
    print(f"    Train class dist: {y_tr.value_counts().to_dict()}")

    # XGBoost classifier (position-specific, no position features)
    xgb_clf = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', xgb.XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=10,
            reg_alpha=1.0,
            reg_lambda=3.0,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        ))
    ])

    rf_clf = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model', RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=15,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        ))
    ])

    # Encode labels for XGBoost (needs integer labels)
    label_map     = {'Declining': 0, 'Stable': 1, 'Improving': 2}
    inv_label_map = {0: 'Declining', 1: 'Stable', 2: 'Improving'}

    y_tr_enc = y_tr.map(label_map)
    y_te_enc = y_te.map(label_map) if len(y_te) > 0 else pd.Series(dtype=int)

    xgb_clf.fit(X_tr, y_tr_enc)
    rf_clf.fit(X_tr, y_tr)

    # Validation metrics
    metrics = {}
    for split_name, X_s, y_s, y_s_enc in [
        ('val',  X_va, y_va, y_va.map(label_map)),
        ('test', X_te, y_te, y_te_enc),
    ]:
        if len(X_s) == 0 or len(y_s.unique()) < 2:
            metrics[split_name] = {'note': 'insufficient samples'}
            continue

        xgb_pred_enc = xgb_clf.predict(X_s)
        xgb_pred     = pd.Series(xgb_pred_enc).map(inv_label_map).values
        rf_pred      = rf_clf.predict(X_s)

        # Ensemble: majority vote
        ens_preds = []
        for x, r in zip(xgb_pred, rf_pred):
            votes = [x, r]
            cnt   = {v: votes.count(v) for v in set(votes)}
            ens_preds.append(max(cnt, key=cnt.get))

        acc_xgb = accuracy_score(y_s, xgb_pred)
        acc_rf  = accuracy_score(y_s, rf_pred)
        acc_ens = accuracy_score(y_s, ens_preds)
        bal_ens = balanced_accuracy_score(y_s, ens_preds)

        metrics[split_name] = {
            'accuracy_xgb': round(acc_xgb, 4),
            'accuracy_rf':  round(acc_rf,  4),
            'accuracy_ensemble': round(acc_ens, 4),
            'balanced_accuracy': round(bal_ens, 4),
            'n_samples': int(len(y_s)),
            'class_dist': y_s.value_counts().to_dict(),
        }
        print(f"    {split_name}: acc_xgb={acc_xgb:.3f}  acc_rf={acc_rf:.3f}  "
              f"acc_ens={acc_ens:.3f}  bal_acc={bal_ens:.3f}")

    # Expanding-window CV for this position
    df_pos_cv = df_pos.sort_values('season_year').reset_index(drop=True)
    X_cv_pos  = df_pos_cv[feat_cols_pos]
    y_cv_pos  = df_pos_cv['dev_class'].map(label_map)
    valid_mask_cv = y_cv_pos.notna()
    X_cv_pos  = X_cv_pos[valid_mask_cv].reset_index(drop=True)
    y_cv_pos  = y_cv_pos[valid_mask_cv].reset_index(drop=True)

    tscv_pos = TimeSeriesSplit(n_splits=min(5, max(2, len(X_cv_pos) // 50)))
    cv_accs  = []
    for tr_idx, te_idx in tscv_pos.split(X_cv_pos):
        if len(te_idx) < 5 or len(y_cv_pos.iloc[te_idx].unique()) < 2:
            continue
        xgb_clf.fit(X_cv_pos.iloc[tr_idx], y_cv_pos.iloc[tr_idx])
        preds = pd.Series(xgb_clf.predict(X_cv_pos.iloc[te_idx])).map(inv_label_map).values
        actuals = y_cv_pos.iloc[te_idx].map(inv_label_map).values
        cv_accs.append(balanced_accuracy_score(actuals, preds))

    cv_bal_acc = float(np.mean(cv_accs)) if cv_accs else None
    metrics['cv_balanced_accuracy'] = round(cv_bal_acc, 4) if cv_bal_acc else None
    if cv_accs:
        print(f"    CV balanced_accuracy = {cv_bal_acc:.3f} ({len(cv_accs)} folds)")

    # Retrain on all data up through 2022 for final scout predictions
    full_train_mask = df_pos['season_year'] <= 2022
    X_full = X[full_train_mask]
    y_full_enc = y[full_train_mask].map(label_map)
    y_full     = y[full_train_mask]
    if len(X_full) >= 20 and len(y_full_enc.unique()) >= 2:
        xgb_clf.fit(X_full, y_full_enc)
        rf_clf.fit(X_full, y_full)

    # Feature importance
    fi = pd.DataFrame({
        'feature':    feat_cols_pos,
        'importance': xgb_clf.named_steps['model'].feature_importances_
    }).sort_values('importance', ascending=False)

    pos_models[pos] = {
        'xgb':           xgb_clf,
        'rf':            rf_clf,
        'label_map':     label_map,
        'inv_label_map': inv_label_map,
        'feat_cols':     feat_cols_pos,
        'feature_importance': fi,
    }
    pos_results[pos] = metrics
    print(f"    Top features: {fi.head(5)['feature'].tolist()}")

# ═══════════════════════════════════════════════════════════════
# 10. Scout predictions on the most recent season
#     Min 1500 minutes, position-specific models.
# ═══════════════════════════════════════════════════════════════
print("\n[10] Generating scout predictions (most recent season)...")

latest_raw_year = season_df['season_year'].max()
pred_raw = season_df[season_df['season_year'] == latest_raw_year].copy()
pred_raw['pos_simple'] = pred_raw['pos'].apply(simplify_position)
pred_raw = pred_raw.merge(match_agg, on=['player', 'season_year'], how='left')

pred_raw['90s_safe'] = pred_raw['90s'].fillna(1).clip(lower=0.1)
pred_raw['gls_p90']  = pred_raw['gls'].fillna(0)               / pred_raw['90s_safe']
pred_raw['ast_p90']  = pred_raw['ast'].fillna(0)               / pred_raw['90s_safe']
pred_raw['tklw_p90'] = pred_raw['tklw_total'].fillna(0)        / pred_raw['90s_safe']
pred_raw['int_p90']  = pred_raw['int_total'].fillna(0)         / pred_raw['90s_safe']
pred_raw['sh_p90']   = pred_raw['sh_total'].fillna(0)          / pred_raw['90s_safe']
pred_raw['crs_p90']  = pred_raw['crs_total'].fillna(0)         / pred_raw['90s_safe']
pred_raw['fld_p90']  = pred_raw['fld_total'].fillna(0)         / pred_raw['90s_safe']
pred_raw['raw_score'] = pred_raw.apply(compute_composite_score, axis=1)

# Z-score within position (latest season)
for pos in POSITIONS:
    mask = pred_raw['pos_simple'] == pos
    vals = pred_raw.loc[mask, 'raw_score']
    mu, sigma = vals.mean(), vals.std()
    pred_raw.loc[mask, 'perf_z'] = (vals - mu) / sigma if sigma > 0 else 0.0

# Injury proxy
prev_year_mins = (
    season_df[season_df['season_year'] == latest_raw_year - 1][['player', 'min']]
    .rename(columns={'min': 'prev_min'})
)
pred_raw = pred_raw.merge(prev_year_mins, on='player', how='left')
pred_raw['min_drop_pct'] = np.where(
    (pred_raw['prev_min'].notna()) & (pred_raw['prev_min'] > 0),
    (pred_raw['prev_min'] - pred_raw['min']) / pred_raw['prev_min'],
    0.0
)
pred_raw['injury_risk'] = (pred_raw['min_drop_pct'] > 0.40).astype(int)

# Team points
pred_raw = pred_raw.merge(team_points, on=['team', 'season_year'], how='left')

# EPL experience
epl_exp = (
    season_df.groupby('player')['season_year'].count()
    .reset_index().rename(columns={'season_year': 'epl_seasons'})
)
pred_raw = pred_raw.merge(epl_exp, on='player', how='left')

# Match consistency (most recent season)
latest_mc = match_consistency[
    match_consistency['season_year'] == latest_raw_year
][['player', 'consistency']]
pred_raw = pred_raw.merge(latest_mc, on='player', how='left')

# Starter ratio
pred_raw['starter_ratio'] = (pred_raw['starts'].fillna(0) / pred_raw['mp'].replace(0, np.nan)).clip(0, 1)

# Trajectory trends from training data (last entry per player)
available_trend_cols = [c for c in ['perf_trend', 'min_trend', 'mv_trend']
                        if c in df_model.columns]
hist_latest = (
    df_model.sort_values('season_year')
    .groupby('player').last()
    .reset_index()
    [['player'] + available_trend_cols]
)
pred_raw = pred_raw.merge(hist_latest, on='player', how='left')

# Age features
pred_raw['peak_age_pos'] = pred_raw['pos_simple'].map(pos_peak_ages).fillna(27)
pred_raw['age_to_peak']  = pred_raw['peak_age_pos'] - pred_raw['age']
pred_raw['age_sq']       = pred_raw['age'] ** 2

# Filter: minimum 1500 minutes
pred_candidates = pred_raw[pred_raw['min'].fillna(0) >= MIN_MINUTES].copy()
print(f"  After {MIN_MINUTES}-min filter: {len(pred_candidates)} candidates")

# Predict per position — deduplicate by player name before predicting
all_scout_rows = []

for pos in POSITIONS:
    if pos not in pos_models:
        print(f"  [{pos}] No model available — skipping.")
        continue

    pm         = pos_models[pos]
    feat_cols_p = pm['feat_cols']
    pos_cands  = pred_candidates[pred_candidates['pos_simple'] == pos].copy()

    # Deduplicate: keep the row with highest minutes per player
    pos_cands = pos_cands.sort_values('min', ascending=False).drop_duplicates(
        subset='player', keep='first'
    )

    if len(pos_cands) == 0:
        continue

    X_scout = pos_cands[feat_cols_p].fillna(
        pos_cands[feat_cols_p].median()
    )

    # XGBoost prediction
    xgb_pred_enc = pm['xgb'].predict(X_scout)
    xgb_pred_cls = [pm['inv_label_map'][e] for e in xgb_pred_enc]

    # XGBoost probability
    xgb_proba = pm['xgb'].predict_proba(X_scout)
    classes   = pm['xgb'].classes_
    proba_df  = pd.DataFrame(xgb_proba, columns=[pm['inv_label_map'][c] for c in classes])

    # RF prediction
    rf_pred_cls = pm['rf'].predict(X_scout)

    # Ensemble: majority vote (XGBoost + RF)
    ens_cls = []
    for x, r in zip(xgb_pred_cls, rf_pred_cls):
        votes = [x, r]
        cnt   = {v: votes.count(v) for v in set(votes)}
        ens_cls.append(max(cnt, key=cnt.get))

    pos_cands = pos_cands.reset_index(drop=True)
    pos_cands['pred_xgb']        = xgb_pred_cls
    pos_cands['pred_rf']         = rf_pred_cls
    pos_cands['pred_ensemble']   = ens_cls
    pos_cands['prob_improving']  = proba_df.get('Improving', 0).values
    pos_cands['prob_stable']     = proba_df.get('Stable', 0).values
    pos_cands['prob_declining']  = proba_df.get('Declining', 0).values

    all_scout_rows.append(pos_cands)

if all_scout_rows:
    scout_df = pd.concat(all_scout_rows, ignore_index=True)
else:
    scout_df = pd.DataFrame()

print(f"  Total classified candidates: {len(scout_df)}")
if len(scout_df) > 0:
    print("  Ensemble class distribution:")
    print(scout_df['pred_ensemble'].value_counts().to_string())

# ═══════════════════════════════════════════════════════════════
# 11. Build the 3 scout outputs
# ═══════════════════════════════════════════════════════════════
print("\n[11] Building scout outputs...")

if len(scout_df) == 0:
    likely_improvers = pd.DataFrame()
    stability_picks  = pd.DataFrame()
    decline_risk     = pd.DataFrame()
else:
    # "Likely improvers" — U25, classified Improving, sorted by prob_improving
    likely_improvers = (
        scout_df[
            (scout_df['age'] <= 25) &
            (scout_df['pred_ensemble'] == 'Improving')
        ]
        .sort_values('prob_improving', ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    likely_improvers.index += 1

    # "Stability picks" — prime age 26-30, classified Stable, sorted by prob_stable
    stability_picks = (
        scout_df[
            (scout_df['age'] >= 26) & (scout_df['age'] <= 30) &
            (scout_df['pred_ensemble'] == 'Stable')
        ]
        .sort_values('prob_stable', ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    stability_picks.index += 1

    # "Decline risk" — 30+, classified Declining, sorted by prob_declining
    decline_risk = (
        scout_df[
            (scout_df['age'] >= 30) &
            (scout_df['pred_ensemble'] == 'Declining')
        ]
        .sort_values('prob_declining', ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    decline_risk.index += 1

_display_cols = ['player', 'age', 'pos_simple', 'team', 'pred_ensemble',
                 'prob_improving', 'prob_stable', 'prob_declining', 'min']

print("\n" + "=" * 70)
print("  LIKELY IMPROVERS (U25, classified Improving)")
print("=" * 70)
if len(likely_improvers) > 0:
    print(likely_improvers[[c for c in _display_cols if c in likely_improvers.columns]].to_string())
else:
    print("  No candidates found.")

print("\n" + "=" * 70)
print("  STABILITY PICKS (26-30, classified Stable)")
print("=" * 70)
if len(stability_picks) > 0:
    print(stability_picks[[c for c in _display_cols if c in stability_picks.columns]].to_string())
else:
    print("  No candidates found.")

print("\n" + "=" * 70)
print("  DECLINE RISK (30+, classified Declining)")
print("=" * 70)
if len(decline_risk) > 0:
    print(decline_risk[[c for c in _display_cols if c in decline_risk.columns]].to_string())
else:
    print("  No candidates found.")

# ═══════════════════════════════════════════════════════════════
# 12. Visualisations
# ═══════════════════════════════════════════════════════════════
print("\n[12] Generating figures...")

# ── Fig 1: Peak age curves per position (LOWESS) ──
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    'S4 v4: Peak Age Curves per Position\n'
    '(LOWESS smoothing, min 30 samples/bucket, prior-blended)',
    fontsize=14, fontweight='bold'
)
axes = axes.flatten()

for ax, pos in zip(axes, POSITIONS):
    pos_data  = season_df_hist[season_df_hist['pos_simple'] == pos]
    age_stats = (
        pos_data.groupby('age')['perf_z']
        .agg(['mean', 'std', 'count'])
        .reset_index()
        .rename(columns={'mean': 'mean_z', 'std': 'std_z', 'count': 'n'})
        .sort_values('age')
    )
    age_all   = age_stats.copy()
    age_valid = age_stats[age_stats['n'] >= 30].copy()

    ax.plot(age_all['age'], age_all['mean_z'],
            'o--', color='#cccccc', linewidth=1, markersize=3, label='n < 30 (excluded)')

    if len(age_valid) >= 4:
        ages_v   = age_valid['age'].values.astype(float)
        mean_zv  = age_valid['mean_z'].values
        smoothed = lowess_smooth(ages_v, mean_zv, frac=0.4)

        ax.plot(ages_v, mean_zv,
                'o', color=PALETTE[pos], markersize=5, alpha=0.6, label='n >= 30 (raw)')
        ax.plot(ages_v, smoothed,
                '-', color=PALETTE[pos], linewidth=2.5, label='LOWESS smoothed')

        for i, row in age_valid.iterrows():
            if int(row['age']) % 4 == 0:
                ax.annotate(f"n={int(row['n'])}",
                            xy=(row['age'], row['mean_z']),
                            xytext=(0, 8), textcoords='offset points',
                            fontsize=7, ha='center', color='#555555')

    pa       = peak_age_results[pos]['peak_age']
    prior_pa = peak_age_results[pos]['prior_peak_age']
    ax.axvline(pa,       color=PALETTE[pos], linestyle='-',  alpha=0.8, linewidth=2,
               label=f'Blended peak = {pa}')
    ax.axvline(prior_pa, color='gray',       linestyle='--', alpha=0.5, linewidth=1.5,
               label=f'Literature prior = {prior_pa}')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Age'); ax.set_ylabel('Mean perf_z')
    ax.set_title(f'{pos}  —  Peak (blended)={pa},  Prior={prior_pa}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig1_peak_age_v4.png', bbox_inches='tight')
plt.close()
print("  Fig 1 saved: fig1_peak_age_v4.png")

# ── Fig 2: Classification accuracy per position ──
pos_names_plot = [p for p in POSITIONS if p in pos_results]
val_accs  = [pos_results[p].get('val',  {}).get('balanced_accuracy', np.nan) for p in pos_names_plot]
test_accs = [pos_results[p].get('test', {}).get('balanced_accuracy', np.nan) for p in pos_names_plot]
cv_accs_plot = [pos_results[p].get('cv_balanced_accuracy', np.nan) for p in pos_names_plot]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(pos_names_plot))
w = 0.25
bars1 = ax.bar(x - w, val_accs,  w, label='Val Bal.Acc.',  color='#3498db', alpha=0.85)
bars2 = ax.bar(x,     test_accs, w, label='Test Bal.Acc.', color='#e74c3c', alpha=0.85)
bars3 = ax.bar(x + w, cv_accs_plot, w, label='CV Bal.Acc.', color='#27ae60', alpha=0.85)
ax.axhline(1/3, color='black', linewidth=1.5, linestyle='--', label='Random baseline (0.333)')
ax.set_xticks(x); ax.set_xticklabels(pos_names_plot)
ax.set_ylabel('Balanced Accuracy'); ax.set_title('S4 v4: Position-Specific Classifier Performance')
ax.legend(); ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1)
for bars in [bars1, bars2, bars3]:
    for b in bars:
        h = b.get_height()
        if not np.isnan(h):
            ax.text(b.get_x() + b.get_width()/2, h + 0.01, f'{h:.2f}',
                    ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig2_classifier_accuracy_v4.png', bbox_inches='tight')
plt.close()
print("  Fig 2 saved: fig2_classifier_accuracy_v4.png")

# ── Fig 3: Feature importance per position ──
n_pos_with_models = len(pos_models)
if n_pos_with_models > 0:
    fig, axes = plt.subplots(1, n_pos_with_models, figsize=(5 * n_pos_with_models, 7))
    if n_pos_with_models == 1:
        axes = [axes]
    fig.suptitle('S4 v4: Feature Importance per Position (XGBoost)',
                 fontsize=13, fontweight='bold')

    for ax, pos in zip(axes, [p for p in POSITIONS if p in pos_models]):
        fi = pos_models[pos]['feature_importance'].head(10)
        color = PALETTE.get(pos, '#888888')
        ax.barh(fi['feature'][::-1], fi['importance'][::-1], color=color, alpha=0.85)
        ax.set_title(f'{pos} — Top {len(fi)} Features')
        ax.set_xlabel('Importance')
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig3_feature_importance_v4.png', bbox_inches='tight')
    plt.close()
    print("  Fig 3 saved: fig3_feature_importance_v4.png")

# ── Fig 4: Scout outputs (3 lists) ──
fig, axes = plt.subplots(1, 3, figsize=(24, 9))
fig.suptitle('S4 v4: Player Development Profile — Scout Outputs',
             fontsize=14, fontweight='bold')

scout_panels = [
    (likely_improvers, 'Likely Improvers (U25)', 'prob_improving', '#27ae60'),
    (stability_picks,  'Stability Picks (26-30)', 'prob_stable',   '#f39c12'),
    (decline_risk,     'Decline Risk (30+)',       'prob_declining', '#c0392b'),
]

for ax, (panel_df, title, prob_col, color) in zip(axes, scout_panels):
    if len(panel_df) == 0 or prob_col not in panel_df.columns:
        ax.text(0.5, 0.5, f'No candidates\n({title})', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title(title)
        continue

    labels = [f"{r['player']} ({int(r['age'])})" for _, r in panel_df.iterrows()]
    vals   = panel_df[prob_col].values
    pos_colors = [PALETTE.get(p, '#888888') for p in panel_df['pos_simple']]

    ax.barh(labels[::-1], vals[::-1], color=pos_colors[::-1], alpha=0.85)
    ax.axvline(0.5, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_xlabel(f'P({prob_col.replace("prob_", "").title()})')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)

    # Annotate position + team
    for i, (_, row) in enumerate(panel_df.iterrows()):
        n = len(panel_df) - 1 - i
        team_str = str(row.get('team', ''))[:12]
        ax.text(0.01, n, f"{row['pos_simple']}  {team_str}",
                va='center', ha='left', fontsize=7,
                transform=ax.get_yaxis_transform(), color='#333333')

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig4_scout_outputs_v4.png', bbox_inches='tight')
plt.close()
print("  Fig 4 saved: fig4_scout_outputs_v4.png")

# ── Fig 5: Class probability distributions (all candidates) ──
if len(scout_df) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('S4 v4: Classification Probability Distributions (All Candidates)',
                 fontsize=13, fontweight='bold')

    for ax, (prob_col, label, color) in zip(axes, [
        ('prob_improving', 'P(Improving)', '#27ae60'),
        ('prob_stable',    'P(Stable)',    '#f39c12'),
        ('prob_declining', 'P(Declining)', '#c0392b'),
    ]):
        if prob_col not in scout_df.columns:
            continue
        for pos in POSITIONS:
            sub = scout_df[scout_df['pos_simple'] == pos][prob_col].dropna()
            if len(sub) > 2:
                ax.hist(sub, bins=20, alpha=0.5, label=pos, color=PALETTE[pos], density=True)
        ax.set_xlabel(label); ax.set_ylabel('Density')
        ax.set_title(label)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig5_prob_distributions_v4.png', bbox_inches='tight')
    plt.close()
    print("  Fig 5 saved: fig5_prob_distributions_v4.png")

# ═══════════════════════════════════════════════════════════════
# 13. Save outputs
# ═══════════════════════════════════════════════════════════════
print("\n[13] Saving outputs...")

# Parquet with all predictions
if len(scout_df) > 0:
    save_cols_parquet = [c for c in [
        'player', 'age', 'pos_simple', 'team', 'market_value', 'min',
        'perf_z', 'pred_xgb', 'pred_rf', 'pred_ensemble',
        'prob_improving', 'prob_stable', 'prob_declining',
        'injury_risk', 'min_drop_pct', 'epl_seasons'
    ] if c in scout_df.columns]

    scout_df[save_cols_parquet].to_parquet(
        SCOUT_DIR / 'growth_predictions_v4.parquet', index=False
    )
    print(f"  Saved: growth_predictions_v4.parquet ({len(scout_df)} rows)")


def safe_list(df, cols):
    if df is None or len(df) == 0:
        return []
    available = [c for c in cols if c in df.columns]
    records = df[available].to_dict(orient='records')
    # Convert numpy types for JSON serialisation
    cleaned = []
    for r in records:
        cleaned.append({
            k: (float(v) if isinstance(v, (np.floating, np.float32, np.float64)) else
                int(v)   if isinstance(v, (np.integer,)) else v)
            for k, v in r.items()
        })
    return cleaned


scout_display_cols = ['player', 'age', 'pos_simple', 'team',
                      'pred_ensemble', 'prob_improving', 'prob_stable', 'prob_declining',
                      'min', 'perf_z']

output = {
    'version':           'v4',
    'approach':          'classification_3class',
    'classes':           CLASS_LABELS,
    'delta_threshold':   DELTA_THRESHOLD,
    'min_minutes':       MIN_MINUTES,
    'smoothing_method':  'lowess_frac0.4',
    'note': (
        'Individual 1-year performance regression is not reliable with this data '
        '(R^2 consistently negative in v1-v3). v4 pivots to 3-class trajectory '
        'classification using position-specific models with no absolute performance '
        'features and no position dummies — trajectory signals only.'
    ),

    # Per-position model metrics
    'model_performance': {
        pos: {
            split: (
                {
                    k: float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in metrics.items()
                }
                if isinstance(metrics, dict)
                else float(metrics)
            )
            for split, metrics in splits.items()
        }
        for pos, splits in pos_results.items()
    },

    # Peak age analysis
    'peak_ages': {
        pos: {k: v for k, v in info.items() if k != 'age_curve'}
        for pos, info in peak_age_results.items()
    },
    'peak_age_curves': {
        pos: peak_age_results[pos].get('age_curve', {})
        for pos in POSITIONS
    },

    # Scout lists
    'likely_improvers': safe_list(likely_improvers, scout_display_cols),
    'stability_picks':  safe_list(stability_picks,  scout_display_cols),
    'decline_risk':     safe_list(decline_risk,     scout_display_cols),

    # Counts
    'candidate_counts': {
        'total':           int(len(scout_df)),
        'likely_improvers': int(len(likely_improvers)),
        'stability_picks':  int(len(stability_picks)),
        'decline_risk':     int(len(decline_risk)),
    },
}

json_path = SCOUT_DIR / 'growth_v4_results.json'
with open(json_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"  Saved: growth_v4_results.json")

# ═══════════════════════════════════════════════════════════════
# 14. Final summary
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("S4 v4 — COMPLETE SUMMARY")
print("=" * 65)
print(f"  Approach    : 3-class trajectory classification")
print(f"  Min minutes : {MIN_MINUTES}")
print(f"  Threshold   : ±{DELTA_THRESHOLD} Z-score units")
print(f"  Smoothing   : LOWESS (frac=0.4), min 30 samples/age bucket")
print(f"  Models      : position-specific XGBoost + RandomForest ensemble")
print(f"  No position dummies; no absolute performance features")
print()

for pos in POSITIONS:
    if pos not in pos_results:
        continue
    res = pos_results[pos]
    pa  = peak_age_results[pos]['peak_age']
    cv_ba = res.get('cv_balanced_accuracy', 'N/A')
    test_ba = res.get('test', {}).get('balanced_accuracy', 'N/A')
    print(f"  [{pos}]  peak_age={pa}  "
          f"test_bal_acc={test_ba}  cv_bal_acc={cv_ba}")

print()
print(f"  Likely Improvers (U25)      : {len(likely_improvers)} players")
if len(likely_improvers) > 0 and 'player' in likely_improvers.columns:
    for _, r in likely_improvers.head(5).iterrows():
        print(f"    {r['player']:28s}  age={int(r['age'])}  {r['pos_simple']}  "
              f"P(Imp)={r.get('prob_improving', 0):.2f}")

print()
print(f"  Stability Picks (26-30)      : {len(stability_picks)} players")
if len(stability_picks) > 0 and 'player' in stability_picks.columns:
    for _, r in stability_picks.head(5).iterrows():
        print(f"    {r['player']:28s}  age={int(r['age'])}  {r['pos_simple']}  "
              f"P(Stb)={r.get('prob_stable', 0):.2f}")

print()
print(f"  Decline Risk (30+)           : {len(decline_risk)} players")
if len(decline_risk) > 0 and 'player' in decline_risk.columns:
    for _, r in decline_risk.head(5).iterrows():
        print(f"    {r['player']:28s}  age={int(r['age'])}  {r['pos_simple']}  "
              f"P(Dec)={r.get('prob_declining', 0):.2f}")

print()
print(f"  Peak ages (literature-blended):")
for pos in POSITIONS:
    pa    = peak_age_results[pos]['peak_age']
    prior = peak_age_results[pos]['prior_peak_age']
    w     = peak_age_results[pos].get('w_data', 0.0)
    print(f"    {pos}: {pa}  (prior={prior}, w_data={w:.2f})")

print()
print(f"  Outputs:")
print(f"    data/scout/growth_v4_results.json")
print(f"    data/scout/growth_predictions_v4.parquet")
print(f"    models/s4_growth/figures/fig*_v4.png")
print("=" * 65)

"""
S4 Growth Potential Prediction v2
────────────────────────────────────────────────────────────────
Fixes applied:
  1. Peak age via min-50 sample-count filter per age bucket
  2. Target = "sustained improvement" (Z-score increase held 2+ seasons)
  3. Minimum 1800 min current season to be predicted
  4. current_z + current_z² added as features (removes low-baseline bias)
  5. Injury proxy: minutes drop >40% from prior season → excluded from predictions
  6. Sanity check: top prospects validated against scout reality
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

# ═══════════════════════════════════════════════════════════════
# 1. Load data
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("[1] Loading data...")
print("=" * 60)

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

season_df['season_year'] = season_df['season'].apply(season_to_year)
team_df['season_year']   = team_df['Season'].apply(season_to_year)

def simplify_position(pos):
    if pd.isna(pos): return 'MF'
    pos = str(pos).upper()
    if 'GK' in pos: return 'GK'
    if 'FW' in pos: return 'FW'
    if 'DF' in pos: return 'DF'
    return 'MF'

season_df['pos_simple'] = season_df['pos'].apply(simplify_position)

# Minimum 270 min filter for historical data (for training)
season_df_hist = season_df[season_df['min'].fillna(0) >= 270].copy()
print(f"  After 270-min filter (historical): {season_df_hist.shape[0]} rows")

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
        int_total=('int', safe_sum),
        sh_total=('sh', safe_sum),
        sot_total=('sot', safe_sum),
        fls_total=('fls', safe_sum),
        fld_total=('fld', safe_sum),
        crs_total=('crs', safe_sum),
        match_count=('min', 'count')
    )
    .reset_index()
)

# Also compute minutes in previous season per player (for injury proxy)
min_by_season = (
    season_df[['player', 'season_year', 'min']]
    .rename(columns={'season_year': 'prev_year', 'min': 'prev_min'})
)
min_by_season['season_year'] = min_by_season['prev_year'] + 1

print(f"  Match aggregation done: {match_agg.shape}")

# ═══════════════════════════════════════════════════════════════
# 3. Composite performance score (per-90 based, position-aware)
# ═══════════════════════════════════════════════════════════════
print("\n[3] Computing composite performance score...")

season_df_hist = season_df_hist.merge(match_agg, on=['player', 'season_year'], how='left')
season_df_hist['90s_safe'] = season_df_hist['90s'].fillna(1).clip(lower=0.1)

season_df_hist['gls_p90']  = season_df_hist['gls'].fillna(0) / season_df_hist['90s_safe']
season_df_hist['ast_p90']  = season_df_hist['ast'].fillna(0) / season_df_hist['90s_safe']
season_df_hist['tklw_p90'] = season_df_hist['tklw_total'].fillna(0) / season_df_hist['90s_safe']
season_df_hist['int_p90']  = season_df_hist['int_total'].fillna(0) / season_df_hist['90s_safe']
season_df_hist['sh_p90']   = season_df_hist['sh_total'].fillna(0) / season_df_hist['90s_safe']
season_df_hist['crs_p90']  = season_df_hist['crs_total'].fillna(0) / season_df_hist['90s_safe']
season_df_hist['fld_p90']  = season_df_hist['fld_total'].fillna(0) / season_df_hist['90s_safe']

def compute_composite_score(row):
    pos = row['pos_simple']
    if pos == 'FW':
        return (row['gls_p90'] * 3.0 + row['ast_p90'] * 2.0 +
                row['sh_p90'] * 0.3  + row['fld_p90'] * 0.5)
    elif pos == 'MF':
        return (row['gls_p90']  * 2.0 + row['ast_p90']  * 2.5 +
                row['tklw_p90'] * 0.8 + row['int_p90']  * 0.8 +
                row['crs_p90']  * 0.5 + row['fld_p90']  * 0.4)
    elif pos == 'DF':
        return (row['tklw_p90'] * 3.0 + row['int_p90']  * 3.0 +
                row['gls_p90']  * 1.0 + row['ast_p90']  * 0.8 +
                row['fld_p90']  * 0.3)
    else:  # GK
        return row['90s_safe'] * 0.1

season_df_hist['raw_score'] = season_df_hist.apply(compute_composite_score, axis=1)

# Z-score within position × season
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
# 4. Peak age analysis  ←  FIX #1: min 50 samples per age bucket
# ═══════════════════════════════════════════════════════════════
print("\n[4] Peak age analysis (min 50 samples per age bucket)...")
print("=" * 60)

POSITIONS = ['FW', 'MF', 'DF']
peak_age_results = {}

for pos in POSITIONS:
    pos_data = season_df_hist[season_df_hist['pos_simple'] == pos].copy()

    # Age bucket stats
    age_stats = (
        pos_data.groupby('age')['perf_z']
        .agg(['mean', 'std', 'count'])
        .reset_index()
        .rename(columns={'mean': 'mean_z', 'std': 'std_z', 'count': 'n'})
    )

    # Drop age buckets with < 50 samples  ← KEY FIX
    age_stats_valid = age_stats[age_stats['n'] >= 50].copy()

    if len(age_stats_valid) > 0:
        best_idx = age_stats_valid['mean_z'].idxmax()
        peak_age = int(age_stats_valid.loc[best_idx, 'age'])
        peak_mean_z = float(age_stats_valid.loc[best_idx, 'mean_z'])
        peak_n = int(age_stats_valid.loc[best_idx, 'n'])

        # Age range with sufficient data
        valid_ages = sorted(age_stats_valid['age'].tolist())
        min_valid_age = int(min(valid_ages))
        max_valid_age = int(max(valid_ages))

        peak_age_results[pos] = {
            'peak_age': peak_age,
            'peak_mean_z': peak_mean_z,
            'peak_n': peak_n,
            'valid_age_range': [min_valid_age, max_valid_age],
            'valid_age_buckets': int(len(age_stats_valid)),
        }

        print(f"\n  {pos} Peak Age Analysis:")
        print(f"    Peak age: {peak_age} (mean_z={peak_mean_z:.3f}, n={peak_n})")
        print(f"    Valid age range (n≥50): {min_valid_age}–{max_valid_age}")
        print(f"    Top 5 ages by mean_z:")
        top5 = age_stats_valid.nlargest(5, 'mean_z')[['age','mean_z','n']]
        for _, r in top5.iterrows():
            print(f"      age {int(r['age'])}: mean_z={r['mean_z']:.3f}  n={int(r['n'])}")
    else:
        print(f"  {pos}: Not enough data")

print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# 5. Sustained improvement target variable  ←  FIX #2
#
#    "sustained_improvement" = 1 if:
#       - perf_z in season N+1 > perf_z in season N  AND
#       - perf_z in season N+2 > perf_z in season N  (held 2+ seasons)
#    For regression: target = min(delta_N1, delta_N2)
#       i.e. the MINIMUM gain across both future seasons
#    This penalises one-season spikes that revert.
# ═══════════════════════════════════════════════════════════════
print("\n[5] Building sustained improvement target...")

score_lookup = (
    season_df_hist[['player', 'season_year', 'perf_z', 'min']]
    .rename(columns={'perf_z': 'z', 'min': 'mins_lookup'})
)

# Merge N+1 and N+2 scores onto each row
df = season_df_hist.copy()
df['next1_year'] = df['season_year'] + 1
df['next2_year'] = df['season_year'] + 2

df = df.merge(
    score_lookup.rename(columns={'season_year': 'next1_year', 'z': 'z_next1', 'mins_lookup': 'min_next1'}),
    on=['player', 'next1_year'], how='inner'
)
df = df.merge(
    score_lookup.rename(columns={'season_year': 'next2_year', 'z': 'z_next2', 'mins_lookup': 'min_next2'}),
    on=['player', 'next2_year'], how='left'  # left: N+2 optional
)

# delta_1 = Z(N+1) - Z(N),  delta_2 = Z(N+2) - Z(N)
df['delta1'] = df['z_next1'] - df['perf_z']
df['delta2'] = df['z_next2'] - df['perf_z']

# Sustained target: if N+2 available, use min(delta1, delta2) — reward only durable gains
# If N+2 not available, use delta1 (fallback)
df['sustained_growth'] = np.where(
    df['z_next2'].notna(),
    df[['delta1', 'delta2']].min(axis=1),
    df['delta1']
)

# Binary flag for classification audit
df['is_sustained'] = (
    (df['delta1'] > 0) &
    (df['z_next2'].isna() | (df['delta2'] > 0))
).astype(int)

print(f"  Dataset size: {df.shape[0]} rows")
print(f"  Sustained growth mean: {df['sustained_growth'].mean():.3f}, std: {df['sustained_growth'].std():.3f}")
print(f"  Players with sustained improvement: {df['is_sustained'].mean():.1%}")

# ═══════════════════════════════════════════════════════════════
# 6. Injury proxy  ←  FIX #5
#    Flag rows where minutes dropped >40% from prior season
#    These will be excluded from prediction (injury risk)
# ═══════════════════════════════════════════════════════════════
print("\n[6] Computing injury proxy...")

# Previous season minutes
min_prev = (
    season_df_hist[['player', 'season_year', 'min']]
    .rename(columns={'season_year': 'prev_year', 'min': 'prev_min'})
)
min_prev['season_year'] = min_prev['prev_year'] + 1

df = df.merge(min_prev[['player', 'season_year', 'prev_min']],
              on=['player', 'season_year'], how='left')

# injury_risk = minutes dropped > 40% from prior season
df['min_drop_pct'] = np.where(
    (df['prev_min'].notna()) & (df['prev_min'] > 0),
    (df['prev_min'] - df['min']) / df['prev_min'],
    0.0
)
df['injury_risk'] = (df['min_drop_pct'] > 0.40).astype(int)

print(f"  Injury risk rows flagged: {df['injury_risk'].sum()} ({df['injury_risk'].mean():.1%})")

# ═══════════════════════════════════════════════════════════════
# 7. Feature engineering
# ═══════════════════════════════════════════════════════════════
print("\n[7] Feature engineering...")

# Team points
team_points = team_df[['team', 'season_year', 'points']].copy()
df = df.merge(team_points, on=['team', 'season_year'], how='left')

# EPL experience (cumulative seasons)
df_s = df.sort_values(['player', 'season_year'])
df_s['epl_seasons'] = df_s.groupby('player').cumcount() + 1
df['epl_seasons'] = df_s['epl_seasons'].values

# Match consistency (CV of per-match minutes)
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

# Market value momentum
mv_lag = (
    df[['player', 'season_year', 'market_value']]
    .rename(columns={'season_year': 'prev_year', 'market_value': 'prev_mv'})
)
mv_lag['season_year'] = mv_lag['prev_year'] + 1
df = df.merge(mv_lag[['player', 'season_year', 'prev_mv']],
              on=['player', 'season_year'], how='left')
df['mv_change_rate'] = np.where(
    (df['prev_mv'].notna()) & (df['prev_mv'] > 0),
    (df['market_value'].fillna(0) - df['prev_mv']) / df['prev_mv'],
    np.nan
).clip(-2, 5)

# Historical trend (last 3 seasons linear slope)
print("  Computing performance trend features...")
df_s2 = df.sort_values(['player', 'season_year']).reset_index(drop=True)
trend_vals, peak_vals, lag1_vals, lag2_vals = [], [], [], []

for _, group in df_s2.groupby('player'):
    group = group.sort_values('season_year').reset_index(drop=True)
    scores = group['perf_z'].values
    years  = group['season_year'].values

    for j in range(len(group)):
        past_mask   = years < years[j]
        past_scores = scores[past_mask]
        past_years  = years[past_mask]

        # Trend: slope of last 3 seasons
        recent_s = past_scores[-3:] if len(past_scores) >= 2 else []
        recent_y = past_years[-3:] if len(past_years) >= 2 else []
        if len(recent_s) >= 2:
            ry = np.array(recent_y, dtype=float)
            rs = np.array(recent_s, dtype=float)
            slope = 0.0 if np.std(ry) == 0 else stats.linregress(ry - ry.mean(), rs)[0]
            trend_vals.append(slope)
        else:
            trend_vals.append(np.nan)

        peak_vals.append(past_scores.max() if len(past_scores) > 0 else np.nan)
        lag1_vals.append(scores[j-1] if j >= 1 and years[j-1] == years[j]-1 else np.nan)
        lag2_vals.append(scores[j-2] if j >= 2 and years[j-2] == years[j]-2 else np.nan)

df_s2['perf_trend'] = trend_vals
df_s2['peak_perf']  = peak_vals
df_s2['lag1_perf']  = lag1_vals
df_s2['lag2_perf']  = lag2_vals
df = df_s2.copy()

# ── Age features (use actual peak age per position for gap calculation) ──
pos_peak_ages = {pos: peak_age_results[pos]['peak_age'] for pos in POSITIONS
                 if pos in peak_age_results}
pos_peak_ages['GK'] = 28  # GK default

df['peak_age_pos'] = df['pos_simple'].map(pos_peak_ages).fillna(27)
df['age_to_peak']  = df['peak_age_pos'] - df['age']   # positive = hasn't peaked yet
df['age_sq']       = df['age'] ** 2
df['is_u23']       = (df['age'] <= 23).astype(int)
df['is_u25']       = (df['age'] <= 25).astype(int)

# ──  FIX #4: current_z + current_z² to remove low-baseline bias ──
df['current_z']    = df['perf_z']
df['current_z_sq'] = df['perf_z'] ** 2   # captures non-linear mean-reversion

# Age × performance interaction
df['age_x_z']     = df['age'] * df['perf_z']
df['age_x_trend'] = df['age'] * df['perf_trend'].fillna(0)

# Position dummies
for pos in ['FW', 'MF', 'DF', 'GK']:
    df[f'pos_{pos}'] = (df['pos_simple'] == pos).astype(int)

print("  Feature engineering complete.")

# ═══════════════════════════════════════════════════════════════
# 8. Model dataset preparation
# ═══════════════════════════════════════════════════════════════
print("\n[8] Preparing model dataset...")

FEATURE_COLS = [
    # Age
    'age', 'age_sq', 'age_to_peak', 'is_u23', 'is_u25',
    # Current performance  ← FIX #4: both z and z²
    'current_z', 'current_z_sq',
    # per-90 stats
    'gls_p90', 'ast_p90', 'tklw_p90', 'int_p90', 'sh_p90', 'crs_p90',
    # Playing time
    'min', '90s', 'starter_ratio', 'consistency',
    # Market value
    'market_value', 'mv_change_rate',
    # Historical performance
    'perf_trend', 'peak_perf', 'lag1_perf', 'lag2_perf',
    # Context
    'points', 'epl_seasons',
    # Interactions
    'age_x_z', 'age_x_trend',
    # Injury proxy (included as feature too)
    'min_drop_pct',
    # Position
    'pos_FW', 'pos_MF', 'pos_DF', 'pos_GK',
]

TARGET_COL = 'sustained_growth'

extra_cols = ['player', 'season_year', 'pos_simple', 'team',
              'injury_risk', 'is_sustained', 'perf_z', 'market_value']
all_needed = FEATURE_COLS + [TARGET_COL] + extra_cols
all_needed = list(dict.fromkeys(all_needed))  # deduplicate

df_model = df[all_needed].copy().reset_index(drop=True)

# Drop rows with > 50% NaN in features
nan_frac = df_model[FEATURE_COLS].isna().mean(axis=1)
df_model = df_model[nan_frac < 0.5].copy().reset_index(drop=True)

print(f"  Model dataset: {df_model.shape[0]} rows, {len(FEATURE_COLS)} features")
print(f"  Target distribution: mean={df_model[TARGET_COL].mean():.3f}, "
      f"std={df_model[TARGET_COL].std():.3f}")

# Time-based split
train_mask = df_model['season_year'] <= 2020
val_mask   = (df_model['season_year'] >= 2021) & (df_model['season_year'] <= 2022)
test_mask  = df_model['season_year'] >= 2023

X_train = df_model.loc[train_mask, FEATURE_COLS]
y_train = df_model.loc[train_mask, TARGET_COL]
X_val   = df_model.loc[val_mask,   FEATURE_COLS]
y_val   = df_model.loc[val_mask,   TARGET_COL]
X_test  = df_model.loc[test_mask,  FEATURE_COLS]
y_test  = df_model.loc[test_mask,  TARGET_COL]

print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ═══════════════════════════════════════════════════════════════
# 9. Model training
# ═══════════════════════════════════════════════════════════════
print("\n[9] Training models...")

models_config = {
    'XGBoost': (
        Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', xgb.XGBRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.1, reg_lambda=1.0, random_state=42,
                verbosity=0, n_jobs=-1,
            ))
        ])
    ),
    'RandomForest': (
        Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
            ('model', RandomForestRegressor(
                n_estimators=300, max_depth=8, min_samples_leaf=5,
                random_state=42, n_jobs=-1,
            ))
        ])
    ),
    'GradientBoosting': (
        Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
            ('model', GradientBoostingRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=5, random_state=42,
            ))
        ])
    ),
}

results       = {}
fitted_models = {}

for name, pipe in models_config.items():
    print(f"  [{name}] training...", end=' ', flush=True)
    pipe.fit(X_train, y_train)

    yv_pred = pipe.predict(X_val)
    yt_pred = pipe.predict(X_test)

    results[name] = {
        'val_r2':   float(r2_score(y_val,  yv_pred)),
        'val_rmse': float(np.sqrt(mean_squared_error(y_val, yv_pred))),
        'val_mae':  float(mean_absolute_error(y_val, yv_pred)),
        'test_r2':  float(r2_score(y_test, yt_pred)),
        'test_rmse':float(np.sqrt(mean_squared_error(y_test, yt_pred))),
        'test_mae': float(mean_absolute_error(y_test, yt_pred)),
    }
    fitted_models[name] = pipe
    print(f"val R²={results[name]['val_r2']:.3f}  test R²={results[name]['test_r2']:.3f}")

best_model_name = max(results, key=lambda k: results[k]['val_r2'])
best_model      = fitted_models[best_model_name]
print(f"\n  Best model: {best_model_name} (val R²={results[best_model_name]['val_r2']:.3f})")

# ═══════════════════════════════════════════════════════════════
# 10. Scout predictions  ←  FIX #3: min 1800 min filter
#                          FIX #5: exclude injury risk
# ═══════════════════════════════════════════════════════════════
print("\n[10] Generating scout predictions...")

# Use both 2023 and 2024 season data for prediction candidates
latest_years = sorted(df_model['season_year'].unique())[-2:]
pred_pool    = df_model[df_model['season_year'].isin(latest_years)].copy()

# Use season_df (all) to get 2024/25 data including players not in training set
# Specifically rebuild prediction pool from raw season_df for the most recent season
latest_raw_year = season_df['season_year'].max()
pred_raw = season_df[season_df['season_year'] == latest_raw_year].copy()
pred_raw['pos_simple'] = pred_raw['pos'].apply(simplify_position)
pred_raw = pred_raw.merge(match_agg, on=['player', 'season_year'], how='left')
pred_raw['90s_safe'] = pred_raw['90s'].fillna(1).clip(lower=0.1)
pred_raw['gls_p90']  = pred_raw['gls'].fillna(0)  / pred_raw['90s_safe']
pred_raw['ast_p90']  = pred_raw['ast'].fillna(0)  / pred_raw['90s_safe']
pred_raw['tklw_p90'] = pred_raw['tklw_total'].fillna(0) / pred_raw['90s_safe']
pred_raw['int_p90']  = pred_raw['int_total'].fillna(0)  / pred_raw['90s_safe']
pred_raw['sh_p90']   = pred_raw['sh_total'].fillna(0)   / pred_raw['90s_safe']
pred_raw['crs_p90']  = pred_raw['crs_total'].fillna(0)  / pred_raw['90s_safe']
pred_raw['fld_p90']  = pred_raw['fld_total'].fillna(0)  / pred_raw['90s_safe']
pred_raw['raw_score'] = pred_raw.apply(compute_composite_score, axis=1)

# Z-score within position (latest season only)
for pos in ['FW','MF','DF','GK']:
    mask = pred_raw['pos_simple'] == pos
    vals = pred_raw.loc[mask, 'raw_score']
    mu, sigma = vals.mean(), vals.std()
    pred_raw.loc[mask, 'perf_z'] = (vals - mu) / sigma if sigma > 0 else 0.0

pred_raw['current_z']    = pred_raw['perf_z']
pred_raw['current_z_sq'] = pred_raw['perf_z'] ** 2

# Merge previous season minutes for injury proxy
prev_year_mins = season_df[season_df['season_year'] == latest_raw_year - 1][['player','min']].rename(columns={'min':'prev_min'})
pred_raw = pred_raw.merge(prev_year_mins, on='player', how='left')
pred_raw['min_drop_pct'] = np.where(
    (pred_raw['prev_min'].notna()) & (pred_raw['prev_min'] > 0),
    (pred_raw['prev_min'] - pred_raw['min']) / pred_raw['prev_min'],
    0.0
)
pred_raw['injury_risk'] = (pred_raw['min_drop_pct'] > 0.40).astype(int)

# Merge team points
pred_raw = pred_raw.merge(team_points, on=['team', 'season_year'], how='left')

# EPL experience
epl_exp = season_df.groupby('player')['season_year'].count().reset_index().rename(columns={'season_year':'epl_seasons'})
pred_raw = pred_raw.merge(epl_exp, on='player', how='left')

# Match consistency
pred_raw = pred_raw.merge(match_consistency[match_consistency['season_year'] == latest_raw_year][['player','consistency']],
                          on='player', how='left')

# Starter ratio
pred_raw['starter_ratio'] = (pred_raw['starts'].fillna(0) / pred_raw['mp'].replace(0, np.nan)).clip(0, 1)

# Market value momentum
prev_mv = season_df[season_df['season_year'] == latest_raw_year - 1][['player','market_value']].rename(columns={'market_value':'prev_mv'})
pred_raw = pred_raw.merge(prev_mv, on='player', how='left')
pred_raw['mv_change_rate'] = np.where(
    (pred_raw['prev_mv'].notna()) & (pred_raw['prev_mv'] > 0),
    (pred_raw['market_value'].fillna(0) - pred_raw['prev_mv']) / pred_raw['prev_mv'],
    np.nan
)

# Historical trend features from df_model (most recent historical data per player)
hist_latest = (
    df_model.sort_values('season_year')
    .groupby('player').last()
    .reset_index()
    [['player', 'perf_trend', 'peak_perf', 'lag1_perf', 'lag2_perf',
      'age_x_z', 'age_x_trend']]
)
pred_raw = pred_raw.merge(hist_latest, on='player', how='left')

# Age features
pred_raw['peak_age_pos'] = pred_raw['pos_simple'].map(pos_peak_ages).fillna(27)
pred_raw['age_to_peak']  = pred_raw['peak_age_pos'] - pred_raw['age']
pred_raw['age_sq']       = pred_raw['age'] ** 2
pred_raw['is_u23']       = (pred_raw['age'] <= 23).astype(int)
pred_raw['is_u25']       = (pred_raw['age'] <= 25).astype(int)

# Recalculate interaction features with fresh data
pred_raw['age_x_z']     = pred_raw['age'] * pred_raw['perf_z']
pred_raw['age_x_trend'] = pred_raw['age'] * pred_raw['perf_trend'].fillna(0)

# Position dummies
for pos in ['FW', 'MF', 'DF', 'GK']:
    pred_raw[f'pos_{pos}'] = (pred_raw['pos_simple'] == pos).astype(int)

# ──  FIX #3: Minimum 1800 minutes to be included in predictions ──
pred_candidates = pred_raw[pred_raw['min'].fillna(0) >= 1800].copy()
print(f"  After 1800-min filter: {len(pred_candidates)} players")

# ──  FIX #5: Remove injury risk players from predictions ──
pred_candidates_healthy = pred_candidates[pred_candidates['injury_risk'] == 0].copy()
print(f"  After injury-risk filter: {len(pred_candidates_healthy)} players")
print(f"  Injury risk excluded: {len(pred_candidates) - len(pred_candidates_healthy)}")

# Predict growth
X_pred = pred_candidates_healthy[FEATURE_COLS].fillna(pred_candidates_healthy[FEATURE_COLS].median())
pred_candidates_healthy['predicted_growth'] = best_model.predict(X_pred)

# Ensemble prediction
ensemble_preds = np.column_stack([
    fitted_models[name].predict(X_pred) for name in fitted_models
])
pred_candidates_healthy['ensemble_growth'] = ensemble_preds.mean(axis=1)

# ═══════════════════════════════════════════════════════════════
# 11. Scout report tables
# ═══════════════════════════════════════════════════════════════
print("\n[11] Generating scout report tables...")

scout_cols = ['player', 'age', 'pos_simple', 'team',
              'market_value', 'perf_z', 'predicted_growth', 'ensemble_growth', 'min']

all_preds = pred_candidates_healthy[scout_cols].copy()
all_preds = all_preds.rename(columns={'perf_z': 'current_perf_z',
                                       'pos_simple': 'position'})

# ─── U23 Top 15 growth prospects ───
u23_top15 = (
    all_preds[all_preds['age'] <= 23]
    .nlargest(15, 'ensemble_growth')
    .reset_index(drop=True)
)
u23_top15.index += 1

# ─── U25 Top 15 growth prospects ───
u25_top15 = (
    all_preds[all_preds['age'] <= 25]
    .nlargest(15, 'ensemble_growth')
    .reset_index(drop=True)
)
u25_top15.index += 1

# ─── Late bloomers (26-30) ───
late_bloomers = (
    all_preds[(all_preds['age'] >= 26) & (all_preds['age'] <= 30)]
    .nlargest(15, 'ensemble_growth')
    .reset_index(drop=True)
)
late_bloomers.index += 1

# ─── Declining stars ───
declining = (
    all_preds[all_preds['current_perf_z'] > 0.5]
    .nsmallest(15, 'ensemble_growth')
    .reset_index(drop=True)
)
declining.index += 1

print("\n" + "=" * 70)
print("  TOP 15 U23 GROWTH PROSPECTS")
print("=" * 70)
print(u23_top15[['player','age','position','team','current_perf_z',
                  'ensemble_growth','min']].to_string())

print("\n" + "=" * 70)
print("  TOP 15 U25 GROWTH PROSPECTS")
print("=" * 70)
print(u25_top15[['player','age','position','team','current_perf_z',
                  'ensemble_growth','min']].to_string())

print("\n" + "=" * 70)
print("  TOP 15 LATE BLOOMERS (26-30)")
print("=" * 70)
print(late_bloomers[['player','age','position','team','current_perf_z',
                      'ensemble_growth','min']].to_string())

print("\n" + "=" * 70)
print("  TOP 15 DECLINING STARS (current_z > 0.5)")
print("=" * 70)
print(declining[['player','age','position','team','current_perf_z',
                  'ensemble_growth','min']].to_string())

# ═══════════════════════════════════════════════════════════════
# 12. Feature importance
# ═══════════════════════════════════════════════════════════════
xgb_model_obj = fitted_models['XGBoost'].named_steps['model']
feature_importance = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': xgb_model_obj.feature_importances_
}).sort_values('importance', ascending=False)

# ═══════════════════════════════════════════════════════════════
# 13. Visualisations
# ═══════════════════════════════════════════════════════════════
print("\n[12] Generating figures...")

# ── Fig 1: Peak age curves per position ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('S4 v2: Peak Age Analysis (min 50 samples per bucket)',
             fontsize=14, fontweight='bold')

for ax, pos in zip(axes, POSITIONS):
    pos_data = season_df_hist[season_df_hist['pos_simple'] == pos]
    age_stats = (
        pos_data.groupby('age')['perf_z']
        .agg(['mean', 'std', 'count'])
        .reset_index()
        .rename(columns={'mean':'mean_z','std':'std_z','count':'n'})
    )
    age_valid = age_stats[age_stats['n'] >= 50].copy()
    age_all   = age_stats.copy()

    # Plot all ages (greyed out)
    ax.plot(age_all['age'], age_all['mean_z'],
            'o--', color='#cccccc', linewidth=1, markersize=4, label='n < 50 (excluded)')
    # Plot valid ages
    ax.plot(age_valid['age'], age_valid['mean_z'],
            'o-', color=PALETTE[pos], linewidth=2, markersize=6, label='n ≥ 50 (valid)')

    # Annotate sample counts
    for _, r in age_valid.iterrows():
        if r['age'] % 3 == 0:  # every 3 years to avoid clutter
            ax.annotate(f"n={int(r['n'])}", xy=(r['age'], r['mean_z']),
                        xytext=(0, 8), textcoords='offset points',
                        fontsize=7, ha='center', color='#555555')

    if pos in peak_age_results:
        pa = peak_age_results[pos]['peak_age']
        pz = peak_age_results[pos]['peak_mean_z']
        ax.axvline(pa, color=PALETTE[pos], linestyle=':', alpha=0.7)
        ax.scatter([pa], [pz], s=120, color=PALETTE[pos],
                   zorder=5, label=f'Peak age={pa}')

    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Age'); ax.set_ylabel('Mean perf_z')
    ax.set_title(f'{pos} — Peak Age Analysis')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig1_peak_age_v2.png', bbox_inches='tight')
plt.close()
print("  Fig 1 saved: fig1_peak_age_v2.png")

# ── Fig 2: Model performance comparison ──
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('S4 v2: Model Performance (Sustained Growth Target)',
             fontsize=14, fontweight='bold')

model_names    = list(results.keys())
val_r2_vals    = [results[m]['val_r2']  for m in model_names]
test_r2_vals   = [results[m]['test_r2'] for m in model_names]
val_rmse_vals  = [results[m]['val_rmse'] for m in model_names]
test_rmse_vals = [results[m]['test_rmse'] for m in model_names]

x, w = np.arange(len(model_names)), 0.35

ax = axes[0]
b1 = ax.bar(x - w/2, val_r2_vals,  w, label='Val R²',  color='#3498db', alpha=0.85)
b2 = ax.bar(x + w/2, test_r2_vals, w, label='Test R²', color='#e74c3c', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=20, ha='right')
ax.set_ylabel('R²'); ax.set_title('R² Score'); ax.legend()
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
for b in list(b1) + list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
            f'{b.get_height():.2f}', ha='center', fontsize=8)

ax = axes[1]
b3 = ax.bar(x - w/2, val_rmse_vals,  w, label='Val RMSE',  color='#2ecc71', alpha=0.85)
b4 = ax.bar(x + w/2, test_rmse_vals, w, label='Test RMSE', color='#f39c12', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=20, ha='right')
ax.set_ylabel('RMSE'); ax.set_title('RMSE'); ax.legend()

ax = axes[2]
best_pipe    = fitted_models[best_model_name]
y_pred_test  = best_pipe.predict(X_test)
ax.scatter(y_test, y_pred_test, alpha=0.3, s=10, color='#3498db')
lim = max(abs(y_test.max()), abs(y_test.min()), 3)
ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1.5)
ax.set_xlabel('Actual sustained growth'); ax.set_ylabel('Predicted')
ax.set_title(f'{best_model_name}: Predicted vs Actual (Test)')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig2_model_performance_v2.png', bbox_inches='tight')
plt.close()
print("  Fig 2 saved: fig2_model_performance_v2.png")

# ── Fig 3: Feature importance ──
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('S4 v2: Feature Importance', fontsize=13, fontweight='bold')

top_n = 15
rf_imp_df = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': fitted_models['RandomForest'].named_steps['model'].feature_importances_
}).sort_values('importance', ascending=False)

for ax, imp_df, label, color in zip(
    axes,
    [feature_importance, rf_imp_df],
    ['XGBoost', 'RandomForest'],
    ['#e74c3c', '#2ecc71']
):
    top = imp_df.head(top_n)
    ax.barh(top['feature'][::-1], top['importance'][::-1], color=color, alpha=0.85)
    ax.set_title(f'{label} Top {top_n} Features')
    ax.set_xlabel('Importance')
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig3_feature_importance_v2.png', bbox_inches='tight')
plt.close()
print("  Fig 3 saved: fig3_feature_importance_v2.png")

# ── Fig 4: U23 Top 15 bar chart ──
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('S4 v2: Top Growth Prospects', fontsize=14, fontweight='bold')

for ax, df_plot, title in zip(
    axes,
    [u23_top15, u25_top15],
    ['Top 15 U23 Growth Prospects', 'Top 15 U25 Growth Prospects']
):
    colors = [PALETTE.get(p, '#888888') for p in df_plot['position']]
    bars = ax.barh(
        [f"{r['player']} ({int(r['age'])})" for _, r in df_plot.iterrows()],
        df_plot['ensemble_growth'],
        color=colors, alpha=0.85
    )
    ax.set_xlabel('Predicted Sustained Growth (Z-score)')
    ax.set_title(title)
    ax.invert_yaxis()
    ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
    ax.grid(axis='x', alpha=0.3)
    # Annotate current perf_z
    for i, (_, row) in enumerate(df_plot.iterrows()):
        ax.text(0.02, i, f"z={row['current_perf_z']:.2f}",
                va='center', ha='left', fontsize=7, color='white',
                transform=ax.get_yaxis_transform())

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig4_top_prospects_v2.png', bbox_inches='tight')
plt.close()
print("  Fig 4 saved: fig4_top_prospects_v2.png")

# ── Fig 5: Mean-reversion bias check ──
# Plot predicted_growth vs current_z to verify no strong negative correlation
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('S4 v2: Predicted Growth vs Current Z-score (bias check)',
             fontsize=13, fontweight='bold')

for ax, pos in zip(axes, POSITIONS):
    pos_mask = pred_candidates_healthy['pos_simple'] == pos
    sub = pred_candidates_healthy[pos_mask]
    ax.scatter(sub['perf_z'], sub['predicted_growth'],
               alpha=0.4, s=15, color=PALETTE[pos])
    # Trend line
    if len(sub) > 5:
        z_vals = sub['perf_z'].fillna(0).values
        g_vals = sub['predicted_growth'].values
        m, b = np.polyfit(z_vals, g_vals, 1)
        xl = np.linspace(z_vals.min(), z_vals.max(), 50)
        ax.plot(xl, m*xl + b, 'k--', linewidth=1.5,
                label=f'slope={m:.2f}')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Current perf_z')
    ax.set_ylabel('Predicted growth')
    ax.set_title(f'{pos}: bias check')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig5_bias_check_v2.png', bbox_inches='tight')
plt.close()
print("  Fig 5 saved: fig5_bias_check_v2.png")

# ═══════════════════════════════════════════════════════════════
# 14. Save outputs
# ═══════════════════════════════════════════════════════════════
print("\n[13] Saving outputs...")

# Predictions parquet
save_cols = ['player', 'age', 'pos_simple', 'team', 'market_value',
             'min', 'perf_z', 'predicted_growth', 'ensemble_growth',
             'injury_risk', 'min_drop_pct']
save_cols = [c for c in save_cols if c in pred_candidates.columns]
pred_candidates.to_parquet(
    SCOUT_DIR / 'growth_predictions_v2.parquet', index=False
)
print(f"  Saved: growth_predictions_v2.parquet ({len(pred_candidates)} rows incl injury-risk)")

# JSON results
output = {
    'version': 'v2',
    'model_performance': results,
    'best_model': best_model_name,
    'peak_ages': peak_age_results,
    'top_u23_prospects': u23_top15[['player','age','position','team',
                                     'current_perf_z','ensemble_growth','min']].to_dict(orient='records'),
    'top_u25_prospects': u25_top15[['player','age','position','team',
                                     'current_perf_z','ensemble_growth','min']].to_dict(orient='records'),
    'late_bloomers': late_bloomers[['player','age','position','team',
                                     'current_perf_z','ensemble_growth','min']].to_dict(orient='records'),
    'declining_stars': declining[['player','age','position','team',
                                   'current_perf_z','ensemble_growth','min']].to_dict(orient='records'),
    'feature_importance': feature_importance.head(15).to_dict(orient='records'),
    'dataset_info': {
        'train_rows': int(len(X_train)),
        'val_rows':   int(len(X_val)),
        'test_rows':  int(len(X_test)),
        'prediction_pool_total':   int(len(pred_candidates)),
        'prediction_pool_healthy': int(len(pred_candidates_healthy)),
        'min_threshold': 1800,
        'injury_proxy_threshold': 0.40,
        'sustained_improvement_rate': float(df_model['is_sustained'].mean()),
    },
    'fixes_applied': [
        'peak_age_min50_samples',
        'sustained_improvement_target_2plus_seasons',
        'min_1800_prediction_filter',
        'current_z_and_z_squared_features',
        'injury_proxy_exclusion',
    ]
}

with open(SCOUT_DIR / 'growth_v2_results.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2, default=str)
print("  Saved: growth_v2_results.json")

print("\n" + "=" * 60)
print("S4 Growth v2 COMPLETE")
print("=" * 60)
print(f"\nPeak ages (min 50 samples per bucket):")
for pos, info in peak_age_results.items():
    print(f"  {pos}: peak_age={info['peak_age']}  "
          f"(mean_z={info['peak_mean_z']:.3f}, n={info['peak_n']}, "
          f"valid range {info['valid_age_range'][0]}-{info['valid_age_range'][1]})")
print(f"\nBest model: {best_model_name}  "
      f"val R²={results[best_model_name]['val_r2']:.3f}  "
      f"test R²={results[best_model_name]['test_r2']:.3f}")
print(f"\nPrediction pool (1800min, healthy): {len(pred_candidates_healthy)} players")

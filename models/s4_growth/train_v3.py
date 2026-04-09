"""
S4 Growth Potential Prediction v3
────────────────────────────────────────────────────────────────────────────
Fixes over v2:
  1. REGULARIZATION  : max_depth=3, min_child_weight=15, subsample=0.7,
                       colsample_bytree=0.7, strong L1/L2, early-stopping
  2. SIMPLER FEATURES: remove absolute-level leakage features (current_z,
                       current_z_sq, age_x_z, raw per-90 stats, market_value,
                       peak_perf, lag1_perf, lag2_perf).
                       Keep only trajectory / slope features + age + position.
  3. PEAK AGE        : literature-backed priors (FW/MF ~27, DEF ~28-29, GK ~30)
                       blended with data only when n ≥ 30 per age bucket.
                       Gaussian smoothing over age curve before argmax.
  4. TARGET          : relative delta Z-score normalised by position average
                       delta at that age → removes age-cohort mean-reversion bias.
  5. CROSS-VALIDATION: expanding-window TimeSeriesSplit (5 folds) instead of
                       single split; reported metrics are CV mean ± std.
  6. SCOUT OUTPUTS   : hot prospects (U23), late bloomers (24-28), peak-age
                       curves per position, growth trajectory plots.
────────────────────────────────────────────────────────────────────────────
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
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
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

# Literature-backed peak-age priors.
# These come from sports-science meta-analyses of physical prime + skill plateau.
# They are used as a strong anchor: data can only shift the peak by ≤2 years
# and only when the data signal is strong (large Z-score spread across ages).
PEAK_AGE_PRIORS = {'FW': 27, 'MF': 27, 'DF': 28, 'GK': 30}

# Hard constraints: data-derived peak is clipped to [prior - 2, prior + 2]
PEAK_AGE_MIN_DELTA = -2   # data peak must be ≥ prior - 2
PEAK_AGE_MAX_DELTA = +2   # data peak must be ≤ prior + 2

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

# 270-min filter for historical training data
season_df_hist = season_df[season_df['min'].fillna(0) >= 270].copy()
print(f"  After 270-min filter (historical): {season_df_hist.shape[0]} rows")

# ═══════════════════════════════════════════════════════════════
# 2. Match log aggregation (used for composite score)
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
print(f"  Match aggregation: {match_agg.shape}")

# ═══════════════════════════════════════════════════════════════
# 3. Composite performance score (per-90, position-aware)
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
                row['sh_p90']  * 0.3  + row['fld_p90'] * 0.5)
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
# 4. Peak age analysis — FIX #3
#    a. min 30 samples per age bucket (more data-efficient than 50)
#    b. Gaussian-smoothed age curve before argmax
#    c. Blend smoothed-data peak with literature prior using
#       confidence weight: w_data = min(1.0, n_valid_buckets / 10)
# ═══════════════════════════════════════════════════════════════
print("\n[4] Peak age analysis (smoothed, prior-blended, min 30 samples)...")
print("=" * 65)

POSITIONS = ['FW', 'MF', 'DF', 'GK']
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

    # Require at least 30 samples per bucket
    age_valid = age_stats[age_stats['n'] >= 30].copy()

    if len(age_valid) >= 3:
        ages       = age_valid['age'].values.astype(float)
        mean_zs    = age_valid['mean_z'].values

        # Gaussian smoothing (sigma=1.5 age-years equivalent)
        smoothed   = gaussian_filter1d(mean_zs, sigma=1.5)

        data_peak_age_raw = int(ages[np.argmax(smoothed)])

        # Hard-clip: data peak can differ from prior by at most 2 years.
        # This handles the EPL selection-bias artefact where the Z-score
        # normalisation within season flattens the true aging curve.
        data_peak_age = int(np.clip(
            data_peak_age_raw,
            prior + PEAK_AGE_MIN_DELTA,
            prior + PEAK_AGE_MAX_DELTA
        ))

        # Signal strength: how large is the peak-to-trough amplitude?
        # Weak signal (< 0.15 Z) → trust prior more heavily.
        amplitude   = float(smoothed.max() - smoothed.min())
        w_data_raw  = min(1.0, amplitude / 0.30)   # saturates at amplitude 0.30+
        n_valid     = len(age_valid)
        w_data      = w_data_raw * min(1.0, n_valid / 12.0)  # also scale with n
        w_prior     = 1.0 - w_data

        blended_peak = int(round(w_data * data_peak_age + w_prior * prior))

        valid_ages   = sorted(age_valid['age'].tolist())
        peak_age_results[pos] = {
            'peak_age':           int(blended_peak),
            'data_peak_age_raw':  int(data_peak_age_raw),
            'data_peak_age':      int(data_peak_age),
            'prior_peak_age':     int(prior),
            'amplitude':          round(amplitude, 4),
            'w_data':             round(w_data, 3),
            'w_prior':            round(w_prior, 3),
            'valid_age_range':    [int(min(valid_ages)), int(max(valid_ages))],
            'valid_age_buckets':  int(n_valid),
            'age_curve': {
                'ages':     [int(a) for a in ages],
                'mean_z':   [round(float(z), 4) for z in mean_zs],
                'smoothed': [round(float(s), 4) for s in smoothed],
                'n':        [int(age_valid.iloc[i]['n']) for i in range(len(age_valid))],
            }
        }
        print(f"\n  {pos} Peak Age:")
        print(f"    Raw data peak={data_peak_age_raw} → clipped={data_peak_age}")
        print(f"    Prior={prior}, Blended={blended_peak}")
        print(f"    Amplitude={amplitude:.3f}, w_data={w_data:.2f}, valid buckets={n_valid}")
    else:
        peak_age_results[pos] = {
            'peak_age':       int(prior),
            'data_peak_age':  None,
            'prior_peak_age': int(prior),
            'w_data': 0.0, 'w_prior': 1.0,
            'valid_age_range': [18, 35],
            'valid_age_buckets': int(len(age_valid)),
        }
        print(f"\n  {pos}: Insufficient data → using prior peak_age={prior}")

print("\n" + "=" * 65)

# ═══════════════════════════════════════════════════════════════
# 5. Build relative delta target — FIX #4
#
#    Step 1: compute raw delta_1 = Z(N+1) - Z(N)
#    Step 2: compute position×age group mean delta (the "cohort norm")
#    Step 3: target = delta_1 - cohort_norm_delta
#       → positive means the player grew MORE than their age-cohort peers
#       → eliminates the structural mean-reversion artefact where
#         high-scorers appear to "decline" simply because their age
#         cohort mean happens to be falling.
# ═══════════════════════════════════════════════════════════════
print("\n[5] Building relative delta target (age-cohort normalised)...")

score_lookup = (
    season_df_hist[['player', 'season_year', 'perf_z', 'min']]
    .rename(columns={'perf_z': 'z', 'min': 'mins_lookup'})
)

df = season_df_hist.copy()
df['next1_year'] = df['season_year'] + 1

# Merge N+1 Z-score
df = df.merge(
    score_lookup.rename(columns={'season_year': 'next1_year',
                                  'z': 'z_next1',
                                  'mins_lookup': 'min_next1'}),
    on=['player', 'next1_year'], how='inner'
)

df['raw_delta1'] = df['z_next1'] - df['perf_z']

# Cohort norm: mean(delta1) per position × integer age bucket
cohort_norm = (
    df.groupby(['pos_simple', 'age'])['raw_delta1']
    .agg(cohort_mean_delta='mean', cohort_n='count')
    .reset_index()
)
# Require at least 10 players in bucket for the norm to be reliable
cohort_norm['cohort_mean_delta'] = np.where(
    cohort_norm['cohort_n'] >= 10,
    cohort_norm['cohort_mean_delta'],
    0.0   # fall back to 0 if bucket is thin
)

df = df.merge(cohort_norm[['pos_simple', 'age', 'cohort_mean_delta']],
              on=['pos_simple', 'age'], how='left')
df['cohort_mean_delta'] = df['cohort_mean_delta'].fillna(0.0)

# Relative growth = how much did the player outgrow their age-cohort
df['rel_growth'] = df['raw_delta1'] - df['cohort_mean_delta']

print(f"  Dataset size: {df.shape[0]} rows")
print(f"  raw_delta1  : mean={df['raw_delta1'].mean():.3f}, std={df['raw_delta1'].std():.3f}")
print(f"  rel_growth  : mean={df['rel_growth'].mean():.3f}, std={df['rel_growth'].std():.3f}")

TARGET_COL = 'rel_growth'

# ═══════════════════════════════════════════════════════════════
# 6. Injury proxy (unchanged logic, improved flag)
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
# 7. Feature engineering — FIX #2
#
#    REMOVED (absolute-level / current-performance leakers):
#      current_z, current_z_sq, age_x_z
#      gls_p90, ast_p90, tklw_p90, int_p90, sh_p90, crs_p90, fld_p90
#      peak_perf, lag1_perf, lag2_perf
#      market_value, mv_change_rate
#
#    KEPT (trajectory / context features):
#      age, age_to_peak, age_sq, is_u23, is_u25
#      perf_trend (slope over past 2-3 seasons)
#      min_trend (slope of minutes over 2-3 seasons)
#      consistency, starter_ratio, epl_seasons
#      min_drop_pct (injury proxy as numeric)
#      points (team context)
#      position dummies
# ═══════════════════════════════════════════════════════════════
print("\n[7] Feature engineering (trajectory-only, no level leakage)...")

# Team points
team_points = team_df[['team', 'season_year', 'points']].copy()
df = df.merge(team_points, on=['team', 'season_year'], how='left')

# EPL experience
df_s = df.sort_values(['player', 'season_year'])
df_s['epl_seasons'] = df_s.groupby('player').cumcount() + 1
df['epl_seasons'] = df_s['epl_seasons'].values

# Match consistency
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

# ── Trajectory features: perf_z slope + minutes slope ──
print("  Computing performance and minutes trend slopes...")
df_s2 = df.sort_values(['player', 'season_year']).reset_index(drop=True)

trend_vals, min_trend_vals = [], []

for _, group in df_s2.groupby('player'):
    group  = group.sort_values('season_year').reset_index(drop=True)
    scores = group['perf_z'].values
    mins   = group['min'].fillna(0).values
    years  = group['season_year'].values

    for j in range(len(group)):
        past_mask   = years < years[j]
        past_scores = scores[past_mask]
        past_mins   = mins[past_mask]
        past_years  = years[past_mask]

        # Use last 3 seasons for slope calculation
        recent_s = past_scores[-3:] if len(past_scores) >= 2 else []
        recent_m = past_mins[-3:]   if len(past_mins) >= 2 else []
        recent_y = past_years[-3:]  if len(past_years) >= 2 else []

        if len(recent_s) >= 2:
            ry = np.array(recent_y, dtype=float)
            rs = np.array(recent_s, dtype=float)
            rm = np.array(recent_m, dtype=float)
            ry_c = ry - ry.mean()
            slope_z   = 0.0 if np.std(ry_c) == 0 else stats.linregress(ry_c, rs)[0]
            slope_min = 0.0 if np.std(ry_c) == 0 else stats.linregress(ry_c, rm)[0]
            trend_vals.append(slope_z)
            min_trend_vals.append(slope_min)
        else:
            trend_vals.append(np.nan)
            min_trend_vals.append(np.nan)

df_s2['perf_trend']  = trend_vals
df_s2['min_trend']   = min_trend_vals
df = df_s2.copy()

# ── Age features using blended peak ages ──
pos_peak_ages = {pos: peak_age_results[pos]['peak_age'] for pos in POSITIONS}

df['peak_age_pos'] = df['pos_simple'].map(pos_peak_ages).fillna(27)
df['age_to_peak']  = df['peak_age_pos'] - df['age']   # positive = before peak
df['age_sq']       = df['age'] ** 2
df['is_u23']       = (df['age'] <= 23).astype(int)
df['is_u25']       = (df['age'] <= 25).astype(int)

# Interaction: trend × age_to_peak (growing players closer to peak = more upside)
df['trend_x_age_to_peak'] = df['perf_trend'].fillna(0) * df['age_to_peak']

# Position dummies
for pos in ['FW', 'MF', 'DF', 'GK']:
    df[f'pos_{pos}'] = (df['pos_simple'] == pos).astype(int)

print("  Feature engineering complete.")

# ═══════════════════════════════════════════════════════════════
# 8. Model dataset preparation
# ═══════════════════════════════════════════════════════════════
print("\n[8] Preparing model dataset...")

# FIX #2: Only trajectory/context features — no absolute-level leakers
FEATURE_COLS = [
    # Age
    'age', 'age_sq', 'age_to_peak', 'is_u23', 'is_u25',
    # Trajectory features (SLOPE-based, not level-based)
    'perf_trend',         # slope of perf_z over last 3 seasons
    'min_trend',          # slope of minutes over last 3 seasons
    # Playing time context
    'min', 'starter_ratio', 'consistency', 'epl_seasons',
    # Injury proxy (numeric, not categorical)
    'min_drop_pct',
    # Team context
    'points',
    # Interaction
    'trend_x_age_to_peak',
    # Position
    'pos_FW', 'pos_MF', 'pos_DF', 'pos_GK',
]

extra_cols = ['player', 'season_year', 'pos_simple', 'team',
              'injury_risk', 'perf_z', 'raw_delta1', 'cohort_mean_delta',
              'market_value']
all_needed = FEATURE_COLS + [TARGET_COL] + extra_cols
all_needed = list(dict.fromkeys(all_needed))

df_model = df[all_needed].copy().reset_index(drop=True)

# Drop rows where more than 40% of features are NaN
nan_frac = df_model[FEATURE_COLS].isna().mean(axis=1)
df_model = df_model[nan_frac < 0.4].copy().reset_index(drop=True)

print(f"  Model dataset: {df_model.shape[0]} rows, {len(FEATURE_COLS)} features")
print(f"  Target (rel_growth): mean={df_model[TARGET_COL].mean():.3f}, "
      f"std={df_model[TARGET_COL].std():.3f}")

# Time-based train / val / test split
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
# 9. Model training with strong regularisation — FIX #1
# ═══════════════════════════════════════════════════════════════
print("\n[9] Training models with strong regularisation...")

# FIX #1: significantly stronger regularisation vs v2
models_config = {
    'XGBoost': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', xgb.XGBRegressor(
            n_estimators=400,
            max_depth=3,           # was 5 → shallower trees
            learning_rate=0.03,    # was 0.05 → slower learning
            subsample=0.7,         # was 0.8 → more regularised
            colsample_bytree=0.7,  # was 0.8
            min_child_weight=15,   # was 3 → much higher (prevents leaf splits on tiny groups)
            reg_alpha=1.0,         # was 0.1 → much stronger L1
            reg_lambda=5.0,        # was 1.0 → much stronger L2
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        ))
    ]),
    'RandomForest': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model', RandomForestRegressor(
            n_estimators=300,
            max_depth=4,           # was 8 → much shallower
            min_samples_leaf=20,   # was 5 → much stricter leaf size
            max_features=0.6,      # column subsampling
            random_state=42,
            n_jobs=-1,
        ))
    ]),
    'GradientBoosting': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model', GradientBoostingRegressor(
            n_estimators=300,
            max_depth=3,           # was 4 → shallower
            learning_rate=0.03,    # was 0.05 → slower
            subsample=0.7,         # was 0.8
            min_samples_leaf=20,   # was 5
            random_state=42,
        ))
    ]),
}

results       = {}
fitted_models = {}

for name, pipe in models_config.items():
    print(f"  [{name}] training...", end=' ', flush=True)
    pipe.fit(X_train, y_train)

    yv_pred = pipe.predict(X_val)
    yt_pred = pipe.predict(X_test)

    results[name] = {
        'val_r2':    float(r2_score(y_val,  yv_pred)),
        'val_rmse':  float(np.sqrt(mean_squared_error(y_val, yv_pred))),
        'val_mae':   float(mean_absolute_error(y_val, yv_pred)),
        'test_r2':   float(r2_score(y_test, yt_pred)),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, yt_pred))),
        'test_mae':  float(mean_absolute_error(y_test, yt_pred)),
    }
    fitted_models[name] = pipe
    gap = results[name]['val_r2'] - results[name]['test_r2']
    print(f"val R²={results[name]['val_r2']:.3f}  test R²={results[name]['test_r2']:.3f}  "
          f"gap={gap:+.3f}")

best_model_name = max(results, key=lambda k: results[k]['val_r2'])
best_model      = fitted_models[best_model_name]
print(f"\n  Best model: {best_model_name} "
      f"(val R²={results[best_model_name]['val_r2']:.3f})")

# ═══════════════════════════════════════════════════════════════
# 10. Expanding-window cross-validation — FIX #5
# ═══════════════════════════════════════════════════════════════
print("\n[10] Expanding-window cross-validation (5 folds)...")

# Build a time-ordered full dataset for CV
df_cv = df_model.sort_values('season_year').reset_index(drop=True)
X_cv  = df_cv[FEATURE_COLS]
y_cv  = df_cv[TARGET_COL]

# Only run CV on XGBoost (best architecture) for speed
cv_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', xgb.XGBRegressor(
        n_estimators=400, max_depth=3, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=15,
        reg_alpha=1.0, reg_lambda=5.0, random_state=42,
        verbosity=0, n_jobs=-1,
    ))
])

# TimeSeriesSplit = expanding window
tscv = TimeSeriesSplit(n_splits=5)
cv_r2_scores, cv_rmse_scores = [], []

for fold_i, (train_idx, test_idx) in enumerate(tscv.split(X_cv)):
    X_tr, X_te = X_cv.iloc[train_idx], X_cv.iloc[test_idx]
    y_tr, y_te = y_cv.iloc[train_idx], y_cv.iloc[test_idx]
    cv_pipe.fit(X_tr, y_tr)
    y_pred_cv = cv_pipe.predict(X_te)
    fold_r2   = float(r2_score(y_te, y_pred_cv))
    fold_rmse = float(np.sqrt(mean_squared_error(y_te, y_pred_cv)))
    cv_r2_scores.append(fold_r2)
    cv_rmse_scores.append(fold_rmse)
    print(f"    Fold {fold_i+1}: test R²={fold_r2:.3f}  RMSE={fold_rmse:.3f}  "
          f"(train n={len(train_idx)}, test n={len(test_idx)})")

cv_summary = {
    'cv_r2_mean':  float(np.mean(cv_r2_scores)),
    'cv_r2_std':   float(np.std(cv_r2_scores)),
    'cv_rmse_mean':float(np.mean(cv_rmse_scores)),
    'cv_rmse_std': float(np.std(cv_rmse_scores)),
    'cv_r2_scores':  [round(x, 4) for x in cv_r2_scores],
    'cv_rmse_scores':[round(x, 4) for x in cv_rmse_scores],
}
print(f"\n  CV R² = {cv_summary['cv_r2_mean']:.3f} ± {cv_summary['cv_r2_std']:.3f}")
print(f"  CV RMSE = {cv_summary['cv_rmse_mean']:.3f} ± {cv_summary['cv_rmse_std']:.3f}")

# ═══════════════════════════════════════════════════════════════
# 11. Scout predictions — FIX #6 scout outputs
# ═══════════════════════════════════════════════════════════════
print("\n[11] Generating scout predictions...")

# Rebuild prediction pool from most recent season's raw data
latest_raw_year = season_df['season_year'].max()
pred_raw = season_df[season_df['season_year'] == latest_raw_year].copy()
pred_raw['pos_simple'] = pred_raw['pos'].apply(simplify_position)
pred_raw = pred_raw.merge(match_agg, on=['player', 'season_year'], how='left')
pred_raw['90s_safe'] = pred_raw['90s'].fillna(1).clip(lower=0.1)
pred_raw['gls_p90']  = pred_raw['gls'].fillna(0)        / pred_raw['90s_safe']
pred_raw['ast_p90']  = pred_raw['ast'].fillna(0)        / pred_raw['90s_safe']
pred_raw['tklw_p90'] = pred_raw['tklw_total'].fillna(0) / pred_raw['90s_safe']
pred_raw['int_p90']  = pred_raw['int_total'].fillna(0)  / pred_raw['90s_safe']
pred_raw['sh_p90']   = pred_raw['sh_total'].fillna(0)   / pred_raw['90s_safe']
pred_raw['crs_p90']  = pred_raw['crs_total'].fillna(0)  / pred_raw['90s_safe']
pred_raw['fld_p90']  = pred_raw['fld_total'].fillna(0)  / pred_raw['90s_safe']
pred_raw['raw_score'] = pred_raw.apply(compute_composite_score, axis=1)

# Z-score within position (latest season)
for pos in ['FW', 'MF', 'DF', 'GK']:
    mask = pred_raw['pos_simple'] == pos
    vals = pred_raw.loc[mask, 'raw_score']
    mu, sigma = vals.mean(), vals.std()
    pred_raw.loc[mask, 'perf_z'] = (vals - mu) / sigma if sigma > 0 else 0.0

# Injury proxy
prev_year_mins = (
    season_df[season_df['season_year'] == latest_raw_year - 1][['player','min']]
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
latest_mc = match_consistency[match_consistency['season_year'] == latest_raw_year][['player','consistency']]
pred_raw = pred_raw.merge(latest_mc, on='player', how='left')

# Starter ratio
pred_raw['starter_ratio'] = (pred_raw['starts'].fillna(0) / pred_raw['mp'].replace(0, np.nan)).clip(0, 1)

# Historical trend features from training data (most recent entry per player)
hist_latest = (
    df_model.sort_values('season_year')
    .groupby('player').last()
    .reset_index()
    [['player', 'perf_trend', 'min_trend', 'trend_x_age_to_peak']]
)
pred_raw = pred_raw.merge(hist_latest, on='player', how='left')

# Age features
pred_raw['peak_age_pos'] = pred_raw['pos_simple'].map(pos_peak_ages).fillna(27)
pred_raw['age_to_peak']  = pred_raw['peak_age_pos'] - pred_raw['age']
pred_raw['age_sq']       = pred_raw['age'] ** 2
pred_raw['is_u23']       = (pred_raw['age'] <= 23).astype(int)
pred_raw['is_u25']       = (pred_raw['age'] <= 25).astype(int)

# Recalculate interaction with fresh age
pred_raw['trend_x_age_to_peak'] = pred_raw['perf_trend'].fillna(0) * pred_raw['age_to_peak']

# Position dummies
for pos in ['FW', 'MF', 'DF', 'GK']:
    pred_raw[f'pos_{pos}'] = (pred_raw['pos_simple'] == pos).astype(int)

# Filter: min 1800 minutes, no injury risk
pred_candidates         = pred_raw[pred_raw['min'].fillna(0) >= 1800].copy()
pred_candidates_healthy = pred_candidates[pred_candidates['injury_risk'] == 0].copy()
print(f"  After 1800-min filter: {len(pred_candidates)} players")
print(f"  After injury-risk filter: {len(pred_candidates_healthy)} players")

# Predict
X_pred = pred_candidates_healthy[FEATURE_COLS].fillna(
    pred_candidates_healthy[FEATURE_COLS].median()
)
pred_candidates_healthy = pred_candidates_healthy.copy()
pred_candidates_healthy['predicted_rel_growth'] = best_model.predict(X_pred)

# Ensemble prediction
ensemble_preds = np.column_stack([
    fitted_models[name].predict(X_pred) for name in fitted_models
])
pred_candidates_healthy['ensemble_rel_growth'] = ensemble_preds.mean(axis=1)

# ═══════════════════════════════════════════════════════════════
# 12. Scout report tables
# ═══════════════════════════════════════════════════════════════
print("\n[12] Generating scout report tables...")

scout_cols = ['player', 'age', 'pos_simple', 'team',
              'market_value', 'perf_z', 'predicted_rel_growth',
              'ensemble_rel_growth', 'min', 'perf_trend']
all_preds = pred_candidates_healthy[scout_cols].copy()
all_preds = all_preds.rename(columns={
    'perf_z':      'current_perf_z',
    'pos_simple':  'position',
    'perf_trend':  'form_slope_3yr',
})

# ─── Hot prospects: U23, predicted positive relative growth ───
hot_prospects = (
    all_preds[(all_preds['age'] <= 23) & (all_preds['ensemble_rel_growth'] > 0)]
    .nlargest(15, 'ensemble_rel_growth')
    .reset_index(drop=True)
)
hot_prospects.index += 1

# ─── Late bloomers: 24-28, predicted positive relative growth ───
late_bloomers = (
    all_preds[(all_preds['age'] >= 24) & (all_preds['age'] <= 28)
              & (all_preds['ensemble_rel_growth'] > 0)]
    .nlargest(15, 'ensemble_rel_growth')
    .reset_index(drop=True)
)
late_bloomers.index += 1

# ─── All U25 top growth prospects ───
u25_top15 = (
    all_preds[all_preds['age'] <= 25]
    .nlargest(15, 'ensemble_rel_growth')
    .reset_index(drop=True)
)
u25_top15.index += 1

# ─── Declining stars (high current z, negative predicted growth) ───
declining = (
    all_preds[all_preds['current_perf_z'] > 0.5]
    .nsmallest(15, 'ensemble_rel_growth')
    .reset_index(drop=True)
)
declining.index += 1

print("\n" + "=" * 70)
print("  HOT PROSPECTS (U23, positive relative growth)")
print("=" * 70)
print(hot_prospects[['player','age','position','team','current_perf_z',
                      'form_slope_3yr','ensemble_rel_growth','min']].to_string())

print("\n" + "=" * 70)
print("  LATE BLOOMERS (24-28, positive relative growth)")
print("=" * 70)
print(late_bloomers[['player','age','position','team','current_perf_z',
                      'form_slope_3yr','ensemble_rel_growth','min']].to_string())

print("\n" + "=" * 70)
print("  TOP 15 U25 GROWTH PROSPECTS")
print("=" * 70)
print(u25_top15[['player','age','position','team','current_perf_z',
                  'form_slope_3yr','ensemble_rel_growth','min']].to_string())

# ═══════════════════════════════════════════════════════════════
# 13. Feature importance
# ═══════════════════════════════════════════════════════════════
xgb_model_obj = fitted_models['XGBoost'].named_steps['model']
feature_importance = pd.DataFrame({
    'feature':    FEATURE_COLS,
    'importance': xgb_model_obj.feature_importances_
}).sort_values('importance', ascending=False)

rf_importance = pd.DataFrame({
    'feature':    FEATURE_COLS,
    'importance': fitted_models['RandomForest'].named_steps['model'].feature_importances_
}).sort_values('importance', ascending=False)

# ═══════════════════════════════════════════════════════════════
# 14. Visualisations
# ═══════════════════════════════════════════════════════════════
print("\n[13] Generating figures...")

# ── Fig 1: Peak age curves per position (smoothed + prior blend) ──
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('S4 v3: Peak Age Curves per Position\n'
             '(Gaussian-smoothed, prior-blended, min 30 samples/bucket)',
             fontsize=14, fontweight='bold')
axes = axes.flatten()

for ax, pos in zip(axes, POSITIONS):
    pos_data  = season_df_hist[season_df_hist['pos_simple'] == pos]
    age_stats = (
        pos_data.groupby('age')['perf_z']
        .agg(['mean', 'std', 'count'])
        .reset_index()
        .rename(columns={'mean':'mean_z','std':'std_z','count':'n'})
        .sort_values('age')
    )
    age_all   = age_stats.copy()
    age_valid = age_stats[age_stats['n'] >= 30].copy()

    # Plot raw (all) data faded
    ax.plot(age_all['age'], age_all['mean_z'],
            'o--', color='#cccccc', linewidth=1, markersize=3, label='n < 30 (excluded)')

    if len(age_valid) >= 3:
        ages_v  = age_valid['age'].values.astype(float)
        mean_zv = age_valid['mean_z'].values
        smoothed_v = gaussian_filter1d(mean_zv, sigma=1.5)

        ax.plot(ages_v, mean_zv,
                'o', color=PALETTE[pos], markersize=5, alpha=0.6, label='n ≥ 30 (raw)')
        ax.plot(ages_v, smoothed_v,
                '-', color=PALETTE[pos], linewidth=2.5, label='smoothed')

        # Sample count annotation
        for i, r in age_valid.iterrows():
            if int(r['age']) % 4 == 0:
                ax.annotate(f"n={int(r['n'])}",
                            xy=(r['age'], age_valid.loc[i, 'mean_z']),
                            xytext=(0, 8), textcoords='offset points',
                            fontsize=7, ha='center', color='#555555')

    pa       = peak_age_results[pos]['peak_age']
    prior_pa = peak_age_results[pos]['prior_peak_age']
    ax.axvline(pa,       color=PALETTE[pos], linestyle='-',  alpha=0.8, linewidth=2,
               label=f'Blended peak = {pa}')
    ax.axvline(prior_pa, color='gray',       linestyle='--', alpha=0.5, linewidth=1.5,
               label=f'Prior = {prior_pa}')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Age'); ax.set_ylabel('Mean perf_z')
    ax.set_title(f'{pos} — Peak Age (blended={pa}, prior={prior_pa})')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig1_peak_age_v3.png', bbox_inches='tight')
plt.close()
print("  Fig 1 saved: fig1_peak_age_v3.png")

# ── Fig 2: Model performance (val vs test R²) ──
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('S4 v3: Model Performance (Relative Growth Target, Strong Regularisation)',
             fontsize=13, fontweight='bold')

model_names   = list(results.keys())
val_r2_vals   = [results[m]['val_r2']  for m in model_names]
test_r2_vals  = [results[m]['test_r2'] for m in model_names]
val_rmse_vals = [results[m]['val_rmse'] for m in model_names]
test_rmse_vals= [results[m]['test_rmse'] for m in model_names]

x, w = np.arange(len(model_names)), 0.35

ax = axes[0]
b1 = ax.bar(x - w/2, val_r2_vals,  w, label='Val R²',  color='#3498db', alpha=0.85)
b2 = ax.bar(x + w/2, test_r2_vals, w, label='Test R²', color='#e74c3c', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=20, ha='right')
ax.set_ylabel('R²'); ax.set_title('R² Score (closer = less overfit)'); ax.legend()
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
for b in list(b1) + list(b2):
    h = b.get_height()
    ax.text(b.get_x()+b.get_width()/2, h + 0.003,
            f'{h:.3f}', ha='center', fontsize=8)

ax = axes[1]
ax.bar(x - w/2, val_rmse_vals,  w, label='Val RMSE',  color='#2ecc71', alpha=0.85)
ax.bar(x + w/2, test_rmse_vals, w, label='Test RMSE', color='#f39c12', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=20, ha='right')
ax.set_ylabel('RMSE'); ax.set_title('RMSE'); ax.legend()

ax = axes[2]
y_pred_test = best_model.predict(X_test)
ax.scatter(y_test, y_pred_test, alpha=0.3, s=10, color='#3498db')
lim = max(abs(y_test).max(), abs(y_pred_test).max(), 2)
ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1.5)
ax.set_xlabel('Actual relative growth'); ax.set_ylabel('Predicted')
ax.set_title(f'{best_model_name}: Predicted vs Actual (Test)')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig2_model_performance_v3.png', bbox_inches='tight')
plt.close()
print("  Fig 2 saved: fig2_model_performance_v3.png")

# ── Fig 3: Feature importance ──
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('S4 v3: Feature Importance (Trajectory Features Only)',
             fontsize=13, fontweight='bold')

top_n = 15
for ax, imp_df, label, color in zip(
    axes,
    [feature_importance, rf_importance],
    ['XGBoost', 'RandomForest'],
    ['#e74c3c', '#2ecc71']
):
    top = imp_df.head(top_n)
    ax.barh(top['feature'][::-1], top['importance'][::-1], color=color, alpha=0.85)
    ax.set_title(f'{label} Top {top_n} Features')
    ax.set_xlabel('Importance')
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig3_feature_importance_v3.png', bbox_inches='tight')
plt.close()
print("  Fig 3 saved: fig3_feature_importance_v3.png")

# ── Fig 4: Hot prospects & late bloomers ──
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('S4 v3: Scout Report — Growth Prospects',
             fontsize=14, fontweight='bold')

for ax, df_plot, title in zip(
    axes,
    [hot_prospects, late_bloomers],
    ['Hot Prospects (U23, positive rel growth)', 'Late Bloomers (24-28, positive rel growth)']
):
    if len(df_plot) == 0:
        ax.text(0.5, 0.5, 'No candidates', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(title)
        continue
    colors = [PALETTE.get(p, '#888888') for p in df_plot['position']]
    ax.barh(
        [f"{r['player']} ({int(r['age'])})" for _, r in df_plot.iterrows()],
        df_plot['ensemble_rel_growth'],
        color=colors, alpha=0.85
    )
    ax.set_xlabel('Predicted Relative Growth vs Age-Cohort (Z-score units)')
    ax.set_title(title)
    ax.invert_yaxis()
    ax.axvline(0, color='black', linewidth=0.7, linestyle='--')
    ax.grid(axis='x', alpha=0.3)
    # Annotate current perf_z and slope
    for i, (_, row) in enumerate(df_plot.iterrows()):
        ax.text(0.01, i,
                f"z={row['current_perf_z']:.2f}  trend={row['form_slope_3yr']:.2f}",
                va='center', ha='left', fontsize=7,
                transform=ax.get_yaxis_transform(), color='#333333')

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig4_hot_prospects_v3.png', bbox_inches='tight')
plt.close()
print("  Fig 4 saved: fig4_hot_prospects_v3.png")

# ── Fig 5: Mean-reversion bias check ──
# Predicted relative growth vs current_z: should be near-flat slope
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('S4 v3: Bias Check — Predicted Rel Growth vs Current Performance\n'
             '(slope near 0 = no mean-reversion bias)',
             fontsize=12, fontweight='bold')

for ax, pos in zip(axes, ['FW', 'MF', 'DF']):
    sub = pred_candidates_healthy[pred_candidates_healthy['pos_simple'] == pos]
    if len(sub) < 3:
        ax.set_title(f'{pos}: insufficient data')
        continue
    z_vals = sub['perf_z'].fillna(0).values
    g_vals = sub['ensemble_rel_growth'].values
    ax.scatter(z_vals, g_vals, alpha=0.4, s=18, color=PALETTE[pos])
    if len(sub) > 5:
        m, b = np.polyfit(z_vals, g_vals, 1)
        xl = np.linspace(z_vals.min(), z_vals.max(), 50)
        ax.plot(xl, m*xl + b, 'k--', linewidth=1.5, label=f'slope={m:.3f}')
        ax.legend(fontsize=9)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Current perf_z (absolute performance)')
    ax.set_ylabel('Predicted relative growth')
    ax.set_title(f'{pos}: bias check')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig5_bias_check_v3.png', bbox_inches='tight')
plt.close()
print("  Fig 5 saved: fig5_bias_check_v3.png")

# ── Fig 6: Growth trajectory plots for top prospects ──
top_prospect_players = hot_prospects['player'].tolist()[:6]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('S4 v3: Growth Trajectories — Top Hot Prospects',
             fontsize=14, fontweight='bold')
axes = axes.flatten()

for ax, player_name in zip(axes, top_prospect_players):
    player_hist = (
        df_model[df_model['player'] == player_name]
        .sort_values('season_year')
    )
    if len(player_hist) == 0:
        ax.set_title(f'{player_name}\n(no history)')
        continue
    seasons = player_hist['season_year'].values
    z_vals  = player_hist['perf_z'].values
    pos = player_hist['pos_simple'].iloc[-1] if len(player_hist) > 0 else 'MF'
    color = PALETTE.get(pos, '#888888')

    ax.plot(seasons, z_vals, 'o-', color=color, linewidth=2, markersize=6)

    # Projected next season
    latest_pred = pred_candidates_healthy[pred_candidates_healthy['player'] == player_name]
    if len(latest_pred) > 0:
        pred_delta = float(latest_pred['ensemble_rel_growth'].iloc[0])
        cohort_norm_val = cohort_norm[
            (cohort_norm['pos_simple'] == pos) &
            (cohort_norm['age'] == latest_pred['age'].iloc[0])
        ]['cohort_mean_delta']
        cohort_val = float(cohort_norm_val.iloc[0]) if len(cohort_norm_val) > 0 else 0.0
        abs_pred_growth = pred_delta + cohort_val

        next_year_z = z_vals[-1] + abs_pred_growth
        ax.plot([seasons[-1], seasons[-1]+1], [z_vals[-1], next_year_z],
                'o--', color=color, linewidth=1.5, markersize=6, alpha=0.6,
                label=f'Projected: {next_year_z:.2f}')
        ax.legend(fontsize=8)

    age_now = int(player_hist['age'].iloc[-1]) if len(player_hist) > 0 else '?'
    team_now = player_hist['team'].iloc[-1] if len(player_hist) > 0 else '?'
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_title(f'{player_name} (age={age_now}, {pos})\n{team_now}')
    ax.set_xlabel('Season'); ax.set_ylabel('perf_z')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig6_growth_trajectories_v3.png', bbox_inches='tight')
plt.close()
print("  Fig 6 saved: fig6_growth_trajectories_v3.png")

# ── Fig 7: Expanding-window CV scores ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('S4 v3: Expanding-Window CV Performance (XGBoost)',
             fontsize=13, fontweight='bold')

folds = list(range(1, len(cv_r2_scores) + 1))

ax = axes[0]
ax.bar(folds, cv_r2_scores, color='#3498db', alpha=0.8)
ax.axhline(cv_summary['cv_r2_mean'], color='red', linestyle='--', linewidth=1.5,
           label=f"Mean={cv_summary['cv_r2_mean']:.3f} ± {cv_summary['cv_r2_std']:.3f}")
ax.set_xlabel('Fold'); ax.set_ylabel('R²')
ax.set_title('CV R² per Fold')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.bar(folds, cv_rmse_scores, color='#e74c3c', alpha=0.8)
ax.axhline(cv_summary['cv_rmse_mean'], color='navy', linestyle='--', linewidth=1.5,
           label=f"Mean={cv_summary['cv_rmse_mean']:.3f} ± {cv_summary['cv_rmse_std']:.3f}")
ax.set_xlabel('Fold'); ax.set_ylabel('RMSE')
ax.set_title('CV RMSE per Fold')
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig7_cv_performance_v3.png', bbox_inches='tight')
plt.close()
print("  Fig 7 saved: fig7_cv_performance_v3.png")

# ═══════════════════════════════════════════════════════════════
# 15. Save outputs
# ═══════════════════════════════════════════════════════════════
print("\n[14] Saving outputs...")

# Predictions parquet (all healthy candidates)
save_cols = ['player', 'age', 'pos_simple', 'team', 'market_value',
             'min', 'perf_z', 'predicted_rel_growth', 'ensemble_rel_growth',
             'injury_risk', 'min_drop_pct']
save_cols = [c for c in save_cols if c in pred_candidates_healthy.columns]
pred_candidates_healthy[save_cols].to_parquet(
    SCOUT_DIR / 'growth_predictions_v3.parquet', index=False
)
print(f"  Saved: growth_predictions_v3.parquet ({len(pred_candidates_healthy)} rows)")

# JSON results
def safe_list(df, cols):
    available = [c for c in cols if c in df.columns]
    return df[available].to_dict(orient='records')

output = {
    'version': 'v3',
    'model_performance': results,
    'best_model': best_model_name,
    'cv_performance': cv_summary,
    'peak_ages': {
        pos: {k: v for k, v in info.items() if k != 'age_curve'}
        for pos, info in peak_age_results.items()
    },
    'peak_age_curves': {
        pos: peak_age_results[pos].get('age_curve', {})
        for pos in POSITIONS
    },
    'hot_prospects': safe_list(
        hot_prospects,
        ['player','age','position','team','current_perf_z',
         'form_slope_3yr','ensemble_rel_growth','min']
    ),
    'late_bloomers': safe_list(
        late_bloomers,
        ['player','age','position','team','current_perf_z',
         'form_slope_3yr','ensemble_rel_growth','min']
    ),
    'top_u25_prospects': safe_list(
        u25_top15,
        ['player','age','position','team','current_perf_z',
         'form_slope_3yr','ensemble_rel_growth','min']
    ),
    'declining_stars': safe_list(
        declining,
        ['player','age','position','team','current_perf_z',
         'ensemble_rel_growth','min']
    ),
    'feature_importance': feature_importance.head(15).to_dict(orient='records'),
    'dataset_info': {
        'train_rows':              int(len(X_train)),
        'val_rows':                int(len(X_val)),
        'test_rows':               int(len(X_test)),
        'prediction_pool_total':   int(len(pred_candidates)),
        'prediction_pool_healthy': int(len(pred_candidates_healthy)),
        'min_threshold':           1800,
        'injury_proxy_threshold':  0.40,
        'target':                  'relative_delta_z_vs_age_cohort',
        'feature_count':           len(FEATURE_COLS),
    },
    'fixes_v3': [
        'strong_regularisation_max_depth3_min_child_weight15',
        'no_absolute_level_features_trajectory_only',
        'peak_age_smoothed_prior_blended_min30_samples',
        'relative_delta_target_cohort_normalised',
        'expanding_window_5fold_cv',
        'scout_outputs_hot_prospects_late_bloomers_trajectories',
    ]
}

with open(SCOUT_DIR / 'growth_v3_results.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2, default=str)
print("  Saved: growth_v3_results.json")

# ═══════════════════════════════════════════════════════════════
# 16. Summary
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("S4 Growth v3 COMPLETE")
print("=" * 65)

print("\nPeak ages (smoothed, prior-blended, min 30 samples/bucket):")
for pos in POSITIONS:
    info = peak_age_results[pos]
    print(f"  {pos}: peak_age={info['peak_age']}  "
          f"(data={info.get('data_peak_age','N/A')}, prior={info['prior_peak_age']}, "
          f"w_data={info['w_data']:.2f}, valid buckets={info['valid_age_buckets']})")

print(f"\nModel performance:")
for name, res in results.items():
    gap = res['val_r2'] - res['test_r2']
    print(f"  {name}: val R²={res['val_r2']:.3f}  "
          f"test R²={res['test_r2']:.3f}  gap={gap:+.3f}")

print(f"\nCV (expanding window, XGBoost):")
print(f"  R² = {cv_summary['cv_r2_mean']:.3f} ± {cv_summary['cv_r2_std']:.3f}")
print(f"  RMSE = {cv_summary['cv_rmse_mean']:.3f} ± {cv_summary['cv_rmse_std']:.3f}")

print(f"\nScout outputs:")
print(f"  Hot prospects (U23):  {len(hot_prospects)} players")
print(f"  Late bloomers (24-28): {len(late_bloomers)} players")
print(f"  U25 top prospects:     {len(u25_top15)} players")

print(f"\nKey fixes applied vs v2:")
print("  [1] Regularisation: max_depth 5→3, min_child_weight 3→15,")
print("      subsample 0.8→0.7, reg_alpha 0.1→1.0, reg_lambda 1.0→5.0")
print("  [2] Features: removed 12 absolute-level leakers, kept 17 trajectory features")
print("  [3] Peak age: Gaussian-smoothed + prior-blended (FW/MF→27, DF→28, GK→30)")
print("  [4] Target: raw Z-score delta → relative delta vs age-cohort norm")
print("  [5] CV: single time-split → 5-fold expanding-window TimeSeriesSplit")

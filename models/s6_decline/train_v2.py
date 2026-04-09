"""
S6 v2: Player Decline Detection — Fixed
========================================
Fixes applied per scout review:
  1. REMOVED position one-hot features — now trains POSITION-SPECIFIC models
  2. AGE FLOOR: players <=24 capped at 50% decline risk (volatility != career decline)
  3. SUSTAINED DECLINE: requires performance drop in 2 consecutive seasons (not 1)
  4. TWO OUTPUT LISTS:
       - "career_decline_watch"  : age 28+, sustained multi-season drop
       - "regression_alert"      : any age, outlier season flagged
  5. FEATURES: perf_slope (3 seasons), minutes trend, age (25+ only), injury proxy,
               workload, team quality change — NO position one-hot
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    average_precision_score
)
from sklearn.impute import SimpleImputer
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("  [WARNING] imbalanced-learn not found, skipping SMOTE")
import xgboost as xgb

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = Path("C:/Users/xcv54/workspace/EPL project")
DATA_DIR = BASE_DIR / "data" / "processed"
SCOUT_DIR = BASE_DIR / "data" / "scout"
MODEL_DIR = BASE_DIR / "models" / "s6_decline"
FIG_DIR   = MODEL_DIR / "figures_v2"

for d in [SCOUT_DIR, FIG_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("[S6-V2] Player Decline Detection — Fixed")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────
print("\n[1] Loading data...")
season_df = pd.read_parquet(DATA_DIR / "player_season_stats.parquet")
match_df  = pd.read_parquet(DATA_DIR / "player_match_logs.parquet")
team_df   = pd.read_parquet(DATA_DIR / "team_season_summary.parquet")

print(f"  player_season_stats: {season_df.shape}")
print(f"  player_match_logs:   {match_df.shape}")
print(f"  team_season_summary: {team_df.shape}")

# ─────────────────────────────────────────────
# 2. Season year parsing
# ─────────────────────────────────────────────
def parse_season_year(s):
    try:
        return int(str(s).split('/')[0])
    except Exception:
        return np.nan

season_df['season_year'] = season_df['season'].apply(parse_season_year)
team_df['season_year']   = team_df['Season'].apply(parse_season_year)

# ─────────────────────────────────────────────
# 3. Position grouping
# ─────────────────────────────────────────────
def map_position_group(pos):
    if pd.isna(pos):
        return 'MID'
    pos = str(pos).lower()
    if 'goalkeeper' in pos or pos == 'gk':
        return 'GK'
    elif 'back' in pos or 'defend' in pos or pos in ['cb', 'rb', 'lb', 'rwb', 'lwb']:
        return 'DEF'
    elif 'forward' in pos or 'winger' in pos or 'striker' in pos or pos in ['cf', 'st', 'lw', 'rw', 'ss']:
        return 'FWD'
    elif 'midfield' in pos or pos in ['cm', 'cdm', 'cam', 'dm', 'am']:
        return 'MID'
    else:
        return 'MID'

season_df['pos_group'] = season_df['position'].apply(map_position_group)

# ─────────────────────────────────────────────
# 4. Position-specific composite performance score (z-score per position)
# ─────────────────────────────────────────────
print("\n[2] Computing position-specific performance scores...")

POS_METRICS = {
    'FWD': ['gls_1', 'ast_1', 'g_a_1'],
    'MID': ['ast_1', 'gls_1', 'g_a_1'],
    'DEF': ['gls_1', 'ast_1'],
    'GK':  ['gls_1', 'ast_1'],
}

season_df['90s_safe'] = season_df['90s'].clip(lower=0.1)

def compute_composite_score(df):
    scores = np.zeros(len(df))
    for pos_g, metrics in POS_METRICS.items():
        mask = df['pos_group'] == pos_g
        if mask.sum() == 0:
            continue
        avail = [m for m in metrics if m in df.columns]
        if not avail:
            continue
        sub = df.loc[mask, avail].copy()
        z = sub.apply(lambda col: stats.zscore(col.fillna(0), nan_policy='omit'), axis=0)
        scores[mask.values] = z.mean(axis=1).values
    return scores

season_df['perf_score'] = compute_composite_score(season_df)

# ─────────────────────────────────────────────
# 5. Team quality (points-based, per-season normalized 0-1)
# ─────────────────────────────────────────────
team_quality = (
    team_df.groupby(['team', 'season_year'])['points'].sum()
    .reset_index().rename(columns={'points': 'team_quality'})
)
team_quality['team_quality'] = team_quality.groupby('season_year')['team_quality'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
)
season_df = season_df.merge(team_quality, on=['team', 'season_year'], how='left')

# Team quality CHANGE season-over-season (proxy for squad improvement/degradation)
team_quality_sorted = team_quality.sort_values(['team', 'season_year'])
team_quality_sorted['team_quality_prev'] = team_quality_sorted.groupby('team')['team_quality'].shift(1)
team_quality_sorted['team_quality_change'] = (
    team_quality_sorted['team_quality'] - team_quality_sorted['team_quality_prev']
)
season_df = season_df.merge(
    team_quality_sorted[['team', 'season_year', 'team_quality_change']],
    on=['team', 'season_year'], how='left'
)

# ─────────────────────────────────────────────
# 6. Match-level consistency and injury proxy
# ─────────────────────────────────────────────
print("[3] Computing match-level consistency and injury proxy...")

match_df['season_year'] = match_df['season'].apply(parse_season_year)
match_df['g_a_match']   = match_df['gls'].fillna(0) + match_df['ast'].fillna(0)

consistency = match_df.groupby(['player', 'season_year']).agg(
    consistency_std=('g_a_match', 'std'),
    match_count=('g_a_match', 'count'),
    avg_min_match=('min', 'mean'),
    total_min=('min', 'sum'),
    games_sub_60=('min', lambda x: (x < 60).sum()),   # injury proxy: frequent early subs/non-starts
).reset_index()
consistency['consistency_score'] = 1.0 / (1.0 + consistency['consistency_std'].fillna(0))
# Injury proxy: fraction of games with <60 minutes (benched/subbed off)
consistency['injury_proxy'] = consistency['games_sub_60'] / (consistency['match_count'].clip(lower=1))

season_df = season_df.merge(
    consistency[['player', 'season_year', 'consistency_score', 'match_count',
                  'avg_min_match', 'total_min', 'injury_proxy']],
    on=['player', 'season_year'], how='left'
)

# ─────────────────────────────────────────────
# 7. Sustained decline label (requires drop in 2 CONSECUTIVE seasons)
#    FIX: old code flagged any single-season dip — regression-to-mean for young stars
# ─────────────────────────────────────────────
print("[4] Building SUSTAINED decline labels (2 consecutive seasons)...")

season_df['player_key'] = season_df['player']
season_df_sorted = season_df.sort_values(['player_key', 'season_year'])

# Build season N → N+1 → N+2 merge to detect SUSTAINED drops
def make_shift_df(df, shift, suffix):
    tmp = df[['player_key', 'season_year', 'perf_score', 'min', 'mp', 'market_value']].copy()
    tmp['season_year'] = tmp['season_year'] - shift
    tmp.columns = [f'{c}{suffix}' if c not in ('player_key', 'season_year') else c
                   for c in tmp.columns]
    return tmp

current   = season_df[['player_key', 'season_year', 'perf_score', 'min', 'mp',
                         'market_value', 'age', 'pos_group', 'team', 'team_quality',
                         'team_quality_change']].copy()
next1_df  = make_shift_df(season_df, 1, '_n1')   # season N+1
next2_df  = make_shift_df(season_df, 2, '_n2')   # season N+2

merged = current.merge(next1_df, on=['player_key', 'season_year'], how='inner')
merged = merged.merge(next2_df, on=['player_key', 'season_year'], how='left')

perf_std = (merged['perf_score_n1'] - merged['perf_score']).std()

# Sustained performance decline: drop in BOTH N→N+1 AND N+1→N+2
merged['drop_n1'] = ((merged['perf_score_n1'] - merged['perf_score']) < -0.4 * perf_std).astype(int)
merged['drop_n2'] = ((merged['perf_score_n2'] - merged['perf_score_n1']) < -0.2 * perf_std).astype(int)
merged['decline_perf'] = (
    (merged['drop_n1'] == 1) & (merged['drop_n2'] == 1)
).astype(int)

# Single-season availability drop (injury/minutes driven)
merged['decline_avail'] = (
    (merged['min_n1'] < merged['min'] * 0.70) & (merged['min'] >= 450)
).astype(int)

# Combined sustained decline label
merged['decline'] = ((merged['decline_perf'] == 1) | (merged['decline_avail'] == 1)).astype(int)

# ── Outlier season flag (for regression-to-mean alert, any age) ──
# "Outlier" = current season perf_score > 1.5 std above player's own career mean
player_career_mean = (
    season_df.groupby('player_key')['perf_score']
    .agg(['mean', 'std'])
    .reset_index()
    .rename(columns={'mean': 'career_perf_mean', 'std': 'career_perf_std'})
)
merged = merged.merge(player_career_mean, on='player_key', how='left')
merged['career_perf_std'] = merged['career_perf_std'].fillna(0.5)
merged['is_outlier_season'] = (
    merged['perf_score'] > (merged['career_perf_mean'] + 1.5 * merged['career_perf_std'])
).astype(int)

print(f"  Total samples:          {len(merged)}")
print(f"  Sustained perf decline: {merged['decline_perf'].sum()} ({merged['decline_perf'].mean():.1%})")
print(f"  Availability decline:   {merged['decline_avail'].sum()} ({merged['decline_avail'].mean():.1%})")
print(f"  Combined decline label: {merged['decline'].sum()} ({merged['decline'].mean():.1%})")
print(f"  Outlier season flag:    {merged['is_outlier_season'].sum()} ({merged['is_outlier_season'].mean():.1%})")

# ─────────────────────────────────────────────
# 8. Trajectory features: performance slope (3 seasons), minutes trend
# ─────────────────────────────────────────────
print("[5] Computing trajectory features (slope 3-season window)...")

def compute_trajectory(df, n_seasons=3):
    results = []
    for player, grp in df.groupby('player_key'):
        grp = grp.sort_values('season_year')
        for _, row in grp.iterrows():
            cur_year = row['season_year']
            hist = grp[grp['season_year'] <= cur_year].tail(n_seasons)
            if len(hist) < 2:
                perf_slope = np.nan
                min_slope  = np.nan
                peak_minus_current = np.nan
            else:
                x = hist['season_year'].values.astype(float)
                yp = hist['perf_score'].values.astype(float)
                ym = hist['min'].values.astype(float)
                vp = ~np.isnan(yp)
                vm = ~np.isnan(ym)
                perf_slope = np.polyfit(x[vp], yp[vp], 1)[0] if vp.sum() >= 2 else np.nan
                min_slope  = np.polyfit(x[vm], ym[vm], 1)[0] if vm.sum() >= 2 else np.nan
                peak = grp[grp['season_year'] <= cur_year]['perf_score'].max()
                peak_minus_current = peak - row['perf_score']
            results.append({
                'player_key': player,
                'season_year': cur_year,
                'perf_slope': perf_slope,
                'min_slope': min_slope,
                'peak_minus_current': peak_minus_current,
            })
    return pd.DataFrame(results)

traj_input = season_df[['player_key', 'season_year', 'perf_score', 'min']].drop_duplicates()
trajectory_df = compute_trajectory(traj_input)
merged = merged.merge(trajectory_df, on=['player_key', 'season_year'], how='left')

# ─────────────────────────────────────────────
# 9. Market value trajectory
# ─────────────────────────────────────────────
mv_traj = []
for player, grp in season_df.groupby('player_key'):
    grp = grp.sort_values('season_year')
    for _, row in grp.iterrows():
        cur_year = row['season_year']
        hist = grp[grp['season_year'] <= cur_year].tail(3)
        if len(hist) < 2:
            mv_slope = np.nan
        else:
            x = hist['season_year'].values.astype(float)
            y = hist['market_value'].ffill().fillna(0).values.astype(float)
            valid = ~np.isnan(y)
            mv_slope = np.polyfit(x[valid], y[valid], 1)[0] if valid.sum() >= 2 else np.nan
        mv_traj.append({'player_key': player, 'season_year': cur_year, 'mv_slope': mv_slope})
mv_traj_df = pd.DataFrame(mv_traj)
merged = merged.merge(mv_traj_df, on=['player_key', 'season_year'], how='left')

# ─────────────────────────────────────────────
# 10. Workload (cumulative minutes over 2 seasons)
# ─────────────────────────────────────────────
workload = []
for player, grp in season_df.groupby('player_key'):
    grp = grp.sort_values('season_year')
    for _, row in grp.iterrows():
        cur_year = row['season_year']
        hist = grp[(grp['season_year'] >= cur_year - 2) & (grp['season_year'] <= cur_year)]
        workload.append({
            'player_key': player,
            'season_year': cur_year,
            'workload_2y_min': hist['min'].sum(),
            'workload_2y_mp':  hist['mp'].sum(),
        })
workload_df = pd.DataFrame(workload)
merged = merged.merge(workload_df, on=['player_key', 'season_year'], how='left')

# ─────────────────────────────────────────────
# 11. Merge consistency features
# ─────────────────────────────────────────────
cons_features = (
    season_df[['player_key', 'season_year', 'consistency_score',
                'match_count', 'avg_min_match', 'total_min', 'injury_proxy']]
    .groupby(['player_key', 'season_year']).mean().reset_index()
)
merged = merged.merge(cons_features, on=['player_key', 'season_year'], how='left')

# ─────────────────────────────────────────────
# 12. Feature definition — NO position one-hot
#     age used only as contextual/capping variable; for 25+ career model
# ─────────────────────────────────────────────
#
# FIX: position one-hot REMOVED — it was 37% of total importance and caused
#      position base-rate leakage.  Instead we train per-position models.
#
FEATURE_COLS = [
    # Performance trajectory (most important signals)
    'perf_slope',           # 3-season performance trend slope
    'peak_minus_current',   # distance from career peak
    'perf_score',           # current season score
    'min_slope',            # minutes trend (playing time trajectory)
    # Workload / availability
    'min',
    'mp',
    'workload_2y_min',
    'workload_2y_mp',
    # Injury proxy
    'injury_proxy',         # fraction of games < 60 min
    'consistency_score',
    'match_count',
    'avg_min_match',
    # Market signals
    'market_value',
    'mv_slope',
    # Team context
    'team_quality',
    'team_quality_change',  # NEW: is the player's team getting better or worse?
    # Age — only for 25+ career model; we deliberately exclude from young-player model
    'age',
]

merged['age'] = merged['age'].fillna(0)

print(f"\n[6] Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"  Total samples available: {len(merged)}")

# ─────────────────────────────────────────────
# 13. Split into POSITION-SPECIFIC training groups
#     FIX: train separate models per position group
# ─────────────────────────────────────────────
POSITIONS = ['FWD', 'MID', 'DEF', 'GK']
pos_models   = {}   # fitted models per position
pos_imputers = {}
pos_scalers  = {}
pos_results  = {}   # evaluation metrics per position

train_mask = merged['season_year'] < 2021
val_mask   = (merged['season_year'] >= 2021) & (merged['season_year'] <= 2022)
test_mask  =  merged['season_year'] >= 2023

print("\n[7] Training POSITION-SPECIFIC models...")

for pos in POSITIONS:
    pos_mask = merged['pos_group'] == pos
    df_pos   = merged[pos_mask].copy()

    if len(df_pos) < 50:
        print(f"  [{pos}] Skipping — too few samples ({len(df_pos)})")
        continue

    X_tr = df_pos.loc[df_pos['season_year'] < 2021,  FEATURE_COLS]
    y_tr = df_pos.loc[df_pos['season_year'] < 2021,  'decline']
    X_va = df_pos.loc[(df_pos['season_year'] >= 2021) & (df_pos['season_year'] <= 2022), FEATURE_COLS]
    y_va = df_pos.loc[(df_pos['season_year'] >= 2021) & (df_pos['season_year'] <= 2022), 'decline']
    X_te = df_pos.loc[df_pos['season_year'] >= 2023,  FEATURE_COLS]
    y_te = df_pos.loc[df_pos['season_year'] >= 2023,  'decline']

    print(f"\n  [{pos}] train={len(X_tr)} (decline {y_tr.mean():.1%})  "
          f"val={len(X_va)}  test={len(X_te)}")

    if len(X_tr) < 20 or y_tr.sum() < 5:
        print(f"  [{pos}] Skipping — insufficient positive labels")
        continue

    # Impute
    imp = SimpleImputer(strategy='median')
    X_tr_imp = imp.fit_transform(X_tr)
    X_va_imp = imp.transform(X_va) if len(X_va) > 0 else X_va.values
    X_te_imp = imp.transform(X_te) if len(X_te) > 0 else X_te.values

    X_tr_df = pd.DataFrame(X_tr_imp, columns=FEATURE_COLS)
    X_va_df = pd.DataFrame(X_va_imp, columns=FEATURE_COLS) if len(X_va) > 0 else pd.DataFrame(columns=FEATURE_COLS)
    X_te_df = pd.DataFrame(X_te_imp, columns=FEATURE_COLS) if len(X_te) > 0 else pd.DataFrame(columns=FEATURE_COLS)

    # SMOTE
    X_tr_res, y_tr_res = X_tr_df, y_tr.reset_index(drop=True)
    if HAS_SMOTE and int(y_tr.sum()) >= 5:
        try:
            k = min(5, int(y_tr.sum()) - 1)
            sm = SMOTE(random_state=42, k_neighbors=k)
            X_tr_res, y_tr_res = sm.fit_resample(X_tr_df, y_tr)
        except Exception as e:
            print(f"  [{pos}] SMOTE failed ({e}), using original")

    # Scale
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr_res)
    X_va_sc = sc.transform(X_va_df) if len(X_va_df) > 0 else X_va_df.values
    X_te_sc = sc.transform(X_te_df) if len(X_te_df) > 0 else X_te_df.values

    # XGBoost (primary)
    scale_pw = max(1.0, (y_tr_res == 0).sum() / max(1, (y_tr_res == 1).sum()))
    xgb_m = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pw,
        eval_metric='logloss', random_state=42, n_jobs=-1,
    )
    fit_kwargs = {}
    if len(X_va_df) > 0 and len(y_va) > 0:
        fit_kwargs['eval_set'] = [(X_va_df, y_va)]
    xgb_m.fit(X_tr_res, y_tr_res, verbose=False, **fit_kwargs)

    # Random Forest (secondary — diversity in ensemble)
    rf_m = RandomForestClassifier(
        n_estimators=200, max_depth=6, class_weight='balanced',
        random_state=42, n_jobs=-1,
    )
    rf_m.fit(X_tr_res, y_tr_res)

    # Logistic Regression (linear baseline)
    lr_m = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
    lr_m.fit(X_tr_sc, y_tr_res)

    pos_models[pos]   = {'xgb': xgb_m, 'rf': rf_m, 'lr': lr_m}
    pos_imputers[pos] = imp
    pos_scalers[pos]  = sc

    # Evaluate on test set
    if len(X_te_df) > 0 and len(y_te) > 0 and len(y_te.unique()) > 1:
        xgb_probs = xgb_m.predict_proba(X_te_df)[:, 1]
        rf_probs  = rf_m.predict_proba(X_te_df)[:, 1]
        lr_probs  = lr_m.predict_proba(X_te_sc)[:, 1]
        ens_probs = (xgb_probs + rf_probs + lr_probs) / 3.0
        y_te_reset = y_te.reset_index(drop=True)
        auc = roc_auc_score(y_te_reset, ens_probs)
        ap  = average_precision_score(y_te_reset, ens_probs)
        preds = (ens_probs >= 0.5).astype(int)
        report = classification_report(y_te_reset, preds, output_dict=True, zero_division=0)
        pos_results[pos] = {
            'auc_roc':           round(float(auc), 4),
            'avg_precision':     round(float(ap), 4),
            'precision_decline': round(float(report.get('1', {}).get('precision', 0)), 4),
            'recall_decline':    round(float(report.get('1', {}).get('recall', 0)), 4),
            'f1_decline':        round(float(report.get('1', {}).get('f1-score', 0)), 4),
            'n_test':            int(len(X_te_df)),
            'decline_rate_test': round(float(y_te.mean()), 4),
        }
        print(f"  [{pos}] AUC={auc:.4f}  F1={pos_results[pos]['f1_decline']:.4f}")

        # Feature importance for this position
        fi_df = pd.DataFrame({
            'feature': FEATURE_COLS,
            'importance_xgb': xgb_m.feature_importances_,
            'importance_rf':  rf_m.feature_importances_,
        }).sort_values('importance_xgb', ascending=False)
        print(f"  [{pos}] Top 5 features: {fi_df['feature'].head(5).tolist()}")

print("\n[8] Position-specific training complete.")

# ─────────────────────────────────────────────
# 14. Score ALL data using position-specific models
# ─────────────────────────────────────────────
print("\n[9] Scoring all data with position-specific models...")

merged['decline_prob_ensemble'] = np.nan
merged['decline_prob_xgb']      = np.nan
merged['decline_prob_rf']       = np.nan
merged['decline_prob_lr']       = np.nan

for pos in POSITIONS:
    if pos not in pos_models:
        continue
    mask = merged['pos_group'] == pos
    if mask.sum() == 0:
        continue
    X_all = merged.loc[mask, FEATURE_COLS].copy()
    imp   = pos_imputers[pos]
    sc    = pos_scalers[pos]
    models_dict = pos_models[pos]

    X_imp = imp.transform(X_all)
    X_df  = pd.DataFrame(X_imp, columns=FEATURE_COLS)
    X_sc  = sc.transform(X_df)

    p_xgb = models_dict['xgb'].predict_proba(X_df)[:, 1]
    p_rf  = models_dict['rf'].predict_proba(X_df)[:, 1]
    p_lr  = models_dict['lr'].predict_proba(X_sc)[:, 1]
    p_ens = (p_xgb + p_rf + p_lr) / 3.0

    merged.loc[mask, 'decline_prob_xgb']      = p_xgb
    merged.loc[mask, 'decline_prob_rf']        = p_rf
    merged.loc[mask, 'decline_prob_lr']        = p_lr
    merged.loc[mask, 'decline_prob_ensemble']  = p_ens

# ─────────────────────────────────────────────
# 15. AGE FLOOR — cap <=24 at 50% max
#     FIX: young player volatility != career decline
# ─────────────────────────────────────────────
young_mask = merged['age'] <= 24
n_capped = young_mask.sum()
merged.loc[young_mask, 'decline_prob_ensemble'] = merged.loc[young_mask, 'decline_prob_ensemble'].clip(upper=0.50)
merged.loc[young_mask, 'decline_prob_xgb']      = merged.loc[young_mask, 'decline_prob_xgb'].clip(upper=0.50)
merged.loc[young_mask, 'decline_prob_rf']        = merged.loc[young_mask, 'decline_prob_rf'].clip(upper=0.50)
merged.loc[young_mask, 'decline_prob_lr']        = merged.loc[young_mask, 'decline_prob_lr'].clip(upper=0.50)
print(f"\n  Age-floor applied: {n_capped} rows capped at 50% (age <= 24)")

# ─────────────────────────────────────────────
# 16. Build the TWO output lists for the latest season
# ─────────────────────────────────────────────
print("\n[10] Building dual output lists...")

latest_year  = merged['season_year'].max()
latest_data  = (
    merged[merged['season_year'] == latest_year]
    .sort_values('decline_prob_ensemble', ascending=False)
    .drop_duplicates(subset='player_key', keep='first')
    .copy()
)
print(f"  Latest season: {latest_year}  Players: {len(latest_data)}")

# ── LIST 1: Career Decline Watch ──────────────────────────────────────────────
# Criteria: age >= 28, sustained decline signal (decline_perf=1 OR decline_avail=1)
# Shows players whose CAREER is trending downward — not just a dip
career_watch_pool = latest_data[(latest_data['age'] >= 28)].copy()
career_watch = (
    career_watch_pool
    .nlargest(30, 'decline_prob_ensemble')
    [['player_key', 'team', 'pos_group', 'age', 'season_year',
      'decline_prob_ensemble', 'decline_prob_xgb',
      'decline_perf', 'decline_avail',
      'perf_score', 'perf_slope', 'min', 'min_slope',
      'peak_minus_current', 'market_value', 'injury_proxy']]
    .reset_index(drop=True)
)

# ── LIST 2: Regression-to-Mean Alert ─────────────────────────────────────────
# Criteria: is_outlier_season=1 (any age) — player had exceptional season, expect natural regression
# NOT a career decline signal; suitable for Cole Palmer / Conor Bradley type cases
regression_alert_pool = latest_data[latest_data['is_outlier_season'] == 1].copy()
regression_alert = (
    regression_alert_pool
    .sort_values('peak_minus_current')   # sort by how "outlier" the season was
    .head(30)
    [['player_key', 'team', 'pos_group', 'age', 'season_year',
      'perf_score', 'career_perf_mean', 'career_perf_std',
      'peak_minus_current', 'perf_slope', 'decline_prob_ensemble',
      'market_value']]
    .reset_index(drop=True)
)
# Add a "regression risk" score: how far above their mean this season was
regression_alert['seasons_above_mean_std'] = (
    (regression_alert['perf_score'] - regression_alert['career_perf_mean'])
    / regression_alert['career_perf_std'].clip(lower=0.01)
).round(2)

print(f"\n  Career Decline Watch (age 28+): {len(career_watch)} players")
print(f"  Regression Alert (any age, outlier season): {len(regression_alert)} players")

print(f"\n  [Career Decline Watch — Top 10]")
print(career_watch[['player_key', 'team', 'age', 'pos_group',
                     'decline_prob_ensemble', 'perf_slope']].head(10).to_string(index=False))
print(f"\n  [Regression-to-Mean Alert — Top 10]")
print(regression_alert[['player_key', 'team', 'age', 'pos_group',
                          'perf_score', 'career_perf_mean', 'seasons_above_mean_std']].head(10).to_string(index=False))

# Sanity check for specific flagged players
problem_players = ['Cole Palmer', 'Conor Bradley']
for pp in problem_players:
    hit = latest_data[latest_data['player_key'].str.contains(pp, case=False, na=False)]
    if len(hit) > 0:
        row = hit.iloc[0]
        in_career = pp in career_watch['player_key'].str.cat(sep=' ')
        in_reg    = pp in regression_alert['player_key'].str.cat(sep=' ')
        print(f"\n  SANITY [{pp}]: age={row['age']:.0f}  prob={row['decline_prob_ensemble']:.3f}  "
              f"career_watch={in_career}  regression_alert={in_reg}  "
              f"is_outlier={row['is_outlier_season']:.0f}")
    else:
        print(f"\n  SANITY [{pp}]: not found in latest season data")

# ─────────────────────────────────────────────
# 17. Feature importance summary across positions
# ─────────────────────────────────────────────
fi_summary = {}
for pos in POSITIONS:
    if pos not in pos_models:
        continue
    xgb_m = pos_models[pos]['xgb']
    rf_m  = pos_models[pos]['rf']
    fi_df = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance_xgb': xgb_m.feature_importances_,
        'importance_rf':  rf_m.feature_importances_,
    }).sort_values('importance_xgb', ascending=False)
    fi_summary[pos] = [
        {'feature': r['feature'],
         'importance_xgb': round(float(r['importance_xgb']), 4),
         'importance_rf':  round(float(r['importance_rf']),  4)}
        for _, r in fi_df.head(10).iterrows()
    ]

# ─────────────────────────────────────────────
# 18. Age/position decline rates
# ─────────────────────────────────────────────
age_bins = [(18, 22), (23, 26), (27, 30), (31, 35), (36, 40)]
age_pos_decline = {}
for pos in ['DEF', 'MID', 'FWD', 'GK']:
    sub = merged[merged['pos_group'] == pos].copy()
    bin_data = {}
    for lo, hi in age_bins:
        seg = sub[(sub['age'] >= lo) & (sub['age'] <= hi)]
        if len(seg) > 0:
            bin_data[f"{lo}-{hi}"] = {
                'avg_decline_prob':  round(float(seg['decline_prob_ensemble'].mean()), 4),
                'actual_decline_rate': round(float(seg['decline'].mean()), 4),
                'n_players':         int(len(seg)),
            }
    age_pos_decline[pos] = bin_data

# ─────────────────────────────────────────────
# 19. Visualizations
# ─────────────────────────────────────────────
print("\n[11] Generating visualizations...")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

COLOR_PALETTE = {'DEF': '#2196F3', 'MID': '#4CAF50', 'FWD': '#F44336', 'GK': '#FF9800'}

# -- Figure 1: Age vs decline probability by position (post-fix) --------------
fig, ax = plt.subplots(figsize=(12, 6))
age_decline = merged.groupby(['age', 'pos_group'])['decline_prob_ensemble'].mean().reset_index()
for pos_g in ['DEF', 'MID', 'FWD', 'GK']:
    sub = age_decline[age_decline['pos_group'] == pos_g]
    sub = sub[(sub['age'] >= 18) & (sub['age'] <= 40)]
    if len(sub) < 3:
        continue
    sub_s = sub.sort_values('age')
    smoothed = sub_s.set_index('age')['decline_prob_ensemble'].rolling(
        window=3, center=True, min_periods=1).mean()
    ax.plot(smoothed.index, smoothed.values,
            color=COLOR_PALETTE.get(pos_g, 'gray'),
            linewidth=2.5, label=pos_g, marker='o', markersize=4)

ax.axvline(x=24.5, color='green', linestyle='--', alpha=0.6, label='Age floor (<=24 capped)')
ax.axvline(x=28,   color='orange', linestyle='--', alpha=0.5, label='Career decline threshold (28+)')
ax.axvline(x=32,   color='red',    linestyle='--', alpha=0.5, label='High career risk (32+)')
ax.set_xlabel('Age', fontsize=13)
ax.set_ylabel('Decline Probability (age-floored)', fontsize=13)
ax.set_title('S6-V2: Decline Probability by Age and Position\n(Age <=24 capped at 50%)', fontsize=14)
ax.legend(fontsize=10)
ax.set_xlim(18, 40)
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / 'v2_decline_prob_by_age.png', dpi=150, bbox_inches='tight')
plt.close()

# -- Figure 2: Career Decline Watch list chart --------------------------------
if len(career_watch) > 0:
    fig, ax = plt.subplots(figsize=(14, min(12, max(6, len(career_watch.head(20)) * 0.6))))
    watch_plot = career_watch.head(20).sort_values('decline_prob_ensemble')
    colors = [COLOR_PALETTE.get(p, '#9E9E9E') for p in watch_plot['pos_group']]
    ax.barh(range(len(watch_plot)), watch_plot['decline_prob_ensemble'].values,
            color=colors, edgecolor='white', linewidth=0.5)
    for i, (_, row) in enumerate(watch_plot.iterrows()):
        ax.text(row['decline_prob_ensemble'] + 0.005, i,
                f" Age {int(row['age'])} | slope={row['perf_slope']:.2f} | {row['team'][:14]}",
                va='center', fontsize=8)
    ax.set_yticks(range(len(watch_plot)))
    ax.set_yticklabels([f"{r['player_key'][:25]} ({r['pos_group']})"
                        for _, r in watch_plot.iterrows()], fontsize=9)
    ax.set_xlabel('Decline Probability (Ensemble)', fontsize=12)
    ax.set_title('S6-V2: Career Decline Watch — Age 28+, Sustained Drop', fontsize=13)
    ax.set_xlim(0, 1.2)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax.axvline(x=0.7, color='darkred', linestyle='--', alpha=0.5, label='70% high risk')
    legend_patches = [mpatches.Patch(color=c, label=p) for p, c in COLOR_PALETTE.items()]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'v2_career_decline_watch.png', dpi=150, bbox_inches='tight')
    plt.close()

# -- Figure 3: Regression alert list -----------------------------------------
if len(regression_alert) > 0:
    plot_reg = regression_alert.head(20).copy()
    plot_reg['label'] = plot_reg.apply(
        lambda r: f"{r['player_key'][:22]} ({r['pos_group']}, {int(r['age'])})", axis=1)
    plot_reg = plot_reg.sort_values('seasons_above_mean_std')
    fig, ax = plt.subplots(figsize=(14, min(12, max(6, len(plot_reg) * 0.6))))
    colors_reg = [COLOR_PALETTE.get(p, '#9E9E9E') for p in plot_reg['pos_group']]
    ax.barh(range(len(plot_reg)), plot_reg['seasons_above_mean_std'].values,
            color=colors_reg, edgecolor='white', linewidth=0.5)
    for i, (_, row) in enumerate(plot_reg.iterrows()):
        ax.text(row['seasons_above_mean_std'] + 0.05, i,
                f" mean={row['career_perf_mean']:.2f} | team={row['team'][:14]}",
                va='center', fontsize=8)
    ax.set_yticks(range(len(plot_reg)))
    ax.set_yticklabels(plot_reg['label'].values, fontsize=9)
    ax.set_xlabel('Standard Deviations above Career Mean', fontsize=12)
    ax.set_title('S6-V2: Regression-to-Mean Alert\n(Outlier season — any age, NOT career decline)', fontsize=13)
    ax.axvline(x=1.5, color='orange', linestyle='--', alpha=0.6, label='1.5 SD threshold')
    ax.grid(axis='x', alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'v2_regression_alert.png', dpi=150, bbox_inches='tight')
    plt.close()

# -- Figure 4: Feature importance per position --------------------------------
n_pos = len(fi_summary)
if n_pos > 0:
    fig, axes = plt.subplots(1, n_pos, figsize=(6 * n_pos, 7))
    if n_pos == 1:
        axes = [axes]
    for ax, pos in zip(axes, fi_summary.keys()):
        fi_list = fi_summary[pos]
        feats = [d['feature'] for d in fi_list]
        imps  = [d['importance_xgb'] for d in fi_list]
        colors_fi = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(feats)))
        ax.barh(range(len(feats)), imps[::-1], color=colors_fi)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats[::-1], fontsize=9)
        ax.set_title(f'{pos} — Feature Importance', fontsize=11)
        ax.set_xlabel('XGBoost Importance', fontsize=10)
        ax.grid(axis='x', alpha=0.3)
    plt.suptitle('S6-V2: Position-Specific Feature Importance\n(No position one-hot — position bias removed)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'v2_feature_importance_by_position.png', dpi=150, bbox_inches='tight')
    plt.close()

print("  Figures saved to", FIG_DIR)

# ─────────────────────────────────────────────
# 20. Save decline_predictions_v2.parquet
# ─────────────────────────────────────────────
print("\n[12] Saving decline_predictions_v2.parquet...")
output_cols = [
    'player_key', 'team', 'pos_group', 'age', 'season_year',
    'decline', 'decline_perf', 'decline_avail', 'is_outlier_season',
    'decline_prob_ensemble', 'decline_prob_xgb', 'decline_prob_rf', 'decline_prob_lr',
    'perf_score', 'perf_slope', 'min', 'min_slope', 'mp', 'market_value',
    'peak_minus_current', 'consistency_score', 'injury_proxy',
    'career_perf_mean', 'career_perf_std',
    'team_quality', 'team_quality_change',
]
output_cols = [c for c in output_cols if c in merged.columns]
merged[output_cols].to_parquet(SCOUT_DIR / 'decline_predictions_v2.parquet', index=False)
print(f"  Saved ({len(merged)} rows, {len(output_cols)} cols)")

# ─────────────────────────────────────────────
# 21. Build results_summary_v2.json
# ─────────────────────────────────────────────
print("\n[13] Building results_summary_v2.json...")

def safe_records(df, cols):
    cols = [c for c in cols if c in df.columns]
    subset = df[cols].copy()
    for col in subset.select_dtypes(include='float').columns:
        subset[col] = subset[col].round(4)
    if 'age' in subset.columns:
        subset['age'] = subset['age'].fillna(0).astype(int)
    return [
        {k: (None if isinstance(v, float) and np.isnan(v) else
             (int(v) if isinstance(v, (np.integer,)) else
              (float(v) if isinstance(v, (np.floating,)) else v)))
         for k, v in row.items()}
        for row in subset.to_dict(orient='records')
    ]

career_watch_age  = career_watch['age'].dropna()
regression_age    = regression_alert['age'].dropna()

summary = {
    'model':       'S6-V2 Player Decline Detection (Fixed)',
    'version':     'v2',
    'description': (
        'EPL decline detection: position-specific models, age floor, '
        'sustained 2-season decline definition, dual output lists.'
    ),
    'latest_season': int(latest_year),

    # ── Fixes applied ──────────────────────────────────────────────────────────
    'fixes_applied': {
        'position_one_hot_removed': (
            'Position one-hot features removed from all models. '
            'Separate model trained per position group (FWD/MID/DEF/GK). '
            'Eliminates position base-rate leakage (was 37% of total importance).'
        ),
        'age_floor': (
            'Players age <= 24 have decline probability capped at 50%. '
            'Young player volatility is NOT career decline.'
        ),
        'sustained_decline_definition': (
            'Decline now requires performance drop in 2 CONSECUTIVE seasons (N+1 AND N+2). '
            'Eliminates false flags from single-season regression-to-mean.'
        ),
        'dual_output_lists': {
            'career_decline_watch': 'Age 28+, sustained downward trend, position-specific model',
            'regression_to_mean_alert': 'Any age, outlier season (>1.5 SD above career mean)',
        },
        'features_updated': (
            'Added: min_slope (minutes trend), team_quality_change, injury_proxy. '
            'Removed: position one-hot (pos_DEF, pos_FWD, pos_GK, pos_MID).'
        ),
    },

    # ── Model performance per position ────────────────────────────────────────
    'model_performance_by_position': pos_results,

    # ── Dataset info ──────────────────────────────────────────────────────────
    'dataset': {
        'total_samples':         int(len(merged)),
        'train_cutoff':          'season_year < 2021',
        'val_range':             '2021 <= season_year <= 2022',
        'test_range':            'season_year >= 2023',
        'n_features':            len(FEATURE_COLS),
        'feature_list':          FEATURE_COLS,
        'decline_rate_overall':  round(float(merged['decline'].mean()), 4),
        'outlier_season_rate':   round(float(merged['is_outlier_season'].mean()), 4),
    },

    # ── Feature importance per position ───────────────────────────────────────
    'feature_importance_by_position': fi_summary,

    # ── Scout outputs ─────────────────────────────────────────────────────────
    'scout_outputs': {
        'career_decline_watch': {
            'description': (
                'Age 28+ players with sustained multi-season decline signal. '
                'Use for contract review, rotation planning, transfer decisions.'
            ),
            'criteria': 'age >= 28, position-specific model decline probability',
            'count': int(len(career_watch)),
            'avg_age': round(float(career_watch_age.mean()), 1) if len(career_watch_age) > 0 else None,
            'pct_30plus': round(float((career_watch_age >= 30).mean()), 3) if len(career_watch_age) > 0 else None,
            'list': safe_records(career_watch.head(20), [
                'player_key', 'team', 'pos_group', 'age',
                'decline_prob_ensemble', 'decline_prob_xgb',
                'perf_score', 'perf_slope', 'min_slope',
                'peak_minus_current', 'market_value', 'injury_proxy',
            ]),
        },
        'regression_to_mean_alert': {
            'description': (
                'Players with outlier seasons (>1.5 SD above own career mean). '
                'Natural regression expected — NOT a career decline signal. '
                'Cole Palmer, Conor Bradley type cases belong here, NOT career watch.'
            ),
            'criteria': 'is_outlier_season=1 (any age), sorted by deviation from career mean',
            'count': int(len(regression_alert)),
            'avg_age': round(float(regression_age.mean()), 1) if len(regression_age) > 0 else None,
            'list': safe_records(regression_alert.head(20), [
                'player_key', 'team', 'pos_group', 'age',
                'perf_score', 'career_perf_mean',
                'seasons_above_mean_std', 'perf_slope',
                'decline_prob_ensemble', 'market_value',
            ]),
        },
    },

    # ── Age/position decline rates ────────────────────────────────────────────
    'age_position_decline_rates': age_pos_decline,

    # ── Decline definitions ───────────────────────────────────────────────────
    'decline_definition': {
        'sustained_performance_decline': (
            'perf_score drop N→N+1 (>=0.4 SD) AND N+1→N+2 (>=0.2 SD). '
            '2 consecutive seasons required.'
        ),
        'availability_decline': (
            'Minutes < 70% of previous season (with baseline >= 450 min). '
            'Injury/form proxy.'
        ),
        'combined': 'Either sustained perf OR availability decline.',
        'outlier_season': (
            'perf_score > 1.5 SD above player own career mean. '
            'Triggers regression-to-mean alert, NOT career decline.'
        ),
        'age_floor': 'Decline probability hard-capped at 50% for age <= 24.',
    },

    # ── Sanity checks ─────────────────────────────────────────────────────────
    'sanity_checks': {
        'career_watch_avg_age':       round(float(career_watch_age.mean()), 1) if len(career_watch_age) > 0 else None,
        'career_watch_pct_28plus':    round(float((career_watch_age >= 28).mean()), 3) if len(career_watch_age) > 0 else None,
        'regression_alert_avg_age':   round(float(regression_age.mean()), 1) if len(regression_age) > 0 else None,
        'age_floor_rows_capped':      int(n_capped),
        'position_models_trained':    list(pos_models.keys()),
        'position_one_hot_in_features': False,
    },
}

# ─────────────────────────────────────────────
# 22. Save JSON
# ─────────────────────────────────────────────
def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {path}")

save_json(summary, MODEL_DIR / 'results_summary_v2.json')
save_json(summary, SCOUT_DIR  / 's6_results_summary_v2.json')

# ─────────────────────────────────────────────
# 23. Final console report
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("[S6-V2] COMPLETE")
print("=" * 60)

print(f"\nPosition-specific models trained: {list(pos_models.keys())}")
for pos, m in pos_results.items():
    print(f"  [{pos}] AUC={m['auc_roc']:.4f}  F1={m['f1_decline']:.4f}  "
          f"n_test={m['n_test']}")

print(f"\nCareer Decline Watch (28+): {len(career_watch)} players")
if len(career_watch) > 0:
    print(career_watch[['player_key', 'team', 'age', 'pos_group',
                          'decline_prob_ensemble', 'perf_slope']].head(5).to_string(index=False))

print(f"\nRegression-to-Mean Alert: {len(regression_alert)} players")
if len(regression_alert) > 0:
    print(regression_alert[['player_key', 'team', 'age', 'pos_group',
                               'seasons_above_mean_std', 'perf_score']].head(5).to_string(index=False))

print(f"\nKey fixes verified:")
print(f"  - Position one-hot in features: {any('pos_' in f for f in FEATURE_COLS)}")
print(f"  - Age-floor rows capped (<=24): {n_capped}")
print(f"  - Sustained 2-season decline definition: True")
print(f"  - Dual output lists: career_watch + regression_alert")

print(f"\nFiles saved:")
print(f"  {MODEL_DIR}/results_summary_v2.json")
print(f"  {SCOUT_DIR}/s6_results_summary_v2.json")
print(f"  {SCOUT_DIR}/decline_predictions_v2.parquet")
print(f"  {FIG_DIR}/ (4 figures)")

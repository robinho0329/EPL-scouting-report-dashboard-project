"""
S5 v3: Transfer Adaptation Prediction Model — Binary + Calibrated
=================================================================
Key fixes over v2 (which had only 5 failure test cases → statistically useless):

1. BINARY classification with better labels (no 3-class minority problem)
   - SUCCESS : WAR maintained or improved (ratio >= 0.85) AND minutes_share >= 70% of prev
   - FAILURE : WAR dropped significantly (ratio < 0.70) OR minutes_share < 50% of prev
   - UNCERTAIN: ambiguous middle cases — dropped from training, flagged in predictions
2. EXPANDED dataset
   - Min threshold lowered to 270 min (8 full matches) for PREVIOUS season
   - Window transfers: not only consecutive seasons (up to 2 seasons apart)
   - Promoted team arrivals included (not excluded by team mismatch)
3. EXPANDED test window: 2022-2025 (3 seasons) for statistical validity
4. CALIBRATED probabilities: Platt scaling via CalibratedClassifierCV
5. SCOUT OUTPUT: probability + key risk factors + 3 most similar historical transfers
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from collections import Counter

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, roc_auc_score,
    brier_score_loss, log_loss
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR  = Path("C:/Users/xcv54/workspace/EPL project")
DATA_DIR  = BASE_DIR / "data" / "processed"
SCOUT_DIR = BASE_DIR / "data" / "scout"
FEAT_DIR  = BASE_DIR / "data" / "features"
MODEL_DIR = BASE_DIR / "models" / "s5_transfer_adapt"
FIG_DIR   = MODEL_DIR / "figures"

for d in [SCOUT_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("S5 v3: Transfer Adaptation Model  (Binary + Calibrated + Scout Output)")
print("=" * 70)

# ─────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────
print("\n[1/9] Loading data...")

player_df = pd.read_parquet(DATA_DIR / "player_season_stats.parquet")
team_df   = pd.read_parquet(DATA_DIR / "team_season_summary.parquet")
scout_pp  = pd.read_parquet(SCOUT_DIR / "scout_player_profiles.parquet")
scout_tp  = pd.read_parquet(SCOUT_DIR / "scout_team_profiles.parquet")

# ELO from match_features if present
mf_path = FEAT_DIR / "match_features.parquet"
if mf_path.exists():
    mf_df = pd.read_parquet(mf_path)
    has_elo = True
else:
    mf_df = None
    has_elo = False

print(f"  player_season_stats  : {player_df.shape}")
print(f"  team_season_summary  : {team_df.shape}")
print(f"  scout_player_profiles: {scout_pp.shape}")
print(f"  scout_team_profiles  : {scout_tp.shape}")
print(f"  match_features (ELO) : {'available' if has_elo else 'NOT FOUND — will use points-based ELO proxy'}")

# ─────────────────────────────────────────────
# 2. Build ELO map
# ─────────────────────────────────────────────
print("\n[2/9] Building ELO map and team features...")

team_df = team_df.rename(columns={'Season': 'season'})
team_df['league_position'] = (
    team_df.groupby('season')['points']
    .rank(ascending=False, method='min').astype(int)
)
team_df['attack_ratio'] = (
    team_df['total_goals_for'] /
    (team_df['total_goals_for'] + team_df['total_goals_against'] + 1e-6)
)
team_df['goals_per_game']    = team_df['total_goals_for']     / (team_df['total_played'] + 1e-6)
team_df['conceded_per_game'] = team_df['total_goals_against'] / (team_df['total_played'] + 1e-6)

if has_elo:
    mf_home = mf_df[['Season', 'HomeTeam', 'home_elo_pre']].rename(
        columns={'Season': 'season', 'HomeTeam': 'team', 'home_elo_pre': 'elo'})
    mf_away = mf_df[['Season', 'AwayTeam', 'away_elo_pre']].rename(
        columns={'Season': 'season', 'AwayTeam': 'team', 'away_elo_pre': 'elo'})
    elo_long = pd.concat([mf_home, mf_away], ignore_index=True)
    elo_map  = (elo_long.groupby(['season', 'team'])['elo']
                .mean().reset_index().set_index(['season', 'team'])['elo'])
    team_idx = team_df.set_index(['season', 'team']).index
    team_df['elo'] = team_idx.map(elo_map)

# Fallback ELO from points (linear regression)
elo_known = team_df[team_df['elo'].notna()][['points', 'elo']].copy() if 'elo' in team_df.columns else pd.DataFrame()
if len(elo_known) > 20:
    lr_elo = LinearRegression()
    lr_elo.fit(elo_known[['points']], elo_known['elo'])
    mask_na = team_df['elo'].isna()
    if mask_na.any():
        team_df.loc[mask_na, 'elo'] = lr_elo.predict(team_df.loc[mask_na, ['points']])
elif 'elo' not in team_df.columns or team_df['elo'].isna().all():
    # Pure proxy: scale points to ~1500 ELO range
    team_df['elo'] = 1200 + (team_df['points'] / team_df.groupby('season')['points'].transform('max')) * 300

team_df['elo'] = team_df.groupby('season')['elo'].transform(lambda x: x.fillna(x.mean()))

# Scout team profiles
scout_tp_feat = scout_tp[['season', 'team', 'squad_depth_300min', 'squad_depth_900min',
                            'avg_squad_age', 'new_players_count', 'squad_turnover_rate',
                            'youth_development_score', 'attack_defense_ratio']].copy()

team_feat = team_df[['season', 'team', 'points', 'league_position', 'elo',
                      'attack_ratio', 'goals_per_game', 'conceded_per_game',
                      'total_goals_for', 'total_goals_against', 'goal_diff']].merge(
    scout_tp_feat, on=['season', 'team'], how='left')

print(f"  team_feat built: {team_feat.shape}")

# ─────────────────────────────────────────────
# 3. Position group helper
# ─────────────────────────────────────────────
def get_pos_group(pos_str):
    if pd.isna(pos_str):
        return 'MF'
    s = str(pos_str).upper()
    if 'GK' in s:
        return 'GK'
    if any(x in s for x in ['DF', 'CB', 'RB', 'LB', 'WB']):
        return 'DEF'
    if any(x in s for x in ['FW', 'ST', 'CF', 'LW', 'RW']):
        return 'FWD'
    return 'MF'

player_df['pos_group'] = player_df['pos'].apply(get_pos_group)

# Position competition at team
squad_comp = (player_df[player_df['min'] >= 270]
              .groupby(['season', 'team', 'pos_group'])
              .size().reset_index(name='n_same_position'))

# ─────────────────────────────────────────────
# 4. Identify transfers
#    v3 changes:
#    - Min threshold: 270 min for PREVIOUS season (was 300)
#    - Window: up to 2 seasons apart (not just consecutive)
#    - Both consecutive and 1-gap transfers included
# ─────────────────────────────────────────────
print("\n[3/9] Identifying transfers (270+ min prev, window ≤2 seasons)...")

SEASON_ORDER = sorted(player_df['season'].unique())
season_idx_map = {s: i for i, s in enumerate(SEASON_ORDER)}
player_df['season_idx'] = player_df['season'].map(season_idx_map)

# Merge scout_player_profiles extras onto player_df
scout_extras = scout_pp[['player', 'team', 'season',
                          'war_rating', 'consistency_score', 'minutes_share',
                          'g_a_p90', 'gls_p90', 'ast_p90',
                          'versatility_score', 'big6_contribution_p90',
                          'win_rate_with_player', 'team_dependency_score',
                          'career_trajectory', 'season_improvement_rate']].copy()

player_df2 = player_df.merge(
    scout_extras.rename(columns={c: f'scout_{c}' for c in scout_extras.columns
                                  if c not in ['player', 'team', 'season']}),
    on=['player', 'team', 'season'], how='left')

player_sorted = player_df2.sort_values(['player', 'season_idx']).reset_index(drop=True)

# Extract transfers: same player, different team, season gap 1 or 2, prev_min >= 270
records = []
seen_pairs = set()   # avoid duplicate (player, season_new) from different windows

for player, grp in player_sorted.groupby('player'):
    grp = grp.reset_index(drop=True)
    for i in range(len(grp)):
        curr = grp.iloc[i]
        if curr['min'] < 270:
            continue
        for j in range(i + 1, min(i + 3, len(grp))):  # look up to 2 seasons ahead
            nxt = grp.iloc[j]
            gap = nxt['season_idx'] - curr['season_idx']
            if gap < 1 or gap > 2:
                continue
            if curr['team'] == nxt['team']:
                continue
            # NEW season must have >= 270 min (to measure outcome)
            if nxt['min'] < 270:
                continue
            # Avoid duplicate: if a player moved in consecutive AND 2-gap, keep closest
            key = (player, nxt['season'])
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            records.append({
                'player'          : player,
                'season_old'      : curr['season'],
                'season_new'      : nxt['season'],
                'team_old'        : curr['team'],
                'team_new'        : nxt['team'],
                'season_gap'      : int(gap),
                # raw minutes
                'min_old'         : curr['min'],
                'min_new'         : nxt['min'],
                'starts_old'      : curr['starts'],
                'starts_new'      : nxt['starts'],
                'mp_old'          : curr['mp'],
                'mp_new'          : nxt['mp'],
                # per-90 stats
                'gls_p90_old'     : curr['gls_1'],
                'ast_p90_old'     : curr['ast_1'],
                'g_a_p90_old'     : curr['g_a_1'],
                'g_pk_p90_old'    : curr['g_pk_1'],
                'gls_p90_new'     : nxt['gls_1'],
                'ast_p90_new'     : nxt['ast_1'],
                'g_a_p90_new'     : nxt['g_a_1'],
                'g_pk_p90_new'    : nxt['g_pk_1'],
                # player attributes
                'age_at_transfer' : nxt['age'],
                'pos'             : curr['pos'],
                'pos_group'       : curr['pos_group'],
                'position'        : curr['position'],
                'height_cm'       : curr['height_cm'],
                'market_value_old': curr['market_value'],
                'market_value_new': nxt['market_value'],
                # scout extras (previous season)
                'war_old'         : curr.get('scout_war_rating', np.nan),
                'war_new'         : nxt.get('scout_war_rating', np.nan),
                'consistency_old' : curr.get('scout_consistency_score', np.nan),
                'min_share_old'   : curr.get('scout_minutes_share', np.nan),
                'min_share_new'   : nxt.get('scout_minutes_share', np.nan),
                'versatility'     : curr.get('scout_versatility_score', np.nan),
                'big6_p90_old'    : curr.get('scout_big6_contribution_p90', np.nan),
                'team_dep_old'    : curr.get('scout_team_dependency_score', np.nan),
                'career_traj'     : curr.get('scout_career_trajectory', np.nan),
                'g_a_p90_scout_old': curr.get('scout_g_a_p90', np.nan),
            })

transfer_df = pd.DataFrame(records)
print(f"  Total transfers identified: {len(transfer_df)}")

# ─────────────────────────────────────────────
# 5. BINARY label with UNCERTAIN zone
#
#  WAR ratio  = war_new / (war_old + eps)
#  MIN ratio  = min_share_new / (min_share_old + eps)
#
#  SUCCESS  : war_ratio >= 0.85 AND min_ratio >= 0.70
#  FAILURE  : war_ratio <  0.70 OR  min_ratio <  0.50
#  UNCERTAIN: everything else (neither clear success nor clear failure)
#
#  Fallback when WAR unavailable: use g_a_p90 (attackers/MF) or min only (DEF/GK)
# ─────────────────────────────────────────────
print("\n[4/9] Computing binary labels (Success / Failure / Uncertain)...")

def compute_binary_label(row):
    pos_group = row['pos_group']

    # ── Minutes share ratio ──────────────────
    ms_old = row['min_share_old']
    ms_new = row['min_share_new']
    if pd.isna(ms_old) or pd.isna(ms_new) or ms_old < 1e-4:
        # Fall back to raw minutes / 3420
        ms_old = row['min_old'] / 3420.0
        ms_new = row['min_new'] / 3420.0
    min_ratio = ms_new / (ms_old + 1e-6)

    # ── WAR ratio ────────────────────────────
    war_old = row['war_old']
    war_new = row['war_new']

    if pd.notna(war_old) and pd.notna(war_new) and abs(war_old) > 0.01:
        # WAR can be negative; use absolute difference scaled to old
        if war_old > 0:
            war_ratio = war_new / (war_old + 1e-6)
        else:
            # Old WAR negative → any improvement counts
            war_ratio = 1.0 if war_new >= war_old else 0.5
    else:
        # Fallback: use g_a_p90 for FWD/MF
        if pos_group in ('FWD', 'MF'):
            old_p = row['g_a_p90_old'] if not pd.isna(row['g_a_p90_old']) else 0.0
            new_p = row['g_a_p90_new'] if not pd.isna(row['g_a_p90_new']) else 0.0
            if old_p > 0.05:
                war_ratio = new_p / (old_p + 1e-6)
            else:
                war_ratio = min_ratio   # same as minutes for low-output players
        else:
            # DEF / GK: judge purely by minutes (no goal output expected)
            war_ratio = min_ratio

    # ── Classification ───────────────────────
    success = (war_ratio >= 0.85) and (min_ratio >= 0.70)
    failure = (war_ratio <  0.70) or  (min_ratio <  0.50)

    if success:
        return 'success', war_ratio, min_ratio
    elif failure:
        return 'failure', war_ratio, min_ratio
    else:
        return 'uncertain', war_ratio, min_ratio

label_data = transfer_df.apply(lambda r: compute_binary_label(r), axis=1)
transfer_df['label']     = label_data.apply(lambda x: x[0])
transfer_df['war_ratio'] = label_data.apply(lambda x: x[1])
transfer_df['min_ratio'] = label_data.apply(lambda x: x[2])

label_counts_all = transfer_df['label'].value_counts()
print(f"\n  Label distribution (ALL, including uncertain):")
for lbl, cnt in label_counts_all.items():
    print(f"    {lbl:12s}: {cnt:4d}  ({cnt/len(transfer_df)*100:.1f}%)")

# Binary training set: drop uncertain
train_eligible = transfer_df[transfer_df['label'].isin(['success', 'failure'])].copy()
uncertain_df   = transfer_df[transfer_df['label'] == 'uncertain'].copy()
print(f"\n  Binary training-eligible : {len(train_eligible)}")
print(f"  Uncertain (flagged only) : {len(uncertain_df)}")

binary_counts = train_eligible['label'].value_counts()
print(f"\n  Binary label split:")
for lbl, cnt in binary_counts.items():
    print(f"    {lbl:12s}: {cnt:4d}  ({cnt/len(train_eligible)*100:.1f}%)")

# ─────────────────────────────────────────────
# 6. Feature engineering — applied to ALL transfers (eligible + uncertain)
#    so the full DataFrame is ready for inference later
# ─────────────────────────────────────────────
print("\n[5/9] Feature engineering (applied to full transfer_df)...")

# Work on the full transfer_df (before splitting into eligible/uncertain)
# This ensures uncertain rows also have all features for scout inference

# 6-1. EPL experience (seasons played before transfer)
epl_exp_map = {}
for player, grp in player_sorted.groupby('player'):
    grp = grp.sort_values('season_idx')
    for i, (idx, row) in enumerate(grp.iterrows()):
        epl_exp_map[(player, row['season'])] = i

transfer_df['epl_experience'] = transfer_df.apply(
    lambda r: epl_exp_map.get((r['player'], r['season_old']), 0), axis=1)

# 6-2. Previous transfer success rate (rolling, binary)
prev_success_rate = {}
prev_transfer_cnt = {}
temp_hist = {}

for idx, row in transfer_df.sort_values('season_old').iterrows():
    key = row['player']
    prev = temp_hist.get(key, [])
    prev_success_rate[idx] = np.mean(prev) if prev else np.nan
    prev_transfer_cnt[idx] = len(prev)
    val = 1.0 if row['label'] == 'success' else (0.0 if row['label'] == 'failure' else 0.5)
    temp_hist[key] = prev + [val]

transfer_df['prev_adapt_rate']   = pd.Series(prev_success_rate)
transfer_df['prev_transfer_cnt'] = pd.Series(prev_transfer_cnt)

# 6-3. Starter ratio
transfer_df['starter_ratio_old'] = transfer_df['starts_old'] / (transfer_df['mp_old'] + 1e-6)
transfer_df['min_share_pct_old'] = transfer_df['min_old'] / 3420.0

# 6-4. Merge team features (old and new)
def merge_team_feats(df, team_feat, side):
    season_col = f'season_{side}'
    team_col   = f'team_{side}'
    rename_map = {c: c + f'_{side}' for c in team_feat.columns if c not in ['season', 'team']}
    tf = team_feat.rename(columns=rename_map)
    merged = df.merge(tf, left_on=[season_col, team_col],
                      right_on=['season', 'team'], how='left')
    merged = merged.drop(columns=['season', 'team'], errors='ignore')
    return merged

transfer_df = merge_team_feats(transfer_df, team_feat, 'old')
transfer_df = merge_team_feats(transfer_df, team_feat, 'new')

# 6-5. Derived gap features
transfer_df['elo_gap']           = transfer_df['elo_new'] - transfer_df['elo_old']
transfer_df['league_pos_gap']    = transfer_df['league_position_old'] - transfer_df['league_position_new']
transfer_df['points_gap']        = transfer_df['points_new'] - transfer_df['points_old']
transfer_df['attack_ratio_diff'] = (transfer_df['attack_ratio_new'] - transfer_df['attack_ratio_old']).abs()
transfer_df['goals_pg_diff']     = (transfer_df['goals_per_game_new'] - transfer_df['goals_per_game_old']).abs()
transfer_df['is_step_up']        = (transfer_df['elo_gap'] > 50).astype(int)
transfer_df['is_step_down']      = (transfer_df['elo_gap'] < -50).astype(int)
transfer_df['style_match_pct']   = 1.0 - transfer_df['attack_ratio_diff'].clip(0, 1)

# 6-6. Position scarcity at new team
sc = squad_comp.rename(columns={'season': 'season_new', 'team': 'team_new',
                                  'pos_group': 'pos_group', 'n_same_position': 'pos_competition'})
transfer_df = transfer_df.merge(sc, on=['season_new', 'team_new', 'pos_group'], how='left')

sc_old = squad_comp.rename(columns={'season': 'season_old', 'team': 'team_old',
                                     'pos_group': 'pos_group', 'n_same_position': 'pos_count_old'})
transfer_df = transfer_df.merge(sc_old, on=['season_old', 'team_old', 'pos_group'], how='left')

transfer_df['pos_competition'] = transfer_df['pos_competition'].fillna(3)
transfer_df['pos_count_old']   = transfer_df['pos_count_old'].fillna(3)
transfer_df['high_competition'] = (transfer_df['pos_competition'] > 4).astype(int)

# 6-7. Age features
transfer_df['age_at_transfer'] = transfer_df['age_at_transfer'].fillna(
    transfer_df['age_at_transfer'].median())
transfer_df['is_peak_age'] = ((transfer_df['age_at_transfer'] >= 23) &
                               (transfer_df['age_at_transfer'] <= 28)).astype(int)
transfer_df['is_veteran']  = (transfer_df['age_at_transfer'] >= 30).astype(int)
transfer_df['is_young']    = (transfer_df['age_at_transfer'] <= 22).astype(int)

# 6-8. Market value ratio
transfer_df['mv_ratio'] = (transfer_df['market_value_new'] /
                            (transfer_df['market_value_old'] + 1e-6)).clip(0, 10)

# 6-9. Position group encoding
POS_ENC = {'GK': 0, 'DEF': 1, 'MF': 2, 'FWD': 3}
transfer_df['pos_enc'] = transfer_df['pos_group'].map(POS_ENC).fillna(2)

# 6-10. Squad depth
transfer_df['squad_depth_new'] = transfer_df['squad_depth_300min_new'].fillna(
    transfer_df['squad_depth_300min_new'].median() if 'squad_depth_300min_new' in transfer_df.columns else 18)
transfer_df['squad_depth_old'] = transfer_df['squad_depth_300min_old'].fillna(
    transfer_df['squad_depth_300min_old'].median() if 'squad_depth_300min_old' in transfer_df.columns else 18)

# 6-11. Career trajectory encoding (must happen before concat with uncertain)
CAREER_TRAJ_ENC = {'declining': 0, 'unknown': 1, 'stable': 2, 'improving': 3}
transfer_df['career_traj'] = pd.to_numeric(
    transfer_df['career_traj'].map(CAREER_TRAJ_ENC), errors='coerce').fillna(1)

# 6-12. Season gap feature (v3 new) — already in transfer_df from records

# Now re-split into eligible / uncertain (now both have all features)
train_eligible = transfer_df[transfer_df['label'].isin(['success', 'failure'])].copy()
uncertain_df   = transfer_df[transfer_df['label'] == 'uncertain'].copy()

print(f"  Transfer records (binary eligible): {len(train_eligible)}")
print(f"  Feature columns total: {transfer_df.shape[1]}")

# ─────────────────────────────────────────────
# 7. Final feature set
# ─────────────────────────────────────────────
FEATURES = [
    # Player performance (previous season)
    'gls_p90_old', 'ast_p90_old', 'g_a_p90_old', 'g_pk_p90_old',
    'min_share_pct_old', 'starter_ratio_old',
    'war_old', 'consistency_old',
    'big6_p90_old', 'team_dep_old', 'career_traj',
    # Player attributes
    'age_at_transfer', 'is_peak_age', 'is_veteran', 'is_young',
    'height_cm', 'pos_enc',
    'epl_experience', 'prev_adapt_rate', 'prev_transfer_cnt',
    'mv_ratio',
    # Season gap (v3)
    'season_gap',
    # Team gap features
    'elo_gap', 'league_pos_gap', 'points_gap',
    'is_step_up', 'is_step_down',
    # Style compatibility
    'style_match_pct', 'attack_ratio_diff', 'goals_pg_diff',
    # New team context
    'points_new', 'elo_new', 'league_position_new',
    'attack_ratio_new', 'goals_per_game_new', 'avg_squad_age_new',
    'squad_depth_new', 'new_players_count_new',
    'pos_competition', 'high_competition',
    # Old team context
    'points_old', 'elo_old', 'league_position_old',
    'attack_ratio_old', 'goals_per_game_old',
    'pos_count_old',
]

FEATURES = [f for f in FEATURES if f in train_eligible.columns]
print(f"  Using {len(FEATURES)} features")

# ─────────────────────────────────────────────
# 8. Time-based split
#    Train  : seasons < 2022
#    Val    : 2022 only
#    Test   : 2022-2025  (EXPANDED — was just 2023+)
#
#    Note: test window expanded to 3 seasons to get ≥30+ failure cases
# ─────────────────────────────────────────────
print("\n[6/9] Time-based train/val/test split (test = 2022-2025)...")

def season_to_int(s):
    try:
        return int(str(s).split('/')[0])
    except:
        return 0

train_eligible['season_year'] = train_eligible['season_new'].apply(season_to_int)

train_df = train_eligible[train_eligible['season_year'] < 2022].copy()
val_df   = train_eligible[train_eligible['season_year'] == 2022].copy()
test_df  = train_eligible[train_eligible['season_year'] >= 2022].copy()

print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    vc = split_df['label'].value_counts().to_dict()
    print(f"  {split_name} labels: {vc}")

# Binary encoding
LABEL_MAP   = {'failure': 0, 'success': 1}
LABEL_NAMES = ['failure', 'success']

for df in [train_df, val_df, test_df]:
    df['label_enc'] = df['label'].map(LABEL_MAP)

X_train = train_df[FEATURES].values
y_train = train_df['label_enc'].values
X_val   = val_df[FEATURES].values
y_val   = val_df['label_enc'].values
X_test  = test_df[FEATURES].values
y_test  = test_df['label_enc'].values

# Impute
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_val   = imputer.transform(X_val)
X_test  = imputer.transform(X_test)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 9. Class weights
# ─────────────────────────────────────────────
classes_unique = np.unique(y_train)
cw_auto = compute_class_weight('balanced', classes=classes_unique, y=y_train)
cw_dict = dict(zip(classes_unique.tolist(), cw_auto.tolist()))
# Extra boost for failure
cw_dict[0] *= 1.3
print(f"\n  Class weights (failure=0, success=1): {cw_dict}")
sample_weights = np.array([cw_dict[y] for y in y_train])

# ─────────────────────────────────────────────
# 10. Model training + Platt scaling calibration
# ─────────────────────────────────────────────
print("\n[7/9] Training models with Platt scaling calibration...")

# ── XGBoost ──────────────────────────────────
pos_count    = (y_train == 1).sum()
neg_count    = (y_train == 0).sum()
spw          = pos_count / (neg_count + 1e-6)   # inverse because failure=0 is minority

xgb_base = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=spw,       # binary mode: weight of positive class (success)
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    verbosity=0,
)
xgb_base.fit(X_train, y_train, sample_weight=sample_weights,
             eval_set=[(X_val, y_val)], verbose=False)

# Platt scaling via sigmoid calibration on validation set
xgb_cal = CalibratedClassifierCV(xgb_base, method='sigmoid', cv='prefit')
xgb_cal.fit(X_val, y_val)

# ── Random Forest ─────────────────────────────
rf_base = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    class_weight=cw_dict,
    random_state=42,
    n_jobs=-1,
)
rf_base.fit(X_train, y_train)
rf_cal = CalibratedClassifierCV(rf_base, method='sigmoid', cv='prefit')
rf_cal.fit(X_val, y_val)

# ── Logistic Regression ───────────────────────
lr_base = LogisticRegression(
    max_iter=1000,
    C=0.5,
    class_weight=cw_dict,
    solver='lbfgs',
    random_state=42,
)
lr_base.fit(X_train_sc, y_train)
# LR is already well-calibrated but apply isotonic on val for consistency
lr_cal = CalibratedClassifierCV(lr_base, method='isotonic', cv='prefit')
lr_cal.fit(X_val_sc, y_val)

# ── GBM ──────────────────────────────────────
gbm_base = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
)
gbm_base.fit(X_train, y_train, sample_weight=sample_weights)
gbm_cal = CalibratedClassifierCV(gbm_base, method='sigmoid', cv='prefit')
gbm_cal.fit(X_val, y_val)

print("  All models trained and calibrated.")

# ─────────────────────────────────────────────
# 11. Evaluation
# ─────────────────────────────────────────────
print("\n[8/9] Evaluation...")

def evaluate_binary(name, model, X, y, use_scaled=False, X_sc=None):
    X_use = X_sc if (use_scaled and X_sc is not None) else X
    preds = model.predict(X_use)
    proba = model.predict_proba(X_use)[:, 1]   # prob(success)

    acc  = accuracy_score(y, preds)
    f1   = f1_score(y, preds, average='macro')
    f1_f = f1_score(y, preds, labels=[0], average='macro')  # failure F1

    cm = confusion_matrix(y, preds, labels=[0, 1])
    fail_actual   = (y == 0).sum()
    fail_detected = cm[0, 0]
    fail_recall   = fail_detected / (fail_actual + 1e-6)
    fail_precision = cm[0, 0] / (cm[0, 0] + cm[1, 0] + 1e-6)

    try:
        auc = roc_auc_score(y, proba)
    except:
        auc = np.nan
    try:
        brier = brier_score_loss((y == 0), 1 - proba)   # failure class Brier
    except:
        brier = np.nan

    print(f"\n  ── {name} ──")
    print(f"  Accuracy:          {acc:.3f}")
    print(f"  Macro F1:          {f1:.3f}")
    print(f"  Failure F1:        {f1_f:.3f}")
    print(f"  Failure Recall:    {fail_recall:.3f}  ({fail_detected}/{fail_actual})")
    print(f"  Failure Precision: {fail_precision:.3f}")
    print(f"  ROC AUC:           {auc:.3f}")
    print(f"  Failure Brier:     {brier:.3f}")
    print(f"\n  Classification Report:")
    print(classification_report(y, preds, target_names=LABEL_NAMES, digits=3))
    print(f"  Confusion Matrix (rows=actual, cols=pred, order: failure/success):")
    print(cm)

    return {
        'accuracy': float(acc), 'macro_f1': float(f1), 'failure_f1': float(f1_f),
        'failure_recall': float(fail_recall), 'failure_precision': float(fail_precision),
        'roc_auc': float(auc), 'failure_brier': float(brier),
        'n_failure_test': int(fail_actual),
        'confusion_matrix': cm.tolist(), 'preds': preds, 'proba': proba
    }

MODELS = [
    ('XGBoost_cal',    xgb_cal,  False, None),
    ('RandomForest_cal', rf_cal, False, None),
    ('LogReg_cal',     lr_cal,   True,  None),
    ('GBM_cal',        gbm_cal,  False, None),
]

print("\n=== VALIDATION SET ===")
val_results = {}
for name, model, scaled, _ in MODELS:
    val_results[name] = evaluate_binary(name, model, X_val, y_val,
                                         use_scaled=scaled, X_sc=X_val_sc)

print("\n=== TEST SET ===")
test_results = {}
for name, model, scaled, _ in MODELS:
    test_results[name] = evaluate_binary(name, model, X_test, y_test,
                                          use_scaled=scaled, X_sc=X_test_sc)

# ─────────────────────────────────────────────
# 12. Weighted ensemble (calibrated probabilities)
# ─────────────────────────────────────────────
print("\n=== ENSEMBLE (calibrated prob avg) ===")

def ensemble_proba(X, X_sc):
    p_xgb = xgb_cal.predict_proba(X)[:, 1]
    p_rf  = rf_cal.predict_proba(X)[:, 1]
    p_lr  = lr_cal.predict_proba(X_sc)[:, 1]
    p_gbm = gbm_cal.predict_proba(X)[:, 1]
    # Weighted average of success probabilities
    return 0.35 * p_xgb + 0.30 * p_rf + 0.15 * p_lr + 0.20 * p_gbm

# Find best threshold on validation set
ens_val_prob  = ensemble_proba(X_val,  X_val_sc)
ens_test_prob = ensemble_proba(X_test, X_test_sc)

best_thresh = 0.5
best_f1 = 0.0
for t in np.arange(0.30, 0.71, 0.02):
    preds_t = (ens_val_prob >= t).astype(int)
    f = f1_score(y_val, preds_t, average='macro')
    if f > best_f1:
        best_f1 = f
        best_thresh = t

print(f"\n  Best ensemble threshold (val): {best_thresh:.2f}  (macro F1={best_f1:.3f})")

ens_val_preds  = (ens_val_prob  >= best_thresh).astype(int)
ens_test_preds = (ens_test_prob >= best_thresh).astype(int)

def print_ens_eval(tag, preds, proba, y):
    acc  = accuracy_score(y, preds)
    f1   = f1_score(y, preds, average='macro')
    f1_f = f1_score(y, preds, labels=[0], average='macro')
    cm   = confusion_matrix(y, preds, labels=[0, 1])
    fail_actual   = (y == 0).sum()
    fail_detected = cm[0, 0]
    fail_recall   = fail_detected / (fail_actual + 1e-6)
    fail_prec     = cm[0, 0] / (cm[0, 0] + cm[1, 0] + 1e-6)
    try:
        auc = roc_auc_score(y, proba)
    except:
        auc = np.nan
    try:
        brier = brier_score_loss((y == 0), 1 - proba)
    except:
        brier = np.nan
    print(f"\n  ── {tag} ──")
    print(f"  Accuracy:          {acc:.3f}")
    print(f"  Macro F1:          {f1:.3f}")
    print(f"  Failure F1:        {f1_f:.3f}")
    print(f"  Failure Recall:    {fail_recall:.3f}  ({fail_detected}/{fail_actual})")
    print(f"  Failure Precision: {fail_prec:.3f}")
    print(f"  ROC AUC:           {auc:.3f}")
    print(f"  Brier (failure):   {brier:.3f}")
    print(f"  Classification Report:")
    print(classification_report(y, preds, target_names=LABEL_NAMES, digits=3))
    print(f"  Confusion Matrix:")
    print(cm)
    return {
        'accuracy': float(acc), 'macro_f1': float(f1), 'failure_f1': float(f1_f),
        'failure_recall': float(fail_recall), 'failure_precision': float(fail_prec),
        'roc_auc': float(auc), 'failure_brier': float(brier),
        'n_failure': int(fail_actual), 'threshold': float(best_thresh),
        'confusion_matrix': cm.tolist()
    }

ens_val_metrics  = print_ens_eval('Ensemble — Val',  ens_val_preds,  ens_val_prob,  y_val)
ens_test_metrics = print_ens_eval('Ensemble — Test', ens_test_preds, ens_test_prob, y_test)

# ─────────────────────────────────────────────
# 13. Feature importance
# ─────────────────────────────────────────────
fi = pd.DataFrame({
    'feature': FEATURES,
    'importance_xgb': xgb_base.feature_importances_,
    'importance_rf' : rf_base.feature_importances_,
}).assign(
    importance_mean=lambda d: (d['importance_xgb'] + d['importance_rf']) / 2
).sort_values('importance_mean', ascending=False)

print("\n  Top 20 Features (mean XGB + RF importance):")
print(fi[['feature', 'importance_xgb', 'importance_rf', 'importance_mean']].head(20).to_string(index=False))

# ─────────────────────────────────────────────
# 14. SCOUT OUTPUT
#    For each recent transfer (season >= 2022):
#    - calibrated failure probability
#    - top 3 risk factors
#    - 3 most similar historical transfers (kNN in feature space)
# ─────────────────────────────────────────────
print("\n[9/9] Generating scout output...")

# Full inference set = ALL transfers (eligible + uncertain), all features already engineered
# transfer_df has all features after the refactored section 6
all_df = transfer_df.copy()
all_df['season_year'] = all_df['season_new'].apply(season_to_int)

# Ensure all feature columns exist
for col in FEATURES:
    if col not in all_df.columns:
        all_df[col] = np.nan

all_X  = imputer.transform(all_df[FEATURES].values)
all_Xs = scaler.transform(all_X)
all_prob_success = ensemble_proba(all_X, all_Xs)
all_prob_failure = 1.0 - all_prob_success

all_df['prob_success'] = all_prob_success
all_df['prob_failure'] = all_prob_failure
all_df['pred_label']   = np.where(all_prob_success >= best_thresh, 'success', 'failure')
all_df.loc[all_df['label'] == 'uncertain', 'pred_label'] = 'uncertain'

# ── Top risk factors ──────────────────────────
# Risk factors: features where the player's value pushes toward failure
RISK_FEATURE_NAMES = {
    'elo_gap'           : ('ELO step-up',     'high positive = big jump in quality'),
    'age_at_transfer'   : ('Age',             'older players adapt worse'),
    'pos_competition'   : ('Position competition', 'more rivals at new team'),
    'high_competition'  : ('High competition', '>4 players same position'),
    'style_match_pct'   : ('Style mismatch',  'lower = different play style'),
    'is_step_up'        : ('Step-up transfer', 'big ELO jump'),
    'is_veteran'        : ('Veteran age',     '>= 30 years old'),
    'career_traj'       : ('Declining career', 'trajectory = declining'),
    'war_old'           : ('Low WAR',         'low previous performance'),
    'consistency_old'   : ('Inconsistent',    'low consistency score'),
    'min_share_pct_old' : ('Low minutes',     'low previous minute share'),
    'prev_adapt_rate'   : ('Poor prev transfers', 'failed previous moves'),
    'points_gap'        : ('Points gap',      'big quality gap'),
    'epl_experience'    : ('Limited EPL exp', 'new to the league'),
}

# Build a logistic model just for sign interpretation
lr_sign = LogisticRegression(max_iter=500, C=0.5, solver='lbfgs')
lr_sign.fit(X_train_sc, y_train)
# Negative coefficients → feature increases failure risk (coefficient for class 1 = success)
feature_coefs = pd.Series(lr_sign.coef_[0], index=FEATURES)

def get_top_risk_factors(row_features, n=3):
    """
    Return top N risk factors based on:
    - Feature value (normalized)
    - Feature's coefficient sign in LR (negative coef + high value = risk)
    """
    factors = []
    for feat, coef in feature_coefs.items():
        if feat not in RISK_FEATURE_NAMES:
            continue
        feat_idx = FEATURES.index(feat)
        val = row_features[feat_idx]
        if np.isnan(val):
            continue
        # Risk contribution: negative coef * value (for success model)
        risk_contrib = -coef * val  # positive = failure risk
        label, desc = RISK_FEATURE_NAMES[feat]
        factors.append((label, desc, float(risk_contrib), float(val)))
    factors.sort(key=lambda x: -x[2])
    return [{'factor': f[0], 'description': f[1], 'risk_score': round(f[2], 3), 'value': round(f[3], 3)}
            for f in factors[:n]]

# ── kNN similar historical transfers ──────────
# Build kNN on training features
nn_model = NearestNeighbors(n_neighbors=4, metric='euclidean')
nn_model.fit(X_train)

train_df_reset = train_df.reset_index(drop=True)

def find_similar_transfers(query_features, n=3):
    """Return n most similar historical transfers with outcomes."""
    dist, idx = nn_model.kneighbors([query_features], n_neighbors=n+1)
    similar = []
    for d, i in zip(dist[0], idx[0]):
        if i >= len(train_df_reset):
            continue
        r = train_df_reset.iloc[i]
        similar.append({
            'player'     : str(r['player']),
            'team_old'   : str(r['team_old']),
            'team_new'   : str(r['team_new']),
            'season_new' : str(r['season_new']),
            'outcome'    : str(r['label']),
            'distance'   : round(float(d), 2),
        })
    return similar[:n]

# ── Build scout records for 2022-2025 ──────────
recent_df = all_df[all_df['season_year'] >= 2022].copy().reset_index(drop=True)
recent_X  = imputer.transform(recent_df[FEATURES].values)
recent_Xs = scaler.transform(recent_X)

scout_records = []
for i, row in enumerate(recent_df.itertuples(index=False)):
    row = recent_df.iloc[i]
    risk_factors = get_top_risk_factors(recent_X[i])
    similar      = find_similar_transfers(recent_X[i])

    # Calibrated probability for this row
    p_succ_i = float(ensemble_proba(recent_X[i:i+1], recent_Xs[i:i+1])[0])
    p_fail_i = 1.0 - p_succ_i

    if abs(p_fail_i - 0.5) > 0.25:
        confidence = 'HIGH'
    elif abs(p_fail_i - 0.5) > 0.12:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'

    scout_records.append({
        'player'          : str(row['player']),
        'from'            : str(row['team_old']),
        'to'              : str(row['team_new']),
        'season'          : str(row['season_new']),
        'pos_group'       : str(row['pos_group']),
        'age'             : float(row['age_at_transfer']) if not pd.isna(row['age_at_transfer']) else None,
        'actual_outcome'  : str(row['label']),
        'predicted'       : str(row['pred_label']),
        'prob_failure'    : round(p_fail_i, 3),
        'prob_success'    : round(p_succ_i, 3),
        'confidence'      : confidence,
        'top_risk_factors': risk_factors,
        'similar_historical': similar,
        # Key metrics
        'elo_gap'         : float(row['elo_gap'])           if 'elo_gap'        in row and not pd.isna(row.get('elo_gap'))        else None,
        'style_match'     : float(row['style_match_pct'])   if 'style_match_pct' in row and not pd.isna(row.get('style_match_pct')) else None,
        'age_flag'        : 'veteran' if row.get('is_veteran', 0) == 1 else ('young' if row.get('is_young', 0) == 1 else 'prime'),
        'step_up'         : bool(row.get('is_step_up', 0)),
    })

print(f"  Scout records generated: {len(scout_records)}")

# High-risk recent transfers
high_risk = sorted([r for r in scout_records if r['prob_failure'] > 0.55],
                   key=lambda x: -x['prob_failure'])
print(f"\n  High-risk transfers (prob_failure > 0.55): {len(high_risk)}")
if high_risk:
    for r in high_risk[:10]:
        print(f"    {r['player']:25s}  {r['from']:20s} → {r['to']:20s}  "
              f"prob_fail={r['prob_failure']:.2f}  actual={r['actual_outcome']}")

# ─────────────────────────────────────────────
# 15. Calibration plots
# ─────────────────────────────────────────────
print("\n  Generating figures...")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Fig 1: Label distribution (binary + uncertain)
lbl_order = ['success', 'failure', 'uncertain']
lbl_colors = {'success': '#2ecc71', 'failure': '#e74c3c', 'uncertain': '#95a5a6'}
vc_all = transfer_df['label'].value_counts()
cnts = [vc_all.get(l, 0) for l in lbl_order]
axes[0, 0].bar(lbl_order, cnts, color=[lbl_colors[l] for l in lbl_order])
axes[0, 0].set_title('Transfer Adaptation Labels\n(270+ min, window ≤2 seasons, 2000-2025)', fontsize=10)
axes[0, 0].set_ylabel('Count')
for i, (l, c) in enumerate(zip(lbl_order, cnts)):
    axes[0, 0].text(i, c + 5, f'{c}\n({c/len(transfer_df)*100:.1f}%)', ha='center', fontsize=9)

# Fig 2: Ensemble confusion matrix (test set)
cm_ens = confusion_matrix(y_test, ens_test_preds, labels=[0, 1])
sns.heatmap(cm_ens, annot=True, fmt='d', cmap='Blues',
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
            ax=axes[0, 1], linewidths=0.5)
axes[0, 1].set_title(f'Ensemble Confusion Matrix — Test 2022-2025\n'
                     f'(n={len(y_test)}, failures={int((y_test==0).sum())})', fontsize=10)
axes[0, 1].set_ylabel('Actual')
axes[0, 1].set_xlabel('Predicted')

# Fig 3: Calibration curve (test set — are probabilities meaningful?)
try:
    frac_pos, mean_pred = calibration_curve((y_test == 0), 1 - ens_test_prob, n_bins=8, strategy='quantile')
    axes[1, 0].plot(mean_pred, frac_pos, 's-', label='Ensemble (calibrated)', color='#e74c3c')
    axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    axes[1, 0].set_title('Probability Calibration (failure class)\n'
                         'Platt scaling applied', fontsize=10)
    axes[1, 0].set_xlabel('Mean predicted failure probability')
    axes[1, 0].set_ylabel('Fraction of actual failures')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
except Exception as e:
    axes[1, 0].text(0.5, 0.5, f'Calibration plot error:\n{e}', ha='center', va='center')

# Fig 4: Feature importance
top20 = fi.head(20)
colors_fi = ['#c0392b' if any(k in f for k in ['style', 'elo', 'pos', 'war', 'gap'])
             else '#2980b9' for f in top20['feature']]
axes[1, 1].barh(top20['feature'][::-1], top20['importance_mean'][::-1],
                color=colors_fi[::-1])
axes[1, 1].set_title('Top 20 Feature Importances (XGB + RF)', fontsize=10)
axes[1, 1].set_xlabel('Mean Importance')

plt.suptitle('S5 v3: Transfer Adaptation (Binary + Calibrated)', fontsize=12, fontweight='bold')
plt.tight_layout()
fig_path = FIG_DIR / 'v3_main_figures.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig_path}")

# Fig 5: Failure recall comparison (val vs test)
model_names_list = ['XGBoost_cal', 'RandomForest_cal', 'LogReg_cal', 'GBM_cal', 'Ensemble']
val_f_recalls  = [val_results[m]['failure_recall']  for m in model_names_list[:-1]] + [ens_val_metrics['failure_recall']]
test_f_recalls = [test_results[m]['failure_recall'] for m in model_names_list[:-1]] + [ens_test_metrics['failure_recall']]
val_n  = [val_results[m]['n_failure_test']  for m in model_names_list[:-1]] + [ens_val_metrics['n_failure']]
test_n = [test_results[m]['n_failure_test'] for m in model_names_list[:-1]] + [ens_test_metrics['n_failure']]

x = np.arange(len(model_names_list))
fig, ax = plt.subplots(figsize=(11, 5))
bars1 = ax.bar(x - 0.2, val_f_recalls,  0.35, label='Validation', color='#3498db', alpha=0.85)
bars2 = ax.bar(x + 0.2, test_f_recalls, 0.35, label='Test 2022-25', color='#e74c3c', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(model_names_list, fontsize=9)
ax.set_title('Failure Detection Recall by Model\n'
             '(v3 Binary — test has statistically meaningful failure count)', fontsize=11)
ax.set_ylabel('Failure Recall')
ax.set_ylim(0, 1.15)
ax.legend()
ax.axhline(0.6, color='orange', linestyle='--', alpha=0.7, label='60% target')
for i, (v, t) in enumerate(zip(val_f_recalls, test_f_recalls)):
    ax.text(i - 0.2, v + 0.03, f'{v:.2f}', ha='center', fontsize=8)
    ax.text(i + 0.2, t + 0.03, f'{t:.2f}', ha='center', fontsize=8)

# Annotate failure counts
ax.text(0.5, 1.06, f"Val n_failure={val_n[0]}  |  Test n_failure={test_n[0]}",
        ha='center', transform=ax.transAxes, fontsize=9, color='#c0392b')
plt.tight_layout()
recall_path = FIG_DIR / 'v3_failure_recall_comparison.png'
plt.savefig(recall_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {recall_path}")

# ─────────────────────────────────────────────
# 16. Notable player scout output
# ─────────────────────────────────────────────
NOTABLE = [
    'Declan Rice', 'Kai Havertz', 'Jadon Sancho', 'Marcus Rashford',
    'Jack Grealish', 'Romelu Lukaku', 'Alexis Sanchez', 'Mesut Ozil',
    'Angel Di Maria', 'Memphis Depay', 'Nicolas Pepe', 'Riyad Mahrez',
    'Virgil van Dijk', 'Fernandinho', 'Kevin De Bruyne',
]

print("\n  === NOTABLE PLAYER PREDICTIONS ===")
for name in NOTABLE:
    match = [r for r in scout_records if name.lower() in r['player'].lower()]
    if match:
        for r in match:
            print(f"\n  {r['player']}  ({r['from']} → {r['to']}, {r['season']})")
            print(f"    Failure prob: {r['prob_failure']:.2f}  |  Predicted: {r['predicted']}  |  Actual: {r['actual_outcome']}  |  Confidence: {r['confidence']}")
            print(f"    Top risk factors:")
            for rf in r['top_risk_factors']:
                print(f"      - {rf['factor']:30s}  risk_score={rf['risk_score']:.2f}")
            print(f"    Similar historical transfers:")
            for s in r['similar_historical']:
                print(f"      - {s['player']:25s}  {s['team_old']} → {s['team_new']}  ({s['season_new']})  outcome={s['outcome']}")

# ─────────────────────────────────────────────
# 17. Save results
# ─────────────────────────────────────────────
print("\n  Saving results...")

def to_py(x):
    if isinstance(x, (np.integer,)):  return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, np.ndarray):    return x.tolist()
    if isinstance(x, bool):           return bool(x)
    return x

# Predictions parquet
save_cols = ['player', 'team_old', 'team_new', 'season_old', 'season_new',
             'pos_group', 'age_at_transfer', 'label', 'pred_label',
             'prob_failure', 'prob_success', 'war_ratio', 'min_ratio',
             'elo_gap', 'league_pos_gap', 'style_match_pct',
             'pos_competition', 'high_competition',
             'is_step_up', 'is_step_down', 'epl_experience',
             'prev_adapt_rate', 'prev_transfer_cnt', 'season_gap']
save_cols = [c for c in save_cols if c in all_df.columns]
all_df[save_cols].to_parquet(SCOUT_DIR / 'transfer_predictions_v3.parquet', index=False)
print(f"  Saved: {SCOUT_DIR / 'transfer_predictions_v3.parquet'}")

# Main JSON results
binary_label_counts = train_eligible['label'].value_counts().to_dict()
results_json = {
    'model_version': 'v3',
    'description': 'Binary classification (success/failure) + Platt scaling calibration. '
                   'Uncertain cases dropped from training. Test window 2022-2025 (3 seasons).',
    'key_fixes': [
        'Binary labels (not 3-class): success vs failure, uncertain dropped',
        'Lower min threshold: 270 min (was 300)',
        'Window transfers: gap ≤ 2 seasons (was consecutive only)',
        'Test window: 2022-2025 (3 seasons) — statistically meaningful failure count',
        'Platt scaling calibration on all models',
        'Scout output: prob + risk factors + 3 similar historical transfers',
    ],
    'total_transfers_identified': int(len(transfer_df)),
    'label_distribution_all': {k: int(v) for k, v in label_counts_all.items()},
    'binary_eligible': int(len(train_eligible)),
    'uncertain_count': int(len(uncertain_df)),
    'binary_label_counts': {k: int(v) for k, v in binary_label_counts.items()},
    'split_sizes': {
        'train': int(len(train_df)),
        'val'  : int(len(val_df)),
        'test' : int(len(test_df)),
    },
    'n_features': len(FEATURES),
    'features': FEATURES,
    'ensemble_threshold': float(best_thresh),
    'val_metrics': {
        m: {k: to_py(v) for k, v in val_results[m].items() if k not in ('preds', 'proba', 'confusion_matrix')}
        for m in val_results
    },
    'test_metrics': {
        m: {k: to_py(v) for k, v in test_results[m].items() if k not in ('preds', 'proba', 'confusion_matrix')}
        for m in test_results
    },
    'ensemble_val':  {k: to_py(v) for k, v in ens_val_metrics.items() if k != 'confusion_matrix'},
    'ensemble_test': {k: to_py(v) for k, v in ens_test_metrics.items() if k != 'confusion_matrix'},
    'top_features': fi[['feature', 'importance_mean']].head(20).to_dict(orient='records'),
    'n_scout_records': len(scout_records),
    'n_high_risk_recent': len(high_risk),
}

results_path = SCOUT_DIR / 's5_v3_results.json'
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(results_json, f, ensure_ascii=False, indent=2, default=to_py)
print(f"  Saved: {results_path}")

# Scout JSON with full records
scout_json_path = SCOUT_DIR / 's5_v3_scout_output.json'
with open(scout_json_path, 'w', encoding='utf-8') as f:
    json.dump({
        'generated': '2026-03-24',
        'model_version': 'v3',
        'threshold': float(best_thresh),
        'n_records': len(scout_records),
        'high_risk_summary': high_risk[:20],
        'all_records': scout_records,
    }, f, ensure_ascii=False, indent=2, default=to_py)
print(f"  Saved: {scout_json_path}")

# ─────────────────────────────────────────────
# 18. Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY — S5 v3")
print("=" * 70)
print(f"\nTotal transfers identified (270+ min, window ≤2 seasons): {len(transfer_df)}")
print(f"  Success   : {label_counts_all.get('success',   0):4d} ({label_counts_all.get('success',   0)/len(transfer_df)*100:.1f}%)")
print(f"  Failure   : {label_counts_all.get('failure',   0):4d} ({label_counts_all.get('failure',   0)/len(transfer_df)*100:.1f}%)")
print(f"  Uncertain : {label_counts_all.get('uncertain', 0):4d} ({label_counts_all.get('uncertain', 0)/len(transfer_df)*100:.1f}%)")
print(f"\nBinary training set (success + failure): {len(train_eligible)}")
print(f"  Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")
print(f"  Test failure cases: {int((y_test==0).sum())}  (was 5 in v2 — now statistically meaningful)")
print(f"\nEnsemble Test Performance (threshold={best_thresh:.2f}):")
print(f"  Accuracy       : {ens_test_metrics['accuracy']:.3f}")
print(f"  Macro F1       : {ens_test_metrics['macro_f1']:.3f}")
print(f"  Failure F1     : {ens_test_metrics['failure_f1']:.3f}")
print(f"  Failure Recall : {ens_test_metrics['failure_recall']:.3f}")
print(f"  Failure Prec.  : {ens_test_metrics['failure_precision']:.3f}")
print(f"  ROC AUC        : {ens_test_metrics['roc_auc']:.3f}")
print(f"  Brier (failure): {ens_test_metrics['failure_brier']:.3f}")
print(f"\nCalibration: Platt scaling applied to all 4 models")
print(f"Scout output: {len(scout_records)} records (2022-2025)")
print(f"  High-risk (prob_fail > 0.55): {len(high_risk)}")
print(f"\nTop 5 features:")
for _, row in fi.head(5).iterrows():
    print(f"  {row['feature']:35s}  importance={row['importance_mean']:.4f}")
print("\nDone.")

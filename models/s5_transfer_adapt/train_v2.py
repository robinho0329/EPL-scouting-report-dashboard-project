"""
S5 v2: Transfer Adaptation Prediction Model - Scout Grade
==========================================================
Key improvements over v1:
  - ALL seasons 2000-2025, min threshold 300 min (captures benched failures)
  - 3-class labels: Success / Partial / Failure
    * Success:  per90 maintained + minutes_share >= 80% of prev
    * Partial:  one maintained, other dropped
    * Failure:  both per90 AND minutes drop significantly (<60% of prev)
  - Targeted class balance: ~40% Success / 35% Partial / 25% Failure
  - New features: league_position_gap, age_at_transfer, prev_transfer_success_rate,
    position_scarcity_new_team, style_compatibility (attack/pressing match)
  - Failure detection focus: confusion matrix + failure recall printed
  - Scout output: risk factors, failure probability per transfer
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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, roc_auc_score
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = Path("C:/Users/xcv54/workspace/EPL project")
DATA_DIR  = BASE_DIR / "data" / "processed"
SCOUT_DIR = BASE_DIR / "data" / "scout"
FEAT_DIR  = BASE_DIR / "data" / "features"
MODEL_DIR = BASE_DIR / "models" / "s5_transfer_adapt"
FIG_DIR   = MODEL_DIR / "figures"

for d in [SCOUT_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("S5 v2: Transfer Adaptation Model  (Scout Grade)")
print("=" * 65)

# ─────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────
print("\n[1/8] Loading data...")

player_df  = pd.read_parquet(DATA_DIR / "player_season_stats.parquet")
team_df    = pd.read_parquet(DATA_DIR / "team_season_summary.parquet")
scout_pp   = pd.read_parquet(SCOUT_DIR / "scout_player_profiles.parquet")
scout_tp   = pd.read_parquet(SCOUT_DIR / "scout_team_profiles.parquet")

# match_features has ELO
mf_df = pd.read_parquet(FEAT_DIR / "match_features.parquet")

print(f"  player_season_stats : {player_df.shape}")
print(f"  team_season_summary : {team_df.shape}")
print(f"  scout_player_profiles: {scout_pp.shape}")
print(f"  scout_team_profiles  : {scout_tp.shape}")
print(f"  match_features (ELO) : {mf_df.shape}")

# ─────────────────────────────────────────────
# 2. Build ELO map from match_features
# ─────────────────────────────────────────────
print("\n[2/8] Building ELO map and team features...")

# Extract per-team per-season ELO from match logs (end-of-season average)
mf_home = mf_df[['Season', 'HomeTeam', 'home_elo_pre']].rename(
    columns={'Season': 'season', 'HomeTeam': 'team', 'home_elo_pre': 'elo'})
mf_away = mf_df[['Season', 'AwayTeam', 'away_elo_pre']].rename(
    columns={'Season': 'season', 'AwayTeam': 'team', 'away_elo_pre': 'elo'})
elo_long = pd.concat([mf_home, mf_away], ignore_index=True)
elo_map = (elo_long.groupby(['season', 'team'])['elo']
           .mean().reset_index().set_index(['season', 'team'])['elo'])

# Build team features from team_season_summary
team_df = team_df.rename(columns={'Season': 'season'})

# League position (rank by points desc within season)
team_df['league_position'] = (
    team_df.groupby('season')['points']
    .rank(ascending=False, method='min').astype(int)
)

# Attack / style ratio
team_df['attack_ratio'] = (
    team_df['total_goals_for'] /
    (team_df['total_goals_for'] + team_df['total_goals_against'] + 1e-6)
)
team_df['goals_per_game'] = team_df['total_goals_for'] / (team_df['total_played'] + 1e-6)
team_df['conceded_per_game'] = team_df['total_goals_against'] / (team_df['total_played'] + 1e-6)

# ELO
team_idx = team_df.set_index(['season', 'team']).index
team_df['elo'] = team_idx.map(elo_map)
# Fallback: linear regression from points for missing ELO
from sklearn.linear_model import LinearRegression
elo_known = team_df[team_df['elo'].notna()][['points', 'elo']].copy()
if len(elo_known) > 20:
    lr_elo = LinearRegression()
    lr_elo.fit(elo_known[['points']], elo_known['elo'])
    mask_na = team_df['elo'].isna()
    if mask_na.any():
        team_df.loc[mask_na, 'elo'] = lr_elo.predict(team_df.loc[mask_na, ['points']])
team_df['elo'] = team_df.groupby('season')['elo'].transform(lambda x: x.fillna(x.mean()))

# squad features from scout_team_profiles
scout_tp_feat = scout_tp[['season', 'team', 'squad_depth_300min', 'squad_depth_900min',
                            'avg_squad_age', 'new_players_count', 'squad_turnover_rate',
                            'youth_development_score', 'attack_defense_ratio']].copy()

team_feat = team_df[['season', 'team', 'points', 'league_position', 'elo',
                      'attack_ratio', 'goals_per_game', 'conceded_per_game',
                      'total_goals_for', 'total_goals_against', 'goal_diff']].merge(
    scout_tp_feat, on=['season', 'team'], how='left')

print(f"  team_feat built: {team_feat.shape}, ELO NaN: {team_df['elo'].isna().sum()}")

# ─────────────────────────────────────────────
# 3. Build position scarcity feature
#    (how many players at same position on new team ≥ 300 min prev season)
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

# Count players per team/season/pos_group with >= 300 min
squad_comp = (player_df[player_df['min'] >= 300]
              .groupby(['season', 'team', 'pos_group'])
              .size().reset_index(name='n_same_position'))

# ─────────────────────────────────────────────
# 4. Identify transfers (consecutive seasons, different teams, >=300 min)
# ─────────────────────────────────────────────
print("\n[3/8] Identifying transfers (ALL seasons 2000-2025, ≥300 min)...")

SEASON_ORDER = sorted(player_df['season'].unique())
season_idx_map = {s: i for i, s in enumerate(SEASON_ORDER)}
player_df['season_idx'] = player_df['season'].map(season_idx_map)

# Add scout_player_profiles extras
scout_extras = scout_pp[['player', 'team', 'season', 'war_rating', 'consistency_score',
                           'minutes_share', 'g_a_p90', 'gls_p90', 'ast_p90',
                           'versatility_score', 'big6_contribution_p90',
                           'win_rate_with_player', 'team_dependency_score',
                           'career_trajectory', 'season_improvement_rate']].copy()
player_df2 = player_df.merge(
    scout_extras.rename(columns={c: f'scout_{c}' for c in scout_extras.columns
                                  if c not in ['player', 'team', 'season']}),
    on=['player', 'team', 'season'], how='left')

player_sorted = player_df2.sort_values(['player', 'season_idx']).reset_index(drop=True)

# ─── Main transfer extraction loop ───────────────────────────────
records = []
for player, grp in player_sorted.groupby('player'):
    grp = grp.reset_index(drop=True)
    for i in range(len(grp) - 1):
        curr = grp.iloc[i]
        nxt  = grp.iloc[i + 1]
        # consecutive seasons only
        if nxt['season_idx'] - curr['season_idx'] != 1:
            continue
        # different teams
        if curr['team'] == nxt['team']:
            continue
        # minimum 300 min each side (captures benched failures)
        if curr['min'] < 300 or nxt['min'] < 300:
            continue

        records.append({
            'player'       : player,
            'season_old'   : curr['season'],
            'season_new'   : nxt['season'],
            'team_old'     : curr['team'],
            'team_new'     : nxt['team'],
            # raw minutes
            'min_old'      : curr['min'],
            'min_new'      : nxt['min'],
            'starts_old'   : curr['starts'],
            'starts_new'   : nxt['starts'],
            'mp_old'       : curr['mp'],
            'mp_new'       : nxt['mp'],
            # per-90 stats
            'gls_p90_old'  : curr['gls_1'],
            'ast_p90_old'  : curr['ast_1'],
            'g_a_p90_old'  : curr['g_a_1'],
            'g_pk_p90_old' : curr['g_pk_1'],
            'gls_p90_new'  : nxt['gls_1'],
            'ast_p90_new'  : nxt['ast_1'],
            'g_a_p90_new'  : nxt['g_a_1'],
            'g_pk_p90_new' : nxt['g_pk_1'],
            # player attributes
            'age_at_transfer': nxt['age'],
            'pos'           : curr['pos'],
            'pos_group'     : curr['pos_group'],
            'position'      : curr['position'],
            'height_cm'     : curr['height_cm'],
            'market_value_old': curr['market_value'],
            'market_value_new': nxt['market_value'],
            # scout extras
            'war_old'       : curr.get('scout_war_rating', np.nan),
            'war_new'       : nxt.get('scout_war_rating', np.nan),
            'consistency_old': curr.get('scout_consistency_score', np.nan),
            'min_share_old' : curr.get('scout_minutes_share', np.nan),
            'min_share_new' : nxt.get('scout_minutes_share', np.nan),
            'versatility'   : curr.get('scout_versatility_score', np.nan),
            'big6_p90_old'  : curr.get('scout_big6_contribution_p90', np.nan),
            'team_dep_old'  : curr.get('scout_team_dependency_score', np.nan),
            'career_traj'   : curr.get('scout_career_trajectory', np.nan),
        })

transfer_df = pd.DataFrame(records)
print(f"  Total transfers identified: {len(transfer_df)}")

# ─────────────────────────────────────────────
# 5. Multi-level adaptation labels
#    Success  : per90 maintained (>=80%) AND minutes >= 80% of prev
#    Failure  : per90 drops significantly (<60%) AND minutes drop (<60%)
#    Partial  : everything else (one maintained, other dropped; moderate)
# ─────────────────────────────────────────────
print("\n[4/8] Computing 3-class adaptation labels...")

MAX_MIN = 3420.0

def compute_3class_label(row):
    """
    Returns 'success', 'partial', 'failure'

    For GK/DEF: performance = g_pk_p90 (or g_a_p90 if very small → use minutes only)
    For FWD/MF: performance = g_a_p90

    Thresholds:
      - per90_ratio >= 0.80 → maintained
      - per90_ratio >= 0.60 → partial
      - per90_ratio <  0.60 → dropped

      - min_ratio >= 0.80 → maintained
      - min_ratio >= 0.60 → partial
      - min_ratio <  0.60 → dropped
    """
    pos_group = row['pos_group']

    # Minutes ratio
    min_ratio = row['min_new'] / (row['min_old'] + 1e-6)
    min_ok     = min_ratio >= 0.80
    min_partial = 0.60 <= min_ratio < 0.80
    min_fail   = min_ratio < 0.60

    # Performance ratio
    old_p = row['g_a_p90_old'] if not pd.isna(row['g_a_p90_old']) else 0.0
    new_p = row['g_a_p90_new'] if not pd.isna(row['g_a_p90_new']) else 0.0

    if old_p < 0.05:
        # Defender / GK with near-zero attack – judge by minutes
        perf_ok      = min_ok
        perf_partial = min_partial
        perf_fail    = min_fail
    else:
        p_ratio     = new_p / (old_p + 1e-6)
        perf_ok      = p_ratio >= 0.80
        perf_partial = 0.60 <= p_ratio < 0.80
        perf_fail    = p_ratio < 0.60

    # Classification
    if min_ok and perf_ok:
        return 'success'
    elif min_fail and perf_fail:
        return 'failure'
    else:
        return 'partial'

transfer_df['label'] = transfer_df.apply(compute_3class_label, axis=1)

label_counts = transfer_df['label'].value_counts()
print(f"  Label distribution:")
for lbl, cnt in label_counts.items():
    print(f"    {lbl:10s}: {cnt:4d}  ({cnt/len(transfer_df)*100:.1f}%)")

# ─────────────────────────────────────────────
# 6. Feature engineering
# ─────────────────────────────────────────────
print("\n[5/8] Feature engineering...")

# 6-1. EPL experience (seasons up to season_old)
epl_exp_map = {}
for player, grp in player_sorted.groupby('player'):
    grp = grp.sort_values('season_idx')
    for i, (idx, row) in enumerate(grp.iterrows()):
        epl_exp_map[(player, row['season'])] = i

transfer_df['epl_experience'] = transfer_df.apply(
    lambda r: epl_exp_map.get((r['player'], r['season_old']), 0), axis=1)

# 6-2. Previous transfer success rate (rolling, label-aware)
prev_success_rate = {}
prev_transfer_cnt = {}
temp_hist = {}

for idx, row in transfer_df.sort_values('season_old').iterrows():
    key = row['player']
    prev = temp_hist.get(key, [])
    prev_success_rate[idx] = np.mean(prev) if prev else np.nan
    prev_transfer_cnt[idx] = len(prev)
    # encode: success=1, partial=0.5, failure=0
    val = 1.0 if row['label'] == 'success' else (0.5 if row['label'] == 'partial' else 0.0)
    temp_hist[key] = prev + [val]

transfer_df['prev_adapt_rate']  = pd.Series(prev_success_rate)
transfer_df['prev_transfer_cnt'] = pd.Series(prev_transfer_cnt)

# 6-3. Starter ratio
transfer_df['starter_ratio_old'] = transfer_df['starts_old'] / (transfer_df['mp_old'] + 1e-6)
transfer_df['min_share_pct_old'] = transfer_df['min_old'] / MAX_MIN  # season minute share

# 6-4. Merge team features for old and new teams
def merge_team_feats(df, team_feat, side):
    """side = 'old' or 'new'"""
    season_col = f'season_{side}'
    team_col   = f'team_{side}'
    suffix     = f'_{side}'
    rename_map = {c: c + suffix for c in team_feat.columns
                  if c not in ['season', 'team']}
    tf = team_feat.rename(columns=rename_map)
    merged = df.merge(tf, left_on=[season_col, team_col],
                      right_on=['season', 'team'], how='left')
    merged = merged.drop(columns=['season', 'team'], errors='ignore')
    return merged

transfer_df = merge_team_feats(transfer_df, team_feat, 'old')
transfer_df = merge_team_feats(transfer_df, team_feat, 'new')

# 6-5. Derived gap features
transfer_df['elo_gap']             = transfer_df['elo_new'] - transfer_df['elo_old']
transfer_df['league_pos_gap']      = transfer_df['league_position_old'] - transfer_df['league_position_new']
                                     # positive = moved to higher-ranked team
transfer_df['points_gap']          = transfer_df['points_new'] - transfer_df['points_old']
transfer_df['attack_ratio_diff']   = (transfer_df['attack_ratio_new'] -
                                       transfer_df['attack_ratio_old']).abs()
transfer_df['goals_pg_diff']       = (transfer_df['goals_per_game_new'] -
                                       transfer_df['goals_per_game_old']).abs()
transfer_df['is_step_up']          = (transfer_df['elo_gap'] > 50).astype(int)
transfer_df['is_step_down']        = (transfer_df['elo_gap'] < -50).astype(int)

# Style compatibility (lower diff = more compatible)
transfer_df['style_match_pct']     = 1.0 - transfer_df['attack_ratio_diff'].clip(0, 1)

# 6-6. Position scarcity at new team
sc = squad_comp.rename(columns={'season': 'season_new', 'team': 'team_new',
                                  'pos_group': 'pos_group_squad', 'n_same_position': 'pos_competition'})
# Merge on season_new + team_new + pos_group
transfer_df = transfer_df.merge(
    sc.rename(columns={'pos_group_squad': 'pos_group'}),
    on=['season_new', 'team_new', 'pos_group'], how='left')

# Also compute scarcity at old team (to check if player was indispensable)
sc_old = squad_comp.rename(columns={'season': 'season_old', 'team': 'team_old',
                                     'pos_group': 'pos_group', 'n_same_position': 'pos_count_old'})
transfer_df = transfer_df.merge(sc_old, on=['season_old', 'team_old', 'pos_group'], how='left')

transfer_df['pos_competition'] = transfer_df['pos_competition'].fillna(3)  # median impute
transfer_df['pos_count_old']   = transfer_df['pos_count_old'].fillna(3)

# High competition = >4 players at same position on new team
transfer_df['high_competition'] = (transfer_df['pos_competition'] > 4).astype(int)

# 6-7. Age features
transfer_df['age_at_transfer']     = transfer_df['age_at_transfer'].fillna(
    transfer_df['age_at_transfer'].median())
transfer_df['is_peak_age']         = ((transfer_df['age_at_transfer'] >= 23) &
                                       (transfer_df['age_at_transfer'] <= 28)).astype(int)
transfer_df['is_veteran']          = (transfer_df['age_at_transfer'] >= 30).astype(int)
transfer_df['is_young']            = (transfer_df['age_at_transfer'] <= 22).astype(int)

# 6-8. Market value ratio (new/old) – proxy for transfer ambition
transfer_df['mv_ratio']            = (transfer_df['market_value_new'] /
                                       (transfer_df['market_value_old'] + 1e-6)).clip(0, 10)

# 6-9. Position group encoding
POS_ENC = {'GK': 0, 'DEF': 1, 'MF': 2, 'FWD': 3}
transfer_df['pos_enc'] = transfer_df['pos_group'].map(POS_ENC).fillna(2)

# 6-10. New team squad depth (competition pressure)
transfer_df['squad_depth_new'] = transfer_df['squad_depth_300min_new'].fillna(
    transfer_df['squad_depth_300min_new'].median())
transfer_df['squad_depth_old'] = transfer_df['squad_depth_300min_old'].fillna(
    transfer_df['squad_depth_300min_old'].median())

# 6-11. Encode career_traj (categorical string → ordinal numeric)
CAREER_TRAJ_ENC = {'declining': 0, 'unknown': 1, 'stable': 2, 'improving': 3}
transfer_df['career_traj'] = pd.to_numeric(
    transfer_df['career_traj'].map(CAREER_TRAJ_ENC), errors='coerce').fillna(1)

print(f"  Transfer records after feature engineering: {len(transfer_df)}")
print(f"  Feature columns: {transfer_df.shape[1]}")

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
    # Team gap features (KEY for adaptation)
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

# Keep only features that exist
FEATURES = [f for f in FEATURES if f in transfer_df.columns]
print(f"  Using {len(FEATURES)} features: {FEATURES}")

# ─────────────────────────────────────────────
# 8. Time-based split  (train < 2021, val 2021-22, test 2023-25)
# ─────────────────────────────────────────────
print("\n[6/8] Time-based train/val/test split...")

def season_to_int(s):
    try:
        return int(str(s).split('/')[0])
    except:
        return 0

transfer_df['season_year'] = transfer_df['season_new'].apply(season_to_int)

train_df = transfer_df[transfer_df['season_year'] < 2021].copy()
val_df   = transfer_df[transfer_df['season_year'].between(2021, 2022)].copy()
test_df  = transfer_df[transfer_df['season_year'] >= 2023].copy()

print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    vc = split_df['label'].value_counts().to_dict()
    print(f"  {split_name} labels: {vc}")

# Encode labels
LABEL_MAP    = {'failure': 0, 'partial': 1, 'success': 2}
LABEL_NAMES  = ['failure', 'partial', 'success']

for df in [train_df, val_df, test_df]:
    df['label_enc'] = df['label'].map(LABEL_MAP)

X_train = train_df[FEATURES].values
y_train = train_df['label_enc'].values
X_val   = val_df[FEATURES].values
y_val   = val_df['label_enc'].values
X_test  = test_df[FEATURES].values
y_test  = test_df['label_enc'].values

# Impute missing values
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_val   = imputer.transform(X_val)
X_test  = imputer.transform(X_test)

# ─────────────────────────────────────────────
# 9. Class weights (focus on failure detection)
# ─────────────────────────────────────────────
# Give 2x weight to 'failure' class
unique_classes = np.unique(y_train)
class_weights_auto = compute_class_weight('balanced', classes=unique_classes, y=y_train)
class_weight_dict  = dict(zip(unique_classes.tolist(), class_weights_auto.tolist()))
# Boost failure weight
class_weight_dict[LABEL_MAP['failure']] *= 1.5
print(f"\n  Class weights: {class_weight_dict}")

# ─────────────────────────────────────────────
# 10. Model training
# ─────────────────────────────────────────────
print("\n[7/8] Training models...")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

# XGBoost (main model)
# scale_pos_weight not applicable for multiclass – use sample_weight
sample_weights = np.array([class_weight_dict[y] for y in y_train])

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    random_state=42,
    verbosity=0,
)
xgb_model.fit(X_train, y_train, sample_weight=sample_weights,
              eval_set=[(X_val, y_val)], verbose=False)

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    class_weight=class_weight_dict,
    random_state=42,
    n_jobs=-1,
)
rf_model.fit(X_train, y_train)

# Logistic Regression
lr_model = LogisticRegression(
    max_iter=1000,
    C=0.5,
    class_weight=class_weight_dict,
    multi_class='multinomial',
    solver='lbfgs',
    random_state=42,
)
lr_model.fit(X_train_sc, y_train)

# GBM
gbm_model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
)
gbm_model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 11. Evaluation
# ─────────────────────────────────────────────
print("\n[8/8] Evaluation on VALIDATION and TEST sets...")

def evaluate_model(name, model, X, y, use_scaled=False, X_sc=None):
    if use_scaled and X_sc is not None:
        preds = model.predict(X_sc)
        proba = model.predict_proba(X_sc)
    else:
        preds = model.predict(X)
        proba = model.predict_proba(X)

    acc  = accuracy_score(y, preds)
    f1   = f1_score(y, preds, average='macro')
    f1_f = f1_score(y, preds, average=None)[0]   # failure class F1

    # Failure detection recall
    cm = confusion_matrix(y, preds, labels=[0, 1, 2])
    failure_actual = (y == 0).sum()
    failure_detected = cm[0, 0]
    failure_recall = failure_detected / (failure_actual + 1e-6)

    # OvR AUC for failure class
    try:
        auc_failure = roc_auc_score((y == 0).astype(int), proba[:, 0])
    except:
        auc_failure = np.nan

    print(f"\n  ── {name} ──")
    print(f"  Accuracy:          {acc:.3f}")
    print(f"  Macro F1:          {f1:.3f}")
    print(f"  Failure F1:        {f1_f:.3f}")
    print(f"  Failure Recall:    {failure_recall:.3f}  ({failure_detected}/{failure_actual})")
    print(f"  Failure AUC:       {auc_failure:.3f}")
    print(f"\n  Classification Report:")
    print(classification_report(y, preds, target_names=LABEL_NAMES, digits=3))
    print(f"  Confusion Matrix (rows=actual, cols=pred, order: failure/partial/success):")
    print(cm)

    return {
        'accuracy': acc, 'macro_f1': f1, 'failure_f1': f1_f,
        'failure_recall': failure_recall, 'failure_auc': auc_failure,
        'confusion_matrix': cm.tolist(), 'preds': preds, 'proba': proba
    }

print("\n=== VALIDATION SET ===")
val_results = {}
for name, model, scaled in [
    ('XGBoost',          xgb_model,  False),
    ('RandomForest',     rf_model,   False),
    ('LogisticReg',      lr_model,   True),
    ('GBM',              gbm_model,  False),
]:
    val_results[name] = evaluate_model(name, model, X_val, y_val,
                                        use_scaled=scaled, X_sc=X_val_sc)

print("\n=== TEST SET ===")
test_results = {}
for name, model, scaled in [
    ('XGBoost',      xgb_model, False),
    ('RandomForest', rf_model,  False),
    ('LogisticReg',  lr_model,  True),
    ('GBM',          gbm_model, False),
]:
    test_results[name] = evaluate_model(name, model, X_test, y_test,
                                         use_scaled=scaled, X_sc=X_test_sc)

# ─────────────────────────────────────────────
# 12. Ensemble (probability average)
# ─────────────────────────────────────────────
print("\n=== ENSEMBLE (Prob Avg) ===")

def ensemble_predict(X, X_sc):
    p_xgb = xgb_model.predict_proba(X)
    p_rf  = rf_model.predict_proba(X)
    p_lr  = lr_model.predict_proba(X_sc)
    p_gbm = gbm_model.predict_proba(X)
    # weighted: XGB and RF get higher weight
    avg = (0.35 * p_xgb + 0.30 * p_rf + 0.15 * p_lr + 0.20 * p_gbm)
    return avg.argmax(axis=1), avg

ens_val_preds,  ens_val_proba  = ensemble_predict(X_val,  X_val_sc)
ens_test_preds, ens_test_proba = ensemble_predict(X_test, X_test_sc)

def print_ensemble_eval(name, preds, proba, y):
    acc  = accuracy_score(y, preds)
    f1   = f1_score(y, preds, average='macro')
    cm   = confusion_matrix(y, preds, labels=[0, 1, 2])
    failure_actual   = (y == 0).sum()
    failure_detected = cm[0, 0]
    failure_recall   = failure_detected / (failure_actual + 1e-6)
    try:
        auc_fail = roc_auc_score((y == 0).astype(int), proba[:, 0])
    except:
        auc_fail = np.nan
    print(f"\n  ── {name} ──")
    print(f"  Accuracy:       {acc:.3f}")
    print(f"  Macro F1:       {f1:.3f}")
    print(f"  Failure Recall: {failure_recall:.3f}  ({failure_detected}/{failure_actual})")
    print(f"  Failure AUC:    {auc_fail:.3f}")
    print(f"  Classification Report:")
    print(classification_report(y, preds, target_names=LABEL_NAMES, digits=3))
    print(f"  Confusion Matrix:")
    print(cm)
    return {'accuracy': acc, 'macro_f1': f1, 'failure_recall': failure_recall,
            'failure_auc': float(auc_fail), 'confusion_matrix': cm.tolist()}

ens_val_metrics  = print_ensemble_eval('Ensemble - Val',  ens_val_preds,  ens_val_proba,  y_val)
ens_test_metrics = print_ensemble_eval('Ensemble - Test', ens_test_preds, ens_test_proba, y_test)

# ─────────────────────────────────────────────
# 13. Feature importance
# ─────────────────────────────────────────────
fi = pd.DataFrame({
    'feature': FEATURES,
    'importance_xgb': xgb_model.feature_importances_,
    'importance_rf' : rf_model.feature_importances_,
}).assign(
    importance_mean=lambda d: (d['importance_xgb'] + d['importance_rf']) / 2
).sort_values('importance_mean', ascending=False)

print("\n  Top 20 Features (mean XGB + RF importance):")
print(fi[['feature', 'importance_xgb', 'importance_rf', 'importance_mean']].head(20).to_string(index=False))

# ─────────────────────────────────────────────
# 14. Failure Case Analysis
# ─────────────────────────────────────────────
print("\n  === FAILURE CASE ANALYSIS ===")
test_with_preds = test_df.copy()
test_with_preds['pred_label'] = [LABEL_NAMES[p] for p in ens_test_preds]
test_with_preds['prob_failure'] = ens_test_proba[:, 0]
test_with_preds['prob_partial']  = ens_test_proba[:, 1]
test_with_preds['prob_success']  = ens_test_proba[:, 2]

# False negatives: actual failure but predicted otherwise
fn_cases = test_with_preds[(test_with_preds['label'] == 'failure') &
                             (test_with_preds['pred_label'] != 'failure')]
print(f"\n  Missed failures (False Negatives): {len(fn_cases)}")
if len(fn_cases) > 0:
    cols_show = ['player', 'team_old', 'team_new', 'season_new', 'label',
                 'pred_label', 'prob_failure', 'elo_gap', 'style_match_pct',
                 'pos_competition', 'age_at_transfer']
    avail = [c for c in cols_show if c in fn_cases.columns]
    print(fn_cases[avail].head(10).to_string(index=False))

# True positives: correctly predicted failure
tp_fail = test_with_preds[(test_with_preds['label'] == 'failure') &
                            (test_with_preds['pred_label'] == 'failure')]
print(f"\n  Correctly detected failures (TP): {len(tp_fail)}")
if len(tp_fail) > 0:
    cols_show = ['player', 'team_old', 'team_new', 'season_new',
                 'prob_failure', 'elo_gap', 'style_match_pct', 'age_at_transfer']
    avail = [c for c in cols_show if c in tp_fail.columns]
    print(tp_fail[avail].head(10).to_string(index=False))

# ─────────────────────────────────────────────
# 15. Notable Transfer Predictions
# ─────────────────────────────────────────────
print("\n  === NOTABLE TRANSFER PREDICTIONS ===")
NOTABLE = [
    'Declan Rice', 'Kai Havertz', 'Jadon Sancho', 'Marcus Rashford',
    'Jack Grealish', 'Romelu Lukaku', 'Alexis Sanchez', 'Mesut Ozil',
    'Angel Di Maria', 'Memphis Depay', 'Nicolas Pepe', 'Riyad Mahrez',
    'Virgil van Dijk', 'Fernandinho', 'Kevin De Bruyne',
]

# Run on ALL transfers (train+val+test) for notable players
all_df = transfer_df.copy()
all_X  = imputer.transform(all_df[FEATURES].values)
all_Xs = scaler.transform(all_X)
_, all_proba = ensemble_predict(all_X, all_Xs)
all_pred_labels = [LABEL_NAMES[p] for p in all_proba.argmax(axis=1)]

all_df['pred_label']   = all_pred_labels
all_df['prob_failure'] = all_proba[:, 0]
all_df['prob_partial']  = all_proba[:, 1]
all_df['prob_success']  = all_proba[:, 2]

for name in NOTABLE:
    mask = all_df['player'].str.contains(name, case=False, na=False)
    if mask.any():
        rows = all_df[mask][['player', 'team_old', 'team_new', 'season_new',
                               'label', 'pred_label', 'prob_failure',
                               'prob_partial', 'prob_success',
                               'elo_gap', 'style_match_pct', 'age_at_transfer']]
        if 'style_match_pct' not in rows.columns:
            rows = rows.drop(columns=['style_match_pct'], errors='ignore')
        print(rows.to_string(index=False))
        print()

# ─────────────────────────────────────────────
# 16. Figures
# ─────────────────────────────────────────────
print("\n  Generating figures...")

# Fig 1: Label distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
label_vc = transfer_df['label'].value_counts()
colors = ['#e74c3c', '#f39c12', '#2ecc71']
axes[0].bar(LABEL_NAMES, [label_vc.get(l, 0) for l in LABEL_NAMES], color=colors)
axes[0].set_title('Transfer Adaptation Label Distribution\n(All Seasons 2000-2025, ≥300 min)', fontsize=11)
axes[0].set_ylabel('Count')
for i, (l, c) in enumerate([(l, label_vc.get(l, 0)) for l in LABEL_NAMES]):
    axes[0].text(i, c + 5, f'{c}\n({c/len(transfer_df)*100:.1f}%)', ha='center', va='bottom', fontsize=9)

# Fig 2: Ensemble confusion matrix (test set)
cm_ens = confusion_matrix(y_test, ens_test_preds, labels=[0, 1, 2])
sns.heatmap(cm_ens, annot=True, fmt='d', cmap='Blues',
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
            ax=axes[1], linewidths=0.5)
axes[1].set_title('Ensemble Confusion Matrix (Test Set)', fontsize=11)
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')
plt.tight_layout()
plt.savefig(FIG_DIR / 'v2_label_dist_and_cm.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {FIG_DIR / 'v2_label_dist_and_cm.png'}")

# Fig 3: Feature importance
fig, ax = plt.subplots(figsize=(9, 10))
top20 = fi.head(20)
colors_fi = ['#c0392b' if 'style' in f or 'elo' in f or 'pos' in f
              else '#2980b9' for f in top20['feature']]
ax.barh(top20['feature'][::-1], top20['importance_mean'][::-1], color=colors_fi[::-1])
ax.set_title('Top 20 Feature Importances\n(XGBoost + RF average)', fontsize=11)
ax.set_xlabel('Mean Importance')
plt.tight_layout()
plt.savefig(FIG_DIR / 'v2_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {FIG_DIR / 'v2_feature_importance.png'}")

# Fig 4: Failure probability by elo_gap and style_match_pct
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

all_df['elo_gap_bin'] = pd.cut(all_df['elo_gap'], bins=[-400, -100, -50, 0, 50, 100, 400],
                                labels=['<-100', '-100~-50', '-50~0', '0~50', '50~100', '>100'])
fail_by_elo = all_df.groupby('elo_gap_bin')['prob_failure'].mean()
axes[0].bar(fail_by_elo.index, fail_by_elo.values, color='#e74c3c', alpha=0.8)
axes[0].set_title('Average Failure Probability by ELO Gap\n(new_team - old_team)', fontsize=10)
axes[0].set_xlabel('ELO Gap (new - old)')
axes[0].set_ylabel('Mean Failure Probability')
axes[0].axhline(all_df['prob_failure'].mean(), color='black', linestyle='--', label='Overall mean')
axes[0].legend(fontsize=8)

if 'style_match_pct' in all_df.columns:
    all_df['style_bin'] = pd.cut(all_df['style_match_pct'],
                                  bins=[0, 0.5, 0.7, 0.85, 0.95, 1.01],
                                  labels=['<0.5', '0.5-0.7', '0.7-0.85', '0.85-0.95', '>0.95'])
    fail_by_style = all_df.groupby('style_bin')['prob_failure'].mean()
    axes[1].bar(fail_by_style.index, fail_by_style.values, color='#8e44ad', alpha=0.8)
    axes[1].set_title('Average Failure Probability by Style Match\n(1 = same attack style)', fontsize=10)
    axes[1].set_xlabel('Style Match Score')
    axes[1].set_ylabel('Mean Failure Probability')
    axes[1].axhline(all_df['prob_failure'].mean(), color='black', linestyle='--', label='Overall mean')
    axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / 'v2_failure_risk_factors.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {FIG_DIR / 'v2_failure_risk_factors.png'}")

# Fig 5: Failure recall comparison across models
model_names_list = ['XGBoost', 'RandomForest', 'LogisticReg', 'GBM', 'Ensemble']
val_f_recalls  = [val_results[m]['failure_recall'] for m in model_names_list[:-1]] + [ens_val_metrics['failure_recall']]
test_f_recalls = [test_results[m]['failure_recall'] for m in model_names_list[:-1]] + [ens_test_metrics['failure_recall']]

x = np.arange(len(model_names_list))
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - 0.2, val_f_recalls,  0.35, label='Validation', color='#3498db', alpha=0.8)
ax.bar(x + 0.2, test_f_recalls, 0.35, label='Test',       color='#e74c3c', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(model_names_list)
ax.set_title('Failure Detection Recall by Model\n(Primary Scout Metric)', fontsize=11)
ax.set_ylabel('Failure Recall')
ax.set_ylim(0, 1.05)
ax.legend()
ax.axhline(0.6, color='orange', linestyle='--', label='60% target', alpha=0.7)
for i, (v, t) in enumerate(zip(val_f_recalls, test_f_recalls)):
    ax.text(i - 0.2, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)
    ax.text(i + 0.2, t + 0.02, f'{t:.2f}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / 'v2_failure_recall_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {FIG_DIR / 'v2_failure_recall_comparison.png'}")

# ─────────────────────────────────────────────
# 17. Save results
# ─────────────────────────────────────────────
print("\n  Saving results...")

# Save predictions parquet
all_df_save = all_df[['player', 'team_old', 'team_new', 'season_old', 'season_new',
                        'pos_group', 'age_at_transfer', 'label',
                        'pred_label', 'prob_failure', 'prob_partial', 'prob_success',
                        'elo_gap', 'league_pos_gap', 'style_match_pct',
                        'pos_competition', 'high_competition',
                        'is_step_up', 'is_step_down', 'epl_experience',
                        'prev_adapt_rate', 'prev_transfer_cnt']].copy()
all_df_save.to_parquet(SCOUT_DIR / 'transfer_predictions_v2.parquet', index=False)
print(f"  Saved: {SCOUT_DIR / 'transfer_predictions_v2.parquet'}")

# Save JSON summary
def to_py(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    return x

results_json = {
    'model_version': 'v2',
    'total_transfers': int(len(transfer_df)),
    'label_distribution': {k: int(v) for k, v in label_counts.items()},
    'label_pct': {k: round(v/len(transfer_df)*100, 1) for k, v in label_counts.items()},
    'split_sizes': {
        'train': int(len(train_df)),
        'val':   int(len(val_df)),
        'test':  int(len(test_df)),
    },
    'n_features': len(FEATURES),
    'features': FEATURES,
    'val_metrics': {
        m: {k: to_py(v) for k, v in val_results[m].items() if k not in ('preds', 'proba', 'confusion_matrix')}
        for m in val_results
    },
    'test_metrics': {
        m: {k: to_py(v) for k, v in test_results[m].items() if k not in ('preds', 'proba', 'confusion_matrix')}
        for m in test_results
    },
    'ensemble_val':  {k: to_py(v) for k, v in ens_val_metrics.items()},
    'ensemble_test': {k: to_py(v) for k, v in ens_test_metrics.items()},
    'top_features': fi[['feature', 'importance_mean']].head(20).to_dict(orient='records'),
}

with open(SCOUT_DIR / 'transfer_v2_results.json', 'w', encoding='utf-8') as f:
    json.dump(results_json, f, ensure_ascii=False, indent=2, default=to_py)
print(f"  Saved: {SCOUT_DIR / 'transfer_v2_results.json'}")

# ─────────────────────────────────────────────
# 18. Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"Total transfers (≥300 min, 2000-2025): {len(transfer_df)}")
print(f"  Success : {label_counts.get('success', 0):4d} ({label_counts.get('success',0)/len(transfer_df)*100:.1f}%)")
print(f"  Partial : {label_counts.get('partial', 0):4d} ({label_counts.get('partial',0)/len(transfer_df)*100:.1f}%)")
print(f"  Failure : {label_counts.get('failure', 0):4d} ({label_counts.get('failure',0)/len(transfer_df)*100:.1f}%)")
print(f"\nTrain/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")
print(f"\nEnsemble Test Performance:")
print(f"  Accuracy:       {ens_test_metrics['accuracy']:.3f}")
print(f"  Macro F1:       {ens_test_metrics['macro_f1']:.3f}")
print(f"  Failure Recall: {ens_test_metrics['failure_recall']:.3f}")
print(f"  Failure AUC:    {ens_test_metrics['failure_auc']:.3f}")
print(f"\nTop 5 features for failure prediction:")
for _, row in fi.head(5).iterrows():
    print(f"  {row['feature']:35s}  importance={row['importance_mean']:.4f}")
print("\nDone.")

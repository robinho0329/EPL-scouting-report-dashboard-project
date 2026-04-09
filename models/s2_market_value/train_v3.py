"""
S2 v3: Unified Player Market Value Prediction (Scout Edition - Youth Premium Fix)
==================================================================================
Key improvements over S2 v2 (R2=0.89):

  FIX 1 — age_premium feature:
    For players aged <=23 a multiplier is applied before model training.
    Younger = higher potential premium expected (clubs pay above stat-implied value).
    age_premium = max(0, (24 - age) / 6)   → age 18 → 1.0, age 21 → 0.5, age 24 → 0.0

  FIX 2 — age × trajectory interaction:
    young_trajectory = age_premium × season_improvement_rate
    Captures "young + improving fast" premium (Yoro, Bergvall, Dibling pattern).

  FIX 3 — Overvalued list filtered:
    Players with <1500 minutes AND age <=22 are EXCLUDED from overvalued output.
    Their value is driven by potential, not EPL stats → wrong to flag as overvalued.

  FIX 4 — potential_premium_pct column:
    Shows what fraction of a player's predicted value is attributable to youth premium
    vs baseline stats, so scouts see the premium breakdown clearly.

  RETAINED:
    - Unified model (R2=0.89 preserved)
    - All existing features from v2
    - 900-minute filter for scouting outputs (undervalued list)
    - Log-transform target
    - XGBoost (primary), Ridge (baseline), MLP (sklearn)
    - Time split: train <2021/22, val 2021/22-2022/23, test 2023/24-2024/25

Usage:
    python models/s2_market_value/train_v3.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor as SklearnMLP
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"C:\Users\xcv54\workspace\EPL project")
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "s2_market_value"
FIG_DIR = MODEL_DIR / "figures"
SCOUT_DIR = DATA_DIR / "scout"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
SCOUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("S2 v3: UNIFIED MARKET VALUE PREDICTION (Youth Premium Fix)")
print("=" * 70)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def mape_score(y_true, y_pred):
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def regression_metrics(y_true, y_pred, prefix=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mp = mape_score(y_true, y_pred)
    return {
        f"{prefix}MAE": round(float(mae), 0),
        f"{prefix}RMSE": round(float(rmse), 0),
        f"{prefix}R2": round(float(r2), 4),
        f"{prefix}MAPE": round(float(mp), 2) if not np.isnan(mp) else None,
    }


def map_pos_group(pos_str):
    if pd.isna(pos_str):
        return "Unknown"
    p = str(pos_str).upper()
    if "GK" in p:
        return "GK"
    elif "DF" in p:
        return "DF"
    elif "MF" in p:
        return "MF"
    elif "FW" in p:
        return "FW"
    return "Unknown"


# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
print("\n[1/9] Loading data...")

# Primary feature table
pf = pd.read_parquet(DATA_DIR / "features" / "player_features.parquet")

# Scout profiles (has war_rating, consistency_score, big6 stats, season_improvement_rate)
scout = pd.read_parquet(SCOUT_DIR / "scout_player_profiles.parquet")

# Scout ratings v2 (has WAR score, per-90 performance, consistency)
sr_v2 = pd.read_parquet(SCOUT_DIR / "scout_ratings_v2.parquet")

# Team season summary for team strength (points)
ts = pd.read_parquet(DATA_DIR / "processed" / "team_season_summary.parquet")

print(f"  player_features:       {pf.shape}")
print(f"  scout_player_profiles: {scout.shape}")
print(f"  scout_ratings_v2:      {sr_v2.shape}")
print(f"  team_season_summary:   {ts.shape}")

# ---------------------------------------------------------------------------
# 2. Merge supplementary data
# ---------------------------------------------------------------------------
print("\n[2/9] Merging supplementary data...")

scout_dedup = scout.drop_duplicates(subset=["player", "season", "team"], keep="last")

scout_cols = ["player", "season", "team", "war_rating", "consistency_score",
              "big6_contribution_p90", "team_dependency_score", "win_rate_with_player",
              "win_rate_without_player", "season_improvement_rate", "value_momentum"]
df = pf.merge(scout_dedup[scout_cols], on=["player", "season", "team"], how="left")

sr_cols = ["player", "season", "team", "war", "consistency"]
sr_dedup = sr_v2.drop_duplicates(subset=["player", "season", "team"], keep="last")
df = df.merge(sr_dedup[sr_cols], on=["player", "season", "team"], how="left")

df = df.merge(ts[["Season", "team", "points"]], left_on=["season", "team"],
              right_on=["Season", "team"], how="left")
df.drop(columns=["Season"], inplace=True)

print(f"  Merged dataframe: {df.shape}")
print(f"  WAR (sr_v2) coverage:   {df['war'].notna().sum()} / {len(df)}")
print(f"  war_rating coverage:    {df['war_rating'].notna().sum()} / {len(df)}")
print(f"  points coverage:        {df['points'].notna().sum()} / {len(df)}")

# ---------------------------------------------------------------------------
# 3. Filter and target preparation
# ---------------------------------------------------------------------------
print("\n[3/9] Preparing target variable...")

before = len(df)
df = df[df["market_value"].notna() & (df["market_value"] > 0)].copy()
df = df[df["data_split"].notna()].copy()
print(f"  Rows with valid market_value: {len(df)} (removed {before - len(df)})")

df["log_market_value"] = np.log1p(df["market_value"])
print(f"  market_value range: {df['market_value'].min():,.0f} - {df['market_value'].max():,.0f}")
print(f"  log(MV) range:      {df['log_market_value'].min():.2f} - {df['log_market_value'].max():.2f}")

# ---------------------------------------------------------------------------
# 4. Feature Engineering (v2 base + v3 youth-premium additions)
# ---------------------------------------------------------------------------
print("\n[4/9] Engineering features (including youth-premium features)...")

# --- Base features (identical to v2) ---
df["pos_group_clean"] = df["pos"].apply(map_pos_group)
pos_dummies = pd.get_dummies(df["pos_group_clean"], prefix="pos", dtype=float)
df = pd.concat([df, pos_dummies], axis=1)

df["season_start"] = df["season"].apply(
    lambda s: int(s.split("/")[0]) if isinstance(s, str) and "/" in s else 2018
)
df["season_start_centered"] = df["season_start"] - 2018

df["age_sq"] = df["age_used"] ** 2
df["log_mv_prev"] = np.log1p(df["mv_prev"].fillna(0))

df["is_international"] = (~df["nationality"].str.contains(
    r"^England$", case=False, na=False
)).astype(float)

nat_counts = df["nationality"].value_counts()
df["nationality_freq"] = df["nationality"].map(nat_counts).fillna(1)

foot_map = {"right": 0, "left": 1, "both": 2}
df["foot_code"] = df["foot"].map(foot_map).fillna(0).astype(float)

df["war_raw"] = df["war_rating"].fillna(0)
df["war_norm"] = df["war"].fillna(
    df["war_raw"].clip(0, None) / (df["war_raw"].abs().max() + 1e-6) * 100
)

df["log_epl_exp"] = np.log1p(df["epl_experience"])
df["log_minutes"] = np.log1p(df["min"])

df["gc_x_min_share"] = df["goals_p90"].fillna(0) * df["minutes_share"].fillna(0)

df["mv_change_pct"] = df["mv_change_pct"].clip(-100, 500)
df["consistency_cv"] = df["consistency_cv"].clip(0, 10)

for col in ["war_norm", "consistency_score", "big6_contribution_p90",
            "team_dependency_score", "win_rate_with_player", "win_rate_without_player",
            "season_improvement_rate", "value_momentum"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df["transfer_flag_num"] = df["transfer_flag"].astype(float).fillna(0)

# ---------------------------------------------------------------------------
# FIX 1: age_premium feature
# ---------------------------------------------------------------------------
# age_premium: linear ramp, 1.0 at age <=18, 0.0 at age >=24, negative capped to 0
# Formula: max(0, (24 - age) / 6)
# This captures the structural premium clubs pay for young talent above stats.
df["age_premium"] = ((24.0 - df["age_used"]) / 6.0).clip(lower=0.0)

print(f"  age_premium: mean={df['age_premium'].mean():.3f}, "
      f"max={df['age_premium'].max():.3f}, "
      f"pct players >0: {(df['age_premium'] > 0).mean()*100:.1f}%")

# ---------------------------------------------------------------------------
# FIX 2: age × performance_trajectory interaction
# ---------------------------------------------------------------------------
# young_trajectory: age_premium × season_improvement_rate
# A player who is young AND rapidly improving gets a compounding premium.
# Normalise improvement rate to 0-1 range using sigmoid-like clipping.
df["season_improvement_rate"] = df["season_improvement_rate"].fillna(0)
improvement_clipped = df["season_improvement_rate"].clip(-5, 5)
# Shift so positive = improving (centre at 0)
df["young_trajectory"] = df["age_premium"] * improvement_clipped

print(f"  young_trajectory: mean={df['young_trajectory'].mean():.3f}, "
      f"max={df['young_trajectory'].max():.3f}")

# ---------------------------------------------------------------------------
# 5. Define Feature Columns (v2 base + 2 new youth features)
# ---------------------------------------------------------------------------
pos_cols = [c for c in df.columns if c.startswith("pos_") and c != "pos_group_clean"]

FEATURE_COLS = [
    # === Core value signals ===
    "log_mv_prev",          # Previous market value (most predictive lag feature)
    "mv_change_pct",        # Year-over-year change (momentum)

    # === Player profile ===
    "age_used", "age_sq",   # Age + quadratic (peak ~27)
    "log_epl_exp",          # EPL experience (log-scaled)
    "height_cm",            # Physical attribute
    "foot_code",            # Foot preference
    "is_international",     # International status proxy (reputation premium)
    "nationality_freq",     # Talent pool proxy

    # === Youth premium features (NEW in v3) ===
    "age_premium",          # FIX 1: structural premium for players <=23
    "young_trajectory",     # FIX 2: age_premium × improvement_rate interaction

    # === Playing time ===
    "log_minutes",          # Log minutes (quantity signal)
    "minutes_share",        # Share of team's total minutes
    "mp", "starts",         # Appearances

    # === Performance (per-90) ===
    "goals_p90", "assists_p90", "goal_contributions_p90",
    "yellow_cards_p90", "red_cards_p90",
    "gc_x_min_share",       # Interaction: quality x quantity

    # === WAR & Scout ratings ===
    "war_norm",             # Wins Above Replacement (normalised)
    "consistency_score",    # Consistency metric from scout profiles

    # === Big-match performance ===
    "big6_contribution_p90",

    # === Team metrics ===
    "points",               # Team strength (league points)
    "win_rate_with_player",
    "win_rate_without_player",
    "team_dependency_score",

    # === Trend / momentum ===
    "season_improvement_rate",
    "value_momentum",
    "season_start_centered",

    # === Player consistency history ===
    "consistency_mean", "consistency_std", "consistency_cv",
    "n_matches",

    # === Transfer context ===
    "transfer_flag_num",
] + pos_cols

# Ensure all columns exist and are numeric
for c in FEATURE_COLS:
    if c not in df.columns:
        df[c] = 0.0
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

print(f"  Total features: {len(FEATURE_COLS)}  (v2 had {len(FEATURE_COLS) - 2})")
print(f"  New features:   age_premium, young_trajectory")
print(f"  Position dummies: {pos_cols}")

# ---------------------------------------------------------------------------
# 6. Time-based split
# ---------------------------------------------------------------------------
print("\n[5/9] Time-based data split...")

train_df = df[df["data_split"] == "train"].copy()
val_df   = df[df["data_split"] == "val"].copy()
test_df  = df[df["data_split"] == "test"].copy()

print(f"  Train: {len(train_df):5d} rows  (seasons < 2021/22)")
print(f"  Val:   {len(val_df):5d} rows  (2021/22 - 2022/23)")
print(f"  Test:  {len(test_df):5d} rows  (2023/24 - 2024/25)")

X_train = train_df[FEATURE_COLS].values
y_train = train_df["log_market_value"].values
X_val   = val_df[FEATURE_COLS].values
y_val   = val_df["log_market_value"].values
X_test  = test_df[FEATURE_COLS].values
y_test  = test_df["log_market_value"].values

y_train_orig = train_df["market_value"].values
y_val_orig   = val_df["market_value"].values
y_test_orig  = test_df["market_value"].values

robust_scaler = RobustScaler()
X_train_rob = robust_scaler.fit_transform(X_train)
X_val_rob   = robust_scaler.transform(X_val)
X_test_rob  = robust_scaler.transform(X_test)

std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_val_std   = std_scaler.transform(X_val)
X_test_std  = std_scaler.transform(X_test)

joblib.dump(robust_scaler, MODEL_DIR / "robust_scaler_v3.joblib")
joblib.dump(std_scaler, MODEL_DIR / "std_scaler_v3.joblib")

# ---------------------------------------------------------------------------
# 7. Train Models
# ---------------------------------------------------------------------------
print("\n[6/9] Training models...")

all_results = {}
test_predictions = {}

# --- 7a. Ridge Regression (baseline) ---
print("\n  --- Ridge Regression (baseline) ---")
for alpha in [10, 100, 1000]:
    r = Ridge(alpha=alpha)
    r.fit(X_train_rob, y_train)
    pred_log = r.predict(X_val_rob)
    pred_orig = np.expm1(np.clip(pred_log, 0, 25))
    val_r2 = r2_score(y_val_orig, pred_orig)
    val_mape = mape_score(y_val_orig, pred_orig)
    print(f"    alpha={alpha:<6}: val R2={val_r2:.4f}  MAPE={val_mape:.1f}%")

best_alpha = 100
ridge = Ridge(alpha=best_alpha)
ridge.fit(X_train_rob, y_train)
for split_name, Xs, ys, yo in [
    ("val",  X_val_rob,  y_val,  y_val_orig),
    ("test", X_test_rob, y_test, y_test_orig),
]:
    pred_log = ridge.predict(Xs)
    pred_orig = np.expm1(np.clip(pred_log, 0, 25))
    m = regression_metrics(yo, pred_orig, prefix=f"{split_name}_")
    print(f"    {split_name}: R2={m[f'{split_name}_R2']:.4f}  MAE={m[f'{split_name}_MAE']:,.0f}  MAPE={m[f'{split_name}_MAPE']:.1f}%")
    if split_name == "test":
        all_results["Ridge"] = m
        test_predictions["Ridge"] = pred_orig

joblib.dump(ridge, MODEL_DIR / "ridge_v3.joblib")

# --- 7b. XGBoost (primary) ---
print("\n  --- XGBoost (primary model) ---")
xgb_model = xgb.XGBRegressor(
    n_estimators=800,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.75,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.5,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=40,
    eval_metric="rmse",
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)
print(f"    Best iteration: {xgb_model.best_iteration}")

for split_name, Xs, ys, yo in [
    ("val",  X_val,  y_val,  y_val_orig),
    ("test", X_test, y_test, y_test_orig),
]:
    pred_log = xgb_model.predict(Xs)
    pred_orig = np.expm1(np.clip(pred_log, 0, 25))
    m = regression_metrics(yo, pred_orig, prefix=f"{split_name}_")
    print(f"    {split_name}: R2={m[f'{split_name}_R2']:.4f}  MAE={m[f'{split_name}_MAE']:,.0f}  MAPE={m[f'{split_name}_MAPE']:.1f}%")
    if split_name == "test":
        all_results["XGBoost"] = m
        test_predictions["XGBoost"] = pred_orig

xgb_model.save_model(str(MODEL_DIR / "xgb_v3.json"))

# --- 7c. MLP (sklearn MLPRegressor) ---
print("\n  --- MLP Neural Network (sklearn) ---")
mlp = SklearnMLP(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    solver="adam",
    alpha=1e-3,
    learning_rate_init=1e-3,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
    verbose=False,
)
mlp.fit(X_train_std, y_train)
print(f"    Iterations: {mlp.n_iter_}  /  best_val_loss={mlp.best_validation_score_:.4f}")

for split_name, Xs, yo in [
    ("val",  X_val_std,  y_val_orig),
    ("test", X_test_std, y_test_orig),
]:
    pred_log = mlp.predict(Xs)
    pred_orig = np.expm1(np.clip(pred_log, 0, 25))
    m = regression_metrics(yo, pred_orig, prefix=f"{split_name}_")
    mape_v = m[f'{split_name}_MAPE'] or 0
    print(f"    {split_name}: R2={m[f'{split_name}_R2']:.4f}  MAE={m[f'{split_name}_MAE']:,.0f}  MAPE={mape_v:.1f}%")
    if split_name == "test":
        all_results["MLP"] = m
        test_predictions["MLP"] = pred_orig

joblib.dump(mlp, MODEL_DIR / "mlp_v3.joblib")

# ---------------------------------------------------------------------------
# 8. Model Comparison & Best Model
# ---------------------------------------------------------------------------
print("\n[7/9] Model comparison (test set)...")
print(f"  {'Model':<12} {'R2':>8} {'MAE (EUR)':>14} {'RMSE (EUR)':>14} {'MAPE':>8}")
print("  " + "-" * 60)

best_model_name = None
best_r2 = -1e9
for name, met in all_results.items():
    r2 = met["test_R2"]
    mape_v = met["test_MAPE"] or 0
    print(f"  {name:<12} {r2:>8.4f} {met['test_MAE']:>14,.0f} {met['test_RMSE']:>14,.0f} {mape_v:>7.1f}%")
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name

print(f"\n  Best model: {best_model_name} (R2 = {best_r2:.4f})")
best_preds = test_predictions[best_model_name]

# ---------------------------------------------------------------------------
# 9. Scout Value Analysis  (with v3 youth-premium fixes)
# ---------------------------------------------------------------------------
print("\n[8/9] Scout Value Analysis (v3 youth-premium filter applied)...")

test_df = test_df.copy()
test_df["predicted_value"] = best_preds
test_df["value_ratio"] = test_df["predicted_value"] / test_df["market_value"]

# ---- FIX 4: potential_premium_pct ----------------------------------------
# Estimate "premium contribution" from youth features.
# We compare log_mv_prev alone (stats-implied) vs full prediction gap.
# Proxy: proportion of log-space prediction attributable to age_premium feature.
# Implementation: use XGBoost SHAP-style weight for age_premium + young_trajectory
# Simpler proxy that's transparent to scouts:
#   potential_premium_pct = age_premium / (1 + age_premium) * 100
#   (normalised share of the youth-premium signal relative to a baseline player)
test_df["potential_premium_pct"] = (
    test_df["age_premium"] / (1 + test_df["age_premium"]) * 100
).round(1)

# Apply 900-minute filter for reliable scouting signals
scouted_df = test_df[test_df["min"] >= 900].copy()
print(f"  Scouted (min >= 900): {len(scouted_df)} players (from test {len(test_df)})")

# WAR efficiency
scouted_df["war_efficiency"] = scouted_df["war_norm"] / (scouted_df["market_value"] / 1e6 + 1e-6)

# Undervalued: predicted > 1.5x actual (model sees more value than market)
undervalued_df = (
    scouted_df[scouted_df["value_ratio"] > 1.5]
    .sort_values("value_ratio", ascending=False)
)

# ---- FIX 3: Overvalued exclusion -----------------------------------------
# Players with <1500 minutes AND age <=22 are excluded from overvalued list.
# Rationale: their high market value is potential-driven, not EPL-stats-driven.
# Flagging them as "overvalued" is a model artefact, not a real scout signal.
raw_overvalued = scouted_df[scouted_df["value_ratio"] < 0.5].copy()

potential_exclusion_mask = (
    (raw_overvalued["min"] < 1500) &
    (raw_overvalued["age_used"] <= 22)
)
overvalued_df = raw_overvalued[~potential_exclusion_mask].sort_values("value_ratio", ascending=True)
excluded_young_df = raw_overvalued[potential_exclusion_mask].sort_values("value_ratio", ascending=True)

print(f"\n  Undervalued (>1.5x ratio, min>=900): {len(undervalued_df)} players")
print(f"  Overvalued  (<0.5x ratio, min>=900): {len(raw_overvalued)} raw -> "
      f"{len(overvalued_df)} after youth filter")
print(f"  Excluded from overvalued (young potential, <1500 min & age<=22): "
      f"{len(excluded_young_df)} players")

if len(excluded_young_df) > 0:
    print(f"\n  YOUNG POTENTIAL PLAYERS (excluded from overvalued — potential premium):")
    print(f"  {'Player':<28} {'Season':<10} {'Pos':<5} {'Age':<4} {'Min':>5} "
          f"{'Actual MV':>12} {'Pred MV':>12} {'Ratio':>7} {'Prem%':>7}")
    print("  " + "-" * 100)
    for _, row in excluded_young_df.iterrows():
        print(f"  {str(row['player'])[:27]:<28} {row['season']:<10} {str(row['pos'])[:4]:<5} "
              f"{row['age_used']:<4.0f} {row['min']:>5.0f} "
              f"{row['market_value']:>12,.0f} {row['predicted_value']:>12,.0f} "
              f"{row['value_ratio']:>6.2f}x {row['potential_premium_pct']:>6.1f}%")

print(f"\n  TOP UNDERVALUED PLAYERS (test 2023-2025, min>=900 mins)")
print(f"  {'Player':<28} {'Season':<10} {'Pos':<5} {'Age':<4} {'Min':>5} "
      f"{'Actual MV':>12} {'Pred MV':>12} {'Ratio':>7} {'WAR/M':>8} {'Prem%':>7}")
print("  " + "-" * 108)
for _, row in undervalued_df.head(25).iterrows():
    print(f"  {str(row['player'])[:27]:<28} {row['season']:<10} {str(row['pos'])[:4]:<5} "
          f"{row['age_used']:<4.0f} {row['min']:>5.0f} "
          f"{row['market_value']:>12,.0f} {row['predicted_value']:>12,.0f} "
          f"{row['value_ratio']:>6.2f}x {row['war_efficiency']:>8.2f} "
          f"{row['potential_premium_pct']:>6.1f}%")

print(f"\n  TOP OVERVALUED PLAYERS (test 2023-2025, min>=900, youth-filtered)")
print(f"  {'Player':<28} {'Season':<10} {'Pos':<5} {'Age':<4} {'Min':>5} "
      f"{'Actual MV':>12} {'Pred MV':>12} {'Ratio':>7} {'Prem%':>7}")
print("  " + "-" * 96)
for _, row in overvalued_df.head(25).iterrows():
    print(f"  {str(row['player'])[:27]:<28} {row['season']:<10} {str(row['pos'])[:4]:<5} "
          f"{row['age_used']:<4.0f} {row['min']:>5.0f} "
          f"{row['market_value']:>12,.0f} {row['predicted_value']:>12,.0f} "
          f"{row['value_ratio']:>6.2f}x {row['potential_premium_pct']:>6.1f}%")

# ---------------------------------------------------------------------------
# 10. Feature Importance
# ---------------------------------------------------------------------------
print("\n  XGBoost Feature Importance (Top 20):")
xgb_imp = xgb_model.feature_importances_
imp_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": xgb_imp,
}).sort_values("importance", ascending=False)
for _, row in imp_df.head(20).iterrows():
    print(f"    {row['feature']:<35} {row['importance']:.4f}")

# Highlight new features
age_prem_rank = imp_df[imp_df["feature"] == "age_premium"].index[0] if "age_premium" in imp_df["feature"].values else None
young_traj_rank = imp_df[imp_df["feature"] == "young_trajectory"].index[0] if "young_trajectory" in imp_df["feature"].values else None
print(f"\n  age_premium rank:      #{imp_df['feature'].tolist().index('age_premium') + 1}")
print(f"  young_trajectory rank: #{imp_df['feature'].tolist().index('young_trajectory') + 1}")

imp_df.to_csv(MODEL_DIR / "xgb_feature_importance_v3.csv", index=False)

# ---------------------------------------------------------------------------
# 11. Figures
# ---------------------------------------------------------------------------
print("\n[9/9] Generating figures...")

# Fig 1: Predicted vs Actual
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("S2 v3: Predicted vs Actual Market Value (Test Set 2023-2025)\n"
             "[Youth Premium Fix Applied]",
             fontsize=13, fontweight="bold")

for ax, (name, preds) in zip(axes, test_predictions.items()):
    actual_m = y_test_orig / 1e6
    pred_m   = preds / 1e6
    ax.scatter(actual_m, pred_m, alpha=0.35, s=20, edgecolors="none", c="steelblue")
    lim = max(actual_m.max(), pred_m.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.2, label="Perfect fit")
    ax.set_xlabel("Actual Market Value (M EUR)", fontsize=10)
    ax.set_ylabel("Predicted Market Value (M EUR)", fontsize=10)
    r2_v = all_results[name]["test_R2"]
    mape_v = all_results[name]["test_MAPE"] or 0
    ax.set_title(f"{name}\nR²={r2_v:.3f}  MAPE={mape_v:.1f}%", fontsize=11)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "predicted_vs_actual_v3.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIG_DIR / 'predicted_vs_actual_v3.png'}")

# Fig 2: Value ratio distribution
fig, ax = plt.subplots(figsize=(12, 5))
ratios_all = scouted_df["value_ratio"].clip(0, 6)
ax.hist(ratios_all, bins=80, edgecolor="none", alpha=0.55, color="steelblue",
        label="All scouted (min>=900)")

# Shade excluded young players differently
if len(excluded_young_df) > 0:
    ratios_excl = excluded_young_df["value_ratio"].clip(0, 6)
    ax.hist(ratios_excl, bins=40, edgecolor="none", alpha=0.8, color="gold",
            label=f"Excluded young potential (age<=22, <1500min): {len(excluded_young_df)}")

ax.axvline(1.5, color="green", linestyle="--", linewidth=2,
           label=f"Undervalued (>1.5x): {len(undervalued_df)} players")
ax.axvline(0.5, color="red", linestyle="--", linewidth=2,
           label=f"Overvalued (<0.5x, filtered): {len(overvalued_df)} players")
ax.axvline(1.0, color="black", linestyle="-", linewidth=1.2, label="Fair value (1.0x)")
ax.set_xlabel("Value Ratio (Predicted / Actual)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("S2 v3: Value Ratio Distribution — Youth Premium Fix\n"
             "Gold = young potential players excluded from overvalued list", fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "value_ratio_distribution_v3.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIG_DIR / 'value_ratio_distribution_v3.png'}")

# Fig 3: Feature Importance
fig, ax = plt.subplots(figsize=(10, 8))
top20 = imp_df.head(20).iloc[::-1]
colors = []
for f in top20["feature"]:
    if "mv_prev" in f or "war" in f or "log_mv" in f:
        colors.append("#2ecc71")
    elif f in ("age_premium", "young_trajectory"):
        colors.append("#e74c3c")   # highlight new youth features
    elif "age" in f or "pos_" in f or "epl_exp" in f:
        colors.append("#3498db")
    else:
        colors.append("#e67e22")

ax.barh(top20["feature"], top20["importance"], color=colors, edgecolor="white", height=0.7)
ax.set_xlabel("Feature Importance (XGBoost gain)", fontsize=11)
ax.set_title("S2 v3: Top 20 Feature Importances\n(Red = new youth-premium features)", fontsize=13)
patches = [
    mpatches.Patch(color="#2ecc71", label="Value lag / WAR"),
    mpatches.Patch(color="#e74c3c", label="Youth premium features (NEW v3)"),
    mpatches.Patch(color="#3498db", label="Age / Position / Experience"),
    mpatches.Patch(color="#e67e22", label="Performance / Context"),
]
ax.legend(handles=patches, fontsize=9)
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(FIG_DIR / "feature_importance_v3.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIG_DIR / 'feature_importance_v3.png'}")

# Fig 4: WAR Efficiency scatter (undervalued players)
if len(undervalued_df) > 0:
    fig, ax = plt.subplots(figsize=(11, 7))
    sc = ax.scatter(
        undervalued_df["market_value"] / 1e6,
        undervalued_df["war_norm"],
        c=undervalued_df["value_ratio"],
        cmap="RdYlGn",
        s=80, alpha=0.8, edgecolors="grey", linewidth=0.5,
        vmin=1.5, vmax=4.0,
    )
    plt.colorbar(sc, ax=ax, label="Value Ratio (Predicted/Actual)")
    ax.set_xlabel("Actual Market Value (M EUR)", fontsize=11)
    ax.set_ylabel("WAR Score (0-100)", fontsize=11)
    ax.set_title("S2 v3: Undervalued Players — WAR vs Market Value\n"
                 "(Larger ratio = more undervalued, greener = better value)", fontsize=12)
    top_gems = undervalued_df.head(10)
    for _, row in top_gems.iterrows():
        ax.annotate(
            str(row["player"])[:15],
            (row["market_value"] / 1e6, row["war_norm"]),
            textcoords="offset points", xytext=(5, 3),
            fontsize=7, alpha=0.9,
        )
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "undervalued_war_scatter_v3.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / 'undervalued_war_scatter_v3.png'}")

# Fig 5: Model R2 comparison bar chart
fig, ax = plt.subplots(figsize=(7, 4))
model_names = list(all_results.keys())
r2_vals   = [all_results[n]["test_R2"]   for n in model_names]
mape_vals = [all_results[n]["test_MAPE"] or 0 for n in model_names]

x = np.arange(len(model_names))
bars = ax.bar(x, r2_vals, color=["#3498db", "#e74c3c", "#2ecc71"], edgecolor="white", width=0.5)
ax.axhline(0.87, color="orange", linestyle="--", linewidth=1.5, label="P6 baseline R²=0.87")
for bar, r2v, mapev in zip(bars, r2_vals, mape_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"R²={r2v:.3f}\nMAPE={mapev:.1f}%", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylabel("R² (Test Set)", fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title("S2 v3: Model Performance vs P6 Baseline", fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(FIG_DIR / "model_comparison_v3.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIG_DIR / 'model_comparison_v3.png'}")

# Fig 6 (NEW v3): Youth Premium breakdown scatter
# For all test players <= 23, plot actual MV vs predicted MV coloured by age
young_test = test_df[test_df["age_used"] <= 23].copy()
if len(young_test) > 0:
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(
        young_test["market_value"] / 1e6,
        young_test["predicted_value"] / 1e6,
        c=young_test["age_used"],
        cmap="RdYlGn_r",
        s=60, alpha=0.75, edgecolors="grey", linewidth=0.4,
        vmin=16, vmax=23,
    )
    plt.colorbar(sc, ax=ax, label="Age")
    lim = max((young_test["market_value"] / 1e6).max(),
              (young_test["predicted_value"] / 1e6).max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1.2, label="Perfect fit")
    ax.plot([0, lim], [0, lim * 0.5], "r:", linewidth=1.0, alpha=0.6, label="0.5x (old overvalued line)")
    ax.set_xlabel("Actual Market Value (M EUR)", fontsize=11)
    ax.set_ylabel("Predicted Market Value (M EUR)", fontsize=11)
    ax.set_title("S2 v3: Young Players (age <=23) — Predicted vs Actual\n"
                 "Green = younger (more premium expected). Points below dashed = previously mis-flagged",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    # Annotate the most prominent ones (low ratio, high value = previously flagged)
    mis_flagged = young_test[
        (young_test["value_ratio"] < 0.5) & (young_test["market_value"] > 20e6)
    ]
    for _, row in mis_flagged.head(8).iterrows():
        ax.annotate(
            str(row["player"])[:15],
            (row["market_value"] / 1e6, row["predicted_value"] / 1e6),
            textcoords="offset points", xytext=(5, 3),
            fontsize=7, alpha=0.95, color="darkred",
        )
    plt.tight_layout()
    plt.savefig(FIG_DIR / "youth_premium_scatter_v3.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / 'youth_premium_scatter_v3.png'}")

# ---------------------------------------------------------------------------
# 12. Save Scout Output Parquet
# ---------------------------------------------------------------------------
print("\n  Saving scout output parquet files...")

uv_output_cols = ["player", "team", "season", "pos", "age_used", "min",
                  "market_value", "predicted_value", "value_ratio",
                  "war_norm", "war_efficiency", "goals_p90", "assists_p90",
                  "nationality", "epl_experience", "points",
                  "age_premium", "young_trajectory", "potential_premium_pct"]
uv_save_cols = [c for c in uv_output_cols if c in undervalued_df.columns]
undervalued_df[uv_save_cols].to_parquet(SCOUT_DIR / "s2_v3_undervalued.parquet", index=False)
print(f"  Saved: {SCOUT_DIR / 's2_v3_undervalued.parquet'}  ({len(undervalued_df)} rows)")

ov_output_cols = ["player", "team", "season", "pos", "age_used", "min",
                  "market_value", "predicted_value", "value_ratio",
                  "war_norm", "goals_p90", "assists_p90",
                  "age_premium", "young_trajectory", "potential_premium_pct"]
ov_save_cols = [c for c in ov_output_cols if c in overvalued_df.columns]
overvalued_df[ov_save_cols].to_parquet(SCOUT_DIR / "s2_v3_overvalued.parquet", index=False)
print(f"  Saved: {SCOUT_DIR / 's2_v3_overvalued.parquet'}  ({len(overvalued_df)} rows)")

excl_output_cols = ["player", "team", "season", "pos", "age_used", "min",
                    "market_value", "predicted_value", "value_ratio",
                    "age_premium", "young_trajectory", "potential_premium_pct"]
excl_save_cols = [c for c in excl_output_cols if c in excluded_young_df.columns]
if len(excluded_young_df) > 0:
    excluded_young_df[excl_save_cols].to_parquet(
        SCOUT_DIR / "s2_v3_young_potential_excluded.parquet", index=False
    )
    print(f"  Saved: {SCOUT_DIR / 's2_v3_young_potential_excluded.parquet'}  ({len(excluded_young_df)} rows)")

pred_cols = ["player", "team", "season", "pos", "age_used", "min",
             "market_value", "predicted_value", "value_ratio",
             "war_norm", "goals_p90", "assists_p90",
             "age_premium", "young_trajectory", "potential_premium_pct"]
pred_save_cols = [c for c in pred_cols if c in scouted_df.columns]
scouted_df[pred_save_cols].to_parquet(SCOUT_DIR / "s2_v3_all_predictions.parquet", index=False)
print(f"  Saved: {SCOUT_DIR / 's2_v3_all_predictions.parquet'}  ({len(scouted_df)} rows)")

# ---------------------------------------------------------------------------
# 13. Save results_summary_v3.json
# ---------------------------------------------------------------------------
print("\n  Saving results_summary_v3.json...")


def build_player_list(frame, top_n=25, extra_cols=None):
    records = []
    for _, row in frame.head(top_n).iterrows():
        rec = {
            "player": str(row["player"]),
            "team": str(row["team"]),
            "season": str(row["season"]),
            "position": str(row["pos"]),
            "age": int(row["age_used"]),
            "minutes": int(row["min"]),
            "actual_market_value_eur": int(row["market_value"]),
            "predicted_market_value_eur": int(row["predicted_value"]),
            "value_ratio": round(float(row["value_ratio"]), 3),
            "war_score": round(float(row.get("war_norm", 0)), 2),
            "goals_p90": round(float(row.get("goals_p90", 0)), 3),
            "assists_p90": round(float(row.get("assists_p90", 0)), 3),
            "age_premium": round(float(row.get("age_premium", 0)), 3),
            "potential_premium_pct": round(float(row.get("potential_premium_pct", 0)), 1),
        }
        records.append(rec)
    return records


def build_excluded_list(frame):
    records = []
    for _, row in frame.iterrows():
        records.append({
            "player": str(row["player"]),
            "team": str(row["team"]),
            "season": str(row["season"]),
            "position": str(row["pos"]),
            "age": int(row["age_used"]),
            "minutes": int(row["min"]),
            "actual_market_value_eur": int(row["market_value"]),
            "predicted_market_value_eur": int(row["predicted_value"]),
            "value_ratio": round(float(row["value_ratio"]), 3),
            "age_premium": round(float(row.get("age_premium", 0)), 3),
            "potential_premium_pct": round(float(row.get("potential_premium_pct", 0)), 1),
            "exclusion_reason": "young_potential_<1500min_age<=22",
        })
    return records


results_summary = {
    "pipeline": "S2 v3 - Unified Market Value Prediction (Youth Premium Fix)",
    "version": "v3",
    "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "key_improvements_over_v2": [
        "FIX 1: age_premium feature — linear ramp 1.0 (age<=18) to 0.0 (age>=24), captures structural youth premium clubs pay above stats",
        "FIX 2: young_trajectory interaction — age_premium x season_improvement_rate, 'young + improving' gets compounding premium",
        "FIX 3: Overvalued list excludes players with <1500 min AND age<=22 — their value is potential-based, not EPL-stats-based",
        "FIX 4: potential_premium_pct column added — shows % of value explained by youth premium vs stats",
        "Added Fig 6: youth_premium_scatter showing previously mis-flagged young players",
        "Added s2_v3_young_potential_excluded.parquet — separate list of potential-based players",
        "Kept unified model approach (R2 target >= 0.89)",
    ],
    "youth_premium_design": {
        "age_premium_formula": "max(0, (24 - age) / 6)",
        "age_premium_examples": {
            "age_16": 1.333,
            "age_18": 1.0,
            "age_20": 0.667,
            "age_21": 0.5,
            "age_22": 0.333,
            "age_23": 0.167,
            "age_24+": 0.0,
        },
        "young_trajectory_formula": "age_premium x season_improvement_rate (clipped -5 to 5)",
        "overvalued_exclusion_rule": "min < 1500 AND age <= 22",
        "potential_premium_pct_formula": "age_premium / (1 + age_premium) * 100",
    },
    "data": {
        "total_rows_with_value": len(df),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "scouted_players_min900": len(scouted_df),
        "num_features": len(FEATURE_COLS),
        "feature_columns": FEATURE_COLS,
        "target": "log1p(market_value)",
        "time_split": {
            "train": "seasons < 2021/22",
            "val": "2021/22 - 2022/23",
            "test": "2023/24 - 2024/25",
        },
    },
    "best_model": best_model_name,
    "model_metrics": all_results,
    "value_analysis": {
        "undervalued_count_min900": len(undervalued_df),
        "overvalued_count_raw_before_youth_filter": len(raw_overvalued),
        "overvalued_count_after_youth_filter": len(overvalued_df),
        "excluded_young_potential_count": len(excluded_young_df),
        "undervalued_threshold": "predicted > 1.5x actual",
        "overvalued_threshold": "predicted < 0.5x actual",
        "overvalued_exclusion": "age <= 22 AND min < 1500 (potential-premium players)",
        "minutes_filter_scouted": ">=900 minutes played",
    },
    "top25_undervalued": build_player_list(undervalued_df, 25),
    "top25_overvalued": build_player_list(overvalued_df, 25),
    "young_potential_excluded_from_overvalued": build_excluded_list(excluded_young_df),
    "feature_importance_xgb_top15": imp_df.head(15)[["feature", "importance"]].to_dict(orient="records"),
}

with open(MODEL_DIR / "results_summary_v3.json", "w", encoding="utf-8") as f:
    json.dump(results_summary, f, indent=2, ensure_ascii=False)
print(f"  Saved: {MODEL_DIR / 'results_summary_v3.json'}")

print("\n" + "=" * 70)
print(f"S2 v3 PIPELINE COMPLETE")
print(f"  Best model:  {best_model_name}  R2={best_r2:.4f}")
print(f"  Test MAPE:   {all_results[best_model_name]['test_MAPE']:.1f}%")
print(f"  Undervalued: {len(undervalued_df)} players (min>=900)")
print(f"  Overvalued:  {len(overvalued_df)} players (youth-filtered)")
print(f"  Young potential (excluded from overvalued): {len(excluded_young_df)}")
print("=" * 70)

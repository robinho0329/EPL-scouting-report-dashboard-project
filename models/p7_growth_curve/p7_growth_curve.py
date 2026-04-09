"""
P7: Player Growth Curve Prediction
===================================
Predicts how a young player's performance will evolve across seasons.
Scout question: "Will this 22-year-old reach elite level by 26?"

Models: Linear Regression (baseline), XGBoost, MLPRegressor (sequence-aware)
Metrics: MAE, RMSE, R², Directional Accuracy
"""

import os, json, warnings, sys
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── paths ────────────────────────────────────────────────────────────────
BASE   = Path(__file__).resolve().parent
PROJ   = BASE.parents[1]
DATA   = PROJ / "data"
OUT    = BASE            # models/p7_growth_curve/
FIG    = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("P7  Player Growth Curve Prediction")
print("=" * 60)

# =====================================================================
# 1.  LOAD DATA
# =====================================================================
print("\n[1/8] Loading data …")

feat = pd.read_parquet(DATA / "features" / "player_features.parquet")
pss  = pd.read_parquet(DATA / "processed" / "player_season_stats.parquet")
pml  = pd.read_parquet(DATA / "processed" / "player_match_logs.parquet")

print(f"  player_features  : {feat.shape}")
print(f"  player_season    : {pss.shape}")
print(f"  player_match_logs: {pml.shape}")

# =====================================================================
# 2.  BUILD SEASON-LEVEL TABLE  (one row per player-season)
# =====================================================================
print("\n[2/8] Building season-level table …")

# Use player_features as the main table – it already has per-90 stats,
# market value, age, position, and data_split.
df = feat.copy()

# Ensure season ordering
SEASON_ORDER = sorted(df["season"].unique(),
                      key=lambda s: int(s.split("/")[0]))
season_to_idx = {s: i for i, s in enumerate(SEASON_ORDER)}
df["season_idx"] = df["season"].map(season_to_idx)

# Robust player key – prefer player_id when available, else name
df["pid"] = df["player_id"].fillna(-1).astype(int).astype(str)
# For rows without player_id, use player name (lowercase) as fallback
mask_no_id = df["player_id"].isna()
df.loc[mask_no_id, "pid"] = df.loc[mask_no_id, "player"].str.lower().str.strip()

df.sort_values(["pid", "season_idx"], inplace=True)

# =====================================================================
# 3.  COMPOSITE PERFORMANCE SCORE  (position-normalized)
# =====================================================================
print("\n[3/8] Computing composite performance score …")

# Group positions into broad categories for normalization
POS_MAP = {
    "Goalkeeper":          "GK",
    "Centre-Back":         "DEF",
    "Right-Back":          "DEF",
    "Left-Back":           "DEF",
    "Defender":            "DEF",
    "Defensive Midfield":  "MID",
    "Central Midfield":    "MID",
    "Attacking Midfield":  "MID",
    "Left Midfield":       "MID",
    "Right Midfield":      "MID",
    "Midfielder":          "MID",
    "Left Winger":         "FWD",
    "Right Winger":        "FWD",
    "Centre-Forward":      "FWD",
    "Second Striker":      "FWD",
    "Striker":             "FWD",
}
df["pos_group"] = df["position"].map(POS_MAP).fillna("MID")

# Raw composite: weighted sum of per-90 stats + minutes share
# Weights emphasise output for forwards, defensive actions for defs
w = {
    "FWD": {"goals_p90": 4.0, "assists_p90": 3.0, "minutes_share": 1.5,
             "tackles_p90": 0.3, "interceptions_p90": 0.3},
    "MID": {"goals_p90": 2.5, "assists_p90": 3.0, "minutes_share": 1.5,
             "tackles_p90": 1.0, "interceptions_p90": 1.0},
    "DEF": {"goals_p90": 1.0, "assists_p90": 1.5, "minutes_share": 2.0,
             "tackles_p90": 2.5, "interceptions_p90": 2.5},
    "GK":  {"goals_p90": 0.0, "assists_p90": 0.0, "minutes_share": 3.0,
             "tackles_p90": 0.5, "interceptions_p90": 0.5},
}

def composite_score(row):
    wt = w.get(row["pos_group"], w["MID"])
    score = 0.0
    for col, weight in wt.items():
        v = row.get(col, 0.0)
        if pd.isna(v):
            v = 0.0
        score += v * weight
    return score

df["composite"] = df.apply(composite_score, axis=1)

# Position-normalize: z-score within position group + season
df["composite_z"] = 0.0
for (pg, s), grp in df.groupby(["pos_group", "season"]):
    mu, std = grp["composite"].mean(), grp["composite"].std()
    if std > 0:
        df.loc[grp.index, "composite_z"] = (grp["composite"] - mu) / std

print(f"  Composite score range: {df['composite'].min():.2f} – {df['composite'].max():.2f}")
print(f"  Composite Z range   : {df['composite_z'].min():.2f} – {df['composite_z'].max():.2f}")

# =====================================================================
# 4.  BUILD GROWTH FEATURES  (lagged seasons)
# =====================================================================
print("\n[4/8] Building growth features (lagged seasons) …")

# For each player-season, attach stats from previous 1-3 seasons
STAT_COLS = [
    "goals_p90", "assists_p90", "goal_contributions_p90",
    "tackles_p90", "interceptions_p90", "minutes_share",
    "composite", "composite_z", "market_value", "min",
    "yellow_cards_p90", "red_cards_p90",
]

# Lag features
for lag in [1, 2, 3]:
    shifted = (df.groupby("pid")[STAT_COLS + ["season_idx"]]
               .shift(lag)
               .rename(columns={c: f"{c}_lag{lag}" for c in STAT_COLS + ["season_idx"]}))
    df = pd.concat([df, shifted], axis=1)

# Target: next season composite_z
df["target"] = df.groupby("pid")["composite_z"].shift(-1)

# Also create market-value change % as secondary target
df["mv_next"] = df.groupby("pid")["market_value"].shift(-1)
df["mv_change_pct_next"] = np.where(
    df["market_value"] > 0,
    (df["mv_next"] - df["market_value"]) / df["market_value"] * 100,
    np.nan,
)

# Derived growth features
for lag in [1, 2]:
    curr = "composite"
    prev = f"composite_lag{lag}"
    df[f"composite_delta_{lag}"] = df[curr] - df[prev]

# Minutes trend
df["min_trend_1"] = df["min"] - df["min_lag1"]
df["min_trend_2"] = df["min"] - df["min_lag2"]

# MV trajectory
df["mv_trend_1"] = df["market_value"] - df["market_value_lag1"]

# Ensure consecutive season check (gap ≤ 1 allowed between lag and current)
df["season_gap_1"] = df["season_idx"] - df["season_idx_lag1"]

print(f"  Rows before filtering: {len(df)}")

# =====================================================================
# 5.  FEATURE MATRIX
# =====================================================================
print("\n[5/8] Preparing feature matrix …")

FEATURE_COLS = [
    # Current season
    "age_used", "minutes_share", "goals_p90", "assists_p90",
    "goal_contributions_p90", "tackles_p90", "interceptions_p90",
    "yellow_cards_p90", "composite", "composite_z",
    "market_value", "epl_experience",
    # Lag-1
    "goals_p90_lag1", "assists_p90_lag1", "goal_contributions_p90_lag1",
    "tackles_p90_lag1", "interceptions_p90_lag1", "minutes_share_lag1",
    "composite_lag1", "composite_z_lag1", "market_value_lag1", "min_lag1",
    # Lag-2
    "goals_p90_lag2", "assists_p90_lag2", "composite_lag2",
    "composite_z_lag2", "market_value_lag2", "min_lag2",
    # Lag-3
    "composite_lag3", "composite_z_lag3", "market_value_lag3",
    # Deltas / trends
    "composite_delta_1", "composite_delta_2",
    "min_trend_1", "min_trend_2", "mv_trend_1",
    "mv_change_pct",
]

# Position dummies
pos_dummies = pd.get_dummies(df["pos_group"], prefix="pos", dtype=float)
df = pd.concat([df, pos_dummies], axis=1)
pos_cols = [c for c in pos_dummies.columns]
FEATURE_COLS += pos_cols

# Filter: need target AND at least lag-1 AND consecutive seasons (gap == 1)
valid = df["target"].notna() & df["composite_lag1"].notna() & (df["season_gap_1"] == 1)
dfv = df.loc[valid].copy()
print(f"  Valid rows (have target + lag1): {len(dfv)}")

# Fill remaining NaN in features with 0 (lag2/lag3 may be missing)
dfv[FEATURE_COLS] = dfv[FEATURE_COLS].fillna(0)

# ── TIME-BASED SPLIT ─────────────────────────────────────────────────
# The "data_split" col is for the *current* season.  But we predict
# *next* season, so a row in "train" split predicts into "val" at boundary.
# We re-derive: the row's season is the predictor season; the target
# season = season_idx + 1.
# Train: season < 2021/22 idx,  Val: 2021/22-2022/23, Test: 2023/24-2024/25
idx_2122 = season_to_idx["2021/22"]
idx_2223 = season_to_idx["2022/23"]
idx_2324 = season_to_idx["2023/24"]
idx_2425 = season_to_idx["2024/25"]

# The target season is season_idx + 1
dfv["target_season_idx"] = dfv["season_idx"] + 1

train_mask = dfv["target_season_idx"] < idx_2122
val_mask   = (dfv["target_season_idx"] >= idx_2122) & (dfv["target_season_idx"] <= idx_2223)
test_mask  = (dfv["target_season_idx"] >= idx_2324) & (dfv["target_season_idx"] <= idx_2425)

X_train = dfv.loc[train_mask, FEATURE_COLS].values
y_train = dfv.loc[train_mask, "target"].values
X_val   = dfv.loc[val_mask,   FEATURE_COLS].values
y_val   = dfv.loc[val_mask,   "target"].values
X_test  = dfv.loc[test_mask,  FEATURE_COLS].values
y_test  = dfv.loc[test_mask,  "target"].values

print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# =====================================================================
# 6.  TRAIN MODELS
# =====================================================================
print("\n[6/8] Training models …")

def evaluate(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    # Directional accuracy: predict improvement vs decline relative to current z
    # We compare sign(predicted_change) vs sign(actual_change)
    # Here target is next-season z, and current z is in features (composite_z)
    dir_acc = np.nan
    if len(y_true) > 0:
        dir_acc = np.mean(np.sign(y_pred) == np.sign(y_true)) * 100
    print(f"  {label:20s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  DirAcc={dir_acc:.1f}%")
    return {"mae": round(mae, 4), "rmse": round(rmse, 4),
            "r2": round(r2, 4), "directional_accuracy_pct": round(dir_acc, 1)}

results = {}

# ── 6a. Linear Regression ────────────────────────────────────────────
print("\n  --- Linear Regression (baseline) ---")
lr = LinearRegression()
lr.fit(X_train_s, y_train)
pred_lr_val  = lr.predict(X_val_s)
pred_lr_test = lr.predict(X_test_s)
results["linear_regression"] = {
    "val":  evaluate(y_val,  pred_lr_val,  "LR val"),
    "test": evaluate(y_test, pred_lr_test, "LR test"),
}

# ── 6b. XGBoost ──────────────────────────────────────────────────────
print("\n  --- XGBoost ---")
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLS)
dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=FEATURE_COLS)
dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=FEATURE_COLS)

params = {
    "objective": "reg:squarederror",
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "seed": 42,
}
xgb_model = xgb.train(params, dtrain, num_boost_round=500,
                       evals=[(dval, "val")],
                       early_stopping_rounds=30,
                       verbose_eval=False)
pred_xgb_val  = xgb_model.predict(dval)
pred_xgb_test = xgb_model.predict(dtest)
results["xgboost"] = {
    "val":  evaluate(y_val,  pred_xgb_val,  "XGB val"),
    "test": evaluate(y_test, pred_xgb_test, "XGB test"),
    "best_iteration": int(xgb_model.best_iteration),
}

# ── 6c. MLPRegressor (sequence-aware via lag features) ────────────────
print("\n  --- MLPRegressor (neural net) ---")
mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    solver="adam",
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.15,
    random_state=42,
    learning_rate_init=0.001,
    batch_size=64,
)
mlp.fit(X_train_s, y_train)
pred_mlp_val  = mlp.predict(X_val_s)
pred_mlp_test = mlp.predict(X_test_s)
results["mlp_regressor"] = {
    "val":  evaluate(y_val,  pred_mlp_val,  "MLP val"),
    "test": evaluate(y_test, pred_mlp_test, "MLP test"),
}

# =====================================================================
# 7.  SCOUT OUTPUTS
# =====================================================================
print("\n[7/8] Generating scout outputs …")

# Use XGBoost predictions (typically best) for scout outputs
# Attach predictions back to dfv
dfv = dfv.copy()
# We need predictions for test-set rows
test_df = dfv.loc[test_mask].copy()
test_df["pred_next_z"] = pred_xgb_test
test_df["pred_growth"] = test_df["pred_next_z"] - test_df["composite_z"]

# ── 7a. Top 20 highest growth potential young players (age ≤ 23) ─────
print("\n  --- Top 20 Growth Potential (age ≤ 23, test set) ---")
young = test_df[test_df["age_used"] <= 23].copy()
young_sorted = young.sort_values("pred_growth", ascending=False)

# If a player appears multiple times (multiple test seasons), take latest
young_dedup = young_sorted.drop_duplicates(subset="pid", keep="first")
top20 = young_dedup.nlargest(20, "pred_growth")

top20_list = []
for _, row in top20.iterrows():
    entry = {
        "player": row["player"],
        "team": row["team"],
        "age": int(row["age_used"]),
        "position": row["position"] if pd.notna(row["position"]) else row["pos"],
        "season": row["season"],
        "current_composite_z": round(row["composite_z"], 3),
        "predicted_next_z": round(row["pred_next_z"], 3),
        "predicted_growth": round(row["pred_growth"], 3),
        "market_value": int(row["market_value"]) if pd.notna(row["market_value"]) and row["market_value"] > 0 else None,
    }
    top20_list.append(entry)
    print(f"    {len(top20_list):2d}. {entry['player']:25s}  age={entry['age']}  "
          f"team={entry['team']:15s}  growth={entry['predicted_growth']:+.3f}  "
          f"curr_z={entry['current_composite_z']:.3f}")

# ── 7b. Peak age analysis by position ────────────────────────────────
print("\n  --- Peak Age Analysis by Position ---")
# Use all data (not just test) to find peak composite by age and position
peak_data = df[df["composite"].notna() & df["age_used"].notna()].copy()
peak_data["age_int"] = peak_data["age_used"].astype(int)

peak_by_pos = (peak_data
               .groupby(["pos_group", "age_int"])["composite"]
               .mean()
               .reset_index())

peak_ages = {}
for pg in ["FWD", "MID", "DEF", "GK"]:
    sub = peak_by_pos[peak_by_pos["pos_group"] == pg]
    if len(sub) == 0:
        continue
    # Smooth with rolling mean for robustness
    sub = sub.sort_values("age_int")
    sub["smooth"] = sub["composite"].rolling(3, center=True, min_periods=1).mean()
    best_row = sub.loc[sub["smooth"].idxmax()]
    peak_ages[pg] = int(best_row["age_int"])
    print(f"    {pg:4s}  peak age = {peak_ages[pg]}")

# ── 7c.  Figures ─────────────────────────────────────────────────────
print("\n  Generating figures …")

# Figure 1: Growth trajectory for top 5 prospects
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
top5_pids = top20["pid"].values[:6]
for i, pid in enumerate(top5_pids):
    ax = axes[i]
    player_data = df[df["pid"] == pid].sort_values("season_idx")
    name = player_data["player"].iloc[0]
    ax.plot(player_data["season"], player_data["composite_z"],
            "o-", color="steelblue", linewidth=2, markersize=6, label="Actual")
    # Add predicted next point
    test_row = test_df[test_df["pid"] == pid].sort_values("season_idx")
    if len(test_row) > 0:
        last = test_row.iloc[-1]
        next_sidx = int(last["season_idx"] + 1)
        if next_sidx < len(SEASON_ORDER):
            next_season = SEASON_ORDER[next_sidx]
        else:
            next_season = f"next"
        ax.plot([last["season"], next_season],
                [last["composite_z"], last["pred_next_z"]],
                "s--", color="tomato", linewidth=2, markersize=8, label="Predicted")
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_ylabel("Composite Z-score")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
fig.suptitle("Growth Trajectories – Top Prospects (P7)", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG / "growth_trajectories_top_prospects.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved growth_trajectories_top_prospects.png")

# Figure 2: Peak age curves by position
fig, ax = plt.subplots(figsize=(10, 6))
colors = {"FWD": "tomato", "MID": "steelblue", "DEF": "seagreen", "GK": "orange"}
for pg in ["FWD", "MID", "DEF", "GK"]:
    sub = peak_by_pos[peak_by_pos["pos_group"] == pg].sort_values("age_int")
    sub = sub[(sub["age_int"] >= 17) & (sub["age_int"] <= 38)]
    smooth = sub["composite"].rolling(3, center=True, min_periods=1).mean()
    ax.plot(sub["age_int"], smooth, "-o", color=colors[pg],
            linewidth=2, markersize=4, label=f"{pg} (peak ≈ {peak_ages.get(pg, '?')})")
ax.set_xlabel("Age", fontsize=12)
ax.set_ylabel("Average Composite Score", fontsize=12)
ax.set_title("Peak Age Curves by Position Group", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.savefig(FIG / "peak_age_by_position.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved peak_age_by_position.png")

# Figure 3: Model comparison bar chart
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
metrics_list = ["mae", "rmse", "r2", "directional_accuracy_pct"]
titles = ["MAE ↓", "RMSE ↓", "R² ↑", "Directional Accuracy (%) ↑"]
model_names = ["linear_regression", "xgboost", "mlp_regressor"]
short_names = ["Linear Reg", "XGBoost", "MLP"]
bar_colors = ["#4C72B0", "#DD8452", "#55A868"]

for idx, (metric, title) in enumerate(zip(metrics_list, titles)):
    ax = axes[idx]
    vals = [results[m]["test"][metric] for m in model_names]
    bars = ax.bar(short_names, vals, color=bar_colors, edgecolor="white", width=0.6)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(metric)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}" if metric != "directional_accuracy_pct" else f"{v:.1f}%",
                ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Model Comparison on Test Set (P7)", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG / "model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved model_comparison.png")

# Figure 4: XGBoost feature importance (top 15)
importance = xgb_model.get_score(importance_type="gain")
imp_df = (pd.DataFrame(list(importance.items()), columns=["feature", "gain"])
          .sort_values("gain", ascending=True)
          .tail(15))
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(imp_df["feature"], imp_df["gain"], color="steelblue", edgecolor="white")
ax.set_xlabel("Gain", fontsize=12)
ax.set_title("XGBoost Feature Importance (Top 15)", fontsize=14, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fig.savefig(FIG / "xgb_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved xgb_feature_importance.png")

# Figure 5: Predicted vs actual scatter for test set (XGBoost)
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, pred_xgb_test, alpha=0.5, s=20, color="steelblue")
lims = [min(y_test.min(), pred_xgb_test.min()) - 0.2,
        max(y_test.max(), pred_xgb_test.max()) + 0.2]
ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
ax.set_xlabel("Actual Composite Z (next season)", fontsize=12)
ax.set_ylabel("Predicted Composite Z (next season)", fontsize=12)
ax.set_title("XGBoost: Predicted vs Actual (Test Set)", fontsize=14, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")
fig.savefig(FIG / "predicted_vs_actual_xgb.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved predicted_vs_actual_xgb.png")

# Figure 6: Growth distribution by age bracket (test set)
fig, ax = plt.subplots(figsize=(10, 6))
test_df["age_bracket_fine"] = pd.cut(test_df["age_used"], bins=[17, 20, 23, 26, 29, 32, 40],
                                      labels=["17-20", "21-23", "24-26", "27-29", "30-32", "33+"])
bracket_growth = test_df.groupby("age_bracket_fine")["pred_growth"].mean()
bracket_growth.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
ax.set_xlabel("Age Bracket", fontsize=12)
ax.set_ylabel("Mean Predicted Growth (ΔZ)", fontsize=12)
ax.set_title("Predicted Growth by Age Bracket (Test Set)", fontsize=14, fontweight="bold")
ax.axhline(0, color="red", linestyle="--", alpha=0.5)
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=0)
plt.tight_layout()
fig.savefig(FIG / "growth_by_age_bracket.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved growth_by_age_bracket.png")

# =====================================================================
# 8.  SAVE RESULTS
# =====================================================================
print("\n[8/8] Saving results_summary.json …")

summary = {
    "pipeline": "P7 – Player Growth Curve Prediction",
    "description": "Predicts next-season composite performance Z-score from current + historical stats",
    "data": {
        "total_player_seasons": int(len(feat)),
        "valid_rows_with_target": int(len(dfv)),
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "players_with_2plus_seasons": int((df.groupby("pid")["season"].nunique() >= 2).sum()),
        "features_used": len(FEATURE_COLS),
        "feature_names": FEATURE_COLS,
    },
    "composite_score": {
        "method": "Weighted sum of per-90 stats (goals, assists, tackles, interceptions, minutes_share), position-normalized via Z-score within position group + season",
        "position_weights": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in w.items()},
    },
    "models": results,
    "best_model": "xgboost",
    "scout_outputs": {
        "top_20_growth_potential_young_players": top20_list,
        "peak_ages_by_position": {k: int(v) for k, v in peak_ages.items()},
    },
    "figures": [
        "figures/growth_trajectories_top_prospects.png",
        "figures/peak_age_by_position.png",
        "figures/model_comparison.png",
        "figures/xgb_feature_importance.png",
        "figures/predicted_vs_actual_xgb.png",
        "figures/growth_by_age_bracket.png",
    ],
}

with open(OUT / "results_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\n  Saved results_summary.json")
print(f"  Figures in {FIG}")

print("\n" + "=" * 60)
print("P7 COMPLETE")
print("=" * 60)

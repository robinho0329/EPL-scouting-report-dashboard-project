"""
P6: Player Market Value Prediction Pipeline
============================================
Predicts a player's fair market value from performance stats,
then identifies undervalued and overvalued players for EPL scouts.

Models: Ridge Regression, XGBoost, Random Forest, Neural Network (MLP)
Target: log(market_value) -- inverse-transformed for final metrics.

Usage:
    python models/p6_market_value/train.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ML
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"C:\Users\xcv54\workspace\EPL project")
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "p6_market_value"
FIG_DIR = MODEL_DIR / "figures"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def mape(y_true, y_pred):
    """Mean Absolute Percentage Error, ignoring zeros."""
    mask = y_true > 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def regression_metrics(y_true, y_pred, prefix=""):
    """Compute MAE, RMSE, R2, MAPE on original euro scale."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mp = mape(y_true, y_pred)
    return {
        f"{prefix}MAE": round(float(mae), 0),
        f"{prefix}RMSE": round(float(rmse), 0),
        f"{prefix}R2": round(float(r2), 4),
        f"{prefix}MAPE": round(float(mp), 2),
    }


# ===================================================================
print("=" * 70)
print("P6: PLAYER MARKET VALUE PREDICTION PIPELINE")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
print("\n[1/8] Loading data...")
features_df = pd.read_parquet(DATA_DIR / "features" / "player_features.parquet")
season_df = pd.read_parquet(DATA_DIR / "processed" / "player_season_stats.parquet")
print(f"  player_features:      {features_df.shape}")
print(f"  player_season_stats:  {season_df.shape}")

# ---------------------------------------------------------------------------
# 2. Prepare target and filter
# ---------------------------------------------------------------------------
print("\n[2/8] Preparing target variable...")
df = features_df.copy()

# Filter out rows with missing or zero market value
before = len(df)
df = df[df["market_value"].notna() & (df["market_value"] > 0)].copy()
print(f"  Rows before filter: {before}")
print(f"  Rows after  filter: {len(df)}  (removed {before - len(df)} with MV=0/NaN)")

# Log-transform target
df["log_market_value"] = np.log1p(df["market_value"])
print(f"  market_value range: {df['market_value'].min():,.0f} - {df['market_value'].max():,.0f}")
print(f"  log(MV) range:      {df['log_market_value'].min():.2f} - {df['log_market_value'].max():.2f}")

# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
print("\n[3/8] Engineering features...")

# Position group encoding
def map_position_group(pos_str):
    if pd.isna(pos_str):
        return "Unknown"
    pos_str = str(pos_str).upper()
    if "GK" in pos_str:
        return "GK"
    elif "DF" in pos_str:
        return "DF"
    elif "MF" in pos_str:
        return "MF"
    elif "FW" in pos_str:
        return "FW"
    return "Unknown"

df["pos_group"] = df["pos"].apply(map_position_group)

# Nationality frequency -- proxy for league / talent pool size
nat_counts = df["nationality"].value_counts()
df["nationality_freq"] = df["nationality"].map(nat_counts).fillna(1)

# Foot encoding
foot_map = {"right": 0, "left": 1, "both": 2}
df["foot_code"] = df["foot"].map(foot_map).fillna(0).astype(int)

# Season as integer (start year) -- centered to reduce extrapolation
df["season_start"] = df["season"].apply(
    lambda s: int(s.split("/")[0]) if isinstance(s, str) and "/" in s else 2020
)
df["season_start"] = df["season_start"] - 2015  # center around 2015

# Age squared (captures non-linear age-value relationship -- peaks ~27)
df["age_sq"] = df["age_used"] ** 2

# Log-transform previous market value (handles inflation / scale issues)
df["log_mv_prev"] = np.log1p(df["mv_prev"].fillna(0))

# Interaction: goals per 90 x minutes share
df["goals_p90_x_mins_share"] = df["goals_p90"].fillna(0) * df["minutes_share"].fillna(0)

# One-hot encode position group
pos_dummies = pd.get_dummies(df["pos_group"], prefix="pos", dtype=int)
df = pd.concat([df, pos_dummies], axis=1)

# ---------------------------------------------------------------------------
# 4. Define feature columns
# ---------------------------------------------------------------------------
numeric_features = [
    # Basic
    "age_used", "age_sq", "mp", "starts", "min", "90s",
    # Counting stats
    "gls", "ast", "g_a", "g_pk", "pk", "pkatt", "crdy", "crdr",
    # Per-90 stats
    "goals_p90", "assists_p90", "goal_contributions_p90",
    "yellow_cards_p90", "red_cards_p90", "penalties_p90",
    # Advanced features
    "log_minutes", "goal_contribution_rate", "minutes_share",
    "epl_experience", "consistency_std", "consistency_mean",
    "consistency_cv", "n_matches", "versatility_positions",
    # Engineered
    "nationality_freq", "foot_code", "season_start",
    "goals_p90_x_mins_share",
    # Previous value signal
    "log_mv_prev", "mv_change_pct",
    # Physical
    "height_cm",
]

# Position dummies (exclude the string column pos_group)
pos_cols = [c for c in df.columns if c.startswith("pos_") and c != "pos_group"]
feature_cols = numeric_features + pos_cols

# Make sure all feature columns exist and fill NaN
for c in feature_cols:
    if c not in df.columns:
        df[c] = 0
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# Clip extreme outliers that break linear models
df["mv_change_pct"] = df["mv_change_pct"].clip(-100, 500)

print(f"  Total features: {len(feature_cols)}")
print(f"  Position dummies: {pos_cols}")

# ---------------------------------------------------------------------------
# 5. Time-based split
# ---------------------------------------------------------------------------
print("\n[4/8] Splitting data (time-based)...")

train_df = df[df["data_split"] == "train"].copy()
val_df = df[df["data_split"] == "val"].copy()
test_df = df[df["data_split"] == "test"].copy()

print(f"  Train: {len(train_df)}  (seasons < 2021/22)")
print(f"  Val:   {len(val_df)}  (2021/22 - 2022/23)")
print(f"  Test:  {len(test_df)}  (2023/24 - 2024/25)")

X_train = train_df[feature_cols].values
y_train = train_df["log_market_value"].values
X_val = val_df[feature_cols].values
y_val = val_df["log_market_value"].values
X_test = test_df[feature_cols].values
y_test = test_df["log_market_value"].values

# Original scale targets for metrics
y_train_orig = train_df["market_value"].values
y_val_orig = val_df["market_value"].values
y_test_orig = test_df["market_value"].values

# Standardize features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc = scaler.transform(X_val)
X_test_sc = scaler.transform(X_test)

joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
print("  Scaler saved.")

# ---------------------------------------------------------------------------
# 6. Train Models
# ---------------------------------------------------------------------------
print("\n[5/8] Training models...")

all_results = {}
test_predictions = {}

# ---- 6a. Ridge Regression (baseline) ----
print("\n  --- Ridge Regression ---")
# Use RobustScaler for Ridge to handle outliers better
ridge_scaler = RobustScaler()
X_train_ridge = ridge_scaler.fit_transform(X_train)
X_val_ridge = ridge_scaler.transform(X_val)
X_test_ridge = ridge_scaler.transform(X_test)
ridge = Ridge(alpha=100.0)
ridge.fit(X_train_ridge, y_train)

for split_name, Xs, ys, yo in [
    ("val", X_val_ridge, y_val, y_val_orig),
    ("test", X_test_ridge, y_test, y_test_orig),
]:
    pred_log = ridge.predict(Xs)
    pred_orig = np.expm1(np.clip(pred_log, 0, 25))
    metrics = regression_metrics(yo, pred_orig, prefix=f"{split_name}_")
    print(f"    {split_name}: R2={metrics[f'{split_name}_R2']:.4f}  "
          f"MAE={metrics[f'{split_name}_MAE']:,.0f}  MAPE={metrics[f'{split_name}_MAPE']:.1f}%")
    if split_name == "test":
        all_results["Ridge"] = metrics
        test_predictions["Ridge"] = pred_orig

joblib.dump(ridge, MODEL_DIR / "ridge_model.joblib")

# ---- 6b. XGBoost ----
print("\n  --- XGBoost ---")
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=30,
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)
print(f"    Best iteration: {xgb_model.best_iteration}")

for split_name, Xs, yo in [("val", X_val, y_val_orig), ("test", X_test, y_test_orig)]:
    pred_log = xgb_model.predict(Xs)
    pred_orig = np.expm1(np.clip(pred_log, 0, 25))
    metrics = regression_metrics(yo, pred_orig, prefix=f"{split_name}_")
    print(f"    {split_name}: R2={metrics[f'{split_name}_R2']:.4f}  "
          f"MAE={metrics[f'{split_name}_MAE']:,.0f}  MAPE={metrics[f'{split_name}_MAPE']:.1f}%")
    if split_name == "test":
        all_results["XGBoost"] = metrics
        test_predictions["XGBoost"] = pred_orig

xgb_model.save_model(str(MODEL_DIR / "xgb_model.json"))

# ---- 6c. Random Forest ----
print("\n  --- Random Forest ---")
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)
rf_model.fit(X_train, y_train)

for split_name, Xs, yo in [("val", X_val, y_val_orig), ("test", X_test, y_test_orig)]:
    pred_log = rf_model.predict(Xs)
    pred_orig = np.expm1(np.clip(pred_log, 0, 25))
    metrics = regression_metrics(yo, pred_orig, prefix=f"{split_name}_")
    print(f"    {split_name}: R2={metrics[f'{split_name}_R2']:.4f}  "
          f"MAE={metrics[f'{split_name}_MAE']:,.0f}  MAPE={metrics[f'{split_name}_MAPE']:.1f}%")
    if split_name == "test":
        all_results["RandomForest"] = metrics
        test_predictions["RandomForest"] = pred_orig

joblib.dump(rf_model, MODEL_DIR / "rf_model.joblib")

# ---- 6d. Neural Network (MLP) ----
print("\n  --- Neural Network (MLP) ---")

class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"    Device: {device}")

mlp = MLPRegressor(X_train_sc.shape[1]).to(device)
optimizer = optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
criterion = nn.HuberLoss(delta=1.0)

# DataLoaders
train_ds = TensorDataset(
    torch.tensor(X_train_sc, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32),
)
val_ds = TensorDataset(
    torch.tensor(X_val_sc, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32),
)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

best_val_loss = float("inf")
patience_counter = 0
max_patience = 25
best_state = None

for epoch in range(200):
    mlp.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = mlp(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)
    train_loss /= len(train_ds)

    mlp.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = mlp(xb)
            val_loss += criterion(pred, yb).item() * len(xb)
    val_loss /= len(val_ds)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
    else:
        patience_counter += 1

    if (epoch + 1) % 25 == 0:
        print(f"    Epoch {epoch+1:3d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    if patience_counter >= max_patience:
        print(f"    Early stopping at epoch {epoch+1}")
        break

mlp.load_state_dict(best_state)
mlp.eval()
torch.save(mlp.state_dict(), MODEL_DIR / "mlp_model.pt")

for split_name, Xs, yo in [("val", X_val_sc, y_val_orig), ("test", X_test_sc, y_test_orig)]:
    with torch.no_grad():
        pred_log = mlp(torch.tensor(Xs, dtype=torch.float32).to(device)).cpu().numpy()
    pred_orig = np.expm1(np.clip(pred_log, 0, 25))
    metrics = regression_metrics(yo, pred_orig, prefix=f"{split_name}_")
    print(f"    {split_name}: R2={metrics[f'{split_name}_R2']:.4f}  "
          f"MAE={metrics[f'{split_name}_MAE']:,.0f}  MAPE={metrics[f'{split_name}_MAPE']:.1f}%")
    if split_name == "test":
        all_results["MLP"] = metrics
        test_predictions["MLP"] = pred_orig

# ---------------------------------------------------------------------------
# 7. Model comparison & best model selection
# ---------------------------------------------------------------------------
print("\n[6/8] Model comparison (test set)...")
print(f"  {'Model':<15} {'R2':>8} {'MAE':>14} {'RMSE':>14} {'MAPE':>8}")
print("  " + "-" * 62)
best_model_name = None
best_r2 = -1e9
for name, met in all_results.items():
    r2 = met["test_R2"]
    print(f"  {name:<15} {r2:>8.4f} {met['test_MAE']:>14,.0f} {met['test_RMSE']:>14,.0f} {met['test_MAPE']:>7.1f}%")
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name

print(f"\n  Best model: {best_model_name} (R2 = {best_r2:.4f})")
best_preds = test_predictions[best_model_name]

# ---------------------------------------------------------------------------
# 8. Value Score & Under/Over-valued players
# ---------------------------------------------------------------------------
print("\n[7/8] Computing Value Scores (test set)...")

test_df = test_df.copy()
test_df["predicted_value"] = best_preds
test_df["value_score"] = test_df["predicted_value"] / test_df["market_value"]

# Undervalued: value_score > 1.5 (model says they are worth MORE than actual)
undervalued = (
    test_df[test_df["value_score"] > 1.0]
    .sort_values("value_score", ascending=False)
    .head(20)
)
# Overvalued: value_score < 0.7
overvalued = (
    test_df[test_df["value_score"] < 1.0]
    .sort_values("value_score", ascending=True)
    .head(20)
)

print("\n  TOP 20 MOST UNDERVALUED PLAYERS (test set 2023-2025)")
print(f"  {'Player':<28} {'Season':<10} {'Pos':<6} {'Age':<5} {'Actual MV':>14} {'Predicted MV':>14} {'Value Score':>12}")
print("  " + "-" * 92)
for _, row in undervalued.iterrows():
    print(f"  {str(row['player'])[:27]:<28} {row['season']:<10} {str(row['pos'])[:5]:<6} "
          f"{row['age_used']:<5.0f} {row['market_value']:>14,.0f} {row['predicted_value']:>14,.0f} "
          f"{row['value_score']:>11.2f}x")

print("\n  TOP 20 MOST OVERVALUED PLAYERS (test set 2023-2025)")
print(f"  {'Player':<28} {'Season':<10} {'Pos':<6} {'Age':<5} {'Actual MV':>14} {'Predicted MV':>14} {'Value Score':>12}")
print("  " + "-" * 92)
for _, row in overvalued.iterrows():
    print(f"  {str(row['player'])[:27]:<28} {row['season']:<10} {str(row['pos'])[:5]:<6} "
          f"{row['age_used']:<5.0f} {row['market_value']:>14,.0f} {row['predicted_value']:>14,.0f} "
          f"{row['value_score']:>11.2f}x")

# Significantly undervalued (>1.5x) and overvalued (<0.7x) counts
sig_under = len(test_df[test_df["value_score"] > 1.5])
sig_over = len(test_df[test_df["value_score"] < 0.7])
print(f"\n  Significantly undervalued (>1.5x): {sig_under} players")
print(f"  Significantly overvalued  (<0.7x): {sig_over} players")

# ---------------------------------------------------------------------------
# 9. Feature Importance
# ---------------------------------------------------------------------------
print("\n[8/8] Feature importance analysis...")

# XGBoost feature importance
xgb_imp = xgb_model.feature_importances_
imp_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": xgb_imp,
}).sort_values("importance", ascending=False)

print("\n  XGBoost Feature Importance (Top 15):")
for i, row in imp_df.head(15).iterrows():
    print(f"    {row['feature']:<35} {row['importance']:.4f}")

# Random Forest feature importance
rf_imp = rf_model.feature_importances_
rf_imp_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf_imp,
}).sort_values("importance", ascending=False)

print("\n  Random Forest Feature Importance (Top 15):")
for i, row in rf_imp_df.head(15).iterrows():
    print(f"    {row['feature']:<35} {row['importance']:.4f}")

imp_df.to_csv(MODEL_DIR / "xgb_feature_importance.csv", index=False)
rf_imp_df.to_csv(MODEL_DIR / "rf_feature_importance.csv", index=False)

# ---------------------------------------------------------------------------
# 10. Scatter plot: Predicted vs Actual
# ---------------------------------------------------------------------------
print("\nGenerating figures...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("P6: Predicted vs Actual Market Value (Test Set)", fontsize=14, fontweight="bold")

for ax, (name, preds) in zip(axes.flatten(), test_predictions.items()):
    actual = y_test_orig / 1e6  # in millions
    predicted = preds / 1e6
    ax.scatter(actual, predicted, alpha=0.3, s=15, edgecolors="none")
    max_val = max(actual.max(), predicted.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual Market Value (M euros)")
    ax.set_ylabel("Predicted Market Value (M euros)")
    ax.set_title(f"{name}  (R2={all_results[name]['test_R2']:.3f})")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / "predicted_vs_actual.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIG_DIR / 'predicted_vs_actual.png'}")

# Value Score distribution
fig, ax = plt.subplots(figsize=(10, 5))
scores = test_df["value_score"].clip(0, 5)
ax.hist(scores, bins=60, edgecolor="black", alpha=0.7, color="steelblue")
ax.axvline(1.5, color="green", linestyle="--", linewidth=1.5, label="Undervalued threshold (1.5x)")
ax.axvline(0.7, color="red", linestyle="--", linewidth=1.5, label="Overvalued threshold (0.7x)")
ax.axvline(1.0, color="black", linestyle="-", linewidth=1, label="Fair value (1.0x)")
ax.set_xlabel("Value Score (predicted / actual)")
ax.set_ylabel("Count")
ax.set_title("Distribution of Value Scores (Test Set 2023-2025)")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "value_score_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIG_DIR / 'value_score_distribution.png'}")

# Feature importance bar chart
fig, ax = plt.subplots(figsize=(10, 7))
top15 = imp_df.head(15).iloc[::-1]
ax.barh(top15["feature"], top15["importance"], color="steelblue", edgecolor="black")
ax.set_xlabel("Importance (XGBoost gain)")
ax.set_title("Top 15 Features for Market Value Prediction")
plt.tight_layout()
plt.savefig(FIG_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIG_DIR / 'feature_importance.png'}")

# ---------------------------------------------------------------------------
# 11. Save results_summary.json
# ---------------------------------------------------------------------------
undervalued_list = []
for _, row in undervalued.iterrows():
    undervalued_list.append({
        "player": str(row["player"]),
        "season": str(row["season"]),
        "team": str(row["team"]),
        "position": str(row["pos"]),
        "age": int(row["age_used"]),
        "actual_market_value": int(row["market_value"]),
        "predicted_market_value": int(row["predicted_value"]),
        "value_score": round(float(row["value_score"]), 3),
    })

overvalued_list = []
for _, row in overvalued.iterrows():
    overvalued_list.append({
        "player": str(row["player"]),
        "season": str(row["season"]),
        "team": str(row["team"]),
        "position": str(row["pos"]),
        "age": int(row["age_used"]),
        "actual_market_value": int(row["market_value"]),
        "predicted_market_value": int(row["predicted_value"]),
        "value_score": round(float(row["value_score"]), 3),
    })

results_summary = {
    "pipeline": "P6 - Player Market Value Prediction",
    "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "data": {
        "total_rows_with_value": len(df),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "num_features": len(feature_cols),
        "feature_columns": feature_cols,
        "target": "log1p(market_value)",
    },
    "best_model": best_model_name,
    "model_metrics": all_results,
    "value_analysis": {
        "significantly_undervalued_count": sig_under,
        "significantly_overvalued_count": sig_over,
        "undervalued_threshold": ">1.5x",
        "overvalued_threshold": "<0.7x",
    },
    "top20_undervalued": undervalued_list,
    "top20_overvalued": overvalued_list,
    "feature_importance_xgb_top10": imp_df.head(10)[["feature", "importance"]].to_dict(orient="records"),
    "feature_importance_rf_top10": rf_imp_df.head(10)[["feature", "importance"]].to_dict(orient="records"),
}

with open(MODEL_DIR / "results_summary.json", "w") as f:
    json.dump(results_summary, f, indent=2)
print(f"\n  Saved: {MODEL_DIR / 'results_summary.json'}")

# Save feature columns for reproducibility
with open(MODEL_DIR / "feature_cols.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

print("\n" + "=" * 70)
print("P6 PIPELINE COMPLETE")
print("=" * 70)

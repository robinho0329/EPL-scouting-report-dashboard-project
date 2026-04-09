"""
P4: MVP (Most Valuable Player) Scoring Pipeline
=================================================
Builds a composite MVP score for EPL player-seasons, then trains
Linear Regression, XGBoost, Neural Network, and LambdaRank models
to predict/rank MVPs.

Usage:
    python models/p4_mvp/mvp_pipeline.py
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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"C:\Users\xcv54\workspace\EPL project")
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "p4_mvp"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
print("=" * 70)
print("P4: MVP SCORING PIPELINE")
print("=" * 70)

print("\n[1/7] Loading data...")
features_df = pd.read_parquet(DATA_DIR / "features" / "player_features.parquet")
match_logs = pd.read_parquet(DATA_DIR / "processed" / "player_match_logs.parquet")
print(f"  player_features: {features_df.shape}")
print(f"  player_match_logs: {match_logs.shape}")


# ---------------------------------------------------------------------------
# 2. Position Grouping
# ---------------------------------------------------------------------------
def map_position_group(pos_str):
    """Map FBref pos column to GK/DEF/MID/FWD."""
    if pd.isna(pos_str):
        return "MID"  # default
    pos = pos_str.upper().split(",")[0].strip()
    if pos == "GK":
        return "GK"
    elif pos == "DF":
        return "DEF"
    elif pos == "MF":
        return "MID"
    elif pos == "FW":
        return "FWD"
    else:
        return "MID"


features_df["pos_group"] = features_df["pos"].apply(map_position_group)
print(f"\n  Position groups: {features_df['pos_group'].value_counts().to_dict()}")


# ---------------------------------------------------------------------------
# 3. Compute Composite MVP Score
# ---------------------------------------------------------------------------
print("\n[2/7] Computing composite MVP score...")

# We build a position-aware composite score from available features.
# The score components:
#   1. Goal contributions per 90 (weighted by position)
#   2. Minutes played (log-scaled, with 1000 min threshold as soft gate)
#   3. Consistency (inverse of CV where available)
#   4. Market value (log-scaled, proxy for quality)
#   5. Appearances share (minutes_share within team)

df = features_df.copy()

# --- 3a. Fill missing per-90 stats with 0 (player didn't play enough) ---
per90_cols = ["goals_p90", "assists_p90", "goal_contributions_p90",
              "yellow_cards_p90", "red_cards_p90"]
for c in per90_cols:
    df[c] = df[c].fillna(0.0)

# Fill other needed columns
df["market_value"] = df["market_value"].fillna(0.0)
df["consistency_cv"] = df["consistency_cv"].fillna(df["consistency_cv"].median())
df["consistency_std"] = df["consistency_std"].fillna(df["consistency_std"].median())
df["consistency_mean"] = df["consistency_mean"].fillna(0.0)
df["minutes_share"] = df["minutes_share"].fillna(0.0)
df["min"] = df["min"].fillna(0.0)
df["gls"] = df["gls"].fillna(0.0)
df["ast"] = df["ast"].fillna(0.0)
df["g_a"] = df["g_a"].fillna(0.0)
df["mp"] = df["mp"].fillna(0)

# --- 3b. Position-specific weights for goal contributions ---
pos_gc_weight = {"GK": 0.15, "DEF": 0.50, "MID": 0.80, "FWD": 1.00}
df["gc_weight"] = df["pos_group"].map(pos_gc_weight)

# --- 3c. Minutes gate: soft sigmoid around 1000 min threshold ---
def minutes_gate(minutes, threshold=1000, steepness=0.005):
    return 1.0 / (1.0 + np.exp(-steepness * (minutes - threshold)))

df["min_gate"] = minutes_gate(df["min"])

# --- 3d. Log market value (add 1 to avoid log(0)) ---
df["log_mv"] = np.log1p(df["market_value"])
# Normalize to 0-1 range
mv_max = df["log_mv"].max()
df["log_mv_norm"] = df["log_mv"] / mv_max if mv_max > 0 else 0.0

# --- 3e. Consistency score: lower CV = more consistent = better ---
# Invert and normalize
cv_median = df["consistency_cv"].median()
cv_max = df["consistency_cv"].quantile(0.99)
df["consistency_score"] = 1.0 - (df["consistency_cv"].clip(0, cv_max) / cv_max)
df["consistency_score"] = df["consistency_score"].fillna(0.5)

# --- 3f. Goals + assists raw contribution (season totals) ---
df["ga_total"] = df["gls"] + df["ast"]

# --- 3g. Composite MVP Score ---
# Weighted formula:
#   MVP = min_gate * (
#       0.30 * goal_contributions_p90 * pos_weight (scaled)
#     + 0.20 * log_mv_norm
#     + 0.15 * consistency_score
#     + 0.20 * minutes_share
#     + 0.15 * ga_total_scaled
#   )

# Scale goal_contributions_p90 to ~0-1
gc_p90_max = df["goal_contributions_p90"].quantile(0.99)
df["gc_p90_scaled"] = (df["goal_contributions_p90"] / gc_p90_max).clip(0, 1) if gc_p90_max > 0 else 0.0

# Scale ga_total
ga_max = df["ga_total"].quantile(0.99)
df["ga_total_scaled"] = (df["ga_total"] / ga_max).clip(0, 1) if ga_max > 0 else 0.0

df["mvp_score"] = df["min_gate"] * (
    0.30 * df["gc_p90_scaled"] * df["gc_weight"]
    + 0.20 * df["log_mv_norm"]
    + 0.15 * df["consistency_score"]
    + 0.20 * df["minutes_share"]
    + 0.15 * df["ga_total_scaled"]
)

# Scale to 0-100
mvp_max = df["mvp_score"].quantile(0.999)
df["mvp_score"] = (df["mvp_score"] / mvp_max * 100).clip(0, 100)

print(f"  MVP score stats:\n{df['mvp_score'].describe()}")

# Show top 10 overall
top10_all = df.nlargest(10, "mvp_score")[["player", "season", "team", "pos_group",
                                           "gls", "ast", "min", "market_value", "mvp_score"]]
print(f"\n  Top 10 MVP scores (all time):\n{top10_all.to_string(index=False)}")


# ---------------------------------------------------------------------------
# 4. Prepare Features for ML Models
# ---------------------------------------------------------------------------
print("\n[3/7] Preparing features for ML models...")

# Feature columns (robust to missing data in early seasons)
FEATURE_COLS = [
    "goals_p90", "assists_p90", "goal_contributions_p90",
    "yellow_cards_p90", "red_cards_p90",
    "log_mv_norm", "consistency_score", "minutes_share",
    "ga_total_scaled", "gc_p90_scaled", "min_gate",
    "mp", "starts", "90s", "gls", "ast", "g_a",
    "age_used", "epl_experience",
]

# Position one-hot
for pg in ["GK", "DEF", "MID", "FWD"]:
    df[f"pos_{pg}"] = (df["pos_group"] == pg).astype(float)
    FEATURE_COLS.append(f"pos_{pg}")

# Fill remaining NaNs
df["age_used"] = df["age_used"].fillna(df["age"].fillna(26))
df["epl_experience"] = df["epl_experience"].fillna(0)
df["starts"] = df["starts"].fillna(0)
df["90s"] = df["90s"].fillna(0)
df["g_a"] = df["g_a"].fillna(0)

TARGET_COL = "mvp_score"

# Split by data_split column
train_df = df[df["data_split"] == "train"].copy()
val_df = df[df["data_split"] == "val"].copy()
test_df = df[df["data_split"] == "test"].copy()

print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Extract arrays
X_train = train_df[FEATURE_COLS].values.astype(np.float32)
y_train = train_df[TARGET_COL].values.astype(np.float32)
X_val = val_df[FEATURE_COLS].values.astype(np.float32)
y_val = val_df[TARGET_COL].values.astype(np.float32)
X_test = test_df[FEATURE_COLS].values.astype(np.float32)
y_test = test_df[TARGET_COL].values.astype(np.float32)

# Handle any remaining NaN in feature arrays
X_train = np.nan_to_num(X_train, nan=0.0)
X_val = np.nan_to_num(X_val, nan=0.0)
X_test = np.nan_to_num(X_test, nan=0.0)
y_train = np.nan_to_num(y_train, nan=0.0)
y_val = np.nan_to_num(y_val, nan=0.0)
y_test = np.nan_to_num(y_test, nan=0.0)

# Scale features
scaler = RobustScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
print(f"  Feature columns: {len(FEATURE_COLS)}")


# ---------------------------------------------------------------------------
# Evaluation Helpers
# ---------------------------------------------------------------------------
def ndcg_at_k(y_true, y_pred, k=10):
    """Compute NDCG@k."""
    # Get top-k predicted indices
    pred_order = np.argsort(-y_pred)[:k]
    true_order = np.argsort(-y_true)[:k]

    # Relevance = actual MVP score (higher = more relevant)
    # DCG
    dcg = 0.0
    for i, idx in enumerate(pred_order):
        rel = y_true[idx]
        dcg += rel / np.log2(i + 2)

    # Ideal DCG
    idcg = 0.0
    sorted_true = np.sort(y_true)[::-1][:k]
    for i, rel in enumerate(sorted_true):
        idcg += rel / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(y_true, y_pred, split_name=""):
    """Compute MAE, RMSE, Spearman, NDCG@10."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    spearman_corr, _ = spearmanr(y_true, y_pred)
    ndcg10 = ndcg_at_k(y_true, y_pred, k=10)
    metrics = {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "Spearman": round(float(spearman_corr), 4),
        "NDCG@10": round(float(ndcg10), 4),
    }
    if split_name:
        print(f"    {split_name}: MAE={mae:.4f}  RMSE={rmse:.4f}  "
              f"Spearman={spearman_corr:.4f}  NDCG@10={ndcg10:.4f}")
    return metrics


# ---------------------------------------------------------------------------
# 5. Train Models
# ---------------------------------------------------------------------------
results = {}

# ---- 5a. Linear Regression (Ridge) ----
print("\n[4/7] Training Linear Regression (Ridge)...")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_s, y_train)

pred_train_lr = ridge.predict(X_train_s)
pred_val_lr = ridge.predict(X_val_s)
pred_test_lr = ridge.predict(X_test_s)

results["linear_regression"] = {
    "train": evaluate_model(y_train, pred_train_lr, "Train"),
    "val": evaluate_model(y_val, pred_val_lr, "Val"),
    "test": evaluate_model(y_test, pred_test_lr, "Test"),
}
joblib.dump(ridge, MODEL_DIR / "ridge_model.joblib")
print("  Saved ridge_model.joblib")


# ---- 5b. XGBoost Regressor ----
print("\n[5/7] Training XGBoost Regressor...")
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
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

pred_train_xgb = xgb_model.predict(X_train)
pred_val_xgb = xgb_model.predict(X_val)
pred_test_xgb = xgb_model.predict(X_test)

results["xgboost"] = {
    "train": evaluate_model(y_train, pred_train_xgb, "Train"),
    "val": evaluate_model(y_val, pred_val_xgb, "Val"),
    "test": evaluate_model(y_test, pred_test_xgb, "Test"),
}
xgb_model.save_model(str(MODEL_DIR / "xgb_model.json"))
print("  Saved xgb_model.json")

# Feature importance
importance = xgb_model.feature_importances_
feat_imp = sorted(zip(FEATURE_COLS, importance), key=lambda x: -x[1])
print("  Top 10 feature importances:")
for fname, imp in feat_imp[:10]:
    print(f"    {fname}: {imp:.4f}")


# ---- 5c. Neural Network (MLP) ----
print("\n[6/7] Training Neural Network (MLP)...")

class MVPNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

input_dim = X_train_s.shape[1]
mlp_model = MVPNet(input_dim).to(device)
optimizer = optim.AdamW(mlp_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
criterion = nn.HuberLoss(delta=5.0)

# DataLoaders
train_ds = TensorDataset(torch.tensor(X_train_s, dtype=torch.float32),
                         torch.tensor(y_train, dtype=torch.float32))
val_ds = TensorDataset(torch.tensor(X_val_s, dtype=torch.float32),
                       torch.tensor(y_val, dtype=torch.float32))
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

best_val_loss = float("inf")
patience_counter = 0
max_patience = 25
best_state = None

for epoch in range(200):
    mlp_model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = mlp_model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)
    train_loss /= len(train_ds)

    # Validation
    mlp_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = mlp_model(xb)
            val_loss += criterion(pred, yb).item() * len(xb)
    val_loss /= len(val_ds)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = {k: v.cpu().clone() for k, v in mlp_model.state_dict().items()}
    else:
        patience_counter += 1

    if (epoch + 1) % 20 == 0:
        print(f"    Epoch {epoch+1}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    if patience_counter >= max_patience:
        print(f"    Early stopping at epoch {epoch+1}")
        break

# Load best model
mlp_model.load_state_dict(best_state)
mlp_model.eval()

def predict_mlp(X_scaled):
    with torch.no_grad():
        t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        return mlp_model(t).cpu().numpy()

pred_train_mlp = predict_mlp(X_train_s)
pred_val_mlp = predict_mlp(X_val_s)
pred_test_mlp = predict_mlp(X_test_s)

results["neural_network"] = {
    "train": evaluate_model(y_train, pred_train_mlp, "Train"),
    "val": evaluate_model(y_val, pred_val_mlp, "Val"),
    "test": evaluate_model(y_test, pred_test_mlp, "Test"),
}
torch.save(best_state, MODEL_DIR / "mlp_model.pt")
print("  Saved mlp_model.pt")


# ---- 5d. LambdaRank (Learning-to-Rank via XGBoost) ----
print("\n[6.5/7] Training LambdaRank (XGBoost rank:ndcg)...")

# Group by season for ranking
train_df_sorted = train_df.sort_values("season").reset_index(drop=True)
val_df_sorted = val_df.sort_values("season").reset_index(drop=True)
test_df_sorted = test_df.sort_values("season").reset_index(drop=True)

X_train_ltr = np.nan_to_num(train_df_sorted[FEATURE_COLS].values.astype(np.float32), nan=0.0)
y_train_ltr_raw = np.nan_to_num(train_df_sorted[TARGET_COL].values.astype(np.float32), nan=0.0)
X_val_ltr = np.nan_to_num(val_df_sorted[FEATURE_COLS].values.astype(np.float32), nan=0.0)
y_val_ltr_raw = np.nan_to_num(val_df_sorted[TARGET_COL].values.astype(np.float32), nan=0.0)
X_test_ltr = np.nan_to_num(test_df_sorted[FEATURE_COLS].values.astype(np.float32), nan=0.0)
y_test_ltr_raw = np.nan_to_num(test_df_sorted[TARGET_COL].values.astype(np.float32), nan=0.0)

# XGBRanker with rank:ndcg needs non-negative integer labels <= 31
# Scale 0-100 to 0-31 range
y_train_ltr = np.round(y_train_ltr_raw * 31.0 / 100.0).astype(int).clip(0, 31)
y_val_ltr = np.round(y_val_ltr_raw * 31.0 / 100.0).astype(int).clip(0, 31)
y_test_ltr = np.round(y_test_ltr_raw * 31.0 / 100.0).astype(int).clip(0, 31)

# Build group sizes (players per season)
train_groups = train_df_sorted.groupby("season").size().values
val_groups = val_df_sorted.groupby("season").size().values
test_groups = test_df_sorted.groupby("season").size().values

ltr_model = xgb.XGBRanker(
    objective="rank:ndcg",
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)
ltr_model.fit(
    X_train_ltr, y_train_ltr,
    group=train_groups,
    eval_set=[(X_val_ltr, y_val_ltr)],
    eval_group=[val_groups],
    verbose=False,
)

pred_train_ltr = ltr_model.predict(X_train_ltr)
pred_val_ltr = ltr_model.predict(X_val_ltr)
pred_test_ltr = ltr_model.predict(X_test_ltr)

results["lambdarank"] = {
    "train": evaluate_model(y_train_ltr_raw, pred_train_ltr, "Train"),
    "val": evaluate_model(y_val_ltr_raw, pred_val_ltr, "Val"),
    "test": evaluate_model(y_test_ltr_raw, pred_test_ltr, "Test"),
}
ltr_model.save_model(str(MODEL_DIR / "ltr_model.json"))
print("  Saved ltr_model.json")


# ---------------------------------------------------------------------------
# 6. Top 10 Predicted MVPs for 2023-2025 Test Seasons
# ---------------------------------------------------------------------------
print("\n[7/7] Top 10 Predicted MVPs for 2023-2025 test seasons...")
print("=" * 70)

# Collect test predictions from each model
test_pred_df = test_df[["player", "season", "team", "pos_group",
                         "gls", "ast", "min", "market_value", "mvp_score"]].copy()
test_pred_df["pred_ridge"] = pred_test_lr
test_pred_df["pred_xgb"] = pred_test_xgb
test_pred_df["pred_mlp"] = pred_test_mlp

# LTR predictions need to be aligned to test_df_sorted
test_pred_ltr_df = test_df_sorted[["player", "season", "team"]].copy()
test_pred_ltr_df["pred_ltr"] = pred_test_ltr
# Merge back
test_pred_df = test_pred_df.merge(
    test_pred_ltr_df[["player", "season", "team", "pred_ltr"]],
    on=["player", "season", "team"],
    how="left",
)

# Ensemble (average of all 4 models)
test_pred_df["pred_ensemble"] = (
    test_pred_df["pred_ridge"]
    + test_pred_df["pred_xgb"]
    + test_pred_df["pred_mlp"]
    + test_pred_df["pred_ltr"].fillna(0)
) / 4

# Show per-season top 10
for season in sorted(test_pred_df["season"].unique()):
    sdf = test_pred_df[test_pred_df["season"] == season]
    print(f"\n--- {season} ---")
    for model_name, col in [("Actual MVP Score", "mvp_score"),
                             ("Ridge", "pred_ridge"),
                             ("XGBoost", "pred_xgb"),
                             ("MLP", "pred_mlp"),
                             ("LambdaRank", "pred_ltr"),
                             ("Ensemble", "pred_ensemble")]:
        top10 = sdf.nlargest(10, col)
        print(f"\n  {model_name} Top 10:")
        for rank, (_, row) in enumerate(top10.iterrows(), 1):
            print(f"    {rank:2d}. {row['player']:25s} ({row['team']:15s}) "
                  f"Pos={row['pos_group']:3s}  Score={row[col]:6.2f}  "
                  f"G={int(row['gls']):2d}  A={int(row['ast']):2d}  "
                  f"Min={int(row['min']):5d}")


# ---------------------------------------------------------------------------
# 7. Save Results Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

# Comparison table
print(f"\n{'Model':<20} {'Split':<8} {'MAE':>8} {'RMSE':>8} {'Spearman':>10} {'NDCG@10':>9}")
print("-" * 65)
for model_name, model_results in results.items():
    for split_name, metrics in model_results.items():
        print(f"{model_name:<20} {split_name:<8} {metrics['MAE']:>8.4f} "
              f"{metrics['RMSE']:>8.4f} {metrics['Spearman']:>10.4f} {metrics['NDCG@10']:>9.4f}")

# Build the overall top-10 for 2023-2025 (ensemble)
overall_top10 = test_pred_df.nlargest(10, "pred_ensemble")[
    ["player", "season", "team", "pos_group", "gls", "ast", "min",
     "mvp_score", "pred_ensemble"]
].to_dict(orient="records")

# Convert numpy types for JSON serialization
def convert_np(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

summary = {
    "pipeline": "P4_MVP_Scoring",
    "created": datetime.now().isoformat(),
    "data_splits": {
        "train": f"2000-2021 ({len(train_df)} samples)",
        "val": f"2021-2023 ({len(val_df)} samples)",
        "test": f"2023-2025 ({len(test_df)} samples)",
    },
    "features": FEATURE_COLS,
    "target": "mvp_score (composite, 0-100)",
    "mvp_formula": {
        "description": "Position-aware weighted composite of per-90 goal contributions, "
                       "market value, consistency, minutes share, and total goal+assist",
        "weights": {
            "goal_contributions_p90 * pos_weight": 0.30,
            "log_market_value_norm": 0.20,
            "consistency_score": 0.15,
            "minutes_share": 0.20,
            "ga_total_scaled": 0.15,
        },
        "minutes_threshold": "Soft sigmoid gate at 1000 minutes",
        "position_weights": pos_gc_weight,
    },
    "models": {},
    "top10_mvp_2023_2025_ensemble": [],
}

for model_name, model_results in results.items():
    summary["models"][model_name] = {}
    for split_name, metrics in model_results.items():
        summary["models"][model_name][split_name] = {
            k: convert_np(v) for k, v in metrics.items()
        }

for rec in overall_top10:
    summary["top10_mvp_2023_2025_ensemble"].append(
        {k: convert_np(v) for k, v in rec.items()}
    )

with open(MODEL_DIR / "results_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=convert_np)

print(f"\nResults saved to {MODEL_DIR / 'results_summary.json'}")

# Also save the full test predictions
test_pred_df.to_parquet(MODEL_DIR / "test_predictions.parquet", index=False)
print(f"Test predictions saved to {MODEL_DIR / 'test_predictions.parquet'}")

# Save feature columns for future use
with open(MODEL_DIR / "feature_cols.json", "w") as f:
    json.dump(FEATURE_COLS, f)

print("\nAll models and artifacts saved to:", MODEL_DIR)
print("Pipeline complete!")

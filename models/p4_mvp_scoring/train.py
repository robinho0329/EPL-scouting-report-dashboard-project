"""
P4: MVP (Most Valuable Player) Scoring Model
=============================================
Builds a composite MVP score per player per season, then trains:
  - XGBoost Regressor
  - MLP (PyTorch)
  - Attention-based MLP (PyTorch)

Time-based split:
  Train : 2000/01 – 2020/21
  Val   : 2021/22 – 2022/23
  Test  : 2023/24 – 2024/25
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import xgboost as xgb

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE = Path("C:/Users/xcv54/workspace/EPL project")
DATA = BASE / "data" / "processed"
OUT  = BASE / "models" / "p4_mvp_scoring"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_SEASONS = [f"{y}/{str(y+1)[-2:]}" for y in range(2000, 2021)]   # 2000/01-2020/21
VAL_SEASONS   = ["2021/22", "2022/23"]
TEST_SEASONS  = ["2023/24", "2024/25"]

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("Loading data …")
pss  = pd.read_parquet(DATA / "player_season_stats.parquet")
logs = pd.read_parquet(DATA / "player_match_logs.parquet")
tss  = pd.read_parquet(DATA / "team_season_summary.parquet")

# Standardise season column name in team table
tss = tss.rename(columns={"Season": "season"})

# ─────────────────────────────────────────────────────────────────────────────
# 2. Derive team league position (rank by points, lower rank = better)
# ─────────────────────────────────────────────────────────────────────────────
tss["team_rank"] = tss.groupby("season")["points"].rank(
    ascending=False, method="min"
)
tss["teams_in_season"] = tss.groupby("season")["team"].transform("count")
# Normalised position: 1 = champion, 0 = last
tss["team_strength"] = 1 - (tss["team_rank"] - 1) / (tss["teams_in_season"] - 1)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Primary position mapping  (FW / MF / DF / GK)
# ─────────────────────────────────────────────────────────────────────────────
def primary_pos(pos_str):
    if pd.isna(pos_str):
        return "MF"
    p = str(pos_str).strip().upper().split(",")[0]
    if p == "GK":
        return "GK"
    if p == "DF":
        return "DF"
    if p == "FW":
        return "FW"
    return "MF"

pss["primary_pos"] = pss["pos"].apply(primary_pos)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Clean-sheet aggregation from match logs
# ─────────────────────────────────────────────────────────────────────────────
print("Computing clean sheets from match logs …")

# A clean sheet match = goals_against == 0 and player played (min > 0)
cs_df = logs[logs["min"].fillna(0) > 0].copy()
cs_df["clean_sheet_match"] = (cs_df["goals_against"] == 0).astype(int)
cs_agg = (
    cs_df.groupby(["player", "season"])["clean_sheet_match"]
    .sum()
    .reset_index()
    .rename(columns={"clean_sheet_match": "clean_sheets"})
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Consistency: std of per-match goals + assists from match logs
# ─────────────────────────────────────────────────────────────────────────────
print("Computing per-match consistency …")
logs["ga_match"] = logs["gls"].fillna(0) + logs["ast"].fillna(0)
cons_agg = (
    logs[logs["min"].fillna(0) > 0]
    .groupby(["player", "season"])["ga_match"]
    .std()
    .reset_index()
    .rename(columns={"ga_match": "ga_std"})
)
# Invert: lower std = more consistent = higher bonus
# We'll use 1/(1+std) as the consistency multiplier

# ─────────────────────────────────────────────────────────────────────────────
# 6. Team goal involvement  (player G+A / team total goals)
# ─────────────────────────────────────────────────────────────────────────────
team_goals = (
    tss[["season", "team", "total_goals_for"]]
    .copy()
)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Merge everything into a single season-player frame
# ─────────────────────────────────────────────────────────────────────────────
print("Merging datasets …")
df = pss.copy()
df = df.merge(cs_agg, on=["player", "season"], how="left")
df = df.merge(cons_agg, on=["player", "season"], how="left")
df = df.merge(team_goals, on=["season", "team"], how="left")
df = df.merge(
    tss[["season", "team", "team_rank", "team_strength", "points"]],
    on=["season", "team"],
    how="left",
)

df["clean_sheets"] = df["clean_sheets"].fillna(0)
df["ga_std"]       = df["ga_std"].fillna(0)
df["total_goals_for"] = df["total_goals_for"].fillna(1)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Minimum minutes filter (900 min = 10 full matches)
# ─────────────────────────────────────────────────────────────────────────────
df = df[df["min"] >= 900].reset_index(drop=True)
print(f"Players with >=900 min: {len(df)}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. MVP Score Construction
# ─────────────────────────────────────────────────────────────────────────────
print("Building MVP composite score …")

GOAL_WEIGHT = {"FW": 1.0, "MF": 1.5, "DF": 2.0, "GK": 3.0}
AST_WEIGHT  = 1.2
CS_WEIGHT   = 1.5

def mvp_score_raw(row):
    pos    = row["primary_pos"]
    gw     = GOAL_WEIGHT.get(pos, 1.5)
    goals  = row["gls"]  if not pd.isna(row["gls"])  else 0
    assists= row["ast"]  if not pd.isna(row["ast"])  else 0
    cs     = row["clean_sheets"]
    minutes= row["min"]

    # Weighted stats
    goal_component   = gw * goals
    assist_component = AST_WEIGHT * assists
    cs_component     = CS_WEIGHT * cs if pos in ("DF", "GK") else 0

    # Minutes played bonus (log scale, normalised to ~38 games = 3420 min)
    minutes_bonus = np.log1p(minutes) / np.log1p(3420)

    # Consistency bonus  (capped between 0 and 1)
    consistency_bonus = 1 / (1 + row["ga_std"])

    # Team contribution: player G+A share of team goals (capped at 1)
    team_goals_total = max(row["total_goals_for"], 1)
    involvement = min((goals + assists) / team_goals_total, 1.0)

    raw = (
        goal_component
        + assist_component
        + cs_component
        + 5 * minutes_bonus          # scale factor so minutes matter
        + 3 * consistency_bonus      # consistency reward
        + 10 * involvement           # team contribution
    )
    return raw

df["mvp_raw"] = df.apply(mvp_score_raw, axis=1)

# Normalise to 0-100 per season
def normalise_season(grp):
    mn, mx = grp["mvp_raw"].min(), grp["mvp_raw"].max()
    if mx == mn:
        grp["mvp_score"] = 50.0
    else:
        grp["mvp_score"] = (grp["mvp_raw"] - mn) / (mx - mn) * 100
    return grp

df = df.groupby("season", group_keys=False).apply(normalise_season)
print(f"MVP score stats:\n{df['mvp_score'].describe()}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
print("Engineering features …")

# 10a. Per-90 stats
safe90 = df["90s"].replace(0, np.nan)
df["gls_p90"]  = df["gls"]  / safe90
df["ast_p90"]  = df["ast"]  / safe90
df["sh_p90"]   = df.get("sh",  pd.Series(np.nan, index=df.index)) / safe90 if "sh"  in df.columns else np.nan
df["sot_p90"]  = df.get("sot", pd.Series(np.nan, index=df.index)) / safe90 if "sot" in df.columns else np.nan

# Per-90 from logs (for players with detail stats)
logs_per90 = (
    logs[logs["detail_stats_available"]]
    .groupby(["player", "season"])
    .agg(
        total_sh   =("sh",   "sum"),
        total_tklw =("tklw", "sum"),
        total_int  =("int",  "sum"),
        total_90s  =("min",  lambda x: x.sum() / 90),
    )
    .reset_index()
)
logs_per90["sh_p90_log"]   = logs_per90["total_sh"]   / logs_per90["total_90s"].replace(0, np.nan)
logs_per90["tklw_p90_log"] = logs_per90["total_tklw"] / logs_per90["total_90s"].replace(0, np.nan)
logs_per90["int_p90_log"]  = logs_per90["total_int"]  / logs_per90["total_90s"].replace(0, np.nan)

df = df.merge(
    logs_per90[["player", "season", "sh_p90_log", "tklw_p90_log", "int_p90_log"]],
    on=["player", "season"],
    how="left",
)

# 10b. Age (use age_tm if available, else age)
df["player_age"] = df["age_tm"].fillna(df["age"])

# 10c. Market value (log-transform, fill missing with 0)
df["log_market_value"] = np.log1p(df["market_value"].fillna(0))

# 10d. Minutes share (% of possible 38 * 90 = 3420 min)
df["minutes_share"] = df["min"] / 3420.0

# 10e. Position dummies
pos_dummies = pd.get_dummies(df["primary_pos"], prefix="pos")
df = pd.concat([df, pos_dummies], axis=1)

# 10f. Experience: cumulative seasons up to and including current
df = df.sort_values(["player", "season"])
df["experience"] = df.groupby("player").cumcount() + 1

# 10g. Improvement trend: delta in per-90 goals vs previous season
df["prev_gls_p90"] = df.groupby("player")["gls_p90"].shift(1)
df["delta_gls_p90"] = df["gls_p90"] - df["prev_gls_p90"]

df["prev_ast_p90"] = df.groupby("player")["ast_p90"].shift(1)
df["delta_ast_p90"] = df["ast_p90"] - df["prev_ast_p90"]

# 10h. Team strength features
df["team_strength"]  = df["team_strength"].fillna(0.5)
df["team_points"]    = df["points"].fillna(df.groupby("season")["points"].transform("mean"))
df["team_rank_norm"] = df["team_rank"].fillna(10) / df["teams_in_season"].fillna(20)

# 10i. Fill remaining NaN per-90 with 0
per90_cols = ["gls_p90", "ast_p90", "sh_p90_log", "tklw_p90_log", "int_p90_log",
              "delta_gls_p90", "delta_ast_p90", "prev_gls_p90", "prev_ast_p90"]
for c in per90_cols:
    if c in df.columns:
        df[c] = df[c].fillna(0)

# ─────────────────────────────────────────────────────────────────────────────
# 11. Feature list
# ─────────────────────────────────────────────────────────────────────────────
BASE_FEATURES = [
    "gls_p90", "ast_p90",
    "sh_p90_log", "tklw_p90_log", "int_p90_log",
    "player_age", "log_market_value",
    "minutes_share", "clean_sheets",
    "team_strength", "team_points", "team_rank_norm",
    "experience",
    "delta_gls_p90", "delta_ast_p90",
    "ga_std",
]
POS_COLS = [c for c in df.columns if c.startswith("pos_")]
FEATURES = BASE_FEATURES + POS_COLS

TARGET = "mvp_score"

print(f"Feature set ({len(FEATURES)}): {FEATURES}")

# ─────────────────────────────────────────────────────────────────────────────
# 12. Time-based train / val / test split
# ─────────────────────────────────────────────────────────────────────────────
train_df = df[df["season"].isin(TRAIN_SEASONS)].copy()
val_df   = df[df["season"].isin(VAL_SEASONS)].copy()
test_df  = df[df["season"].isin(TEST_SEASONS)].copy()

print(f"\nSplit sizes  → train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

X_train = train_df[FEATURES].values.astype(np.float32)
y_train = train_df[TARGET].values.astype(np.float32)
X_val   = val_df[FEATURES].values.astype(np.float32)
y_val   = val_df[TARGET].values.astype(np.float32)
X_test  = test_df[FEATURES].values.astype(np.float32)
y_test  = test_df[TARGET].values.astype(np.float32)

# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# 13. Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────
def regression_metrics(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  [{label}] MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}")
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def top10_spearman(df_split, y_pred, label=""):
    """Spearman correlation of predicted vs actual ranks among top-10 per season."""
    tmp = df_split[["player", "season", TARGET]].copy()
    tmp["pred"] = y_pred
    corrs = []
    for season, grp in tmp.groupby("season"):
        top_actual = grp.nlargest(10, TARGET)
        pred_vals  = top_actual["pred"].values
        true_vals  = top_actual[TARGET].values
        if len(true_vals) >= 2:
            rho, _ = spearmanr(true_vals, pred_vals)
            corrs.append(rho)
    mean_corr = float(np.nanmean(corrs))
    print(f"  [{label}] Top-10 Spearman corr = {mean_corr:.4f}")
    return mean_corr


def get_top10_per_season(df_split, y_pred, label=""):
    tmp = df_split[["player", "season", "team", "primary_pos", TARGET]].copy()
    tmp["pred_score"] = y_pred
    results = {}
    for season, grp in tmp.groupby("season"):
        top10 = grp.nlargest(10, "pred_score")[
            ["player", "team", "primary_pos", "pred_score", TARGET]
        ].reset_index(drop=True)
        top10.index += 1
        results[season] = top10.to_dict(orient="records")
    return results

# ─────────────────────────────────────────────────────────────────────────────
# 14.  Model A – XGBoost
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Training XGBoost …")

xgb_model = xgb.XGBRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1,
    verbosity=0,
    eval_metric="rmse",
    early_stopping_rounds=30,
)
xgb_model.fit(
    X_train_s, y_train,
    eval_set=[(X_val_s, y_val)],
    verbose=False,
)

xgb_val_pred  = xgb_model.predict(X_val_s)
xgb_test_pred = xgb_model.predict(X_test_s)

print("  Validation:")
xgb_val_metrics  = regression_metrics(y_val,  xgb_val_pred,  "XGB-val")
xgb_val_sp       = top10_spearman(val_df,  xgb_val_pred,  "XGB-val")
print("  Test:")
xgb_test_metrics = regression_metrics(y_test, xgb_test_pred, "XGB-test")
xgb_test_sp      = top10_spearman(test_df, xgb_test_pred, "XGB-test")

# Feature importance
feat_imp = dict(zip(FEATURES, xgb_model.feature_importances_.tolist()))
feat_imp_sorted = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))
print("\n  Top-10 features by importance:")
for i, (k, v) in enumerate(list(feat_imp_sorted.items())[:10]):
    print(f"    {i+1:2d}. {k:<30s} {v:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 15.  Model B – MLP (PyTorch)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Training MLP (PyTorch) …")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Using device: {DEVICE}")

def make_loader(X, y, batch_size=256, shuffle=True):
    t_x = torch.tensor(X, dtype=torch.float32)
    t_y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    ds  = TensorDataset(t_x, t_y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X_train_s, y_train, shuffle=True)
val_loader   = make_loader(X_val_s,   y_val,   shuffle=False)
test_loader  = make_loader(X_test_s,  y_test,  shuffle=False)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_model(model, train_loader, val_loader, epochs=150, lr=1e-3, patience=20):
    model = model.to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
    loss_fn = nn.MSELoss()
    best_val, best_ep, best_state = np.inf, 0, None

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_losses.append(loss_fn(model(xb), yb).item())
        val_loss = np.mean(val_losses)
        sched.step(val_loss)

        if val_loss < best_val:
            best_val, best_ep = val_loss, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep - best_ep >= patience:
            print(f"    Early stop at epoch {ep} (best={best_ep}, val_loss={best_val:.4f})")
            break

    model.load_state_dict(best_state)
    return model


def predict_loader(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(preds).squeeze()


n_feat = X_train_s.shape[1]
mlp = MLP(in_dim=n_feat, hidden=(256, 128, 64), dropout=0.3)
mlp = train_model(mlp, train_loader, val_loader, epochs=200, lr=1e-3, patience=25)

mlp_val_pred  = predict_loader(mlp, val_loader)
mlp_test_pred = predict_loader(mlp, test_loader)

print("  Validation:")
mlp_val_metrics  = regression_metrics(y_val,  mlp_val_pred,  "MLP-val")
mlp_val_sp       = top10_spearman(val_df,  mlp_val_pred,  "MLP-val")
print("  Test:")
mlp_test_metrics = regression_metrics(y_test, mlp_test_pred, "MLP-test")
mlp_test_sp      = top10_spearman(test_df, mlp_test_pred, "MLP-test")

# ─────────────────────────────────────────────────────────────────────────────
# 16.  Model C – Attention-based MLP (PyTorch)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Training Attention-MLP (PyTorch) …")


class AttentionMLP(nn.Module):
    """
    Treats each feature as a 'token', computes self-attention over features,
    then feeds the attended representation through an MLP head.
    """
    def __init__(self, in_dim, embed_dim=64, n_heads=4, mlp_hidden=(128, 64), dropout=0.2):
        super().__init__()
        # Linear embedding of each feature to embed_dim
        self.feature_embed = nn.Linear(1, embed_dim)
        # Self-attention over the in_dim 'tokens'
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        # Aggregation: mean pooling over tokens
        # MLP head
        layers = []
        prev = in_dim * embed_dim   # flattened after attention
        # Use a smaller head to avoid over-parameterisation
        for h in mlp_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, F)
        # Treat each feature as a token: (B, F, 1) → embed → (B, F, E)
        tokens = self.feature_embed(x.unsqueeze(-1))     # (B, F, E)
        attn_out, _ = self.attn(tokens, tokens, tokens)  # (B, F, E)
        attn_out = self.norm1(attn_out + tokens)          # residual
        flat = attn_out.flatten(start_dim=1)              # (B, F*E)
        return self.head(flat)


attn_mlp = AttentionMLP(
    in_dim=n_feat, embed_dim=32, n_heads=4,
    mlp_hidden=(256, 128), dropout=0.3
)
attn_mlp = train_model(attn_mlp, train_loader, val_loader, epochs=200, lr=5e-4, patience=25)

attn_val_pred  = predict_loader(attn_mlp, val_loader)
attn_test_pred = predict_loader(attn_mlp, test_loader)

print("  Validation:")
attn_val_metrics  = regression_metrics(y_val,  attn_val_pred,  "AttnMLP-val")
attn_val_sp       = top10_spearman(val_df,  attn_val_pred,  "AttnMLP-val")
print("  Test:")
attn_test_metrics = regression_metrics(y_test, attn_test_pred, "AttnMLP-test")
attn_test_sp      = top10_spearman(test_df, attn_test_pred, "AttnMLP-test")

# ─────────────────────────────────────────────────────────────────────────────
# 17.  Ensemble (simple average of all three)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Ensemble (mean of XGB + MLP + AttnMLP) …")

ens_val_pred  = (xgb_val_pred  + mlp_val_pred  + attn_val_pred)  / 3
ens_test_pred = (xgb_test_pred + mlp_test_pred + attn_test_pred) / 3

print("  Validation:")
ens_val_metrics  = regression_metrics(y_val,  ens_val_pred,  "Ensemble-val")
ens_val_sp       = top10_spearman(val_df,  ens_val_pred,  "Ensemble-val")
print("  Test:")
ens_test_metrics = regression_metrics(y_test, ens_test_pred, "Ensemble-test")
ens_test_sp      = top10_spearman(test_df, ens_test_pred, "Ensemble-test")

# ─────────────────────────────────────────────────────────────────────────────
# 18.  Top-10 MVPs per test season  (best model: ensemble)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Top-10 MVPs per test season (Ensemble):")
top10_results = get_top10_per_season(test_df, ens_test_pred, "Ensemble-test")
for season, rows in top10_results.items():
    print(f"\n  {season}:")
    for rank, r in enumerate(rows, 1):
        print(f"    {rank:2d}. {r['player']:<30s} {r['team']:<20s} "
              f"pos={r['primary_pos']:<3s}  pred={r['pred_score']:.1f}  actual={r[TARGET]:.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# 19.  Save outputs
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Saving outputs …")

# results.json
results = {
    "models": {
        "xgboost": {
            "val":  {**xgb_val_metrics,  "top10_spearman": xgb_val_sp},
            "test": {**xgb_test_metrics, "top10_spearman": xgb_test_sp},
        },
        "mlp": {
            "val":  {**mlp_val_metrics,  "top10_spearman": mlp_val_sp},
            "test": {**mlp_test_metrics, "top10_spearman": mlp_test_sp},
        },
        "attention_mlp": {
            "val":  {**attn_val_metrics, "top10_spearman": attn_val_sp},
            "test": {**attn_test_metrics,"top10_spearman": attn_test_sp},
        },
        "ensemble": {
            "val":  {**ens_val_metrics,  "top10_spearman": ens_val_sp},
            "test": {**ens_test_metrics, "top10_spearman": ens_test_sp},
        },
    },
    "feature_importance_xgboost": feat_imp_sorted,
    "top10_mvp_per_test_season": top10_results,
    "config": {
        "train_seasons": TRAIN_SEASONS,
        "val_seasons":   VAL_SEASONS,
        "test_seasons":  TEST_SEASONS,
        "min_minutes_threshold": 900,
        "features": FEATURES,
        "target": TARGET,
        "goal_weights": GOAL_WEIGHT,
        "assist_weight": AST_WEIGHT,
        "clean_sheet_weight": CS_WEIGHT,
    },
}

with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"  Saved: {OUT / 'results.json'}")

# Save MVP scores dataset
df[["player", "season", "team", "primary_pos", "min", "gls", "ast",
    "clean_sheets", "mvp_raw", "mvp_score"]].to_parquet(OUT / "mvp_scores.parquet", index=False)
print(f"  Saved: {OUT / 'mvp_scores.parquet'}")

# Save predictions on test set
test_df = test_df.copy()
test_df["xgb_pred"]   = xgb_test_pred
test_df["mlp_pred"]   = mlp_test_pred
test_df["attn_pred"]  = attn_test_pred
test_df["ens_pred"]   = ens_test_pred
test_df[["player", "season", "team", "primary_pos", TARGET,
         "xgb_pred", "mlp_pred", "attn_pred", "ens_pred"]].to_parquet(
    OUT / "test_predictions.parquet", index=False
)
print(f"  Saved: {OUT / 'test_predictions.parquet'}")

# Save models
xgb_model.save_model(str(OUT / "xgb_model.json"))
print(f"  Saved: {OUT / 'xgb_model.json'}")

torch.save(mlp.state_dict(),      OUT / "mlp_weights.pt")
torch.save(attn_mlp.state_dict(), OUT / "attn_mlp_weights.pt")
print(f"  Saved: {OUT / 'mlp_weights.pt'}  &  {OUT / 'attn_mlp_weights.pt'}")

import joblib
joblib.dump(scaler, OUT / "scaler.pkl")
print(f"  Saved: {OUT / 'scaler.pkl'}")

# ─────────────────────────────────────────────────────────────────────────────
# 20.  Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
header = f"{'Model':<20s} {'Val MAE':>8s} {'Val R²':>8s} {'Val Sp':>8s}   "
header += f"{'Test MAE':>9s} {'Test R²':>9s} {'Test Sp':>9s}"
print(header)
print("-" * 70)
rows_summary = [
    ("XGBoost",      xgb_val_metrics,  xgb_val_sp,  xgb_test_metrics,  xgb_test_sp),
    ("MLP",          mlp_val_metrics,  mlp_val_sp,  mlp_test_metrics,  mlp_test_sp),
    ("Attention-MLP",attn_val_metrics, attn_val_sp, attn_test_metrics, attn_test_sp),
    ("Ensemble",     ens_val_metrics,  ens_val_sp,  ens_test_metrics,  ens_test_sp),
]
for name, vm, vs, tm, ts in rows_summary:
    print(f"{name:<20s} {vm['mae']:>8.3f} {vm['r2']:>8.4f} {vs:>8.4f}   "
          f"{tm['mae']:>9.3f} {tm['r2']:>9.4f} {ts:>9.4f}")
print("=" * 70)
print("Done. All outputs saved to:", OUT)

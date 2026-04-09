"""
P3: EPL Relegation Prediction Model
Predicts which teams will be relegated at end of season (bottom 3).

Data splits:
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

# ── Sklearn ──────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
from sklearn.pipeline import Pipeline

# ── XGBoost ──────────────────────────────────────────────────────────────────
import xgboost as xgb

# ── Imbalance ────────────────────────────────────────────────────────────────
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ── PyTorch ──────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# =============================================================================
# Paths
# =============================================================================
ROOT = Path("C:/Users/xcv54/workspace/EPL project")
DATA = ROOT / "data" / "processed"
OUT  = ROOT / "models" / "p3_relegation"
OUT.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 1. Load raw data
# =============================================================================
print("Loading data …")
match_df   = pd.read_parquet(DATA / "match_results.parquet")
summary_df = pd.read_parquet(DATA / "team_season_summary.parquet")
player_df  = pd.read_parquet(DATA / "player_season_stats.parquet")

# Normalise season column name in player_df
player_df = player_df.rename(columns={"season": "Season"})

# =============================================================================
# 2. Build relegation target
# =============================================================================
print("Building relegation labels …")

def assign_relegation(df: pd.DataFrame) -> pd.DataFrame:
    """Mark the 3 teams with lowest points each season as relegated.
    Ties broken by goal_diff, then total_goals_for."""
    df = df.copy()
    df["rank"] = (
        df.groupby("Season")[["points", "goal_diff", "total_goals_for"]]
        .rank(method="min", ascending=True)
        .mean(axis=1)          # combined rank so we can sort
    )
    # Proper ranking: sort ascending by points, gd, gf → bottom 3 = rank ≤ 3
    df["pts_rank"] = df.groupby("Season")["points"].rank(
        method="first", ascending=True
    )
    df["gd_rank"] = df.groupby("Season")["goal_diff"].rank(
        method="first", ascending=True
    )
    df["sort_key"] = df["points"] * 10000 + df["goal_diff"] * 100 + df["total_goals_for"]
    df["final_rank"] = df.groupby("Season")["sort_key"].rank(
        method="first", ascending=True
    )
    df["relegated"] = (df["final_rank"] <= 3).astype(int)
    return df

summary_df = assign_relegation(summary_df)

print("Relegated teams per season (sample):")
rel_sample = summary_df[summary_df["relegated"] == 1][["Season","team","points","goal_diff"]].sort_values("Season")
print(rel_sample.head(12).to_string(index=False))
print(f"\nTotal relegated observations: {summary_df['relegated'].sum()} / {len(summary_df)}")

# =============================================================================
# 3. Feature Engineering
# =============================================================================
print("\nEngineering features …")

SEASONS_ORDERED = sorted(summary_df["Season"].unique())

# ── 3a. Cumulative standings at matchday N ───────────────────────────────────

def compute_cumulative_standings(match_df: pd.DataFrame) -> pd.DataFrame:
    """
    From match_results, compute cumulative points / GD for each team
    after exactly N matches played in the season (N = 10, 15, 20).
    """
    rows = []
    for season, sg in match_df.groupby("Season"):
        sg = sg.sort_values("MatchDate").reset_index(drop=True)
        # Track match count per team
        team_match_count: dict = {}
        team_stats: dict = {}

        for _, row in sg.iterrows():
            ht = row["HomeTeam"]
            at = row["AwayTeam"]
            for t in [ht, at]:
                if t not in team_match_count:
                    team_match_count[t] = 0
                    team_stats[t] = {"pts": 0, "gf": 0, "ga": 0}

            # Outcome
            hg, ag = row["FullTimeHomeGoals"], row["FullTimeAwayGoals"]
            if hg > ag:
                hp, ap = 3, 0
            elif hg == ag:
                hp, ap = 1, 1
            else:
                hp, ap = 0, 3

            team_stats[ht]["pts"] += hp; team_stats[ht]["gf"] += hg; team_stats[ht]["ga"] += ag
            team_stats[at]["pts"] += ap; team_stats[at]["gf"] += ag; team_stats[at]["ga"] += hg
            team_match_count[ht] += 1
            team_match_count[at] += 1

            # Snapshot at matchday 10, 15, 20 for each team
            for N in [10, 15, 20]:
                for t in [ht, at]:
                    if team_match_count[t] == N:
                        rows.append({
                            "Season": season, "team": t, "checkpoint": N,
                            "pts_at_N": team_stats[t]["pts"],
                            "gd_at_N": team_stats[t]["gf"] - team_stats[t]["ga"],
                        })

    return pd.DataFrame(rows)

cum_df = compute_cumulative_standings(match_df)
# Pivot to wide format
cum_wide = cum_df.pivot_table(
    index=["Season", "team"], columns="checkpoint",
    values=["pts_at_N", "gd_at_N"]
)
cum_wide.columns = [f"{v}_md{c}" for v, c in cum_wide.columns]
cum_wide = cum_wide.reset_index()

# ── 3b. Home/Away split from summary ────────────────────────────────────────
# home pts ratio
summary_df["home_pts"] = summary_df["home_wins"] * 3 + summary_df["home_draws"]
summary_df["away_pts"] = summary_df["away_wins"] * 3 + summary_df["away_draws"]
summary_df["home_away_ratio"] = (
    summary_df["home_pts"] / (summary_df["away_pts"] + 1e-6)
).clip(0, 10)

# shots on target ratio (proxy for quality)
# (derived from match_df below)

# ── 3c. Squad features from player_df ───────────────────────────────────────
def compute_squad_features(player_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (season, team), g in player_df.groupby(["Season", "team"]):
        # market value: only players with data
        mv_data = g[g["no_value_data"] == False]["market_value"]
        avg_mv  = mv_data.mean() if len(mv_data) > 0 else np.nan
        sum_mv  = mv_data.sum()  if len(mv_data) > 0 else np.nan

        # squad age
        avg_age = g["age"].mean()

        # squad depth: unique players who played ≥1 match
        depth = (g["mp"] >= 1).sum()

        rows.append({
            "Season": season, "team": team,
            "avg_market_value": avg_mv,
            "sum_market_value": sum_mv,
            "avg_age": avg_age,
            "squad_depth": depth,
        })
    return pd.DataFrame(rows)

squad_feats = compute_squad_features(player_df)

# ── 3d. Previous season finish & newly promoted flag ────────────────────────
def compute_prev_season_features(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Season, team): previous season finish rank + newly_promoted flag.
    """
    rows = []
    seasons = sorted(summary_df["Season"].unique())
    # Build rank per season
    season_rank = {}
    for s, sg in summary_df.groupby("Season"):
        sg = sg.sort_values("sort_key", ascending=False).reset_index(drop=True)
        sg["finish_rank"] = sg.index + 1          # 1 = champion
        for _, row in sg.iterrows():
            season_rank[(s, row["team"])] = row["finish_rank"]

    for i, season in enumerate(seasons):
        teams_this = summary_df[summary_df["Season"] == season]["team"].unique()
        if i == 0:
            prev_teams = set()
            prev_season = None
        else:
            prev_season = seasons[i - 1]
            prev_teams = set(summary_df[summary_df["Season"] == prev_season]["team"].unique())

        for team in teams_this:
            newly_promoted = int(team not in prev_teams)
            if prev_season and (prev_season, team) in season_rank:
                prev_rank = season_rank[(prev_season, team)]
            else:
                prev_rank = np.nan      # newly promoted, no prior rank

            rows.append({
                "Season": season, "team": team,
                "newly_promoted": newly_promoted,
                "prev_season_rank": prev_rank,
            })
    return pd.DataFrame(rows)

prev_feats = compute_prev_season_features(summary_df)

# ── 3e. Historical relegation frequency ─────────────────────────────────────
def compute_rel_history(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Season, team): fraction of past seasons where team was relegated.
    Expanding window — only uses data strictly before current season.
    """
    seasons = sorted(summary_df["Season"].unique())
    rows = []
    hist: dict = {}   # team -> list of relegation outcomes in past seasons

    for season in seasons:
        sg = summary_df[summary_df["Season"] == season]
        for _, row in sg.iterrows():
            team = row["team"]
            past = hist.get(team, [])
            rel_freq = np.mean(past) if past else 0.0
            rows.append({"Season": season, "team": team, "rel_freq_hist": rel_freq})

        # Update history AFTER recording features
        for _, row in sg.iterrows():
            hist.setdefault(row["team"], []).append(row["relegated"])

    return pd.DataFrame(rows)

rel_hist = compute_rel_history(summary_df)

# ── 3f. Simple ELO rating ────────────────────────────────────────────────────
def compute_elo(match_df: pd.DataFrame, k: int = 32, initial: int = 1500) -> dict:
    """
    Compute ELO ratings match-by-match.
    Returns dict: {(season, team): elo_at_season_start}
    """
    elo: dict = {}
    seasons = sorted(match_df["Season"].unique())
    season_start_elo: dict = {}

    for season in seasons:
        sg = match_df[match_df["Season"] == season].sort_values("MatchDate")
        # Record ELO at season start (before any game this season)
        for team in pd.concat([sg["HomeTeam"], sg["AwayTeam"]]).unique():
            season_start_elo[(season, team)] = elo.get(team, initial)

        for _, row in sg.iterrows():
            ht, at = row["HomeTeam"], row["AwayTeam"]
            r_h = elo.get(ht, initial)
            r_a = elo.get(at, initial)

            e_h = 1 / (1 + 10 ** ((r_a - r_h) / 400))
            e_a = 1 - e_h

            if row["FullTimeResult"] == "H":
                s_h, s_a = 1, 0
            elif row["FullTimeResult"] == "D":
                s_h, s_a = 0.5, 0.5
            else:
                s_h, s_a = 0, 1

            elo[ht] = r_h + k * (s_h - e_h)
            elo[at] = r_a + k * (s_a - e_a)

    return season_start_elo

print("  Computing ELO …")
elo_map = compute_elo(match_df)
elo_df = pd.DataFrame(
    [{"Season": s, "team": t, "elo_start": v} for (s, t), v in elo_map.items()]
)

# ── 3g. Shots on target ratio from match_df ──────────────────────────────────
def compute_shot_ratio(match_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, sg in match_df.groupby("Season"):
        teams = pd.concat([sg["HomeTeam"], sg["AwayTeam"]]).unique()
        for team in teams:
            h = sg[sg["HomeTeam"] == team]
            a = sg[sg["AwayTeam"] == team]
            sot_for  = h["HomeShotsOnTarget"].sum() + a["AwayShotsOnTarget"].sum()
            sot_ag   = h["AwayShotsOnTarget"].sum() + a["HomeShotsOnTarget"].sum()
            total    = sot_for + sot_ag + 1e-6
            rows.append({"Season": season, "team": team,
                         "sot_ratio": sot_for / total})
    return pd.DataFrame(rows)

sot_df = compute_shot_ratio(match_df)

# =============================================================================
# 4. Merge all features
# =============================================================================
print("Merging features …")
feat = summary_df[["Season","team","points","goal_diff",
                    "home_pts","away_pts","home_away_ratio",
                    "total_wins","total_draws","total_losses",
                    "total_goals_for","total_goals_against",
                    "relegated"]].copy()

feat = feat.merge(cum_wide,   on=["Season","team"], how="left")
feat = feat.merge(squad_feats, on=["Season","team"], how="left")
feat = feat.merge(prev_feats,  on=["Season","team"], how="left")
feat = feat.merge(rel_hist,    on=["Season","team"], how="left")
feat = feat.merge(elo_df,      on=["Season","team"], how="left")
feat = feat.merge(sot_df,      on=["Season","team"], how="left")

print(f"Feature matrix shape: {feat.shape}")
print(f"Missing values:\n{feat.isnull().sum()[feat.isnull().sum()>0]}")

# =============================================================================
# 5. Fill missing / encode
# =============================================================================
# For seasons/teams with no market value data (early seasons 2000-2003),
# we'll use the seasonal median (imputed per season)
for col in ["avg_market_value", "sum_market_value"]:
    feat[col] = feat.groupby("Season")[col].transform(
        lambda x: x.fillna(x.median())
    )
    feat[col] = feat[col].fillna(0)

# prev_season_rank: newly promoted teams get rank 21 (just outside top flight)
feat["prev_season_rank"] = feat["prev_season_rank"].fillna(21)

# Remaining numeric NaNs: fill with column median
num_cols = feat.select_dtypes(include=[np.number]).columns.tolist()
for c in num_cols:
    if c in ("relegated",):
        continue
    feat[c] = feat[c].fillna(feat[c].median())

# ── Feature list ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    # Season-end summary (would be available mid-season at MD38, but we keep full season for target)
    "home_pts", "away_pts", "home_away_ratio",
    "total_wins", "total_draws", "total_losses",
    "total_goals_for", "total_goals_against", "goal_diff",
    # Cumulative checkpoints
    "pts_at_N_md10", "gd_at_N_md10",
    "pts_at_N_md15", "gd_at_N_md15",
    "pts_at_N_md20", "gd_at_N_md20",
    # Squad features
    "avg_market_value", "sum_market_value", "avg_age", "squad_depth",
    # History
    "newly_promoted", "prev_season_rank", "rel_freq_hist",
    # ELO
    "elo_start",
    # Shot quality
    "sot_ratio",
]

# Verify all columns exist
for c in FEATURE_COLS:
    if c not in feat.columns:
        print(f"WARNING: column {c!r} not found, adding zeros")
        feat[c] = 0.0

print(f"\nUsing {len(FEATURE_COLS)} features")

# =============================================================================
# 6. Train / Val / Test split
# =============================================================================
TRAIN_SEASONS = [s for s in SEASONS_ORDERED if s <= "2020/21"]
VAL_SEASONS   = [s for s in SEASONS_ORDERED if "2021/22" <= s <= "2022/23"]
TEST_SEASONS  = [s for s in SEASONS_ORDERED if s >= "2023/24"]

train = feat[feat["Season"].isin(TRAIN_SEASONS)].reset_index(drop=True)
val   = feat[feat["Season"].isin(VAL_SEASONS)].reset_index(drop=True)
test  = feat[feat["Season"].isin(TEST_SEASONS)].reset_index(drop=True)

X_train = train[FEATURE_COLS].values.astype(np.float32)
y_train = train["relegated"].values.astype(np.float32)
X_val   = val[FEATURE_COLS].values.astype(np.float32)
y_val   = val["relegated"].values.astype(np.float32)
X_test  = test[FEATURE_COLS].values.astype(np.float32)
y_test  = test["relegated"].values.astype(np.float32)

print(f"\nSplit sizes  →  Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
print(f"Positive rate  →  Train: {y_train.mean():.3f}  Val: {y_val.mean():.3f}  Test: {y_test.mean():.3f}")

# =============================================================================
# 7. Evaluation helpers
# =============================================================================

def evaluate(y_true, y_pred_proba, y_pred_bin, label: str, df_meta: pd.DataFrame):
    """Compute metrics and per-season top-3 accuracy."""
    auc  = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else float("nan")
    f1   = f1_score(y_true, y_pred_bin, zero_division=0)
    prec = precision_score(y_true, y_pred_bin, zero_division=0)
    rec  = recall_score(y_true, y_pred_bin, zero_division=0)
    print(f"\n--- {label} ---")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(classification_report(y_true, y_pred_bin, target_names=["Safe","Relegated"]))

    # Per-season top-3 prediction accuracy
    df_meta = df_meta.copy()
    df_meta["prob"]     = y_pred_proba
    df_meta["pred_bin"] = y_pred_bin
    df_meta["actual"]   = y_true

    season_results = []
    correct_seasons = 0
    for season, sg in df_meta.groupby("Season"):
        top3_pred   = set(sg.nlargest(3, "prob")["team"])
        top3_actual = set(sg[sg["actual"] == 1]["team"])
        overlap     = len(top3_pred & top3_actual)
        correct     = int(top3_pred == top3_actual)
        correct_seasons += correct
        season_results.append({
            "season": season,
            "actual_relegated": sorted(top3_actual),
            "predicted_top3":   sorted(top3_pred),
            "overlap": overlap,
            "exact_match": correct,
        })
    top3_acc = correct_seasons / len(season_results)
    print(f"  Top-3 exact match accuracy: {correct_seasons}/{len(season_results)} = {top3_acc:.3f}")

    return {
        "auc_roc": round(auc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "top3_exact_match_accuracy": round(top3_acc, 4),
        "season_results": season_results,
    }


def threshold_top3(proba: np.ndarray, df_meta: pd.DataFrame) -> np.ndarray:
    """
    For each season, predict the 3 teams with highest probability as relegated.
    Returns binary predictions array aligned with df_meta index.
    """
    df_meta = df_meta.copy()
    df_meta["prob"] = proba
    df_meta["pred"] = 0
    for season, sg in df_meta.groupby("Season"):
        top3_idx = sg.nlargest(3, "prob").index
        df_meta.loc[top3_idx, "pred"] = 1
    return df_meta["pred"].values


# =============================================================================
# 8a. Model 1: Logistic Regression
# =============================================================================
print("\n" + "="*60)
print("MODEL 1: Logistic Regression")
print("="*60)

lr_pipe = ImbPipeline([
    ("smote", SMOTE(random_state=42, k_neighbors=4)),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        C=0.1,
        solver="lbfgs",
        random_state=42,
    )),
])

lr_pipe.fit(X_train, y_train)

lr_val_proba  = lr_pipe.predict_proba(X_val)[:, 1]
lr_val_bin    = threshold_top3(lr_val_proba, val[["Season","team"]])
lr_val_stats  = evaluate(y_val, lr_val_proba, lr_val_bin, "LR – Validation", val[["Season","team"]])

lr_test_proba = lr_pipe.predict_proba(X_test)[:, 1]
lr_test_bin   = threshold_top3(lr_test_proba, test[["Season","team"]])
lr_test_stats = evaluate(y_test, lr_test_proba, lr_test_bin, "LR – Test", test[["Season","team"]])

# Coefficients
lr_coef = dict(zip(FEATURE_COLS, lr_pipe.named_steps["clf"].coef_[0].tolist()))
print("\nTop positive coefficients (relegation risk):")
sorted_coef = sorted(lr_coef.items(), key=lambda x: x[1], reverse=True)
for k, v in sorted_coef[:8]:
    print(f"  {k:35s}: {v:+.4f}")

# =============================================================================
# 8b. Model 2: XGBoost
# =============================================================================
print("\n" + "="*60)
print("MODEL 2: XGBoost Classifier")
print("="*60)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
)

# Scale features for consistency (XGBoost doesn't strictly need it but uniform)
scaler_xgb = StandardScaler()
X_train_s  = scaler_xgb.fit_transform(X_train)
X_val_s    = scaler_xgb.transform(X_val)
X_test_s   = scaler_xgb.transform(X_test)

xgb_model.fit(
    X_train_s, y_train,
    eval_set=[(X_val_s, y_val)],
    verbose=False,
)

xgb_val_proba  = xgb_model.predict_proba(X_val_s)[:, 1]
xgb_val_bin    = threshold_top3(xgb_val_proba, val[["Season","team"]])
xgb_val_stats  = evaluate(y_val, xgb_val_proba, xgb_val_bin, "XGBoost – Validation", val[["Season","team"]])

xgb_test_proba = xgb_model.predict_proba(X_test_s)[:, 1]
xgb_test_bin   = threshold_top3(xgb_test_proba, test[["Season","team"]])
xgb_test_stats = evaluate(y_test, xgb_test_proba, xgb_test_bin, "XGBoost – Test", test[["Season","team"]])

# Feature importance
xgb_imp = dict(zip(FEATURE_COLS, xgb_model.feature_importances_.tolist()))
print("\nTop XGBoost feature importances:")
for k, v in sorted(xgb_imp.items(), key=lambda x: x[1], reverse=True)[:8]:
    print(f"  {k:35s}: {v:.4f}")

# =============================================================================
# 8c. Model 3: MLP (PyTorch)
# =============================================================================
print("\n" + "="*60)
print("MODEL 3: MLP (PyTorch)")
print("="*60)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class RelMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# SMOTE for MLP
smote = SMOTE(random_state=42, k_neighbors=4)
X_train_sm, y_train_sm = smote.fit_resample(X_train_s, y_train)

scaler_mlp = StandardScaler()
X_train_sm = scaler_mlp.fit_transform(X_train_sm)
X_val_mlp  = scaler_mlp.transform(X_val_s)
X_test_mlp = scaler_mlp.transform(X_test_s)

t_X_train = torch.tensor(X_train_sm, dtype=torch.float32).to(DEVICE)
t_y_train = torch.tensor(y_train_sm, dtype=torch.float32).to(DEVICE)
t_X_val   = torch.tensor(X_val_mlp,  dtype=torch.float32).to(DEVICE)
t_X_test  = torch.tensor(X_test_mlp, dtype=torch.float32).to(DEVICE)

train_ds = TensorDataset(t_X_train, t_y_train)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

mlp = RelMLP(in_dim=len(FEATURE_COLS)).to(DEVICE)

# Weighted BCE for safety even though we have SMOTE
pos_weight = torch.tensor([scale_pos_weight], dtype=torch.float32).to(DEVICE)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer  = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

EPOCHS = 200
best_val_auc  = 0.0
best_state    = None
patience      = 40
no_improve    = 0

for epoch in range(1, EPOCHS + 1):
    mlp.train()
    epoch_loss = 0.0
    for Xb, yb in train_dl:
        optimizer.zero_grad()
        logits = mlp(Xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()

    # Validation AUC
    mlp.eval()
    with torch.no_grad():
        val_logits = mlp(t_X_val).cpu().numpy()
        val_proba  = 1 / (1 + np.exp(-val_logits))   # sigmoid
    cur_auc = roc_auc_score(y_val, val_proba) if len(np.unique(y_val)) > 1 else 0.0
    if cur_auc > best_val_auc:
        best_val_auc = cur_auc
        best_state   = {k: v.clone() for k, v in mlp.state_dict().items()}
        no_improve   = 0
    else:
        no_improve += 1

    if epoch % 50 == 0:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={epoch_loss/len(train_dl):.4f}  val_auc={cur_auc:.4f}  best={best_val_auc:.4f}")

    if no_improve >= patience:
        print(f"  Early stopping at epoch {epoch}")
        break

# Load best weights
mlp.load_state_dict(best_state)
mlp.eval()

with torch.no_grad():
    mlp_val_logits  = mlp(t_X_val).cpu().numpy()
    mlp_test_logits = mlp(t_X_test).cpu().numpy()

mlp_val_proba  = 1 / (1 + np.exp(-mlp_val_logits))
mlp_test_proba = 1 / (1 + np.exp(-mlp_test_logits))

mlp_val_bin    = threshold_top3(mlp_val_proba, val[["Season","team"]])
mlp_test_bin   = threshold_top3(mlp_test_proba, test[["Season","team"]])

mlp_val_stats  = evaluate(y_val, mlp_val_proba, mlp_val_bin, "MLP – Validation", val[["Season","team"]])
mlp_test_stats = evaluate(y_test, mlp_test_proba, mlp_test_bin, "MLP – Test", test[["Season","team"]])

# =============================================================================
# 9. Ensemble (average probabilities)
# =============================================================================
print("\n" + "="*60)
print("ENSEMBLE: Average of LR + XGBoost + MLP")
print("="*60)

ens_val_proba  = (lr_val_proba + xgb_val_proba + mlp_val_proba) / 3
ens_val_bin    = threshold_top3(ens_val_proba, val[["Season","team"]])
ens_val_stats  = evaluate(y_val, ens_val_proba, ens_val_bin, "Ensemble – Validation", val[["Season","team"]])

ens_test_proba = (lr_test_proba + xgb_test_proba + mlp_test_proba) / 3
ens_test_bin   = threshold_top3(ens_test_proba, test[["Season","team"]])
ens_test_stats = evaluate(y_test, ens_test_proba, ens_test_bin, "Ensemble – Test", test[["Season","team"]])

# =============================================================================
# 10. Full dataset predictions (all seasons)
# =============================================================================
print("\nGenerating full-dataset predictions …")
X_all  = feat[FEATURE_COLS].values.astype(np.float32)
X_all_s = scaler_xgb.transform(X_all)

lr_all_proba  = lr_pipe.predict_proba(X_all)[:, 1]
xgb_all_proba = xgb_model.predict_proba(X_all_s)[:, 1]

X_all_mlp = scaler_mlp.transform(X_all_s)
t_X_all   = torch.tensor(X_all_mlp, dtype=torch.float32).to(DEVICE)
mlp.eval()
with torch.no_grad():
    mlp_all_logits = mlp(t_X_all).cpu().numpy()
mlp_all_proba = 1 / (1 + np.exp(-mlp_all_logits))

ens_all_proba = (lr_all_proba + xgb_all_proba + mlp_all_proba) / 3

feat_out = feat.copy()
feat_out["prob_lr"]  = lr_all_proba
feat_out["prob_xgb"] = xgb_all_proba
feat_out["prob_mlp"] = mlp_all_proba
feat_out["prob_ens"] = ens_all_proba

# Per-season top-3 predictions (ensemble)
feat_out["pred_relegated_ens"] = threshold_top3(ens_all_proba, feat_out[["Season","team"]])

# =============================================================================
# 11. Results summary
# =============================================================================
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

results = {
    "model_description": "P3 EPL Relegation Prediction",
    "features": FEATURE_COLS,
    "data_splits": {
        "train": {"seasons": TRAIN_SEASONS, "n_observations": int(len(train)), "n_relegated": int(y_train.sum())},
        "val":   {"seasons": VAL_SEASONS,   "n_observations": int(len(val)),   "n_relegated": int(y_val.sum())},
        "test":  {"seasons": TEST_SEASONS,  "n_observations": int(len(test)),  "n_relegated": int(y_test.sum())},
    },
    "models": {
        "logistic_regression": {
            "val":  lr_val_stats,
            "test": lr_test_stats,
            "top_coefficients": {k: round(v, 4) for k, v in sorted_coef[:10]},
        },
        "xgboost": {
            "val":  xgb_val_stats,
            "test": xgb_test_stats,
            "top_feature_importances": {k: round(v, 4) for k, v in
                                        sorted(xgb_imp.items(), key=lambda x: x[1], reverse=True)[:10]},
        },
        "mlp": {
            "val":  mlp_val_stats,
            "test": mlp_test_stats,
            "best_val_auc": round(best_val_auc, 4),
        },
        "ensemble": {
            "val":  ens_val_stats,
            "test": ens_test_stats,
        },
    },
}

# Per-season all-time predictions table
all_season_preds = []
for season, sg in feat_out.groupby("Season"):
    sg_sorted = sg.sort_values("prob_ens", ascending=False)
    for _, row in sg_sorted.iterrows():
        all_season_preds.append({
            "season": row["Season"],
            "team": row["team"],
            "actual_relegated": int(row["relegated"]),
            "prob_lr":  round(float(row["prob_lr"]),  4),
            "prob_xgb": round(float(row["prob_xgb"]), 4),
            "prob_mlp": round(float(row["prob_mlp"]), 4),
            "prob_ens": round(float(row["prob_ens"]), 4),
            "pred_relegated_ens": int(row["pred_relegated_ens"]),
        })

results["all_season_predictions"] = all_season_preds

# Save results.json
results_path = OUT / "results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {results_path}")

# Save predictions CSV
pred_csv = feat_out[["Season","team","relegated",
                       "prob_lr","prob_xgb","prob_mlp","prob_ens",
                       "pred_relegated_ens"]].sort_values(["Season","prob_ens"], ascending=[True, False])
pred_csv.to_csv(OUT / "predictions.csv", index=False)
print(f"Predictions CSV saved to: {OUT / 'predictions.csv'}")

# Save MLP model weights
torch.save(mlp.state_dict(), OUT / "mlp_weights.pt")
print(f"MLP weights saved to: {OUT / 'mlp_weights.pt'}")

# =============================================================================
# 12. Final comparison table
# =============================================================================
print("\n" + "="*60)
print("FINAL MODEL COMPARISON")
print("="*60)
header = f"{'Model':<20} {'Set':<8} {'AUC':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} {'Top3':>7}"
print(header)
print("-" * len(header))
for mname, mdata in results["models"].items():
    for split, sdata in [("Val", mdata["val"]), ("Test", mdata["test"])]:
        print(f"{mname:<20} {split:<8} "
              f"{sdata['auc_roc']:7.4f} "
              f"{sdata['f1']:7.4f} "
              f"{sdata['precision']:7.4f} "
              f"{sdata['recall']:7.4f} "
              f"{sdata['top3_exact_match_accuracy']:7.4f}")

print("\nDone.")

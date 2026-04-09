"""
P2: Goal Scoring Prediction Model — English Premier League
==========================================================
Sub-task A: Match total goals prediction (regression)
Sub-task B: Player goal probability (classification)

Models:
  - XGBoost Regressor         (match goals)
  - Poisson GLM via statsmodels (match goals)
  - MLP Regressor via sklearn  (match goals)
  - XGBoost Classifier        (player goal probability)
  - MLP Classifier via sklearn (player goal probability)

Time-based split:
  Train  : 2000/01 – 2020/21
  Val    : 2021/22 – 2022/23
  Test   : 2023/24 – 2024/25
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

# scikit-learn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, f1_score, precision_score, average_precision_score,
    classification_report,
)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import Pipeline

# XGBoost
import xgboost as xgb

# statsmodels for Poisson GLM
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE = Path("C:/Users/xcv54/workspace/EPL project")
DATA = BASE / "data" / "processed"
OUT  = BASE / "models" / "p2_goal_scoring"
OUT.mkdir(parents=True, exist_ok=True)

MATCH_FILE   = DATA / "match_results.parquet"
PLAYER_FILE  = DATA / "player_match_logs.parquet"
SEASON_FILE  = DATA / "player_season_stats.parquet"

# ─────────────────────────────────────────────
# Season split helpers
# ─────────────────────────────────────────────
TRAIN_SEASONS = [f"{y}/{str(y+1)[-2:]}" for y in range(2000, 2021)]
VAL_SEASONS   = ["2021/22", "2022/23"]
TEST_SEASONS  = ["2023/24", "2024/25"]

def season_split(df, season_col="Season"):
    train = df[df[season_col].isin(TRAIN_SEASONS)]
    val   = df[df[season_col].isin(VAL_SEASONS)]
    test  = df[df[season_col].isin(TEST_SEASONS)]
    return train, val, test

# ─────────────────────────────────────────────────────────────────────────────
# ╔══════════════════════════════════════════════════════╗
# ║  PART A – MATCH TOTAL GOALS PREDICTION (REGRESSION) ║
# ╚══════════════════════════════════════════════════════╝
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART A — Match Total Goals Regression")
print("=" * 70)

# ── Load & basic prep ──────────────────────────────────────────────────────
mr = pd.read_parquet(MATCH_FILE)
mr = mr.sort_values("MatchDate").reset_index(drop=True)
mr["TotalGoals"]    = mr["FullTimeHomeGoals"] + mr["FullTimeAwayGoals"]
mr["HomeGoals"]     = mr["FullTimeHomeGoals"]
mr["AwayGoals"]     = mr["FullTimeAwayGoals"]

# ── Rolling team strength features ─────────────────────────────────────────
print("Building match-level rolling features...")

# Build per-team rolling stats using historical rows only (avoid leakage)
# We iterate in chronological order and track a rolling dictionary.
WINDOWS = [5, 10, 20]

for team_role in ("Home", "Away"):
    for w in WINDOWS:
        mr[f"{team_role}Team_AvgScored_{w}"]   = np.nan
        mr[f"{team_role}Team_AvgConceded_{w}"] = np.nan

# track last N results per team — store (scored, conceded) tuples
from collections import defaultdict, deque

team_history: dict = defaultdict(lambda: deque(maxlen=max(WINDOWS)))

for idx, row in mr.iterrows():
    ht = row["HomeTeam"]
    at = row["AwayTeam"]
    hg = row["HomeGoals"]
    ag = row["AwayGoals"]

    for w in WINDOWS:
        # Home team rolling
        hist = list(team_history[ht])[-w:]
        if hist:
            mr.at[idx, f"HomeTeam_AvgScored_{w}"]   = np.mean([x[0] for x in hist])
            mr.at[idx, f"HomeTeam_AvgConceded_{w}"] = np.mean([x[1] for x in hist])
        # Away team rolling
        hist_a = list(team_history[at])[-w:]
        if hist_a:
            mr.at[idx, f"AwayTeam_AvgScored_{w}"]   = np.mean([x[0] for x in hist_a])
            mr.at[idx, f"AwayTeam_AvgConceded_{w}"] = np.mean([x[1] for x in hist_a])

    # update history after features computed (no leakage)
    team_history[ht].append((hg, ag))
    team_history[at].append((ag, hg))

# ── H2H average goals ───────────────────────────────────────────────────────
print("Building H2H features...")

h2h_history: dict = defaultdict(deque)

mr["H2H_AvgTotalGoals"] = np.nan
mr["H2H_Count"]         = 0

for idx, row in mr.iterrows():
    key = tuple(sorted([row["HomeTeam"], row["AwayTeam"]]))
    hist = list(h2h_history[key])
    if hist:
        mr.at[idx, "H2H_AvgTotalGoals"] = np.mean(hist)
        mr.at[idx, "H2H_Count"]         = len(hist)
    h2h_history[key].append(row["TotalGoals"])

# ── Season average goals trend ──────────────────────────────────────────────
season_avg = (
    mr.groupby("Season")["TotalGoals"]
    .mean()
    .rename("SeasonAvgGoals")
    .reset_index()
)
mr = mr.merge(season_avg, on="Season", how="left")

# ── Home/away scoring rates per season ─────────────────────────────────────
# We compute cumulative seasonal average up to but not including current match
mr = mr.sort_values(["Season", "MatchDate"]).reset_index(drop=True)
mr["SeasonMatchIdx"] = mr.groupby("Season").cumcount()

# expanding within season — shift(1) to avoid leakage
mr["SeasonCumAvgGoals"] = (
    mr.groupby("Season")["TotalGoals"]
    .transform(lambda x: x.expanding().mean().shift(1))
)

# ── Assemble match feature matrix ───────────────────────────────────────────
match_feature_cols = (
    [f"HomeTeam_AvgScored_{w}"   for w in WINDOWS] +
    [f"HomeTeam_AvgConceded_{w}" for w in WINDOWS] +
    [f"AwayTeam_AvgScored_{w}"   for w in WINDOWS] +
    [f"AwayTeam_AvgConceded_{w}" for w in WINDOWS] +
    ["H2H_AvgTotalGoals", "H2H_Count",
     "SeasonMatchIdx", "SeasonCumAvgGoals"]
)

TARGET_MATCH = "TotalGoals"

# Drop rows where ALL rolling features are NaN (first few rows per team)
mr_model = mr.dropna(subset=[f"HomeTeam_AvgScored_{WINDOWS[0]}",
                               f"AwayTeam_AvgScored_{WINDOWS[0]}"],
                      how="all").copy()

# Fill remaining NaNs with median
for c in match_feature_cols:
    mr_model[c] = mr_model[c].fillna(mr_model[c].median())

print(f"Match dataset: {len(mr_model)} rows after dropping cold-start rows")

# ── Split ───────────────────────────────────────────────────────────────────
train_m, val_m, test_m = season_split(mr_model, "Season")
print(f"  Train: {len(train_m)}  Val: {len(val_m)}  Test: {len(test_m)}")

X_train_m = train_m[match_feature_cols].values
y_train_m = train_m[TARGET_MATCH].values
X_val_m   = val_m[match_feature_cols].values
y_val_m   = val_m[TARGET_MATCH].values
X_test_m  = test_m[match_feature_cols].values
y_test_m  = test_m[TARGET_MATCH].values

X_trainval_m = np.vstack([X_train_m, X_val_m])
y_trainval_m = np.concatenate([y_train_m, y_val_m])

scaler_m = StandardScaler()
X_train_m_sc  = scaler_m.fit_transform(X_train_m)
X_val_m_sc    = scaler_m.transform(X_val_m)
X_test_m_sc   = scaler_m.transform(X_test_m)
X_trainval_m_sc = scaler_m.transform(X_trainval_m)

def match_metrics(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  {label:30s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}

results: dict = {"match_goals": {}, "player_goals": {}}

# ── Model A1: XGBoost Regressor ─────────────────────────────────────────────
print("\n[A1] XGBoost Regressor — tuning on val set")

xgb_reg = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    objective="reg:squarederror",
    eval_metric="mae",
    early_stopping_rounds=30,
    random_state=42,
    verbosity=0,
)
xgb_reg.fit(
    X_train_m, y_train_m,
    eval_set=[(X_val_m, y_val_m)],
    verbose=False,
)

pred_val_xgb  = xgb_reg.predict(X_val_m)
pred_test_xgb = xgb_reg.predict(X_test_m)

print("  Validation:")
results["match_goals"]["xgboost_val"]  = match_metrics(y_val_m,  pred_val_xgb,  "XGBoost val")
print("  Test:")
results["match_goals"]["xgboost_test"] = match_metrics(y_test_m, pred_test_xgb, "XGBoost test")

# Feature importance
feat_imp = pd.DataFrame({
    "feature": match_feature_cols,
    "importance": xgb_reg.feature_importances_,
}).sort_values("importance", ascending=False)
feat_imp.to_csv(OUT / "match_xgb_feature_importance.csv", index=False)
print(f"  Top 5 features: {feat_imp['feature'].head(5).tolist()}")

# ── Model A2: Poisson GLM (statsmodels) ─────────────────────────────────────
print("\n[A2] Poisson GLM")

# Fit on train only, evaluate on val & test
# Add small constant to target to avoid log(0) issues (Poisson handles 0 natively)
train_poisson_df = train_m[match_feature_cols + [TARGET_MATCH]].copy()
val_poisson_df   = val_m[match_feature_cols + [TARGET_MATCH]].copy()
test_poisson_df  = test_m[match_feature_cols + [TARGET_MATCH]].copy()

# Use scaled arrays via sm.GLM
X_train_poi = sm.add_constant(X_train_m_sc)
X_val_poi   = sm.add_constant(X_val_m_sc)
X_test_poi  = sm.add_constant(X_test_m_sc)

poisson_model = sm.GLM(
    y_train_m, X_train_poi,
    family=sm.families.Poisson()
).fit(disp=False)

pred_val_poi  = poisson_model.predict(X_val_poi)
pred_test_poi = poisson_model.predict(X_test_poi)

print("  Validation:")
results["match_goals"]["poisson_val"]  = match_metrics(y_val_m,  pred_val_poi,  "Poisson val")
print("  Test:")
results["match_goals"]["poisson_test"] = match_metrics(y_test_m, pred_test_poi, "Poisson test")

# Save Poisson summary
with open(OUT / "poisson_summary.txt", "w") as f:
    f.write(str(poisson_model.summary()))

# ── Model A3: MLP Regressor ─────────────────────────────────────────────────
print("\n[A3] MLP Regressor (sklearn)")

mlp_reg = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    solver="adam",
    learning_rate_init=1e-3,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
    batch_size=256,
)
mlp_reg.fit(X_train_m_sc, y_train_m)

pred_val_mlp_r  = mlp_reg.predict(X_val_m_sc)
pred_test_mlp_r = mlp_reg.predict(X_test_m_sc)

print("  Validation:")
results["match_goals"]["mlp_val"]  = match_metrics(y_val_m,  pred_val_mlp_r,  "MLP val")
print("  Test:")
results["match_goals"]["mlp_test"] = match_metrics(y_test_m, pred_test_mlp_r, "MLP test")

# ── Summary comparison (match goals) ────────────────────────────────────────
print("\n--- Match Goals Summary (Test set) ---")
models_test_preds = {
    "XGBoost":  pred_test_xgb,
    "Poisson":  pred_test_poi,
    "MLP":      pred_test_mlp_r,
}
best_model_name = None
best_mae = float("inf")
for name, preds in models_test_preds.items():
    m = match_metrics(y_test_m, preds, name)
    if m["mae"] < best_mae:
        best_mae = m["mae"]
        best_model_name = name

print(f"  Best match model (by MAE): {best_model_name} — MAE={best_mae:.4f}")
results["match_goals"]["best_model"] = best_model_name

# ─────────────────────────────────────────────────────────────────────────────
# ╔══════════════════════════════════════════════════════════╗
# ║  PART B – PLAYER GOAL PROBABILITY (CLASSIFICATION)      ║
# ╚══════════════════════════════════════════════════════════╝
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART B — Player Goal Probability Classification")
print("=" * 70)

# ── Load player match logs ──────────────────────────────────────────────────
ml = pd.read_parquet(PLAYER_FILE)
ps = pd.read_parquet(SEASON_FILE)

# Filter to players who actually played (not squad non-participant)
ml = ml[ml["pos"] != "On matchday squad, but did not play"].copy()
ml = ml[ml["min"].notna() & (ml["min"] > 0)].copy()

# Ensure date sorted
ml = ml.sort_values(["player", "date"]).reset_index(drop=True)

# Binary target
ml["scored"] = (ml["gls"] > 0).astype(int)

print(f"Player match log rows after filtering: {len(ml)}")
print(f"  Scored: {ml['scored'].sum()} ({100*ml['scored'].mean():.2f}%)")

# ── Position encoding ──────────────────────────────────────────────────────
# Consolidate positions
POS_MAP = {
    "FW": 4, "AM": 3, "LW": 3, "RW": 3,
    "CM": 2, "LM": 2, "RM": 2,
    "DM": 1, "CB": 0, "LB": 0, "RB": 0, "GK": -1,
    "None": 1, None: 1,
}
ml["pos_score"] = ml["pos"].map(POS_MAP).fillna(1)  # default = mid

# ── Merge market value from player_season_stats ─────────────────────────────
# Use per-season market value
ps_mv = ps[["player", "season", "market_value", "position"]].dropna(subset=["player", "season"])
ps_mv = ps_mv.rename(columns={"season": "season", "market_value": "mv_season"})
ps_mv["position_tm"] = ps_mv["position"].fillna("Unknown")

ml = ml.merge(ps_mv[["player", "season", "mv_season", "position_tm"]],
              on=["player", "season"], how="left")
ml["mv_season"] = ml["mv_season"].fillna(0)

# Log-transform market value
ml["log_mv"] = np.log1p(ml["mv_season"])

# ── Rolling player goal rate ────────────────────────────────────────────────
print("Computing player rolling goal rates...")

PLAYER_WINDOWS = [5, 10]

for w in PLAYER_WINDOWS:
    ml[f"player_goal_rate_{w}"] = np.nan
    ml[f"player_min_avg_{w}"]   = np.nan

# Group by player and compute rolling stats with shift(1) to avoid leakage
ml = ml.sort_values(["player", "date"]).reset_index(drop=True)

for w in PLAYER_WINDOWS:
    grp = ml.groupby("player")
    ml[f"player_goal_rate_{w}"] = grp["gls"].transform(
        lambda x: x.shift(1).rolling(w, min_periods=1).mean()
    )
    ml[f"player_min_avg_{w}"] = grp["min"].transform(
        lambda x: x.shift(1).rolling(w, min_periods=1).mean()
    )

# Season cumulative goal rate (within season, shifted)
ml["player_season_goal_rate"] = ml.groupby(["player", "season"])["gls"].transform(
    lambda x: x.shift(1).expanding().mean()
)

# ── Opponent defensive strength ─────────────────────────────────────────────
print("Computing opponent defensive strength...")

# From match_results, build opponent avg goals conceded (rolling)
# We use team_history already built for match model
# Simpler: compute season-level avg goals conceded per team from match_results
opp_conceded = pd.concat([
    mr[["Season", "AwayTeam", "HomeGoals"]].rename(
        columns={"AwayTeam": "team", "HomeGoals": "goals_conceded"}),
    mr[["Season", "HomeTeam", "AwayGoals"]].rename(
        columns={"HomeTeam": "team", "AwayGoals": "goals_conceded"}),
])
opp_season_def = (
    opp_conceded.groupby(["Season", "team"])["goals_conceded"]
    .mean().rename("opp_avg_conceded").reset_index()
)
opp_season_def.columns = ["season", "opponent", "opp_avg_conceded"]

ml = ml.merge(opp_season_def, on=["season", "opponent"], how="left")
ml["opp_avg_conceded"] = ml["opp_avg_conceded"].fillna(ml["opp_avg_conceded"].median())

# ── Home/Away flag ────────────────────────────────────────────────────────
ml["is_home"] = (ml["venue"] == "Home").astype(int)

# ── Assemble player feature matrix ─────────────────────────────────────────
player_feature_cols = (
    [f"player_goal_rate_{w}" for w in PLAYER_WINDOWS] +
    [f"player_min_avg_{w}"   for w in PLAYER_WINDOWS] +
    ["player_season_goal_rate",
     "pos_score", "log_mv",
     "opp_avg_conceded", "is_home", "min"]
)

TARGET_PLAYER = "scored"

ml_model = ml.copy()

# Fill NaN rolling stats with 0 for early matches (cold start)
for c in player_feature_cols:
    ml_model[c] = ml_model[c].fillna(0)

print(f"Player dataset: {len(ml_model)} rows")

# ── Split ─────────────────────────────────────────────────────────────────
train_p, val_p, test_p = season_split(ml_model, "season")
print(f"  Train: {len(train_p)}  Val: {len(val_p)}  Test: {len(test_p)}")

X_train_p = train_p[player_feature_cols].values
y_train_p = train_p[TARGET_PLAYER].values
X_val_p   = val_p[player_feature_cols].values
y_val_p   = val_p[TARGET_PLAYER].values
X_test_p  = test_p[player_feature_cols].values
y_test_p  = test_p[TARGET_PLAYER].values

X_trainval_p = np.vstack([X_train_p, X_val_p])
y_trainval_p = np.concatenate([y_train_p, y_val_p])

scaler_p = StandardScaler()
X_train_p_sc    = scaler_p.fit_transform(X_train_p)
X_val_p_sc      = scaler_p.transform(X_val_p)
X_test_p_sc     = scaler_p.transform(X_test_p)
X_trainval_p_sc = scaler_p.transform(X_trainval_p)

def player_metrics(y_true, y_prob, y_pred=None, label="", k=500):
    auc = roc_auc_score(y_true, y_prob)
    if y_pred is None:
        y_pred = (y_prob >= 0.5).astype(int)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    ap  = average_precision_score(y_true, y_prob)

    # precision@k  — top-k by predicted probability
    top_k_idx = np.argsort(y_prob)[::-1][:k]
    prec_k = y_true[top_k_idx].mean()

    print(f"  {label:35s}  AUC={auc:.4f}  F1={f1:.4f}  "
          f"AP={ap:.4f}  Prec@{k}={prec_k:.4f}")
    return {
        "auc": round(auc, 4), "f1": round(f1, 4),
        "avg_precision": round(ap, 4), f"precision_at_{k}": round(prec_k, 4),
    }

# ── Class weight for imbalanced data ───────────────────────────────────────
pos_rate = y_train_p.mean()
neg_rate = 1 - pos_rate
scale_pos = neg_rate / pos_rate  # ~9x

print(f"\n  Class imbalance: {pos_rate:.3%} positive — scale_pos_weight={scale_pos:.1f}")

# ── Model B1: XGBoost Classifier ────────────────────────────────────────────
print("\n[B1] XGBoost Classifier")

xgb_clf = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    scale_pos_weight=scale_pos,
    objective="binary:logistic",
    eval_metric="auc",
    early_stopping_rounds=30,
    use_label_encoder=False,
    random_state=42,
    verbosity=0,
)
xgb_clf.fit(
    X_train_p, y_train_p,
    eval_set=[(X_val_p, y_val_p)],
    verbose=False,
)

prob_val_xgb  = xgb_clf.predict_proba(X_val_p)[:, 1]
prob_test_xgb = xgb_clf.predict_proba(X_test_p)[:, 1]
pred_val_xgb_c  = xgb_clf.predict(X_val_p)
pred_test_xgb_c = xgb_clf.predict(X_test_p)

print("  Validation:")
results["player_goals"]["xgboost_val"]  = player_metrics(y_val_p, prob_val_xgb, pred_val_xgb_c, "XGBoost val")
print("  Test:")
results["player_goals"]["xgboost_test"] = player_metrics(y_test_p, prob_test_xgb, pred_test_xgb_c, "XGBoost test")

# Feature importance
feat_imp_p = pd.DataFrame({
    "feature": player_feature_cols,
    "importance": xgb_clf.feature_importances_,
}).sort_values("importance", ascending=False)
feat_imp_p.to_csv(OUT / "player_xgb_feature_importance.csv", index=False)
print(f"  Top 5 features: {feat_imp_p['feature'].head(5).tolist()}")

# ── Model B2: MLP Classifier ────────────────────────────────────────────────
print("\n[B2] MLP Classifier (sklearn)")

mlp_clf = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    solver="adam",
    learning_rate_init=1e-3,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
    batch_size=512,
)
mlp_clf.fit(X_train_p_sc, y_train_p)

prob_val_mlp_c  = mlp_clf.predict_proba(X_val_p_sc)[:, 1]
prob_test_mlp_c = mlp_clf.predict_proba(X_test_p_sc)[:, 1]
pred_val_mlp_c  = mlp_clf.predict(X_val_p_sc)
pred_test_mlp_c = mlp_clf.predict(X_test_p_sc)

print("  Validation:")
results["player_goals"]["mlp_val"]  = player_metrics(y_val_p, prob_val_mlp_c, pred_val_mlp_c, "MLP val")
print("  Test:")
results["player_goals"]["mlp_test"] = player_metrics(y_test_p, prob_test_mlp_c, pred_test_mlp_c, "MLP test")

# ── Summary comparison (player goals) ───────────────────────────────────────
print("\n--- Player Goal Probability Summary (Test set) ---")
player_models_test = {
    "XGBoost": (prob_test_xgb, pred_test_xgb_c),
    "MLP":     (prob_test_mlp_c, pred_test_mlp_c),
}
best_player_model = None
best_auc = 0.0
for name, (prob, pred) in player_models_test.items():
    m = player_metrics(y_test_p, prob, pred, name)
    if m["auc"] > best_auc:
        best_auc = m["auc"]
        best_player_model = name

print(f"  Best player model (by AUC): {best_player_model} — AUC={best_auc:.4f}")
results["player_goals"]["best_model"] = best_player_model

# ─────────────────────────────────────────────────────────────────────────────
# Save predictions
# ─────────────────────────────────────────────────────────────────────────────
print("\nSaving predictions...")

# Match goals predictions
match_pred_df = test_m[["Season", "MatchDate", "HomeTeam", "AwayTeam",
                          "FullTimeHomeGoals", "FullTimeAwayGoals", "TotalGoals"]].copy()
match_pred_df["pred_xgboost"] = np.round(pred_test_xgb, 2)
match_pred_df["pred_poisson"] = np.round(pred_test_poi, 2)
match_pred_df["pred_mlp"]     = np.round(pred_test_mlp_r, 2)
match_pred_df.to_csv(OUT / "match_goal_predictions.csv", index=False)

# Player goal predictions
player_pred_df = test_p[["season", "date", "player", "squad", "opponent",
                           "venue", "pos", "min", "gls", "scored"]].copy()
player_pred_df["prob_xgboost"] = np.round(prob_test_xgb, 4)
player_pred_df["prob_mlp"]     = np.round(prob_test_mlp_c, 4)
player_pred_df.to_csv(OUT / "player_goal_predictions.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# Save final results.json
# ─────────────────────────────────────────────────────────────────────────────
results["meta"] = {
    "train_seasons": f"{TRAIN_SEASONS[0]} – {TRAIN_SEASONS[-1]}",
    "val_seasons":   f"{VAL_SEASONS[0]} – {VAL_SEASONS[-1]}",
    "test_seasons":  f"{TEST_SEASONS[0]} – {TEST_SEASONS[-1]}",
    "match_features":  match_feature_cols,
    "player_features": player_feature_cols,
    "match_train_rows":  int(len(train_m)),
    "match_val_rows":    int(len(val_m)),
    "match_test_rows":   int(len(test_m)),
    "player_train_rows": int(len(train_p)),
    "player_val_rows":   int(len(val_p)),
    "player_test_rows":  int(len(test_p)),
    "player_positive_rate": float(round(pos_rate, 4)),
}

with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("ALL DONE")
print(f"  Results  → {OUT / 'results.json'}")
print(f"  Match predictions   → {OUT / 'match_goal_predictions.csv'}")
print(f"  Player predictions  → {OUT / 'player_goal_predictions.csv'}")
print(f"  Match XGB feat imp  → {OUT / 'match_xgb_feature_importance.csv'}")
print(f"  Player XGB feat imp → {OUT / 'player_xgb_feature_importance.csv'}")
print(f"  Poisson summary     → {OUT / 'poisson_summary.txt'}")
print("=" * 70)

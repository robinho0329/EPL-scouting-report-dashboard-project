"""
P3: Relegation Prediction Pipeline
===================================
Predicts whether a team will be relegated (bottom 3 in final standings).

Models: Logistic Regression, XGBoost, Random Forest, MLP
Split:  Train 2000-2021 | Val 2021-2023 | Test 2023-2025

Two prediction modes:
  - Full season: uses all 38 matchdays of aggregated data
  - Mid-season: uses data up to matchday 19 only
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report, confusion_matrix
)
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import joblib

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────
PROJECT  = Path(r"C:/Users/xcv54/workspace/EPL project")
DATA_DIR = PROJECT / "data"
OUT_DIR  = PROJECT / "models" / "p3_relegation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── season helpers ───────────────────────────────────────────────────────
def season_start_year(season_str):
    """'2000/01' -> 2000"""
    return int(season_str.split("/")[0])

def get_split(season_str):
    y = season_start_year(season_str)
    if y <= 2020:
        return "train"
    elif y <= 2022:
        return "val"
    else:
        return "test"

# ══════════════════════════════════════════════════════════════════════════
# 1.  BUILD TEAM-SEASON FEATURES + RELEGATION LABEL
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("P3  RELEGATION PREDICTION – Loading data …")
print("=" * 70)

mr = pd.read_parquet(DATA_DIR / "processed" / "match_results.parquet")
mf = pd.read_parquet(DATA_DIR / "features"  / "match_features.parquet")
ts = pd.read_parquet(DATA_DIR / "processed" / "team_season_summary.parquet")
pf = pd.read_parquet(DATA_DIR / "features"  / "player_features.parquet")

print(f"  match_results  : {mr.shape}")
print(f"  match_features : {mf.shape}")
print(f"  team_season    : {ts.shape}")
print(f"  player_features: {pf.shape}")

# ── 1a. Derive relegation label from team_season_summary ────────────────
ts["season_year"] = ts["Season"].apply(season_start_year)

# Bottom 3 teams by points each season = relegated
def mark_relegated(group):
    group = group.sort_values("points", ascending=True)
    group["relegated"] = 0
    group.iloc[:3, group.columns.get_loc("relegated")] = 1
    return group

ts = ts.groupby("Season", group_keys=False).apply(mark_relegated)
print(f"\nRelegation label distribution:\n{ts['relegated'].value_counts()}")
print(f"  Fraction relegated: {ts['relegated'].mean():.3f}")

# ── 1b. Build match-level team rows (one row per team per match) ────────
# We need cumulative stats per team per season at each matchday
print("\nBuilding per-team per-match cumulative features …")

# Assign matchweek numbers within each season (by date order)
mr = mr.sort_values(["Season", "MatchDate"]).reset_index(drop=True)
mr["matchweek"] = mr.groupby("Season").cumcount() // 10 + 1  # approx, 10 games per matchday

# Better: use the matchweek from match_features if available
if "matchweek" in mf.columns:
    mf_mw = mf[["Season", "MatchDate", "HomeTeam", "AwayTeam", "matchweek"]].copy()
    mr = mr.drop(columns=["matchweek"], errors="ignore")
    mr = mr.merge(mf_mw, on=["Season", "MatchDate", "HomeTeam", "AwayTeam"], how="left")

# Build team-match rows: one row per team per match
rows = []
for _, m in mr.iterrows():
    season = m["Season"]
    mw = m.get("matchweek", np.nan)
    # Home team row
    home_pts = 3 if m["FullTimeResult"] == "H" else (1 if m["FullTimeResult"] == "D" else 0)
    rows.append({
        "Season": season, "team": m["HomeTeam"], "matchweek": mw,
        "goals_for": m["FullTimeHomeGoals"], "goals_against": m["FullTimeAwayGoals"],
        "points": home_pts,
        "win": int(m["FullTimeResult"] == "H"),
        "draw": int(m["FullTimeResult"] == "D"),
        "loss": int(m["FullTimeResult"] == "A"),
        "shots": m["HomeShots"], "sot": m["HomeShotsOnTarget"],
        "corners": m["HomeCorners"], "fouls": m["HomeFouls"],
        "yellows": m["HomeYellowCards"], "reds": m["HomeRedCards"],
    })
    # Away team row
    away_pts = 3 if m["FullTimeResult"] == "A" else (1 if m["FullTimeResult"] == "D" else 0)
    rows.append({
        "Season": season, "team": m["AwayTeam"], "matchweek": mw,
        "goals_for": m["FullTimeAwayGoals"], "goals_against": m["FullTimeHomeGoals"],
        "points": away_pts,
        "win": int(m["FullTimeResult"] == "A"),
        "draw": int(m["FullTimeResult"] == "D"),
        "loss": int(m["FullTimeResult"] == "H"),
        "shots": m["AwayShots"], "sot": m["AwayShotsOnTarget"],
        "corners": m["AwayCorners"], "fouls": m["AwayFouls"],
        "yellows": m["AwayYellowCards"], "reds": m["AwayRedCards"],
    })

team_matches = pd.DataFrame(rows)
team_matches = team_matches.sort_values(["Season", "team", "matchweek"]).reset_index(drop=True)

# Cumulative aggregation per team-season
cum_cols = ["goals_for", "goals_against", "points", "win", "draw", "loss",
            "shots", "sot", "corners", "fouls", "yellows", "reds"]

for col in cum_cols:
    team_matches[f"cum_{col}"] = team_matches.groupby(["Season", "team"])[col].cumsum()

team_matches["cum_gd"] = team_matches["cum_goals_for"] - team_matches["cum_goals_against"]
team_matches["cum_games"] = team_matches.groupby(["Season", "team"]).cumcount() + 1
team_matches["cum_ppg"] = team_matches["cum_points"] / team_matches["cum_games"]
team_matches["cum_win_rate"] = team_matches["cum_win"] / team_matches["cum_games"]

# Rolling form (last 5 games)
team_matches["form_5"] = (
    team_matches.groupby(["Season", "team"])["points"]
    .transform(lambda x: x.rolling(5, min_periods=1).mean())
)

print(f"  Team-match rows: {team_matches.shape[0]}")

# ── 1c. Build full-season and mid-season aggregates per team-season ─────
def aggregate_at_matchday(df, max_matchweek=None, suffix=""):
    """Aggregate team-season stats up to a given matchweek."""
    if max_matchweek is not None:
        df = df[df["matchweek"] <= max_matchweek]

    # Take the last row per team-season (which has cumulative totals)
    agg = df.sort_values("matchweek").groupby(["Season", "team"]).last().reset_index()

    features = {
        "Season": agg["Season"],
        "team": agg["team"],
        f"points{suffix}": agg["cum_points"],
        f"goal_diff{suffix}": agg["cum_gd"],
        f"goals_for{suffix}": agg["cum_goals_for"],
        f"goals_against{suffix}": agg["cum_goals_against"],
        f"wins{suffix}": agg["cum_win"],
        f"draws{suffix}": agg["cum_draw"],
        f"losses{suffix}": agg["cum_loss"],
        f"ppg{suffix}": agg["cum_ppg"],
        f"win_rate{suffix}": agg["cum_win_rate"],
        f"shots{suffix}": agg["cum_shots"],
        f"sot{suffix}": agg["cum_sot"],
        f"corners{suffix}": agg["cum_corners"],
        f"fouls{suffix}": agg["cum_fouls"],
        f"yellows{suffix}": agg["cum_yellows"],
        f"reds{suffix}": agg["cum_reds"],
        f"form_5{suffix}": agg["form_5"],
        f"games_played{suffix}": agg["cum_games"],
    }
    return pd.DataFrame(features)

full_season = aggregate_at_matchday(team_matches, max_matchweek=None, suffix="")
mid_season  = aggregate_at_matchday(team_matches, max_matchweek=19, suffix="")

print(f"  Full-season aggregates: {full_season.shape}")
print(f"  Mid-season aggregates : {mid_season.shape}")

# ── 1d. Add ELO ratings from match_features ────────────────────────────
# Average ELO per team per season
elo_home = mf[["Season", "HomeTeam", "home_elo_pre"]].rename(
    columns={"HomeTeam": "team", "home_elo_pre": "elo"})
elo_away = mf[["Season", "AwayTeam", "away_elo_pre"]].rename(
    columns={"AwayTeam": "team", "away_elo_pre": "elo"})
elo_all = pd.concat([elo_home, elo_away], ignore_index=True)
elo_avg = elo_all.groupby(["Season", "team"])["elo"].mean().reset_index().rename(
    columns={"elo": "avg_elo"})

# Last ELO of the season
elo_last = elo_all.groupby(["Season", "team"])["elo"].last().reset_index().rename(
    columns={"elo": "last_elo"})

# ── 1e. Player-level team aggregates ───────────────────────────────────
print("Building player-level team aggregates …")
pf_agg = pf.groupby(["season", "team"]).agg(
    squad_size=("player", "count"),
    avg_age=("age_used", "mean"),
    avg_market_value=("market_value", "mean"),
    total_market_value=("market_value", "sum"),
    avg_minutes=("min", "mean"),
    avg_goals_p90=("goals_p90", "mean"),
    avg_assists_p90=("assists_p90", "mean"),
    avg_gc_p90=("goal_contributions_p90", "mean"),
    avg_epl_experience=("epl_experience", "mean"),
    total_minutes=("min", "sum"),
).reset_index().rename(columns={"season": "Season"})

print(f"  Player aggregates: {pf_agg.shape}")

# ── 1f. Promoted flag ──────────────────────────────────────────────────
# A team is promoted if it wasn't in the league the previous season
all_seasons = sorted(ts["Season"].unique())
promoted_rows = []
for i, s in enumerate(all_seasons):
    teams_this = set(ts[ts["Season"] == s]["team"])
    if i > 0:
        teams_prev = set(ts[ts["Season"] == all_seasons[i - 1]]["team"])
        for t in teams_this:
            promoted_rows.append({"Season": s, "team": t, "promoted": int(t not in teams_prev)})
    else:
        for t in teams_this:
            promoted_rows.append({"Season": s, "team": t, "promoted": 0})

promoted_df = pd.DataFrame(promoted_rows)

# ── 1g. Merge everything ──────────────────────────────────────────────
def build_dataset(season_agg, label="full"):
    df = season_agg.merge(elo_avg, on=["Season", "team"], how="left")
    df = df.merge(elo_last, on=["Season", "team"], how="left")
    df = df.merge(pf_agg, on=["Season", "team"], how="left")
    df = df.merge(promoted_df, on=["Season", "team"], how="left")
    df = df.merge(ts[["Season", "team", "relegated"]], on=["Season", "team"], how="left")
    df["split"] = df["Season"].apply(get_split)
    print(f"  [{label}] Dataset shape: {df.shape}  |  "
          f"train={len(df[df['split']=='train'])}  "
          f"val={len(df[df['split']=='val'])}  "
          f"test={len(df[df['split']=='test'])}")
    return df

print("\nMerging features …")
df_full = build_dataset(full_season, "full-season")
df_mid  = build_dataset(mid_season,  "mid-season")

# ══════════════════════════════════════════════════════════════════════════
# 2.  FEATURE SELECTION & PREPARATION
# ══════════════════════════════════════════════════════════════════════════

id_cols = ["Season", "team", "split", "relegated"]

def prepare_splits(df, label=""):
    feature_cols = [c for c in df.columns if c not in id_cols]

    # Drop columns with >50% missing
    missing_frac = df[feature_cols].isnull().mean()
    keep = missing_frac[missing_frac < 0.5].index.tolist()
    print(f"  [{label}] Keeping {len(keep)}/{len(feature_cols)} features (dropped high-missing)")

    X = df[keep].copy()
    y = df["relegated"].copy()
    split = df["split"].copy()

    # Fill remaining NaN with median from training set
    train_mask = split == "train"
    medians = X[train_mask].median()
    X = X.fillna(medians)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_mask])
    X_val   = scaler.transform(X[split == "val"])
    X_test  = scaler.transform(X[split == "test"])

    y_train = y[train_mask].values
    y_val   = y[split == "val"].values
    y_test  = y[split == "test"].values

    meta_val  = df[split == "val"][["Season", "team"]].reset_index(drop=True)
    meta_test = df[split == "test"][["Season", "team"]].reset_index(drop=True)

    print(f"  [{label}] Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    print(f"  [{label}] Train relegation rate: {y_train.mean():.3f}")
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, keep, meta_val, meta_test

print("\nPreparing data splits …")
(X_tr_f, y_tr_f, X_v_f, y_v_f, X_te_f, y_te_f,
 scaler_f, feat_f, meta_v_f, meta_te_f) = prepare_splits(df_full, "full")

(X_tr_m, y_tr_m, X_v_m, y_v_m, X_te_m, y_te_m,
 scaler_m, feat_m, meta_v_m, meta_te_m) = prepare_splits(df_mid, "mid")


# ══════════════════════════════════════════════════════════════════════════
# 3.  MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════

def get_models():
    """Return dict of model_name -> model."""
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=2000, C=0.5, random_state=42
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            scale_pos_weight=5.67,  # ~(1-0.15)/0.15
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, use_label_encoder=False,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=5,
            class_weight="balanced", random_state=42,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu",
            max_iter=1000, early_stopping=True, validation_fraction=0.15,
            random_state=42, alpha=0.01,
        ),
    }


# ══════════════════════════════════════════════════════════════════════════
# 4.  TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════

def evaluate(model, X, y, label=""):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)

    acc  = accuracy_score(y, y_pred)
    f1   = f1_score(y, y_pred, zero_division=0)
    prec = precision_score(y, y_pred, zero_division=0)
    rec  = recall_score(y, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y, y_prob)
    except ValueError:
        auc = float("nan")

    print(f"    {label:12s}  Acc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}")
    return {"accuracy": round(acc, 4), "f1": round(f1, 4), "auc_roc": round(auc, 4),
            "precision": round(prec, 4), "recall": round(rec, 4)}


def train_and_evaluate(X_tr, y_tr, X_v, y_v, X_te, y_te, mode_label, meta_test):
    """Train all models, return results dict."""
    results = {}
    models_dict = get_models()

    for name, model in models_dict.items():
        print(f"\n  ── {name} ({mode_label}) ──")
        model.fit(X_tr, y_tr)

        train_metrics = evaluate(model, X_tr, y_tr, "Train")
        val_metrics   = evaluate(model, X_v,  y_v,  "Val")
        test_metrics  = evaluate(model, X_te, y_te, "Test")

        # Detailed test predictions
        y_pred_test = model.predict(X_te)
        y_prob_test = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else y_pred_test.astype(float)

        print(f"    Confusion (test):\n{confusion_matrix(y_te, y_pred_test)}")

        # Save model
        model_path = OUT_DIR / f"{name.lower()}_{mode_label}.joblib"
        joblib.dump(model, model_path)

        results[name] = {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }

        # Print test predictions with team names
        if meta_test is not None:
            pred_df = meta_test.copy()
            pred_df["prob_relegated"] = y_prob_test
            pred_df["pred_relegated"] = y_pred_test
            pred_df["actual_relegated"] = y_te
            # Show predicted relegation candidates
            top_risk = pred_df.sort_values("prob_relegated", ascending=False)
            print(f"    Top relegation risks (test):")
            for _, row in top_risk.head(10).iterrows():
                flag = "✓" if row["actual_relegated"] == 1 else " "
                print(f"      {flag} {row['Season']} {row['team']:20s}  "
                      f"prob={row['prob_relegated']:.3f}  pred={int(row['pred_relegated'])}  "
                      f"actual={int(row['actual_relegated'])}")

    return results


print("\n" + "=" * 70)
print("FULL-SEASON MODELS")
print("=" * 70)
results_full = train_and_evaluate(X_tr_f, y_tr_f, X_v_f, y_v_f, X_te_f, y_te_f,
                                   "full", meta_te_f)

print("\n" + "=" * 70)
print("MID-SEASON MODELS (up to matchday 19)")
print("=" * 70)
results_mid = train_and_evaluate(X_tr_m, y_tr_m, X_v_m, y_v_m, X_te_m, y_te_m,
                                  "mid", meta_te_m)


# ══════════════════════════════════════════════════════════════════════════
# 5.  SAVE RESULTS SUMMARY
# ══════════════════════════════════════════════════════════════════════════

# Also save scalers & feature lists
joblib.dump(scaler_f, OUT_DIR / "scaler_full.joblib")
joblib.dump(scaler_m, OUT_DIR / "scaler_mid.joblib")

summary = {
    "task": "P3 Relegation Prediction",
    "target": "relegated (binary: bottom 3 in final standings)",
    "data_split": {"train": "2000-2021", "val": "2021-2023", "test": "2023-2025"},
    "class_balance": f"{ts['relegated'].mean():.3f} (15% relegated)",
    "imbalance_handling": "class_weight=balanced / scale_pos_weight for XGBoost",
    "features_full_season": feat_f,
    "features_mid_season": feat_m,
    "results_full_season": results_full,
    "results_mid_season": results_mid,
}

with open(OUT_DIR / "results_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print("\n── Full Season (Test Set) ──")
for name, r in results_full.items():
    m = r["test"]
    print(f"  {name:22s}  Acc={m['accuracy']:.3f}  F1={m['f1']:.3f}  "
          f"AUC={m['auc_roc']:.3f}  Prec={m['precision']:.3f}  Rec={m['recall']:.3f}")

print("\n── Mid Season (Test Set) ──")
for name, r in results_mid.items():
    m = r["test"]
    print(f"  {name:22s}  Acc={m['accuracy']:.3f}  F1={m['f1']:.3f}  "
          f"AUC={m['auc_roc']:.3f}  Prec={m['precision']:.3f}  Rec={m['recall']:.3f}")

print(f"\nAll models and results saved to: {OUT_DIR}")
print("Done!")

"""
P8: Transfer Adaptation Prediction
===================================
Predict whether a player will adapt well after transferring to a new EPL team.
Scout question: "If we sign Player X, will they perform well in our system?"

Pipeline:
1. Identify transfers (player appearing for different teams in consecutive seasons)
2. Define adaptation success (binary: maintained/improved per-90 stats)
3. Engineer features (player profile, style match, team strength, etc.)
4. Train Logistic Regression, XGBoost, Random Forest, MLP
5. Evaluate and produce scout-friendly outputs
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score, classification_report,
                             confusion_matrix, roc_curve)
from sklearn.impute import SimpleImputer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE = Path(r"C:\Users\xcv54\workspace\EPL project")
DATA = BASE / "data" / "processed"
OUT  = BASE / "models" / "p8_transfer_adaptation"
OUT.mkdir(parents=True, exist_ok=True)

# ─── 1. Load data ───────────────────────────────────────────────────────────
print("=" * 70)
print("P8: Transfer Adaptation Prediction")
print("=" * 70)

print("\n[1/7] Loading data...")
pss = pd.read_parquet(DATA / "player_season_stats.parquet")
clusters = pd.read_parquet(BASE / "models" / "p5_clustering" / "cluster_assignments_k6.parquet")
team_summary = pd.read_parquet(DATA / "team_season_summary.parquet")
match_results = pd.read_parquet(DATA / "match_results.parquet")

print(f"  player_season_stats: {pss.shape}")
print(f"  cluster_assignments: {clusters.shape}")
print(f"  team_season_summary: {team_summary.shape}")
print(f"  match_results:       {match_results.shape}")

# ─── Helper: season ordering ────────────────────────────────────────────────
ALL_SEASONS = sorted(pss["season"].unique())
SEASON_ORDER = {s: i for i, s in enumerate(ALL_SEASONS)}

def next_season(s):
    idx = SEASON_ORDER.get(s)
    if idx is None or idx + 1 >= len(ALL_SEASONS):
        return None
    return ALL_SEASONS[idx + 1]

def prev_season(s):
    idx = SEASON_ORDER.get(s)
    if idx is None or idx - 1 < 0:
        return None
    return ALL_SEASONS[idx - 1]

def season_start_year(s):
    return int(s.split("/")[0])

# ─── 2. Compute ELO ratings per team-season ────────────────────────────────
print("\n[2/7] Computing team ELO ratings...")

# Build ELO from match_results
elo_ratings = {}  # team -> current elo
K = 20
HOME_ADV = 50

def expected(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))

# Sort matches by date
mr = match_results.sort_values("MatchDate").copy()

# Track ELO snapshots per team per season
team_season_elo = {}  # (team, season) -> list of elos

for _, row in mr.iterrows():
    ht, at, season = row["HomeTeam"], row["AwayTeam"], row["Season"]
    ra = elo_ratings.get(ht, 1500)
    rb = elo_ratings.get(at, 1500)

    ea = expected(ra + HOME_ADV, rb)
    eb = 1 - ea

    res = row["FullTimeResult"]
    if res == "H":
        sa, sb = 1, 0
    elif res == "A":
        sa, sb = 0, 1
    else:
        sa, sb = 0.5, 0.5

    elo_ratings[ht] = ra + K * (sa - ea)
    elo_ratings[at] = rb + K * (sb - eb)

    team_season_elo.setdefault((ht, season), []).append(elo_ratings[ht])
    team_season_elo.setdefault((at, season), []).append(elo_ratings[at])

# Average ELO per team-season
elo_df = pd.DataFrame([
    {"team": t, "season": s, "elo": np.mean(vals)}
    for (t, s), vals in team_season_elo.items()
])
print(f"  ELO ratings computed for {len(elo_df)} team-seasons")

# ─── 3. Identify transfers and build adaptation dataset ─────────────────────
print("\n[3/7] Identifying transfers and defining adaptation success...")

# Sort player seasons
pss_sorted = pss.sort_values(["player", "season"]).copy()
pss_sorted["season_idx"] = pss_sorted["season"].map(SEASON_ORDER)

# For each player, find consecutive season pairs where team changed
transfers = []
for player, grp in pss_sorted.groupby("player"):
    grp = grp.sort_values("season_idx")
    rows = grp.to_dict("records")
    for i in range(len(rows) - 1):
        curr = rows[i]
        nxt = rows[i + 1]
        # Must be consecutive seasons
        if nxt["season_idx"] != curr["season_idx"] + 1:
            continue
        # Team must have changed
        if curr["team"] == nxt["team"]:
            continue
        # Both must have meaningful minutes (>= 450 min = ~5 full games)
        if curr["min"] < 450 or nxt["min"] < 450:
            continue

        transfers.append({
            "player": player,
            "season_old": curr["season"],
            "season_new": nxt["season"],
            "team_old": curr["team"],
            "team_new": nxt["team"],
            # Per-90 stats old team
            "g_a_per90_old": curr["g_a_1"],
            "gls_per90_old": curr["gls_1"],
            "ast_per90_old": curr["ast_1"],
            "min_old": curr["min"],
            "90s_old": curr["90s"],
            "mp_old": curr["mp"],
            "starts_old": curr["starts"],
            # Per-90 stats new team
            "g_a_per90_new": nxt["g_a_1"],
            "gls_per90_new": nxt["gls_1"],
            "ast_per90_new": nxt["ast_1"],
            "min_new": nxt["min"],
            "90s_new": nxt["90s"],
            "mp_new": nxt["mp"],
            "starts_new": nxt["starts"],
            # Player info
            "age": nxt["age"],
            "pos": nxt["pos"],
            "position": nxt["position"],
            "market_value": nxt["market_value"],
            "market_value_old": curr["market_value"],
            "height_cm": nxt["height_cm"],
        })

tf = pd.DataFrame(transfers)
print(f"  Found {len(tf)} transfer events (min 450 mins each season)")

# ─── Define adaptation success ──────────────────────────────────────────────
# Primary metric: goals+assists per 90
# For defenders/GKs who have 0 g+a, also consider minutes share
# Success = at least 80% of previous per-90 output OR improvement

def compute_adaptation_label(row):
    old_ga = row["g_a_per90_old"]
    new_ga = row["g_a_per90_new"]

    # Minutes share: proportion of available minutes played
    # Use 90s as proxy
    old_mins_share = row["min_old"] / max(row["mp_old"] * 90, 1)
    new_mins_share = row["min_new"] / max(row["mp_new"] * 90, 1)

    # For attackers (MF, FW): primarily g+a per 90
    # For defenders: primarily minutes share (playing time = adaptation)
    pos = str(row["pos"]) if pd.notna(row["pos"]) else ""

    if old_ga > 0.05:  # Player had meaningful attacking output
        ga_ratio = new_ga / max(old_ga, 0.01)
        mins_ratio = new_mins_share / max(old_mins_share, 0.01)
        # Weighted: 60% attacking output, 40% playing time
        composite = 0.6 * ga_ratio + 0.4 * mins_ratio
        return 1 if composite >= 0.80 else 0
    else:
        # Defensive/low-output player: playing time is key metric
        mins_ratio = new_mins_share / max(old_mins_share, 0.01)
        return 1 if mins_ratio >= 0.75 else 0

tf["adapted"] = tf.apply(compute_adaptation_label, axis=1)
print(f"  Adaptation labels: {tf['adapted'].value_counts().to_dict()}")
print(f"  Adaptation rate: {tf['adapted'].mean():.1%}")

# ─── 4. Feature engineering ─────────────────────────────────────────────────
print("\n[4/7] Engineering features...")

# 4a. Merge cluster info
cluster_map = clusters[["player", "season", "cluster", "cluster_name"]].copy()
tf = tf.merge(cluster_map.rename(columns={"season": "season_old", "cluster": "cluster_old",
                                            "cluster_name": "cluster_name_old"}),
              on=["player", "season_old"], how="left")

# 4b. Team dominant cluster profile
# For each team-season, compute distribution of player clusters
team_cluster_dist = (
    clusters.groupby(["team", "season", "cluster"])
    .size()
    .unstack(fill_value=0)
)
# Normalize to proportions
team_cluster_dist = team_cluster_dist.div(team_cluster_dist.sum(axis=1), axis=0)
team_cluster_dist.columns = [f"team_cluster_{c}_pct" for c in team_cluster_dist.columns]
team_cluster_dist = team_cluster_dist.reset_index()

# Merge new team cluster profile
tf = tf.merge(
    team_cluster_dist.rename(columns={"team": "team_new", "season": "season_new"}),
    on=["team_new", "season_new"], how="left"
)

# Style match: what fraction of the new team shares the player's cluster?
cluster_pct_cols = [c for c in tf.columns if c.startswith("team_cluster_") and c.endswith("_pct")]
def style_match(row):
    if pd.isna(row.get("cluster_old")):
        return np.nan
    col = f"team_cluster_{int(row['cluster_old'])}_pct"
    return row.get(col, np.nan)

tf["style_match_pct"] = tf.apply(style_match, axis=1)

# 4c. Team strength (ELO)
tf = tf.merge(elo_df.rename(columns={"team": "team_old", "season": "season_old", "elo": "elo_old"}),
              on=["team_old", "season_old"], how="left")
tf = tf.merge(elo_df.rename(columns={"team": "team_new", "season": "season_new", "elo": "elo_new"}),
              on=["team_new", "season_new"], how="left")
tf["elo_diff"] = tf["elo_new"] - tf["elo_old"]  # positive = moving to stronger team
tf["moving_up"] = (tf["elo_diff"] > 0).astype(int)

# 4d. Team points from summary
team_pts = team_summary[["Season", "team", "points", "goal_diff", "total_goals_for", "total_goals_against"]].copy()
team_pts = team_pts.rename(columns={"Season": "season"})

tf = tf.merge(team_pts.rename(columns={"team": "team_new", "season": "season_new",
                                         "points": "new_team_points", "goal_diff": "new_team_gd",
                                         "total_goals_for": "new_team_gf"}),
              on=["team_new", "season_new"], how="left")
tf = tf.merge(team_pts.rename(columns={"team": "team_old", "season": "season_old",
                                         "points": "old_team_points", "goal_diff": "old_team_gd",
                                         "total_goals_for": "old_team_gf"}),
              on=["team_old", "season_old"], how="left")
tf["points_diff"] = tf["new_team_points"] - tf["old_team_points"]

# 4e. Market value relative to new squad average
squad_avg_mv = (
    pss.groupby(["team", "season"])["market_value"]
    .mean()
    .reset_index()
    .rename(columns={"market_value": "squad_avg_mv"})
)
tf = tf.merge(
    squad_avg_mv.rename(columns={"team": "team_new", "season": "season_new"}),
    on=["team_new", "season_new"], how="left"
)
tf["mv_vs_squad"] = tf["market_value"] / tf["squad_avg_mv"].replace(0, np.nan)

# 4f. EPL experience (seasons in EPL before transfer)
player_seasons = pss.groupby("player")["season"].apply(set).to_dict()
def epl_experience(row):
    seasons = player_seasons.get(row["player"], set())
    return sum(1 for s in seasons if SEASON_ORDER.get(s, 999) < SEASON_ORDER.get(row["season_new"], 0))

tf["epl_experience"] = tf.apply(epl_experience, axis=1)

# 4g. Historical adaptation rate (if player transferred before)
tf = tf.sort_values(["player", "season_new"])
tf["transfer_count"] = tf.groupby("player").cumcount()

# Compute rolling adaptation rate for repeat transfers
def historical_adapt_rate(row):
    if row["transfer_count"] == 0:
        return np.nan  # First transfer, no history
    mask = (tf["player"] == row["player"]) & (tf["season_new"] < row["season_new"])
    prev = tf.loc[mask, "adapted"]
    if len(prev) == 0:
        return np.nan
    return prev.mean()

tf["hist_adapt_rate"] = tf.apply(historical_adapt_rate, axis=1)

# 4h. Position encoding
pos_map = {"GK": 0, "DF": 1, "DF,MF": 2, "MF": 3, "MF,DF": 2, "MF,FW": 4, "FW,MF": 4, "FW": 5}
tf["pos_code"] = tf["pos"].map(pos_map).fillna(3)

# 4i. Age bucket
tf["age_bucket"] = pd.cut(tf["age"], bins=[0, 23, 27, 30, 40], labels=[0, 1, 2, 3]).astype(float)

# 4j. Performance metrics from old team
tf["was_starter"] = (tf["starts_old"] / tf["mp_old"].replace(0, np.nan)).fillna(0)

print(f"  Feature engineering complete. Dataset shape: {tf.shape}")
print(f"  Columns: {tf.columns.tolist()}")

# ─── 5. Prepare ML dataset ─────────────────────────────────────────────────
print("\n[5/7] Preparing ML dataset and training models...")

FEATURE_COLS = [
    "age", "pos_code", "age_bucket", "epl_experience", "transfer_count",
    "g_a_per90_old", "gls_per90_old", "ast_per90_old",
    "90s_old", "was_starter",
    "market_value", "mv_vs_squad",
    "elo_old", "elo_new", "elo_diff", "moving_up",
    "style_match_pct",
    "hist_adapt_rate",
    "height_cm",
    "old_team_points", "new_team_points", "points_diff",
]
# Add team cluster distribution cols
FEATURE_COLS += cluster_pct_cols

TARGET = "adapted"

# Time-based split
def get_split(season):
    yr = season_start_year(season)
    if yr < 2021:
        return "train"
    elif yr <= 2022:
        return "val"
    else:
        return "test"

tf["split"] = tf["season_new"].apply(get_split)
print(f"  Split distribution:\n{tf['split'].value_counts().to_string()}")

# Filter to available features
available_features = [c for c in FEATURE_COLS if c in tf.columns]
print(f"  Using {len(available_features)} features: {available_features}")

X = tf[available_features].copy()
y = tf[TARGET].copy()

# Impute missing values
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=available_features, index=X.index)

# Scale
scaler = StandardScaler()

train_mask = tf["split"] == "train"
val_mask = tf["split"] == "val"
test_mask = tf["split"] == "test"

X_train = X_imputed[train_mask]
y_train = y[train_mask]
X_val = X_imputed[val_mask]
y_val = y[val_mask]
X_test = X_imputed[test_mask]
y_test = y[test_mask]

print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print(f"  Train adaptation rate: {y_train.mean():.1%}")
print(f"  Val adaptation rate:   {y_val.mean():.1%}")
print(f"  Test adaptation rate:  {y_test.mean():.1%}")

# Scale features
X_train_sc = scaler.fit_transform(X_train)
X_val_sc = scaler.transform(X_val)
X_test_sc = scaler.transform(X_test)

# ─── 6. Train models ────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=0.5),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5,
                                             random_state=42, class_weight="balanced"),
    "XGBoost": GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                           min_samples_leaf=5, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42,
                          early_stopping=True, validation_fraction=0.15, alpha=0.01),
}

results = {}

for name, model in models.items():
    print(f"\n  Training {name}...")
    if name in ["Logistic Regression", "MLP"]:
        model.fit(X_train_sc, y_train)
        val_pred = model.predict(X_val_sc)
        val_proba = model.predict_proba(X_val_sc)[:, 1] if hasattr(model, "predict_proba") else val_pred
        test_pred = model.predict(X_test_sc)
        test_proba = model.predict_proba(X_test_sc)[:, 1] if hasattr(model, "predict_proba") else test_pred
    else:
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_proba = model.predict_proba(X_val)[:, 1]
        test_pred = model.predict(X_test)
        test_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    def calc_metrics(y_true, y_pred, y_prob):
        m = {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        }
        try:
            m["auc_roc"] = round(roc_auc_score(y_true, y_prob), 4)
        except ValueError:
            m["auc_roc"] = None
        return m

    val_metrics = calc_metrics(y_val, val_pred, val_proba)
    test_metrics = calc_metrics(y_test, test_pred, test_proba)

    results[name] = {
        "val": val_metrics,
        "test": test_metrics,
        "model": model,
        "test_proba": test_proba,
        "test_pred": test_pred,
    }

    print(f"    Val  - Acc: {val_metrics['accuracy']:.3f} | F1: {val_metrics['f1']:.3f} | AUC: {val_metrics.get('auc_roc', 'N/A')}")
    print(f"    Test - Acc: {test_metrics['accuracy']:.3f} | F1: {test_metrics['f1']:.3f} | AUC: {test_metrics.get('auc_roc', 'N/A')}")

# ─── 7. Analysis and scout outputs ──────────────────────────────────────────
print("\n[6/7] Generating analysis and scout outputs...")

# Pick best model by val AUC
best_name = max(results, key=lambda k: results[k]["val"].get("auc_roc") or 0)
best = results[best_name]
print(f"\n  Best model (by val AUC): {best_name}")

# ─── Feature importance ─────────────────────────────────────────────────────
# Use XGBoost or RF feature importances
fi_model_name = "XGBoost" if "XGBoost" in results else "Random Forest"
fi_model = results[fi_model_name]["model"]
importances = fi_model.feature_importances_
fi_df = pd.DataFrame({
    "feature": available_features,
    "importance": importances
}).sort_values("importance", ascending=False)

print(f"\n  Top 10 features predicting transfer adaptation ({fi_model_name}):")
for _, row in fi_df.head(10).iterrows():
    print(f"    {row['feature']:30s} {row['importance']:.4f}")

# ─── Figure 1: Model comparison ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart of metrics
metric_names = ["accuracy", "f1", "auc_roc", "precision", "recall"]
model_names = list(results.keys())
x = np.arange(len(metric_names))
width = 0.18

for i, mname in enumerate(model_names):
    vals = [results[mname]["test"].get(m) or 0 for m in metric_names]
    axes[0].bar(x + i * width, vals, width, label=mname, alpha=0.85)

axes[0].set_xticks(x + width * 1.5)
axes[0].set_xticklabels(metric_names, fontsize=10)
axes[0].set_ylabel("Score")
axes[0].set_title("Test Set: Model Comparison")
axes[0].legend(fontsize=8)
axes[0].set_ylim(0, 1.05)

# ROC curves
for mname in model_names:
    proba = results[mname]["test_proba"]
    try:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = results[mname]["test"].get("auc_roc") or 0
        axes[1].plot(fpr, tpr, label=f"{mname} (AUC={auc:.3f})")
    except:
        pass
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curves (Test Set)")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT / "model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved model_comparison.png")

# ─── Figure 2: Feature importance ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
top_fi = fi_df.head(15)
ax.barh(range(len(top_fi)), top_fi["importance"].values, color="steelblue")
ax.set_yticks(range(len(top_fi)))
ax.set_yticklabels(top_fi["feature"].values)
ax.invert_yaxis()
ax.set_xlabel("Feature Importance")
ax.set_title(f"Top 15 Features for Transfer Adaptation ({fi_model_name})")
plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved feature_importance.png")

# ─── Figure 3: Adaptation by key factors ────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Age vs adaptation
age_bins = pd.cut(tf["age"], bins=[0, 23, 26, 29, 40])
age_adapt = tf.groupby(age_bins, observed=True)["adapted"].mean()
axes[0, 0].bar(range(len(age_adapt)), age_adapt.values, color="coral")
axes[0, 0].set_xticks(range(len(age_adapt)))
axes[0, 0].set_xticklabels(["<=23", "24-26", "27-29", "30+"], fontsize=10)
axes[0, 0].set_ylabel("Adaptation Rate")
axes[0, 0].set_title("Adaptation Rate by Age Group")

# ELO diff vs adaptation
elo_bins = pd.cut(tf["elo_diff"].dropna(), bins=5)
elo_adapt = tf.groupby(pd.cut(tf["elo_diff"], bins=5), observed=True)["adapted"].mean()
axes[0, 1].bar(range(len(elo_adapt)), elo_adapt.values, color="teal")
axes[0, 1].set_xticks(range(len(elo_adapt)))
labels = [f"{iv.left:.0f} to {iv.right:.0f}" for iv in elo_adapt.index]
axes[0, 1].set_xticklabels(labels, fontsize=8, rotation=15)
axes[0, 1].set_ylabel("Adaptation Rate")
axes[0, 1].set_title("Adaptation by Team Strength Change (ELO diff)")

# EPL experience vs adaptation
exp_bins = pd.cut(tf["epl_experience"], bins=[0, 1, 3, 5, 25], include_lowest=True)
exp_adapt = tf.groupby(exp_bins, observed=True)["adapted"].mean()
axes[1, 0].bar(range(len(exp_adapt)), exp_adapt.values, color="mediumpurple")
axes[1, 0].set_xticks(range(len(exp_adapt)))
axes[1, 0].set_xticklabels(["0-1 yr", "2-3 yr", "4-5 yr", "6+ yr"], fontsize=10)
axes[1, 0].set_ylabel("Adaptation Rate")
axes[1, 0].set_title("Adaptation Rate by EPL Experience")

# Position vs adaptation
pos_adapt = tf.groupby("pos")["adapted"].agg(["mean", "count"])
pos_adapt = pos_adapt[pos_adapt["count"] >= 5].sort_values("mean", ascending=False)
axes[1, 1].barh(range(len(pos_adapt)), pos_adapt["mean"].values, color="goldenrod")
axes[1, 1].set_yticks(range(len(pos_adapt)))
axes[1, 1].set_yticklabels(pos_adapt.index.values)
axes[1, 1].set_xlabel("Adaptation Rate")
axes[1, 1].set_title("Adaptation Rate by Position")

plt.suptitle("Transfer Adaptation Insights for Scouts", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT / "adaptation_insights.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved adaptation_insights.png")

# ─── Figure 4: Confusion matrix for best model ──────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, best["test_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Struggled", "Adapted"], yticklabels=["Struggled", "Adapted"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix - {best_name} (Test Set)")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved confusion_matrix.png")

# ─── Example predictions for recent test-set transfers ───────────────────────
print("\n[7/7] Generating scout example predictions...")

test_df = tf[test_mask].copy()
test_df["adapt_probability"] = best["test_proba"]
test_df["predicted"] = best["test_pred"]

# Show top predictions
test_display = test_df.sort_values("adapt_probability", ascending=False)[
    ["player", "season_new", "team_old", "team_new", "age", "pos",
     "adapt_probability", "predicted", "adapted"]
].head(20)

print("\n  Top 20 Test-Set Transfer Predictions (highest adaptation probability):")
print("  " + "-" * 110)
print(f"  {'Player':<25s} {'Season':<10s} {'From':<18s} {'To':<18s} {'Age':>4s} {'Pos':<6s} {'P(adapt)':>9s} {'Pred':>5s} {'Actual':>6s}")
print("  " + "-" * 110)
for _, r in test_display.iterrows():
    marker = "OK" if r["predicted"] == r["adapted"] else "XX"
    print(f"  {r['player']:<25s} {r['season_new']:<10s} {r['team_old']:<18s} {r['team_new']:<18s} "
          f"{r['age']:>4.0f} {r['pos']:<6s} {r['adapt_probability']:>8.1%} {r['predicted']:>5d} {r['adapted']:>6d} {marker}")

# ─── Scout function: predict for any player + target team ────────────────────
print("\n  Scout Prediction Function (example usage):")
print("  predict_adaptation(player='Cole Palmer', target_team='Chelsea', season='2023/24')")

def predict_adaptation_example():
    """Show example of how scout would use the model."""
    # Find a notable transfer in test set
    notable = test_df[test_df["adapt_probability"] > 0.5].head(3)
    examples = []
    for _, r in notable.iterrows():
        examples.append({
            "player": r["player"],
            "from_team": r["team_old"],
            "to_team": r["team_new"],
            "season": r["season_new"],
            "adaptation_probability": round(float(r["adapt_probability"]), 3),
            "prediction": "Will adapt" if r["predicted"] == 1 else "May struggle",
            "actual_outcome": "Adapted" if r["adapted"] == 1 else "Struggled",
        })
    return examples

scout_examples = predict_adaptation_example()
for ex in scout_examples:
    print(f"\n    Player: {ex['player']}")
    print(f"    Transfer: {ex['from_team']} -> {ex['to_team']} ({ex['season']})")
    print(f"    Adaptation Probability: {ex['adaptation_probability']:.1%}")
    print(f"    Prediction: {ex['prediction']}")
    print(f"    Actual: {ex['actual_outcome']}")

# ─── Save results summary ───────────────────────────────────────────────────
summary = {
    "task": "P8: Transfer Adaptation Prediction",
    "description": "Predict whether a player will adapt well after transferring to a new EPL team",
    "dataset": {
        "total_transfers": len(tf),
        "train_size": int(train_mask.sum()),
        "val_size": int(val_mask.sum()),
        "test_size": int(test_mask.sum()),
        "adaptation_rate_overall": round(float(tf["adapted"].mean()), 4),
        "adaptation_rate_train": round(float(y_train.mean()), 4),
        "adaptation_rate_val": round(float(y_val.mean()), 4),
        "adaptation_rate_test": round(float(y_test.mean()), 4),
        "min_minutes_threshold": 450,
        "adaptation_threshold": "80% of previous per-90 output (composite metric)",
    },
    "features_used": available_features,
    "top_features": fi_df.head(10)[["feature", "importance"]].to_dict("records"),
    "models": {},
    "best_model": best_name,
    "scout_examples": scout_examples,
    "key_findings": [],
}

for mname in model_names:
    summary["models"][mname] = {
        "val_metrics": results[mname]["val"],
        "test_metrics": results[mname]["test"],
    }

# Key findings
findings = []
findings.append(f"Best model: {best_name} with test AUC of {results[best_name]['test'].get('auc_roc', 'N/A')}")
findings.append(f"Top predictive feature: {fi_df.iloc[0]['feature']} (importance: {fi_df.iloc[0]['importance']:.4f})")
findings.append(f"Overall adaptation rate: {tf['adapted'].mean():.1%} of transfers result in successful adaptation")

# Age insight
young = tf[tf["age"] <= 23]["adapted"].mean()
prime = tf[(tf["age"] > 23) & (tf["age"] <= 29)]["adapted"].mean()
older = tf[tf["age"] > 29]["adapted"].mean()
findings.append(f"Age effect: Young (<=23): {young:.1%}, Prime (24-29): {prime:.1%}, Senior (30+): {older:.1%} adaptation rate")

# ELO insight
up = tf[tf["elo_diff"] > 0]["adapted"].mean()
down = tf[tf["elo_diff"] < 0]["adapted"].mean()
findings.append(f"Moving up vs down: Players moving to stronger teams adapt {up:.1%} vs {down:.1%} for weaker teams")

summary["key_findings"] = findings

with open(OUT / "results_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\n  Saved results_summary.json")

# Save the transfer dataset for further analysis
tf_save = tf.drop(columns=["split"], errors="ignore")
tf_save.to_parquet(OUT / "transfer_dataset.parquet", index=False)
print(f"  Saved transfer_dataset.parquet ({tf_save.shape})")

# ─── Final summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("P8 COMPLETE: Transfer Adaptation Prediction")
print("=" * 70)
print(f"\n  Dataset: {len(tf)} transfers identified")
print(f"  Best model: {best_name}")
print(f"  Test metrics:")
for m in ["accuracy", "f1", "auc_roc", "precision", "recall"]:
    v = results[best_name]["test"].get(m)
    print(f"    {m:<12s}: {v:.4f}" if v else f"    {m:<12s}: N/A")

print(f"\n  Key findings:")
for f in findings:
    print(f"    - {f}")

print(f"\n  Outputs saved to: {OUT}")
print(f"    - model_comparison.png")
print(f"    - feature_importance.png")
print(f"    - adaptation_insights.png")
print(f"    - confusion_matrix.png")
print(f"    - results_summary.json")
print(f"    - transfer_dataset.parquet")
print()

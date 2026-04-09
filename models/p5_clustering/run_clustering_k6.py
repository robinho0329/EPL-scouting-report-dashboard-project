"""
P5 Supplementary: K=6 Clustering for More Granular Player Archetypes
====================================================================
Builds on the main pipeline's PCA/scaling. Uses K=6 to find more
distinct player types. Saves additional figures and updates results.
"""

import os
import json
import warnings
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(BASE, "..", ".."))
FIG_DIR = os.path.join(BASE, "figures")

FEATURES_PATH = os.path.join(PROJECT, "data", "features", "player_features.parquet")

# ─── Load & prepare (same as main pipeline) ──────────────────────────────────
print("Loading and preparing data...")
df = pd.read_parquet(FEATURES_PATH)
df = df[df["min"] >= 900].copy()

CLUSTERING_FEATURES = [
    "goals_p90", "assists_p90", "goal_contributions_p90",
    "yellow_cards_p90", "red_cards_p90", "penalties_p90",
    "gls", "ast", "g_a", "g_pk",
    "90s", "min", "mp", "starts",
    "age_used", "height_cm",
    "market_value",
    "goal_contribution_rate", "minutes_share",
    "epl_experience",
    "consistency_mean", "consistency_cv",
]
available = [f for f in CLUSTERING_FEATURES if f in df.columns]

df_clust = df.dropna(subset=available, how="all").copy()
X_raw = df_clust[available].copy()
for col in available:
    X_raw[col] = X_raw[col].fillna(X_raw[col].median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

pca = PCA(n_components=8)
X_pca = pca.fit_transform(X_scaled)

meta = df_clust[["player", "pos", "position", "season", "team",
                  "age_used", "market_value", "age_bracket"]].reset_index(drop=True)
X_raw = X_raw.reset_index(drop=True)

# 2D embedding
if HAS_UMAP:
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)
    X_2d = reducer.fit_transform(X_scaled)
    embed_name = "UMAP"
else:
    reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    X_2d = reducer.fit_transform(X_scaled)
    embed_name = "t-SNE"

# ─── K=6 Clustering ──────────────────────────────────────────────────────────
K = 6
print(f"\nRunning K-Means with K={K}...")
km = KMeans(n_clusters=K, n_init=20, random_state=42)
labels = km.fit_predict(X_pca)

metrics = {
    "silhouette": float(silhouette_score(X_pca, labels)),
    "davies_bouldin": float(davies_bouldin_score(X_pca, labels)),
    "calinski_harabasz": float(calinski_harabasz_score(X_pca, labels)),
}
print(f"  Silhouette: {metrics['silhouette']:.4f}")
print(f"  Davies-Bouldin: {metrics['davies_bouldin']:.4f}")
print(f"  Calinski-Harabasz: {metrics['calinski_harabasz']:.1f}")

# ─── Profile clusters ────────────────────────────────────────────────────────
profile_df = X_raw.copy()
profile_df["cluster"] = labels
for col in ["player", "pos", "position", "season", "team", "age_used",
            "market_value", "age_bracket"]:
    profile_df[col] = meta[col].values

# Compute global means
global_means = {f: profile_df[f].mean() for f in available}

cluster_profiles = {}
for c in range(K):
    sub = profile_df[profile_df["cluster"] == c]
    cluster_profiles[c] = {
        "size": int(len(sub)),
        "mean_stats": {col: float(round(sub[col].mean(), 4)) for col in available},
        "dominant_positions": sub["pos"].value_counts().head(3).to_dict(),
        "detailed_positions": sub["position"].dropna().value_counts().head(5).to_dict(),
        "market_value_mean": float(sub["market_value"].mean()),
        "market_value_median": float(sub["market_value"].median()),
        "age_mean": float(sub["age_used"].mean()),
        "age_brackets": sub["age_bracket"].value_counts().to_dict(),
        "top_players": sub.nlargest(5, "market_value")[
            ["player", "season", "team", "market_value"]].to_dict("records"),
    }

# ─── Smart naming ────────────────────────────────────────────────────────────
def name_cluster(p, gm):
    s = p["mean_stats"]
    pos = p["dominant_positions"]
    top_pos = list(pos.keys())[0] if pos else ""
    gc = s.get("goal_contributions_p90", 0)
    goals = s.get("goals_p90", 0)
    assists = s.get("assists_p90", 0)
    mv = p["market_value_mean"]
    age = p["age_mean"]
    mins = s.get("minutes_share", 0)
    yellows = s.get("yellow_cards_p90", 0)
    exp = s.get("epl_experience", 0)
    consist = s.get("consistency_mean", 0)

    # Check position composition
    gk_pct = pos.get("GK", 0) / p["size"]
    df_pct = (pos.get("DF", 0) + pos.get("DF,MF", 0)) / p["size"]
    fw_pct = (pos.get("FW", 0) + pos.get("FW,MF", 0)) / p["size"]
    mf_pct = (pos.get("MF", 0) + pos.get("MF,FW", 0) + pos.get("MF,DF", 0)) / p["size"]

    if gk_pct > 0.4:
        return "Goalkeepers"
    elif gc > gm["goal_contributions_p90"] * 2.0 and mv > gm["market_value"] * 2:
        return "Elite Attackers"
    elif gc > gm["goal_contributions_p90"] * 1.5 and fw_pct > 0.4:
        return "Goal Threats"
    elif assists > gm["assists_p90"] * 1.5 and gc > gm["goal_contributions_p90"] * 1.2:
        return "Creative Playmakers"
    elif df_pct > 0.5 and mv > gm["market_value"] * 1.5:
        return "Top Defenders"
    elif df_pct > 0.5 and yellows > gm["yellow_cards_p90"] * 1.1:
        return "Defensive Workhorses"
    elif df_pct > 0.5:
        return "Solid Defenders"
    elif age < 25 and mv < gm["market_value"] * 0.6:
        return "Young Prospects"
    elif exp > gm["epl_experience"] * 1.5 and age > 28:
        return "Experienced Veterans"
    elif mins > gm["minutes_share"] * 1.2 and consist > gm.get("consistency_mean", 0) * 1.1:
        return "Reliable Starters"
    elif mf_pct > 0.4:
        return "Midfield Engine"
    elif mins < gm["minutes_share"] * 0.8:
        return "Squad Rotation Players"
    else:
        return "Versatile Contributors"


cluster_names = {}
used_names = set()
for c in range(K):
    name = name_cluster(cluster_profiles[c], global_means)
    if name in used_names:
        name = name + f" II"
    used_names.add(name)
    cluster_names[c] = name
    cluster_profiles[c]["cluster_name"] = name

print("\n  Cluster Profiles (K=6):")
for c in range(K):
    p = cluster_profiles[c]
    print(f"\n  C{c}: {cluster_names[c]} (n={p['size']})")
    print(f"    Positions: {p['dominant_positions']}")
    print(f"    Market Value: mean={p['market_value_mean']:,.0f}, median={p['market_value_median']:,.0f}")
    print(f"    Age: {p['age_mean']:.1f}")
    print(f"    Goals/90: {p['mean_stats']['goals_p90']:.3f}, Assists/90: {p['mean_stats']['assists_p90']:.3f}")
    print(f"    Top: {[x['player'] for x in p['top_players']]}")

# ─── Visualizations ──────────────────────────────────────────────────────────
palette = sns.color_palette("husl", K)

# UMAP scatter K=6
fig, ax = plt.subplots(figsize=(14, 9))
for c in range(K):
    mask = labels == c
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=[palette[c]], label=f"C{c}: {cluster_names[c]} (n={cluster_profiles[c]['size']})",
               alpha=0.5, s=15, edgecolors="none")
ax.set_xlabel(f"{embed_name} 1")
ax.set_ylabel(f"{embed_name} 2")
ax.set_title(f"Player Archetypes (K=6) - {embed_name}\nSilhouette={metrics['silhouette']:.3f}")
ax.legend(fontsize=9, loc="best", markerscale=2)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "clusters_k6_umap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\n  Saved: clusters_k6_umap.png")

# Radar chart K=6
RADAR_FEATURES = ["goals_p90", "assists_p90", "goal_contributions_p90",
                   "yellow_cards_p90", "goal_contribution_rate",
                   "minutes_share", "consistency_mean"]
radar_avail = [f for f in RADAR_FEATURES if f in available]

n_feat = len(radar_avail)
angles = np.linspace(0, 2 * np.pi, n_feat, endpoint=False).tolist()
angles += angles[:1]

all_vals = {f: [cluster_profiles[c]["mean_stats"][f] for c in range(K)] for f in radar_avail}
feat_min = {f: min(v) for f, v in all_vals.items()}
feat_max = {f: max(v) for f, v in all_vals.items()}

fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(polar=True))
for c in range(K):
    ax = axes[c // 3, c % 3]
    values = []
    for f in radar_avail:
        v = cluster_profiles[c]["mean_stats"][f]
        rng = feat_max[f] - feat_min[f]
        values.append((v - feat_min[f]) / rng if rng > 0 else 0.5)
    values += values[:1]
    ax.fill(angles, values, alpha=0.25, color=palette[c])
    ax.plot(angles, values, "o-", markersize=5, color=palette[c], linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.replace("_p90", "\n/90").replace("_", "\n") for f in radar_avail],
                       fontsize=7)
    ax.set_title(f"C{c}: {cluster_names[c]}\n(n={cluster_profiles[c]['size']})", fontsize=10, pad=20)
    ax.set_ylim(0, 1.1)

plt.suptitle("Player Archetype Radar Charts (K=6)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cluster_k6_radar.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: cluster_k6_radar.png")

# Size distribution K=6
fig, ax = plt.subplots(figsize=(12, 6))
sizes = [cluster_profiles[c]["size"] for c in range(K)]
bar_labels = [f"C{c}\n{cluster_names[c]}" for c in range(K)]
bars = ax.bar(bar_labels, sizes, color=palette, edgecolor="black", linewidth=0.5)
for bar, size in zip(bars, sizes):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
            str(size), ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Number of Player-Seasons")
ax.set_title("Player Archetype Distribution (K=6)")
plt.xticks(rotation=25, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cluster_k6_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: cluster_k6_distribution.png")

# Heatmap K=6
fig, ax = plt.subplots(figsize=(18, 5))
heat_data = pd.DataFrame({
    f"C{c}: {cluster_names[c]}": cluster_profiles[c]["mean_stats"]
    for c in range(K)
}).T
heat_norm = (heat_data - heat_data.mean()) / (heat_data.std() + 1e-8)
sns.heatmap(heat_norm, cmap="RdYlBu_r", center=0, linewidths=0.5, ax=ax)
ax.set_title("Player Archetype Feature Profiles (K=6, Z-score normalized)")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cluster_k6_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: cluster_k6_heatmap.png")

# ─── Save ─────────────────────────────────────────────────────────────────────
# Save K=6 model
with open(os.path.join(BASE, "kmeans_k6_model.pkl"), "wb") as f:
    pickle.dump(km, f)

# Save assignments
assign = meta.copy()
assign["cluster"] = labels
assign["cluster_name"] = [cluster_names[c] for c in labels]
assign.to_parquet(os.path.join(BASE, "cluster_assignments_k6.parquet"), index=False)

# Update results summary
def make_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    return obj

results_path = os.path.join(BASE, "results_summary.json")
with open(results_path, "r") as f:
    results = json.load(f)

results["kmeans_k6"] = {
    "metrics": make_serializable(metrics),
    "n_clusters": K,
    "cluster_profiles": make_serializable(cluster_profiles),
    "cluster_names": make_serializable(cluster_names),
}
results["figures"].extend([
    "clusters_k6_umap.png",
    "cluster_k6_radar.png",
    "cluster_k6_distribution.png",
    "cluster_k6_heatmap.png",
])

with open(results_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print("  Updated: results_summary.json")

print("\nK=6 supplementary analysis complete.")

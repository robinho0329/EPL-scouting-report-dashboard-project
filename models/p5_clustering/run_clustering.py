"""
P5: Player Clustering Pipeline
==============================
Clusters EPL players based on performance features using K-Means, DBSCAN, and GMM.
Produces visualizations (PCA, UMAP, radar charts) and a results_summary.json.
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
from matplotlib.patches import FancyBboxPatch

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

# Try UMAP, fall back to t-SNE
try:
    import umap
    HAS_UMAP = True
    print("[INFO] UMAP available.")
except ImportError:
    HAS_UMAP = False
    print("[INFO] UMAP not available, will use t-SNE for 2D projection.")
    from sklearn.manifold import TSNE

# ─── paths ───────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(BASE, "..", ".."))
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

FEATURES_PATH = os.path.join(PROJECT, "data", "features", "player_features.parquet")
SEASON_PATH = os.path.join(PROJECT, "data", "processed", "player_season_stats.parquet")

# ─── 1. Load data ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 1: Loading data")
print("=" * 70)

df_feat = pd.read_parquet(FEATURES_PATH)
df_season = pd.read_parquet(SEASON_PATH)
print(f"  player_features  : {df_feat.shape}")
print(f"  player_season    : {df_season.shape}")

# We'll work primarily with player_features since it already has per-90 stats
df = df_feat.copy()

# ─── 2. Feature preparation ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Feature preparation")
print("=" * 70)

# Filter: minimum 900 minutes played
df = df[df["min"] >= 900].copy()
print(f"  After 900-min filter: {len(df)} player-seasons")

# Select clustering features
CLUSTERING_FEATURES = [
    # per-90 stats
    "goals_p90", "assists_p90", "goal_contributions_p90",
    "yellow_cards_p90", "red_cards_p90", "penalties_p90",
    # raw counting (normalised by 90s implicitly via per-90, but also raw context)
    "gls", "ast", "g_a", "g_pk",
    # workload / involvement
    "90s", "min", "mp", "starts",
    # age and physicality
    "age_used", "height_cm",
    # market value
    "market_value",
    # derived features
    "goal_contribution_rate", "minutes_share",
    "epl_experience",
    "consistency_mean", "consistency_cv",
]

# Keep only features that actually exist
available = [f for f in CLUSTERING_FEATURES if f in df.columns]
print(f"  Selected {len(available)} features: {available}")

# Drop rows where all selected features are NaN
df_clust = df.dropna(subset=available, how="all").copy()
print(f"  After dropping all-NaN rows: {len(df_clust)}")

# Impute remaining NaN with column median
X_raw = df_clust[available].copy()
for col in available:
    median_val = X_raw[col].median()
    X_raw[col] = X_raw[col].fillna(median_val)

print(f"  Remaining NaN after imputation: {X_raw.isnull().sum().sum()}")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
print(f"  Scaled feature matrix: {X_scaled.shape}")

# Save meta for later profiling
meta = df_clust[["player", "pos", "position", "season", "team",
                  "age_used", "market_value", "age_bracket"]].copy()
meta = meta.reset_index(drop=True)
X_raw = X_raw.reset_index(drop=True)

# ─── 3. PCA ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3: PCA dimensionality reduction")
print("=" * 70)

pca_full = PCA().fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)

# Choose n_components for 90% variance
n_components = int(np.argmax(cumvar >= 0.90)) + 1
print(f"  Components for 90% variance: {n_components}")
print(f"  Explained variance (first 10): {np.round(pca_full.explained_variance_ratio_[:10], 4)}")

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)
print(f"  PCA matrix: {X_pca.shape}")

# Also get 2-component PCA for visualization
pca_2d = PCA(n_components=2)
X_pca2d = pca_2d.fit_transform(X_scaled)

# Plot explained variance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
            pca_full.explained_variance_ratio_, alpha=0.7, label="Individual")
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Explained Variance Ratio")
axes[0].set_title("PCA Explained Variance")
axes[0].legend()

axes[1].plot(range(1, len(cumvar) + 1), cumvar, "o-", markersize=4)
axes[1].axhline(y=0.90, color="r", linestyle="--", label="90% threshold")
axes[1].axvline(x=n_components, color="g", linestyle="--", label=f"n={n_components}")
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Explained Variance")
axes[1].set_title("Cumulative Explained Variance")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "pca_explained_variance.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: pca_explained_variance.png")

# ─── 4. UMAP / t-SNE for 2D embedding ────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: 2D embedding (UMAP or t-SNE)")
print("=" * 70)

if HAS_UMAP:
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)
    X_2d = reducer.fit_transform(X_scaled)
    embed_name = "UMAP"
else:
    reducer = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    X_2d = reducer.fit_transform(X_scaled)
    embed_name = "t-SNE"

print(f"  {embed_name} embedding: {X_2d.shape}")

# ─── 5. K-Means ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5: K-Means clustering")
print("=" * 70)

K_range = range(3, 16)
inertias = []
sil_scores = []
db_scores = []
ch_scores = []

for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
    labels = km.fit_predict(X_pca)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_pca, labels))
    db_scores.append(davies_bouldin_score(X_pca, labels))
    ch_scores.append(calinski_harabasz_score(X_pca, labels))
    print(f"  K={k:2d}  Silhouette={sil_scores[-1]:.4f}  DB={db_scores[-1]:.4f}  CH={ch_scores[-1]:.1f}")

# Elbow + Silhouette plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(list(K_range), inertias, "o-", markersize=5)
axes[0].set_xlabel("K")
axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow Method")

axes[1].plot(list(K_range), sil_scores, "o-", markersize=5, color="green")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score vs K")

axes[2].plot(list(K_range), ch_scores, "o-", markersize=5, color="purple")
axes[2].set_xlabel("K")
axes[2].set_ylabel("Calinski-Harabasz Index")
axes[2].set_title("Calinski-Harabasz vs K")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "kmeans_elbow_silhouette.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: kmeans_elbow_silhouette.png")

# Pick optimal K: highest silhouette
best_k = list(K_range)[np.argmax(sil_scores)]
print(f"\n  Optimal K (max silhouette): {best_k}")

km_final = KMeans(n_clusters=best_k, n_init=10, random_state=42)
kmeans_labels = km_final.fit_predict(X_pca)

kmeans_metrics = {
    "silhouette": float(silhouette_score(X_pca, kmeans_labels)),
    "davies_bouldin": float(davies_bouldin_score(X_pca, kmeans_labels)),
    "calinski_harabasz": float(calinski_harabasz_score(X_pca, kmeans_labels)),
    "n_clusters": best_k,
}
print(f"  K-Means final metrics: {kmeans_metrics}")

# ─── 6. DBSCAN ───────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6: DBSCAN clustering")
print("=" * 70)

# Use k-distance graph to find good eps
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_pca)
distances, _ = nn.kneighbors(X_pca)
k_distances = np.sort(distances[:, -1])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(k_distances)
ax.set_xlabel("Points (sorted)")
ax.set_ylabel("5th Nearest Neighbor Distance")
ax.set_title("K-Distance Graph for DBSCAN eps Selection")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "dbscan_kdistance.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: dbscan_kdistance.png")

# Try multiple eps values
best_dbscan_sil = -1
best_dbscan_params = {}
best_dbscan_labels = None

# Use percentiles of k-distances for eps candidates
eps_candidates = np.percentile(k_distances, [70, 75, 80, 85, 90, 95])
eps_candidates = np.unique(np.round(eps_candidates, 2))
min_samples_candidates = [5, 7, 10, 15]

for eps in eps_candidates:
    for ms in min_samples_candidates:
        db = DBSCAN(eps=eps, min_samples=ms)
        labels = db.fit_predict(X_pca)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        if n_clusters >= 2 and n_noise < len(labels) * 0.5:
            mask = labels != -1
            if mask.sum() > n_clusters:
                sil = silhouette_score(X_pca[mask], labels[mask])
                if sil > best_dbscan_sil:
                    best_dbscan_sil = sil
                    best_dbscan_params = {"eps": float(eps), "min_samples": ms}
                    best_dbscan_labels = labels.copy()
                    print(f"  eps={eps:.2f}, min_samples={ms}: {n_clusters} clusters, "
                          f"{n_noise} noise, silhouette={sil:.4f}")

if best_dbscan_labels is not None:
    mask = best_dbscan_labels != -1
    dbscan_metrics = {
        "silhouette": float(silhouette_score(X_pca[mask], best_dbscan_labels[mask])),
        "davies_bouldin": float(davies_bouldin_score(X_pca[mask], best_dbscan_labels[mask])),
        "calinski_harabasz": float(calinski_harabasz_score(X_pca[mask], best_dbscan_labels[mask])),
        "n_clusters": int(len(set(best_dbscan_labels)) - (1 if -1 in best_dbscan_labels else 0)),
        "n_noise": int((best_dbscan_labels == -1).sum()),
        "params": best_dbscan_params,
    }
    print(f"\n  Best DBSCAN: {dbscan_metrics}")
else:
    print("  [WARN] DBSCAN could not find good clustering. Using fallback.")
    dbscan_metrics = {"error": "no valid clustering found"}
    best_dbscan_labels = np.full(len(X_pca), -1)

# ─── 7. Gaussian Mixture Model ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 7: Gaussian Mixture Model")
print("=" * 70)

gmm_bics = []
gmm_aics = []
gmm_sils = []

for k in K_range:
    gmm = GaussianMixture(n_components=k, covariance_type="full",
                           random_state=42, n_init=3, max_iter=300)
    gmm.fit(X_pca)
    labels = gmm.predict(X_pca)
    gmm_bics.append(gmm.bic(X_pca))
    gmm_aics.append(gmm.aic(X_pca))
    gmm_sils.append(silhouette_score(X_pca, labels))
    print(f"  K={k:2d}  BIC={gmm_bics[-1]:.0f}  AIC={gmm_aics[-1]:.0f}  Silhouette={gmm_sils[-1]:.4f}")

# Plot BIC/AIC
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(list(K_range), gmm_bics, "o-", label="BIC", color="blue")
axes[0].plot(list(K_range), gmm_aics, "s-", label="AIC", color="orange")
axes[0].set_xlabel("Number of Components")
axes[0].set_ylabel("Information Criterion")
axes[0].set_title("GMM: BIC and AIC")
axes[0].legend()

axes[1].plot(list(K_range), gmm_sils, "o-", color="green")
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("GMM: Silhouette Score")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "gmm_bic_aic.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: gmm_bic_aic.png")

# Choose best GMM by BIC (lower is better)
best_gmm_k = list(K_range)[np.argmin(gmm_bics)]
print(f"\n  Optimal GMM components (min BIC): {best_gmm_k}")

gmm_final = GaussianMixture(n_components=best_gmm_k, covariance_type="full",
                             random_state=42, n_init=3)
gmm_final.fit(X_pca)
gmm_labels = gmm_final.predict(X_pca)

gmm_metrics = {
    "silhouette": float(silhouette_score(X_pca, gmm_labels)),
    "davies_bouldin": float(davies_bouldin_score(X_pca, gmm_labels)),
    "calinski_harabasz": float(calinski_harabasz_score(X_pca, gmm_labels)),
    "n_clusters": best_gmm_k,
    "bic": float(gmm_final.bic(X_pca)),
    "aic": float(gmm_final.aic(X_pca)),
}
print(f"  GMM final metrics: {gmm_metrics}")

# ─── 8. Select primary model & profile clusters ──────────────────────────────
print("\n" + "=" * 70)
print("STEP 8: Cluster profiling (using K-Means as primary)")
print("=" * 70)

# Use K-Means as the primary clustering for profiling
primary_labels = kmeans_labels
primary_name = "kmeans"
n_clusters_primary = best_k

# Build profiling dataframe
profile_df = X_raw.copy()
profile_df["cluster"] = primary_labels
for col in ["player", "pos", "position", "season", "team", "age_used",
            "market_value", "age_bracket"]:
    profile_df[col] = meta[col].values

# Cluster profiles
RADAR_FEATURES = ["goals_p90", "assists_p90", "goal_contributions_p90",
                   "yellow_cards_p90", "goal_contribution_rate",
                   "minutes_share", "consistency_mean"]
radar_available = [f for f in RADAR_FEATURES if f in available]

cluster_profiles = {}
for c in sorted(profile_df["cluster"].unique()):
    sub = profile_df[profile_df["cluster"] == c]
    profile = {
        "size": int(len(sub)),
        "mean_stats": {col: float(round(sub[col].mean(), 4)) for col in available},
        "dominant_positions": sub["pos"].value_counts().head(3).to_dict(),
        "detailed_positions": sub["position"].dropna().value_counts().head(5).to_dict(),
        "market_value_mean": float(round(sub["market_value"].mean(), 0)),
        "market_value_median": float(round(sub["market_value"].median(), 0)),
        "market_value_min": float(round(sub["market_value"].min(), 0)),
        "market_value_max": float(round(sub["market_value"].max(), 0)),
        "age_mean": float(round(sub["age_used"].mean(), 1)),
        "age_brackets": sub["age_bracket"].value_counts().to_dict(),
        "top_players": sub.nlargest(5, "market_value")[["player", "season", "team",
                                                         "market_value"]].to_dict("records"),
        "seasons": sub["season"].value_counts().head(5).to_dict(),
    }
    cluster_profiles[int(c)] = profile
    print(f"\n  Cluster {c} (n={profile['size']}):")
    print(f"    Positions: {profile['dominant_positions']}")
    print(f"    Market Value: mean={profile['market_value_mean']:,.0f}, "
          f"median={profile['market_value_median']:,.0f}")
    print(f"    Age: {profile['age_mean']:.1f}")
    print(f"    Top players: {[p['player'] for p in profile['top_players']]}")

# ─── 9. Name clusters ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 9: Auto-naming clusters")
print("=" * 70)

def auto_name_cluster(profile, all_profiles):
    """Heuristic naming based on cluster stats."""
    stats = profile["mean_stats"]
    pos = profile["dominant_positions"]
    mv = profile["market_value_mean"]
    age = profile["age_mean"]
    size = profile["size"]

    # Compute global averages for comparison
    global_means = {}
    total = sum(p["size"] for p in all_profiles.values())
    for feat in available:
        global_means[feat] = sum(p["mean_stats"][feat] * p["size"]
                                  for p in all_profiles.values()) / total

    goals = stats.get("goals_p90", 0)
    assists = stats.get("assists_p90", 0)
    gc = stats.get("goal_contributions_p90", 0)
    yellows = stats.get("yellow_cards_p90", 0)
    minutes = stats.get("min", 0)
    consistency = stats.get("consistency_mean", 0)
    mins_share = stats.get("minutes_share", 0)

    top_pos = list(pos.keys())[0] if pos else ""

    # Decision tree for naming
    if gc > global_means.get("goal_contributions_p90", 0) * 1.8 and mv > 15_000_000:
        return "Elite Attackers"
    elif gc > global_means.get("goal_contributions_p90", 0) * 1.5:
        return "Goal Threats"
    elif top_pos == "GK":
        return "Goalkeepers"
    elif top_pos in ("DF", "DF,MF") and yellows > global_means.get("yellow_cards_p90", 0) * 1.1:
        return "Defensive Workhorses"
    elif top_pos in ("DF", "DF,MF"):
        return "Solid Defenders"
    elif age < 24 and mv < global_means.get("market_value", 5e6) * 0.5:
        return "Young Prospects"
    elif mins_share > global_means.get("minutes_share", 0) * 1.3 and consistency > 0:
        return "Reliable Starters"
    elif assists > global_means.get("assists_p90", 0) * 1.5:
        return "Creative Playmakers"
    elif mv > 20_000_000:
        return "High-Value Stars"
    elif age > 30:
        return "Experienced Veterans"
    elif mins_share < global_means.get("minutes_share", 0) * 0.7:
        return "Squad Rotation Players"
    else:
        return "All-Round Midfielders"


cluster_names = {}
for c, profile in cluster_profiles.items():
    name = auto_name_cluster(profile, cluster_profiles)
    # Avoid duplicate names
    if name in cluster_names.values():
        name = name + f" (Group {c})"
    cluster_names[c] = name
    profile["cluster_name"] = name
    print(f"  Cluster {c}: {name}")

# ─── 10. Visualizations ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 10: Creating visualizations")
print("=" * 70)

palette = sns.color_palette("husl", n_clusters_primary)

# 10a. PCA 2D scatter
fig, ax = plt.subplots(figsize=(12, 8))
for c in range(n_clusters_primary):
    mask = primary_labels == c
    ax.scatter(X_pca2d[mask, 0], X_pca2d[mask, 1],
               c=[palette[c]], label=f"C{c}: {cluster_names[c]}",
               alpha=0.5, s=15, edgecolors="none")
ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} var)")
ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} var)")
ax.set_title("Player Clusters - PCA 2D Projection")
ax.legend(fontsize=8, loc="best", markerscale=2)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "clusters_pca_2d.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: clusters_pca_2d.png")

# 10b. UMAP/t-SNE 2D scatter
fig, ax = plt.subplots(figsize=(12, 8))
for c in range(n_clusters_primary):
    mask = primary_labels == c
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=[palette[c]], label=f"C{c}: {cluster_names[c]}",
               alpha=0.5, s=15, edgecolors="none")
ax.set_xlabel(f"{embed_name} 1")
ax.set_ylabel(f"{embed_name} 2")
ax.set_title(f"Player Clusters - {embed_name} 2D Projection")
ax.legend(fontsize=8, loc="best", markerscale=2)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, f"clusters_{embed_name.lower()}_2d.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: clusters_{embed_name.lower()}_2d.png")

# 10c. Cluster size distribution
fig, ax = plt.subplots(figsize=(10, 6))
sizes = [cluster_profiles[c]["size"] for c in sorted(cluster_profiles.keys())]
labels_bar = [f"C{c}\n{cluster_names[c]}" for c in sorted(cluster_profiles.keys())]
bars = ax.bar(labels_bar, sizes, color=palette[:len(sizes)], edgecolor="black", linewidth=0.5)
for bar, size in zip(bars, sizes):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
            str(size), ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Number of Player-Seasons")
ax.set_title("Cluster Size Distribution (K-Means)")
plt.xticks(rotation=30, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cluster_size_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: cluster_size_distribution.png")

# 10d. Radar charts for cluster profiles
def radar_chart(cluster_profiles, features, cluster_names, filename):
    """Create radar chart comparing cluster profiles."""
    n_features = len(features)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    # Compute min/max for normalization
    all_vals = {f: [] for f in features}
    for c, p in cluster_profiles.items():
        for f in features:
            all_vals[f].append(p["mean_stats"].get(f, 0))

    feat_min = {f: min(v) for f, v in all_vals.items()}
    feat_max = {f: max(v) for f, v in all_vals.items()}

    n_clusters = len(cluster_profiles)
    # Arrange in grid
    ncols = min(3, n_clusters)
    nrows = (n_clusters + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows),
                              subplot_kw=dict(polar=True))
    if n_clusters == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, (c, p) in enumerate(sorted(cluster_profiles.items())):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]

        values = []
        for f in features:
            v = p["mean_stats"].get(f, 0)
            rng = feat_max[f] - feat_min[f]
            if rng > 0:
                values.append((v - feat_min[f]) / rng)
            else:
                values.append(0.5)
        values += values[:1]

        ax.fill(angles, values, alpha=0.25, color=palette[c])
        ax.plot(angles, values, "o-", markersize=4, color=palette[c], linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f.replace("_p90", "\n/90").replace("_", "\n") for f in features],
                           fontsize=7)
        ax.set_title(f"C{c}: {cluster_names[c]}\n(n={p['size']})", fontsize=10, pad=20)
        ax.set_ylim(0, 1.1)

    # Hide unused subplots
    for idx in range(n_clusters, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].set_visible(False)

    plt.suptitle("Cluster Profile Radar Charts", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()

radar_chart(cluster_profiles, radar_available, cluster_names, "cluster_radar_charts.png")
print("  Saved: cluster_radar_charts.png")

# 10e. Heatmap of cluster means (normalized)
fig, ax = plt.subplots(figsize=(16, max(4, n_clusters_primary * 0.8)))
heat_data = pd.DataFrame({
    f"C{c}: {cluster_names[c]}": cluster_profiles[c]["mean_stats"]
    for c in sorted(cluster_profiles.keys())
}).T

# Z-score normalize columns for better visualization
heat_norm = (heat_data - heat_data.mean()) / (heat_data.std() + 1e-8)
sns.heatmap(heat_norm, cmap="RdYlBu_r", center=0, annot=False,
            fmt=".1f", linewidths=0.5, ax=ax, xticklabels=True)
ax.set_title("Cluster Feature Profiles (Z-score normalized)")
plt.xticks(rotation=45, ha="right", fontsize=7)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cluster_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: cluster_heatmap.png")

# 10f. Compare all methods on same 2D plot
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# K-Means
for c in range(n_clusters_primary):
    mask = kmeans_labels == c
    axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1], c=[palette[c]], alpha=0.4, s=10)
axes[0].set_title(f"K-Means (K={best_k})\nSil={kmeans_metrics['silhouette']:.3f}")
axes[0].set_xlabel(f"{embed_name} 1")
axes[0].set_ylabel(f"{embed_name} 2")

# DBSCAN
if "error" not in dbscan_metrics:
    unique_db = sorted(set(best_dbscan_labels))
    db_palette = sns.color_palette("husl", len(unique_db))
    for i, c in enumerate(unique_db):
        mask = best_dbscan_labels == c
        color = "gray" if c == -1 else db_palette[i]
        label = "Noise" if c == -1 else f"C{c}"
        axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color], alpha=0.4, s=10, label=label)
    axes[1].set_title(f"DBSCAN (eps={best_dbscan_params.get('eps', '?')})\n"
                      f"Sil={dbscan_metrics.get('silhouette', 0):.3f}")
else:
    axes[1].set_title("DBSCAN (no valid clustering)")
axes[1].set_xlabel(f"{embed_name} 1")
axes[1].set_ylabel(f"{embed_name} 2")

# GMM
gmm_palette = sns.color_palette("husl", best_gmm_k)
for c in range(best_gmm_k):
    mask = gmm_labels == c
    axes[2].scatter(X_2d[mask, 0], X_2d[mask, 1], c=[gmm_palette[c]], alpha=0.4, s=10)
axes[2].set_title(f"GMM (K={best_gmm_k})\nSil={gmm_metrics['silhouette']:.3f}")
axes[2].set_xlabel(f"{embed_name} 1")
axes[2].set_ylabel(f"{embed_name} 2")

plt.suptitle(f"Clustering Methods Comparison ({embed_name} projection)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "clustering_methods_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: clustering_methods_comparison.png")

# ─── 11. Save models & results ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 11: Saving models and results")
print("=" * 70)

# Save models
with open(os.path.join(BASE, "kmeans_model.pkl"), "wb") as f:
    pickle.dump(km_final, f)
with open(os.path.join(BASE, "gmm_model.pkl"), "wb") as f:
    pickle.dump(gmm_final, f)
with open(os.path.join(BASE, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(BASE, "pca_model.pkl"), "wb") as f:
    pickle.dump(pca, f)
print("  Saved: kmeans_model.pkl, gmm_model.pkl, scaler.pkl, pca_model.pkl")

# Save cluster assignments
assignments_df = meta.copy()
assignments_df["kmeans_cluster"] = kmeans_labels
assignments_df["kmeans_cluster_name"] = [cluster_names[c] for c in kmeans_labels]
assignments_df["gmm_cluster"] = gmm_labels
assignments_df["dbscan_cluster"] = best_dbscan_labels
assignments_df.to_parquet(os.path.join(BASE, "cluster_assignments.parquet"), index=False)
print("  Saved: cluster_assignments.parquet")

# Build results summary
# Convert all numpy types to Python native for JSON serialization
def make_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    return obj

results_summary = {
    "pipeline": "P5 Player Clustering",
    "data": {
        "source_features": FEATURES_PATH,
        "source_season": SEASON_PATH,
        "total_records": int(len(df_feat)),
        "after_900min_filter": int(len(df)),
        "after_nan_drop": int(len(X_raw)),
        "features_used": available,
        "n_features": len(available),
    },
    "pca": {
        "n_components": n_components,
        "explained_variance_90pct": float(cumvar[n_components - 1]),
        "total_components": len(pca_full.explained_variance_ratio_),
    },
    "embedding": embed_name,
    "kmeans": {
        "metrics": kmeans_metrics,
        "cluster_profiles": make_serializable(cluster_profiles),
        "cluster_names": make_serializable(cluster_names),
    },
    "dbscan": {
        "metrics": make_serializable(dbscan_metrics),
    },
    "gmm": {
        "metrics": make_serializable(gmm_metrics),
    },
    "figures": [
        "pca_explained_variance.png",
        "kmeans_elbow_silhouette.png",
        "dbscan_kdistance.png",
        "gmm_bic_aic.png",
        "clusters_pca_2d.png",
        f"clusters_{embed_name.lower()}_2d.png",
        "cluster_size_distribution.png",
        "cluster_radar_charts.png",
        "cluster_heatmap.png",
        "clustering_methods_comparison.png",
    ],
}

with open(os.path.join(BASE, "results_summary.json"), "w") as f:
    json.dump(results_summary, f, indent=2, default=str)
print("  Saved: results_summary.json")

# ─── Done ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("P5 CLUSTERING PIPELINE COMPLETE")
print("=" * 70)
print(f"\n  Primary model: K-Means (K={best_k})")
print(f"  Silhouette: {kmeans_metrics['silhouette']:.4f}")
print(f"  Davies-Bouldin: {kmeans_metrics['davies_bouldin']:.4f}")
print(f"  Calinski-Harabasz: {kmeans_metrics['calinski_harabasz']:.1f}")
print(f"\n  Cluster names:")
for c, name in sorted(cluster_names.items()):
    print(f"    C{c}: {name} (n={cluster_profiles[c]['size']})")
print(f"\n  Outputs in: {BASE}")
print(f"  Figures in: {FIG_DIR}")

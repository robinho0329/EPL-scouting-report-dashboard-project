"""
S3 v3: Scout Player Similarity Engine — Bug Fixes
EPL Data Project

Changes from v2:
  1. CLUSTER DEDUPLICATION: After K-Means, merge clusters whose centroids share
     cosine similarity > 0.85. Duplicate archetype names (e.g., three "Attacking
     Full-Back") are eliminated by either merging or re-labelling based on the
     dominant stat differentiator between the near-identical clusters.
  2. BARGAIN FILTER: Minimum 900 minutes enforced for every player that appears
     in bargain or recommendation lists (fixes Origi-style tiny-sample p90 noise).
  3. REPLACEMENT SEARCH: New `find_replacement(player, season, max_budget_eur,
     max_age, min_minutes=900)` function plus 5 demo searches in results_summary.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import umap

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
# Path setup
# ──────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent.parent
DATA_DIR  = BASE_DIR / "data" / "processed"
SCOUT_DIR = BASE_DIR / "data" / "scout"
FIG_DIR   = Path(__file__).resolve().parent / "figures_v3"
MODEL_DIR = Path(__file__).resolve().parent
S2_JSON   = BASE_DIR / "models" / "s2_market_value" / "results_summary.json"

SCOUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_SEASON = "2024/25"

# Cosine-similarity threshold above which two cluster centroids are treated as
# duplicates and merged into one.
MERGE_COSINE_THRESHOLD = 0.85

# Minimum minutes for bargain list and replacement search
MIN_MINUTES_BARGAIN = 900


def _season_to_year(s: str) -> int:
    """'2023/24' -> 2024 (ending year)"""
    try:
        return int(s.split('/')[0]) + 1
    except Exception:
        return 2000


CURRENT_YEAR = _season_to_year(CURRENT_SEASON)


# ──────────────────────────────────────────────────────────────
# 1. Data load and feature engineering  (unchanged from v2)
# ──────────────────────────────────────────────────────────────

def load_and_engineer_features():
    print("[1/7] Loading data...")
    season_df = pd.read_parquet(DATA_DIR / "player_season_stats.parquet")
    match_df  = pd.read_parquet(DATA_DIR / "player_match_logs.parquet")
    feat_df   = pd.read_parquet(BASE_DIR / "data" / "features" / "player_features.parquet")

    agg_cols = {
        'sh': 'sum', 'sot': 'sum', 'tklw': 'sum', 'int': 'sum',
        'crs': 'sum', 'fls': 'sum', 'fld': 'sum',
        'min': 'sum', 'gls': 'sum', 'ast': 'sum',
    }
    match_agg = (
        match_df.groupby(['player', 'season', 'team'])
        .agg({k: 'sum' for k in agg_cols})
        .reset_index()
    )
    match_agg.columns = ['player', 'season', 'team'] + [f'ml_{c}' for c in agg_cols]

    df = season_df.merge(match_agg, on=['player', 'season', 'team'], how='left')

    aux_cols = ['player', 'season', 'team',
                'goal_contribution_rate', 'consistency_cv', 'epl_experience',
                'minutes_share', 'n_matches', 'versatility_positions']
    aux = feat_df[[c for c in aux_cols if c in feat_df.columns]].drop_duplicates(
        subset=['player', 'season', 'team'])
    df = df.merge(aux, on=['player', 'season', 'team'], how='left')

    df['90s_safe'] = df['90s'].fillna(0).replace(0, np.nan)

    df['goals_p90']      = df['gls']     / df['90s_safe']
    df['assists_p90']    = df['ast']     / df['90s_safe']
    df['shots_p90']      = df['ml_sh']   / df['90s_safe']
    df['sot_p90']        = df['ml_sot']  / df['90s_safe']
    df['tackles_p90']    = df['ml_tklw'] / df['90s_safe']
    df['interc_p90']     = df['ml_int']  / df['90s_safe']
    df['crosses_p90']    = df['ml_crs']  / df['90s_safe']
    df['fouls_p90']      = df['ml_fls']  / df['90s_safe']
    df['key_passes_p90'] = df['crosses_p90']

    df['g_plus_a_p90']   = df['goals_p90'] + df['assists_p90']
    df['def_actions_p90'] = df['tackles_p90'] + df['interc_p90']
    df['shot_conversion'] = np.where(
        df['ml_sh'].fillna(0) > 0,
        df['gls'] / df['ml_sh'].replace(0, np.nan),
        0.0
    )

    df['starter_ratio'] = np.where(
        df['mp'] > 0, df['starts'].fillna(0) / df['mp'], 0.0)

    pos_map = {
        'Centre-Forward': 'FW', 'Second Striker': 'FW', 'Striker': 'FW',
        'Right Winger': 'AM', 'Left Winger': 'AM', 'Attacking Midfield': 'AM',
        'Left Midfield': 'AM', 'Right Midfield': 'AM',
        'Central Midfield': 'CM', 'Defensive Midfield': 'DM', 'Midfielder': 'CM',
        'Right-Back': 'DEF', 'Left-Back': 'DEF', 'Centre-Back': 'DEF', 'Defender': 'DEF',
        'Goalkeeper': 'GK',
    }
    df['pos_group'] = df['position'].map(pos_map).fillna('Unknown')

    df = df[df['90s'].fillna(0) >= 1.0].copy()

    df['age_raw']          = df['age'].fillna(df['age_tm']).fillna(25.0)
    df['age_filled']       = df['age_raw']
    df['height_cm']        = df['height_cm'].fillna(
        df.groupby('pos_group')['height_cm'].transform('median')).fillna(180.0)
    df['market_value_raw'] = df['market_value'].fillna(0.0)
    df['market_value_log'] = np.log1p(df['market_value_raw'])

    # total minutes from match logs (more reliable than season_stats 'min')
    df['min'] = df['ml_min'].fillna(df['min']).fillna(0.0)

    df['season_year']    = df['season'].apply(_season_to_year)
    df['seasons_ago']    = CURRENT_YEAR - df['season_year']
    df['recency_weight'] = np.where(
        df['seasons_ago'] <= 2, 1.3,
        np.where(df['seasons_ago'] <= 4, 1.0, 0.7)
    )

    for col in ['goal_contribution_rate', 'consistency_cv', 'epl_experience', 'minutes_share']:
        if col in df.columns:
            df[col] = df[col].fillna(
                df.groupby('pos_group')[col].transform('median')).fillna(0.0)

    FEATURE_COLS = [
        'goals_p90', 'assists_p90', 'shots_p90', 'sot_p90',
        'tackles_p90', 'interc_p90', 'key_passes_p90',
        'g_plus_a_p90', 'def_actions_p90', 'shot_conversion',
        'age_filled', 'height_cm', 'starter_ratio',
        'market_value_log',
        'goal_contribution_rate',
        'consistency_cv',
        'gls', 'ast',
    ]

    for col in FEATURE_COLS:
        if col in df.columns:
            med = df.groupby('pos_group')[col].transform('median')
            df[col] = df[col].fillna(med).fillna(0.0)
        else:
            df[col] = 0.0

    df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], 0.0)

    print(f"   Player-seasons after filter: {len(df):,}")
    return df, FEATURE_COLS


# ──────────────────────────────────────────────────────────────
# 2. Normalization  (unchanged from v2)
# ──────────────────────────────────────────────────────────────

def normalize_features(df, feature_cols):
    print("[2/7] Position-group normalization + recency weighting...")
    df = df.copy()
    scaled_data = np.zeros((len(df), len(feature_cols)))

    for pos_group in df['pos_group'].unique():
        mask = (df['pos_group'] == pos_group).values
        if mask.sum() < 2:
            continue
        scaler = StandardScaler()
        scaled_data[mask] = scaler.fit_transform(df.loc[mask, feature_cols])

    rw = df['recency_weight'].values.reshape(-1, 1)
    scaled_data = scaled_data * rw

    scaled_cols = [f'sc_{c}' for c in feature_cols]
    df[scaled_cols] = scaled_data

    pos_dummies = pd.get_dummies(df['pos_group'], prefix='pos').astype(float) * 2.5
    pos_oh_cols = pos_dummies.columns.tolist()
    df = pd.concat([df.reset_index(drop=True), pos_dummies.reset_index(drop=True)], axis=1)

    all_scaled_cols = scaled_cols + pos_oh_cols
    return df, all_scaled_cols


# ──────────────────────────────────────────────────────────────
# 3. K-Means + centroid-merge deduplication  [FIX #1]
# ──────────────────────────────────────────────────────────────

def run_kmeans(df, scaled_cols):
    """
    K=10..18 sweep, pick best silhouette K.
    Then merge clusters whose centroids have cosine-similarity > MERGE_COSINE_THRESHOLD
    to eliminate duplicate archetype names.
    """
    print(f"[3/7] K-Means clustering (K=10..18, merge threshold={MERGE_COSINE_THRESHOLD})...")
    X = df[scaled_cols].values

    best_k, best_score, best_labels, best_km = 10, -1, None, None
    scores = {}
    for k in range(10, 19):
        km = KMeans(n_clusters=k, random_state=42, n_init=15, max_iter=500)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels, sample_size=min(5000, len(X)), random_state=42)
        scores[k] = score
        print(f"   K={k}: silhouette = {score:.4f}")
        if score > best_score:
            best_k, best_score, best_labels, best_km = k, score, labels, km

    print(f"   Best K = {best_k} (silhouette: {best_score:.4f})")

    # ── Centroid-merge step ────────────────────────────────────
    centroids = best_km.cluster_centers_          # shape (best_k, n_features)
    cos_sim   = cosine_similarity(centroids)       # (best_k, best_k)
    np.fill_diagonal(cos_sim, 0.0)

    # Build union-find to group clusters that are too similar
    parent = list(range(best_k))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    merge_pairs = []
    for i in range(best_k):
        for j in range(i + 1, best_k):
            if cos_sim[i, j] > MERGE_COSINE_THRESHOLD:
                union(i, j)
                merge_pairs.append((i, j, float(cos_sim[i, j])))

    # Map old cluster ids → new cluster ids (canonical root)
    old_to_new = {}
    roots_seen = {}
    new_id = 0
    for old in range(best_k):
        root = find(old)
        if root not in roots_seen:
            roots_seen[root] = new_id
            new_id += 1
        old_to_new[old] = roots_seen[root]

    n_merged = best_k - new_id
    if n_merged > 0:
        print(f"   Merged {n_merged} near-duplicate cluster(s) (pairs: {merge_pairs})")
        print(f"   Cluster count after merge: {new_id}")
    else:
        print("   No clusters needed merging.")

    merged_labels = np.array([old_to_new[lbl] for lbl in best_labels])
    df['cluster'] = merged_labels

    return df, best_k, new_id, scores, merge_pairs


# ──────────────────────────────────────────────────────────────
# 4. PCA / UMAP  (unchanged from v2)
# ──────────────────────────────────────────────────────────────

def run_embeddings(df, scaled_cols):
    print("[4/7] PCA/UMAP 2D embedding...")
    X = df[scaled_cols].values

    pca = PCA(n_components=2, random_state=42)
    pca_emb = pca.fit_transform(X)
    df['pca_x'] = pca_emb[:, 0]
    df['pca_y'] = pca_emb[:, 1]
    print(f"   PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=20, min_dist=0.05)
    umap_emb = reducer.fit_transform(X)
    df['umap_x'] = umap_emb[:, 0]
    df['umap_y'] = umap_emb[:, 1]

    return df, pca


# ──────────────────────────────────────────────────────────────
# 5. Archetype naming  [FIX #1 — unique names via stat tiebreak]
# ──────────────────────────────────────────────────────────────

def _stat_tiebreak(prof1: dict, prof2: dict, name: str):
    """
    Given two clusters that would both get `name`, differentiate them using
    the most discriminating per-90 stat between the pair.

    Returns (name_for_cluster_1, name_for_cluster_2).
    """
    # Differentiator priority list per base name
    differentiator_map = {
        "Attacking Full-Back": [
            ("goals_p90",      "High-Output",  "Wide"),
            ("assists_p90",    "Creative",     "Defensive"),
            ("def_actions_p90","Press-Heavy",  "Carrying"),
        ],
        "Ball-Winning Defender": [
            ("height_cm",      "Aerial",       "Compact"),
            ("interc_p90",     "Reading",      "Aggressive"),
            ("tackles_p90",    "Tenacious",    "Positional"),
        ],
        "Creative Winger": [
            ("goals_p90",      "Goal-Threat",  "Chance-Creating"),
            ("assists_p90",    "Assist-Focused","Dribbling"),
        ],
        "Goal Poacher": [
            ("shots_p90",      "High-Volume",  "Efficient"),
            ("assists_p90",    "Link-Up",      "Pure"),
        ],
        "Central Midfielder": [
            ("def_actions_p90","Industrious",  "Technical"),
            ("assists_p90",    "Progressive",  "Pressing"),
        ],
        "Defensive Midfielder": [
            ("tackles_p90",    "Disruptive",   "Holding"),
            ("interc_p90",     "Intercepting", "Screening"),
        ],
    }

    differentiators = differentiator_map.get(name, [
        ("goals_p90",      "Offensive",  "Balanced"),
        ("def_actions_p90","Defensive",  "Technical"),
    ])

    best_diff, best_pair = 0, None
    for stat, tag_high, tag_low in differentiators:
        v1 = prof1.get(stat, 0)
        v2 = prof2.get(stat, 0)
        diff = abs(v1 - v2)
        if diff > best_diff:
            best_diff = diff
            if v1 >= v2:
                best_pair = (f"{tag_high} {name}", f"{tag_low} {name}")
            else:
                best_pair = (f"{tag_low} {name}", f"{tag_high} {name}")

    if best_pair is None:
        return f"{name} A", f"{name} B"
    return best_pair


def assign_archetypes(df, scaled_cols):
    """
    Name each cluster with a unique archetype string.
    If two clusters produce the same base name, use stat tiebreaking
    to differentiate them — so the final archetype dict has NO duplicates.
    """
    print("[5/7] Assigning unique archetype names...")

    profile_cols = [
        'goals_p90', 'assists_p90', 'shots_p90',
        'tackles_p90', 'interc_p90', 'key_passes_p90',
        'age_filled', 'height_cm', 'market_value_log',
        'starter_ratio', 'gls', 'ast', 'def_actions_p90', 'g_plus_a_p90'
    ]

    global_p75 = df[profile_cols].quantile(0.75)

    cluster_profiles = {}
    raw_names = {}        # cluster_id -> base name before dedup

    for c in sorted(df['cluster'].unique()):
        mask = df['cluster'] == c
        sub  = df[mask]
        prof = sub[profile_cols].mean().to_dict()
        prof['count']    = int(mask.sum())
        prof['top_pos']  = sub['pos_group'].value_counts().idxmax()
        prof['pos_dist'] = sub['pos_group'].value_counts(normalize=True).round(3).to_dict()
        cluster_profiles[int(c)] = prof

        pos_dist   = sub['pos_group'].value_counts(normalize=True)
        top_pos    = pos_dist.idxmax()
        top_frac   = pos_dist.max()
        second_pos = pos_dist.index[1] if len(pos_dist) > 1 else top_pos

        goals   = prof['goals_p90']
        tackles = prof['tackles_p90']
        assists = prof['assists_p90']
        height  = prof['height_cm']
        kp      = prof['key_passes_p90']
        interc  = prof['interc_p90']

        high_goals   = goals   >= global_p75['goals_p90']
        high_tackles = tackles >= global_p75['tackles_p90']
        high_assists = assists >= global_p75['assists_p90']
        high_interc  = interc  >= global_p75['interc_p90']
        high_kp      = kp      >= global_p75['key_passes_p90']
        tall         = height  >= 185.0

        if top_pos == 'GK':
            name = "Goalkeeper"
        elif top_pos == 'DEF':
            if top_frac < 0.6 and second_pos == 'AM':
                name = "Attacking Full-Back"
            elif high_goals or high_assists:
                name = "Attacking Full-Back"
            elif high_tackles and high_interc:
                name = "Ball-Winning Defender"
            elif tall and high_interc:
                name = "Aerial Defender"
            elif 'Left-Back' in sub['position'].value_counts().index[:2] or \
                 'Right-Back' in sub['position'].value_counts().index[:2]:
                name = "Modern Full-Back"
            else:
                name = "Centre-Back"
        elif top_pos == 'DM':
            if high_tackles:
                name = "Ball-Winning Midfielder"
            elif high_assists or high_kp:
                name = "Deep-Lying Playmaker"
            else:
                name = "Defensive Midfielder"
        elif top_pos == 'CM':
            if high_assists or high_kp:
                name = "Creative Playmaker"
            elif high_tackles:
                name = "Box-to-Box Midfielder"
            else:
                name = "Central Midfielder"
        elif top_pos == 'AM':
            if high_goals and high_assists:
                name = "Attacking Wide Forward"
            elif high_goals:
                name = "Goal-Scoring Winger"
            elif high_assists or high_kp:
                name = "Creative Winger"
            else:
                name = "Wide Midfielder"
        elif top_pos == 'FW':
            if high_goals:
                name = "Goal Poacher"
            elif high_assists:
                name = "Advanced Forward"
            elif tall:
                name = "Target Man"
            else:
                name = "Centre-Forward"
        else:
            name = f"Mixed Role (C{c})"

        raw_names[int(c)] = name

    # ── Deduplicate names using stat tiebreaking ───────────────
    # Group clusters by base name
    name_to_clusters = defaultdict(list)
    for cid, name in raw_names.items():
        name_to_clusters[name].append(cid)

    archetype_names = dict(raw_names)  # start with raw, then patch dupes

    for base_name, cluster_ids in name_to_clusters.items():
        if len(cluster_ids) == 1:
            continue  # unique — no action needed

        print(f"   Duplicate base name '{base_name}' in clusters {cluster_ids} — applying stat tiebreak")

        # Sort by cluster_id for determinism, then rename pairwise
        cluster_ids_sorted = sorted(cluster_ids)

        # Iterative two-way split: pick the two most different and split,
        # then if a third appears assign a letter suffix
        if len(cluster_ids_sorted) == 2:
            c1, c2 = cluster_ids_sorted
            n1, n2 = _stat_tiebreak(cluster_profiles[c1], cluster_profiles[c2], base_name)
            archetype_names[c1] = n1
            archetype_names[c2] = n2
        else:
            # 3+ duplicates: use High/Mid/Low ordering on the most variable stat
            # Find the stat with highest variance across the group
            stats_to_check = [
                'goals_p90', 'assists_p90', 'tackles_p90',
                'interc_p90', 'def_actions_p90', 'shots_p90',
            ]
            variances = {}
            for stat in stats_to_check:
                vals = [cluster_profiles[cid].get(stat, 0) for cid in cluster_ids_sorted]
                variances[stat] = np.var(vals)
            best_stat = max(variances, key=variances.get)

            vals = [(cluster_profiles[cid].get(best_stat, 0), cid) for cid in cluster_ids_sorted]
            vals_sorted = sorted(vals, reverse=True)
            suffixes = ["High-Volume", "Standard", "Low-Block", "Deep", "Compact"]
            for rank, (_, cid) in enumerate(vals_sorted):
                suffix = suffixes[rank] if rank < len(suffixes) else f"Type{rank+1}"
                archetype_names[cid] = f"{suffix} {base_name}"

    # Final uniqueness check & fallback
    seen = {}
    for cid, name in list(archetype_names.items()):
        if name in seen:
            archetype_names[cid] = f"{name} II"
        else:
            seen[name] = cid

    for c in sorted(df['cluster'].unique()):
        prof = cluster_profiles[int(c)]
        name = archetype_names[int(c)]
        print(f"   C{c}: {name} (n={prof['count']}, top_pos={prof['top_pos']}, "
              f"G/90={prof['goals_p90']:.2f}, A/90={prof['assists_p90']:.2f}, "
              f"Tkl/90={prof['tackles_p90']:.2f})")

    return cluster_profiles, archetype_names


# ──────────────────────────────────────────────────────────────
# 6. Top-K similarity matrix  (unchanged from v2)
# ──────────────────────────────────────────────────────────────

def build_topk_similarity(df, scaled_cols, top_k=15):
    print("[6/7] Building top-K similarity matrix (within position, k=15)...")
    records = []

    for pos_group in df['pos_group'].unique():
        mask = df['pos_group'] == pos_group
        sub  = df[mask].reset_index(drop=True)
        if len(sub) < 2:
            continue

        X       = sub[scaled_cols].values
        sim_mat = cosine_similarity(X)
        np.fill_diagonal(sim_mat, -1)

        for i in range(len(sub)):
            top_idx = np.argsort(sim_mat[i])[::-1][:top_k]
            for rank, j in enumerate(top_idx):
                records.append({
                    'player':     sub.loc[i, 'player'],
                    'season':     sub.loc[i, 'season'],
                    'neighbor':   sub.loc[j, 'player'],
                    'nbr_season': sub.loc[j, 'season'],
                    'pos_group':  pos_group,
                    'rank':       rank + 1,
                    'cosine_sim': float(sim_mat[i, j]),
                })

    sim_df = pd.DataFrame(records)
    print(f"   Top-K neighbour rows: {len(sim_df):,}")
    return sim_df


# ──────────────────────────────────────────────────────────────
# 7. Core query function  (unchanged from v2, min_minutes=900 default)
# ──────────────────────────────────────────────────────────────

def query(
    df,
    scaled_cols,
    player_name,
    season,
    max_age=None,
    max_value=None,
    min_minutes=900,
    top_n=10,
    same_pos_only=True,
):
    """
    Scout similarity query with market value / age / minutes filters.
    """
    mask_name   = df['player'].str.contains(player_name, case=False, na=False)
    mask_season = df['season'] == season
    query_rows  = df[mask_name & mask_season]

    if len(query_rows) == 0:
        query_rows = df[mask_name].sort_values('season', ascending=False)
        if len(query_rows) == 0:
            print(f"  [WARNING] Player '{player_name}' not found.")
            return pd.DataFrame()
        used = query_rows.iloc[0]['season']
        print(f"  [WARNING] Season {season} not found -> using latest ({used})")

    query_row = query_rows.iloc[0]
    query_vec = query_row[scaled_cols].values.reshape(1, -1)
    query_pos = query_row['pos_group']

    compare_df = df[
        ~(mask_name & (df['season'] == query_row['season']))
    ].copy()

    if same_pos_only:
        compare_df = compare_df[compare_df['pos_group'] == query_pos]

    compare_df = compare_df[compare_df['min'].fillna(0) >= min_minutes]

    if max_age is not None:
        compare_df = compare_df[compare_df['age_raw'].fillna(99) <= max_age]

    if max_value is not None:
        compare_df = compare_df[
            (compare_df['market_value_raw'].fillna(0) <= max_value) |
            (compare_df['market_value_raw'].fillna(0) == 0)
        ]

    if len(compare_df) == 0:
        print("  [WARNING] No players match the filter criteria.")
        return pd.DataFrame()

    compare_X  = compare_df[scaled_cols].values
    sim_scores = cosine_similarity(query_vec, compare_X)[0]
    compare_df = compare_df.copy()
    compare_df['similarity'] = sim_scores

    result = compare_df.nlargest(top_n, 'similarity')

    out_cols = [
        'player', 'season', 'team', 'position', 'pos_group',
        'age_raw', 'market_value_raw',
        'min',
        'gls', 'ast', 'goals_p90', 'assists_p90',
        'tackles_p90', 'interc_p90',
        'similarity', 'cluster', 'archetype'
    ]
    out_cols = [c for c in out_cols if c in result.columns]
    return result[out_cols].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# 8. find_replacement  [FIX #3 — dedicated replacement function]
# ──────────────────────────────────────────────────────────────

def find_replacement(
    df,
    scaled_cols,
    player: str,
    season: str,
    max_budget_eur: float,
    max_age=None,
    min_minutes: int = 900,
    top_n: int = 8,
):
    """
    Find the most statistically similar players who fit the scouting constraints.

    Parameters
    ----------
    player         : player name (partial match, case-insensitive)
    season         : reference season string e.g. '2023/24'
    max_budget_eur : hard budget cap in EUR (market_value_raw <= this)
    max_age        : optional age ceiling (inclusive)
    min_minutes    : minimum minutes played in that season (default 900)
    top_n          : number of candidates to return

    Returns
    -------
    dict with keys: query_player, query_season, filters, results (list of dicts)
    """
    label_parts = [f"€{max_budget_eur/1e6:.0f}M budget"]
    if max_age is not None:
        label_parts.append(f"age ≤{max_age}")
    label_parts.append(f"≥{min_minutes} min")
    label = f"Replace {player} ({season}) — {', '.join(label_parts)}"

    res = query(
        df=df,
        scaled_cols=scaled_cols,
        player_name=player,
        season=season,
        max_age=max_age,
        max_value=max_budget_eur,
        min_minutes=min_minutes,
        top_n=top_n,
        same_pos_only=True,
    )

    if res.empty:
        print(f"  No replacements found for {label}")
        return {"label": label, "query_player": player,
                "query_season": season,
                "filters": {"max_budget_eur": max_budget_eur,
                             "max_age": max_age,
                             "min_minutes": min_minutes},
                "results": []}

    # Print table
    display = res[[
        'player', 'season', 'team', 'position',
        'age_raw', 'market_value_raw', 'min',
        'goals_p90', 'assists_p90', 'similarity', 'archetype'
    ]].copy()
    display.columns = [
        'Player', 'Season', 'Team', 'Position',
        'Age', 'Market Value', 'Minutes',
        'G/90', 'A/90', 'Similarity', 'Archetype'
    ]
    display['Market Value'] = display['Market Value'].apply(
        lambda x: f"€{x/1e6:.1f}M" if x > 0 else "N/A"
    )
    display['Minutes'] = display['Minutes'].apply(lambda x: f"{int(x)}")
    display['G/90'] = display['G/90'].round(3)
    display['A/90'] = display['A/90'].round(3)
    display['Similarity'] = display['Similarity'].round(4)
    display.index = range(1, len(display) + 1)

    print(f"\n{'─'*68}")
    print(f"  {label}")
    print(f"{'─'*68}")
    print(display.to_string())

    records = []
    for _, r in res.iterrows():
        records.append({
            "player":           r['player'],
            "season":           r['season'],
            "team":             r.get('team', ''),
            "position":         str(r.get('position', '')),
            "age":              float(r.get('age_raw', 0)),
            "market_value_eur": float(r.get('market_value_raw', 0)),
            "minutes":          int(r.get('min', 0)),
            "goals_p90":        round(float(r['goals_p90']), 3),
            "assists_p90":      round(float(r['assists_p90']), 3),
            "similarity":       round(float(r['similarity']), 4),
            "archetype":        str(r.get('archetype', '')),
        })

    return {
        "label":        label,
        "query_player": player,
        "query_season": season,
        "filters": {
            "max_budget_eur": max_budget_eur,
            "max_age":        max_age,
            "min_minutes":    min_minutes,
        },
        "results": records,
    }


# ──────────────────────────────────────────────────────────────
# 9. Bargain list  [FIX #2 — 900-minute minimum enforced]
# ──────────────────────────────────────────────────────────────

def build_bargain_list(df, archetype_names, top_n=5):
    """
    Best-value players per cluster.

    FIX: Only players with >= MIN_MINUTES_BARGAIN (900) minutes are eligible.
    This prevents tiny-sample per-90 artifacts (e.g., Origi on 90 minutes).

    bargain_score = (goals_p90 + assists_p90 + def_actions_p90/3) / log1p(market_value_raw)
    Adjusted by S2 value_score bonus.
    Scope: recent seasons (2021/22+), players with known market value.
    """
    print(f"[bargain] Building per-cluster bargain list (min {MIN_MINUTES_BARGAIN} min)...")

    s2_uv_map = {}
    if S2_JSON.exists():
        with open(S2_JSON, 'r', encoding='utf-8') as f:
            s2 = json.load(f)
        for entry in s2.get('top20_undervalued', []):
            key = (entry['player'], entry['season'])
            s2_uv_map[key] = entry.get('value_score', 1.0)

    recent_seasons = ['2021/22', '2022/23', '2023/24', '2024/25']
    sub = df[
        (df['season'].isin(recent_seasons)) &
        (df['market_value_raw'].fillna(0) > 0) &
        (df['min'].fillna(0) >= MIN_MINUTES_BARGAIN)          # ← FIX #2
    ].copy()

    sub['perf_score'] = (
        sub['goals_p90'].fillna(0) +
        sub['assists_p90'].fillna(0) +
        sub['def_actions_p90'].fillna(0) / 3
    )
    sub['bargain_score'] = sub['perf_score'] / np.log1p(sub['market_value_raw'])
    sub['s2_value_score'] = sub.apply(
        lambda r: s2_uv_map.get((r['player'], r['season']), 1.0), axis=1)
    sub['bargain_score_adj'] = sub['bargain_score'] * np.log1p(sub['s2_value_score'])

    bargain_dict = {}
    for c in sorted(sub['cluster'].unique()):
        mask = sub['cluster'] == c
        top  = sub[mask].nlargest(top_n, 'bargain_score_adj')
        name = archetype_names.get(int(c), f"Cluster {c}")

        records = []
        for idx, row in top.iterrows():
            records.append({
                'rank':             len(records) + 1,
                'player':           row['player'],
                'season':           row['season'],
                'team':             row['team'],
                'position':         row.get('position', ''),
                'age':              float(row.get('age_raw', 0)),
                'minutes':          int(row.get('min', 0)),
                'market_value_eur': float(row['market_value_raw']),
                'goals_p90':        round(float(row['goals_p90']), 3),
                'assists_p90':      round(float(row['assists_p90']), 3),
                'def_actions_p90':  round(float(row.get('def_actions_p90', 0)), 3),
                'bargain_score':    round(float(row['bargain_score_adj']), 4),
                's2_value_score':   round(float(row['s2_value_score']), 3),
            })
        bargain_dict[int(c)] = {'archetype': name, 'players': records}
        names_list = [f"{r['player']} ({r['minutes']}min)" for r in records]
        print(f"   C{c} ({name}): {names_list}")

    return bargain_dict


# ──────────────────────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────────────────────

def plot_umap_scatter(df, archetype_names):
    fig, ax = plt.subplots(figsize=(16, 11))
    n_clusters = df['cluster'].nunique()
    palette    = sns.color_palette("tab20", n_clusters)

    for i, c in enumerate(sorted(df['cluster'].unique())):
        mask = df['cluster'] == c
        name = archetype_names.get(int(c), f"Cluster {c}")
        ax.scatter(
            df.loc[mask, 'umap_x'], df.loc[mask, 'umap_y'],
            c=[palette[i]], label=f"C{c}: {name}",
            alpha=0.45, s=10, linewidths=0
        )

    ax.set_title("UMAP 2D — Player Cluster Distribution v3", fontsize=14)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=7)
    plt.tight_layout()
    path = FIG_DIR / "umap_v3.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   UMAP saved: {path}")


def plot_replacement_results(label, result_df, fig_name):
    if result_df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    colors  = sns.color_palette("Blues_r", len(result_df))
    labels  = [f"{r['player']} ({r['season']})" for _, r in result_df.iterrows()]
    scores  = result_df['similarity'].values
    bars    = ax.barh(labels[::-1], scores[::-1], color=colors)
    ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Cosine Similarity")
    ax.set_title(label, fontsize=10)
    plt.tight_layout()
    path = FIG_DIR / fig_name
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Figure saved: {path}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 68)
    print("S3 v3: Player Similarity Engine — Duplicate / Bargain / Budget fixes")
    print("=" * 68)

    df, feature_cols = load_and_engineer_features()
    df, scaled_cols  = normalize_features(df, feature_cols)

    df, best_k, n_final_clusters, sil_scores, merge_pairs = run_kmeans(df, scaled_cols)
    df, pca_model = run_embeddings(df, scaled_cols)

    cluster_profiles, archetype_names = assign_archetypes(df, scaled_cols)
    df['archetype'] = df['cluster'].map(archetype_names)

    # ── pos_group vs archetype 정합성 검증 및 자동 수정 ──
    # K-Means 클러스터 경계에서 포지션 불일치 선수를 stats 기반으로 재분류
    POS_VALID = {
        "GK":  {"Goalkeeper"},
        "DEF": {"Aerial Ball-Winning Defender", "Compact Ball-Winning Defender",
                "Modern Full-Back", "High-Volume Attacking Full-Back",
                "Standard Attacking Full-Back", "Low-Block Attacking Full-Back",
                "Attacking Full-Back", "Ball-Winning Defender"},
        "DM":  {"Defensive Midfielder", "Ball-Winning Midfielder", "Central Midfielder"},
        "CM":  {"Central Midfielder", "Ball-Winning Midfielder", "Creative Playmaker",
                "Defensive Midfielder"},
        "AM":  {"Attacking Wide Forward", "Assist-Focused Creative Winger",
                "Dribbling Creative Winger", "Creative Playmaker"},
        "FW":  {"Attacking Wide Forward", "Efficient Goal Poacher", "High-Volume Goal Poacher",
                "Assist-Focused Creative Winger", "Dribbling Creative Winger"},
    }

    def _pos_fix(row):
        pos = row.get("pos_group", "")
        arch = row.get("archetype", "")
        valid = POS_VALID.get(pos)
        if valid is None or arch in valid:
            return arch
        # DEF: stats 기반 세분화
        if pos == "DEF":
            goals   = row.get("goals_p90", 0) or 0
            assists = row.get("assists_p90", 0) or 0
            tackles = row.get("tackles_p90", 0) or 0
            if goals > 0.10 or assists > 0.10:
                return "Attacking Full-Back"
            elif tackles > 1.5:
                return "Ball-Winning Defender"
            return "Modern Full-Back"
        if pos == "GK":
            return "Goalkeeper"
        # CM/DM/AM/FW 폴백
        fallback = {
            "DM": "Defensive Midfielder",
            "CM": "Central Midfielder",
            "AM": "Assist-Focused Creative Winger",
            "FW": "Efficient Goal Poacher",
        }
        return fallback.get(pos, arch)

    before = (df["pos_group"] != df["archetype"].map(
        lambda a: next((p for p, s in POS_VALID.items() if a in s), "?")
    )).sum()
    df["archetype"] = df.apply(_pos_fix, axis=1)
    after_mismatch = sum(
        1 for _, r in df.iterrows()
        if POS_VALID.get(r["pos_group"]) and r["archetype"] not in POS_VALID.get(r["pos_group"], set())
    )
    logging.info(f"[S3] archetype pos mismatch 수정: {before}건 → 잔여 {after_mismatch}건")

    sim_df = build_topk_similarity(df, scaled_cols, top_k=15)

    # ── Save parquet outputs ──────────────────────────────────
    save_cols = [
        'player', 'season', 'team', 'position', 'pos_group',
        'cluster', 'archetype',
        'pca_x', 'pca_y', 'umap_x', 'umap_y',
        'gls', 'ast', 'goals_p90', 'assists_p90',
        'shots_p90', 'tackles_p90', 'interc_p90', 'key_passes_p90',
        'def_actions_p90', 'g_plus_a_p90',
        'age_raw', 'height_cm', 'market_value_raw', 'market_value_log',
        'starter_ratio', 'min',
    ]
    save_cols = [c for c in save_cols if c in df.columns]

    cluster_path = SCOUT_DIR / "cluster_assignments_v3.parquet"
    df[save_cols].to_parquet(cluster_path, index=False)
    print(f"\nCluster assignments saved: {cluster_path}")

    sim_path = SCOUT_DIR / "similarity_matrix_v3.parquet"
    sim_df.to_parquet(sim_path, index=False)
    print(f"Similarity matrix saved: {sim_path}")

    # ── Visualisation ─────────────────────────────────────────
    print("[7/7] Generating visualisations...")
    plot_umap_scatter(df, archetype_names)

    # ── Bargain list  (FIX #2: 900-min filter inside) ─────────
    bargain_dict = build_bargain_list(df, archetype_names, top_n=5)

    # ── 5 Replacement searches  (FIX #3) ─────────────────────
    print("\n" + "=" * 68)
    print("REPLACEMENT SEARCH DEMOS  (find_replacement)")
    print("=" * 68)

    replacement_searches = [
        dict(player="Mohamed Salah",   season="2024/25", max_budget_eur=40_000_000, max_age=27),
        dict(player="Erling Haaland",  season="2023/24", max_budget_eur=60_000_000, max_age=None),
        dict(player="Virgil van Dijk", season="2023/24", max_budget_eur=30_000_000, max_age=26),
        dict(player="Declan Rice",     season="2023/24", max_budget_eur=50_000_000, max_age=25),
        dict(player="Bukayo Saka",     season="2023/24", max_budget_eur=35_000_000, max_age=23),
    ]

    replacement_results = []
    for i, params in enumerate(replacement_searches, start=1):
        result_obj = find_replacement(
            df=df,
            scaled_cols=scaled_cols,
            min_minutes=900,
            top_n=8,
            **params,
        )
        replacement_results.append(result_obj)

        # Plot
        if result_obj["results"]:
            res_df = pd.DataFrame(result_obj["results"])
            res_df['similarity'] = res_df['similarity'].astype(float)
            plot_replacement_results(
                result_obj["label"],
                res_df,
                f"replacement_demo_{i}.png"
            )

    # ── JSON results_summary_v3.json ─────────────────────────
    results_summary = {
        "pipeline": "S3 v3 — Similarity Engine (dedup clusters, 900-min bargain, budget search)",
        "created":  pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fixes_applied": [
            "FIX1: K-Means centroids merged at cosine_sim > 0.85; archetype names deduplicated via stat tiebreak",
            "FIX2: Bargain list requires >= 900 minutes (eliminates tiny-sample p90 noise)",
            "FIX3: find_replacement() function with 5 demo searches (Salah/Haaland/VanDijk/Rice/Saka)",
        ],
        "metadata": {
            "best_k_before_merge":  best_k,
            "n_final_clusters":     n_final_clusters,
            "merge_pairs":          [
                {"cluster_a": a, "cluster_b": b, "cosine_sim": round(s, 4)}
                for a, b, s in merge_pairs
            ],
            "silhouette_scores":   {str(k): round(float(v), 4) for k, v in sil_scores.items()},
            "best_silhouette":     round(max(sil_scores.values()), 4),
            "archetypes":          {str(k): v for k, v in archetype_names.items()},
            "n_player_seasons":    len(df),
            "seasons_covered":     sorted(df['season'].unique().tolist()),
            "features_used":       feature_cols,
        },
        "cluster_profiles": {},
        "bargain_players":   bargain_dict,
        "replacement_searches": replacement_results,
    }

    for c, prof in cluster_profiles.items():
        results_summary["cluster_profiles"][str(c)] = {
            k: (float(v) if isinstance(v, (float, np.floating)) else
                int(v)   if isinstance(v, (int, np.integer))    else v)
            for k, v in prof.items()
            if k != 'pos_dist'
        }
        results_summary["cluster_profiles"][str(c)]['pos_dist'] = {
            k2: float(v2) for k2, v2 in prof.get('pos_dist', {}).items()
        }

    json_path = MODEL_DIR / "results_summary_v3.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"\nresults_summary_v3.json saved: {json_path}")

    bargain_path = SCOUT_DIR / "bargain_players_v3.json"
    with open(bargain_path, 'w', encoding='utf-8') as f:
        json.dump(bargain_dict, f, ensure_ascii=False, indent=2)
    print(f"Bargain list saved: {bargain_path}")

    # ── Archetype uniqueness audit ────────────────────────────
    names_list = list(archetype_names.values())
    names_unique = len(set(names_list))
    if names_unique == len(names_list):
        print(f"\nArchetype audit PASSED — all {len(names_list)} names are unique.")
    else:
        from collections import Counter
        dupes = [n for n, cnt in Counter(names_list).items() if cnt > 1]
        print(f"\n[WARN] Archetype audit: {len(names_list) - names_unique} duplicate(s) remain: {dupes}")

    print("\n" + "=" * 68)
    print("S3 v3 build complete!")
    print(f"  Cluster assignments : {cluster_path}")
    print(f"  Similarity matrix   : {sim_path}")
    print(f"  Results JSON        : {json_path}")
    print(f"  Bargain list        : {bargain_path}")
    print(f"  Figures             : {FIG_DIR}")
    print(f"  Final cluster count : {n_final_clusters}  (from K={best_k})")
    print(f"  Best silhouette     : {max(sil_scores.values()):.4f}")
    print("=" * 68)


if __name__ == "__main__":
    main()

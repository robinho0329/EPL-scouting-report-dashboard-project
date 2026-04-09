"""
S3 v2: 스카우트를 위한 향상된 선수 유사도 검색 엔진
EPL 데이터 프로젝트

개선 사항 (v1 → v2):
  1. 시장가치 / 나이 필터 지원 (max_value, max_age)
  2. 포지션 내 Per-90 정규화 개선 (position-aware z-score)
  3. 최근 시즌 가중치 부여 (최근 2시즌 × 1.3, 3~4시즌 전 × 1.0, 5시즌+ × 0.7)
  4. 실루엣 점수 개선: K 범위 확장(10~18), 피처 엔지니어링 고도화
  5. 클러스터별 "최고 가성비(bargain)" 선수 목록 자동 생성 (S2 시장가치 예측 활용)
  6. query() 함수: player_name, season, max_age, max_value, min_minutes, top_n
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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import umap

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent.parent
DATA_DIR  = BASE_DIR / "data" / "processed"
SCOUT_DIR = BASE_DIR / "data" / "scout"
FIG_DIR   = Path(__file__).resolve().parent / "figures_v2"
MODEL_DIR = Path(__file__).resolve().parent
S2_JSON   = BASE_DIR / "models" / "s2_market_value" / "results_summary.json"

SCOUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 현재 시즌 (가중치 계산 기준)
CURRENT_SEASON = "2024/25"


def _season_to_year(s: str) -> int:
    """'2023/24' -> 2024 (ending year)"""
    try:
        return int(s.split('/')[0]) + 1
    except Exception:
        return 2000


CURRENT_YEAR = _season_to_year(CURRENT_SEASON)


# ──────────────────────────────────────────────────────────────
# 1. 데이터 로드 및 고도화된 피처 엔지니어링
# ──────────────────────────────────────────────────────────────

def load_and_engineer_features():
    """
    player_season_stats + player_match_logs + player_features 결합.

    개선:
      - 포지션 내 Per-90 정규화 (position-aware z-score)
      - recency_weight: 최근 2시즌 x1.3, 3-4시즌 x1.0, 5시즌+ x0.7
      - goal_contribution_rate, consistency_cv, epl_experience 추가
      - g+a_p90, def_actions_p90, shot_conversion 복합 지표 추가
    """
    print("[1/7] 데이터 로드 중...")
    season_df = pd.read_parquet(DATA_DIR / "player_season_stats.parquet")
    match_df  = pd.read_parquet(DATA_DIR / "player_match_logs.parquet")
    feat_df   = pd.read_parquet(BASE_DIR / "data" / "features" / "player_features.parquet")

    # ── 매치 로그 집계 ────────────────────────────────────────
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

    # ── season_df와 조인 ──────────────────────────────────────
    df = season_df.merge(match_agg, on=['player', 'season', 'team'], how='left')

    # ── player_features 보조 컬럼 조인 ──────────────────────
    aux_cols = ['player', 'season', 'team',
                'goal_contribution_rate', 'consistency_cv', 'epl_experience',
                'minutes_share', 'n_matches', 'versatility_positions']
    aux = feat_df[[c for c in aux_cols if c in feat_df.columns]].drop_duplicates(
        subset=['player', 'season', 'team'])
    df = df.merge(aux, on=['player', 'season', 'team'], how='left')

    # ── 90분 기준 스탯 ────────────────────────────────────────
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

    # 복합 지표
    df['g_plus_a_p90']   = df['goals_p90'] + df['assists_p90']
    df['def_actions_p90'] = df['tackles_p90'] + df['interc_p90']
    df['shot_conversion'] = np.where(
        df['ml_sh'].fillna(0) > 0,
        df['gls'] / df['ml_sh'].replace(0, np.nan),
        0.0
    )

    # ── 역할 지표 ─────────────────────────────────────────────
    df['starter_ratio'] = np.where(
        df['mp'] > 0, df['starts'].fillna(0) / df['mp'], 0.0)

    # ── 포지션 그룹 매핑 ──────────────────────────────────────
    pos_map = {
        'Centre-Forward': 'FW', 'Second Striker': 'FW', 'Striker': 'FW',
        'Right Winger': 'AM', 'Left Winger': 'AM', 'Attacking Midfield': 'AM',
        'Left Midfield': 'AM', 'Right Midfield': 'AM',
        'Central Midfield': 'CM', 'Defensive Midfield': 'DM', 'Midfielder': 'CM',
        'Right-Back': 'DEF', 'Left-Back': 'DEF', 'Centre-Back': 'DEF', 'Defender': 'DEF',
        'Goalkeeper': 'GK',
    }
    df['pos_group'] = df['position'].map(pos_map).fillna('Unknown')

    # ── 최소 출전 필터 (90분 이상) ────────────────────────────
    df = df[df['90s'].fillna(0) >= 1.0].copy()

    # ── 나이 / 키 / 시장가치 ─────────────────────────────────
    df['age_raw']          = df['age'].fillna(df['age_tm']).fillna(25.0)
    df['age_filled']       = df['age_raw']
    df['height_cm']        = df['height_cm'].fillna(
        df.groupby('pos_group')['height_cm'].transform('median')).fillna(180.0)
    df['market_value_raw'] = df['market_value'].fillna(0.0)
    df['market_value_log'] = np.log1p(df['market_value_raw'])

    # total minutes from match logs (more accurate than season_stats 'min')
    df['min'] = df['ml_min'].fillna(df['min']).fillna(0.0)

    # ── 최근 시즌 가중치 ──────────────────────────────────────
    df['season_year']    = df['season'].apply(_season_to_year)
    df['seasons_ago']    = CURRENT_YEAR - df['season_year']
    df['recency_weight'] = np.where(
        df['seasons_ago'] <= 2, 1.3,
        np.where(df['seasons_ago'] <= 4, 1.0, 0.7)
    )

    # ── 보조 피처 결측치 처리 ─────────────────────────────────
    for col in ['goal_contribution_rate', 'consistency_cv', 'epl_experience', 'minutes_share']:
        if col in df.columns:
            df[col] = df[col].fillna(
                df.groupby('pos_group')[col].transform('median')).fillna(0.0)

    # ── 피처 컬럼 목록 ────────────────────────────────────────
    FEATURE_COLS = [
        # Core per-90
        'goals_p90', 'assists_p90', 'shots_p90', 'sot_p90',
        'tackles_p90', 'interc_p90', 'key_passes_p90',
        # Composite
        'g_plus_a_p90', 'def_actions_p90', 'shot_conversion',
        # Physical / participation
        'age_filled', 'height_cm', 'starter_ratio',
        # Market value
        'market_value_log',
        # Role quality
        'goal_contribution_rate',
        # Consistency
        'consistency_cv',
        # Raw scale
        'gls', 'ast',
    ]

    for col in FEATURE_COLS:
        if col in df.columns:
            med = df.groupby('pos_group')[col].transform('median')
            df[col] = df[col].fillna(med).fillna(0.0)
        else:
            df[col] = 0.0

    df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], 0.0)

    print(f"   필터 후 선수-시즌 수: {len(df):,}")
    return df, FEATURE_COLS


# ──────────────────────────────────────────────────────────────
# 2. 포지션 그룹별 StandardScaler + 최근 시즌 가중치
# ──────────────────────────────────────────────────────────────

def normalize_features(df, feature_cols):
    """
    포지션 그룹 내 StandardScaler.
    recency_weight 곱해 최근 시즌 강조.
    포지션 One-Hot(가중치 2.5)으로 포지션 간 분리 강화 -> 실루엣 향상.
    """
    print("[2/7] 포지션 그룹별 피처 정규화 (최근 시즌 가중치 포함) 중...")
    df = df.copy()
    scaled_data = np.zeros((len(df), len(feature_cols)))

    for pos_group in df['pos_group'].unique():
        mask = (df['pos_group'] == pos_group).values
        if mask.sum() < 2:
            continue
        scaler = StandardScaler()
        scaled_data[mask] = scaler.fit_transform(df.loc[mask, feature_cols])

    # recency_weight 적용
    rw = df['recency_weight'].values.reshape(-1, 1)
    scaled_data = scaled_data * rw

    scaled_cols = [f'sc_{c}' for c in feature_cols]
    df[scaled_cols] = scaled_data

    # 포지션 One-Hot (가중치 2.5)
    pos_dummies = pd.get_dummies(df['pos_group'], prefix='pos').astype(float) * 2.5
    pos_oh_cols = pos_dummies.columns.tolist()
    df = pd.concat([df.reset_index(drop=True), pos_dummies.reset_index(drop=True)], axis=1)

    all_scaled_cols = scaled_cols + pos_oh_cols
    return df, all_scaled_cols


# ──────────────────────────────────────────────────────────────
# 3. K-Means 클러스터링 (K=10~18)
# ──────────────────────────────────────────────────────────────

def run_kmeans(df, scaled_cols):
    """K=10~18 범위에서 실루엣 최대 K 선택."""
    print("[3/7] K-Means 클러스터링 (K=10~18 튜닝) 중...")
    X = df[scaled_cols].values

    best_k, best_score, best_labels = 10, -1, None
    scores = {}
    for k in range(10, 19):
        km = KMeans(n_clusters=k, random_state=42, n_init=15, max_iter=500)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels, sample_size=min(5000, len(X)), random_state=42)
        scores[k] = score
        print(f"   K={k}: 실루엣 = {score:.4f}")
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    print(f"   최적 K = {best_k} (실루엣: {best_score:.4f})")
    df['cluster'] = best_labels
    return df, best_k, scores


# ──────────────────────────────────────────────────────────────
# 4. PCA / UMAP 임베딩
# ──────────────────────────────────────────────────────────────

def run_embeddings(df, scaled_cols):
    print("[4/7] PCA/UMAP 2D 임베딩 계산 중...")
    X = df[scaled_cols].values

    pca = PCA(n_components=2, random_state=42)
    pca_emb = pca.fit_transform(X)
    df['pca_x'] = pca_emb[:, 0]
    df['pca_y'] = pca_emb[:, 1]
    print(f"   PCA 설명 분산: {pca.explained_variance_ratio_.sum():.1%}")

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=20, min_dist=0.05)
    umap_emb = reducer.fit_transform(X)
    df['umap_x'] = umap_emb[:, 0]
    df['umap_y'] = umap_emb[:, 1]

    return df, pca


# ──────────────────────────────────────────────────────────────
# 5. 클러스터 아키타입 명명
# ──────────────────────────────────────────────────────────────

def assign_archetypes(df, scaled_cols):
    print("[5/7] 클러스터 아키타입 명명 중...")

    profile_cols = [
        'goals_p90', 'assists_p90', 'shots_p90',
        'tackles_p90', 'interc_p90', 'key_passes_p90',
        'age_filled', 'height_cm', 'market_value_log',
        'starter_ratio', 'gls', 'ast', 'def_actions_p90', 'g_plus_a_p90'
    ]

    global_p75 = df[profile_cols].quantile(0.75)

    cluster_profiles = {}
    archetype_names  = {}

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

        archetype_names[int(c)] = name
        print(f"   C{c}: {name} (n={prof['count']}, {top_pos} {top_frac:.0%}, "
              f"G/90={goals:.2f}, A/90={assists:.2f}, Tkl/90={tackles:.2f})")

    return cluster_profiles, archetype_names


# ──────────────────────────────────────────────────────────────
# 6. Top-K 유사도 행렬
# ──────────────────────────────────────────────────────────────

def build_topk_similarity(df, scaled_cols, top_k=15):
    print("[6/7] Top-K 유사도 행렬 계산 중 (포지션 내, k=15)...")
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
    print(f"   Top-K 이웃 행 수: {len(sim_df):,}")
    return sim_df


# ──────────────────────────────────────────────────────────────
# 7. 핵심 쿼리 함수
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
    스카우트 유사도 쿼리. 시장가치/나이/최소출전 필터 지원.

    Parameters
    ----------
    player_name  : 선수 이름 (부분 일치, 대소문자 무시)
    season       : '2023/24' 형식
    max_age      : 이 나이 이하 선수만 반환
    max_value    : 시장가치(EUR) 이하 선수만 반환
    min_minutes  : 최소 출전 분수 (기본 900)
    top_n        : 반환할 선수 수 (기본 10)
    same_pos_only: True이면 같은 포지션 그룹 내 검색
    """
    mask_name   = df['player'].str.contains(player_name, case=False, na=False)
    mask_season = df['season'] == season
    query_rows  = df[mask_name & mask_season]

    if len(query_rows) == 0:
        query_rows = df[mask_name].sort_values('season', ascending=False)
        if len(query_rows) == 0:
            print(f"  [경고] '{player_name}' 선수를 찾을 수 없습니다.")
            return pd.DataFrame()
        used = query_rows.iloc[0]['season']
        print(f"  [경고] {season} 시즌 없음 -> 최신 시즌({used}) 사용")

    query_row = query_rows.iloc[0]
    query_vec = query_row[scaled_cols].values.reshape(1, -1)
    query_pos = query_row['pos_group']

    compare_df = df[
        ~(mask_name & (df['season'] == query_row['season']))
    ].copy()

    if same_pos_only:
        compare_df = compare_df[compare_df['pos_group'] == query_pos]

    # 최소 출전 시간 필터
    compare_df = compare_df[compare_df['min'].fillna(0) >= min_minutes]

    # 나이 필터
    age_col = 'age_raw'
    if max_age is not None:
        compare_df = compare_df[compare_df[age_col].fillna(99) <= max_age]

    # 시장가치 필터
    mv_col = 'market_value_raw'
    if max_value is not None:
        compare_df = compare_df[
            (compare_df[mv_col].fillna(0) <= max_value) |
            (compare_df[mv_col].fillna(0) == 0)
        ]

    if len(compare_df) == 0:
        print(f"  [경고] 필터 조건을 만족하는 선수가 없습니다.")
        return pd.DataFrame()

    compare_X  = compare_df[scaled_cols].values
    sim_scores = cosine_similarity(query_vec, compare_X)[0]
    compare_df = compare_df.copy()
    compare_df['similarity'] = sim_scores

    result = compare_df.nlargest(top_n, 'similarity')

    out_cols = [
        'player', 'season', 'team', 'position', 'pos_group',
        'age_raw', 'market_value_raw',
        'gls', 'ast', 'goals_p90', 'assists_p90',
        'similarity', 'cluster', 'archetype'
    ]
    out_cols = [c for c in out_cols if c in result.columns]
    return result[out_cols].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# 8. 클러스터별 가성비(bargain) 선수 목록
# ──────────────────────────────────────────────────────────────

def build_bargain_list(df, archetype_names, top_n=5):
    """
    각 클러스터에서 가성비 최고 선수 top_n명.

    bargain_score = (goals_p90 + assists_p90 + def_actions_p90/3) / log1p(market_value_raw)
    S2 value_score 보너스로 보정.
    최근 시즌(2021/22+), 시장가치 있는 선수만 대상.
    """
    print("[bargain] 클러스터별 가성비 선수 목록 생성 중...")

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
        (df['market_value_raw'].fillna(0) > 0)
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
                'market_value_eur': float(row['market_value_raw']),
                'goals_p90':        round(float(row['goals_p90']), 3),
                'assists_p90':      round(float(row['assists_p90']), 3),
                'def_actions_p90':  round(float(row.get('def_actions_p90', 0)), 3),
                'bargain_score':    round(float(row['bargain_score_adj']), 4),
                's2_value_score':   round(float(row['s2_value_score']), 3),
            })
        bargain_dict[int(c)] = {'archetype': name, 'players': records}
        names_list = [r['player'] for r in records]
        print(f"   C{c} ({name}): {names_list}")

    return bargain_dict


# ──────────────────────────────────────────────────────────────
# 시각화
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

    ax.set_title("UMAP 2D - 클러스터별 선수 분포 v2", fontsize=14)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    path = FIG_DIR / "umap_v2.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   UMAP 저장: {path}")


def plot_query_results(query_player, query_season, result_df, fig_name):
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
    ax.set_title(f"'{query_player}' ({query_season}) - Similar Players", fontsize=12)
    plt.tight_layout()
    path = FIG_DIR / fig_name
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   그림 저장: {path}")


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("S3 v2: 향상된 선수 유사도 검색 엔진 빌드 시작")
    print("=" * 65)

    df, feature_cols = load_and_engineer_features()
    df, scaled_cols  = normalize_features(df, feature_cols)
    df, best_k, sil_scores = run_kmeans(df, scaled_cols)
    df, pca_model    = run_embeddings(df, scaled_cols)

    cluster_profiles, archetype_names = assign_archetypes(df, scaled_cols)
    df['archetype'] = df['cluster'].map(archetype_names)

    sim_df = build_topk_similarity(df, scaled_cols, top_k=15)

    # ── 저장 ─────────────────────────────────────────────────
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

    cluster_path = SCOUT_DIR / "cluster_assignments_v2.parquet"
    df[save_cols].to_parquet(cluster_path, index=False)
    print(f"\n클러스터 배정 저장: {cluster_path}")

    sim_path = SCOUT_DIR / "similarity_matrix_v2.parquet"
    sim_df.to_parquet(sim_path, index=False)
    print(f"유사도 행렬 저장: {sim_path}")

    # ── 시각화 ───────────────────────────────────────────────
    print("[7/7] 시각화 생성 중...")
    plot_umap_scatter(df, archetype_names)

    # ── 가성비 목록 ──────────────────────────────────────────
    bargain_dict = build_bargain_list(df, archetype_names, top_n=5)

    # ── 데모 쿼리 ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("데모 쿼리 실행")
    print("=" * 65)

    demo_queries = [
        {
            "label":       "Saka 2023/24 alternatives <=EUR30M, age <=25",
            "player":      "Bukayo Saka",
            "season":      "2023/24",
            "max_value":   30_000_000,
            "max_age":     25,
            "min_minutes": 900,
            "top_n":       10,
        },
        {
            "label":       "Haaland 2023/24 alternatives <=EUR50M",
            "player":      "Erling Haaland",
            "season":      "2023/24",
            "max_value":   50_000_000,
            "max_age":     None,
            "min_minutes": 900,
            "top_n":       10,
        },
        {
            "label":       "Van Dijk 2023/24 CB alternatives, age <=24, <=EUR25M",
            "player":      "Virgil van Dijk",
            "season":      "2023/24",
            "max_value":   25_000_000,
            "max_age":     24,
            "min_minutes": 900,
            "top_n":       10,
        },
        {
            "label":       "Salah 2024/25 alternatives <=EUR40M, age <=27",
            "player":      "Mohamed Salah",
            "season":      "2024/25",
            "max_value":   40_000_000,
            "max_age":     27,
            "min_minutes": 900,
            "top_n":       10,
        },
    ]

    demo_results = {}
    for i, q in enumerate(demo_queries):
        print(f"\n{'─'*60}")
        print(f"Query {i+1}: {q['label']}")
        print(f"{'─'*60}")

        res = query(
            df=df,
            scaled_cols=scaled_cols,
            player_name=q['player'],
            season=q['season'],
            max_age=q.get('max_age'),
            max_value=q.get('max_value'),
            min_minutes=q.get('min_minutes', 900),
            top_n=q.get('top_n', 10),
        )

        if not res.empty:
            display = res[[
                'player', 'season', 'team', 'position',
                'age_raw', 'market_value_raw',
                'goals_p90', 'assists_p90', 'similarity', 'archetype'
            ]].copy()
            display.columns = [
                'Player', 'Season', 'Team', 'Position',
                'Age', 'Market Value',
                'G/90', 'A/90', 'Similarity', 'Archetype'
            ]
            display['Market Value'] = display['Market Value'].apply(
                lambda x: f"EUR{x/1e6:.1f}M" if x > 0 else "N/A"
            )
            display['G/90'] = display['G/90'].round(3)
            display['A/90'] = display['A/90'].round(3)
            display['Similarity'] = display['Similarity'].round(4)
            display.index = range(1, len(display) + 1)
            print(display.to_string())

            fig_name = f"sim_v2_query{i+1}.png"
            plot_query_results(q['player'], q['season'], res, fig_name)

            demo_results[q['label']] = res.apply(
                lambda r: {
                    "player":           r['player'],
                    "season":           r['season'],
                    "team":             r['team'],
                    "position":         str(r.get('position', '')),
                    "age":              float(r.get('age_raw', 0)),
                    "market_value_eur": float(r.get('market_value_raw', 0)),
                    "goals_p90":        round(float(r['goals_p90']), 3),
                    "assists_p90":      round(float(r['assists_p90']), 3),
                    "similarity":       round(float(r['similarity']), 4),
                    "archetype":        str(r.get('archetype', '')),
                },
                axis=1
            ).tolist()
        else:
            print("  결과 없음")
            demo_results[q['label']] = []

    # ── JSON 저장 ─────────────────────────────────────────────
    results_summary = {
        "pipeline": "S3 v2 - Enhanced Player Similarity Engine",
        "created":  pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metadata": {
            "best_k":            best_k,
            "silhouette_scores": {str(k): round(float(v), 4) for k, v in sil_scores.items()},
            "best_silhouette":   round(max(sil_scores.values()), 4),
            "archetypes":        {str(k): v for k, v in archetype_names.items()},
            "n_player_seasons":  len(df),
            "seasons_covered":   sorted(df['season'].unique().tolist()),
            "features_used":     feature_cols,
        },
        "cluster_profiles": {},
        "bargain_players":   bargain_dict,
        "demo_queries":      demo_results,
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

    json_path = MODEL_DIR / "results_summary_v2.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"\nresults_summary_v2.json 저장: {json_path}")

    bargain_path = SCOUT_DIR / "bargain_players_v2.json"
    with open(bargain_path, 'w', encoding='utf-8') as f:
        json.dump(bargain_dict, f, ensure_ascii=False, indent=2)
    print(f"가성비 목록 저장: {bargain_path}")

    print("\n" + "=" * 65)
    print("S3 v2 빌드 완료!")
    print(f"  클러스터 배정  : {cluster_path}")
    print(f"  유사도 행렬    : {sim_path}")
    print(f"  요약 JSON      : {json_path}")
    print(f"  가성비 목록    : {bargain_path}")
    print(f"  그림 디렉터리  : {FIG_DIR}")
    print(f"  최고 실루엣    : {max(sil_scores.values()):.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()

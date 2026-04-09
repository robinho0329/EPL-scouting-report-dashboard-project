"""
S3: 스카우트를 위한 선수 유사도 검색 엔진 (Player Similarity Search Engine)
EPL 데이터 프로젝트 - 코사인 유사도, K-Means 클러스터링, UMAP/PCA 임베딩
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI 없이 파일로 저장
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
BASE_DIR    = Path(__file__).resolve().parent.parent.parent
DATA_DIR    = BASE_DIR / "data" / "processed"
SCOUT_DIR   = BASE_DIR / "data" / "scout"
FIG_DIR     = Path(__file__).resolve().parent / "figures"
MODEL_DIR   = Path(__file__).resolve().parent

SCOUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 1. 데이터 로드 및 피처 엔지니어링
# ──────────────────────────────────────────────────────────────

def load_and_engineer_features() -> pd.DataFrame:
    """
    player_season_stats와 player_match_logs를 결합하여
    유사도 계산에 필요한 피처를 생성한다.
    """
    print("[1/7] 데이터 로드 중...")
    season_df  = pd.read_parquet(DATA_DIR / "player_season_stats.parquet")
    match_df   = pd.read_parquet(DATA_DIR / "player_match_logs.parquet")

    # ── 매치 로그에서 경기당 집계 스탯 계산 ──────────────────
    # 구버전 시즌(2010 이전)은 상세 스탯(sh, tklw, int 등)이 없을 수 있으므로 안전하게 처리
    agg_cols = {
        'sh':    'sum',   # 슈팅 합계
        'sot':   'sum',   # 유효슈팅 합계
        'tklw':  'sum',   # 태클 성공 합계
        'int':   'sum',   # 인터셉트 합계
        'crs':   'sum',   # 크로스 합계
        'fls':   'sum',   # 파울 합계
        'fld':   'sum',   # 파울 유도 합계
        'min':   'sum',   # 출전 시간 (검증용)
        'gls':   'sum',
        'ast':   'sum',
    }
    match_agg = (
        match_df.groupby(['player', 'season', 'team'])
        .agg({k: ('sum' if v == 'sum' else v) for k, v in agg_cols.items()})
        .reset_index()
    )
    match_agg.columns = ['player', 'season', 'team'] + [f'ml_{c}' for c in agg_cols.keys()]

    # ── season_df와 조인 ──────────────────────────────────────
    df = season_df.merge(
        match_agg,
        on=['player', 'season', 'team'],
        how='left'
    )

    # ── 90분 기준 스탯 계산 ───────────────────────────────────
    # '90s' 컬럼: 해당 시즌 출전한 90분 수
    df['90s_safe'] = df['90s'].fillna(0).replace(0, np.nan)

    # Per-90 스탯 생성
    df['goals_p90']   = df['gls']         / df['90s_safe']
    df['assists_p90'] = df['ast']         / df['90s_safe']
    df['shots_p90']   = df['ml_sh']       / df['90s_safe']
    df['sot_p90']     = df['ml_sot']      / df['90s_safe']
    df['tackles_p90'] = df['ml_tklw']     / df['90s_safe']
    df['interc_p90']  = df['ml_int']      / df['90s_safe']
    df['crosses_p90'] = df['ml_crs']      / df['90s_safe']
    df['fouls_p90']   = df['ml_fls']      / df['90s_safe']

    # key_passes(크로스 + gls_1의 별칭)는 매치로그 크로스로 대체
    df['key_passes_p90'] = df['crosses_p90']

    # ── 역할 지표 ─────────────────────────────────────────────
    # 선발 비율 (90분을 소화할 선발 경기수 / 전체 출전 경기수)
    df['starter_ratio'] = np.where(
        df['mp'] > 0,
        df['starts'].fillna(0) / df['mp'],
        0.0
    )

    # 포지션 그룹 매핑 (5가지 대분류)
    pos_map = {
        'Centre-Forward':    'FW',
        'Second Striker':    'FW',
        'Striker':           'FW',
        'Right Winger':      'AM',
        'Left Winger':       'AM',
        'Attacking Midfield':'AM',
        'Left Midfield':     'AM',
        'Right Midfield':    'AM',
        'Central Midfield':  'CM',
        'Defensive Midfield':'DM',
        'Midfielder':        'CM',
        'Right-Back':        'DEF',
        'Left-Back':         'DEF',
        'Centre-Back':       'DEF',
        'Defender':          'DEF',
        'Goalkeeper':        'GK',
    }
    df['pos_group'] = df['position'].map(pos_map).fillna('Unknown')

    # ── 최소 출전 시간 필터 (90분 미만 선수 제외) ─────────────
    df = df[df['90s'].fillna(0) >= 1.0].copy()

    # ── 시장 가치 정규화 (로그 변환) ─────────────────────────
    # 시장 가치가 없는 구버전 시즌 선수는 0으로 처리
    df['market_value_log'] = np.log1p(df['market_value'].fillna(0))

    # ── 키 (height_cm) ────────────────────────────────────────
    df['height_cm'] = df['height_cm'].fillna(df.groupby('pos_group')['height_cm'].transform('median'))
    df['height_cm'] = df['height_cm'].fillna(180.0)  # 전체 중간값으로 fallback

    # ── 나이 ─────────────────────────────────────────────────
    df['age_filled'] = df['age'].fillna(df['age_tm']).fillna(25.0)

    # ── 최종 피처 목록 ────────────────────────────────────────
    FEATURE_COLS = [
        # Per-90 스탯
        'goals_p90', 'assists_p90', 'shots_p90', 'sot_p90',
        'tackles_p90', 'interc_p90', 'key_passes_p90',
        # 물리적 / 참여도
        'age_filled', 'height_cm',
        # 시장 가치
        'market_value_log',
        # 역할
        'starter_ratio',
        # 절대 스탯 (규모 파악용)
        'gls', 'ast',
    ]

    # 결측치를 포지션 그룹 중앙값으로 대체 (구버전 시즌 대응)
    for col in FEATURE_COLS:
        if col in df.columns:
            median_by_pos = df.groupby('pos_group')[col].transform('median')
            df[col] = df[col].fillna(median_by_pos).fillna(0.0)
        else:
            df[col] = 0.0

    # inf 값 제거
    df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], 0.0)

    print(f"   필터 후 선수-시즌 수: {len(df):,}")
    return df, FEATURE_COLS


# ──────────────────────────────────────────────────────────────
# 2. 포지션 그룹별 StandardScaler 정규화
# ──────────────────────────────────────────────────────────────

def normalize_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    포지션 그룹 내에서 StandardScaler로 피처를 정규화한다.
    서로 다른 포지션(예: GK vs FW)은 비교 기준이 다르므로 분리 정규화.
    추가로 포지션 그룹을 One-Hot 인코딩하여 K-Means가 포지션을
    군집화의 주요 축으로 활용할 수 있도록 가중치 포함.
    """
    print("[2/7] 포지션 그룹별 피처 정규화 + One-Hot 포지션 인코딩 중...")
    df = df.copy()
    scaled_data = np.zeros((len(df), len(feature_cols)))

    for pos_group in df['pos_group'].unique():
        mask = df['pos_group'] == pos_group
        if mask.sum() < 2:
            continue
        scaler = StandardScaler()
        scaled_data[mask] = scaler.fit_transform(df.loc[mask, feature_cols])

    scaled_cols = [f'sc_{c}' for c in feature_cols]
    df[scaled_cols] = scaled_data

    # 포지션 One-Hot 인코딩 (가중치 2.0 적용 - 포지션이 군집화의 핵심 축)
    pos_dummies = pd.get_dummies(df['pos_group'], prefix='pos').astype(float) * 2.0
    pos_oh_cols = pos_dummies.columns.tolist()
    df = pd.concat([df.reset_index(drop=True), pos_dummies.reset_index(drop=True)], axis=1)

    # 최종 스케일된 피처: 정규화 스탯 + 포지션 One-Hot
    all_scaled_cols = scaled_cols + pos_oh_cols

    return df, all_scaled_cols


# ──────────────────────────────────────────────────────────────
# 3. K-Means 클러스터링 (실루엣 점수로 K 튜닝)
# ──────────────────────────────────────────────────────────────

def run_kmeans(df: pd.DataFrame, scaled_cols: list) -> pd.DataFrame:
    """
    K=8~12 범위에서 실루엣 점수가 최대인 K를 선택하고
    클러스터 레이블을 부여한다.
    """
    print("[3/7] K-Means 클러스터링 (K=8~12 튜닝) 중...")
    X = df[scaled_cols].values

    best_k, best_score, best_labels = 8, -1, None
    scores = {}
    for k in range(8, 13):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels, sample_size=min(5000, len(X)), random_state=42)
        scores[k] = score
        print(f"   K={k}: 실루엣 점수 = {score:.4f}")
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    print(f"   최적 K = {best_k} (실루엣 점수: {best_score:.4f})")
    df['cluster'] = best_labels
    return df, best_k, scores


# ──────────────────────────────────────────────────────────────
# 4. PCA 및 UMAP 2D 임베딩
# ──────────────────────────────────────────────────────────────

def run_embeddings(df: pd.DataFrame, scaled_cols: list) -> pd.DataFrame:
    """
    PCA(2D)와 UMAP(2D) 임베딩을 계산하여 시각화에 사용한다.
    """
    print("[4/7] PCA/UMAP 2D 임베딩 계산 중...")
    X = df[scaled_cols].values

    # PCA
    pca = PCA(n_components=2, random_state=42)
    pca_emb = pca.fit_transform(X)
    df['pca_x'] = pca_emb[:, 0]
    df['pca_y'] = pca_emb[:, 1]
    explained = pca.explained_variance_ratio_.sum()
    print(f"   PCA 설명 분산 비율 (2D): {explained:.1%}")

    # UMAP
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_emb = reducer.fit_transform(X)
    df['umap_x'] = umap_emb[:, 0]
    df['umap_y'] = umap_emb[:, 1]

    return df, pca


# ──────────────────────────────────────────────────────────────
# 5. 클러스터 아키타입 명명
# ──────────────────────────────────────────────────────────────

def assign_archetypes(df: pd.DataFrame, scaled_cols: list) -> dict:
    """
    각 클러스터의 평균 스탯을 기반으로 아키타입 이름을 자동 부여한다.
    포지션 그룹 내 백분위 순위를 사용하여 절대값 의존성을 제거한다.
    """
    print("[5/7] 클러스터 아키타입 명명 중...")

    cluster_profiles = {}
    archetype_names  = {}

    # 포지션 그룹별 전체 스탯 분포 계산 (상대 비교용)
    profile_cols = [
        'goals_p90', 'assists_p90', 'shots_p90',
        'tackles_p90', 'interc_p90', 'key_passes_p90',
        'age_filled', 'height_cm', 'market_value_log',
        'starter_ratio', 'gls', 'ast'
    ]

    # 각 stat에 대해 전체 75th 백분위 계산
    global_p75 = df[profile_cols].quantile(0.75)
    global_p50 = df[profile_cols].quantile(0.50)

    for c in sorted(df['cluster'].unique()):
        mask = df['cluster'] == c
        sub  = df[mask]
        prof = sub[profile_cols].mean().to_dict()
        prof['count']     = int(mask.sum())
        prof['top_pos']   = sub['pos_group'].value_counts().idxmax()
        prof['pos_dist']  = sub['pos_group'].value_counts(normalize=True).round(3).to_dict()
        cluster_profiles[int(c)] = prof

        # 포지션 분포에서 dominant pos 판단 (50% 이상이면 단일 포지션으로 간주)
        pos_dist = sub['pos_group'].value_counts(normalize=True)
        top_pos  = pos_dist.idxmax()
        top_frac = pos_dist.max()

        # 포지션이 섞여 있으면 가장 많은 두 그룹 고려
        second_pos = pos_dist.index[1] if len(pos_dist) > 1 else top_pos

        goals    = prof['goals_p90']
        tackles  = prof['tackles_p90']
        assists  = prof['assists_p90']
        height   = prof['height_cm']
        kp       = prof['key_passes_p90']
        interc   = prof['interc_p90']

        # 전체 대비 높은/낮은 스탯 판단 (75th 퍼센타일 기준)
        high_goals   = goals   >= global_p75['goals_p90']
        high_tackles = tackles >= global_p75['tackles_p90']
        high_assists = assists >= global_p75['assists_p90']
        high_interc  = interc  >= global_p75['interc_p90']
        high_kp      = kp      >= global_p75['key_passes_p90']
        tall         = height  >= 185.0

        # ── 아키타입 규칙 ──────────────────────────────────────
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
        print(f"   클러스터 {c}: {name} (n={prof['count']}, top_pos={top_pos}({top_frac:.0%}), "
              f"goals_p90={goals:.2f}, tackles_p90={tackles:.2f}, assists_p90={assists:.2f})")

    return cluster_profiles, archetype_names


# ──────────────────────────────────────────────────────────────
# 6. 유사도 검색 함수
# ──────────────────────────────────────────────────────────────

def find_similar_players(
    df: pd.DataFrame,
    scaled_cols: list,
    player_name: str,
    season: str,
    top_k: int = 10,
    method: str = 'cosine',   # 'cosine' | 'euclidean'
    same_pos_only: bool = True
) -> pd.DataFrame:
    """
    주어진 선수+시즌과 가장 유사한 상위 K명을 반환한다.

    Parameters
    ----------
    player_name  : 검색할 선수 이름 (부분 일치 허용)
    season       : '2023/24' 형식
    top_k        : 반환할 유사 선수 수
    method       : 유사도 측정 방법 ('cosine' | 'euclidean')
    same_pos_only: True이면 같은 포지션 그룹 내에서만 검색
    """
    # 선수 찾기 (대소문자 무시, 부분 일치)
    mask_name   = df['player'].str.contains(player_name, case=False, na=False)
    mask_season = df['season'] == season
    query_rows  = df[mask_name & mask_season]

    if len(query_rows) == 0:
        # 시즌 없이 최신 시즌 검색
        query_rows = df[mask_name].sort_values('season', ascending=False)
        if len(query_rows) == 0:
            print(f"  [경고] '{player_name}' 선수를 찾을 수 없습니다.")
            return pd.DataFrame()
        print(f"  [경고] {season} 시즌 없음 → 최신 시즌({query_rows.iloc[0]['season']}) 사용")

    # 여러 행이 있으면 첫 번째 행 사용 (동명이인 처리)
    query_row = query_rows.iloc[0]
    query_vec = query_row[scaled_cols].values.reshape(1, -1)
    query_pos = query_row['pos_group']

    # 비교 대상 필터 (자기 자신 제외)
    compare_df = df[
        ~((df['player'].str.contains(player_name, case=False, na=False)) & (df['season'] == season))
    ].copy()

    if same_pos_only:
        compare_df = compare_df[compare_df['pos_group'] == query_pos]

    if len(compare_df) == 0:
        print(f"  [경고] 비교 대상 선수가 없습니다.")
        return pd.DataFrame()

    compare_X = compare_df[scaled_cols].values

    # 유사도 / 거리 계산
    if method == 'cosine':
        sim_scores = cosine_similarity(query_vec, compare_X)[0]
        compare_df = compare_df.copy()
        compare_df['similarity'] = sim_scores
        result = compare_df.nlargest(top_k, 'similarity')
    else:  # euclidean (PCA 공간 거리)
        query_pca   = np.array([[query_row['pca_x'], query_row['pca_y']]])
        compare_pca = compare_df[['pca_x', 'pca_y']].values
        dists       = euclidean_distances(query_pca, compare_pca)[0]
        compare_df  = compare_df.copy()
        compare_df['similarity'] = 1 / (1 + dists)  # 거리를 유사도로 변환
        result = compare_df.nlargest(top_k, 'similarity')

    # 출력 컬럼 정리
    output_cols = ['player', 'season', 'team', 'position', 'pos_group',
                   'gls', 'ast', 'goals_p90', 'assists_p90', 'similarity', 'cluster']
    output_cols = [c for c in output_cols if c in result.columns]
    return result[output_cols].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# 7. 시각화
# ──────────────────────────────────────────────────────────────

def plot_umap_scatter(df: pd.DataFrame, archetype_names: dict):
    """UMAP 산점도 - 클러스터별 색상"""
    fig, ax = plt.subplots(figsize=(14, 10))
    n_clusters = df['cluster'].nunique()
    palette    = sns.color_palette("tab10", n_clusters)

    for i, c in enumerate(sorted(df['cluster'].unique())):
        mask = df['cluster'] == c
        name = archetype_names.get(int(c), f"Cluster {c}")
        ax.scatter(
            df.loc[mask, 'umap_x'], df.loc[mask, 'umap_y'],
            c=[palette[i]], label=f"C{c}: {name}",
            alpha=0.5, s=12, linewidths=0
        )

    ax.set_title("UMAP 2D 임베딩 - 클러스터별 선수 분포 (EPL 2000/01~2024/25)", fontsize=14)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    path = FIG_DIR / "umap_cluster_scatter.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   저장: {path}")


def plot_radar_charts(df: pd.DataFrame, archetype_names: dict):
    """클러스터별 레이더 차트 (평균 Per-90 스탯)"""
    radar_cols = ['goals_p90', 'assists_p90', 'shots_p90',
                  'tackles_p90', 'interc_p90', 'key_passes_p90']
    labels     = ['Goals\n/90', 'Assists\n/90', 'Shots\n/90',
                  'Tackles\n/90', 'Intercept\n/90', 'Key Pass\n/90']

    n_clusters = df['cluster'].nunique()
    n_cols     = 4
    n_rows     = (n_clusters + n_cols - 1) // n_cols
    fig, axes  = plt.subplots(n_rows, n_cols,
                               figsize=(5 * n_cols, 5 * n_rows),
                               subplot_kw=dict(polar=True))
    axes = axes.flatten()
    palette = sns.color_palette("tab10", n_clusters)

    # 전체 최대값으로 레이더 스케일 통일
    global_max = df[radar_cols].quantile(0.95).values

    angles = np.linspace(0, 2 * np.pi, len(radar_cols), endpoint=False).tolist()
    angles += angles[:1]  # 닫힌 다각형

    for i, c in enumerate(sorted(df['cluster'].unique())):
        ax   = axes[i]
        mask = df['cluster'] == c
        means = df.loc[mask, radar_cols].mean().values
        # 0~1 정규화
        normed = np.where(global_max > 0, means / global_max, 0.0)
        values = normed.tolist() + normed[:1].tolist()

        ax.plot(angles, values, 'o-', linewidth=2, color=palette[i])
        ax.fill(angles, values, alpha=0.25, color=palette[i])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'], size=6)
        name = archetype_names.get(int(c), f"Cluster {c}")
        ax.set_title(f"C{c}: {name}\n(n={mask.sum()})", size=9, pad=10)

    # 남은 subplot 숨기기
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("클러스터별 레이더 차트 (평균 Per-90 스탯)", fontsize=14, y=1.01)
    plt.tight_layout()
    path = FIG_DIR / "cluster_radar_charts.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   저장: {path}")


def plot_cluster_size_distribution(df: pd.DataFrame, archetype_names: dict):
    """클러스터 크기 분포 막대 그래프"""
    counts = df.groupby('cluster').size().reset_index(name='count')
    counts['name'] = counts['cluster'].map(archetype_names)
    counts = counts.sort_values('count', ascending=False)

    palette = sns.color_palette("tab10", len(counts))
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(
        [f"C{r['cluster']}: {r['name']}" for _, r in counts.iterrows()],
        counts['count'],
        color=palette
    )
    ax.bar_label(bars, padding=4, fontsize=9)
    ax.set_xlabel("선수-시즌 수", fontsize=11)
    ax.set_title("클러스터 크기 분포", fontsize=13)
    plt.tight_layout()
    path = FIG_DIR / "cluster_size_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   저장: {path}")


def plot_similarity_search_results(
    query_player: str,
    query_season: str,
    result_df: pd.DataFrame,
    archetype_names: dict,
    fig_name: str
):
    """유사도 검색 결과 시각화 - 수평 막대 그래프"""
    if result_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = sns.color_palette("Blues_r", len(result_df))

    labels = [f"{r['player']} ({r['season']})" for _, r in result_df.iterrows()]
    scores = result_df['similarity'].values

    bars = ax.barh(labels[::-1], scores[::-1], color=colors)
    ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("코사인 유사도", fontsize=11)
    ax.set_title(f"'{query_player}' ({query_season})과 유사한 선수 Top 10", fontsize=12)
    plt.tight_layout()
    path = FIG_DIR / fig_name
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   저장: {path}")


# ──────────────────────────────────────────────────────────────
# 8. Top-K 이웃 유사도 행렬 저장 (space-efficient)
# ──────────────────────────────────────────────────────────────

def build_topk_similarity(df: pd.DataFrame, scaled_cols: list, top_k: int = 10) -> pd.DataFrame:
    """
    전체 선수-시즌 쌍에 대해 코사인 유사도 Top-K 이웃을 계산한다.
    포지션 그룹별로 분리 계산하여 메모리 효율을 높인다.
    """
    print("[6/7] Top-K 유사도 행렬 계산 중...")
    records = []

    for pos_group in df['pos_group'].unique():
        mask = df['pos_group'] == pos_group
        sub  = df[mask].reset_index(drop=True)
        if len(sub) < 2:
            continue

        X      = sub[scaled_cols].values
        sim_mat = cosine_similarity(X)
        np.fill_diagonal(sim_mat, -1)  # 자기 자신 제외

        for i in range(len(sub)):
            top_idx  = np.argsort(sim_mat[i])[::-1][:top_k]
            for rank, j in enumerate(top_idx):
                records.append({
                    'player':       sub.loc[i, 'player'],
                    'season':       sub.loc[i, 'season'],
                    'neighbor':     sub.loc[j, 'player'],
                    'nbr_season':   sub.loc[j, 'season'],
                    'pos_group':    pos_group,
                    'rank':         rank + 1,
                    'cosine_sim':   float(sim_mat[i, j]),
                })

    sim_df = pd.DataFrame(records)
    print(f"   Top-K 이웃 행 수: {len(sim_df):,}")
    return sim_df


# ──────────────────────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("S3: 선수 유사도 검색 엔진 빌드 시작")
    print("=" * 60)

    # ── 피처 엔지니어링 ───────────────────────────────────────
    df, feature_cols = load_and_engineer_features()

    # ── 정규화 ────────────────────────────────────────────────
    df, scaled_cols = normalize_features(df, feature_cols)

    # ── K-Means ───────────────────────────────────────────────
    df, best_k, sil_scores = run_kmeans(df, scaled_cols)

    # ── PCA / UMAP 임베딩 ─────────────────────────────────────
    df, pca_model = run_embeddings(df, scaled_cols)

    # ── 클러스터 아키타입 ─────────────────────────────────────
    cluster_profiles, archetype_names = assign_archetypes(df, scaled_cols)
    df['archetype'] = df['cluster'].map(archetype_names)

    # ── 클러스터 배정 저장 ────────────────────────────────────
    save_cols = [
        'player', 'season', 'team', 'position', 'pos_group',
        'cluster', 'archetype',
        'pca_x', 'pca_y', 'umap_x', 'umap_y',
        'gls', 'ast', 'goals_p90', 'assists_p90',
        'shots_p90', 'tackles_p90', 'interc_p90', 'key_passes_p90',
        'age_filled', 'height_cm', 'market_value_log', 'starter_ratio'
    ]
    save_cols = [c for c in save_cols if c in df.columns]
    cluster_path = SCOUT_DIR / "cluster_assignments.parquet"
    df[save_cols].to_parquet(cluster_path, index=False)
    print(f"\n클러스터 배정 저장: {cluster_path}")

    # ── Top-K 유사도 행렬 저장 ────────────────────────────────
    sim_df = build_topk_similarity(df, scaled_cols, top_k=10)
    sim_path = SCOUT_DIR / "similarity_matrix.parquet"
    sim_df.to_parquet(sim_path, index=False)
    print(f"Top-K 유사도 행렬 저장: {sim_path}")

    # ── 시각화 ────────────────────────────────────────────────
    print("[7/7] 시각화 생성 중...")
    plot_umap_scatter(df, archetype_names)
    plot_radar_charts(df, archetype_names)
    plot_cluster_size_distribution(df, archetype_names)

    # ── 데모 유사도 검색 ──────────────────────────────────────
    demo_searches = [
        ("Erling Haaland",    "2023/24"),
        ("Bukayo Saka",       "2023/24"),
        ("Virgil van Dijk",   "2023/24"),
    ]
    fig_names = [
        "sim_haaland_2324.png",
        "sim_saka_2324.png",
        "sim_vandijk_2324.png",
    ]

    results_summary = {
        "metadata": {
            "best_k":           best_k,
            "silhouette_scores": {str(k): float(v) for k, v in sil_scores.items()},
            "archetypes":       {str(k): v for k, v in archetype_names.items()},
            "n_players":        len(df),
            "seasons_covered":  sorted(df['season'].unique().tolist()),
        },
        "cluster_profiles": {},
        "demo_searches":    {},
    }

    # 클러스터 프로파일을 JSON 직렬화 가능 형태로 변환
    for c, prof in cluster_profiles.items():
        results_summary["cluster_profiles"][str(c)] = {
            k: (float(v) if isinstance(v, (float, np.floating)) else
                int(v)   if isinstance(v, (int, np.integer))    else v)
            for k, v in prof.items()
            if k != 'pos_dist'  # dict 내 dict는 별도 처리
        }
        results_summary["cluster_profiles"][str(c)]['pos_dist'] = {
            k2: float(v2) for k2, v2 in prof.get('pos_dist', {}).items()
        }

    for (player, season), fig_name in zip(demo_searches, fig_names):
        print(f"\n--- 유사도 검색: {player} ({season}) ---")
        res = find_similar_players(df, scaled_cols, player, season, top_k=10, method='cosine')
        if not res.empty:
            print(res[['player', 'season', 'team', 'position', 'similarity']].to_string())
            plot_similarity_search_results(player, season, res, archetype_names, fig_name)

            # JSON 직렬화
            results_summary["demo_searches"][f"{player}_{season}"] = res.apply(
                lambda r: {
                    "rank":       int(r.name) + 1,
                    "player":     r['player'],
                    "season":     r['season'],
                    "team":       r['team'],
                    "position":   str(r.get('position', '')),
                    "similarity": float(r['similarity']),
                    "goals":      float(r.get('gls', 0)),
                    "assists":    float(r.get('ast', 0)),
                },
                axis=1
            ).tolist()

    # ── results_summary.json 저장 ─────────────────────────────
    json_path = MODEL_DIR / "results_summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"\nresults_summary.json 저장: {json_path}")

    print("\n" + "=" * 60)
    print("S3 유사도 검색 엔진 빌드 완료!")
    print(f"  클러스터 배정: {cluster_path}")
    print(f"  유사도 행렬:   {sim_path}")
    print(f"  요약 JSON:     {json_path}")
    print(f"  그림 디렉터리: {FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
S3 v4: Scout Player Similarity Engine — Position-Stratified Clustering
=======================================================================

v3 대비 핵심 변경사항:
  FIX 1 — 포지션별 분리 클러스터링:
    전체 선수를 한 번에 클러스터링하던 방식 → FW/MID/DEF/GK 각각 별도 K-Means.
    Thierry Henry(FW) → "Low-Block Attacking Full-Back" 같은 cross-pos 오배정 완전 제거.

  FIX 2 — 순수 스타일 피처만 사용:
    age_filled, height_cm, market_value_log 제거.
    클러스터 = 선수 스타일(플레이 패턴), 나이/키/가격은 별도 필터링에 사용.

  FIX 3 — pos_group을 WAR 모델과 통일 (FW/MID/DEF/GK):
    AM/CM/DM → MID 통합.
    FB/CB 구분 없이 DEF 통합 후 내부 스타일 클러스터로 자동 분리.

  FIX 4 — 포지션별 최적 K 탐색:
    FW: k=4~7, MID: k=5~8, DEF: k=4~7, GK: k=2~4.
    각 포지션에서 silhouette이 가장 높은 K 자동 선택.

  FIX 5 — 유사도 계산 포지션 내에서만:
    FW와 DEF 비교 불가. 포지션 내에서만 코사인 유사도 계산.

  FIX 6 — 아키타입 레이블 포지션별 의미 있는 이름으로 자동 부여:
    각 클러스터 센트로이드의 지배적 스탯을 기반으로 레이블 자동 결정.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent.parent
DATA_DIR  = BASE_DIR / "data" / "processed"
SCOUT_DIR = BASE_DIR / "data" / "scout"
FIG_DIR   = Path(__file__).resolve().parent / "figures_v4"
MODEL_DIR = Path(__file__).resolve().parent

SCOUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_SEASON = "2024/25"
MIN_MINUTES    = 900   # 스타일 피처 통계적 안정성 확보


def _season_to_year(s: str) -> int:
    try:
        return int(s.split('/')[0]) + 1
    except Exception:
        return 2000


CURRENT_YEAR = _season_to_year(CURRENT_SEASON)


# ──────────────────────────────────────────────────────────────
# 포지션별 피처 정의 (FIX 1, 2, 3)
# ──────────────────────────────────────────────────────────────
POS_FEATURES = {
    'FW': [
        'goals_p90', 'assists_p90', 'shots_p90', 'sot_p90',
        'g_plus_a_p90', 'shot_conversion',
        'fouls_drawn_p90', 'minutes_share', 'starter_ratio',
    ],
    'MID': [
        'assists_p90', 'key_passes_p90',
        'goals_p90', 'shots_p90',
        'tackles_p90', 'interc_p90', 'def_actions_p90',
        'fouls_drawn_p90', 'minutes_share', 'starter_ratio',
    ],
    'DEF': [
        'tackles_p90', 'interc_p90', 'def_actions_p90',
        'fouls_p90', 'crosses_p90',
        'goals_p90', 'assists_p90',
        'minutes_share', 'starter_ratio',
    ],
    'GK': [
        'gk_save_pct', 'gk_cs_pct', 'gk_ga_p90_inv',
        'minutes_share', 'starter_ratio',
    ],
}

# 포지션별 K 탐색 범위 (FIX 4)
POS_K_RANGE = {
    'FW':  range(4, 8),
    'MID': range(5, 9),
    'DEF': range(4, 8),
    'GK':  range(2, 5),
}

# 포지션별 아키타입 후보 레이블 — 지배 스탯 매핑 (FIX 6)
FW_ARCHETYPE_RULES = [
    # (조건 함수(row), 레이블)
    ('goals_p90',      0.50, '⚽ 박스 스트라이커'),     # 골/90분 높음
    ('shot_conversion',0.15, '🎯 효율형 득점왕'),        # 슈팅 효율 최고
    ('shots_p90',      2.50, '🔫 고볼륨 슈터'),          # 슈팅 많음
    ('assists_p90',    0.30, '🎨 찬스 메이커'),           # 어시 높음
    ('fouls_drawn_p90',2.00, '🏃 압박형 전방'),           # 파울 유도 높음
]
MID_ARCHETYPE_RULES = [
    ('def_actions_p90', 6.0, '🛡️ 수비형 미드필더'),
    ('tackles_p90',     3.0, '💪 볼 위너'),
    ('assists_p90',     0.25, '🎨 창의적 플레이메이커'),
    ('goals_p90',       0.15, '📈 박스 투 박스'),
    ('key_passes_p90',  2.5,  '🔑 딥라잉 플레이메이커'),
]
DEF_ARCHETYPE_RULES = [
    ('crosses_p90',    2.5,  '🏃 공격형 풀백'),       # 크로스 많음 = 측면 활용 풀백
    ('assists_p90',    0.12, '🏃 오버래핑 풀백'),      # 어시스트 있는 공격적 풀백
    ('tackles_p90',    2.0,  '💪 강압형 센터백'),      # 태클 많음 = 적극적 CB
    ('def_actions_p90',3.5,  '💪 강압형 센터백'),      # 수비액션 많음
    ('goals_p90',      0.06, '🛡️ 볼배급 센터백'),     # 공격 기여 낮은 순수 수비형 CB
]
GK_ARCHETYPE_RULES = [
    ('gk_save_pct',    0.73, '🧤 슈팅 스토퍼'),
    ('gk_cs_pct',      0.40, '🧱 클린시트 머신'),
]

POS_ARCHETYPE_RULES = {
    'FW':  FW_ARCHETYPE_RULES,
    'MID': MID_ARCHETYPE_RULES,
    'DEF': DEF_ARCHETYPE_RULES,
    'GK':  GK_ARCHETYPE_RULES,
}


# ──────────────────────────────────────────────────────────────
# 1. 데이터 로드 & 피처 생성
# ──────────────────────────────────────────────────────────────
def load_and_engineer():
    print("[1/6] 데이터 로드 중...")

    season_df = pd.read_parquet(DATA_DIR / "player_season_stats.parquet")
    match_df  = pd.read_parquet(DATA_DIR / "player_match_logs.parquet")

    # match log 집계
    agg = match_df.groupby(['player', 'season', 'team']).agg(
        ml_min   = ('min',  'sum'),
        ml_gls   = ('gls',  'sum'),
        ml_ast   = ('ast',  'sum'),
        ml_sh    = ('sh',   'sum'),
        ml_sot   = ('sot',  'sum'),
        ml_tklw  = ('tklw', 'sum'),
        ml_int   = ('int',  'sum'),
        ml_crs   = ('crs',  'sum'),
        ml_fls   = ('fls',  'sum'),
        ml_fld   = ('fld',  'sum'),
        ml_games = ('min',  'count'),
        ml_starts= ('started', 'sum'),
    ).reset_index()

    df = season_df.merge(agg, on=['player', 'season', 'team'], how='left')

    # 출전 시간
    df['total_min'] = df['ml_min'].fillna(df.get('min', pd.Series(dtype=float))).fillna(0.0)

    # 최소 출전 필터
    df = df[df['total_min'] >= MIN_MINUTES].copy()
    print(f"   900분 이상 선수-시즌: {len(df):,}행")

    # Per-90
    s90 = (df['total_min'] / 90.0).replace(0, np.nan)
    df['goals_p90']      = df['ml_gls'].fillna(df.get('gls', 0)) / s90
    df['assists_p90']    = df['ml_ast'].fillna(df.get('ast', 0)) / s90
    df['shots_p90']      = df['ml_sh'].fillna(0) / s90
    df['sot_p90']        = df['ml_sot'].fillna(0) / s90
    df['tackles_p90']    = df['ml_tklw'].fillna(0) / s90
    df['interc_p90']     = df['ml_int'].fillna(0) / s90
    df['crosses_p90']    = df['ml_crs'].fillna(0) / s90
    df['fouls_p90']      = df['ml_fls'].fillna(0) / s90
    df['fouls_drawn_p90']= df['ml_fld'].fillna(0) / s90
    df['key_passes_p90'] = df['crosses_p90']   # 크로스를 key_pass 대리 지표로

    df['g_plus_a_p90']   = df['goals_p90'] + df['assists_p90']
    df['def_actions_p90']= df['tackles_p90'] + df['interc_p90']
    df['shot_conversion'] = np.where(
        df['ml_sh'].fillna(0) > 0,
        df['ml_gls'].fillna(0) / df['ml_sh'].replace(0, np.nan),
        0.0
    )
    df['minutes_share'] = (df['total_min'] / 3420.0).clip(0, 1)
    df['starter_ratio'] = np.where(
        df['ml_games'].fillna(0) > 0,
        df['ml_starts'].fillna(0) / df['ml_games'].fillna(1),
        0.0
    ).clip(0, 1)

    # GK 전용 스탯 — scout_ratings에서 로드
    sr_path = SCOUT_DIR / "scout_ratings_v3.parquet"
    if sr_path.exists():
        sr = pd.read_parquet(sr_path)
        gk_cols = ['player', 'season', 'team',
                   'gk_save_pct', 'gk_cs_pct', 'gk_ga_p90']
        gk_cols = [c for c in gk_cols if c in sr.columns]
        df = df.merge(sr[gk_cols], on=['player', 'season', 'team'], how='left')
        df['gk_ga_p90_inv'] = -df.get('gk_ga_p90', pd.Series(0.0, index=df.index)).fillna(0)
    else:
        df['gk_save_pct']   = np.nan
        df['gk_cs_pct']     = np.nan
        df['gk_ga_p90_inv'] = 0.0

    # pos_group → FW / MID / DEF / GK 4분류 (FIX 3)
    pos_map = {
        'Centre-Forward': 'FW', 'Second Striker': 'FW', 'Striker': 'FW',
        'Right Winger': 'FW',   'Left Winger': 'FW',    # 윙어 → FW
        'Attacking Midfield': 'MID',
        'Left Midfield': 'MID', 'Right Midfield': 'MID',
        'Central Midfield': 'MID', 'Defensive Midfield': 'MID', 'Midfielder': 'MID',
        'Right-Back': 'DEF', 'Left-Back': 'DEF',
        'Centre-Back': 'DEF', 'Defender': 'DEF',
        'Goalkeeper': 'GK',
    }
    df['pos_group'] = df['position'].map(pos_map)

    # 폴백: pos 컬럼으로 추론
    fallback_map = {
        'FW': 'FW', 'MF': 'MID', 'DF': 'DEF', 'GK': 'GK',
        'FW,MF': 'MID', 'MF,FW': 'MID', 'DF,MF': 'DEF', 'MF,DF': 'DEF',
    }
    mask_null = df['pos_group'].isna()
    if mask_null.any() and 'pos' in df.columns:
        df.loc[mask_null, 'pos_group'] = df.loc[mask_null, 'pos'].map(fallback_map)

    df = df[df['pos_group'].isin(['FW', 'MID', 'DEF', 'GK'])].copy()
    print(f"   포지션 분포:\n{df['pos_group'].value_counts().to_string()}")

    # recency weight (최근 2시즌 1.3x, 3~4시즌 1.0x, 오래된 0.7x)
    df['season_year']    = df['season'].apply(_season_to_year)
    df['seasons_ago']    = CURRENT_YEAR - df['season_year']
    df['recency_weight'] = np.where(
        df['seasons_ago'] <= 2, 1.3,
        np.where(df['seasons_ago'] <= 4, 1.0, 0.7)
    )

    # inf / NaN 정리
    for pos, feats in POS_FEATURES.items():
        for col in feats:
            if col in df.columns:
                med = df[df['pos_group'] == pos][col].median()
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(med if not pd.isna(med) else 0.0)

    return df


# ──────────────────────────────────────────────────────────────
# 2. 포지션별 클러스터링 (FIX 1, 4)
# ──────────────────────────────────────────────────────────────
def cluster_by_position(df):
    print("\n[2/6] 포지션별 분리 K-Means 클러스터링...")

    all_rows = []
    cluster_meta = {}   # pos → {k, silhouette, centroids, scaler, feature_cols}
    global_id_offset = 0

    for pos in ['FW', 'MID', 'DEF', 'GK']:
        sub = df[df['pos_group'] == pos].copy()
        feats = [f for f in POS_FEATURES[pos] if f in sub.columns]
        X_raw = sub[feats].fillna(0.0).values

        if len(sub) < 20:
            print(f"   {pos}: 데이터 부족 ({len(sub)}행) → 단일 클러스터 처리")
            sub['cluster_local'] = 0
            sub['cluster'] = global_id_offset
            sub['archetype'] = f"{pos} 선수"
            all_rows.append(sub)
            global_id_offset += 1
            continue

        # StandardScaler (포지션 내 정규화)
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        # recency weighting
        rw = sub['recency_weight'].values.reshape(-1, 1)
        X_weighted = X * rw

        # K 탐색
        best_k, best_score, best_labels = POS_K_RANGE[pos].start, -1, None
        scores = {}
        for k in POS_K_RANGE[pos]:
            if len(sub) < k * 3:
                continue
            km = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
            labels = km.fit_predict(X_weighted)
            score  = silhouette_score(X_weighted, labels,
                                      sample_size=min(3000, len(X_weighted)),
                                      random_state=42)
            scores[k] = round(score, 4)
            print(f"   {pos} K={k}: silhouette={score:.4f}")
            if score > best_score:
                best_k, best_score, best_labels = k, score, labels
                best_km = km

        print(f"   → {pos} 최적 K={best_k} (silhouette={best_score:.4f})")

        sub['cluster_local'] = best_labels
        sub['cluster']       = best_labels + global_id_offset

        # 아키타입 레이블 부여 (FIX 6)
        centroids_scaled = best_km.cluster_centers_          # (k, n_feats)
        centroids_raw    = scaler.inverse_transform(centroids_scaled)
        centroid_df      = pd.DataFrame(centroids_raw, columns=feats)

        label_map = _assign_archetypes(centroid_df, pos, feats)
        sub['archetype'] = sub['cluster_local'].map(label_map)

        # PCA 2D (시각화용)
        pca = PCA(n_components=2, random_state=42)
        pca_coords = pca.fit_transform(X_weighted)
        sub['pca_x'] = pca_coords[:, 0]
        sub['pca_y'] = pca_coords[:, 1]

        cluster_meta[pos] = {
            'best_k':     best_k,
            'silhouette': round(best_score, 4),
            'k_scores':   scores,
            'features':   feats,
            'archetypes': label_map,
            'centroids':  centroid_df.round(4).to_dict(),
        }

        all_rows.append(sub)
        global_id_offset += best_k

    result = pd.concat(all_rows, ignore_index=True)
    return result, cluster_meta


# ──────────────────────────────────────────────────────────────
# 3. 아키타입 레이블 자동 부여 (FIX 6)
# ──────────────────────────────────────────────────────────────
def _assign_archetypes(centroid_df: pd.DataFrame, pos: str, feats: list) -> dict:
    """
    각 클러스터 센트로이드의 지배적 스탯을 기반으로 아키타입 이름 부여.
    규칙 우선순위 순으로 적용, 미매칭 시 'Generic {pos}' 반환.
    """
    rules = POS_ARCHETYPE_RULES.get(pos, [])
    label_map   = {}
    used_labels = set()

    # 1순위: 규칙 기반
    for cid, row in centroid_df.iterrows():
        for (col, threshold, label) in rules:
            if col not in row.index:
                continue
            if row[col] >= threshold and label not in used_labels:
                label_map[cid] = label
                used_labels.add(label)
                break

    # 2순위: 미매칭 클러스터 → 지배 스탯 이름
    generic_names = {
        'FW': ['🎯 측면 공격수', '⚽ 스트라이커', '🎨 창의적 공격수', '🏃 압박 전방'],
        'MID': ['🔑 플레이메이커', '💪 수비형 MF', '📈 박스 투 박스', '🎯 공격형 MF', '⚖️ 올라운더 MF'],
        'DEF': ['🛡️ 볼배급 센터백', '💪 강압형 센터백', '🏃 공격형 풀백', '🛡️ 수비형 풀백'],
        'GK':  ['🧤 슈팅 스토퍼', '🧱 클린시트 키퍼', '🧠 스윕 키퍼'],
    }
    fallbacks = generic_names.get(pos, [f'{pos} 타입 A', f'{pos} 타입 B'])
    fi = 0
    for cid in centroid_df.index:
        if cid not in label_map:
            while fi < len(fallbacks) and fallbacks[fi] in used_labels:
                fi += 1
            lbl = fallbacks[fi] if fi < len(fallbacks) else f'{pos} 타입 {cid}'
            label_map[cid] = lbl
            used_labels.add(lbl)
            fi += 1

    return label_map


# ──────────────────────────────────────────────────────────────
# 4. 포지션 내 유사도 매트릭스 생성 (FIX 5)
# ──────────────────────────────────────────────────────────────
def build_similarity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 포지션 내에서 최근 3시즌(2022/23~2024/25) 선수들의
    코사인 유사도 top-10 유사 선수 목록을 생성.
    """
    print("\n[3/6] 포지션 내 유사도 매트릭스 생성...")

    recent_seasons = {'2022/23', '2023/24', '2024/25'}
    sim_rows = []

    for pos in ['FW', 'MID', 'DEF', 'GK']:
        sub = df[(df['pos_group'] == pos) &
                 (df['season'].isin(recent_seasons))].copy()
        feats = [f for f in POS_FEATURES[pos] if f in sub.columns]
        if len(sub) < 5:
            continue

        X = sub[feats].fillna(0.0).values
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        cos_sim = cosine_similarity(X_s)   # (n, n)
        idx = sub.index.tolist()
        players = sub['player'].values
        seasons = sub['season'].values

        for i in range(len(sub)):
            sims = cos_sim[i].copy()
            sims[i] = -1   # 자기 자신 제외
            top10_idx = np.argsort(sims)[::-1][:10]

            for rank, j in enumerate(top10_idx, 1):
                sim_rows.append({
                    'player':     players[i],
                    'season':     seasons[i],
                    'pos_group':  pos,
                    'neighbor':   players[j],
                    'nbr_season': seasons[j],
                    'cosine_sim': round(float(sims[j]), 4),
                    'rank':       rank,
                })

    sim_df = pd.DataFrame(sim_rows)
    print(f"   유사도 행: {len(sim_df):,}")
    return sim_df


# ──────────────────────────────────────────────────────────────
# 5. 결과 저장
# ──────────────────────────────────────────────────────────────
def save_results(df: pd.DataFrame, sim_df: pd.DataFrame, cluster_meta: dict):
    print("\n[4/6] 결과 저장 중...")

    save_cols = [
        'player', 'season', 'team', 'position', 'pos_group',
        'cluster', 'cluster_local', 'archetype',
        'pca_x', 'pca_y',
        'total_min', 'goals_p90', 'assists_p90',
        'shots_p90', 'sot_p90', 'tackles_p90', 'interc_p90',
        'key_passes_p90', 'def_actions_p90', 'g_plus_a_p90',
        'fouls_drawn_p90', 'crosses_p90',
        'shot_conversion', 'minutes_share', 'starter_ratio',
        'gk_save_pct', 'gk_cs_pct',
        'age', 'age_tm', 'height_cm', 'market_value',
        'recency_weight',
    ]
    save_cols = [c for c in save_cols if c in df.columns]

    out_cluster = SCOUT_DIR / "cluster_assignments_v4.parquet"
    df[save_cols].to_parquet(out_cluster, index=False)
    print(f"   cluster_assignments_v4.parquet 저장 ({len(df):,}행)")

    out_sim = SCOUT_DIR / "similarity_matrix_v4.parquet"
    sim_df.to_parquet(out_sim, index=False)
    print(f"   similarity_matrix_v4.parquet 저장 ({len(sim_df):,}행)")

    # 결과 요약 JSON
    summary = {
        'version':        'v4',
        'created_at':     datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'description':    'S3 v4 — Position-Stratified Clustering + Style Features Only',
        'min_minutes':    MIN_MINUTES,
        'current_season': CURRENT_SEASON,
        'total_player_seasons': int(len(df)),
        'cluster_meta':   {},
    }

    for pos, meta in cluster_meta.items():
        sub24 = df[(df['pos_group'] == pos) & (df['season'] == CURRENT_SEASON)]
        archetype_dist = sub24['archetype'].value_counts().to_dict() if not sub24.empty else {}
        summary['cluster_meta'][pos] = {
            'best_k':       meta['best_k'],
            'silhouette':   meta['silhouette'],
            'k_scores':     meta['k_scores'],
            'features':     meta['features'],
            'archetypes':   meta['archetypes'],
            '2024_25_dist': archetype_dist,
        }

    out_json = MODEL_DIR / "results_summary_v4.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"   results_summary_v4.json 저장")

    return out_cluster, out_sim


# ──────────────────────────────────────────────────────────────
# 6. 시각화
# ──────────────────────────────────────────────────────────────
def visualize(df: pd.DataFrame):
    print("\n[5/6] 시각화 생성 중...")

    colors = ['#e90052', '#00ff87', '#04f5ff', '#ffd700', '#ff6b6b',
              '#a8e6cf', '#dda0dd', '#87ceeb']

    for pos in ['FW', 'MID', 'DEF', 'GK']:
        sub = df[(df['pos_group'] == pos) & df['pca_x'].notna()].copy()
        if len(sub) < 5:
            continue

        fig, ax = plt.subplots(figsize=(10, 7))
        archetypes = sub['archetype'].unique()
        for i, arch in enumerate(archetypes):
            mask = sub['archetype'] == arch
            ax.scatter(
                sub.loc[mask, 'pca_x'], sub.loc[mask, 'pca_y'],
                c=colors[i % len(colors)], label=arch,
                alpha=0.6, s=30, edgecolors='none'
            )

        # 2024/25 선수 이름 표시
        recent = sub[sub['season'] == CURRENT_SEASON]
        for _, row in recent.iterrows():
            ax.annotate(
                row['player'], (row['pca_x'], row['pca_y']),
                fontsize=5, alpha=0.7
            )

        ax.set_title(f'{pos} 아키타입 군집 (v4)', fontsize=13)
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.legend(loc='upper right', fontsize=7, markerscale=1.5)
        ax.set_facecolor('#f8f8f8')
        plt.tight_layout()
        fig.savefig(FIG_DIR / f'cluster_v4_{pos}.png', dpi=120)
        plt.close(fig)
        print(f"   {pos} 클러스터 시각화 저장")


# ──────────────────────────────────────────────────────────────
# 7. 검증 출력
# ──────────────────────────────────────────────────────────────
def validate(df: pd.DataFrame, sim_df: pd.DataFrame):
    print("\n[6/6] 검증 출력...")

    print("\n=== 포지션별 아키타입 분포 (전체) ===")
    for pos in ['FW', 'MID', 'DEF', 'GK']:
        sub = df[df['pos_group'] == pos]
        print(f"\n{pos}:")
        print(sub['archetype'].value_counts().to_string())

    print("\n=== 2024/25 아키타입별 대표 선수 ===")
    sub24 = df[df['season'] == CURRENT_SEASON].copy()
    for pos in ['FW', 'MID', 'DEF', 'GK']:
        p24 = sub24[sub24['pos_group'] == pos]
        if p24.empty:
            continue
        print(f"\n{pos}:")
        for arch in p24['archetype'].unique():
            top3 = p24[p24['archetype'] == arch].nlargest(3, 'minutes_share')
            names = ', '.join(top3['player'].tolist())
            print(f"  {arch}: {names}")

    # Henry 검증: FW에만 배정되어야 함
    henry = df[df['player'].str.contains('Henry', case=False, na=False)]
    if not henry.empty:
        print(f"\n✅ Thierry Henry 배정 확인:")
        print(henry[['player', 'season', 'pos_group', 'archetype']].head(5).to_string(index=False))

    # 유사 선수 검증 (살라)
    salah_sim = sim_df[sim_df['player'].str.contains('Salah', case=False, na=False)].head(5)
    if not salah_sim.empty:
        print(f"\n✅ 살라 유사 선수 (2024/25):")
        s24 = sim_df[(sim_df['player'].str.contains('Salah', case=False, na=False)) &
                     (sim_df['season'] == '2024/25')]
        print(s24[['player', 'neighbor', 'nbr_season', 'cosine_sim']].head(8).to_string(index=False))


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("S3 v4: Position-Stratified Archetype Clustering 시작")
    print("=" * 60)

    df            = load_and_engineer()
    df, meta      = cluster_by_position(df)
    sim_df        = build_similarity_matrix(df)
    save_results(df, sim_df, meta)
    visualize(df)
    validate(df, sim_df)

    print("\n" + "=" * 60)
    print("S3 v4 완료!")
    for pos, m in meta.items():
        print(f"  {pos}: K={m['best_k']}, silhouette={m['silhouette']:.4f}, "
              f"archetypes={list(m['archetypes'].values())}")
    print("=" * 60)

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

# 시즌을 박아두면 새 시즌을 크롤링해도 산출물이 옛 시즌에 머문다.
# 실제로 2025/26 수집 후에도 유사도 매트릭스가 2024/25까지만 나왔다.
# config의 SEASONS 마지막 값을 기준으로 삼는다.
_PROFILES_PATH = SCOUT_DIR / "scout_player_profiles.parquet"
_ALL_SEASONS = sorted(
    pd.read_parquet(_PROFILES_PATH, columns=["season"])["season"].astype(str).unique()
)
CURRENT_SEASON = _ALL_SEASONS[-1]
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
# ──────────────────────────────────────────────────────────────
# 역할 체계 — 포지션(구조) × 스탯(스타일)
# ──────────────────────────────────────────────────────────────
# 스탯만 KMeans에 던지면 "주전이냐 후보냐"로 갈리고 축구 용어와 어긋난다.
# Transfermarkt의 세부 포지션을 구조로 깔고, 그 안에서 스타일로 나눈다.
POSITION_GROUP = {
    "Goalkeeper":         "GK",
    "Centre-Back":        "CB",
    "Left-Back":          "FB",
    "Right-Back":         "FB",
    "Defensive Midfield": "DM",
    "Central Midfield":   "CM",
    "Left Midfield":      "CM",
    "Right Midfield":     "CM",
    "Attacking Midfield": "AM",
    "Left Winger":        "WG",
    "Right Winger":       "WG",
    "Centre-Forward":     "CF",
}

# 그룹별 스타일 축.
# 원칙 — 합성 피처(g_plus_a, def_actions)는 구성 요소와 함께 쓰지 않는다.
#        출전량(minutes_share, starter_ratio)은 스타일이 아니므로 뺀다.
#        (MIN_MINUTES 필터가 이미 표본을 거른다)
GROUP_FEATURES = {
    "GK": ["gk_save_pct", "gk_cs_pct"],
    "CB": ["tackles_p90", "interc_p90", "fouls_p90", "crosses_p90", "goals_p90"],
    "FB": ["crosses_p90", "assists_p90", "tackles_p90", "interc_p90"],
    "DM": ["tackles_p90", "interc_p90", "assists_p90", "goals_p90", "crosses_p90"],
    "CM": ["goals_p90", "assists_p90", "tackles_p90", "interc_p90", "crosses_p90"],
    "AM": ["assists_p90", "goals_p90", "shots_p90", "crosses_p90"],
    "WG": ["crosses_p90", "shots_p90", "goals_p90", "assists_p90"],
    "CF": ["goals_p90", "shots_p90", "fouls_drawn_p90", "offside_p90",
           "assists_p90", "shot_conversion"],
}

# K 하한은 그 그룹에 정의한 역할 수에 맞춘다. 역할이 4개인데 군집이 2개면
# 나머지 역할은 영영 나오지 않는다(실제로 CF가 36명 중 35명이 한 유형이었다).
GROUP_K_RANGE = {
    "GK": range(2, 4), "CB": range(3, 5), "FB": range(2, 4),
    "DM": range(2, 4), "CM": range(3, 6), "AM": range(2, 4),
    "WG": range(2, 4), "CF": range(3, 6),
}

# 역할 정의 — (축, 방향, 이름, 설명).
# 클러스터 중심의 그룹 내 z-점수에서 가장 두드러진 축으로 배정한다.
# 절대 임계값은 시즌 수준이 바뀌면 무너지므로 쓰지 않는다.
GROUP_ROLES = {
    "GK": [
        ("gk_save_pct", "+", "🧤 슈팅 스토퍼",   "선방률이 높은 유형"),
        ("gk_cs_pct",   "+", "🧱 클린시트 키퍼", "실점을 적게 허용하는 유형"),
    ],
    "CB": [
        ("fouls_p90",   "+", "🪓 스토퍼",       "앞으로 나가 끊는 유형 — 태클·파울이 많다"),
        ("tackles_p90", "+", "🪓 스토퍼",       "앞으로 나가 끊는 유형 — 태클·파울이 많다"),
        ("interc_p90",  "+", "🧹 커버형 CB",    "위치로 차단하는 유형 — 인터셉트가 많고 파울이 적다"),
        ("crosses_p90", "+", "🦶 빌드업 CB",    "전진 배급에 관여하는 유형"),
    ],
    "FB": [
        ("crosses_p90", "+", "🏃 오버래핑 풀백", "측면을 올라가 크로스를 올리는 유형"),
        ("assists_p90", "+", "🏃 오버래핑 풀백", "측면을 올라가 크로스를 올리는 유형"),
        ("tackles_p90", "+", "🛡️ 수비형 풀백",  "전진을 자제하고 수비에 무게를 두는 유형"),
        ("interc_p90",  "+", "🛡️ 수비형 풀백",  "전진을 자제하고 수비에 무게를 두는 유형"),
    ],
    "DM": [
        ("tackles_p90", "+", "⚔️ 볼란치",       "중원에서 끊어내는 파괴형"),
        ("interc_p90",  "+", "⚔️ 볼란치",       "중원에서 끊어내는 파괴형"),
        ("assists_p90", "+", "🎼 레지스타",     "후방에서 공격을 조립하는 유형 (수비형 MF 중 공격 기여가 높은 쪽)"),
        ("crosses_p90", "+", "🎼 레지스타",     "후방에서 공격을 조립하는 유형 (수비형 MF 중 공격 기여가 높은 쪽)"),
    ],
    "CM": [
        ("goals_p90",   "+", "🔄 박스투박스",   "양쪽 박스를 오가며 공수에 모두 관여"),
        ("assists_p90", "+", "🎨 메짤라",       "공격 쪽으로 치우쳐 기회를 만드는 유형"),
        ("crosses_p90", "+", "🎨 메짤라",       "공격 쪽으로 치우쳐 기회를 만드는 유형"),
        ("tackles_p90", "+", "🧱 홀딩 미드필더", "뒤를 지키며 균형을 잡는 유형"),
        ("interc_p90",  "+", "🧱 홀딩 미드필더", "뒤를 지키며 균형을 잡는 유형"),
    ],
    "AM": [
        ("assists_p90", "+", "🔟 클래식 10번",   "최전방 바로 뒤에서 패스로 풀어주는 유형"),
        ("goals_p90",   "+", "👤 섀도 스트라이커", "직접 골을 노리며 침투하는 유형"),
        ("shots_p90",   "+", "👤 섀도 스트라이커", "직접 골을 노리며 침투하는 유형"),
        ("crosses_p90", "+", "↔️ 와이드 플레이메이커", "측면으로 벌려 공급하는 유형"),
    ],
    "WG": [
        ("crosses_p90", "+", "🚩 클래식 윙어",   "측면을 파고들어 크로스를 올리는 유형"),
        ("shots_p90",   "+", "↩️ 인버티드 윙어", "안쪽으로 접어 들어와 직접 슛하는 유형"),
        ("goals_p90",   "+", "↩️ 인버티드 윙어", "안쪽으로 접어 들어와 직접 슛하는 유형"),
        ("assists_p90", "+", "🚩 클래식 윙어",   "측면을 파고들어 크로스를 올리는 유형"),
    ],
    "CF": [
        ("shot_conversion",  "+", "🎯 포처",         "적은 슈팅으로 마무리하는 골문 앞 유형"),
        ("offside_p90",      "+", "🎯 포처",         "적은 슈팅으로 마무리하는 골문 앞 유형"),
        ("fouls_drawn_p90",  "+", "🗼 타겟맨",       "몸으로 버티며 볼을 지켜주는 유형"),
        ("assists_p90",      "+", "🌟 컴플리트 포워드", "득점과 연계를 겸하는 유형"),
        ("shots_p90",        "+", "💥 볼륨 슈터",    "슈팅 시도가 많은 유형"),
    ],
}

# 클러스터 최소 인원 — 이보다 작으면 가장 가까운 중심으로 병합한다.
# 1~2명짜리는 아키타입이 아니라 이상치다.
MIN_CLUSTER_FRAC = 0.05
MIN_CLUSTER_ABS  = 8

# K 검약 규칙 — 최고 실루엣과 이 값 이내면 더 작은 K를 택한다.
# MID가 0.0012 차이로 K=7을 골라 1~2명짜리 군집을 만든 적이 있다.
# 다만 너무 관대하면 반대로 K가 눌려 역할이 안 나온다. 0.005가 균형점.
K_TOLERANCE = 0.005


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
        ml_off   = ('off',  'sum'),
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
    df['offside_p90']    = df['ml_off'].fillna(0) / s90   # 포처 판별축
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
        # (구 매핑은 POSITION_GROUP으로 대체됨)
    }
    # 세부 포지션 → 역할 그룹 8종. 스타일 클러스터링은 이 그룹 안에서만 돈다.
    df['pos_group'] = df['position'].map(POSITION_GROUP)

    # 폴백 — TM 포지션이 비면 FBref pos로 대략 배정한다.
    fallback_map = {
        'GK': 'GK', 'DF': 'CB', 'MF': 'CM', 'FW': 'CF',
        'DF,MF': 'FB', 'MF,DF': 'DM', 'MF,FW': 'AM', 'FW,MF': 'WG',
    }
    mask_null = df['pos_group'].isna()
    if mask_null.any() and 'pos' in df.columns:
        df.loc[mask_null, 'pos_group'] = df.loc[mask_null, 'pos'].map(fallback_map)

    df = df[df['pos_group'].isin(GROUP_FEATURES.keys())].copy()
    print(f"   포지션 분포:\n{df['pos_group'].value_counts().to_string()}")

    # recency weight (최근 2시즌 1.3x, 3~4시즌 1.0x, 오래된 0.7x)
    df['season_year']    = df['season'].apply(_season_to_year)
    df['seasons_ago']    = CURRENT_YEAR - df['season_year']
    df['recency_weight'] = np.where(
        df['seasons_ago'] <= 2, 1.3,
        np.where(df['seasons_ago'] <= 4, 1.0, 0.7)
    )

    # inf / NaN 정리
    for pos, feats in GROUP_FEATURES.items():
        for col in feats:
            if col in df.columns:
                med = df[df['pos_group'] == pos][col].median()
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(med if not pd.isna(med) else 0.0)

    return df


# ──────────────────────────────────────────────────────────────
# 2. 포지션별 클러스터링 (FIX 1, 4)
# ──────────────────────────────────────────────────────────────
def _pick_k(X, k_range, sample_weight=None):
    """실루엣으로 K 선택 — 단, 최고점과 K_TOLERANCE 이내면 더 작은 K를 택한다.

    차이가 0.001 수준이면 그건 구조가 아니라 노이즈다. 그걸 따라가면
    1~2명짜리 군집이 생긴다(실제로 MID가 0.0012 차이로 K=7을 골랐다).
    """
    scores = {}
    for k in k_range:
        if k >= len(X):
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
        lab = km.fit_predict(X)
        if len(set(lab)) < 2:
            continue
        scores[k] = silhouette_score(X, lab, sample_size=min(3000, len(X)), random_state=42)
    if not scores:
        return None, {}, None
    best = max(scores.values())
    k = min(kk for kk, s in scores.items() if s >= best - K_TOLERANCE)
    return k, scores, scores[k]


def _merge_small_clusters(X, labels, min_size):
    """최소 인원 미달 군집을 가장 가까운 중심으로 흡수한다."""
    labels = labels.copy()
    for _ in range(10):
        uniq, cnt = np.unique(labels, return_counts=True)
        small = uniq[cnt < min_size]
        if len(small) == 0 or len(uniq) <= 2:
            break
        cents = {c: X[labels == c].mean(axis=0) for c in uniq}
        victim = small[np.argmin([cnt[list(uniq).index(c)] for c in small])]
        others = [c for c in uniq if c != victim]
        target = min(others, key=lambda c: np.linalg.norm(cents[c] - cents[victim]))
        labels[labels == victim] = target
    # 라벨 0..n-1로 재정렬
    remap = {old: i for i, old in enumerate(sorted(set(labels)))}
    return np.array([remap[v] for v in labels])


def _assign_roles(centroid_z: pd.DataFrame, group: str) -> dict:
    """클러스터 중심의 그룹 내 z-점수에서 가장 두드러진 축으로 역할을 배정.

    절대 임계값(goals_p90 > 0.5 같은)은 시즌 수준이 바뀌면 무너지고,
    여러 규칙이 동시에 걸리면 먼저 나온 게 이겨서 라벨이 겹친다.
    상대 순위로 판정하면 그 문제가 없다.
    """
    rules = GROUP_ROLES.get(group, [])
    used, out = set(), {}
    # z가 가장 큰 클러스터부터 배정해 강한 특성이 먼저 이름을 가져간다.
    order = sorted(centroid_z.index, key=lambda c: -centroid_z.loc[c].abs().max())
    for cid in order:
        row = centroid_z.loc[cid]
        best = None
        for axis, direction, name, desc in rules:
            if axis not in row.index:
                continue
            v = row[axis] if direction == "+" else -row[axis]
            if best is None or v > best[0]:
                best = (v, name, desc)
        if best is None:
            out[cid] = (f"{group} 유형", "")
            continue
        # 같은 이름이 이미 쓰였으면 다음 후보로 넘어간다
        name, desc = best[1], best[2]
        if name in used:
            alts = [(row[a] if d == "+" else -row[a], n, ds)
                    for a, d, n, ds in rules if a in row.index and n not in used]
            if alts:
                _, name, desc = max(alts)
        used.add(name)
        out[cid] = (name, desc)
    return out


def cluster_by_position(df):
    print("\n[2/6] 역할 그룹별 스타일 클러스터링...")

    all_rows, cluster_meta = [], {}
    offset = 0

    for group in GROUP_FEATURES:
        sub = df[df['pos_group'] == group].copy()
        feats = [f for f in GROUP_FEATURES[group] if f in sub.columns]
        if len(sub) < 20 or len(feats) < 2:
            print(f"   {group}: 표본 {len(sub)}행 → 단일 유형 처리")
            sub['cluster_local'] = 0
            sub['cluster'] = offset
            sub['archetype'] = f"{group} 유형"
            sub['archetype_desc'] = ""
            all_rows.append(sub); offset += 1
            continue

        scaler = StandardScaler()
        X = scaler.fit_transform(sub[feats].fillna(0.0).values)
        X = X * sub['recency_weight'].values.reshape(-1, 1)

        k, scores, sil = _pick_k(X, GROUP_K_RANGE[group])
        if k is None:
            sub['cluster_local'] = 0; sub['cluster'] = offset
            sub['archetype'] = f"{group} 유형"; sub['archetype_desc'] = ""
            all_rows.append(sub); offset += 1
            continue

        km = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
        labels = _merge_small_clusters(
            X, km.fit_predict(X),
            max(MIN_CLUSTER_ABS, int(len(sub) * MIN_CLUSTER_FRAC)),
        )
        sub['cluster_local'] = labels
        sub['cluster'] = labels + offset

        # 산점도용 2차원 좌표 — 시각화 전용이며 클러스터링에는 쓰지 않는다.
        if X.shape[1] >= 2:
            xy = PCA(n_components=2, random_state=42).fit_transform(X)
            sub['pca_x'], sub['pca_y'] = xy[:, 0], xy[:, 1]
        else:
            sub['pca_x'], sub['pca_y'] = X[:, 0], 0.0

        # 그룹 내 z-점수 중심 → 역할 배정
        raw = sub.groupby('cluster_local')[feats].mean()
        z = (raw - sub[feats].mean()) / sub[feats].std().replace(0, np.nan)
        roles = _assign_roles(z.fillna(0), group)
        sub['archetype'] = sub['cluster_local'].map(lambda c: roles[c][0])
        sub['archetype_desc'] = sub['cluster_local'].map(lambda c: roles[c][1])

        n_final = sub['cluster_local'].nunique()
        print(f"   {group}: {len(sub):4d}명 → K={k}(병합 후 {n_final}) "
              f"silhouette={sil:.4f}  {sorted(sub['archetype'].unique())}")

        cluster_meta[group] = {
            'k': int(n_final), 'silhouette': round(float(sil), 4),
            'k_scores': {int(kk): round(float(v), 4) for kk, v in scores.items()},
            'features': feats,
            'archetypes': {int(c): roles[c][0] for c in roles},
            'centroids': raw.round(4).to_dict(),
        }
        all_rows.append(sub); offset += n_final

    return pd.concat(all_rows, ignore_index=True), cluster_meta


def build_similarity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 포지션 내에서 최근 3시즌(2022/23~2024/25) 선수들의
    코사인 유사도 top-10 유사 선수 목록을 생성.
    """
    print("\n[3/6] 포지션 내 유사도 매트릭스 생성...")

    # 최신 3시즌 — 하드코딩 대신 CURRENT_SEASON에서 역산한다.
    recent_seasons = set(_ALL_SEASONS[-3:])
    sim_rows = []

    for pos in GROUP_FEATURES:
        sub = df[(df['pos_group'] == pos) &
                 (df['season'].isin(recent_seasons))].copy()
        feats = [f for f in GROUP_FEATURES[pos] if f in sub.columns]
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
            'best_k':       meta['k'],
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

    for pos in GROUP_FEATURES:
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
    for pos in GROUP_FEATURES:
        sub = df[df['pos_group'] == pos]
        print(f"\n{pos}:")
        print(sub['archetype'].value_counts().to_string())

    print("\n=== 2024/25 아키타입별 대표 선수 ===")
    sub24 = df[df['season'] == CURRENT_SEASON].copy()
    for pos in GROUP_FEATURES:
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
        print(f"  {pos}: K={m['k']}, silhouette={m['silhouette']:.4f}, "
              f"archetypes={list(m['archetypes'].values())}")
    print("=" * 60)

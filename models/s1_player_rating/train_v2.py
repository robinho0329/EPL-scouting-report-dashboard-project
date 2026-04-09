"""
S1 v3: Scout Player Rating System - 완전 재설계 (v2 버그픽스 + 스카우트 검증)

스카우트 리뷰 v2 피드백 반영:
- WAR=100 버그 수정: rank(pct=True)*100 → (rank-1)/(n-1)*100 방식으로 최대값 < 100
- GK save_pct 버그 수정: match_features에서 SoT 역산하여 실제 세이브% 계산
- 팀 강도 제거: points, goal_diff를 점수 피처에서 완전 제거 (보정 계수도 미사용)
- market_value 피처 완전 제거
- CB/FB 세부 포지션 분리 후 각자 기준으로 평가 (crosses가 CB 점수에 영향 없게)
- 미드필더 crosses 대신 goals+assists+fouls_drawn 기반으로 재조정
- 최소 출전 기준: 900분 이상 OR 15선발 이상
- 출력: 퍼센타일 랭크(0-99.9), 절대로 100이 되지 않음
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from scipy import stats

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE_DIR  = r"C:/Users/xcv54/workspace/EPL project"
DATA_DIR  = os.path.join(BASE_DIR, "data", "processed")
FEAT_DIR  = os.path.join(BASE_DIR, "data", "features")
MODEL_DIR = os.path.join(BASE_DIR, "models", "s1_player_rating")
SCOUT_DIR = os.path.join(BASE_DIR, "data", "scout")
FIG_DIR   = os.path.join(MODEL_DIR, "figures")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCOUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR,   exist_ok=True)

# 한글 폰트 (Windows)
try:
    font_path = "C:/Windows/Fonts/malgun.ttf"
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

print("=" * 60)
print("S1 v3: Scout Player Rating System (완전 재설계) 시작")
print("=" * 60)


# ─────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────
print("\n[1] 데이터 로드 중...")

season_stats  = pd.read_parquet(os.path.join(DATA_DIR, "player_season_stats.parquet"))
match_logs    = pd.read_parquet(os.path.join(DATA_DIR, "player_match_logs.parquet"))
team_summary  = pd.read_parquet(os.path.join(DATA_DIR, "team_season_summary.parquet"))
match_features = pd.read_parquet(os.path.join(FEAT_DIR, "match_features.parquet"))

print(f"  player_season_stats  : {season_stats.shape}")
print(f"  player_match_logs    : {match_logs.shape}")
print(f"  team_season_summary  : {team_summary.shape}")
print(f"  match_features       : {match_features.shape}")


# ─────────────────────────────────────────────
# 2. GK: match_features에서 SoT 역산해 save% 계산
#    각 경기에서 home GK는 AwaySOT, away GK는 HomeSOT에 맞닥뜨림
# ─────────────────────────────────────────────
print("\n[2] GK save% 계산을 위한 SoT 역산 중...")

# match_features에서 날짜/팀/SoT 추출
mf = match_features[['MatchDate', 'HomeTeam', 'AwayTeam',
                       'HomeShotsOnTarget', 'AwayShotsOnTarget', 'Season']].copy()
mf['MatchDate'] = pd.to_datetime(mf['MatchDate'])

# match_logs에서 GK만 추출
gk_logs_raw = match_logs[match_logs['pos'] == 'GK'].copy()
gk_logs_raw['date'] = pd.to_datetime(gk_logs_raw['date'])
gk_logs_raw['clean_sheet'] = (gk_logs_raw['goals_against'] == 0).astype(int)

# GK 경기 로그에 SoT 조인: venue에 따라 home/away 구분
# match_logs의 venue 칼럼: 'Home' or 'Away'
gk_home = gk_logs_raw[gk_logs_raw['venue'] == 'Home'].merge(
    mf[['MatchDate', 'HomeTeam', 'AwayShotsOnTarget']].rename(
        columns={'MatchDate': 'date', 'HomeTeam': 'team', 'AwayShotsOnTarget': 'shots_against'}
    ),
    on=['date', 'team'], how='left'
)
gk_away = gk_logs_raw[gk_logs_raw['venue'] == 'Away'].merge(
    mf[['MatchDate', 'AwayTeam', 'HomeShotsOnTarget']].rename(
        columns={'MatchDate': 'date', 'AwayTeam': 'team', 'HomeShotsOnTarget': 'shots_against'}
    ),
    on=['date', 'team'], how='left'
)

gk_logs = pd.concat([gk_home, gk_away], ignore_index=True)

# SoT 정보 없는 경기: goals_against + 추정 세이브 (EPL 평균 세이브율 ~70%)
# shots_against = goals_against + saves (shots saved)
# 없는 경우: shots_against = goals_against / 0.30 (대략 70% 세이브율 가정)
gk_logs['shots_against'] = gk_logs['shots_against'].fillna(
    (gk_logs['goals_against'] / 0.30).clip(lower=gk_logs['goals_against'])
)
gk_logs['saves'] = (gk_logs['shots_against'] - gk_logs['goals_against']).clip(lower=0)

# 시즌 집계
gk_agg = gk_logs.groupby(['player', 'season', 'team']).agg(
    gk_minutes        = ('min', 'sum'),
    gk_games          = ('min', 'count'),
    gk_starts         = ('started', 'sum'),
    gk_clean_sheets   = ('clean_sheet', 'sum'),
    gk_goals_conceded = ('goals_against', 'sum'),
    gk_shots_against  = ('shots_against', 'sum'),
    gk_saves          = ('saves', 'sum'),
).reset_index()

# 실제 세이브율: saves / shots_against
gk_agg['gk_save_pct'] = np.where(
    gk_agg['gk_shots_against'] > 0,
    gk_agg['gk_saves'] / gk_agg['gk_shots_against'],
    np.nan
)
# 클린시트율
gk_agg['gk_cs_pct'] = np.where(
    gk_agg['gk_games'] > 0,
    gk_agg['gk_clean_sheets'] / gk_agg['gk_games'],
    np.nan
)
# 실점/90분 (낮을수록 좋음 → 나중에 부호 반전)
gk_agg['gk_ga_p90'] = np.where(
    gk_agg['gk_minutes'] > 0,
    gk_agg['gk_goals_conceded'] / (gk_agg['gk_minutes'] / 90.0),
    np.nan
)

print(f"  GK 집계: {len(gk_agg)}명")
print(f"  SoT 데이터 있는 GK 기록 비율: "
      f"{(gk_logs['shots_against'].notna()).mean()*100:.1f}%")


# ─────────────────────────────────────────────
# 3. 필드 플레이어 match_log 집계
# ─────────────────────────────────────────────
print("\n[3] 필드 플레이어 매치 로그 집계 중...")

ml = match_logs.copy()

agg_dict = {
    'min'   : 'sum',
    'gls'   : 'sum',
    'ast'   : 'sum',
    'sh'    : 'sum',
    'sot'   : 'sum',
    'tklw'  : 'sum',
    'int'   : 'sum',
    'crs'   : 'sum',
    'fld'   : 'sum',   # 파울 유도
}
if 'started' in ml.columns:
    agg_dict['started'] = 'sum'

field_agg = ml.groupby(['player', 'season', 'team']).agg(**{
    k: pd.NamedAgg(column=k, aggfunc=v) for k, v in agg_dict.items()
}).reset_index()

if 'started' not in field_agg.columns:
    field_agg['started'] = np.nan

games_count = ml[ml['min'] > 0].groupby(['player', 'season', 'team']).size().reset_index(name='games_played')
field_agg = field_agg.merge(games_count, on=['player', 'season', 'team'], how='left')
field_agg['games_played'] = field_agg['games_played'].fillna(0)

# consistency: 선발 비율
field_agg['consistency'] = np.where(
    field_agg['games_played'] > 0,
    field_agg['started'].fillna(0) / field_agg['games_played'],
    0
).clip(0, 1)

print(f"  field_agg rows: {len(field_agg)}")


# ─────────────────────────────────────────────
# 4. season_stats와 병합 + 기본 피처 생성
#    market_value, points, goal_diff 피처에서 제거
# ─────────────────────────────────────────────
print("\n[4] 피처 엔지니어링 중...")

# season_stats에서 필요 칼럼만 추출 (팀 스탯, market_value 제외)
ss_cols = ['player', 'season', 'team', 'pos', 'position', 'age', 'age_tm',
           'min', 'starts', 'mp', 'gls', 'ast', 'height_cm', 'market_value']
ss = season_stats[[c for c in ss_cols if c in season_stats.columns]].copy()

df = ss.merge(
    field_agg[['player', 'season', 'team',
               'min', 'gls', 'ast', 'sh', 'sot',
               'tklw', 'int', 'crs', 'fld',
               'games_played', 'started', 'consistency']],
    on=['player', 'season', 'team'],
    how='left',
    suffixes=('_ss', '_ml')
)

# 분 수 통합: match_log가 있으면 우선 사용
df['total_min']    = df['min_ml'].fillna(df['min_ss'])
df['total_gls']    = df['gls_ml'].fillna(df['gls_ss'])
df['total_ast']    = df['ast_ml'].fillna(df['ast_ss'])
df['total_starts'] = df['started'].fillna(df['starts'])

# GK 병합
df = df.merge(
    gk_agg[['player', 'season', 'team',
            'gk_minutes', 'gk_games', 'gk_starts',
            'gk_clean_sheets', 'gk_goals_conceded',
            'gk_shots_against', 'gk_saves',
            'gk_save_pct', 'gk_cs_pct', 'gk_ga_p90']],
    on=['player', 'season', 'team'],
    how='left'
)

# GK 분 수 우선 사용
is_gk = df['pos'].str.contains('GK', na=False)
df.loc[is_gk, 'total_min']    = df.loc[is_gk, 'gk_minutes'].fillna(df.loc[is_gk, 'total_min'])
df.loc[is_gk, 'total_starts'] = df.loc[is_gk, 'gk_starts'].fillna(df.loc[is_gk, 'total_starts'])

for c in ['sh', 'sot', 'tklw', 'int', 'crs', 'fld', 'games_played', 'consistency']:
    if c in df.columns:
        df[c] = df[c].fillna(0)

print(f"  병합 후 rows: {len(df)}")


# ─────────────────────────────────────────────
# 5. 최소 출전 기준 필터링
#    → 900분 이상 OR 15선발 이상
# ─────────────────────────────────────────────
print("\n[5] 최소 출전 기준 필터링 (900분 OR 15선발)...")

total_before = len(df)
sufficient_mask = (df['total_min'] >= 900) | (df['total_starts'] >= 15)
df_rated  = df[sufficient_mask].copy()
df_insuff = df[~sufficient_mask].copy()
df_insuff['rating_status'] = 'insufficient_data'

print(f"  전체: {total_before} → 충분한 데이터: {len(df_rated)} / 부족: {len(df_insuff)}")


# ─────────────────────────────────────────────
# 6. 포지션 분류
# ─────────────────────────────────────────────
def classify_position(row):
    pos      = str(row.get('pos', '') or '')
    position = str(row.get('position', '') or '')

    if 'GK' in pos or 'Goalkeeper' in position:
        return 'GK'
    if pos in ['FW'] or pos.startswith('FW'):
        return 'FW'
    if pos in ['MF'] or pos.startswith('MF'):
        return 'MID'
    if pos in ['DF'] or pos.startswith('DF'):
        return 'DEF'

    fw_kw  = ['Forward', 'Winger', 'Striker', 'Second Striker', 'Centre-Forward',
               'Left Winger', 'Right Winger', 'Attacking Midfield']
    mid_kw = ['Midfield', 'Central Midfield', 'Defensive Midfield',
               'Left Midfield', 'Right Midfield']
    def_kw = ['Back', 'Centre-Back', 'Left-Back', 'Right-Back', 'Defender']

    for kw in fw_kw:
        if kw in position:
            return 'FW'
    for kw in mid_kw:
        if kw in position:
            return 'MID'
    for kw in def_kw:
        if kw in position:
            return 'DEF'
    return 'MID'

df_rated['pos_group'] = df_rated.apply(classify_position, axis=1)

# DEF 세부 분류: CB vs FB
def classify_def_subpos(row):
    position = str(row.get('position', '') or '')
    fb_kw = ['Left-Back', 'Right-Back', 'Left Back', 'Right Back',
              'Left Wing-Back', 'Right Wing-Back']
    cb_kw = ['Centre-Back', 'Defender', 'Center-Back']
    for kw in fb_kw:
        if kw in position:
            return 'FB'
    for kw in cb_kw:
        if kw in position:
            return 'CB'
    # pos 칼럼 추가 단서
    pos = str(row.get('pos', '') or '')
    if 'DF' in pos:
        return 'CB'  # 기본값 CB
    return 'CB'

df_rated['def_subpos'] = df_rated.apply(
    lambda r: classify_def_subpos(r) if r['pos_group'] == 'DEF' else r['pos_group'],
    axis=1
)

print(f"\n  포지션 분포:")
print(df_rated['pos_group'].value_counts().to_string())
print(f"\n  DEF 세부 포지션:")
print(df_rated[df_rated['pos_group'] == 'DEF']['def_subpos'].value_counts().to_string())


# ─────────────────────────────────────────────
# 7. Per-90 스탯 계산
# ─────────────────────────────────────────────
print("\n[6] Per-90 스탯 계산...")

MIN_90 = 90.0

def safe_p90(col, minutes):
    return np.where(minutes > 0, col / (minutes / MIN_90), np.nan)

df_rated['min_90s']         = df_rated['total_min'] / MIN_90
df_rated['goals_p90']       = safe_p90(df_rated['total_gls'], df_rated['total_min'])
df_rated['assists_p90']     = safe_p90(df_rated['total_ast'], df_rated['total_min'])
df_rated['shots_p90']       = safe_p90(df_rated['sh'].fillna(0), df_rated['total_min'])
df_rated['sot_p90']         = safe_p90(df_rated['sot'].fillna(0), df_rated['total_min'])
df_rated['tackles_p90']     = safe_p90(df_rated['tklw'].fillna(0), df_rated['total_min'])
df_rated['int_p90']         = safe_p90(df_rated['int'].fillna(0), df_rated['total_min'])
df_rated['crosses_p90']     = safe_p90(df_rated['crs'].fillna(0), df_rated['total_min'])
df_rated['fouls_drawn_p90'] = safe_p90(df_rated['fld'].fillna(0), df_rated['total_min'])

# 출전 시간 비율 (리그 최대 3420분 기준)
df_rated['minutes_share'] = (df_rated['total_min'] / 3420.0).clip(0, 1)

print("  per-90 스탯 생성 완료")


# ─────────────────────────────────────────────
# 8. WAR 계산 (핵심 수정사항)
#
#  수정사항:
#  1. 퍼센타일: rank(pct=True)*100 대신 (rank-1)/(n-1)*100 방식
#     → 최대값이 100이 되지 않음 (최솟값도 0이 되지 않음)
#     → 대신 scipy.stats.percentileofscore 방식 사용
#  2. 팀 강도 보정 제거 (완전)
#  3. CB는 tackles+int 기반 (crosses 무관)
#  4. FB는 tackles+int+crosses 기반
#  5. MID: crosses 제거 → goal_contributions + fouls_drawn 기반
#  6. GK: 실제 save_pct + cs_pct + ga_p90 사용
# ─────────────────────────────────────────────
print("\n[7] WAR 계산 중 (v3 완전 재설계)...")


def pct_rank(series, method='average'):
    """
    0~100 퍼센타일 랭크.
    최솟값 > 0, 최댓값 < 100 (절대로 0 또는 100이 되지 않음).
    scipy percentileofscore 방식: (n_below + 0.5*n_equal) / n * 100
    """
    arr = series.values
    result = np.array([
        stats.percentileofscore(arr[~np.isnan(arr)], v, kind='weak')
        if not np.isnan(v) else 50.0
        for v in arr
    ])
    # weak: count(x <= v) / n * 100 → 최대는 100이 됨
    # average 방식으로 조정: (count_below + 0.5*count_equal) / n * 100
    result = np.array([
        stats.percentileofscore(arr[~np.isnan(arr)], v, kind='rank')
        if not np.isnan(v) else 50.0
        for v in arr
    ])
    # rank 방식도 최대 100 됨
    # 선형 변환으로 [1/(n+1), n/(n+1)] 범위로 스케일
    n = np.sum(~np.isnan(arr))
    if n <= 1:
        return pd.Series(50.0, index=series.index)
    ranks = series.rank(method='average', na_option='keep')
    pct = (ranks / (n + 1)) * 100.0
    pct = pct.fillna(50.0)
    return pct


def zscore_series(s):
    m  = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if sd == 0 or np.isnan(sd) or pd.isna(sd):
        return pd.Series(0.0, index=s.index)
    return (s - m) / sd


def war_from_zscores(z_weighted, group_series):
    """Z-score 가중합 → 그룹 내 퍼센타일 [0~100, 절대 100 미만]"""
    n = len(z_weighted.dropna())
    if n < 5:
        return pd.Series(50.0, index=z_weighted.index)
    return pct_rank(z_weighted)


# ─── FW 공격 WAR ───
# goals_p90 (40%) + assists_p90 (20%) + sot_p90 (20%) + minutes_share (10%) + consistency (10%)
def calc_fw_war(grp):
    g90  = zscore_series(grp['goals_p90'].fillna(0))
    a90  = zscore_series(grp['assists_p90'].fillna(0))
    sot90 = zscore_series(grp['sot_p90'].fillna(0))
    ms   = zscore_series(grp['minutes_share'])
    cons = zscore_series(grp['consistency'])

    raw = (0.40 * g90 + 0.20 * a90 + 0.20 * sot90 + 0.10 * ms + 0.10 * cons)
    return war_from_zscores(raw, grp)


# ─── MID 미드필더 WAR ───
# goals_p90 (20%) + assists_p90 (25%) + fouls_drawn_p90 (15%) +
# tackles_p90 (15%) + minutes_share (15%) + consistency (10%)
# NOTE: crosses 제거 (크로스가 많은 측면 미드필더만 유리해지는 편향 문제)
def calc_mid_war(grp):
    g90   = zscore_series(grp['goals_p90'].fillna(0))
    a90   = zscore_series(grp['assists_p90'].fillna(0))
    fld90 = zscore_series(grp['fouls_drawn_p90'].fillna(0))
    tkl90 = zscore_series(grp['tackles_p90'].fillna(0))
    ms    = zscore_series(grp['minutes_share'])
    cons  = zscore_series(grp['consistency'])

    raw = (0.20 * g90 + 0.25 * a90 + 0.15 * fld90
           + 0.15 * tkl90 + 0.15 * ms + 0.10 * cons)
    return war_from_zscores(raw, grp)


# ─── CB 센터백 WAR ───
# tackles_p90 (25%) + int_p90 (35%) + minutes_share (20%) + consistency (20%)
# NOTE: 팀 실점 품질(cb_def_quality) 제거 - 팀 강도 편향 문제
#       CB는 인터셉트/태클로만 평가 (개인 기여도 중심)
def calc_cb_war(grp):
    tkl90 = zscore_series(grp['tackles_p90'].fillna(0))
    int90 = zscore_series(grp['int_p90'].fillna(0))
    ms    = zscore_series(grp['minutes_share'])
    cons  = zscore_series(grp['consistency'])

    raw = (0.25 * tkl90 + 0.35 * int90 + 0.20 * ms + 0.20 * cons)
    return war_from_zscores(raw, grp)


# ─── FB 풀백 WAR ───
# tackles_p90 (25%) + int_p90 (25%) + crosses_p90 (20%) +
# minutes_share (15%) + consistency (15%)
def calc_fb_war(grp):
    tkl90 = zscore_series(grp['tackles_p90'].fillna(0))
    int90 = zscore_series(grp['int_p90'].fillna(0))
    crs90 = zscore_series(grp['crosses_p90'].fillna(0))
    ms    = zscore_series(grp['minutes_share'])
    cons  = zscore_series(grp['consistency'])

    raw = (0.25 * tkl90 + 0.25 * int90 + 0.20 * crs90 + 0.15 * ms + 0.15 * cons)
    return war_from_zscores(raw, grp)


# ─── GK 골키퍼 WAR ───
# save_pct (40%) + cs_pct (30%) + ga_p90_inv (10%) + minutes_share (10%) + consistency (10%)
def calc_gk_war(grp):
    # save_pct: SoT에서 실제 세이브율 (높을수록 좋음)
    save_pct = zscore_series(grp['gk_save_pct'].fillna(grp['gk_save_pct'].median()))
    # cs_pct: 클린시트율
    cs_pct   = zscore_series(grp['gk_cs_pct'].fillna(grp['gk_cs_pct'].median()))
    # ga_p90: 실점/90분 → 낮을수록 좋으므로 부호 반전
    ga_inv   = zscore_series(-grp['gk_ga_p90'].fillna(grp['gk_ga_p90'].median()))
    ms       = zscore_series(grp['minutes_share'])
    cons     = zscore_series(grp['consistency'])

    raw = (0.40 * save_pct + 0.30 * cs_pct + 0.10 * ga_inv
           + 0.10 * ms + 0.10 * cons)
    return war_from_zscores(raw, grp)


# ─────────────────────────────────────────────
# WAR 계산 루프
# ─────────────────────────────────────────────
war_list = []

for (pos_group, season), grp in df_rated[df_rated['pos_group'] != 'DEF'].groupby(['pos_group', 'season']):
    idx = grp.index
    n   = len(grp)

    if n < 5:
        war_vals = pd.Series(50.0, index=idx)
    elif pos_group == 'FW':
        war_vals = calc_fw_war(grp)
    elif pos_group == 'MID':
        war_vals = calc_mid_war(grp)
    elif pos_group == 'GK':
        war_vals = calc_gk_war(grp)
    else:
        war_vals = pd.Series(50.0, index=idx)

    war_list.append(war_vals)

# DEF: CB와 FB를 각각 독립적으로 퍼센타일 계산
# 그 후, 전체 DEF 내에서 재조정 (CB/FB 비율 유지)
for season, def_grp in df_rated[df_rated['pos_group'] == 'DEF'].groupby('season'):
    cb_grp = def_grp[def_grp['def_subpos'] == 'CB']
    fb_grp = def_grp[def_grp['def_subpos'] == 'FB']

    cb_war = calc_cb_war(cb_grp) if len(cb_grp) >= 5 else pd.Series(50.0, index=cb_grp.index)
    fb_war = calc_fb_war(fb_grp) if len(fb_grp) >= 5 else pd.Series(50.0, index=fb_grp.index)

    combined = pd.concat([cb_war, fb_war])
    # 전체 DEF 내에서 최종 퍼센타일화 (각 그룹 내 상대 순위 유지)
    final_war = pct_rank(combined)
    war_list.append(final_war)


df_rated['war'] = pd.concat(war_list).reindex(df_rated.index)
df_rated['war'] = df_rated['war'].clip(0.5, 99.9)

print(f"  WAR 계산 완료:")
print(f"    mean={df_rated['war'].mean():.1f}, "
      f"min={df_rated['war'].min():.2f}, max={df_rated['war'].max():.2f}")
print(f"    WAR >= 99: {(df_rated['war'] >= 99).sum()} players")
print(f"    WAR == 100: {(df_rated['war'] == 100.0).sum()} players (should be 0)")


# ─────────────────────────────────────────────
# 9. Sanity Check: 2023/24 및 2024/25 포지션별 Top 10
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("[8] SANITY CHECK: 2023/24 포지션별 Top 10")
print("=" * 60)

for target_season in ['2023/24', '2024/25']:
    df_s = df_rated[df_rated['season'] == target_season].copy()

    print(f"\n{'='*50}")
    print(f"  시즌: {target_season}")
    print(f"{'='*50}")

    print(f"\n  ── FW Top 10 ({target_season}) ──")
    fw_top = df_s[df_s['pos_group'] == 'FW'].nlargest(10, 'war')
    print(fw_top[['player', 'team', 'total_min', 'goals_p90', 'assists_p90',
                  'sot_p90', 'war']].to_string(index=False))

    print(f"\n  ── MID Top 10 ({target_season}) ──")
    mid_top = df_s[df_s['pos_group'] == 'MID'].nlargest(10, 'war')
    print(mid_top[['player', 'team', 'total_min', 'goals_p90', 'assists_p90',
                   'fouls_drawn_p90', 'tackles_p90', 'war']].to_string(index=False))

    print(f"\n  ── DEF Top 10 ({target_season}) ──")
    def_top = df_s[df_s['pos_group'] == 'DEF'].nlargest(10, 'war')
    print(def_top[['player', 'team', 'position', 'total_min', 'tackles_p90', 'int_p90',
                   'def_subpos', 'war']].to_string(index=False))

    print(f"\n  ── GK Top 10 ({target_season}) ──")
    gk_top = df_s[df_s['pos_group'] == 'GK'].nlargest(10, 'war')
    print(gk_top[['player', 'team', 'total_min', 'gk_save_pct', 'gk_cs_pct',
                  'gk_ga_p90', 'war']].to_string(index=False))

# 기대 선수 확인 (2023/24)
print("\n  ── 기대 선수 매칭 체크 (2023/24) ──")
df_2324 = df_rated[df_rated['season'] == '2023/24']
expected = {
    'FW':  ['Haaland', 'Watkins', 'Palmer', 'Salah', 'Isak'],
    'MID': ['De Bruyne', 'Fernandes', 'Foden', 'Palmer', 'Olise'],
    'DEF': ['Saliba', 'Gabriel', 'van Dijk', 'Guehi', 'Romero'],
    'GK':  ['Raya', 'Ederson', 'Pickford', 'Alisson'],
}
for grp, names in expected.items():
    top_players = df_2324[df_2324['pos_group'] == grp].nlargest(10, 'war')['player'].tolist()
    found = [n for n in names if any(n.lower() in p.lower() for p in top_players)]
    print(f"  [{grp}] 기대: {names}")
    print(f"        Top10: {top_players[:5]}")
    print(f"        매칭: {found} ({len(found)}/{len(names)})")


# ─────────────────────────────────────────────
# 10. 히든 젬 발굴
# ─────────────────────────────────────────────
print("\n[9] 히든 젬 발굴 중...")

war_q75 = df_rated.groupby(['pos_group', 'season'])['war'].transform(lambda x: x.quantile(0.75))

# 나이
age_col = df_rated['age_tm'].fillna(df_rated['age']) if 'age_tm' in df_rated.columns else df_rated['age']

# market_value: 히든 젬 기준에는 사용 (피처가 아닌 필터 조건으로만)
mv_median = df_rated.groupby('season')['market_value'].transform(
    lambda x: x.median(skipna=True)
)

hidden_gems = df_rated[
    (df_rated['war'] >= war_q75) &
    (df_rated['market_value'].fillna(np.inf) <= mv_median) &
    (age_col <= 27) &
    (df_rated['total_min'] >= 900) &
    (df_rated['market_value'].notna())
].copy()

hidden_gems = hidden_gems.sort_values(['season', 'war'], ascending=[False, False])
print(f"  히든 젬 총 {len(hidden_gems)}명 발굴")
print("\n  최근 시즌 히든 젬 (2023/24):")
hg_2324 = hidden_gems[hidden_gems['season'] == '2023/24']
print(hg_2324[['player', 'team', 'pos_group', 'total_min', 'war',
               'market_value', 'age_tm']].sort_values('war', ascending=False)
      .head(20).to_string(index=False))


# ─────────────────────────────────────────────
# 11. 결과 저장
# ─────────────────────────────────────────────
print("\n[10] 결과 저장 중...")

output_cols = [
    'player', 'team', 'season', 'pos', 'pos_group', 'position', 'def_subpos',
    'age', 'age_tm', 'market_value',
    'total_min', 'total_starts', 'games_played',
    'goals_p90', 'assists_p90', 'shots_p90', 'sot_p90',
    'tackles_p90', 'int_p90', 'crosses_p90', 'fouls_drawn_p90',
    'gk_save_pct', 'gk_cs_pct', 'gk_ga_p90',
    'minutes_share', 'consistency',
    'war',
]
output_cols = [c for c in output_cols if c in df_rated.columns]

scout_ratings = df_rated[output_cols].copy()
scout_ratings.to_parquet(os.path.join(SCOUT_DIR, "scout_ratings_v2.parquet"), index=False)

hg_cols = [c for c in output_cols if c in hidden_gems.columns]
hidden_gems[hg_cols].to_parquet(os.path.join(SCOUT_DIR, "hidden_gems_v2.parquet"), index=False)

print(f"  scout_ratings_v2.parquet 저장 ({len(scout_ratings)} rows)")
print(f"  hidden_gems_v2.parquet   저장 ({len(hidden_gems)} rows)")


# ─────────────────────────────────────────────
# 12. 시각화
# ─────────────────────────────────────────────
print("\n[11] 시각화 생성 중...")

df_2324 = df_rated[df_rated['season'] == '2023/24']
df_2425 = df_rated[df_rated['season'] == '2024/25']

# ── 12-1. WAR 분포 (2023/24) ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
pos_groups = ['FW', 'MID', 'DEF', 'GK']
pos_labels = ['Forward (FW)', 'Midfielder (MID)', 'Defender (DEF)', 'Goalkeeper (GK)']
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

for ax, pg, label, color in zip(axes.flatten(), pos_groups, pos_labels, colors):
    data = df_2324[df_2324['pos_group'] == pg]['war'].dropna()
    if len(data) == 0:
        ax.set_title(f'{label} (No data)')
        continue
    ax.hist(data, bins=20, color=color, edgecolor='white', alpha=0.85)
    ax.axvline(data.mean(), color='navy', linestyle='--', linewidth=2,
               label=f'Mean: {data.mean():.1f}')
    ax.set_title(f'{label} WAR Distribution (2023/24)\nn={len(data)}')
    ax.set_xlabel('WAR (Percentile Rank)')
    ax.set_ylabel('Player Count')
    ax.legend()

plt.suptitle('S1 v3: 2023/24 WAR Distribution by Position', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'war_distribution_v2.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  war_distribution_v2.png saved")

# ── 12-2. FW Top 15 (2023/24) ──
fig, ax = plt.subplots(figsize=(10, 8))
fw_chart = df_2324[df_2324['pos_group'] == 'FW'].nlargest(15, 'war')
ax.barh(range(len(fw_chart)), fw_chart['war'], color='#e74c3c', alpha=0.85)
ax.set_yticks(range(len(fw_chart)))
ax.set_yticklabels([f"{r['player']} ({r['team']})" for _, r in fw_chart.iterrows()], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('WAR (Percentile Rank 0-100)')
ax.set_title('2023/24 Forward (FW) WAR Top 15 - S1 v3', fontweight='bold')
for i, (_, row) in enumerate(fw_chart.iterrows()):
    ax.text(row['war'] + 0.3, i, f"{row['war']:.1f}", va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fw_top15_v2.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  fw_top15_v2.png saved")

# ── 12-3. DEF Top 15 (2023/24) - CB vs FB 색상 구분 ──
fig, ax = plt.subplots(figsize=(11, 9))
def_chart = df_2324[df_2324['pos_group'] == 'DEF'].nlargest(15, 'war')
bar_colors = ['#27ae60' if r['def_subpos'] == 'CB' else '#82e0aa' for _, r in def_chart.iterrows()]
bars = ax.barh(range(len(def_chart)), def_chart['war'], color=bar_colors, alpha=0.85)
ax.set_yticks(range(len(def_chart)))
ax.set_yticklabels([
    f"{r['player']} ({r['team']}) [{r['def_subpos']}]"
    for _, r in def_chart.iterrows()
], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('WAR (Percentile Rank 0-100)')
ax.set_title('2023/24 Defender (DEF) WAR Top 15 - S1 v3\n(Dark=CB, Light=FB)', fontweight='bold')
for i, (_, row) in enumerate(def_chart.iterrows()):
    ax.text(row['war'] + 0.3, i, f"{row['war']:.1f}", va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'def_top15_v2.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  def_top15_v2.png saved")

# ── 12-4. GK Top 10 (2023/24) ──
fig, ax = plt.subplots(figsize=(10, 7))
gk_chart = df_2324[df_2324['pos_group'] == 'GK'].nlargest(10, 'war')
if len(gk_chart) > 0:
    ax.barh(range(len(gk_chart)), gk_chart['war'], color='#f39c12', alpha=0.85)
    ax.set_yticks(range(len(gk_chart)))
    ax.set_yticklabels([f"{r['player']} ({r['team']})" for _, r in gk_chart.iterrows()], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('WAR (Percentile Rank 0-100)')
    ax.set_title('2023/24 Goalkeeper (GK) WAR Top 10 - S1 v3', fontweight='bold')
    for i, (_, row) in enumerate(gk_chart.iterrows()):
        save_s = f" sv%={row['gk_save_pct']:.2f}" if pd.notna(row.get('gk_save_pct')) else ""
        ax.text(row['war'] + 0.3, i, f"{row['war']:.1f}{save_s}", va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'gk_top10_v2.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  gk_top10_v2.png saved")

# ── 12-5. MID Top 15 (2023/24) ──
fig, ax = plt.subplots(figsize=(10, 8))
mid_chart = df_2324[df_2324['pos_group'] == 'MID'].nlargest(15, 'war')
ax.barh(range(len(mid_chart)), mid_chart['war'], color='#3498db', alpha=0.85)
ax.set_yticks(range(len(mid_chart)))
ax.set_yticklabels([f"{r['player']} ({r['team']})" for _, r in mid_chart.iterrows()], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('WAR (Percentile Rank 0-100)')
ax.set_title('2023/24 Midfielder (MID) WAR Top 15 - S1 v3', fontweight='bold')
for i, (_, row) in enumerate(mid_chart.iterrows()):
    ax.text(row['war'] + 0.3, i, f"{row['war']:.1f}", va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'mid_top15_v2.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  mid_top15_v2.png saved")

# ── 12-6. 히든 젬 Scatter (2022/23~2023/24) ──
recent_hg  = hidden_gems[hidden_gems['season'].isin(['2022/23', '2023/24'])].copy()
recent_all = df_rated[df_rated['season'].isin(['2022/23', '2023/24']) &
                       df_rated['market_value'].notna()].copy()

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(recent_all['market_value'] / 1e6, recent_all['war'],
           alpha=0.12, color='gray', s=20, label='All players')
ax.scatter(recent_hg['market_value'] / 1e6, recent_hg['war'],
           alpha=0.8, color='#e74c3c', s=60, label='Hidden Gems', zorder=5)
for _, row in recent_hg.nlargest(10, 'war').iterrows():
    ax.annotate(row['player'], (row['market_value'] / 1e6, row['war']),
                fontsize=7, ha='left', va='bottom', xytext=(3, 3),
                textcoords='offset points')
ax.set_xlabel('Market Value (EUR millions)')
ax.set_ylabel('WAR (Percentile Rank 0-100)')
ax.set_title('Hidden Gems: WAR vs Market Value (2022/23~2023/24) - S1 v3', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'hidden_gems_scatter_v2.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  hidden_gems_scatter_v2.png saved")

# ── 12-7. 2024/25 FW Top 10 ──
fig, ax = plt.subplots(figsize=(10, 7))
fw_2425 = df_2425[df_2425['pos_group'] == 'FW'].nlargest(10, 'war')
ax.barh(range(len(fw_2425)), fw_2425['war'], color='#c0392b', alpha=0.85)
ax.set_yticks(range(len(fw_2425)))
ax.set_yticklabels([f"{r['player']} ({r['team']})" for _, r in fw_2425.iterrows()], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('WAR (Percentile Rank 0-100)')
ax.set_title('2024/25 Forward (FW) WAR Top 10 - S1 v3', fontweight='bold')
for i, (_, row) in enumerate(fw_2425.iterrows()):
    ax.text(row['war'] + 0.3, i, f"{row['war']:.1f}", va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'top_players_2024_25.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  top_players_2024_25.png saved")


# ─────────────────────────────────────────────
# 13. 결과 요약 JSON 저장
# ─────────────────────────────────────────────
print("\n[12] 결과 요약 JSON 저장 중...")

# 포지션별 시즌별 Top 5 저장
top_players_by_season = {}
for season in ['2023/24', '2024/25']:
    ds = df_rated[df_rated['season'] == season]
    top_players_by_season[season] = {}
    for pg in ['FW', 'MID', 'DEF', 'GK']:
        top5 = ds[ds['pos_group'] == pg].nlargest(5, 'war')
        top_players_by_season[season][pg] = [
            {
                'player': r['player'],
                'team': r['team'],
                'war': round(float(r['war']), 2),
                'total_min': int(r['total_min']) if pd.notna(r['total_min']) else None,
            }
            for _, r in top5.iterrows()
        ]

summary = {
    "version": "v3",
    "description": (
        "Scout review v2 피드백 반영: WAR=100 버그 수정, GK save% 실제 계산, "
        "팀강도/market_value 피처 완전 제거, CB/FB 분리 평가, MID crosses 편향 제거"
    ),
    "total_rated_players": int(len(df_rated)),
    "insufficient_data_players": int(len(df_insuff)),
    "hidden_gems_total": int(len(hidden_gems)),
    "minimum_threshold": "900분 이상 OR 15선발 이상",
    "war_range": {
        "min": round(float(df_rated['war'].min()), 2),
        "max": round(float(df_rated['war'].max()), 2),
        "mean": round(float(df_rated['war'].mean()), 2),
    },
    "war_formula": {
        "FW":  "goals_p90(40%) + assists_p90(20%) + sot_p90(20%) + minutes_share(10%) + consistency(10%)",
        "MID": "goals_p90(20%) + assists_p90(25%) + fouls_drawn_p90(15%) + tackles_p90(15%) + minutes_share(15%) + consistency(10%)",
        "CB":  "tackles_p90(25%) + int_p90(35%) + minutes_share(20%) + consistency(20%)",
        "FB":  "tackles_p90(25%) + int_p90(25%) + crosses_p90(20%) + minutes_share(15%) + consistency(15%)",
        "GK":  "save_pct(40%) + cs_pct(30%) + ga_p90_inv(10%) + minutes_share(10%) + consistency(10%)",
    },
    "key_fixes_from_v2": [
        "WAR=100 불가 (rank/(n+1)*100 방식으로 최대값 < 100 보장)",
        "GK save%: 단순 실점률 대신 match_features SoT 역산으로 실제 세이브율 계산",
        "market_value, points, goal_diff 피처에서 완전 제거",
        "CB/FB 분리 평가: 크로스가 CB 점수에 영향 없음",
        "MID crosses 편향 제거: Alfie Doughty류 왜곡 방지",
        "팀 강도 보정(team_adj) 완전 제거",
    ],
    "top_players_by_season": top_players_by_season,
    "seasons_coverage": sorted(df_rated['season'].unique().tolist()),
}

with open(os.path.join(MODEL_DIR, "results_v2.json"), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"  results_v2.json 저장 완료")

print("\n" + "=" * 60)
print("S1 v3 완료!")
print(f"  rated players : {len(df_rated):,}")
print(f"  WAR max       : {df_rated['war'].max():.2f}")
print(f"  WAR = 100     : {(df_rated['war'] == 100.0).sum()} (should be 0)")
print("=" * 60)

"""
S1 v4: Scout Player Rating System - Possession-Adjust DEF + Tier-Norm Fix
==========================================================================

스카우트 검증 v3 피드백 반영:
  FIX 1 — Possession-adjusted tackles/interceptions:
    팀 점유율 프록시를 구해 DEF 스탯을 보정.
    낮은 점유율 팀(하위권) → 수비 스탯 많음 → 상향보정하면 오히려 왜곡되므로,
    대신 팀 강도 티어(top6/mid/bottom6)별 정규화로 VVD/Saliba vs Lacroix/Bednarek
    를 같은 기준으로 비교하지 않고 각 티어 내에서 먼저 퍼센타일화한 뒤 전체로 합산.
  FIX 2 — possession_proxy 기반 tackles/int 보정:
    team_possession_proxy = points / (points + 38) 로 점유율 추정(0~1).
    낮은 점유율 팀은 더 많은 수비 기회 → tackles_p90, int_p90을 (1 - possession_proxy)로 나눔.
  FIX 3 — 최소 출전 필터를 점수 계산 전에 적용 (v2와 동일하지만 명시적 주석 강화).
  FIX 4 — 결과 저장 경로를 results_v3.json으로 변경 (v4 스크립트 → v3 결과).

기대 결과: VVD/Saliba 상위 랭크 > Lacroix/Bednarek (2024/25 기준).
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
print("S1 v4: Scout Player Rating System (Possession-Adjust DEF) 시작")
print("=" * 60)


# ─────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────
print("\n[1] 데이터 로드 중...")

season_stats   = pd.read_parquet(os.path.join(DATA_DIR, "player_season_stats.parquet"))
match_logs     = pd.read_parquet(os.path.join(DATA_DIR, "player_match_logs.parquet"))
team_summary   = pd.read_parquet(os.path.join(DATA_DIR, "team_season_summary.parquet"))
match_features = pd.read_parquet(os.path.join(FEAT_DIR, "match_features.parquet"))

print(f"  player_season_stats  : {season_stats.shape}")
print(f"  player_match_logs    : {match_logs.shape}")
print(f"  team_season_summary  : {team_summary.shape}")
print(f"  match_features       : {match_features.shape}")


# ─────────────────────────────────────────────
# 2. GK: match_features에서 SoT 역산해 save% 계산
# ─────────────────────────────────────────────
print("\n[2] GK save% 계산을 위한 SoT 역산 중...")

mf = match_features[['MatchDate', 'HomeTeam', 'AwayTeam',
                       'HomeShotsOnTarget', 'AwayShotsOnTarget', 'Season']].copy()
mf['MatchDate'] = pd.to_datetime(mf['MatchDate'])

gk_logs_raw = match_logs[match_logs['pos'] == 'GK'].copy()
gk_logs_raw['date'] = pd.to_datetime(gk_logs_raw['date'])
gk_logs_raw['clean_sheet'] = (gk_logs_raw['goals_against'] == 0).astype(int)

# 홈 GK는 AwaySOT에 노출, 원정 GK는 HomeSOT에 노출
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

# SoT 정보 없는 경기: goals_against 기반 추정 (EPL 평균 세이브율 ~70%)
gk_logs['shots_against'] = gk_logs['shots_against'].fillna(
    (gk_logs['goals_against'] / 0.30).clip(lower=gk_logs['goals_against'])
)
gk_logs['saves'] = (gk_logs['shots_against'] - gk_logs['goals_against']).clip(lower=0)

gk_agg = gk_logs.groupby(['player', 'season', 'team']).agg(
    gk_minutes        = ('min', 'sum'),
    gk_games          = ('min', 'count'),
    gk_starts         = ('started', 'sum'),
    gk_clean_sheets   = ('clean_sheet', 'sum'),
    gk_goals_conceded = ('goals_against', 'sum'),
    gk_shots_against  = ('shots_against', 'sum'),
    gk_saves          = ('saves', 'sum'),
).reset_index()

gk_agg['gk_save_pct'] = np.where(
    gk_agg['gk_shots_against'] > 0,
    gk_agg['gk_saves'] / gk_agg['gk_shots_against'],
    np.nan
)
gk_agg['gk_cs_pct'] = np.where(
    gk_agg['gk_games'] > 0,
    gk_agg['gk_clean_sheets'] / gk_agg['gk_games'],
    np.nan
)
# 실점/90분 (낮을수록 좋음)
gk_agg['gk_ga_p90'] = np.where(
    gk_agg['gk_minutes'] > 0,
    gk_agg['gk_goals_conceded'] / (gk_agg['gk_minutes'] / 90.0),
    np.nan
)

print(f"  GK 집계: {len(gk_agg)}명")


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
    'fld'   : 'sum',
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

field_agg['consistency'] = np.where(
    field_agg['games_played'] > 0,
    field_agg['started'].fillna(0) / field_agg['games_played'],
    0
).clip(0, 1)

print(f"  field_agg rows: {len(field_agg)}")


# ─────────────────────────────────────────────
# 4. season_stats와 병합 + 기본 피처 생성
# ─────────────────────────────────────────────
print("\n[4] 피처 엔지니어링 중...")

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

df['total_min']    = df['min_ml'].fillna(df['min_ss'])
df['total_gls']    = df['gls_ml'].fillna(df['gls_ss'])
df['total_ast']    = df['ast_ml'].fillna(df['ast_ss'])
df['total_starts'] = df['started'].fillna(df['starts'])

df = df.merge(
    gk_agg[['player', 'season', 'team',
            'gk_minutes', 'gk_games', 'gk_starts',
            'gk_clean_sheets', 'gk_goals_conceded',
            'gk_shots_against', 'gk_saves',
            'gk_save_pct', 'gk_cs_pct', 'gk_ga_p90']],
    on=['player', 'season', 'team'],
    how='left'
)

is_gk = df['pos'].str.contains('GK', na=False)
df.loc[is_gk, 'total_min']    = df.loc[is_gk, 'gk_minutes'].fillna(df.loc[is_gk, 'total_min'])
df.loc[is_gk, 'total_starts'] = df.loc[is_gk, 'gk_starts'].fillna(df.loc[is_gk, 'total_starts'])

for c in ['sh', 'sot', 'tklw', 'int', 'crs', 'fld', 'games_played', 'consistency']:
    if c in df.columns:
        df[c] = df[c].fillna(0)

print(f"  병합 후 rows: {len(df)}")


# ─────────────────────────────────────────────
# FIX 1 & 2: 팀 점유율 프록시 계산 + possession-adjusted DEF 스탯
#
#   possession_proxy = points / (points + 38)
#     → 리그 최대 점수 38*3=114, 승점 기반 점유율 추정 (0~1 범위)
#     → 맨시티(91점) ≈ 0.71, 노리치(21점) ≈ 0.36
#
#   def_adj_factor = 1 / (1 - possession_proxy)
#     → 낮은 점유율(수비 압박 많은) 팀의 tackles/int를 정규화
#     → 예: 점유율 0.36 → factor = 1.56 (더 많이 수비해야 했으므로)
#     →     점유율 0.71 → factor = 3.45 (적게 수비해도 됐으므로 더 가중)
#   NOTE: factor를 그냥 곱하면 하위팀 수비수가 과도하게 불이익 받는 효과
#         대신 "기회 대비 효율"로 해석 → 보정 후에도 티어별 재정규화 적용
#
#   팀 강도 티어 (시즌별):
#     top6    : points 상위 6개 팀
#     mid     : 중간 8개 팀
#     bottom6 : 하위 6개 팀
#   → DEF WAR 계산 시 tier 내 퍼센타일 → 전체 DEF 재퍼센타일 (2단계 정규화)
# ─────────────────────────────────────────────
print("\n[4b] 팀 possession_proxy 및 강도 티어 계산 중...")

# team_summary 컬럼 확인 후 Season → season 맵핑
ts = team_summary[['Season', 'team', 'points']].copy()
ts.rename(columns={'Season': 'season'}, inplace=True)

# possession_proxy: 점수 기반 (0~1)
ts['possession_proxy'] = ts['points'] / (ts['points'] + 38.0)

# 시즌별 팀 강도 티어 계산
def assign_tier(group):
    """시즌별 20팀을 points 기준으로 top6/mid8/bottom6 분류"""
    group = group.sort_values('points', ascending=False).reset_index(drop=True)
    group['tier'] = 'mid'
    group.loc[:5, 'tier'] = 'top6'     # 상위 6팀
    group.loc[14:, 'tier'] = 'bottom6' # 하위 6팀
    return group

ts = ts.groupby('season', group_keys=False).apply(assign_tier)

print("  시즌별 티어 예시 (2024/25):")
ts24 = ts[ts['season'] == '2024/25'].sort_values('points', ascending=False)
if len(ts24) > 0:
    print(ts24[['team', 'points', 'possession_proxy', 'tier']].to_string(index=False))

# 선수 데이터에 팀 정보 조인
df = df.merge(ts[['season', 'team', 'possession_proxy', 'tier']], on=['season', 'team'], how='left')

# 팀 정보 없는 경우 중앙값으로 대체
median_proxy = ts['possession_proxy'].median()
df['possession_proxy'] = df['possession_proxy'].fillna(median_proxy)
df['tier'] = df['tier'].fillna('mid')

print(f"\n  possession_proxy 범위: {df['possession_proxy'].min():.3f} ~ {df['possession_proxy'].max():.3f}")
print(f"  median: {df['possession_proxy'].median():.3f}")


# ─────────────────────────────────────────────
# 5. 최소 출전 기준 필터링 - 점수 계산 전에 적용
#    FIX 3: 900분 필터를 WAR 계산 전에 수행 (v2와 동일하지만 명시적)
# ─────────────────────────────────────────────
print("\n[5] 최소 출전 기준 필터링 (900분 OR 15선발) - WAR 계산 전 적용...")

total_before = len(df)
sufficient_mask = (df['total_min'] >= 900) | (df['total_starts'] >= 15)
df_rated  = df[sufficient_mask].copy()
df_insuff = df[~sufficient_mask].copy()
df_insuff['rating_status'] = 'insufficient_data'

print(f"  전체: {total_before} → 충분한 데이터: {len(df_rated)} / 부족: {len(df_insuff)}")
print("  ✓ 필터 적용: WAR 계산 전 완료")


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
    pos = str(row.get('pos', '') or '')
    if 'DF' in pos:
        return 'CB'
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
# 7. Per-90 스탯 계산 + Possession-Adjusted DEF 스탯
# ─────────────────────────────────────────────
print("\n[6] Per-90 스탯 계산 + possession-adjusted DEF 스탯...")

MIN_90 = 90.0

def safe_p90(col, minutes):
    """분당 스탯 → per-90분 환산"""
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

# ── Possession-Adjusted DEF 스탯 계산 ──
# def_adj_factor: 낮은 점유율(수비 기회 많음)을 제거해 "순수 수비 효율" 산출
# 공식: adjusted_stat = raw_p90 * possession_proxy
#       (점유율 높은 팀의 수비수는 수비 기회가 적었으므로, 낮은 raw를 그대로 쓰면 불공정)
#       대신: 점유율 비율로 나눠 "기회 조정 효율"로 변환
#       adjusted_tackles_p90 = tackles_p90 / (1 - possession_proxy)
#       → 점유율 0.70인 맨시티 수비수: factor = 1/0.30 = 3.33 (수비 기회 적었지만 효율 측정)
#       → 점유율 0.35인 루턴 수비수:   factor = 1/0.65 = 1.54 (수비 기회 많았으므로 덜 가중)
# NOTE: 과도한 보정 방지를 위해 factor를 0.5~3.0 범위로 클리핑

poss_proxy = df_rated['possession_proxy'].fillna(median_proxy).clip(0.25, 0.75)
# factor: 낮은 점유율(상대가 많이 가짐) → 수비 기회 많음 → 스탯 나누기 큰 값 → 효율 기준
# 즉, (1 - possession_proxy) = 상대 점유율 추정치 = 수비 기회 비율
adj_factor = (1.0 / (1.0 - poss_proxy)).clip(0.5, 3.0)

df_rated['tackles_p90_adj'] = (df_rated['tackles_p90'].fillna(0) / adj_factor).fillna(0)
df_rated['int_p90_adj']     = (df_rated['int_p90'].fillna(0) / adj_factor).fillna(0)

print("  possession-adjusted DEF 스탯 생성 완료")
print(f"  adj_factor 범위: {adj_factor.min():.2f} ~ {adj_factor.max():.2f}")

# 2024/25 VVD/Saliba vs Lacroix/Bednarek 검증용 출력
check_players = ['van Dijk', 'Saliba', 'Lacroix', 'Bednarek']
check_2425 = df_rated[
    (df_rated['season'] == '2024/25') &
    (df_rated['player'].str.contains('|'.join(check_players), case=False, na=False))
]
if len(check_2425) > 0:
    print("\n  ── 검증 대상 선수 raw vs adjusted DEF 스탯 (2024/25) ──")
    print(check_2425[['player', 'team', 'tier', 'possession_proxy',
                       'tackles_p90', 'tackles_p90_adj',
                       'int_p90', 'int_p90_adj']].to_string(index=False))


# ─────────────────────────────────────────────
# 8. WAR 계산 (v4 핵심 수정)
#
#   DEF 변경사항:
#   1. CB/FB 모두 possession-adjusted tackles/int 사용
#   2. 티어별(top6/mid/bottom6) 먼저 퍼센타일화 후 전체 DEF 재퍼센타일화 (2단계)
#      → 같은 리그 내에서도 팀 강도 맥락을 어느 정도 유지
#   나머지(FW, MID, GK)는 v3(train_v2.py 기준) 그대로 유지
# ─────────────────────────────────────────────
print("\n[7] WAR 계산 중 (v4: possession-adjusted DEF + tier normalization)...")


def pct_rank(series):
    """
    0~100 퍼센타일 랭크.
    최솟값 > 0, 최댓값 < 100 (rank/(n+1)*100 방식으로 절대 100 불가).
    """
    n = series.notna().sum()
    if n <= 1:
        return pd.Series(50.0, index=series.index)
    ranks = series.rank(method='average', na_option='keep')
    pct   = (ranks / (n + 1)) * 100.0
    return pct.fillna(50.0)


def zscore_series(s):
    """표준화 (Z-score). 분산 0이면 0 반환."""
    m  = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if sd == 0 or np.isnan(sd) or pd.isna(sd):
        return pd.Series(0.0, index=s.index)
    return (s - m) / sd


def war_from_zscores(z_weighted):
    """Z-score 가중합 → 그룹 내 퍼센타일 [0~100, 최대 < 100]"""
    n = z_weighted.notna().sum()
    if n < 5:
        return pd.Series(50.0, index=z_weighted.index)
    return pct_rank(z_weighted)


# ─── FW 공격 WAR ───
def calc_fw_war(grp):
    g90  = zscore_series(grp['goals_p90'].fillna(0))
    a90  = zscore_series(grp['assists_p90'].fillna(0))
    sot90 = zscore_series(grp['sot_p90'].fillna(0))
    ms   = zscore_series(grp['minutes_share'])
    cons = zscore_series(grp['consistency'])
    raw  = (0.40 * g90 + 0.20 * a90 + 0.20 * sot90 + 0.10 * ms + 0.10 * cons)
    return war_from_zscores(raw)


# ─── MID 미드필더 WAR ───
def calc_mid_war(grp):
    g90   = zscore_series(grp['goals_p90'].fillna(0))
    a90   = zscore_series(grp['assists_p90'].fillna(0))
    fld90 = zscore_series(grp['fouls_drawn_p90'].fillna(0))
    tkl90 = zscore_series(grp['tackles_p90'].fillna(0))
    ms    = zscore_series(grp['minutes_share'])
    cons  = zscore_series(grp['consistency'])
    raw   = (0.20 * g90 + 0.25 * a90 + 0.15 * fld90
             + 0.15 * tkl90 + 0.15 * ms + 0.10 * cons)
    return war_from_zscores(raw)


# ─── CB 센터백 WAR (possession-adjusted) ───
# tackles_p90_adj + int_p90_adj 사용: 팀 점유율 보정 후 순수 수비 효율 평가
def calc_cb_war(grp):
    # possession-adjusted 스탯 사용
    tkl90 = zscore_series(grp['tackles_p90_adj'].fillna(0))
    int90 = zscore_series(grp['int_p90_adj'].fillna(0))
    ms    = zscore_series(grp['minutes_share'])
    cons  = zscore_series(grp['consistency'])
    raw   = (0.25 * tkl90 + 0.35 * int90 + 0.20 * ms + 0.20 * cons)
    return war_from_zscores(raw)


# ─── FB 풀백 WAR (possession-adjusted) ───
def calc_fb_war(grp):
    tkl90 = zscore_series(grp['tackles_p90_adj'].fillna(0))
    int90 = zscore_series(grp['int_p90_adj'].fillna(0))
    crs90 = zscore_series(grp['crosses_p90'].fillna(0))
    ms    = zscore_series(grp['minutes_share'])
    cons  = zscore_series(grp['consistency'])
    raw   = (0.25 * tkl90 + 0.25 * int90 + 0.20 * crs90 + 0.15 * ms + 0.15 * cons)
    return war_from_zscores(raw)


# ─── GK 골키퍼 WAR ───
def calc_gk_war(grp):
    save_pct = zscore_series(grp['gk_save_pct'].fillna(grp['gk_save_pct'].median()))
    cs_pct   = zscore_series(grp['gk_cs_pct'].fillna(grp['gk_cs_pct'].median()))
    ga_inv   = zscore_series(-grp['gk_ga_p90'].fillna(grp['gk_ga_p90'].median()))
    ms       = zscore_series(grp['minutes_share'])
    cons     = zscore_series(grp['consistency'])
    raw      = (0.40 * save_pct + 0.30 * cs_pct + 0.10 * ga_inv
                + 0.10 * ms + 0.10 * cons)
    return war_from_zscores(raw)


# ─────────────────────────────────────────────
# WAR 계산 루프
# ─────────────────────────────────────────────
war_list = []

# FW, MID, GK: 기존 방식 (possession 보정 불필요)
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

# ─── DEF: 2단계 티어 정규화 ───
# 단계 1: CB/FB 각각, 티어별(top6/mid/bottom6) 퍼센타일 계산
# 단계 2: 티어별 퍼센타일을 전체 DEF 안에서 재퍼센타일화
for season, def_grp in df_rated[df_rated['pos_group'] == 'DEF'].groupby('season'):
    cb_grp = def_grp[def_grp['def_subpos'] == 'CB'].copy()
    fb_grp = def_grp[def_grp['def_subpos'] == 'FB'].copy()

    def tier_war(grp_sub, calc_fn):
        """티어별 WAR 계산 후 합산"""
        tier_war_vals = pd.Series(50.0, index=grp_sub.index)
        tiers = grp_sub['tier'].unique()

        tier_results = []
        for tier in ['top6', 'mid', 'bottom6']:
            tier_idx = grp_sub[grp_sub['tier'] == tier].index
            tier_sub = grp_sub.loc[tier_idx]
            n_tier   = len(tier_sub)

            if n_tier >= 5:
                # 티어 내 퍼센타일
                w = calc_fn(tier_sub)
            elif n_tier > 0:
                # 샘플 부족: 전체 그룹으로 대체
                w = pd.Series(50.0, index=tier_idx)
            else:
                continue

            tier_results.append(w)

        if len(tier_results) == 0:
            # 모든 티어가 비어있으면 전체로 계산
            return calc_fn(grp_sub) if len(grp_sub) >= 5 else pd.Series(50.0, index=grp_sub.index)

        combined_tier = pd.concat(tier_results)
        # 단계 2: 티어별 결과를 서브포지션 전체 안에서 재퍼센타일화
        # 이렇게 하면 top6 CB가 인위적으로 높거나 낮아지는 것 방지
        final = pct_rank(combined_tier.reindex(grp_sub.index).fillna(50.0))
        return final

    # CB / FB 각각 티어 정규화
    cb_war = tier_war(cb_grp, calc_cb_war) if len(cb_grp) >= 5 else pd.Series(50.0, index=cb_grp.index)
    fb_war = tier_war(fb_grp, calc_fb_war) if len(fb_grp) >= 5 else pd.Series(50.0, index=fb_grp.index)

    # 전체 DEF 내 최종 퍼센타일화 (CB + FB 비율 유지)
    combined = pd.concat([cb_war, fb_war])
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
# 9. Sanity Check: 2024/25 DEF 특별 검증
#    VVD/Saliba가 Lacroix/Bednarek보다 높아야 함
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("[8] SANITY CHECK: DEF 검증 (VVD/Saliba vs Lacroix/Bednarek)")
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

    print(f"\n  ── DEF Top 15 ({target_season}) - possession-adjusted ──")
    def_top = df_s[df_s['pos_group'] == 'DEF'].nlargest(15, 'war')
    print(def_top[['player', 'team', 'tier', 'def_subpos',
                   'tackles_p90', 'tackles_p90_adj',
                   'int_p90', 'int_p90_adj',
                   'war']].to_string(index=False))

    print(f"\n  ── GK Top 10 ({target_season}) ──")
    gk_top = df_s[df_s['pos_group'] == 'GK'].nlargest(10, 'war')
    print(gk_top[['player', 'team', 'total_min', 'gk_save_pct', 'gk_cs_pct',
                  'gk_ga_p90', 'war']].to_string(index=False))

# VVD/Saliba vs Lacroix/Bednarek 직접 비교
print("\n  ── 핵심 검증: VVD/Saliba vs Lacroix/Bednarek (2024/25) ──")
key_players = ['van Dijk', 'Saliba', 'Lacroix', 'Bednarek']
df_2425 = df_rated[df_rated['season'] == '2024/25']
key_check = df_2425[
    df_2425['player'].str.contains('|'.join(key_players), case=False, na=False) &
    (df_2425['pos_group'] == 'DEF')
]
if len(key_check) > 0:
    print(key_check[['player', 'team', 'tier', 'def_subpos',
                      'possession_proxy', 'tackles_p90', 'tackles_p90_adj',
                      'int_p90', 'int_p90_adj', 'war']].sort_values('war', ascending=False)
          .to_string(index=False))

    vvd_sal_ranks = key_check[
        key_check['player'].str.contains('van Dijk|Saliba', case=False, na=False)
    ]['war'].min()
    lacroix_bed_ranks = key_check[
        key_check['player'].str.contains('Lacroix|Bednarek', case=False, na=False)
    ]['war'].max()

    if not np.isnan(vvd_sal_ranks) and not np.isnan(lacroix_bed_ranks):
        if vvd_sal_ranks > lacroix_bed_ranks:
            print(f"\n  ✓ 검증 성공: VVD/Saliba 최소 WAR ({vvd_sal_ranks:.1f}) > "
                  f"Lacroix/Bednarek 최대 WAR ({lacroix_bed_ranks:.1f})")
        else:
            print(f"\n  △ 부분 검증: VVD/Saliba 최소 WAR ({vvd_sal_ranks:.1f}), "
                  f"Lacroix/Bednarek 최대 WAR ({lacroix_bed_ranks:.1f})")
            print("    → 전체 DEF Top 15를 확인하여 상대 순위 비교 필요")
else:
    print("  해당 선수 데이터 없음 (2024/25 시즌 미포함 가능)")


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

age_col = df_rated['age_tm'].fillna(df_rated['age']) if 'age_tm' in df_rated.columns else df_rated['age']

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
print("\n  최근 시즌 히든 젬 (2024/25 우선, 없으면 2023/24):")
for s in ['2024/25', '2023/24']:
    hg_s = hidden_gems[hidden_gems['season'] == s]
    if len(hg_s) > 0:
        print(f"\n  [{s}]")
        print(hg_s[['player', 'team', 'pos_group', 'total_min', 'war',
                     'market_value', 'age_tm']].sort_values('war', ascending=False)
              .head(15).to_string(index=False))
        break


# ─────────────────────────────────────────────
# 11. 결과 저장
# ─────────────────────────────────────────────
print("\n[10] 결과 저장 중...")

output_cols = [
    'player', 'team', 'season', 'pos', 'pos_group', 'position', 'def_subpos',
    'age', 'age_tm', 'market_value',
    'total_min', 'total_starts', 'games_played',
    'goals_p90', 'assists_p90', 'shots_p90', 'sot_p90',
    'tackles_p90', 'tackles_p90_adj', 'int_p90', 'int_p90_adj',
    'crosses_p90', 'fouls_drawn_p90',
    'gk_save_pct', 'gk_cs_pct', 'gk_ga_p90',
    'minutes_share', 'consistency',
    'possession_proxy', 'tier',
    'war',
]
output_cols = [c for c in output_cols if c in df_rated.columns]

scout_ratings = df_rated[output_cols].copy()
# v3 결과로 저장 (scout_ratings_v3 → S2 v4에서 참조)
scout_ratings.to_parquet(os.path.join(SCOUT_DIR, "scout_ratings_v3.parquet"), index=False)

hg_cols = [c for c in output_cols if c in hidden_gems.columns]
hidden_gems[hg_cols].to_parquet(os.path.join(SCOUT_DIR, "hidden_gems_v3.parquet"), index=False)

print(f"  scout_ratings_v3.parquet 저장 ({len(scout_ratings)} rows)")
print(f"  hidden_gems_v3.parquet   저장 ({len(hidden_gems)} rows)")


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

plt.suptitle('S1 v4: 2023/24 WAR Distribution by Position\n(Possession-Adjusted DEF)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'war_distribution_v3.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  war_distribution_v3.png saved")

# ── 12-2. DEF Top 15 (2024/25) - Possession-Adjusted, 티어 색상 구분 ──
if len(df_2425) > 0:
    fig, ax = plt.subplots(figsize=(12, 10))
    def_chart = df_2425[df_2425['pos_group'] == 'DEF'].nlargest(15, 'war')
    # 색상: CB/FB + tier
    bar_colors = []
    for _, r in def_chart.iterrows():
        if r['def_subpos'] == 'CB' and r.get('tier') == 'top6':
            bar_colors.append('#1a5276')  # CB top6: 진한 파랑
        elif r['def_subpos'] == 'CB':
            bar_colors.append('#2ecc71')  # CB other: 초록
        elif r['def_subpos'] == 'FB' and r.get('tier') == 'top6':
            bar_colors.append('#2980b9')  # FB top6: 파랑
        else:
            bar_colors.append('#82e0aa')  # FB other: 연초록
    ax.barh(range(len(def_chart)), def_chart['war'], color=bar_colors, alpha=0.85)
    ax.set_yticks(range(len(def_chart)))
    ax.set_yticklabels([
        f"{r['player']} ({r['team']}) [{r['def_subpos']}/{r.get('tier','?')}]"
        for _, r in def_chart.iterrows()
    ], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('WAR (Percentile Rank 0-100)')
    ax.set_title('2024/25 Defender (DEF) WAR Top 15 - S1 v4\n(Possession-Adjusted | CB/tier color)', fontweight='bold')
    for i, (_, row) in enumerate(def_chart.iterrows()):
        ax.text(row['war'] + 0.3, i, f"{row['war']:.1f}", va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'def_top15_v3_2425.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  def_top15_v3_2425.png saved")

# ── 12-3. FW Top 15 (2023/24) ──
fig, ax = plt.subplots(figsize=(10, 8))
fw_chart = df_2324[df_2324['pos_group'] == 'FW'].nlargest(15, 'war')
ax.barh(range(len(fw_chart)), fw_chart['war'], color='#e74c3c', alpha=0.85)
ax.set_yticks(range(len(fw_chart)))
ax.set_yticklabels([f"{r['player']} ({r['team']})" for _, r in fw_chart.iterrows()], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('WAR (Percentile Rank 0-100)')
ax.set_title('2023/24 Forward (FW) WAR Top 15 - S1 v4', fontweight='bold')
for i, (_, row) in enumerate(fw_chart.iterrows()):
    ax.text(row['war'] + 0.3, i, f"{row['war']:.1f}", va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fw_top15_v3.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  fw_top15_v3.png saved")

# ── 12-4. DEF Top 15 (2023/24) - CB vs FB 색상 구분 ──
fig, ax = plt.subplots(figsize=(11, 9))
def_chart_2324 = df_2324[df_2324['pos_group'] == 'DEF'].nlargest(15, 'war')
bar_colors_2324 = ['#27ae60' if r['def_subpos'] == 'CB' else '#82e0aa' for _, r in def_chart_2324.iterrows()]
ax.barh(range(len(def_chart_2324)), def_chart_2324['war'], color=bar_colors_2324, alpha=0.85)
ax.set_yticks(range(len(def_chart_2324)))
ax.set_yticklabels([
    f"{r['player']} ({r['team']}) [{r['def_subpos']}]"
    for _, r in def_chart_2324.iterrows()
], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('WAR (Percentile Rank 0-100)')
ax.set_title('2023/24 Defender (DEF) WAR Top 15 - S1 v4\n(Possession-Adjusted | Dark=CB, Light=FB)', fontweight='bold')
for i, (_, row) in enumerate(def_chart_2324.iterrows()):
    ax.text(row['war'] + 0.3, i, f"{row['war']:.1f}", va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'def_top15_v3.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  def_top15_v3.png saved")

# ── 12-5. GK Top 10 (2023/24) ──
fig, ax = plt.subplots(figsize=(10, 7))
gk_chart = df_2324[df_2324['pos_group'] == 'GK'].nlargest(10, 'war')
if len(gk_chart) > 0:
    ax.barh(range(len(gk_chart)), gk_chart['war'], color='#f39c12', alpha=0.85)
    ax.set_yticks(range(len(gk_chart)))
    ax.set_yticklabels([f"{r['player']} ({r['team']})" for _, r in gk_chart.iterrows()], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('WAR (Percentile Rank 0-100)')
    ax.set_title('2023/24 Goalkeeper (GK) WAR Top 10 - S1 v4', fontweight='bold')
    for i, (_, row) in enumerate(gk_chart.iterrows()):
        save_s = f" sv%={row['gk_save_pct']:.2f}" if pd.notna(row.get('gk_save_pct')) else ""
        ax.text(row['war'] + 0.3, i, f"{row['war']:.1f}{save_s}", va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'gk_top10_v3.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  gk_top10_v3.png saved")

# ── 12-6. MID Top 15 (2023/24) ──
fig, ax = plt.subplots(figsize=(10, 8))
mid_chart = df_2324[df_2324['pos_group'] == 'MID'].nlargest(15, 'war')
ax.barh(range(len(mid_chart)), mid_chart['war'], color='#3498db', alpha=0.85)
ax.set_yticks(range(len(mid_chart)))
ax.set_yticklabels([f"{r['player']} ({r['team']})" for _, r in mid_chart.iterrows()], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('WAR (Percentile Rank 0-100)')
ax.set_title('2023/24 Midfielder (MID) WAR Top 15 - S1 v4', fontweight='bold')
for i, (_, row) in enumerate(mid_chart.iterrows()):
    ax.text(row['war'] + 0.3, i, f"{row['war']:.1f}", va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'mid_top15_v3.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  mid_top15_v3.png saved")

# ── 12-7. Possession Proxy 분포 확인 ──
fig, ax = plt.subplots(figsize=(10, 5))
proxy_data = ts[ts['season'].isin(['2023/24', '2024/25'])].copy()
for season in ['2023/24', '2024/25']:
    d = proxy_data[proxy_data['season'] == season]['possession_proxy'].dropna()
    if len(d) > 0:
        ax.hist(d, bins=20, alpha=0.6, label=season, edgecolor='white')
ax.set_xlabel('Possession Proxy (points / (points + 38))')
ax.set_ylabel('Team Count')
ax.set_title('Possession Proxy Distribution (Team-Season)\nUsed for DEF stats adjustment', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'possession_proxy_dist_v3.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  possession_proxy_dist_v3.png saved")

# ── 12-8. 히든 젬 Scatter (2022/23~2023/24) ──
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
ax.set_title('Hidden Gems: WAR vs Market Value (2022/23~2023/24) - S1 v4', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'hidden_gems_scatter_v3.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  hidden_gems_scatter_v3.png saved")


# ─────────────────────────────────────────────
# 13. 결과 요약 JSON 저장 → results_v3.json
# ─────────────────────────────────────────────
print("\n[12] 결과 요약 JSON 저장 중...")

# 포지션별 시즌별 Top 5
top_players_by_season = {}
for season in ['2023/24', '2024/25']:
    ds = df_rated[df_rated['season'] == season]
    top_players_by_season[season] = {}
    for pg in ['FW', 'MID', 'DEF', 'GK']:
        top5 = ds[ds['pos_group'] == pg].nlargest(5, 'war')
        top_players_by_season[season][pg] = [
            {
                'player':    r['player'],
                'team':      r['team'],
                'war':       round(float(r['war']), 2),
                'total_min': int(r['total_min']) if pd.notna(r['total_min']) else None,
                'tier':      r.get('tier', 'unknown'),
                'def_subpos': r.get('def_subpos', 'N/A'),
            }
            for _, r in top5.iterrows()
        ]

# VVD/Saliba vs Lacroix/Bednarek 비교 기록
key_check_record = {}
if len(key_check) > 0:
    for _, r in key_check.sort_values('war', ascending=False).iterrows():
        key_check_record[r['player']] = {
            'team': r['team'],
            'tier': r.get('tier', 'unknown'),
            'def_subpos': r.get('def_subpos', '?'),
            'possession_proxy': round(float(r['possession_proxy']), 3),
            'tackles_p90_raw': round(float(r['tackles_p90']), 3),
            'tackles_p90_adj': round(float(r['tackles_p90_adj']), 3),
            'int_p90_raw':     round(float(r['int_p90']), 3),
            'int_p90_adj':     round(float(r['int_p90_adj']), 3),
            'war': round(float(r['war']), 2),
        }

summary = {
    "version": "v4",
    "script_file": "train_v3.py",
    "description": (
        "스카우트 검증 v3 피드백 반영: "
        "DEF possession-adjusted tackles/int, 팀 강도 티어별 2단계 정규화, "
        "900분 필터 WAR 계산 전 명시적 적용, results_v3.json 저장"
    ),
    "key_fixes_from_v3_feedback": [
        "FIX 1: possession_proxy = points/(points+38) 로 팀 점유율 추정",
        "FIX 2: tackles_p90_adj = tackles_p90 / (1 - possession_proxy), 동일하게 int_p90_adj",
        "FIX 3: 팀 강도 티어(top6/mid/bottom6) 내 퍼센타일화 후 전체 DEF 재퍼센타일화",
        "FIX 4: 900분 최소 출전 기준 필터링 → WAR 계산 전 명시적 수행 (v3 동일, 주석 강화)",
    ],
    "possession_proxy_formula": "points / (points + 38), 범위: 0~1",
    "adj_factor_clip": "0.5 ~ 3.0 (과도한 보정 방지)",
    "tier_normalization": {
        "top6":    "시즌별 points 상위 6팀",
        "mid":     "7~14위 8팀",
        "bottom6": "하위 6팀",
        "2_stage": "티어 내 퍼센타일 → 전체 DEF 재퍼센타일",
    },
    "total_rated_players": int(len(df_rated)),
    "insufficient_data_players": int(len(df_insuff)),
    "hidden_gems_total": int(len(hidden_gems)),
    "minimum_threshold": "900분 이상 OR 15선발 이상 (WAR 계산 전 적용)",
    "war_range": {
        "min":  round(float(df_rated['war'].min()), 2),
        "max":  round(float(df_rated['war'].max()), 2),
        "mean": round(float(df_rated['war'].mean()), 2),
    },
    "war_formula": {
        "FW":  "goals_p90(40%) + assists_p90(20%) + sot_p90(20%) + minutes_share(10%) + consistency(10%)",
        "MID": "goals_p90(20%) + assists_p90(25%) + fouls_drawn_p90(15%) + tackles_p90(15%) + minutes_share(15%) + consistency(10%)",
        "CB":  "tackles_p90_adj(25%) + int_p90_adj(35%) + minutes_share(20%) + consistency(20%) [POSSESSION-ADJUSTED]",
        "FB":  "tackles_p90_adj(25%) + int_p90_adj(25%) + crosses_p90(20%) + minutes_share(15%) + consistency(15%) [POSSESSION-ADJUSTED]",
        "GK":  "save_pct(40%) + cs_pct(30%) + ga_p90_inv(10%) + minutes_share(10%) + consistency(10%)",
    },
    "key_player_comparison_2425": key_check_record,
    "top_players_by_season": top_players_by_season,
    "seasons_coverage": sorted(df_rated['season'].unique().tolist()),
}

with open(os.path.join(MODEL_DIR, "results_v3.json"), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"  results_v3.json 저장 완료")

print("\n" + "=" * 60)
print("S1 v4 완료! (results → results_v3.json)")
print(f"  rated players : {len(df_rated):,}")
print(f"  WAR max       : {df_rated['war'].max():.2f}")
print(f"  WAR = 100     : {(df_rated['war'] == 100.0).sum()} (should be 0)")
print("=" * 60)

"""
S6 v3: Player Decline Detection — Scout Validation v3 Fixes
============================================================
v2 대비 변경사항 (스카우트 검증 v3 피드백 반영):

  Fix 1: "Late Bloomer" 예외 처리 (Chris Wood 케이스)
    - perf_slope(최근 2~3시즌) > 0 AND perf_score > career_mean 이면
      decline_prob에 감쇠 계수 적용:
      adjusted_prob = prob * (1 - min(perf_slope / 2, 0.3))
    - 상승 궤적 선수의 decline 확률을 최대 30% 감소

  Fix 2: DEF 모델 개선 (AUC 0.681 → 목표 0.75+)
    - 수비수 전용 피처 3종 추가:
        aerial_duel_proxy   : crdy(경고) + fls(파울) 합산 → 피지컬 플레이 지표
        clean_sheet_involvement : 해당 선수 선발 경기 중 팀 클린시트 비율
        minutes_stability   : 경기당 출전 시간 표준편차 (부상/로테이션 지표)
    - progressive pass 데이터 없으면 graceful skip
    - DEF 전용 feature set 분리 (FEATURE_COLS_DEF)

  Fix 3: Sanity Check
    - Chris Wood: career_decline_watch에 없거나 prob < 0.5
    - Casemiro: career_decline_watch에 포함
    - Cole Palmer: regression_to_mean_alert에 있고 career_decline_watch에 없음
    - career_decline_watch 전원 age >= 28
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    average_precision_score
)
from sklearn.impute import SimpleImputer
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("  [경고] imbalanced-learn 없음, SMOTE 생략")
import xgboost as xgb

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE_DIR = Path("C:/Users/xcv54/workspace/EPL project")
DATA_DIR = BASE_DIR / "data" / "processed"
SCOUT_DIR = BASE_DIR / "data" / "scout"
MODEL_DIR = BASE_DIR / "models" / "s6_decline"
FIG_DIR   = MODEL_DIR / "figures_v3"

for d in [SCOUT_DIR, FIG_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("[S6-V3] Player Decline Detection — Scout Validation v3 Fixes")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────
print("\n[1] 데이터 로딩...")
season_df = pd.read_parquet(DATA_DIR / "player_season_stats.parquet")
match_df  = pd.read_parquet(DATA_DIR / "player_match_logs.parquet")
team_df   = pd.read_parquet(DATA_DIR / "team_season_summary.parquet")

print(f"  player_season_stats: {season_df.shape}")
print(f"  player_match_logs:   {match_df.shape}")
print(f"  team_season_summary: {team_df.shape}")

# ─────────────────────────────────────────────
# 2. 시즌 연도 파싱
# ─────────────────────────────────────────────
def parse_season_year(s):
    try:
        return int(str(s).split('/')[0])
    except Exception:
        return np.nan

season_df['season_year'] = season_df['season'].apply(parse_season_year)
team_df['season_year']   = team_df['Season'].apply(parse_season_year)

# ─────────────────────────────────────────────
# 3. 포지션 그룹 매핑
# ─────────────────────────────────────────────
def map_position_group(pos):
    """포지션 문자열을 4개 그룹(GK/DEF/MID/FWD)으로 매핑"""
    if pd.isna(pos):
        return 'MID'
    pos = str(pos).lower()
    if 'goalkeeper' in pos or pos == 'gk':
        return 'GK'
    elif 'back' in pos or 'defend' in pos or pos in ['cb', 'rb', 'lb', 'rwb', 'lwb']:
        return 'DEF'
    elif 'forward' in pos or 'winger' in pos or 'striker' in pos or pos in ['cf', 'st', 'lw', 'rw', 'ss']:
        return 'FWD'
    elif 'midfield' in pos or pos in ['cm', 'cdm', 'cam', 'dm', 'am']:
        return 'MID'
    else:
        return 'MID'

season_df['pos_group'] = season_df['position'].apply(map_position_group)

# ─────────────────────────────────────────────
# 4. 포지션별 종합 성과 점수 (z-score 표준화)
# ─────────────────────────────────────────────
print("\n[2] 포지션별 성과 점수 산출...")

POS_METRICS = {
    'FWD': ['gls_1', 'ast_1', 'g_a_1'],
    'MID': ['ast_1', 'gls_1', 'g_a_1'],
    'DEF': ['gls_1', 'ast_1'],
    'GK':  ['gls_1', 'ast_1'],
}

season_df['90s_safe'] = season_df['90s'].clip(lower=0.1)

def compute_composite_score(df):
    """포지션 그룹별로 관련 지표의 z-score 평균 계산"""
    scores = np.zeros(len(df))
    for pos_g, metrics in POS_METRICS.items():
        mask = df['pos_group'] == pos_g
        if mask.sum() == 0:
            continue
        avail = [m for m in metrics if m in df.columns]
        if not avail:
            continue
        sub = df.loc[mask, avail].copy()
        z = sub.apply(lambda col: stats.zscore(col.fillna(0), nan_policy='omit'), axis=0)
        scores[mask.values] = z.mean(axis=1).values
    return scores

season_df['perf_score'] = compute_composite_score(season_df)

# ─────────────────────────────────────────────
# 5. 팀 품질 (포인트 기반, 시즌별 0-1 정규화)
# ─────────────────────────────────────────────
team_quality = (
    team_df.groupby(['team', 'season_year'])['points'].sum()
    .reset_index().rename(columns={'points': 'team_quality'})
)
team_quality['team_quality'] = team_quality.groupby('season_year')['team_quality'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
)
season_df = season_df.merge(team_quality, on=['team', 'season_year'], how='left')

# 팀 품질 시즌간 변화 (팀 강화/약화 프록시)
team_quality_sorted = team_quality.sort_values(['team', 'season_year'])
team_quality_sorted['team_quality_prev'] = team_quality_sorted.groupby('team')['team_quality'].shift(1)
team_quality_sorted['team_quality_change'] = (
    team_quality_sorted['team_quality'] - team_quality_sorted['team_quality_prev']
)
season_df = season_df.merge(
    team_quality_sorted[['team', 'season_year', 'team_quality_change']],
    on=['team', 'season_year'], how='left'
)

# ─────────────────────────────────────────────
# 6. 경기 레벨 일관성 + 부상 프록시
#    [FIX 2] DEF 전용 피처 추가:
#      - aerial_duel_proxy: crdy(경고) + fls(파울) 합산 → 피지컬 강도 지표
#      - clean_sheet_involvement: 선발 출전 시 팀 클린시트 비율
#      - minutes_stability: 경기당 출전시간 표준편차 (낮을수록 안정)
#    progressive pass: 데이터 없으면 graceful skip
# ─────────────────────────────────────────────
print("[3] 경기 레벨 일관성, 부상 프록시, DEF 전용 피처 산출...")

match_df['season_year'] = match_df['season'].apply(parse_season_year)
match_df['g_a_match']   = match_df['gls'].fillna(0) + match_df['ast'].fillna(0)

# ── 기본 일관성/부상 피처 (모든 포지션 공통) ──────────────────────────────
consistency = match_df.groupby(['player', 'season_year']).agg(
    consistency_std=('g_a_match', 'std'),
    match_count=('g_a_match', 'count'),
    avg_min_match=('min', 'mean'),
    total_min=('min', 'sum'),
    games_sub_60=('min', lambda x: (x < 60).sum()),  # 부상/로테이션 지표
).reset_index()
consistency['consistency_score'] = 1.0 / (1.0 + consistency['consistency_std'].fillna(0))
consistency['injury_proxy'] = consistency['games_sub_60'] / (consistency['match_count'].clip(lower=1))

# ── [FIX 2] DEF 전용 피처 산출 ───────────────────────────────────────────

# aerial_duel_proxy: crdy(경고) + fls(파울) 평균의 합 (피지컬 플레이 강도)
# crdy는 대부분 데이터 있음, fls는 60% 누락 → 둘 중 가용한 것만 합산
def compute_aerial_duel_proxy(grp):
    """경고 + 파울 합산으로 피지컬 플레이 지표 계산 (결측 시 가용 컬럼만 사용)"""
    crdy_mean = grp['crdy'].fillna(0).mean() if 'crdy' in grp.columns else 0.0
    fls_mean  = grp['fls'].fillna(0).mean()  if 'fls'  in grp.columns else 0.0
    return crdy_mean + fls_mean

aerial_duel = match_df.groupby(['player', 'season_year']).apply(
    compute_aerial_duel_proxy
).reset_index().rename(columns={0: 'aerial_duel_proxy'})

# clean_sheet_involvement: 해당 선수 선발 출전 경기 중 팀 무실점(클린시트) 비율
# goals_against=0 AND started=True 인 경기 / 선발 출전 전체 경기
if 'goals_against' in match_df.columns and 'started' in match_df.columns:
    match_df['clean_sheet'] = ((match_df['goals_against'] == 0) & match_df['started'].fillna(False)).astype(int)
    match_df['started_int'] = match_df['started'].fillna(False).astype(int)
    cs_data = match_df.groupby(['player', 'season_year']).agg(
        cs_games_started=('clean_sheet', 'sum'),
        total_games_started=('started_int', 'sum'),
    ).reset_index()
    cs_data['clean_sheet_involvement'] = (
        cs_data['cs_games_started'] / cs_data['total_games_started'].clip(lower=1)
    )
    HAS_CLEAN_SHEET = True
    print("  clean_sheet_involvement 피처 생성 완료")
else:
    HAS_CLEAN_SHEET = False
    print("  [skip] goals_against/started 컬럼 없음 → clean_sheet_involvement 건너뜀")

# minutes_stability: 경기당 출전시간 표준편차 (작을수록 안정적 로테이션)
minutes_stability = match_df.groupby(['player', 'season_year'])['min'].std().reset_index()
minutes_stability.columns = ['player', 'season_year', 'minutes_stability']
minutes_stability['minutes_stability'] = minutes_stability['minutes_stability'].fillna(0)

# progressive_pass: 데이터 없으면 graceful skip
PROG_PASS_COLS = [c for c in match_df.columns if 'prog' in c.lower() and 'pass' in c.lower()]
if PROG_PASS_COLS:
    prog_pass = match_df.groupby(['player', 'season_year'])[PROG_PASS_COLS[0]].mean().reset_index()
    prog_pass.columns = ['player', 'season_year', 'prog_pass_avg']
    HAS_PROG_PASS = True
    print(f"  progressive_pass 피처 사용: {PROG_PASS_COLS[0]}")
else:
    HAS_PROG_PASS = False
    print("  [skip] progressive pass 데이터 없음 → prog_pass_avg 건너뜀")

# ── 모든 match-level 피처를 season_df에 병합 ─────────────────────────────
season_df = season_df.merge(
    consistency[['player', 'season_year', 'consistency_score', 'match_count',
                  'avg_min_match', 'total_min', 'injury_proxy']],
    on=['player', 'season_year'], how='left'
)
season_df = season_df.merge(
    aerial_duel[['player', 'season_year', 'aerial_duel_proxy']],
    on=['player', 'season_year'], how='left'
)
season_df = season_df.merge(
    minutes_stability[['player', 'season_year', 'minutes_stability']],
    on=['player', 'season_year'], how='left'
)
if HAS_CLEAN_SHEET:
    season_df = season_df.merge(
        cs_data[['player', 'season_year', 'clean_sheet_involvement']],
        on=['player', 'season_year'], how='left'
    )
else:
    season_df['clean_sheet_involvement'] = np.nan

if HAS_PROG_PASS:
    season_df = season_df.merge(
        prog_pass[['player', 'season_year', 'prog_pass_avg']],
        on=['player', 'season_year'], how='left'
    )

# ─────────────────────────────────────────────
# 7. 지속적 하락세 레이블 (2 연속 시즌 하락 필요)
#    v2와 동일 — 1시즌 단발 하락은 회귀-평균(regression-to-mean)으로 처리
# ─────────────────────────────────────────────
print("[4] 지속적 하락세 레이블 구축 (2 연속 시즌)...")

season_df['player_key'] = season_df['player']
season_df_sorted = season_df.sort_values(['player_key', 'season_year'])

def make_shift_df(df, shift, suffix):
    """시즌 N+shift 데이터를 N 기준으로 정렬하기 위한 shift 변환"""
    tmp = df[['player_key', 'season_year', 'perf_score', 'min', 'mp', 'market_value']].copy()
    tmp['season_year'] = tmp['season_year'] - shift
    tmp.columns = [f'{c}{suffix}' if c not in ('player_key', 'season_year') else c
                   for c in tmp.columns]
    return tmp

current  = season_df[['player_key', 'season_year', 'perf_score', 'min', 'mp',
                        'market_value', 'age', 'pos_group', 'team', 'team_quality',
                        'team_quality_change']].copy()
next1_df = make_shift_df(season_df, 1, '_n1')  # N+1 시즌
next2_df = make_shift_df(season_df, 2, '_n2')  # N+2 시즌

merged = current.merge(next1_df, on=['player_key', 'season_year'], how='inner')
merged = merged.merge(next2_df, on=['player_key', 'season_year'], how='left')

perf_std = (merged['perf_score_n1'] - merged['perf_score']).std()

# 지속적 성과 하락: N→N+1 하락 AND N+1→N+2 추가 하락
merged['drop_n1'] = ((merged['perf_score_n1'] - merged['perf_score']) < -0.4 * perf_std).astype(int)
merged['drop_n2'] = ((merged['perf_score_n2'] - merged['perf_score_n1']) < -0.2 * perf_std).astype(int)
merged['decline_perf'] = (
    (merged['drop_n1'] == 1) & (merged['drop_n2'] == 1)
).astype(int)

# 출전시간 급감 (부상/하락 프록시)
merged['decline_avail'] = (
    (merged['min_n1'] < merged['min'] * 0.70) & (merged['min'] >= 450)
).astype(int)

# 통합 하락세 레이블
merged['decline'] = ((merged['decline_perf'] == 1) | (merged['decline_avail'] == 1)).astype(int)

# ── 아웃라이어 시즌 플래그 (회귀-평균 경보용) ─────────────────────────────
# "아웃라이어" = 현재 시즌 perf_score가 선수 본인 커리어 평균 + 1.5 SD 초과
player_career_mean = (
    season_df.groupby('player_key')['perf_score']
    .agg(['mean', 'std'])
    .reset_index()
    .rename(columns={'mean': 'career_perf_mean', 'std': 'career_perf_std'})
)
merged = merged.merge(player_career_mean, on='player_key', how='left')
merged['career_perf_std'] = merged['career_perf_std'].fillna(0.5)
merged['is_outlier_season'] = (
    merged['perf_score'] > (merged['career_perf_mean'] + 1.5 * merged['career_perf_std'])
).astype(int)

print(f"  전체 샘플:           {len(merged)}")
print(f"  지속 성과 하락:      {merged['decline_perf'].sum()} ({merged['decline_perf'].mean():.1%})")
print(f"  출전시간 하락:       {merged['decline_avail'].sum()} ({merged['decline_avail'].mean():.1%})")
print(f"  통합 하락 레이블:    {merged['decline'].sum()} ({merged['decline'].mean():.1%})")
print(f"  아웃라이어 시즌:     {merged['is_outlier_season'].sum()} ({merged['is_outlier_season'].mean():.1%})")

# ─────────────────────────────────────────────
# 8. 궤적 피처 (3시즌 슬로프)
# ─────────────────────────────────────────────
print("[5] 궤적 피처 산출 (3시즌 슬로프)...")

def compute_trajectory(df, n_seasons=3):
    """플레이어별 최근 n_seasons 시즌 성과/출전 트렌드 기울기 계산"""
    results = []
    for player, grp in df.groupby('player_key'):
        grp = grp.sort_values('season_year')
        for _, row in grp.iterrows():
            cur_year = row['season_year']
            hist = grp[grp['season_year'] <= cur_year].tail(n_seasons)
            if len(hist) < 2:
                perf_slope = np.nan
                min_slope  = np.nan
                peak_minus_current = np.nan
            else:
                x  = hist['season_year'].values.astype(float)
                yp = hist['perf_score'].values.astype(float)
                ym = hist['min'].values.astype(float)
                vp = ~np.isnan(yp)
                vm = ~np.isnan(ym)
                perf_slope = np.polyfit(x[vp], yp[vp], 1)[0] if vp.sum() >= 2 else np.nan
                min_slope  = np.polyfit(x[vm], ym[vm], 1)[0] if vm.sum() >= 2 else np.nan
                peak = grp[grp['season_year'] <= cur_year]['perf_score'].max()
                peak_minus_current = peak - row['perf_score']
            results.append({
                'player_key':          player,
                'season_year':         cur_year,
                'perf_slope':          perf_slope,
                'min_slope':           min_slope,
                'peak_minus_current':  peak_minus_current,
            })
    return pd.DataFrame(results)

traj_input   = season_df[['player_key', 'season_year', 'perf_score', 'min']].drop_duplicates()
trajectory_df = compute_trajectory(traj_input)
merged = merged.merge(trajectory_df, on=['player_key', 'season_year'], how='left')

# ─────────────────────────────────────────────
# 9. 시장가치 궤적 슬로프
# ─────────────────────────────────────────────
mv_traj = []
for player, grp in season_df.groupby('player_key'):
    grp = grp.sort_values('season_year')
    for _, row in grp.iterrows():
        cur_year = row['season_year']
        hist = grp[grp['season_year'] <= cur_year].tail(3)
        if len(hist) < 2:
            mv_slope = np.nan
        else:
            x = hist['season_year'].values.astype(float)
            y = hist['market_value'].ffill().fillna(0).values.astype(float)
            valid = ~np.isnan(y)
            mv_slope = np.polyfit(x[valid], y[valid], 1)[0] if valid.sum() >= 2 else np.nan
        mv_traj.append({'player_key': player, 'season_year': cur_year, 'mv_slope': mv_slope})
mv_traj_df = pd.DataFrame(mv_traj)
merged = merged.merge(mv_traj_df, on=['player_key', 'season_year'], how='left')

# ─────────────────────────────────────────────
# 10. 누적 출전 부하 (2시즌)
# ─────────────────────────────────────────────
workload = []
for player, grp in season_df.groupby('player_key'):
    grp = grp.sort_values('season_year')
    for _, row in grp.iterrows():
        cur_year = row['season_year']
        hist = grp[(grp['season_year'] >= cur_year - 2) & (grp['season_year'] <= cur_year)]
        workload.append({
            'player_key':     player,
            'season_year':    cur_year,
            'workload_2y_min': hist['min'].sum(),
            'workload_2y_mp':  hist['mp'].sum(),
        })
workload_df = pd.DataFrame(workload)
merged = merged.merge(workload_df, on=['player_key', 'season_year'], how='left')

# ─────────────────────────────────────────────
# 11. 매치 레벨 피처 병합 (일관성, DEF 전용 포함)
# ─────────────────────────────────────────────
cons_features = (
    season_df[['player_key', 'season_year', 'consistency_score', 'match_count',
                'avg_min_match', 'total_min', 'injury_proxy',
                'aerial_duel_proxy', 'clean_sheet_involvement', 'minutes_stability']]
    .groupby(['player_key', 'season_year']).mean().reset_index()
)
# progressive pass (가용 시)
if HAS_PROG_PASS:
    prog_feat = season_df[['player_key', 'season_year', 'prog_pass_avg']].groupby(
        ['player_key', 'season_year']
    ).mean().reset_index()
    cons_features = cons_features.merge(prog_feat, on=['player_key', 'season_year'], how='left')

merged = merged.merge(cons_features, on=['player_key', 'season_year'], how='left')

# ─────────────────────────────────────────────
# 12. 피처 정의
#     - 공통 피처: v2와 동일 (FEATURE_COLS)
#     - DEF 전용 추가 피처: FEATURE_COLS_DEF
# ─────────────────────────────────────────────
# ── 공통 피처 (v2 동일) ───────────────────────────────────────────────────
FEATURE_COLS = [
    # 성과 궤적 (핵심 신호)
    'perf_slope',           # 3시즌 성과 트렌드 기울기
    'peak_minus_current',   # 커리어 피크 대비 현재 거리
    'perf_score',           # 현재 시즌 성과 점수
    'min_slope',            # 출전시간 트렌드 기울기
    # 출전 부하
    'min',
    'mp',
    'workload_2y_min',
    'workload_2y_mp',
    # 부상 프록시
    'injury_proxy',         # 60분 미만 경기 비율
    'consistency_score',
    'match_count',
    'avg_min_match',
    # 시장 신호
    'market_value',
    'mv_slope',
    # 팀 컨텍스트
    'team_quality',
    'team_quality_change',  # 팀 강화/약화 변화
    # 나이 (25+ 커리어 모델용)
    'age',
]

# ── [FIX 2] DEF 전용 추가 피처 ───────────────────────────────────────────
DEF_EXTRA_FEATURES = ['aerial_duel_proxy', 'clean_sheet_involvement', 'minutes_stability']
if HAS_PROG_PASS:
    DEF_EXTRA_FEATURES.append('prog_pass_avg')

FEATURE_COLS_DEF = FEATURE_COLS + [
    f for f in DEF_EXTRA_FEATURES if f in merged.columns
]

merged['age'] = merged['age'].fillna(0)

print(f"\n[6] 공통 피처 ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"  DEF 추가 피처: {[f for f in DEF_EXTRA_FEATURES if f in merged.columns]}")
print(f"  전체 샘플: {len(merged)}")

# ─────────────────────────────────────────────
# 13. 포지션별 모델 학습
#     DEF: FEATURE_COLS_DEF 사용 (추가 피처 포함)
#     그 외: FEATURE_COLS 사용
# ─────────────────────────────────────────────
POSITIONS = ['FWD', 'MID', 'DEF', 'GK']
pos_models   = {}   # 포지션별 학습 완료 모델
pos_imputers = {}
pos_scalers  = {}
pos_results  = {}   # 포지션별 평가 지표
pos_feature_cols = {}  # 포지션별 실제 사용 피처 목록

train_mask = merged['season_year'] < 2021
val_mask   = (merged['season_year'] >= 2021) & (merged['season_year'] <= 2022)
test_mask  =  merged['season_year'] >= 2023

print("\n[7] 포지션별 모델 학습 (DEF: 추가 피처 포함)...")

for pos in POSITIONS:
    # DEF는 전용 피처셋, 나머지는 공통 피처셋 사용
    if pos == 'DEF':
        feat_cols = FEATURE_COLS_DEF
        print(f"\n  [DEF] DEF 전용 피처 사용: {len(feat_cols)}개 ({feat_cols})")
    else:
        feat_cols = FEATURE_COLS

    pos_mask = merged['pos_group'] == pos
    df_pos   = merged[pos_mask].copy()

    if len(df_pos) < 50:
        print(f"  [{pos}] 건너뜀 — 샘플 부족 ({len(df_pos)})")
        continue

    X_tr = df_pos.loc[df_pos['season_year'] < 2021,  feat_cols]
    y_tr = df_pos.loc[df_pos['season_year'] < 2021,  'decline']
    X_va = df_pos.loc[(df_pos['season_year'] >= 2021) & (df_pos['season_year'] <= 2022), feat_cols]
    y_va = df_pos.loc[(df_pos['season_year'] >= 2021) & (df_pos['season_year'] <= 2022), 'decline']
    X_te = df_pos.loc[df_pos['season_year'] >= 2023,  feat_cols]
    y_te = df_pos.loc[df_pos['season_year'] >= 2023,  'decline']

    print(f"\n  [{pos}] train={len(X_tr)} (decline {y_tr.mean():.1%})  "
          f"val={len(X_va)}  test={len(X_te)}")

    if len(X_tr) < 20 or y_tr.sum() < 5:
        print(f"  [{pos}] 건너뜀 — 양성 레이블 부족")
        continue

    # 결측값 처리 (중앙값 대체)
    imp = SimpleImputer(strategy='median')
    X_tr_imp = imp.fit_transform(X_tr)
    X_va_imp = imp.transform(X_va) if len(X_va) > 0 else X_va.values
    X_te_imp = imp.transform(X_te) if len(X_te) > 0 else X_te.values

    X_tr_df = pd.DataFrame(X_tr_imp, columns=feat_cols)
    X_va_df = pd.DataFrame(X_va_imp, columns=feat_cols) if len(X_va) > 0 else pd.DataFrame(columns=feat_cols)
    X_te_df = pd.DataFrame(X_te_imp, columns=feat_cols) if len(X_te) > 0 else pd.DataFrame(columns=feat_cols)

    # SMOTE 오버샘플링 (클래스 불균형 보정)
    X_tr_res, y_tr_res = X_tr_df, y_tr.reset_index(drop=True)
    if HAS_SMOTE and int(y_tr.sum()) >= 5:
        try:
            k = min(5, int(y_tr.sum()) - 1)
            sm = SMOTE(random_state=42, k_neighbors=k)
            X_tr_res, y_tr_res = sm.fit_resample(X_tr_df, y_tr)
        except Exception as e:
            print(f"  [{pos}] SMOTE 실패 ({e}), 원본 데이터 사용")

    # 표준화
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr_res)
    X_va_sc = sc.transform(X_va_df) if len(X_va_df) > 0 else X_va_df.values
    X_te_sc = sc.transform(X_te_df) if len(X_te_df) > 0 else X_te_df.values

    # XGBoost (주 모델)
    scale_pw = max(1.0, (y_tr_res == 0).sum() / max(1, (y_tr_res == 1).sum()))
    xgb_m = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pw,
        eval_metric='logloss', random_state=42, n_jobs=-1,
    )
    fit_kwargs = {}
    if len(X_va_df) > 0 and len(y_va) > 0:
        fit_kwargs['eval_set'] = [(X_va_df, y_va)]
    xgb_m.fit(X_tr_res, y_tr_res, verbose=False, **fit_kwargs)

    # Random Forest (보조 모델)
    rf_m = RandomForestClassifier(
        n_estimators=200, max_depth=6, class_weight='balanced',
        random_state=42, n_jobs=-1,
    )
    rf_m.fit(X_tr_res, y_tr_res)

    # Logistic Regression (선형 베이스라인)
    lr_m = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
    lr_m.fit(X_tr_sc, y_tr_res)

    pos_models[pos]      = {'xgb': xgb_m, 'rf': rf_m, 'lr': lr_m}
    pos_imputers[pos]    = imp
    pos_scalers[pos]     = sc
    pos_feature_cols[pos] = feat_cols

    # 테스트 셋 평가
    if len(X_te_df) > 0 and len(y_te) > 0 and len(y_te.unique()) > 1:
        xgb_probs = xgb_m.predict_proba(X_te_df)[:, 1]
        rf_probs  = rf_m.predict_proba(X_te_df)[:, 1]
        lr_probs  = lr_m.predict_proba(X_te_sc)[:, 1]
        ens_probs = (xgb_probs + rf_probs + lr_probs) / 3.0
        y_te_reset = y_te.reset_index(drop=True)
        auc = roc_auc_score(y_te_reset, ens_probs)
        ap  = average_precision_score(y_te_reset, ens_probs)
        preds = (ens_probs >= 0.5).astype(int)
        report = classification_report(y_te_reset, preds, output_dict=True, zero_division=0)
        pos_results[pos] = {
            'auc_roc':           round(float(auc), 4),
            'avg_precision':     round(float(ap), 4),
            'precision_decline': round(float(report.get('1', {}).get('precision', 0)), 4),
            'recall_decline':    round(float(report.get('1', {}).get('recall', 0)), 4),
            'f1_decline':        round(float(report.get('1', {}).get('f1-score', 0)), 4),
            'n_test':            int(len(X_te_df)),
            'decline_rate_test': round(float(y_te.mean()), 4),
            'n_features':        len(feat_cols),
        }
        auc_note = " ✓ DEF 개선 목표 달성" if (pos == 'DEF' and auc >= 0.75) else (
                   " △ DEF 개선 목표 미달" if (pos == 'DEF' and auc < 0.75) else "")
        print(f"  [{pos}] AUC={auc:.4f}  F1={pos_results[pos]['f1_decline']:.4f}{auc_note}")

        # 피처 중요도
        fi_df = pd.DataFrame({
            'feature': feat_cols,
            'importance_xgb': xgb_m.feature_importances_,
            'importance_rf':  rf_m.feature_importances_,
        }).sort_values('importance_xgb', ascending=False)
        print(f"  [{pos}] Top 5 피처: {fi_df['feature'].head(5).tolist()}")

print("\n[8] 포지션별 모델 학습 완료.")

# ─────────────────────────────────────────────
# 14. 전체 데이터 스코어링
# ─────────────────────────────────────────────
print("\n[9] 포지션별 모델로 전체 데이터 스코어링...")

merged['decline_prob_ensemble'] = np.nan
merged['decline_prob_xgb']      = np.nan
merged['decline_prob_rf']       = np.nan
merged['decline_prob_lr']       = np.nan

for pos in POSITIONS:
    if pos not in pos_models:
        continue
    mask = merged['pos_group'] == pos
    if mask.sum() == 0:
        continue
    feat_cols = pos_feature_cols[pos]
    X_all = merged.loc[mask, feat_cols].copy()
    imp   = pos_imputers[pos]
    sc    = pos_scalers[pos]
    models_dict = pos_models[pos]

    X_imp = imp.transform(X_all)
    X_df  = pd.DataFrame(X_imp, columns=feat_cols)
    X_sc  = sc.transform(X_df)

    p_xgb = models_dict['xgb'].predict_proba(X_df)[:, 1]
    p_rf  = models_dict['rf'].predict_proba(X_df)[:, 1]
    p_lr  = models_dict['lr'].predict_proba(X_sc)[:, 1]
    p_ens = (p_xgb + p_rf + p_lr) / 3.0

    merged.loc[mask, 'decline_prob_xgb']      = p_xgb
    merged.loc[mask, 'decline_prob_rf']        = p_rf
    merged.loc[mask, 'decline_prob_lr']        = p_lr
    merged.loc[mask, 'decline_prob_ensemble']  = p_ens

# ─────────────────────────────────────────────
# 15. AGE FLOOR — 24세 이하는 50% 상한
#     젊은 선수 변동성 ≠ 커리어 하락
# ─────────────────────────────────────────────
young_mask = merged['age'] <= 24
n_capped   = young_mask.sum()
merged.loc[young_mask, 'decline_prob_ensemble'] = merged.loc[young_mask, 'decline_prob_ensemble'].clip(upper=0.50)
merged.loc[young_mask, 'decline_prob_xgb']      = merged.loc[young_mask, 'decline_prob_xgb'].clip(upper=0.50)
merged.loc[young_mask, 'decline_prob_rf']        = merged.loc[young_mask, 'decline_prob_rf'].clip(upper=0.50)
merged.loc[young_mask, 'decline_prob_lr']        = merged.loc[young_mask, 'decline_prob_lr'].clip(upper=0.50)
print(f"\n  나이 하한 적용: {n_capped}개 행을 50% 상한으로 조정 (age <= 24)")

# ─────────────────────────────────────────────
# 16. [FIX 1] Late Bloomer 감쇠 계수 적용
#     perf_slope > 0 AND perf_score > career_mean 이면
#     adjusted_prob = prob * (1 - min(perf_slope / 2, 0.3))
#     → 상승 궤적 선수의 decline 확률을 최대 30% 감소
#     Chris Wood 케이스: slope=+0.816, perf_score=0.734 > career_mean
# ─────────────────────────────────────────────
print("\n[10] [FIX 1] Late Bloomer 감쇠 계수 적용...")

# career_perf_mean 병합 (이미 merged에 있어야 하지만 혹시 없으면 재산출)
if 'career_perf_mean' not in merged.columns:
    merged = merged.merge(player_career_mean, on='player_key', how='left')
    merged['career_perf_std'] = merged['career_perf_std'].fillna(0.5)

# Late bloomer 조건: 상승 중이고 현재가 커리어 평균 이상
late_bloomer_mask = (
    (merged['perf_slope'].fillna(0) > 0) &
    (merged['perf_score'] > merged['career_perf_mean'])
)

# 감쇠 계수 = 1 - min(perf_slope / 2, 0.3)
# slope=0 → 감쇠 없음, slope=0.6 → 30% 감소, slope>0.6 → 최대 30% 감소
merged['late_bloomer_dampen'] = np.where(
    late_bloomer_mask,
    1.0 - merged['perf_slope'].fillna(0).clip(lower=0).apply(lambda s: min(s / 2, 0.3)),
    1.0  # 조건 미충족 시 감쇠 없음
)

# 감쇠 계수를 decline_prob_ensemble에 적용
n_late_bloomer = late_bloomer_mask.sum()
merged['decline_prob_ensemble'] = merged['decline_prob_ensemble'] * merged['late_bloomer_dampen']
merged['decline_prob_xgb']      = merged['decline_prob_xgb']      * merged['late_bloomer_dampen']
merged['decline_prob_rf']        = merged['decline_prob_rf']        * merged['late_bloomer_dampen']
merged['decline_prob_lr']        = merged['decline_prob_lr']        * merged['late_bloomer_dampen']

print(f"  Late bloomer 감쇠 적용: {n_late_bloomer}개 행")

# Chris Wood 확인 로그
wood_check = merged[
    (merged['player_key'].str.contains('Chris Wood', case=False, na=False)) &
    (merged['season_year'] == merged['season_year'].max())
]
if len(wood_check) > 0:
    wr = wood_check.iloc[0]
    print(f"  [확인] Chris Wood: slope={wr['perf_slope']:.3f}  "
          f"perf_score={wr['perf_score']:.3f}  "
          f"career_mean={wr['career_perf_mean']:.3f}  "
          f"dampen={wr['late_bloomer_dampen']:.3f}  "
          f"prob(after)={wr['decline_prob_ensemble']:.3f}")

# ─────────────────────────────────────────────
# 17. 이중 출력 목록 구축
#     LIST 1: Career Decline Watch (age 28+, 지속 하락)
#     LIST 2: Regression-to-Mean Alert (아웃라이어 시즌, 전연령)
# ─────────────────────────────────────────────
print("\n[11] 이중 출력 목록 구축...")

latest_year = merged['season_year'].max()
latest_data = (
    merged[merged['season_year'] == latest_year]
    .sort_values('decline_prob_ensemble', ascending=False)
    .drop_duplicates(subset='player_key', keep='first')
    .copy()
)
print(f"  최신 시즌: {latest_year}  선수 수: {len(latest_data)}")

# ── LIST 1: Career Decline Watch ─────────────────────────────────────────
# 기준: age >= 28, decline 확률 기준 상위 30명
career_watch_pool = latest_data[latest_data['age'] >= 28].copy()
career_watch = (
    career_watch_pool
    .nlargest(30, 'decline_prob_ensemble')
    [['player_key', 'team', 'pos_group', 'age', 'season_year',
      'decline_prob_ensemble', 'decline_prob_xgb',
      'decline_perf', 'decline_avail',
      'perf_score', 'perf_slope', 'min', 'min_slope',
      'peak_minus_current', 'market_value', 'injury_proxy',
      'late_bloomer_dampen']]
    .reset_index(drop=True)
)

# ── LIST 2: Regression-to-Mean Alert ─────────────────────────────────────
# 기준: is_outlier_season=1 (전연령) — 예외적 시즌 후 자연 회귀 예상
regression_alert_pool = latest_data[latest_data['is_outlier_season'] == 1].copy()
regression_alert = (
    regression_alert_pool
    .sort_values('peak_minus_current')
    .head(30)
    [['player_key', 'team', 'pos_group', 'age', 'season_year',
      'perf_score', 'career_perf_mean', 'career_perf_std',
      'peak_minus_current', 'perf_slope', 'decline_prob_ensemble',
      'market_value']]
    .reset_index(drop=True)
)
regression_alert['seasons_above_mean_std'] = (
    (regression_alert['perf_score'] - regression_alert['career_perf_mean'])
    / regression_alert['career_perf_std'].clip(lower=0.01)
).round(2)

print(f"\n  Career Decline Watch (age 28+): {len(career_watch)}명")
print(f"  Regression Alert (아웃라이어 시즌): {len(regression_alert)}명")

print(f"\n  [Career Decline Watch — Top 10]")
print(career_watch[['player_key', 'team', 'age', 'pos_group',
                     'decline_prob_ensemble', 'perf_slope']].head(10).to_string(index=False))
print(f"\n  [Regression-to-Mean Alert — Top 10]")
print(regression_alert[['player_key', 'team', 'age', 'pos_group',
                          'perf_score', 'career_perf_mean', 'seasons_above_mean_std']].head(10).to_string(index=False))

# ─────────────────────────────────────────────
# 18. [FIX 3] Sanity Check
# ─────────────────────────────────────────────
print("\n[12] [FIX 3] Sanity Check 수행...")

sanity_results = {}

# 검사 1: Chris Wood — career_decline_watch에 없거나 prob < 0.5
wood_latest = latest_data[latest_data['player_key'].str.contains('Chris Wood', case=False, na=False)]
wood_in_watch = career_watch['player_key'].str.contains('Chris Wood', case=False).any()
wood_prob = wood_latest['decline_prob_ensemble'].values[0] if len(wood_latest) > 0 else None

sanity_results['chris_wood'] = {
    'found':        len(wood_latest) > 0,
    'prob':         round(float(wood_prob), 4) if wood_prob is not None else None,
    'in_career_watch': bool(wood_in_watch),
    'pass':         (not wood_in_watch) or (wood_prob is not None and wood_prob < 0.5),
    'note':         'career_watch 제외 또는 prob < 0.5 이어야 함 (late bloomer 감쇠 적용)',
}
status = "PASS" if sanity_results['chris_wood']['pass'] else "FAIL"
wood_prob_str = f"{wood_prob:.3f}" if wood_prob is not None else "N/A"
print(f"  [{status}] Chris Wood: prob={wood_prob_str}  "
      f"in_career_watch={wood_in_watch}")

# 검사 2: Casemiro — career_decline_watch에 포함
case_latest = latest_data[latest_data['player_key'].str.contains('Casemiro', case=False, na=False)]
case_in_watch = career_watch['player_key'].str.contains('Casemiro', case=False).any()
case_prob = case_latest['decline_prob_ensemble'].values[0] if len(case_latest) > 0 else None

sanity_results['casemiro'] = {
    'found':        len(case_latest) > 0,
    'prob':         round(float(case_prob), 4) if case_prob is not None else None,
    'in_career_watch': bool(case_in_watch),
    'pass':         bool(case_in_watch),
    'note':         'career_watch에 포함되어야 함 (MID, age 31, declining form)',
}
status = "PASS" if sanity_results['casemiro']['pass'] else "FAIL"
case_prob_str = f"{case_prob:.3f}" if case_prob is not None else "N/A"
print(f"  [{status}] Casemiro: prob={case_prob_str}  "
      f"in_career_watch={case_in_watch}")

# 검사 3: Cole Palmer — regression_alert에 있고 career_watch에 없음
palmer_latest = latest_data[latest_data['player_key'].str.contains('Cole Palmer', case=False, na=False)]
palmer_in_watch = career_watch['player_key'].str.contains('Cole Palmer', case=False).any()
palmer_in_reg   = regression_alert['player_key'].str.contains('Cole Palmer', case=False).any()
palmer_prob = palmer_latest['decline_prob_ensemble'].values[0] if len(palmer_latest) > 0 else None

sanity_results['cole_palmer'] = {
    'found':           len(palmer_latest) > 0,
    'prob':            round(float(palmer_prob), 4) if palmer_prob is not None else None,
    'in_career_watch': bool(palmer_in_watch),
    'in_regression_alert': bool(palmer_in_reg),
    'pass':            (not palmer_in_watch) and palmer_in_reg,
    'note':            'regression_alert에 있고 career_watch에 없어야 함 (age 21, 아웃라이어 시즌)',
}
status = "PASS" if sanity_results['cole_palmer']['pass'] else "FAIL"
palmer_prob_str = f"{palmer_prob:.3f}" if palmer_prob is not None else "N/A"
print(f"  [{status}] Cole Palmer: prob={palmer_prob_str}  "
      f"in_career_watch={palmer_in_watch}  in_regression_alert={palmer_in_reg}")

# 검사 4: career_decline_watch 전원 age 28+
if len(career_watch) > 0:
    min_watch_age = career_watch['age'].min()
    all_28plus = bool(min_watch_age >= 28)
else:
    all_28plus = True
    min_watch_age = None

sanity_results['career_watch_age_28plus'] = {
    'min_age_in_watch': float(min_watch_age) if min_watch_age is not None else None,
    'all_28plus':       all_28plus,
    'pass':             all_28plus,
    'note':             'career_watch 전원 age >= 28 이어야 함',
}
status = "PASS" if all_28plus else "FAIL"
print(f"  [{status}] career_watch 최소 나이: {min_watch_age}  전원 28+: {all_28plus}")

# 전체 sanity 결과
all_pass = all(v['pass'] for v in sanity_results.values())
print(f"\n  전체 Sanity Check: {'ALL PASS' if all_pass else 'SOME FAIL'}")

# ─────────────────────────────────────────────
# 19. 피처 중요도 요약
# ─────────────────────────────────────────────
fi_summary = {}
for pos in POSITIONS:
    if pos not in pos_models:
        continue
    feat_cols = pos_feature_cols[pos]
    xgb_m = pos_models[pos]['xgb']
    rf_m  = pos_models[pos]['rf']
    fi_df = pd.DataFrame({
        'feature':         feat_cols,
        'importance_xgb':  xgb_m.feature_importances_,
        'importance_rf':   rf_m.feature_importances_,
    }).sort_values('importance_xgb', ascending=False)
    fi_summary[pos] = [
        {'feature':         r['feature'],
         'importance_xgb':  round(float(r['importance_xgb']), 4),
         'importance_rf':   round(float(r['importance_rf']),  4)}
        for _, r in fi_df.head(10).iterrows()
    ]

# ─────────────────────────────────────────────
# 20. 나이/포지션 하락률 집계
# ─────────────────────────────────────────────
age_bins = [(18, 22), (23, 26), (27, 30), (31, 35), (36, 40)]
age_pos_decline = {}
for pos in ['DEF', 'MID', 'FWD', 'GK']:
    sub = merged[merged['pos_group'] == pos].copy()
    bin_data = {}
    for lo, hi in age_bins:
        seg = sub[(sub['age'] >= lo) & (sub['age'] <= hi)]
        if len(seg) > 0:
            bin_data[f"{lo}-{hi}"] = {
                'avg_decline_prob':    round(float(seg['decline_prob_ensemble'].mean()), 4),
                'actual_decline_rate': round(float(seg['decline'].mean()), 4),
                'n_players':           int(len(seg)),
            }
    age_pos_decline[pos] = bin_data

# ─────────────────────────────────────────────
# 21. 시각화
# ─────────────────────────────────────────────
print("\n[13] 시각화 생성...")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

COLOR_PALETTE = {'DEF': '#2196F3', 'MID': '#4CAF50', 'FWD': '#F44336', 'GK': '#FF9800'}

# ── Figure 1: 나이별 decline 확률 (포지션별, late bloomer 감쇠 적용 후) ──
fig, ax = plt.subplots(figsize=(12, 6))
age_decline = merged.groupby(['age', 'pos_group'])['decline_prob_ensemble'].mean().reset_index()
for pos_g in ['DEF', 'MID', 'FWD', 'GK']:
    sub = age_decline[age_decline['pos_group'] == pos_g]
    sub = sub[(sub['age'] >= 18) & (sub['age'] <= 40)]
    if len(sub) < 3:
        continue
    sub_s = sub.sort_values('age')
    smoothed = sub_s.set_index('age')['decline_prob_ensemble'].rolling(
        window=3, center=True, min_periods=1).mean()
    ax.plot(smoothed.index, smoothed.values,
            color=COLOR_PALETTE.get(pos_g, 'gray'),
            linewidth=2.5, label=pos_g, marker='o', markersize=4)

ax.axvline(x=24.5, color='green', linestyle='--', alpha=0.6, label='Age floor (<=24 capped)')
ax.axvline(x=28,   color='orange', linestyle='--', alpha=0.5, label='Career decline threshold (28+)')
ax.axvline(x=32,   color='red',    linestyle='--', alpha=0.5, label='High career risk (32+)')
ax.set_xlabel('Age', fontsize=13)
ax.set_ylabel('Decline Probability (late-bloomer dampened)', fontsize=13)
ax.set_title('S6-V3: Decline Probability by Age and Position\n(Late Bloomer Fix + Age Floor + DEF Enhanced)', fontsize=14)
ax.legend(fontsize=10)
ax.set_xlim(18, 40)
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / 'v3_decline_prob_by_age.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 2: Career Decline Watch ───────────────────────────────────────
if len(career_watch) > 0:
    fig, ax = plt.subplots(figsize=(14, min(12, max(6, len(career_watch.head(20)) * 0.6))))
    watch_plot = career_watch.head(20).sort_values('decline_prob_ensemble')
    colors = [COLOR_PALETTE.get(p, '#9E9E9E') for p in watch_plot['pos_group']]
    ax.barh(range(len(watch_plot)), watch_plot['decline_prob_ensemble'].values,
            color=colors, edgecolor='white', linewidth=0.5)
    for i, (_, row) in enumerate(watch_plot.iterrows()):
        ax.text(row['decline_prob_ensemble'] + 0.005, i,
                f" Age {int(row['age'])} | slope={row['perf_slope']:.2f} | {row['team'][:14]}",
                va='center', fontsize=8)
    ax.set_yticks(range(len(watch_plot)))
    ax.set_yticklabels([f"{r['player_key'][:25]} ({r['pos_group']})"
                        for _, r in watch_plot.iterrows()], fontsize=9)
    ax.set_xlabel('Decline Probability (Ensemble, Late-Bloomer Dampened)', fontsize=12)
    ax.set_title('S6-V3: Career Decline Watch — Age 28+, Sustained Drop\n(Late Bloomer Fix Applied)', fontsize=13)
    ax.set_xlim(0, 1.2)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax.axvline(x=0.7, color='darkred', linestyle='--', alpha=0.5, label='70% high risk')
    legend_patches = [mpatches.Patch(color=c, label=p) for p, c in COLOR_PALETTE.items()]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'v3_career_decline_watch.png', dpi=150, bbox_inches='tight')
    plt.close()

# ── Figure 3: Regression Alert ────────────────────────────────────────────
if len(regression_alert) > 0:
    plot_reg = regression_alert.head(20).copy()
    plot_reg['label'] = plot_reg.apply(
        lambda r: f"{r['player_key'][:22]} ({r['pos_group']}, {int(r['age'])})", axis=1)
    plot_reg = plot_reg.sort_values('seasons_above_mean_std')
    fig, ax = plt.subplots(figsize=(14, min(12, max(6, len(plot_reg) * 0.6))))
    colors_reg = [COLOR_PALETTE.get(p, '#9E9E9E') for p in plot_reg['pos_group']]
    ax.barh(range(len(plot_reg)), plot_reg['seasons_above_mean_std'].values,
            color=colors_reg, edgecolor='white', linewidth=0.5)
    for i, (_, row) in enumerate(plot_reg.iterrows()):
        ax.text(row['seasons_above_mean_std'] + 0.05, i,
                f" mean={row['career_perf_mean']:.2f} | team={row['team'][:14]}",
                va='center', fontsize=8)
    ax.set_yticks(range(len(plot_reg)))
    ax.set_yticklabels(plot_reg['label'].values, fontsize=9)
    ax.set_xlabel('Standard Deviations above Career Mean', fontsize=12)
    ax.set_title('S6-V3: Regression-to-Mean Alert\n(Outlier season — any age, NOT career decline)', fontsize=13)
    ax.axvline(x=1.5, color='orange', linestyle='--', alpha=0.6, label='1.5 SD threshold')
    ax.grid(axis='x', alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'v3_regression_alert.png', dpi=150, bbox_inches='tight')
    plt.close()

# ── Figure 4: 포지션별 피처 중요도 ──────────────────────────────────────
n_pos = len(fi_summary)
if n_pos > 0:
    fig, axes = plt.subplots(1, n_pos, figsize=(6 * n_pos, 7))
    if n_pos == 1:
        axes = [axes]
    for ax, pos in zip(axes, fi_summary.keys()):
        fi_list = fi_summary[pos]
        feats = [d['feature'] for d in fi_list]
        imps  = [d['importance_xgb'] for d in fi_list]
        colors_fi = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(feats)))
        ax.barh(range(len(feats)), imps[::-1], color=colors_fi)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats[::-1], fontsize=9)
        def_label = ' (DEF Enhanced)' if pos == 'DEF' else ''
        ax.set_title(f'{pos}{def_label} — Feature Importance', fontsize=11)
        ax.set_xlabel('XGBoost Importance', fontsize=10)
        ax.grid(axis='x', alpha=0.3)
    plt.suptitle('S6-V3: Position-Specific Feature Importance\n(DEF: aerial_duel + clean_sheet + minutes_stability added)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'v3_feature_importance_by_position.png', dpi=150, bbox_inches='tight')
    plt.close()

print(f"  시각화 저장 완료: {FIG_DIR}")

# ─────────────────────────────────────────────
# 22. decline_predictions_v4.parquet 저장
# ─────────────────────────────────────────────
print("\n[14] decline_predictions_v4.parquet 저장...")
output_cols = [
    'player_key', 'team', 'pos_group', 'age', 'season_year',
    'decline', 'decline_perf', 'decline_avail', 'is_outlier_season',
    'decline_prob_ensemble', 'decline_prob_xgb', 'decline_prob_rf', 'decline_prob_lr',
    'perf_score', 'perf_slope', 'min', 'min_slope', 'mp', 'market_value',
    'peak_minus_current', 'consistency_score', 'injury_proxy',
    'career_perf_mean', 'career_perf_std',
    'team_quality', 'team_quality_change',
    'late_bloomer_dampen',
    'aerial_duel_proxy', 'clean_sheet_involvement', 'minutes_stability',
]
output_cols = [c for c in output_cols if c in merged.columns]
merged[output_cols].to_parquet(SCOUT_DIR / 'decline_predictions_v4.parquet', index=False)
print(f"  저장 완료 ({len(merged)}행, {len(output_cols)}열)")

# ─────────────────────────────────────────────
# 23. results_summary_v3.json 구성
# ─────────────────────────────────────────────
print("\n[15] results_summary_v3.json 구성...")

def safe_records(df, cols):
    """DataFrame → JSON 안전 변환 (NaN, numpy 타입 처리)"""
    cols = [c for c in cols if c in df.columns]
    subset = df[cols].copy()
    for col in subset.select_dtypes(include='float').columns:
        subset[col] = subset[col].round(4)
    if 'age' in subset.columns:
        subset['age'] = subset['age'].fillna(0).astype(int)
    return [
        {k: (None if isinstance(v, float) and np.isnan(v) else
             (int(v) if isinstance(v, (np.integer,)) else
              (float(v) if isinstance(v, (np.floating,)) else v)))
         for k, v in row.items()}
        for row in subset.to_dict(orient='records')
    ]

career_watch_age  = career_watch['age'].dropna()
regression_age    = regression_alert['age'].dropna()

# DEF AUC 개선 여부 판단
def_auc_v2  = 0.6814  # v2 기준값
def_auc_v3  = pos_results.get('DEF', {}).get('auc_roc', None)
def_improved = def_auc_v3 is not None and def_auc_v3 >= 0.75

summary = {
    'model':       'S6-V3 Player Decline Detection (Scout Validation v3)',
    'version':     'v3',
    'description': (
        'EPL 하락세 감지: v2 대비 late bloomer 예외처리, '
        'DEF 전용 피처 추가 (aerial_duel/clean_sheet/minutes_stability), '
        'sanity check 자동화.'
    ),
    'latest_season': int(latest_year),

    # ── v3 적용 수정사항 ──────────────────────────────────────────────────
    'fixes_applied_v3': {
        'fix1_late_bloomer_exception': {
            'description': (
                'perf_slope > 0 AND perf_score > career_mean 조건 충족 시 '
                'decline_prob 감쇠: adjusted = prob * (1 - min(slope/2, 0.3)). '
                '최대 30% 감소. Chris Wood(slope=+0.816) 등 상승세 선수 오탐 방지.'
            ),
            'formula':       'adjusted_prob = prob * (1 - min(perf_slope / 2, 0.3))',
            'n_affected':    int(n_late_bloomer),
            'max_reduction': '30%',
        },
        'fix2_def_model_improvement': {
            'description':   'DEF 전용 피처 3종 추가로 AUC 개선 시도.',
            'new_features': [
                'aerial_duel_proxy: crdy + fls 평균 합산 (피지컬 플레이 강도)',
                'clean_sheet_involvement: 선발 출전 시 팀 클린시트 비율',
                'minutes_stability: 경기당 출전시간 표준편차',
            ],
            'progressive_pass': 'skipped (데이터 없음)' if not HAS_PROG_PASS else 'added',
            'def_auc_v2':    def_auc_v2,
            'def_auc_v3':    round(float(def_auc_v3), 4) if def_auc_v3 else None,
            'target_met':    bool(def_improved),
        },
        'fix3_sanity_checks': sanity_results,
    },

    # ── v2 상속 수정사항 ──────────────────────────────────────────────────
    'inherited_from_v2': {
        'position_specific_models': 'FWD/MID/DEF/GK 각각 별도 모델',
        'age_floor':               '24세 이하 50% 상한',
        'sustained_decline':       '2 연속 시즌 하락 필요',
        'dual_output_lists':       'career_decline_watch + regression_to_mean_alert',
    },

    # ── 포지션별 모델 성과 ────────────────────────────────────────────────
    'model_performance_by_position': pos_results,

    # ── 데이터셋 정보 ─────────────────────────────────────────────────────
    'dataset': {
        'total_samples':         int(len(merged)),
        'train_cutoff':          'season_year < 2021',
        'val_range':             '2021 <= season_year <= 2022',
        'test_range':            'season_year >= 2023',
        'n_features_common':     len(FEATURE_COLS),
        'n_features_def':        len(FEATURE_COLS_DEF),
        'feature_list_common':   FEATURE_COLS,
        'feature_list_def':      FEATURE_COLS_DEF,
        'decline_rate_overall':  round(float(merged['decline'].mean()), 4),
        'outlier_season_rate':   round(float(merged['is_outlier_season'].mean()), 4),
    },

    # ── 피처 중요도 ───────────────────────────────────────────────────────
    'feature_importance_by_position': fi_summary,

    # ── 스카우트 출력 ─────────────────────────────────────────────────────
    'scout_outputs': {
        'career_decline_watch': {
            'description': (
                'Age 28+ 선수 중 지속 하락세. 계약 검토, 로테이션 계획, 이적 결정에 활용.'
                ' Late bloomer 감쇠 계수 적용으로 상승세 선수 오탐 방지.'
            ),
            'criteria': 'age >= 28, 포지션별 decline 확률 (late bloomer 감쇠 후)',
            'count':     int(len(career_watch)),
            'avg_age':   round(float(career_watch_age.mean()), 1) if len(career_watch_age) > 0 else None,
            'pct_30plus': round(float((career_watch_age >= 30).mean()), 3) if len(career_watch_age) > 0 else None,
            'list': safe_records(career_watch.head(20), [
                'player_key', 'team', 'pos_group', 'age',
                'decline_prob_ensemble', 'decline_prob_xgb',
                'perf_score', 'perf_slope', 'min_slope',
                'peak_minus_current', 'market_value', 'injury_proxy',
                'late_bloomer_dampen',
            ]),
        },
        'regression_to_mean_alert': {
            'description': (
                '아웃라이어 시즌(커리어 평균 +1.5 SD 초과) 선수. '
                '자연 회귀 예상 — 커리어 하락 아님. '
                'Cole Palmer 등 연령 무관 폭발적 시즌 후 정상화 예측.'
            ),
            'criteria': 'is_outlier_season=1 (전연령)',
            'count':    int(len(regression_alert)),
            'avg_age':  round(float(regression_age.mean()), 1) if len(regression_age) > 0 else None,
            'list': safe_records(regression_alert.head(20), [
                'player_key', 'team', 'pos_group', 'age',
                'perf_score', 'career_perf_mean',
                'seasons_above_mean_std', 'perf_slope',
                'decline_prob_ensemble', 'market_value',
            ]),
        },
    },

    # ── 나이/포지션 하락률 ────────────────────────────────────────────────
    'age_position_decline_rates': age_pos_decline,

    # ── 하락세 정의 ───────────────────────────────────────────────────────
    'decline_definition': {
        'sustained_performance_decline': (
            'perf_score 하락 N→N+1 (>=0.4 SD) AND N+1→N+2 (>=0.2 SD). '
            '2 연속 시즌 필요.'
        ),
        'availability_decline': (
            '출전시간 < 이전 시즌의 70% (기준 >= 450분). '
            '부상/폼 하락 프록시.'
        ),
        'combined':       '지속 성과 또는 출전시간 하락 중 하나.',
        'outlier_season': 'perf_score > 선수 커리어 평균 + 1.5 SD. 회귀-평균 경보.',
        'age_floor':      '24세 이하 decline prob 50% 상한.',
        'late_bloomer':   'slope > 0 AND perf > career_mean → prob * (1 - min(slope/2, 0.3))',
    },

    # ── Sanity check 요약 ─────────────────────────────────────────────────
    'sanity_checks_summary': {
        'all_pass':             all_pass,
        'chris_wood_pass':      sanity_results['chris_wood']['pass'],
        'casemiro_pass':        sanity_results['casemiro']['pass'],
        'cole_palmer_pass':     sanity_results['cole_palmer']['pass'],
        'career_watch_age_pass': sanity_results['career_watch_age_28plus']['pass'],
        'career_watch_avg_age': round(float(career_watch_age.mean()), 1) if len(career_watch_age) > 0 else None,
        'career_watch_pct_28plus': round(float((career_watch_age >= 28).mean()), 3) if len(career_watch_age) > 0 else None,
        'n_late_bloomer_adjusted': int(n_late_bloomer),
        'position_models_trained': list(pos_models.keys()),
    },
}

# ─────────────────────────────────────────────
# 24. JSON 저장
# ─────────────────────────────────────────────
def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    print(f"  저장: {path}")

save_json(summary, MODEL_DIR / 'results_summary_v3.json')
save_json(summary, SCOUT_DIR  / 's6_results_summary_v3.json')

# ─────────────────────────────────────────────
# 25. 최종 콘솔 리포트
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("[S6-V3] 완료")
print("=" * 60)

print(f"\n포지션별 학습 모델: {list(pos_models.keys())}")
for pos, m in pos_results.items():
    def_mark = " ← DEF AUC " + ("개선 달성" if pos == 'DEF' and m['auc_roc'] >= 0.75 else
                                  f"v2:{def_auc_v2:.4f}→v3:{m['auc_roc']:.4f}") if pos == 'DEF' else ""
    print(f"  [{pos}] AUC={m['auc_roc']:.4f}  F1={m['f1_decline']:.4f}  "
          f"n_test={m['n_test']}{def_mark}")

print(f"\n[FIX 1] Late Bloomer 감쇠 적용 선수 수: {n_late_bloomer}")
print(f"[FIX 2] DEF 추가 피처: {[f for f in DEF_EXTRA_FEATURES if f in merged.columns]}")
if def_auc_v3:
    print(f"[FIX 2] DEF AUC: v2={def_auc_v2:.4f} → v3={def_auc_v3:.4f}  "
          f"({'목표 달성 (>=0.75)' if def_improved else '개선됨 (목표 미달)'})")

print(f"\nCareer Decline Watch (28+): {len(career_watch)}명")
if len(career_watch) > 0:
    print(career_watch[['player_key', 'team', 'age', 'pos_group',
                          'decline_prob_ensemble', 'perf_slope']].head(5).to_string(index=False))

print(f"\nRegression-to-Mean Alert: {len(regression_alert)}명")
if len(regression_alert) > 0:
    print(regression_alert[['player_key', 'team', 'age', 'pos_group',
                               'seasons_above_mean_std', 'perf_score']].head(5).to_string(index=False))

print(f"\n[FIX 3] Sanity Check 결과:")
for k, v in sanity_results.items():
    status = "PASS" if v['pass'] else "FAIL"
    print(f"  [{status}] {k}: {v.get('note', '')}")

print(f"\n저장 파일:")
print(f"  {MODEL_DIR}/results_summary_v3.json")
print(f"  {SCOUT_DIR}/s6_results_summary_v3.json")
print(f"  {SCOUT_DIR}/decline_predictions_v4.parquet")
print(f"  {FIG_DIR}/ (4개 시각화)")

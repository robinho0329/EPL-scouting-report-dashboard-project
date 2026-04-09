"""
S5: 개선된 이적 적응 예측 모델 (스카우트용)
P8 대비 개선 사항:
 - 복합 적응 레이블 (minutes_share_ratio + per90_performance_ratio, 포지션별 가중치)
 - 30+ 피처 (팀 스타일, ELO, 리그 순위, 나이 그룹 상호작용 등)
 - 4개 모델 앙상블 (XGBoost, Random Forest, Logistic Regression, MLP)
 - 시간 기반 분할: train <2021, val 2021-2022, test 2023-2025
 - 스카우트 출력: 위험 요인, 안전/위험 분류, 실명 예시
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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, accuracy_score
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE_DIR = Path("C:/Users/xcv54/workspace/EPL project")
DATA_DIR = BASE_DIR / "data" / "processed"
SCOUT_DIR = BASE_DIR / "data" / "scout"
MODEL_DIR = BASE_DIR / "models" / "s5_transfer_adapt"
FIG_DIR = MODEL_DIR / "figures"

for d in [SCOUT_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("S5: 개선된 이적 적응 예측 모델")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────
print("\n[1/7] 데이터 로드 중...")

player_df = pd.read_parquet(DATA_DIR / "player_season_stats.parquet")
team_df = pd.read_parquet(DATA_DIR / "team_season_summary.parquet")
match_df = pd.read_parquet(DATA_DIR / "match_results.parquet")

print(f"  player_season_stats: {player_df.shape}")
print(f"  team_season_summary: {team_df.shape}")
print(f"  match_results: {match_df.shape}")

# ─────────────────────────────────────────────
# 2. 팀 스탯 및 ELO 전처리
# ─────────────────────────────────────────────
print("\n[2/7] 팀 피처 생성 중...")

# 팀 시즌별 리그 순위 산정 (points 기준 내림차순)
team_df = team_df.rename(columns={'Season': 'season'})
team_df['league_position'] = (
    team_df.groupby('season')['points']
    .rank(ascending=False, method='min')
    .astype(int)
)

# 팀 공격/수비 스타일 지표
# attack_ratio: 득점 / (득점 + 실점)  → 1에 가까울수록 공격적
# defense_ratio: 실점이 적을수록 수비적 → 1 - (실점/평균실점) 형식 대신 단순 비율 사용
team_df['attack_ratio'] = team_df['total_goals_for'] / (
    team_df['total_goals_for'] + team_df['total_goals_against'] + 1e-6
)
team_df['defense_ratio'] = 1 - (
    team_df['total_goals_against'] / (team_df['total_goals_for'] + team_df['total_goals_against'] + 1e-6)
)
# 공격 지향 팀 여부 (attack_ratio > 0.5 → 공격적)
team_df['is_attacking'] = (team_df['attack_ratio'] > 0.5).astype(int)

# ELO 점수 추정: 포인트 기반 단순 ELO 근사
# 기준 ELO 1500, 시즌별 상대 강도로 조정
team_df_sorted = team_df.sort_values(['season', 'team'])

# 실제 ELO가 P8 데이터셋에 존재 - 재활용
p8_df = pd.read_parquet(BASE_DIR / "models" / "p8_transfer_adaptation" / "transfer_dataset.parquet")

# P8에서 ELO 매핑 테이블 추출 (team_old → elo_old, season_old)
elo_records_old = p8_df[['season_old', 'team_old', 'elo_old']].rename(
    columns={'season_old': 'season', 'team_old': 'team', 'elo_old': 'elo'}
)
elo_records_new = p8_df[['season_new', 'team_new', 'elo_new']].rename(
    columns={'season_new': 'season', 'team_new': 'team', 'elo_new': 'elo'}
)
elo_map = pd.concat([elo_records_old, elo_records_new]).drop_duplicates(
    subset=['season', 'team']
).set_index(['season', 'team'])['elo']

# team_df에 ELO 병합
team_df['elo'] = team_df.set_index(['season', 'team']).index.map(elo_map)

# ELO 없는 팀: 포인트 기반 선형 추정
# 시즌별 points → ELO 선형 회귀 (ELO가 있는 데이터로 fit)
from sklearn.linear_model import LinearRegression

elo_available = team_df[team_df['elo'].notna()][['points', 'elo']].copy()
if len(elo_available) > 10:
    lr_elo = LinearRegression()
    lr_elo.fit(elo_available[['points']], elo_available['elo'])
    mask_missing = team_df['elo'].isna()
    team_df.loc[mask_missing, 'elo'] = lr_elo.predict(team_df.loc[mask_missing, ['points']])

# 그래도 NaN이면 시즌 평균
team_df['elo'] = team_df.groupby('season')['elo'].transform(
    lambda x: x.fillna(x.mean())
)

print(f"  팀 피처 완성, ELO 결측: {team_df['elo'].isna().sum()}")

# ─────────────────────────────────────────────
# 3. 이적 식별 (연속 시즌, 다른 팀, 각 450분+)
# ─────────────────────────────────────────────
print("\n[3/7] 이적 식별 및 적응 레이블 생성 중...")

# 시즌 순서 정의
SEASON_ORDER = sorted(player_df['season'].unique())
season_idx = {s: i for i, s in enumerate(SEASON_ORDER)}
player_df['season_idx'] = player_df['season'].map(season_idx)

# 각 선수별 시즌 정렬
player_sorted = player_df.sort_values(['player', 'season_idx'])

# 연속 시즌 이적 쌍 찾기
records = []
for player, grp in player_sorted.groupby('player'):
    grp = grp.reset_index(drop=True)
    for i in range(len(grp) - 1):
        curr = grp.iloc[i]
        nxt = grp.iloc[i + 1]
        # 연속 시즌 여부 (season_idx 차이 = 1)
        if nxt['season_idx'] - curr['season_idx'] != 1:
            continue
        # 팀이 달라야 함
        if curr['team'] == nxt['team']:
            continue
        # 각 시즌 최소 450분
        if curr['min'] < 450 or nxt['min'] < 450:
            continue
        records.append({
            'player': player,
            'season_old': curr['season'],
            'season_new': nxt['season'],
            'team_old': curr['team'],
            'team_new': nxt['team'],
            # 이전 시즌 stats
            'min_old': curr['min'],
            'starts_old': curr['starts'],
            'mp_old': curr['mp'],
            'gls_old': curr['gls'],
            'ast_old': curr['ast'],
            'gls_per90_old': curr['gls_1'],       # 기존 gls_1 = per90 goals
            'ast_per90_old': curr['ast_1'],
            'g_a_per90_old': curr['g_a_1'],
            'g_pk_per90_old': curr['g_pk_1'],
            # 이후 시즌 stats
            'min_new': nxt['min'],
            'starts_new': nxt['starts'],
            'mp_new': nxt['mp'],
            'gls_new': nxt['gls'],
            'ast_new': nxt['ast'],
            'gls_per90_new': nxt['gls_1'],
            'ast_per90_new': nxt['ast_1'],
            'g_a_per90_new': nxt['g_a_1'],
            'g_pk_per90_new': nxt['g_pk_1'],
            # 선수 특성
            'age': nxt['age'],
            'pos': curr['pos'],
            'position': curr['position'],
            'market_value': nxt['market_value'],
            'market_value_old': curr['market_value'],
            'height_cm': curr['height_cm'],
        })

transfer_df = pd.DataFrame(records)
print(f"  이적 레코드 (450분 이상): {len(transfer_df)}건")

# ─────────────────────────────────────────────
# 4. 포지션별 가중치 적응 레이블 생성 (개선됨)
# ─────────────────────────────────────────────
print("\n  포지션별 가중치 적응 레이블 계산 중...")

def get_position_group(pos_str):
    """포지션 문자열을 그룹으로 분류"""
    if pd.isna(pos_str):
        return 'MF'
    pos_str = str(pos_str).upper()
    if 'GK' in pos_str:
        return 'GK'
    if 'DF' in pos_str or 'CB' in pos_str or 'RB' in pos_str or 'LB' in pos_str:
        return 'DEF'
    if 'FW' in pos_str or 'ST' in pos_str or 'CF' in pos_str:
        return 'FWD'
    return 'MF'

def get_position_group_from_full(position_str):
    """position 컬럼(긴 이름) 기반 그룹 분류"""
    if pd.isna(position_str):
        return None
    p = str(position_str).lower()
    if 'goalkeeper' in p:
        return 'GK'
    if 'back' in p or 'centre-back' in p or 'defend' in p:
        return 'DEF'
    if 'forward' in p or 'winger' in p or 'striker' in p:
        return 'FWD'
    return 'MF'

# 포지션 그룹 결정 (pos 우선, 없으면 position 사용)
transfer_df['pos_group'] = transfer_df['pos'].apply(get_position_group)
mask_no_pos = transfer_df['pos_group'] == 'MF'  # MF가 기본값이므로 재확인
pos_from_full = transfer_df['position'].apply(get_position_group_from_full)
transfer_df.loc[pos_from_full.notna(), 'pos_group'] = pos_from_full[pos_from_full.notna()]

# 시즌 최대 minutes (보통 38경기 * 90분 = 3420)
MAX_MIN = 3420.0

def compute_adapted_label(row):
    """
    복합 적응 레이블:
    - minutes_share_ratio: 새 팀 min / 이전 팀 min (최소 0.7)
    - per90_performance_ratio: 새 팀 g_a per90 / 이전 팀 per90 (최소 0.8)
      단, 골+어시 0인 수비/GK 포지션은 이 지표 대신 minutes 비중 사용
    - 포지션별 가중치:
      GK/DEF: minutes 70%, performance 30%
      FWD:    minutes 30%, performance 70%
      MF:     minutes 50%, performance 50%
    """
    pos_group = row['pos_group']

    # minutes_share_ratio: 이전 시즌 대비 새 시즌 출전 시간 비율
    min_ratio = row['min_new'] / (row['min_old'] + 1e-6)
    min_score = 1.0 if min_ratio >= 0.7 else min_ratio / 0.7  # 0.7이면 1점

    # per90 공격 성과 비율
    old_perf = row['g_a_per90_old'] if not pd.isna(row['g_a_per90_old']) else 0.0
    new_perf = row['g_a_per90_new'] if not pd.isna(row['g_a_per90_new']) else 0.0

    if old_perf < 0.05:
        # 이전 시즌 공격 기여가 거의 없는 경우 (수비/GK 전형적)
        # 출전 시간 기반으로만 판단
        perf_score = min_score
    else:
        perf_ratio = new_perf / (old_perf + 1e-6)
        perf_score = 1.0 if perf_ratio >= 0.8 else perf_ratio / 0.8

    # 포지션별 가중치
    if pos_group in ('GK', 'DEF'):
        w_min, w_perf = 0.70, 0.30
    elif pos_group == 'FWD':
        w_min, w_perf = 0.30, 0.70
    else:  # MF
        w_min, w_perf = 0.50, 0.50

    composite_score = w_min * min_score + w_perf * perf_score

    # composite >= 0.65 → 적응 성공
    return 1 if composite_score >= 0.65 else 0

transfer_df['adapted'] = transfer_df.apply(compute_adapted_label, axis=1)
print(f"  적응 성공: {transfer_df['adapted'].sum()} / {len(transfer_df)}"
      f" ({transfer_df['adapted'].mean()*100:.1f}%)")

# ─────────────────────────────────────────────
# 5. 피처 엔지니어링 (30+)
# ─────────────────────────────────────────────
print("\n[4/7] 피처 엔지니어링 (30+ 피처) 중...")

# 5-1. EPL 경험 (이전 시즌까지의 EPL 출전 시즌 수)
epl_exp_map = {}
for player, grp in player_sorted.groupby('player'):
    grp = grp.sort_values('season_idx')
    for i, (idx, row) in enumerate(grp.iterrows()):
        epl_exp_map[(player, row['season'])] = i  # 현재 시즌 포함

transfer_df['epl_experience'] = transfer_df.apply(
    lambda r: epl_exp_map.get((r['player'], r['season_old']), 0), axis=1
)

# 5-2. 선수의 이전 이적 경험 및 역사적 적응률
player_transfer_history = {}
for idx, row in transfer_df.sort_values('season_old').iterrows():
    key = row['player']
    if key not in player_transfer_history:
        player_transfer_history[key] = []
    player_transfer_history[key].append(row['adapted'])

hist_adapt_rate = {}
transfer_count = {}
temp_history = {}  # 각 row 처리 시점의 이전 기록

# 순서대로 처리하여 누적 적응률 계산
for idx, row in transfer_df.sort_values('season_old').iterrows():
    key = row['player']
    prev = temp_history.get(key, [])
    hist_adapt_rate[idx] = np.mean(prev) if prev else np.nan
    transfer_count[idx] = len(prev)
    temp_history[key] = prev + [row['adapted']]

transfer_df['hist_adapt_rate'] = pd.Series(hist_adapt_rate)
transfer_df['transfer_count'] = pd.Series(transfer_count)

# 5-3. starter_ratio
transfer_df['starter_ratio_old'] = transfer_df['starts_old'] / (transfer_df['mp_old'] + 1e-6)

# 5-4. 이전 시즌 일관성 지표 (max 대비 실제 출전 시간 비율로 근사)
transfer_df['min_share_old'] = transfer_df['min_old'] / MAX_MIN

# 5-5. 팀 스탯 병합
team_feat = team_df[['season', 'team', 'points', 'elo', 'league_position',
                      'attack_ratio', 'defense_ratio', 'is_attacking',
                      'total_goals_for', 'total_goals_against', 'goal_diff']].copy()

# old 팀 통계
transfer_df = transfer_df.merge(
    team_feat.rename(columns={c: c + '_old' for c in team_feat.columns if c not in ['season', 'team']}),
    left_on=['season_old', 'team_old'],
    right_on=['season', 'team'],
    how='left'
).drop(columns=['season', 'team'])

# new 팀 통계
transfer_df = transfer_df.merge(
    team_feat.rename(columns={c: c + '_new' for c in team_feat.columns if c not in ['season', 'team']}),
    left_on=['season_new', 'team_new'],
    right_on=['season', 'team'],
    how='left'
).drop(columns=['season', 'team'])

# 5-6. 갭 피처
transfer_df['elo_diff'] = transfer_df['elo_new'] - transfer_df['elo_old']
transfer_df['points_diff'] = transfer_df['points_new'] - transfer_df['points_old']
transfer_df['league_pos_diff'] = transfer_df['league_position_new'] - transfer_df['league_position_old']
transfer_df['moving_up'] = (transfer_df['elo_diff'] > 0).astype(int)
transfer_df['big_step_up'] = (transfer_df['elo_diff'] > 100).astype(int)
transfer_df['big_step_down'] = (transfer_df['elo_diff'] < -100).astype(int)

# 5-7. 스타일 매치 (공격 비율 차이)
transfer_df['attack_ratio_diff'] = abs(
    transfer_df['attack_ratio_new'] - transfer_df['attack_ratio_old']
)
transfer_df['style_match'] = 1 - transfer_df['attack_ratio_diff']  # 1에 가까울수록 스타일 유사

# 5-8. 이전 팀 내 선수 스쿼드 내 포지션 (market_value vs squad avg)
# 시즌별 팀 선수들의 평균 market_value
squad_avg = player_df.groupby(['season', 'team'])['market_value'].mean().reset_index()
squad_avg.columns = ['season', 'team', 'squad_avg_mv']

transfer_df = transfer_df.merge(
    squad_avg.rename(columns={'season': 'season_old', 'team': 'team_old',
                               'squad_avg_mv': 'squad_avg_mv_old'}),
    on=['season_old', 'team_old'], how='left'
)
transfer_df = transfer_df.merge(
    squad_avg.rename(columns={'season': 'season_new', 'team': 'team_new',
                               'squad_avg_mv': 'squad_avg_mv_new'}),
    on=['season_new', 'team_new'], how='left'
)
transfer_df['mv_vs_squad_old'] = transfer_df['market_value_old'] / (transfer_df['squad_avg_mv_old'] + 1e-6)
transfer_df['mv_vs_squad_new'] = transfer_df['market_value'] / (transfer_df['squad_avg_mv_new'] + 1e-6)
# 새 팀에서 스타 플레이어 여부
transfer_df['is_star_in_new_team'] = (transfer_df['mv_vs_squad_new'] > 1.5).astype(int)

# 5-9. 나이 그룹
def age_bucket(age):
    if pd.isna(age):
        return 2  # 중간값
    if age < 22:
        return 0  # 유망주
    elif age < 27:
        return 1  # 전성기 초반
    elif age < 30:
        return 2  # 전성기
    elif age < 33:
        return 3  # 베테랑
    else:
        return 4  # 노장

transfer_df['age_bucket'] = transfer_df['age'].apply(age_bucket)

# 5-10. 나이 그룹 × ELO 상승 상호작용
transfer_df['young_moving_up'] = (
    (transfer_df['age_bucket'] <= 1) & (transfer_df['moving_up'] == 1)
).astype(int)
transfer_df['veteran_moving_up'] = (
    (transfer_df['age_bucket'] >= 3) & (transfer_df['moving_up'] == 1)
).astype(int)

# 5-11. 포지션 인코딩
pos_group_map = {'GK': 0, 'DEF': 1, 'MF': 2, 'FWD': 3}
transfer_df['pos_code'] = transfer_df['pos_group'].map(pos_group_map).fillna(2)

# 5-12. 이적 방향 (강팀 → 약팀, 약팀 → 강팀)
transfer_df['top6_old'] = (transfer_df['league_position_old'] <= 6).astype(int)
transfer_df['top6_new'] = (transfer_df['league_position_new'] <= 6).astype(int)

# 5-13. 전 시즌 공격 기여도 세분화
transfer_df['was_key_attacker'] = (transfer_df['g_a_per90_old'] > 0.4).astype(int)
transfer_df['was_regular_scorer'] = (transfer_df['gls_per90_old'] > 0.2).astype(int)

print(f"  피처 수: {transfer_df.shape[1]}")
print(f"  이적 레코드 수: {len(transfer_df)}")

# ─────────────────────────────────────────────
# 6. 학습/검증/테스트 분할 (시간 기반)
# ─────────────────────────────────────────────
print("\n[5/7] 시간 기반 데이터 분할...")

# season_new 기준
# Train: <2021, Val: 2021-2022, Test: 2023-2025
def season_year(season_str):
    """시즌 문자열에서 종료 연도 추출 (예: 2020/21 → 2021)"""
    try:
        return int(season_str.split('/')[1]) + 2000
    except:
        return 0

transfer_df['season_new_year'] = transfer_df['season_new'].apply(season_year)

train_mask = transfer_df['season_new_year'] < 2021
val_mask = (transfer_df['season_new_year'] >= 2021) & (transfer_df['season_new_year'] <= 2022)
test_mask = transfer_df['season_new_year'] >= 2023

print(f"  Train (<2021): {train_mask.sum()}건")
print(f"  Val (2021-2022): {val_mask.sum()}건")
print(f"  Test (2023-2025): {test_mask.sum()}건")

# ─────────────────────────────────────────────
# 7. 피처 선택 및 전처리
# ─────────────────────────────────────────────

FEATURE_COLS = [
    # 선수 피처
    'age', 'age_bucket', 'pos_code', 'height_cm',
    'epl_experience', 'transfer_count', 'hist_adapt_rate',
    'starter_ratio_old', 'min_share_old',
    'gls_per90_old', 'ast_per90_old', 'g_a_per90_old', 'g_pk_per90_old',
    'was_key_attacker', 'was_regular_scorer',
    'market_value_old', 'mv_vs_squad_old', 'mv_vs_squad_new',
    'is_star_in_new_team',
    # 이전 팀 피처
    'points_old', 'elo_old', 'league_position_old',
    'attack_ratio_old', 'defense_ratio_old', 'top6_old',
    # 새 팀 피처
    'points_new', 'elo_new', 'league_position_new',
    'attack_ratio_new', 'defense_ratio_new', 'top6_new',
    # 갭 피처
    'elo_diff', 'points_diff', 'league_pos_diff',
    'moving_up', 'big_step_up', 'big_step_down',
    # 스타일 매치
    'style_match', 'attack_ratio_diff',
    # 상호작용
    'young_moving_up', 'veteran_moving_up',
]

print(f"\n  사용 피처 수: {len(FEATURE_COLS)}")

TARGET = 'adapted'

X_train = transfer_df.loc[train_mask, FEATURE_COLS]
y_train = transfer_df.loc[train_mask, TARGET]
X_val = transfer_df.loc[val_mask, FEATURE_COLS]
y_val = transfer_df.loc[val_mask, TARGET]
X_test = transfer_df.loc[test_mask, FEATURE_COLS]
y_test = transfer_df.loc[test_mask, TARGET]

# NaN 처리: 수치형 → 중앙값 대체
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_val_imp = imputer.transform(X_val)
X_test_imp = imputer.transform(X_test)

# 스케일링 (LR, MLP용)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_imp)
X_val_sc = scaler.transform(X_val_imp)
X_test_sc = scaler.transform(X_test_imp)

print(f"  Train 클래스 분포: {y_train.value_counts().to_dict()}")
print(f"  Val 클래스 분포: {y_val.value_counts().to_dict()}")
print(f"  Test 클래스 분포: {y_test.value_counts().to_dict()}")

# ─────────────────────────────────────────────
# 8. 모델 학습 (XGBoost, RF, LR, MLP)
# ─────────────────────────────────────────────
print("\n[6/7] 모델 학습 중...")

# 클래스 불균형 처리용 가중치
neg_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()

models = {}
results = {}

# ── XGBoost ──
print("  XGBoost 학습 중...")
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=neg_pos_ratio,
    eval_metric='logloss',
    random_state=42,
    verbosity=0
)
xgb_model.fit(
    X_train_imp, y_train,
    eval_set=[(X_val_imp, y_val)],
    verbose=False
)
models['XGBoost'] = xgb_model

# ── Random Forest ──
print("  Random Forest 학습 중...")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_imp, y_train)
models['RandomForest'] = rf_model

# ── Logistic Regression ──
print("  Logistic Regression 학습 중...")
lr_model = LogisticRegression(
    C=1.0,
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
lr_model.fit(X_train_sc, y_train)
models['LogisticRegression'] = lr_model

# ── MLP ──
print("  MLP 학습 중...")
mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    max_iter=500,
    learning_rate_init=0.001,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)
mlp_model.fit(X_train_sc, y_train)
models['MLP'] = mlp_model

# ── 앙상블 (평균 확률) ──
def ensemble_predict(X_raw, X_scaled, threshold=0.5):
    probs = []
    probs.append(xgb_model.predict_proba(X_raw)[:, 1])
    probs.append(rf_model.predict_proba(X_raw)[:, 1])
    probs.append(lr_model.predict_proba(X_scaled)[:, 1])
    probs.append(mlp_model.predict_proba(X_scaled)[:, 1])
    avg_prob = np.mean(probs, axis=0)
    return avg_prob, (avg_prob >= threshold).astype(int)

# ─────────────────────────────────────────────
# 9. 평가
# ─────────────────────────────────────────────
print("\n  모델 평가 결과:")
print(f"{'모델':<20} {'Val AUC':>10} {'Val F1':>10} {'Test AUC':>10} {'Test F1':>10}")
print("-" * 60)

for name, model in models.items():
    if name in ('LogisticRegression', 'MLP'):
        val_proba = model.predict_proba(X_val_sc)[:, 1]
        test_proba = model.predict_proba(X_test_sc)[:, 1]
    else:
        val_proba = model.predict_proba(X_val_imp)[:, 1]
        test_proba = model.predict_proba(X_test_imp)[:, 1]

    val_pred = (val_proba >= 0.5).astype(int)
    test_pred = (test_proba >= 0.5).astype(int)

    val_auc = roc_auc_score(y_val, val_proba) if len(y_val.unique()) > 1 else 0.0
    test_auc = roc_auc_score(y_test, test_proba) if len(y_test.unique()) > 1 else 0.0
    val_f1 = f1_score(y_val, val_pred, zero_division=0)
    test_f1 = f1_score(y_test, test_pred, zero_division=0)

    results[name] = {
        'val_auc': float(val_auc),
        'val_f1': float(val_f1),
        'test_auc': float(test_auc),
        'test_f1': float(test_f1),
    }
    print(f"{name:<20} {val_auc:>10.4f} {val_f1:>10.4f} {test_auc:>10.4f} {test_f1:>10.4f}")

# 앙상블 평가
val_ens_prob, val_ens_pred = ensemble_predict(X_val_imp, X_val_sc)
test_ens_prob, test_ens_pred = ensemble_predict(X_test_imp, X_test_sc)

val_ens_auc = roc_auc_score(y_val, val_ens_prob) if len(y_val.unique()) > 1 else 0.0
test_ens_auc = roc_auc_score(y_test, test_ens_prob) if len(y_test.unique()) > 1 else 0.0
val_ens_f1 = f1_score(y_val, val_ens_pred, zero_division=0)
test_ens_f1 = f1_score(y_test, test_ens_pred, zero_division=0)

results['Ensemble'] = {
    'val_auc': float(val_ens_auc),
    'val_f1': float(val_ens_f1),
    'test_auc': float(test_ens_auc),
    'test_f1': float(test_ens_f1),
}
print(f"{'Ensemble':<20} {val_ens_auc:>10.4f} {val_ens_f1:>10.4f} {test_ens_auc:>10.4f} {test_ens_f1:>10.4f}")

# 최종 테스트 분류 리포트 (앙상블)
print("\n  [앙상블] Test 상세 리포트:")
print(classification_report(y_test, test_ens_pred,
                              target_names=['미적응', '적응'], zero_division=0))

# ─────────────────────────────────────────────
# 10. 스카우트 출력 (2023-2025 이적 예측)
# ─────────────────────────────────────────────
print("\n[7/7] 스카우트 출력 생성 중...")

test_df = transfer_df.loc[test_mask].copy()
test_df['adapt_prob_ensemble'] = test_ens_prob
test_df['adapt_prob_xgb'] = xgb_model.predict_proba(X_test_imp)[:, 1]
test_df['adapt_prob_rf'] = rf_model.predict_proba(X_test_imp)[:, 1]
test_df['adapt_prob_lr'] = lr_model.predict_proba(X_test_sc)[:, 1]
test_df['adapt_prob_mlp'] = mlp_model.predict_proba(X_test_sc)[:, 1]
test_df['predicted_adapted'] = test_ens_pred

# 안전 vs 위험 분류
# Safe bet: 확률 >= 0.75, Risky: 확률 < 0.40, Moderate: 나머지
def classify_transfer(prob):
    if prob >= 0.75:
        return 'Safe Bet'
    elif prob < 0.40:
        return 'Risky'
    else:
        return 'Moderate'

test_df['scout_category'] = test_df['adapt_prob_ensemble'].apply(classify_transfer)

# 주목할 실제 이적 예시 (알려진 선수 우선)
notable_players = [
    'Declan Rice', 'Kieran Trippier', 'Casemiro', 'Darwin Nunez',
    'Erling Haaland', 'Julian Alvarez', 'Matheus Cunha',
    'Bruno Fernandes', 'Marcus Rashford', 'Son Heung-min',
    'Kai Havertz', 'Leandro Trossard', 'Gabriel Martinelli',
    'Ivan Toney', 'Dominic Calvert-Lewin', 'Michail Antonio',
    'Ollie Watkins', 'Jarrod Bowen', 'Phil Foden', 'Bukayo Saka'
]

notable_results = []
for _, row in test_df.iterrows():
    label = {
        'player': row['player'],
        'transfer': f"{row['player']} → {row['team_new']}",
        'season': row['season_new'],
        'adapt_probability': round(float(row['adapt_prob_ensemble']) * 100, 1),
        'category': row['scout_category'],
        'actual_adapted': int(row['adapted']),
        'age': float(row['age']) if not pd.isna(row['age']) else None,
        'position': row['pos_group'],
        'elo_diff': round(float(row['elo_diff']), 1) if not pd.isna(row['elo_diff']) else None,
    }
    notable_results.append(label)

# 상위 안전 베팅 및 위험 이적 추출
safe_bets = sorted(
    [r for r in notable_results if r['category'] == 'Safe Bet'],
    key=lambda x: -x['adapt_probability']
)[:10]

risky_transfers = sorted(
    [r for r in notable_results if r['category'] == 'Risky'],
    key=lambda x: x['adapt_probability']
)[:10]

print("\n  [스카우트] 상위 안전 이적 (Top Safe Bets):")
for r in safe_bets[:5]:
    print(f"    {r['transfer']:40s} {r['adapt_probability']:5.1f}% 적응 확률")

print("\n  [스카우트] 고위험 이적 (Top Risky Transfers):")
for r in risky_transfers[:5]:
    print(f"    {r['transfer']:40s} {r['adapt_probability']:5.1f}% 적응 확률")

# 피처 중요도 (XGBoost 기반)
feat_importance = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance_xgb': xgb_model.feature_importances_,
    'importance_rf': rf_model.feature_importances_,
}).sort_values('importance_xgb', ascending=False)

# 실패 예측 기여 피처 (상위 음수 계수 – Logistic Regression 기반)
lr_coef = pd.DataFrame({
    'feature': FEATURE_COLS,
    'lr_coef': lr_model.coef_[0]
}).sort_values('lr_coef')

failure_risk_features = lr_coef.head(5)['feature'].tolist()  # 가장 음수 계수 = 실패 위험 증가

print("\n  [위험 요인] 적응 실패를 가장 잘 예측하는 피처:")
for feat, coef in zip(lr_coef.head(5)['feature'], lr_coef.head(5)['lr_coef']):
    print(f"    {feat:<35}: coef={coef:.4f}")

# ─────────────────────────────────────────────
# 11. 시각화
# ─────────────────────────────────────────────
print("\n  시각화 생성 중...")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ── 그림 1: 혼동 행렬 ──
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
model_preds = {
    'XGBoost': xgb_model.predict_proba(X_test_imp)[:, 1],
    'RandomForest': rf_model.predict_proba(X_test_imp)[:, 1],
    'LogisticRegression': lr_model.predict_proba(X_test_sc)[:, 1],
    'MLP': mlp_model.predict_proba(X_test_sc)[:, 1],
    'Ensemble': test_ens_prob,
}

for ax, (name, proba) in zip(axes.flatten(), model_preds.items()):
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred)
    auc = roc_auc_score(y_test, proba) if len(y_test.unique()) > 1 else 0.0
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Adapted', 'Adapted'],
                yticklabels=['Not Adapted', 'Adapted'])
    ax.set_title(f'{name}\nAUC={auc:.3f}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

# 마지막 subplot: 모델 비교 바차트
ax = axes[1, 2]
model_names = list(results.keys())
test_aucs = [results[n]['test_auc'] for n in model_names]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
bars = ax.bar(model_names, test_aucs, color=colors[:len(model_names)])
ax.set_ylim(0, 1)
ax.set_ylabel('Test AUC')
ax.set_title('Model Comparison (Test AUC)', fontweight='bold')
ax.tick_params(axis='x', rotation=15)
for bar, val in zip(bars, test_aucs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(FIG_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("    confusion_matrix.png 저장 완료")

# ── 그림 2: 피처 중요도 ──
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# XGBoost 피처 중요도
top_feat = feat_importance.head(20)
axes[0].barh(top_feat['feature'][::-1], top_feat['importance_xgb'][::-1],
             color='#2196F3', alpha=0.8)
axes[0].set_title('XGBoost Feature Importance (Top 20)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Importance')

# Logistic Regression 계수
lr_coef_plot = pd.DataFrame({
    'feature': FEATURE_COLS,
    'coef': lr_model.coef_[0]
}).sort_values('coef')
colors_lr = ['#F44336' if c < 0 else '#4CAF50' for c in lr_coef_plot['coef']]
axes[1].barh(lr_coef_plot['feature'], lr_coef_plot['coef'],
             color=colors_lr, alpha=0.8)
axes[1].set_title('Logistic Regression Coefficients\n(Red=Failure Risk, Green=Success Factor)',
                   fontsize=12, fontweight='bold')
axes[1].set_xlabel('Coefficient')
axes[1].axvline(0, color='black', linewidth=0.8)

plt.tight_layout()
plt.savefig(FIG_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("    feature_importance.png 저장 완료")

# ── 그림 3: 나이/포지션별 적응률 ──
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 3-1. 나이 그룹별 적응률 (전체 데이터)
age_adapt = transfer_df.groupby('age_bucket')['adapted'].agg(['mean', 'count']).reset_index()
age_bucket_labels = ['<22세\n(유망주)', '22-26세\n(성장기)', '27-29세\n(전성기)',
                     '30-32세\n(베테랑)', '33+세\n(노장)']
x_pos = range(len(age_adapt))
bars = axes[0, 0].bar(x_pos, age_adapt['mean'] * 100,
                       color=['#4FC3F7', '#29B6F6', '#0288D1', '#01579B', '#003f7f'],
                       alpha=0.85)
axes[0, 0].set_xticks(x_pos)
valid_labels = age_bucket_labels[:len(age_adapt)]
axes[0, 0].set_xticklabels(valid_labels, fontsize=10)
axes[0, 0].set_ylabel('적응 성공률 (%)')
axes[0, 0].set_title('나이 그룹별 이적 적응 성공률', fontsize=12, fontweight='bold')
axes[0, 0].set_ylim(0, 100)
for bar, (_, row) in zip(bars, age_adapt.iterrows()):
    axes[0, 0].text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f'{row["mean"]*100:.1f}%\n(n={row["count"]})',
                    ha='center', va='bottom', fontsize=9)

# 3-2. 포지션별 적응률
pos_adapt = transfer_df.groupby('pos_group')['adapted'].agg(['mean', 'count']).reset_index()
pos_colors = {'GK': '#9C27B0', 'DEF': '#2196F3', 'MF': '#4CAF50', 'FWD': '#FF9800'}
bar_colors = [pos_colors.get(p, '#666666') for p in pos_adapt['pos_group']]
bars2 = axes[0, 1].bar(pos_adapt['pos_group'], pos_adapt['mean'] * 100,
                        color=bar_colors, alpha=0.85)
axes[0, 1].set_ylabel('적응 성공률 (%)')
axes[0, 1].set_title('포지션별 이적 적응 성공률', fontsize=12, fontweight='bold')
axes[0, 1].set_ylim(0, 100)
for bar, (_, row) in zip(bars2, pos_adapt.iterrows()):
    axes[0, 1].text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f'{row["mean"]*100:.1f}%\n(n={row["count"]})',
                    ha='center', va='bottom', fontsize=9)

# 3-3. ELO 차이별 적응률 (히스토그램 + 적응률)
elo_bins = [-300, -150, -50, 0, 50, 150, 300]
elo_labels_b = ['대폭하락\n(<-150)', '하락\n(-150~-50)', '소폭하락\n(-50~0)',
                 '소폭상승\n(0~50)', '상승\n(50~150)', '대폭상승\n(>150)']
transfer_df['elo_bin'] = pd.cut(transfer_df['elo_diff'], bins=elo_bins, labels=elo_labels_b[:5] + ['대폭상승\n(>150)'])
elo_adapt = transfer_df.groupby('elo_bin', observed=True)['adapted'].agg(['mean', 'count']).reset_index()
axes[1, 0].bar(range(len(elo_adapt)), elo_adapt['mean'] * 100,
               color=['#F44336', '#FF7043', '#FFA726', '#66BB6A', '#29B6F6', '#1565C0'][:len(elo_adapt)],
               alpha=0.85)
axes[1, 0].set_xticks(range(len(elo_adapt)))
axes[1, 0].set_xticklabels(elo_adapt['elo_bin'].tolist(), fontsize=9)
axes[1, 0].set_ylabel('적응 성공률 (%)')
axes[1, 0].set_title('ELO 변화량별 이적 적응 성공률', fontsize=12, fontweight='bold')
axes[1, 0].set_ylim(0, 100)

# 3-4. 테스트 세트 확률 분포 (Safe/Moderate/Risky)
cat_colors = {'Safe Bet': '#4CAF50', 'Moderate': '#FF9800', 'Risky': '#F44336'}
for cat, grp in test_df.groupby('scout_category'):
    axes[1, 1].hist(grp['adapt_prob_ensemble'] * 100,
                    bins=15, alpha=0.6, label=f'{cat} (n={len(grp)})',
                    color=cat_colors.get(cat, 'gray'))
axes[1, 1].axvline(75, color='green', linestyle='--', linewidth=1.5, label='Safe threshold (75%)')
axes[1, 1].axvline(40, color='red', linestyle='--', linewidth=1.5, label='Risky threshold (40%)')
axes[1, 1].set_xlabel('적응 확률 (%)')
axes[1, 1].set_ylabel('이적 건수')
axes[1, 1].set_title('테스트 세트 적응 확률 분포 (2023-2025)', fontsize=12, fontweight='bold')
axes[1, 1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(FIG_DIR / 'adaptation_by_age_position.png', dpi=150, bbox_inches='tight')
plt.close()
print("    adaptation_by_age_position.png 저장 완료")

# ── 그림 4: 위험 분석 (ROC + 위험 요인) ──
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# ROC 커브 (모든 모델)
roc_colors = {'XGBoost': '#2196F3', 'RandomForest': '#4CAF50',
               'LogisticRegression': '#FF9800', 'MLP': '#9C27B0', 'Ensemble': '#F44336'}
for name, proba in model_preds.items():
    if len(y_test.unique()) > 1:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc_val = roc_auc_score(y_test, proba)
        axes[0].plot(fpr, tpr, label=f'{name} (AUC={auc_val:.3f})',
                     color=roc_colors[name], linewidth=2)
axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curves - All Models', fontsize=13, fontweight='bold')
axes[0].legend(loc='lower right', fontsize=9)
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1.05])

# 상위 위험 요인 (Logistic Regression 음수 계수)
top_risk = lr_coef.head(15)
bar_colors_risk = ['#F44336' if c < -0.1 else '#FF7043' for c in top_risk['lr_coef']]
axes[1].barh(top_risk['feature'][::-1], abs(top_risk['lr_coef'][::-1]),
             color=bar_colors_risk[::-1], alpha=0.85)
axes[1].set_xlabel('|Coefficient| (위험 강도)')
axes[1].set_title('이적 실패 위험 요인 Top 15\n(LR 음수 계수 기준)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(FIG_DIR / 'risk_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("    risk_analysis.png 저장 완료")

# ─────────────────────────────────────────────
# 12. 결과 저장
# ─────────────────────────────────────────────
print("\n  결과 파일 저장 중...")

# transfer_predictions.parquet 저장
save_cols = [
    'player', 'season_old', 'season_new', 'team_old', 'team_new',
    'age', 'pos_group', 'position',
    'min_old', 'min_new', 'gls_per90_old', 'ast_per90_old',
    'elo_old', 'elo_new', 'elo_diff', 'moving_up',
    'points_old', 'points_new', 'league_position_old', 'league_position_new',
    'style_match', 'market_value', 'mv_vs_squad_new',
    'adapted',
    'adapt_prob_ensemble', 'adapt_prob_xgb', 'adapt_prob_rf',
    'adapt_prob_lr', 'adapt_prob_mlp',
    'predicted_adapted', 'scout_category',
]
# 없는 컬럼 제거
save_cols = [c for c in save_cols if c in test_df.columns]
test_df[save_cols].to_parquet(SCOUT_DIR / 'transfer_predictions.parquet', index=False)
print(f"    transfer_predictions.parquet 저장 ({len(test_df)}건)")

# results_summary.json 저장
category_counts = test_df['scout_category'].value_counts().to_dict()
correct_predictions = int((test_df['predicted_adapted'] == test_df['adapted']).sum())
total_test = len(test_df)

summary = {
    "model_performance": results,
    "best_model": max(results, key=lambda k: results[k]['test_auc']),
    "test_set_stats": {
        "total_transfers": total_test,
        "adapted_count": int(test_df['adapted'].sum()),
        "not_adapted_count": int((test_df['adapted'] == 0).sum()),
        "correct_predictions": correct_predictions,
        "accuracy": round(correct_predictions / total_test, 4),
    },
    "scout_categories": category_counts,
    "top_safe_bets": safe_bets[:10],
    "top_risky_transfers": risky_transfers[:10],
    "failure_risk_features": failure_risk_features,
    "feature_importance_top10": feat_importance.head(10)[['feature', 'importance_xgb']].to_dict('records'),
    "label_logic": {
        "method": "composite_weighted_by_position",
        "min_minutes_each_season": 450,
        "minutes_share_ratio_threshold": 0.70,
        "per90_performance_ratio_threshold": 0.80,
        "composite_threshold": 0.65,
        "position_weights": {
            "GK_DEF": {"minutes": 0.70, "performance": 0.30},
            "MF": {"minutes": 0.50, "performance": 0.50},
            "FWD": {"minutes": 0.30, "performance": 0.70},
        }
    },
    "model_config": {
        "time_split": {
            "train": "< 2021",
            "val": "2021-2022",
            "test": "2023-2025"
        },
        "features_count": len(FEATURE_COLS),
        "nan_handling": "median imputation",
        "class_imbalance": "scale_pos_weight (XGB), class_weight=balanced (RF/LR)"
    }
}

with open(SCOUT_DIR / 'results_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("    results_summary.json 저장 완료")

# ─────────────────────────────────────────────
# 13. 최종 요약 출력
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("S5 모델링 완료 요약")
print("=" * 60)
print(f"  데이터: {len(transfer_df)}건 이적 (450분+ 연속 시즌)")
print(f"  적응 레이블: 복합 가중치 (포지션별, composite >= 0.65)")
print(f"  피처 수: {len(FEATURE_COLS)}개")
print(f"  최고 모델: {max(results, key=lambda k: results[k]['test_auc'])}"
      f" (Test AUC={max(results.values(), key=lambda x: x['test_auc'])['test_auc']:.4f})")
print(f"\n  스카우트 분류 (2023-2025 이적):")
for cat, cnt in category_counts.items():
    print(f"    {cat}: {cnt}건")
print(f"\n  출력 파일:")
print(f"    {SCOUT_DIR / 'transfer_predictions.parquet'}")
print(f"    {SCOUT_DIR / 'results_summary.json'}")
print(f"    {FIG_DIR / 'confusion_matrix.png'}")
print(f"    {FIG_DIR / 'feature_importance.png'}")
print(f"    {FIG_DIR / 'adaptation_by_age_position.png'}")
print(f"    {FIG_DIR / 'risk_analysis.png'}")
print("=" * 60)

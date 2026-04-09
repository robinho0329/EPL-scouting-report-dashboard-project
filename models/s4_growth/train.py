"""
S4: 스카우트를 위한 선수 성장 잠재력 예측 모델
EPL 데이터 프로젝트 - 포지션 인식 복합 성과 점수 기반 성장 예측
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb

warnings.filterwarnings('ignore')

# ───────────────────────── 경로 설정 ──────────────────────────
BASE_DIR   = Path("C:/Users/xcv54/workspace/EPL project")
DATA_DIR   = BASE_DIR / "data" / "processed"
SCOUT_DIR  = BASE_DIR / "data" / "scout"
FIG_DIR    = BASE_DIR / "models" / "s4_growth" / "figures"
MODEL_DIR  = BASE_DIR / "models" / "s4_growth"

SCOUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ───────────────────────── 스타일 설정 ────────────────────────
plt.rcParams.update({
    'figure.dpi': 120,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
})
PALETTE = {
    'FW': '#e74c3c', 'MF': '#3498db', 'DF': '#2ecc71', 'GK': '#f39c12'
}

# ═══════════════════════════════════════════════════════════════
# 1. 데이터 로드 및 전처리
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("[1] 데이터 로드 중...")
print("=" * 60)

season_df = pd.read_parquet(DATA_DIR / "player_season_stats.parquet")
match_df  = pd.read_parquet(DATA_DIR / "player_match_logs.parquet")
team_df   = pd.read_parquet(DATA_DIR / "team_season_summary.parquet")

print(f"  시즌 스탯: {season_df.shape}")
print(f"  경기 로그: {match_df.shape}")
print(f"  팀 요약:   {team_df.shape}")

# ───── 시즌 번호 변환 (2000/01 → 2000) ─────
def season_to_year(s):
    """시즌 문자열에서 시작 연도 추출"""
    try:
        return int(str(s).split('/')[0])
    except:
        return np.nan

season_df['season_year'] = season_df['season'].apply(season_to_year)
team_df['season_year']   = team_df['Season'].apply(season_to_year)

# ───── 포지션 단순화 ─────
def simplify_position(pos):
    """포지션을 FW/MF/DF/GK 4개로 단순화"""
    if pd.isna(pos):
        return 'MF'
    pos = str(pos).upper()
    if 'GK' in pos:
        return 'GK'
    if 'FW' in pos:
        return 'FW'
    if 'DF' in pos:
        return 'DF'
    return 'MF'

season_df['pos_simple'] = season_df['pos'].apply(simplify_position)

# ───── 최소 출전 필터 (팀당 경기의 30% 이상) ─────
season_df = season_df[season_df['min'].fillna(0) >= 270].copy()
print(f"  최소 출전(270분) 필터 후: {season_df.shape[0]}행")

# ═══════════════════════════════════════════════════════════════
# 2. 경기 로그에서 수비 스탯 집계 (포지션별 가중치에 활용)
# ═══════════════════════════════════════════════════════════════

print("\n[2] 경기 로그 집계 중...")

# 상세 스탯이 있는 데이터만 사용
match_detail = match_df[match_df['detail_stats_available'] == True].copy()
match_detail['season_year'] = match_detail['season'].apply(season_to_year)

# 선수-시즌별 수비 스탯 집계
def safe_sum(x):
    return x.fillna(0).sum()

match_agg = (
    match_detail
    .groupby(['player', 'season_year'])
    .agg(
        tklw_total=('tklw', safe_sum),
        int_total=('int', safe_sum),
        sh_total=('sh', safe_sum),
        sot_total=('sot', safe_sum),
        fls_total=('fls', safe_sum),
        fld_total=('fld', safe_sum),
        crs_total=('crs', safe_sum),
        match_count=('min', 'count')
    )
    .reset_index()
)
print(f"  경기 집계 완료: {match_agg.shape}")

# ═══════════════════════════════════════════════════════════════
# 3. 포지션별 복합 성과 점수 계산
# ═══════════════════════════════════════════════════════════════

print("\n[3] 복합 성과 점수 계산 중...")

# 시즌 스탯에 수비 스탯 병합
season_df = season_df.merge(match_agg, on=['player', 'season_year'], how='left')

# 90분당 스탯 (이미 season_df에 90s 컬럼 존재)
season_df['90s_safe'] = season_df['90s'].fillna(1).clip(lower=0.1)

# per-90 스탯 계산
season_df['gls_p90']  = season_df['gls'].fillna(0) / season_df['90s_safe']
season_df['ast_p90']  = season_df['ast'].fillna(0) / season_df['90s_safe']
season_df['tklw_p90'] = season_df['tklw_total'].fillna(0) / season_df['90s_safe']
season_df['int_p90']  = season_df['int_total'].fillna(0) / season_df['90s_safe']
season_df['sh_p90']   = season_df['sh_total'].fillna(0) / season_df['90s_safe']
season_df['crs_p90']  = season_df['crs_total'].fillna(0) / season_df['90s_safe']
season_df['fld_p90']  = season_df['fld_total'].fillna(0) / season_df['90s_safe']

def compute_composite_score(row):
    """
    포지션별 복합 성과 점수 계산
    FW: 득점+도움 중심
    MF: 득점+도움+크로스+파울 유도 균형
    DF: 태클+인터셉트 중심
    GK: 기본 출전 기반
    """
    pos = row['pos_simple']
    if pos == 'FW':
        score = (row['gls_p90'] * 3.0 +
                 row['ast_p90'] * 2.0 +
                 row['sh_p90']  * 0.3 +
                 row['fld_p90'] * 0.5)
    elif pos == 'MF':
        score = (row['gls_p90']  * 2.0 +
                 row['ast_p90']  * 2.5 +
                 row['tklw_p90'] * 0.8 +
                 row['int_p90']  * 0.8 +
                 row['crs_p90']  * 0.5 +
                 row['fld_p90']  * 0.4)
    elif pos == 'DF':
        score = (row['tklw_p90'] * 3.0 +
                 row['int_p90']  * 3.0 +
                 row['gls_p90']  * 1.0 +
                 row['ast_p90']  * 0.8 +
                 row['fld_p90']  * 0.3)
    else:  # GK
        score = row['90s_safe'] * 0.1  # 골키퍼는 출전 시간 기반 기본 점수
    return score

season_df['raw_score'] = season_df.apply(compute_composite_score, axis=1)

# 포지션×시즌 내 Z-점수 표준화
def zscore_within_group(df, col, group_cols):
    """그룹 내 Z-점수 계산 (그룹 크기 < 3이면 0 반환)"""
    result = pd.Series(np.nan, index=df.index)
    for key, idx in df.groupby(group_cols).groups.items():
        vals = df.loc[idx, col]
        if len(vals) >= 3:
            mu, sigma = vals.mean(), vals.std()
            if sigma > 0:
                result.loc[idx] = (vals - mu) / sigma
            else:
                result.loc[idx] = 0.0
        else:
            result.loc[idx] = 0.0
    return result

season_df['perf_score'] = zscore_within_group(
    season_df, 'raw_score', ['pos_simple', 'season_year']
)
print(f"  성과 점수 계산 완료: {season_df['perf_score'].describe()}")

# ═══════════════════════════════════════════════════════════════
# 4. 성장 타겟 변수 생성 (N+1 시즌 점수 - N 시즌 점수)
# ═══════════════════════════════════════════════════════════════

print("\n[4] 성장 타겟 변수 생성 중...")

# 다음 시즌 점수 병합
next_season_score = (
    season_df[['player', 'season_year', 'perf_score', 'age', 'min', 'team']]
    .rename(columns={
        'season_year': 'next_season_year',
        'perf_score': 'next_perf_score',
        'age': 'next_age',
        'min': 'next_min',
        'team': 'next_team'
    })
)

# 현재 시즌에 다음 시즌(+1) 점수를 붙이기 위해 조인 키 생성
season_df['next_season_year'] = season_df['season_year'] + 1

df = season_df.merge(
    next_season_score,
    on=['player', 'next_season_year'],
    how='inner'
)

# 성장 타겟: 다음 시즌 점수 - 현재 시즌 점수
df['growth_target'] = df['next_perf_score'] - df['perf_score']

print(f"  성장 타겟 데이터 크기: {df.shape[0]}행")
print(f"  성장 분포: mean={df['growth_target'].mean():.3f}, std={df['growth_target'].std():.3f}")
print(f"  성장 선수 비율: {(df['growth_target'] > 0).mean():.1%}")

# ═══════════════════════════════════════════════════════════════
# 5. 피처 엔지니어링 (28+ 피처)
# ═══════════════════════════════════════════════════════════════

print("\n[5] 피처 엔지니어링 중...")

# ───── 팀 포인트 병합 ─────
team_points = team_df[['team', 'season_year', 'points']].copy()
df = df.merge(team_points, on=['team', 'season_year'], how='left')

# ───── EPL 누적 시즌 수 계산 ─────
df_sorted = df.sort_values(['player', 'season_year'])
df_sorted['epl_seasons'] = df_sorted.groupby('player').cumcount() + 1
df['epl_seasons'] = df_sorted['epl_seasons'].values

# ───── 출전 일관성: 경기당 출전시간 변동계수 계산 ─────
# (match_df에서 선수-시즌별 std/mean of min)
match_consistency = (
    match_df[match_df['min'].notna()]
    .groupby(['player', 'season'])
    .agg(
        min_mean=('min', 'mean'),
        min_std=('min', 'std'),
        match_count_all=('min', 'count')
    )
    .reset_index()
)
match_consistency['season_year'] = match_consistency['season'].apply(season_to_year)
match_consistency['consistency'] = np.where(
    match_consistency['min_mean'] > 0,
    1 - (match_consistency['min_std'].fillna(0) / match_consistency['min_mean']).clip(0, 1),
    0
)
df = df.merge(
    match_consistency[['player', 'season_year', 'consistency', 'match_count_all']],
    on=['player', 'season_year'],
    how='left'
)

# ───── 스타터 비율 ─────
df['starter_ratio'] = (df['starts'].fillna(0) / df['mp'].replace(0, np.nan)).clip(0, 1)

# ───── 마켓 밸류 모멘텀 (전 시즌 대비 변화율) ─────
mv_lag = (
    df[['player', 'season_year', 'market_value']]
    .rename(columns={'season_year': 'prev_year', 'market_value': 'prev_mv'})
)
mv_lag['season_year'] = mv_lag['prev_year'] + 1
df = df.merge(mv_lag[['player', 'season_year', 'prev_mv']], on=['player', 'season_year'], how='left')
df['mv_change_rate'] = np.where(
    (df['prev_mv'].notna()) & (df['prev_mv'] > 0),
    (df['market_value'].fillna(0) - df['prev_mv']) / df['prev_mv'],
    np.nan
)
df['mv_change_rate'] = df['mv_change_rate'].clip(-2, 5)

# ───── 과거 성과 트렌드 (최근 2-3시즌 기울기) ─────
# 선수별 최근 3시즌 성과 점수의 선형 기울기
def compute_trend_features(df_in, n_seasons=3):
    """최근 n_seasons 시즌의 성과 점수 기울기와 피크 성과 계산"""
    df_sorted = df_in.sort_values(['player', 'season_year'])
    trends, peaks, lag1_scores, lag2_scores = [], [], [], []

    for _, group in df_sorted.groupby('player'):
        group = group.sort_values('season_year').reset_index(drop=True)
        slope_list, peak_list, lag1_list, lag2_list = [], [], [], []

        for i, row in group.iterrows():
            # 현재 행 이전 데이터 (현재 시즌 포함하지 않음)
            past = group[group['season_year'] < row['season_year']].tail(n_seasons)

            # 기울기 계산
            if len(past) >= 2:
                x = past['season_year'].values
                y = past['perf_score'].values
                slope, _, _, _, _ = stats.linregress(x - x.mean(), y)
                slope_list.append(slope)
            else:
                slope_list.append(np.nan)

            # 피크 성과 (이전 시즌들 중 최대)
            if len(past) >= 1:
                peak_list.append(past['perf_score'].max())
            else:
                peak_list.append(np.nan)

            # 1시즌 전 성과
            if i >= 1 and group.loc[i-1, 'season_year'] == row['season_year'] - 1:
                lag1_list.append(group.loc[i-1, 'perf_score'])
            else:
                lag1_list.append(np.nan)

            # 2시즌 전 성과
            if i >= 2 and group.loc[i-2, 'season_year'] == row['season_year'] - 2:
                lag2_list.append(group.loc[i-2, 'perf_score'])
            else:
                lag2_list.append(np.nan)

        trends.extend(slope_list)
        peaks.extend(peak_list)
        lag1_scores.extend(lag1_list)
        lag2_scores.extend(lag2_list)

    df_in = df_in.copy()
    df_in['perf_trend'] = trends
    df_in['peak_perf'] = peaks
    df_in['lag1_perf'] = lag1_list if len(lag1_list) == len(df_in) else np.nan
    df_in['lag2_perf'] = lag2_list if len(lag2_list) == len(df_in) else np.nan
    return df_in, trends, peaks, lag1_scores, lag2_scores

print("  과거 트렌드 계산 중... (시간이 걸릴 수 있음)")
df_sorted_all = df.sort_values(['player', 'season_year']).reset_index(drop=True)

# 벡터화된 방식으로 트렌드 계산
trend_vals, peak_vals, lag1_vals, lag2_vals = [], [], [], []

for player_name, group in df_sorted_all.groupby('player'):
    group = group.sort_values('season_year').reset_index()
    orig_indices = group['index'].values
    scores = group['perf_score'].values
    years  = group['season_year'].values

    for j in range(len(group)):
        past_mask = years < years[j]
        past_scores = scores[past_mask]
        past_years  = years[past_mask]

        # 트렌드: 최근 3시즌
        recent = past_scores[-3:] if len(past_scores) >= 2 else []
        recent_years = past_years[-3:] if len(past_years) >= 2 else []
        if len(recent) >= 2:
            ry = np.array(recent_years, dtype=float)
            rs = np.array(recent, dtype=float)
            # x 값이 모두 동일하면 기울기 0으로 처리
            if np.std(ry) == 0:
                trend_vals.append(0.0)
            else:
                slope, *_ = stats.linregress(ry - ry.mean(), rs)
                trend_vals.append(slope)
        else:
            trend_vals.append(np.nan)

        # 피크
        peak_vals.append(past_scores.max() if len(past_scores) > 0 else np.nan)

        # 1,2 시즌 전
        if j >= 1 and years[j-1] == years[j] - 1:
            lag1_vals.append(scores[j-1])
        else:
            lag1_vals.append(np.nan)

        if j >= 2 and years[j-2] == years[j] - 2:
            lag2_vals.append(scores[j-2])
        else:
            lag2_vals.append(np.nan)

df_sorted_all['perf_trend']  = trend_vals
df_sorted_all['peak_perf']   = peak_vals
df_sorted_all['lag1_perf']   = lag1_vals
df_sorted_all['lag2_perf']   = lag2_vals
df = df_sorted_all.copy()

# ───── Age 상호작용 피처 ─────
df['age_x_perf']     = df['age'] * df['perf_score']           # 나이 × 현재 성과
df['age_x_trend']    = df['age'] * df['perf_trend'].fillna(0) # 나이 × 트렌드
df['age_sq']         = df['age'] ** 2                          # 나이 제곱 (비선형)
df['is_u23']         = (df['age'] <= 23).astype(int)           # U23 지시변수
df['age_peak_gap']   = np.abs(df['age'] - 26)                  # 피크 나이(26)까지 거리

# ───── 포지션 원-핫 인코딩 ─────
for pos in ['FW', 'MF', 'DF', 'GK']:
    df[f'pos_{pos}'] = (df['pos_simple'] == pos).astype(int)

print(f"  피처 엔지니어링 완료")

# ═══════════════════════════════════════════════════════════════
# 6. 최종 피처 선택 및 데이터셋 준비
# ═══════════════════════════════════════════════════════════════

print("\n[6] 피처 선택 및 데이터셋 준비 중...")

FEATURE_COLS = [
    # 기본 선수 정보
    'age', 'age_sq', 'age_peak_gap', 'is_u23',
    # 현재 시즌 성과
    'perf_score', 'gls_p90', 'ast_p90', 'tklw_p90', 'int_p90', 'sh_p90',
    # 출전 관련
    'min', '90s', 'starter_ratio', 'consistency',
    # 시장 가치
    'market_value', 'mv_change_rate',
    # 과거 성과 트렌드
    'perf_trend', 'peak_perf', 'lag1_perf', 'lag2_perf',
    # 컨텍스트
    'points', 'epl_seasons',
    # Age 상호작용
    'age_x_perf', 'age_x_trend',
    # 포지션 원-핫
    'pos_FW', 'pos_MF', 'pos_DF', 'pos_GK',
]

TARGET_COL = 'growth_target'

# NaN이 너무 많은 행 제거 (피처의 50% 이상 NaN)
# age, market_value가 FEATURE_COLS에 이미 포함되어 있으므로 중복 제거
extra_cols = [c for c in [TARGET_COL, 'player', 'season_year', 'pos_simple', 'team']
              if c not in FEATURE_COLS]
df_model = df[FEATURE_COLS + extra_cols].copy()
nan_frac = df_model[FEATURE_COLS].isna().mean(axis=1)
df_model = df_model[nan_frac < 0.5].copy()

print(f"  NaN 처리 후 데이터: {df_model.shape[0]}행")
print(f"  사용 피처 수: {len(FEATURE_COLS)}")

# ───── 시간 기반 분할 ─────
# train: ~2020, val: 2021-2022, test: 2023-2025
# reset_index 하여 인덱스 정합성 확보
df_model = df_model.reset_index(drop=True)

train_mask = df_model['season_year'] <= 2020
val_mask   = (df_model['season_year'] >= 2021) & (df_model['season_year'] <= 2022)
test_mask  = df_model['season_year'] >= 2023

X_train = df_model.loc[train_mask, FEATURE_COLS]
y_train = df_model.loc[train_mask, TARGET_COL]
X_val   = df_model.loc[val_mask,   FEATURE_COLS]
y_val   = df_model.loc[val_mask,   TARGET_COL]
X_test  = df_model.loc[test_mask,  FEATURE_COLS]
y_test  = df_model.loc[test_mask,  TARGET_COL]

print(f"  학습: {len(X_train)}, 검증: {len(X_val)}, 테스트: {len(X_test)}")

# ═══════════════════════════════════════════════════════════════
# 7. 모델 학습 및 평가
# ═══════════════════════════════════════════════════════════════

print("\n[7] 모델 학습 중...")

# 전처리 파이프라인 (결측치 → 중앙값 대치 → 스케일링)
def make_pipeline(model):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model',   model),
    ])

models_config = {
    'XGBoost': xgb.XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    ),
    'RandomForest': RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    ),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    ),
    'MLP': MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        learning_rate_init=0.001,
    ),
}

# XGBoost는 별도 처리 (imputer만 적용, scaler 불필요)
xgb_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model',   models_config['XGBoost']),
])

results = {}
fitted_models = {}

for name, model in models_config.items():
    print(f"  [{name}] 학습 중...", end=' ')
    if name == 'XGBoost':
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model',   model),
        ])
    else:
        pipe = make_pipeline(model)

    pipe.fit(X_train, y_train)

    # 평가
    y_val_pred  = pipe.predict(X_val)
    y_test_pred = pipe.predict(X_test)

    val_rmse  = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae   = mean_absolute_error(y_val, y_val_pred)
    val_r2    = r2_score(y_val, y_val_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae  = mean_absolute_error(y_test, y_test_pred)
    test_r2   = r2_score(y_test, y_test_pred)

    results[name] = {
        'val_rmse': float(val_rmse),
        'val_mae':  float(val_mae),
        'val_r2':   float(val_r2),
        'test_rmse': float(test_rmse),
        'test_mae':  float(test_mae),
        'test_r2':   float(test_r2),
    }
    fitted_models[name] = pipe
    print(f"검증 R²={val_r2:.3f}, 테스트 R²={test_r2:.3f}")

# ───── 최고 모델 선택 (검증 R² 기준) ─────
best_model_name = max(results, key=lambda k: results[k]['val_r2'])
best_model = fitted_models[best_model_name]
print(f"\n  최고 모델: {best_model_name} (검증 R²={results[best_model_name]['val_r2']:.3f})")

# ═══════════════════════════════════════════════════════════════
# 8. 전체 데이터 예측 (스카우트 리포트용)
# ═══════════════════════════════════════════════════════════════

print("\n[8] 전체 데이터 성장 예측 중...")

# 모든 피처 데이터에 예측 적용
all_X = df_model[FEATURE_COLS]
df_model['predicted_growth'] = best_model.predict(all_X)

# 앙상블 예측 (모든 모델 평균)
ensemble_preds = np.column_stack([
    fitted_models[name].predict(all_X) for name in fitted_models
])
df_model['ensemble_growth'] = ensemble_preds.mean(axis=1)

# 메타 정보 재결합
df_predictions = df_model[['player', 'season_year', 'age', 'pos_simple',
                             'team', 'market_value', 'perf_score',
                             'predicted_growth', 'ensemble_growth',
                             TARGET_COL]].copy()

print(f"  예측 완료: {df_predictions.shape[0]}행")

# ═══════════════════════════════════════════════════════════════
# 9. U23 별도 모델 학습
# ═══════════════════════════════════════════════════════════════

print("\n[9] U23 전용 모델 학습 중...")

# 인덱스 정합성 문제 방지를 위해 .values 사용
u23_mask_train = train_mask.values & (df_model['age'] <= 23).values
u23_mask_val   = val_mask.values   & (df_model['age'] <= 23).values
u23_mask_test  = test_mask.values  & (df_model['age'] <= 23).values

X_u23_train = df_model.loc[u23_mask_train, FEATURE_COLS]
y_u23_train = df_model.loc[u23_mask_train, TARGET_COL]
X_u23_val   = df_model.loc[u23_mask_val,   FEATURE_COLS]
y_u23_val   = df_model.loc[u23_mask_val,   TARGET_COL]
X_u23_test  = df_model.loc[u23_mask_test,  FEATURE_COLS]
y_u23_test  = df_model.loc[u23_mask_test,  TARGET_COL]

print(f"  U23 학습: {len(X_u23_train)}, 검증: {len(X_u23_val)}, 테스트: {len(X_u23_test)}")

u23_model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    ))
])
u23_model.fit(X_u23_train, y_u23_train)

u23_results = {}
if len(X_u23_val) > 5:
    y_u23_val_pred  = u23_model.predict(X_u23_val)
    y_u23_test_pred = u23_model.predict(X_u23_test)
    u23_results = {
        'val_r2':   float(r2_score(y_u23_val,  y_u23_val_pred)),
        'test_r2':  float(r2_score(y_u23_test, y_u23_test_pred)),
        'val_rmse': float(np.sqrt(mean_squared_error(y_u23_val, y_u23_val_pred))),
    }
    print(f"  U23 모델: 검증 R²={u23_results['val_r2']:.3f}, 테스트 R²={u23_results['test_r2']:.3f}")

# 최신 시즌(2023/24) U23 예측
latest_u23 = df_model[(df_model['season_year'] >= 2023) & (df_model['age'] <= 23)].copy()
if len(latest_u23) > 0:
    latest_u23['u23_predicted_growth'] = u23_model.predict(latest_u23[FEATURE_COLS])

# ═══════════════════════════════════════════════════════════════
# 10. 스카우트 리포트 생성
# ═══════════════════════════════════════════════════════════════

print("\n[10] 스카우트 리포트 생성 중...")

# 최신 시즌 데이터 (2023/24 또는 2024/25)
latest_year = df_model['season_year'].max()
latest_data = df_model[df_model['season_year'] == latest_year].copy()
latest_data['predicted_growth'] = best_model.predict(latest_data[FEATURE_COLS])

print(f"  최신 시즌 ({latest_year}) 데이터: {len(latest_data)}명")

# ─── Hot Prospects (U23 높은 성장 예측) ───
hot_prospects = (
    latest_data[latest_data['age'] <= 23]
    .nlargest(20, 'predicted_growth')
    [['player', 'age', 'pos_simple', 'team', 'market_value',
      'perf_score', 'predicted_growth']]
    .reset_index(drop=True)
)
hot_prospects.index += 1

# ─── Late Bloomers (24-28, 여전히 성장 중) ───
late_bloomers = (
    latest_data[(latest_data['age'] >= 24) & (latest_data['age'] <= 28)]
    .nlargest(20, 'predicted_growth')
    [['player', 'age', 'pos_simple', 'team', 'market_value',
      'perf_score', 'predicted_growth']]
    .reset_index(drop=True)
)
late_bloomers.index += 1

# ─── Declining Stars (높은 현재 성과 + 하락 예측) ───
declining_stars = (
    latest_data[
        (latest_data['perf_score'] > 0.5) &  # 현재 성과 상위권
        (latest_data['predicted_growth'] < -0.3)
    ]
    .nsmallest(20, 'predicted_growth')
    [['player', 'age', 'pos_simple', 'team', 'market_value',
      'perf_score', 'predicted_growth']]
    .reset_index(drop=True)
)
declining_stars.index += 1

print(f"  Hot Prospects: {len(hot_prospects)}명")
print(f"  Late Bloomers: {len(late_bloomers)}명")
print(f"  Declining Stars: {len(declining_stars)}명")

# ─── 포지션별 피크 나이 분석 ───
peak_age_analysis = {}
for pos in ['FW', 'MF', 'DF']:
    pos_data = df_model[df_model['pos_simple'] == pos].copy()
    if len(pos_data) > 50:
        age_growth = pos_data.groupby('age')['growth_target'].agg(['mean', 'std', 'count'])
        age_growth = age_growth[age_growth['count'] >= 5]
        if len(age_growth) > 0:
            peak_age = int(age_growth['mean'].idxmax())
            peak_age_analysis[pos] = {
                'peak_age': peak_age,
                'avg_growth_at_peak': float(age_growth.loc[peak_age, 'mean']),
                'sample_count': int(age_growth.loc[peak_age, 'count'])
            }

print(f"  포지션별 피크 나이: {peak_age_analysis}")

# ═══════════════════════════════════════════════════════════════
# 11. 피처 중요도 계산
# ═══════════════════════════════════════════════════════════════

print("\n[11] 피처 중요도 계산 중...")

# XGBoost 피처 중요도
xgb_pipe = fitted_models['XGBoost']
xgb_model_obj = xgb_pipe.named_steps['model']
feature_importance = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': xgb_model_obj.feature_importances_
}).sort_values('importance', ascending=False)

# RF 피처 중요도
rf_pipe = fitted_models['RandomForest']
rf_model_obj = rf_pipe.named_steps['model']
rf_importance = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': rf_model_obj.feature_importances_
}).sort_values('importance', ascending=False)

# ═══════════════════════════════════════════════════════════════
# 12. 시각화
# ═══════════════════════════════════════════════════════════════

print("\n[12] 시각화 생성 중...")

# ────── Fig 1: 모델 성능 비교 ──────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('S4: 성장 예측 모델 성능 비교', fontsize=14, fontweight='bold')

model_names = list(results.keys())
val_r2_vals  = [results[m]['val_r2']  for m in model_names]
test_r2_vals = [results[m]['test_r2'] for m in model_names]
val_rmse_vals = [results[m]['val_rmse'] for m in model_names]
test_rmse_vals = [results[m]['test_rmse'] for m in model_names]

x = np.arange(len(model_names))
w = 0.35

ax = axes[0]
bars1 = ax.bar(x - w/2, val_r2_vals,  w, label='Val R²',  color='#3498db', alpha=0.85)
bars2 = ax.bar(x + w/2, test_r2_vals, w, label='Test R²', color='#e74c3c', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=20, ha='right')
ax.set_ylabel('R²'); ax.set_title('R² 점수'); ax.legend()
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
for bar in bars1: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                           f'{bar.get_height():.2f}', ha='center', fontsize=8)
for bar in bars2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                           f'{bar.get_height():.2f}', ha='center', fontsize=8)

ax = axes[1]
bars3 = ax.bar(x - w/2, val_rmse_vals,  w, label='Val RMSE',  color='#2ecc71', alpha=0.85)
bars4 = ax.bar(x + w/2, test_rmse_vals, w, label='Test RMSE', color='#f39c12', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=20, ha='right')
ax.set_ylabel('RMSE'); ax.set_title('RMSE'); ax.legend()

ax = axes[2]
best_pipe = fitted_models[best_model_name]
y_pred_all = best_pipe.predict(df_model.loc[test_mask, FEATURE_COLS])
y_true_all = df_model.loc[test_mask, TARGET_COL]
ax.scatter(y_true_all, y_pred_all, alpha=0.3, s=10, color='#3498db')
lim = max(abs(y_true_all.max()), abs(y_true_all.min()), 3)
ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1.5, label='Perfect')
ax.set_xlabel('실제 성장값'); ax.set_ylabel('예측 성장값')
ax.set_title(f'{best_model_name} 예측 vs 실제 (테스트)')
ax.legend()

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig1_model_performance.png', bbox_inches='tight')
plt.close()
print("  Fig 1 저장: fig1_model_performance.png")

# ────── Fig 2: 피처 중요도 ──────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('S4: 피처 중요도 (XGBoost vs Random Forest)', fontsize=13, fontweight='bold')

top_n = 15
for ax, imp_df, model_label, color in zip(
    axes,
    [feature_importance, rf_importance],
    ['XGBoost', 'Random Forest'],
    ['#e74c3c', '#2ecc71']
):
    top = imp_df.head(top_n)
    ax.barh(top['feature'][::-1], top['importance'][::-1], color=color, alpha=0.85)
    ax.set_title(f'{model_label} 피처 중요도 (상위 {top_n})')
    ax.set_xlabel('중요도')

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig2_feature_importance.png', bbox_inches='tight')
plt.close()
print("  Fig 2 저장: fig2_feature_importance.png")

# ────── Fig 3: 나이-성장 곡선 (포지션별) ──────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('S4: 포지션별 나이-성장 곡선', fontsize=13, fontweight='bold')

for ax, pos in zip(axes, ['FW', 'MF', 'DF']):
    pos_data = df_model[df_model['pos_simple'] == pos].copy()
    age_grouped = pos_data.groupby('age').agg(
        mean_growth=('growth_target', 'mean'),
        mean_pred=('predicted_growth', 'mean'),
        count=('growth_target', 'count')
    ).reset_index()
    age_grouped = age_grouped[age_grouped['count'] >= 5]

    ax.plot(age_grouped['age'], age_grouped['mean_growth'], 'o-',
            color=PALETTE.get(pos, '#999'), label='실제 성장', linewidth=2)
    ax.plot(age_grouped['age'], age_grouped['mean_pred'], 's--',
            color='gray', label='예측 성장', linewidth=1.5, alpha=0.7)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_title(f'{pos} 포지션')
    ax.set_xlabel('나이')
    ax.set_ylabel('평균 성장값 (Z-점수)')
    ax.legend(fontsize=8)

    # 피크 나이 표시
    if pos in peak_age_analysis:
        peak = peak_age_analysis[pos]['peak_age']
        ax.axvline(peak, color='red', linestyle=':', alpha=0.6,
                   label=f'피크: {peak}세')
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig3_age_growth_curves.png', bbox_inches='tight')
plt.close()
print("  Fig 3 저장: fig3_age_growth_curves.png")

# ────── Fig 4: Hot Prospects 랭킹 ──────
fig, ax = plt.subplots(figsize=(12, 7))
if len(hot_prospects) > 0:
    colors = [PALETTE.get(p, '#999') for p in hot_prospects['pos_simple']]
    bars = ax.barh(
        hot_prospects['player'][::-1],
        hot_prospects['predicted_growth'][::-1],
        color=colors[::-1], alpha=0.85
    )
    ax.set_xlabel('예측 성장 점수 (Z-점수 단위)')
    ax.set_title(f'S4: Hot Prospects - U23 최고 성장 유망주 ({latest_year}시즌 기준)')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')

    # 나이 표시
    for i, (_, row) in enumerate(hot_prospects.iloc[::-1].iterrows()):
        ax.text(0.01, i, f" {int(row['age'])}세 | {row['pos_simple']}",
                va='center', fontsize=8, color='white' if row['predicted_growth'] > 0.5 else 'black')

# 범례
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=PALETTE[p], label=p) for p in ['FW', 'MF', 'DF']]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig4_hot_prospects.png', bbox_inches='tight')
plt.close()
print("  Fig 4 저장: fig4_hot_prospects.png")

# ────── Fig 5: 성장 궤적 플롯 (상위 유망주) ──────
fig, ax = plt.subplots(figsize=(13, 7))
top_players = hot_prospects['player'].head(8).tolist()

for player_name in top_players:
    player_data = df_model[df_model['player'] == player_name].sort_values('season_year')
    if len(player_data) >= 2:
        ax.plot(player_data['season_year'], player_data['perf_score'],
                'o-', linewidth=2, label=player_name, alpha=0.85)

ax.set_xlabel('시즌 연도')
ax.set_ylabel('성과 점수 (Z-점수)')
ax.set_title('S4: Hot Prospects 성과 궤적')
ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig5_growth_trajectories.png', bbox_inches='tight')
plt.close()
print("  Fig 5 저장: fig5_growth_trajectories.png")

# ────── Fig 6: Late Bloomers vs Declining Stars ──────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('S4: Late Bloomers vs Declining Stars', fontsize=13, fontweight='bold')

ax = axes[0]
if len(late_bloomers) > 0:
    colors_lb = [PALETTE.get(p, '#999') for p in late_bloomers['pos_simple']]
    ax.barh(late_bloomers['player'][::-1],
            late_bloomers['predicted_growth'][::-1],
            color=colors_lb[::-1], alpha=0.85)
    ax.set_title('Late Bloomers (24-28세)')
    ax.set_xlabel('예측 성장 점수')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')

ax = axes[1]
if len(declining_stars) > 0:
    ax.barh(declining_stars['player'][::-1],
            declining_stars['predicted_growth'][::-1],
            color='#e74c3c', alpha=0.7)
    ax.set_title('Declining Stars (현재 성과 상위 + 하락 예측)')
    ax.set_xlabel('예측 성장 점수')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig6_late_bloomers_declining.png', bbox_inches='tight')
plt.close()
print("  Fig 6 저장: fig6_late_bloomers_declining.png")

# ────── Fig 7: 성장 분포 히스토그램 ──────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('S4: 성장 분포 (포지션별)', fontsize=13, fontweight='bold')

for ax, pos in zip(axes.flat, ['FW', 'MF', 'DF', 'GK']):
    pos_data = df_model[df_model['pos_simple'] == pos]['growth_target'].dropna()
    pred_data = df_model[df_model['pos_simple'] == pos]['predicted_growth'].dropna()
    if len(pos_data) > 10:
        ax.hist(pos_data, bins=30, alpha=0.6, color=PALETTE.get(pos, '#999'),
                label='실제 성장', density=True)
        ax.hist(pred_data, bins=30, alpha=0.4, color='gray',
                label='예측 성장', density=True)
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.set_title(f'{pos} (n={len(pos_data)})')
        ax.set_xlabel('성장값'); ax.set_ylabel('밀도')
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig7_growth_distributions.png', bbox_inches='tight')
plt.close()
print("  Fig 7 저장: fig7_growth_distributions.png")

# ═══════════════════════════════════════════════════════════════
# 13. 결과 저장
# ═══════════════════════════════════════════════════════════════

print("\n[13] 결과 저장 중...")

# ─── growth_predictions.parquet ───
growth_preds_out = df_model[['player', 'season_year', 'age', 'pos_simple',
                              'team', 'market_value', 'perf_score',
                              'predicted_growth', 'ensemble_growth',
                              TARGET_COL]].copy()
growth_preds_out.to_parquet(SCOUT_DIR / 'growth_predictions.parquet', index=False)
print(f"  growth_predictions.parquet 저장: {growth_preds_out.shape}")

# ─── results_summary.json ───
summary = {
    'model_performance': results,
    'best_model': best_model_name,
    'u23_model_performance': u23_results,
    'feature_count': len(FEATURE_COLS),
    'features_used': FEATURE_COLS,
    'data_split': {
        'train': int(train_mask.sum()),
        'val':   int(val_mask.sum()),
        'test':  int(test_mask.sum()),
    },
    'peak_age_by_position': peak_age_analysis,
    'scout_lists': {
        'hot_prospects': hot_prospects.to_dict('records'),
        'late_bloomers': late_bloomers.to_dict('records'),
        'declining_stars': declining_stars.to_dict('records'),
    },
    'top_features_xgb': feature_importance.head(10).to_dict('records'),
    'top_features_rf':  rf_importance.head(10).to_dict('records'),
    'growth_stats': {
        'mean': float(df_model['growth_target'].mean()),
        'std':  float(df_model['growth_target'].std()),
        'pct_positive': float((df_model['growth_target'] > 0).mean()),
    }
}

with open(SCOUT_DIR / 'results_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("  results_summary.json 저장 완료")

# ═══════════════════════════════════════════════════════════════
# 14. 최종 결과 출력
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("S4 학습 완료 요약")
print("=" * 60)
print(f"\n[모델 성능] (최고: {best_model_name})")
for name, res in results.items():
    star = " ★" if name == best_model_name else ""
    print(f"  {name+star:<22}: Val R²={res['val_r2']:.3f} | Test R²={res['test_r2']:.3f} | "
          f"Test RMSE={res['test_rmse']:.3f}")

print(f"\n[포지션별 피크 나이]")
for pos, info in peak_age_analysis.items():
    print(f"  {pos}: {info['peak_age']}세 (평균 성장={info['avg_growth_at_peak']:.3f})")

print(f"\n[Hot Prospects Top 5 (U23)]")
for _, row in hot_prospects.head(5).iterrows():
    print(f"  {row['player']:<25} | {int(row['age'])}세 | {row['pos_simple']} | "
          f"예측성장={row['predicted_growth']:.3f}")

print(f"\n[Late Bloomers Top 5 (24-28)]")
for _, row in late_bloomers.head(5).iterrows():
    print(f"  {row['player']:<25} | {int(row['age'])}세 | {row['pos_simple']} | "
          f"예측성장={row['predicted_growth']:.3f}")

print(f"\n[Declining Stars Top 5]")
for _, row in declining_stars.head(5).iterrows():
    print(f"  {row['player']:<25} | {int(row['age'])}세 | 현재점수={row['perf_score']:.3f} | "
          f"예측성장={row['predicted_growth']:.3f}")

print(f"\n[저장 파일]")
print(f"  {SCOUT_DIR}/results_summary.json")
print(f"  {SCOUT_DIR}/growth_predictions.parquet")
for i in range(1, 8):
    print(f"  {FIG_DIR}/fig{i}_*.png")

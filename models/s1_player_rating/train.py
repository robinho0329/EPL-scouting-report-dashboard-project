"""
S1: Scout Player Rating System
WAR 스타일 복합 선수 평가 시스템 (0-100 스케일)
- 포지션별 가중치 기반 평점 계산
- 예측 모델 학습 (Ridge, XGBoost, MLP)
- 히든 젬 선수 발굴
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서 렌더링
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import joblib

from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE_DIR = r"C:/Users/xcv54/workspace/EPL project"
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models", "s1_player_rating")
SCOUT_DIR = os.path.join(BASE_DIR, "data", "scout")
FIG_DIR = os.path.join(MODEL_DIR, "figures")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCOUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# 한글 폰트 설정 (Windows 환경)
try:
    font_path = "C:/Windows/Fonts/malgun.ttf"
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

print("=" * 60)
print("S1: Scout Player Rating System 시작")
print("=" * 60)


# ─────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────
print("\n[1] 데이터 로드 중...")

season_stats = pd.read_parquet(os.path.join(DATA_DIR, "player_season_stats.parquet"))
match_logs = pd.read_parquet(os.path.join(DATA_DIR, "player_match_logs.parquet"))
team_summary = pd.read_parquet(os.path.join(DATA_DIR, "team_season_summary.parquet"))

print(f"  player_season_stats: {season_stats.shape}")
print(f"  player_match_logs:   {match_logs.shape}")
print(f"  team_season_summary: {team_summary.shape}")


# ─────────────────────────────────────────────
# 2. 매치 로그에서 시즌별 고급 스탯 집계
# ─────────────────────────────────────────────
print("\n[2] 매치 로그에서 시즌별 스탯 집계 중...")

# detail_stats_available == True인 행만 고급 스탯 사용 (2014/15 이후)
ml_detail = match_logs[match_logs['detail_stats_available'] == True].copy()

# 선수별 시즌별 집계 (고급 스탯)
agg_adv = ml_detail.groupby(['player', 'season', 'team']).agg(
    sh_sum=('sh', 'sum'),
    sot_sum=('sot', 'sum'),
    tklw_sum=('tklw', 'sum'),
    int_sum=('int', 'sum'),
    match_count_detail=('min', 'count'),
).reset_index()

# 전체 매치 로그 집계 (기본 스탯 - 모든 시즌)
agg_basic = match_logs.groupby(['player', 'season', 'team']).agg(
    total_min=('min', 'sum'),
    match_count=('min', 'count'),
    gls_sum=('gls', 'sum'),
    ast_sum=('ast', 'sum'),
    clean_sheet_count=('result', lambda x: (x == 'W').sum()),  # GK 클린시트 근사
).reset_index()

# GK 세이브 수 집계 (match_logs에 saves 컬럼 없으므로 결과 기반 근사)
# sot_against를 세이브로 사용할 수 없어 별도 처리

# 기본 + 고급 스탯 병합
agg_all = agg_basic.merge(agg_adv, on=['player', 'season', 'team'], how='left')
print(f"  집계된 시즌 레코드: {agg_all.shape[0]}")


# ─────────────────────────────────────────────
# 3. 시즌 스탯과 매치 집계 병합
# ─────────────────────────────────────────────
print("\n[3] 데이터 병합 및 피처 엔지니어링 중...")

# season_stats에서 주요 컬럼 선택
ss_cols = ['player', 'season', 'team', 'pos', 'age', 'min', '90s',
           'gls', 'ast', 'gls_1', 'ast_1',  # per-90 미리 계산된 값
           'market_value', 'position', 'birth_year', 'height_cm']
ss = season_stats[ss_cols].copy()

# team_summary에서 팀 강도(포인트) 추가
ts_cols = ['Season', 'team', 'points', 'goal_diff']
ts = team_summary[ts_cols].rename(columns={'Season': 'season'})

df = ss.merge(ts, on=['season', 'team'], how='left')
df = df.merge(agg_all, on=['player', 'season', 'team'], how='left')

print(f"  병합 후 데이터: {df.shape}")


# ─────────────────────────────────────────────
# 4. 포지션 그룹 분류
# ─────────────────────────────────────────────
print("\n[4] 포지션 그룹 분류 중...")

def classify_position_group(row):
    """포지션 문자열을 FW/MID/DEF/GK 4개 그룹으로 분류"""
    pos_fbref = str(row.get('pos', '')).upper()
    pos_tm = str(row.get('position', '')).lower()

    # transfermarkt 포지션 우선 활용
    if 'goalkeeper' in pos_tm or 'gk' in pos_fbref:
        return 'GK'
    if any(x in pos_tm for x in ['centre-back', 'left-back', 'right-back', 'defender']):
        return 'DEF'
    if any(x in pos_tm for x in ['defensive midfield', 'central midfield', 'right midfield',
                                   'left midfield', 'midfielder']):
        return 'MID'
    if any(x in pos_tm for x in ['attacking midfield', 'second striker', 'left winger',
                                   'right winger', 'centre-forward', 'striker']):
        return 'FW'

    # fbref pos 폴백
    if 'GK' in pos_fbref:
        return 'GK'
    if 'DF' in pos_fbref:
        return 'DEF'
    if 'MF' in pos_fbref:
        return 'MID'
    if 'FW' in pos_fbref:
        return 'FW'

    return 'MID'  # 기본값

df['pos_group'] = df.apply(classify_position_group, axis=1)
print(f"  포지션 분포:\n{df['pos_group'].value_counts()}")


# ─────────────────────────────────────────────
# 5. Per-90 스탯 계산
# ─────────────────────────────────────────────
print("\n[5] Per-90 스탯 계산 중...")

# 90s (90분 단위) 컬럼이 있으면 사용, 없으면 min에서 계산
df['nineties'] = df['90s'].fillna(df['min'] / 90.0)
df['nineties'] = df['nineties'].replace(0, np.nan)

# 최소 출전 시간 필터 (90분 이상 = 1경기)
df = df[df['min'].fillna(0) >= 90].copy()

def per90(col, nineties):
    """Per-90 계산 (0으로 나누기 방지)"""
    return col.fillna(0) / nineties.clip(lower=0.1)

df['gls_p90'] = per90(df['gls_sum'].fillna(df['gls']), df['nineties'])
df['ast_p90'] = per90(df['ast_sum'].fillna(df['ast']), df['nineties'])
df['sh_p90'] = per90(df['sh_sum'], df['nineties'])
df['tklw_p90'] = per90(df['tklw_sum'], df['nineties'])
df['int_p90'] = per90(df['int_sum'], df['nineties'])
# clearances: match_logs에 없으므로 tackles + interceptions 합산으로 근사
df['clr_p90'] = per90(
    df['tklw_sum'].fillna(0) + df['int_sum'].fillna(0),
    df['nineties']
)
# 키 패스: sot를 key_pass 근사로 활용 (MID 평가)
df['kp_p90'] = per90(df['sot_sum'].fillna(df['ast_sum'].fillna(0)), df['nineties'])
# GK: 결과 기반 클린시트 (무실점 경기)
# match_logs에서 GK 클린시트 = 결과가 W 또는 D이고 상대 득점 0인 경기
gk_clean = match_logs[match_logs['goals_against'] == 0].groupby(
    ['player', 'season', 'team']
)['min'].count().reset_index().rename(columns={'min': 'clean_sheets'})
df = df.merge(gk_clean, on=['player', 'season', 'team'], how='left')
df['clean_sheets'] = df['clean_sheets'].fillna(0)
df['cs_p90'] = per90(df['clean_sheets'], df['nineties'])
# saves 근사: sot_allowed 정보 없으므로 클린시트 비율 활용
df['save_pct'] = (df['clean_sheets'] / df['match_count'].clip(lower=1)).clip(0, 1)

print(f"  Per-90 스탯 계산 완료. 유효 레코드: {df.shape[0]}")


# ─────────────────────────────────────────────
# 6. Z-score 정규화 (포지션 그룹 + 시즌별)
# ─────────────────────────────────────────────
print("\n[6] 포지션-시즌 그룹 내 Z-score 정규화 중...")

STAT_COLS = ['gls_p90', 'ast_p90', 'sh_p90', 'tklw_p90', 'int_p90',
             'clr_p90', 'kp_p90', 'cs_p90', 'save_pct']

def zscore_within_group(series):
    """그룹 내 Z-score 정규화 (분산=0인 경우 0 반환)"""
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std

for col in STAT_COLS:
    z_col = col + '_z'
    df[z_col] = df.groupby(['pos_group', 'season'])[col].transform(zscore_within_group)
    df[z_col] = df[z_col].fillna(0).clip(-3, 3)  # ±3 클리핑으로 이상치 제어

print("  Z-score 정규화 완료.")


# ─────────────────────────────────────────────
# 7. WAR 스타일 복합 평점 계산
# ─────────────────────────────────────────────
print("\n[7] WAR 스타일 복합 평점 계산 중...")

# 포지션별 가중치 정의
# minutes는 출전 비율로 정규화하여 반영
WEIGHTS = {
    'FW':  {'gls_p90': 0.40, 'ast_p90': 0.20, 'sh_p90': 0.15,   'min_ratio': 0.25},
    'MID': {'ast_p90': 0.30, 'gls_p90': 0.25, 'kp_p90': 0.20,   'min_ratio': 0.25},
    'DEF': {'tklw_p90': 0.25, 'int_p90': 0.25, 'clr_p90': 0.20, 'min_ratio': 0.30},
    'GK':  {'save_pct': 0.35, 'cs_p90': 0.35,                   'min_ratio': 0.30},
}

# 시즌별 최대 출전 가능 90s (38경기 × 90분 기준)
MAX_NINETIES = 38.0

def compute_raw_score(row):
    """포지션별 가중 합산 점수 계산"""
    pg = row['pos_group']
    w = WEIGHTS.get(pg, WEIGHTS['MID'])

    # 출전 시간 비율 Z-score (같은 포지션-시즌 내에서 상대 평가)
    min_ratio_z = row.get('min_ratio_z', 0)

    score = 0.0
    if pg == 'FW':
        score = (w['gls_p90'] * row['gls_p90_z']
                 + w['ast_p90'] * row['ast_p90_z']
                 + w['sh_p90'] * row['sh_p90_z']
                 + w['min_ratio'] * min_ratio_z)
    elif pg == 'MID':
        score = (w['ast_p90'] * row['ast_p90_z']
                 + w['gls_p90'] * row['gls_p90_z']
                 + w['kp_p90'] * row['kp_p90_z']
                 + w['min_ratio'] * min_ratio_z)
    elif pg == 'DEF':
        score = (w['tklw_p90'] * row['tklw_p90_z']
                 + w['int_p90'] * row['int_p90_z']
                 + w['clr_p90'] * row['clr_p90_z']
                 + w['min_ratio'] * min_ratio_z)
    elif pg == 'GK':
        score = (w['save_pct'] * row['save_pct_z']
                 + w['cs_p90'] * row['cs_p90_z']
                 + w['min_ratio'] * min_ratio_z)
    return score

# 출전 비율 Z-score 추가 (시즌별 최대 90s 대비)
df['min_ratio'] = df['nineties'] / MAX_NINETIES
df['min_ratio_z'] = df.groupby(['pos_group', 'season'])['min_ratio'].transform(zscore_within_group)
df['min_ratio_z'] = df['min_ratio_z'].fillna(0).clip(-3, 3)

# 원시 점수 계산
df['raw_score'] = df.apply(compute_raw_score, axis=1)

# 시즌-포지션 그룹 내에서 min-max → 0~100 스케일
def minmax_scale_group(series):
    """그룹 내 min-max 정규화"""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50.0, index=series.index)
    return (series - mn) / (mx - mn) * 100

df['war_rating'] = df.groupby(['pos_group', 'season'])['raw_score'].transform(minmax_scale_group)
df['war_rating'] = df['war_rating'].round(2)

print(f"  평점 통계:\n{df['war_rating'].describe()}")
print(f"\n  상위 5명:\n{df.nlargest(5, 'war_rating')[['player', 'season', 'team', 'pos_group', 'war_rating']]}")


# ─────────────────────────────────────────────
# 8. 예측 모델 학습 (Ridge, XGBoost, MLP)
# ─────────────────────────────────────────────
print("\n[8] 예측 모델 학습 중...")

# 시즌 연도 추출 (예: '2020/21' → 2020)
def season_to_year(s):
    try:
        return int(str(s).split('/')[0])
    except Exception:
        return np.nan

df['season_year'] = df['season'].apply(season_to_year)

# 팀 강도 피처 (points, goal_diff)
# 포지션 그룹 인코딩
pos_encoder = LabelEncoder()
df['pos_group_enc'] = pos_encoder.fit_transform(df['pos_group'].fillna('MID'))

# 피처 선택
FEATURES = [
    'age',           # 나이
    'market_value',  # 시장 가치
    'pos_group_enc', # 포지션 그룹
    'points',        # 팀 강도 (시즌 포인트)
    'goal_diff',     # 팀 골 득실 차
    'nineties',      # 출전 90분 단위
    'match_count',   # 경기 수
    'season_year',   # 시즌 연도
]

TARGET = 'war_rating'

# NaN 처리 (중앙값 대체)
all_model_cols = list(dict.fromkeys(FEATURES + [TARGET, 'season_year', 'player', 'season',
                                                 'team', 'pos_group', 'market_value']))
df_model = df[all_model_cols].copy()
df_model = df_model.reset_index(drop=True)

for col in FEATURES:
    median_val = df_model[col].median()
    df_model[col] = df_model[col].fillna(median_val)

df_model = df_model.dropna(subset=[TARGET]).reset_index(drop=True)

# 시간 기반 분할: train <2021, val 2021-2022, test 2023-2025
train_idx = df_model.index[df_model['season_year'] < 2021]
val_idx = df_model.index[df_model['season_year'].isin([2021, 2022])]
test_idx = df_model.index[df_model['season_year'] >= 2023]

train_mask = df_model['season_year'] < 2021
val_mask = df_model['season_year'].isin([2021, 2022])
test_mask = df_model['season_year'] >= 2023

X_train = df_model.loc[train_idx, FEATURES]
y_train = df_model.loc[train_idx, TARGET]
X_val = df_model.loc[val_idx, FEATURES]
y_val = df_model.loc[val_idx, TARGET]
X_test = df_model.loc[test_idx, FEATURES]
y_test = df_model.loc[test_idx, TARGET]

print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

def evaluate(name, y_true, y_pred):
    """모델 평가 메트릭 계산"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"  [{name}] RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}")
    return {'rmse': round(rmse, 4), 'mae': round(mae, 4), 'r2': round(r2, 4)}

metrics = {}

# ── Ridge 회귀 ──
print("\n  Ridge 회귀 학습 중...")
ridge_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=10.0))
])
ridge_pipe.fit(X_train, y_train)
# 검증셋 기반 알파 미세조정 생략 (단순성 유지)
val_pred_ridge = ridge_pipe.predict(X_val)
test_pred_ridge = ridge_pipe.predict(X_test)
metrics['Ridge'] = {
    'val': evaluate('Ridge Val', y_val, val_pred_ridge),
    'test': evaluate('Ridge Test', y_test, test_pred_ridge)
}
joblib.dump(ridge_pipe, os.path.join(MODEL_DIR, "ridge_model.pkl"))
print("  Ridge 모델 저장 완료.")

# ── XGBoost ──
print("\n  XGBoost 학습 중...")
xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)
val_pred_xgb = xgb_model.predict(X_val)
test_pred_xgb = xgb_model.predict(X_test)
metrics['XGBoost'] = {
    'val': evaluate('XGBoost Val', y_val, val_pred_xgb),
    'test': evaluate('XGBoost Test', y_test, test_pred_xgb)
}
joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgb_model.pkl"))
print("  XGBoost 모델 저장 완료.")

# ── MLP 신경망 ──
print("\n  MLP 신경망 학습 중...")
mlp_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        learning_rate_init=0.001
    ))
])
mlp_pipe.fit(X_train, y_train)
val_pred_mlp = mlp_pipe.predict(X_val)
test_pred_mlp = mlp_pipe.predict(X_test)
metrics['MLP'] = {
    'val': evaluate('MLP Val', y_val, val_pred_mlp),
    'test': evaluate('MLP Test', y_test, test_pred_mlp)
}
joblib.dump(mlp_pipe, os.path.join(MODEL_DIR, "mlp_model.pkl"))
print("  MLP 모델 저장 완료.")

# 앙상블 예측 (단순 평균)
test_pred_ensemble = (test_pred_ridge + test_pred_xgb + test_pred_mlp) / 3
metrics['Ensemble'] = {
    'test': evaluate('Ensemble Test', y_test, test_pred_ensemble)
}

# LabelEncoder 저장
joblib.dump(pos_encoder, os.path.join(MODEL_DIR, "pos_encoder.pkl"))
print("\n  모든 모델 저장 완료.")


# ─────────────────────────────────────────────
# 9. 결과 정리 및 저장
# ─────────────────────────────────────────────
print("\n[9] 결과 저장 중...")

# 테스트 셋 (2023-2025) 상위 20명 - 시즌별
test_df = df_model[test_mask].copy()
test_df['xgb_pred'] = test_pred_xgb

# 시즌별 상위 20명
top20_by_season = {}
for season in sorted(test_df['season'].unique()):
    s_df = test_df[test_df['season'] == season].nlargest(20, TARGET)
    top20_by_season[season] = s_df[['player', 'team', 'pos_group', TARGET]].to_dict('records')

# 포지션별 상위 10명 (전체 테스트셋)
top_by_pos = {}
for pos in ['FW', 'MID', 'DEF', 'GK']:
    p_df = test_df[test_df['pos_group'] == pos].nlargest(10, TARGET)
    top_by_pos[pos] = p_df[['player', 'season', 'team', TARGET]].to_dict('records')

results_summary = {
    'model_metrics': metrics,
    'top20_by_season': top20_by_season,
    'top_by_position': top_by_pos,
    'features_used': FEATURES,
    'train_size': int(X_train.shape[0]),
    'val_size': int(X_val.shape[0]),
    'test_size': int(X_test.shape[0]),
    'total_players_rated': int(df.shape[0]),
}

with open(os.path.join(MODEL_DIR, "results_summary.json"), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2, default=str)
print("  results_summary.json 저장 완료.")

# scout_ratings.parquet 저장
scout_cols = ['player', 'season', 'team', 'pos_group', 'age', 'min',
              'nineties', 'match_count', 'market_value', 'war_rating',
              'gls_p90', 'ast_p90', 'sh_p90', 'tklw_p90', 'int_p90', 'kp_p90']
scout_df = df[[c for c in scout_cols if c in df.columns]].copy()
scout_df = scout_df.sort_values(['season', 'war_rating'], ascending=[True, False])
scout_df.to_parquet(os.path.join(SCOUT_DIR, "scout_ratings.parquet"), index=False)
print("  scout_ratings.parquet 저장 완료.")


# ─────────────────────────────────────────────
# 10. 시각화
# ─────────────────────────────────────────────
print("\n[10] 시각화 생성 중...")

# ── 그림 1: 전체 평점 분포 (포지션별) ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('WAR Player Rating Distribution by Position', fontsize=16, fontweight='bold')

colors = {'FW': '#e74c3c', 'MID': '#3498db', 'DEF': '#2ecc71', 'GK': '#f39c12'}
for idx, pos in enumerate(['FW', 'MID', 'DEF', 'GK']):
    ax = axes[idx // 2][idx % 2]
    pos_data = df[df['pos_group'] == pos]['war_rating'].dropna()
    ax.hist(pos_data, bins=40, color=colors[pos], alpha=0.75, edgecolor='white', linewidth=0.5)
    ax.axvline(pos_data.mean(), color='black', linestyle='--', linewidth=1.5,
               label=f'Mean: {pos_data.mean():.1f}')
    ax.set_title(f'{pos} (n={len(pos_data):,})', fontsize=13, fontweight='bold')
    ax.set_xlabel('WAR Rating (0-100)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "rating_distribution.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  rating_distribution.png 저장 완료.")

# ── 그림 2: 테스트셋 시즌별 상위 선수 바 차트 ──
for season in sorted(test_df['season'].unique()):
    fig, ax = plt.subplots(figsize=(12, 7))
    s_df = test_df[test_df['season'] == season].nlargest(15, TARGET).sort_values(TARGET)
    bar_colors = [colors.get(p, '#95a5a6') for p in s_df['pos_group']]
    bars = ax.barh(s_df['player'] + ' (' + s_df['team'] + ')',
                   s_df[TARGET], color=bar_colors, alpha=0.85, edgecolor='white')
    ax.set_xlabel('WAR Rating (0-100)', fontsize=12)
    ax.set_title(f'Top 15 Players - {season}', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 105)
    for bar, val in zip(bars, s_df[TARGET]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}', va='center', fontsize=9)
    # 범례
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=v, label=k) for k, v in colors.items()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fname = f"top_players_{season.replace('/', '_')}.png"
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()

print(f"  top_players 바 차트 저장 완료 ({len(test_df['season'].unique())} 시즌).")

# ── 그림 3: 평점 vs 시장 가치 산점도 (테스트셋) ──
fig, ax = plt.subplots(figsize=(12, 8))
plot_df = test_df[test_df['market_value'].notna() & (test_df['market_value'] > 0)].copy()
plot_df['mv_log'] = np.log10(plot_df['market_value'])

for pos, color in colors.items():
    mask = plot_df['pos_group'] == pos
    ax.scatter(plot_df.loc[mask, 'mv_log'],
               plot_df.loc[mask, TARGET],
               c=color, alpha=0.4, s=20, label=pos)

# 추세선
if len(plot_df) > 10:
    z = np.polyfit(plot_df['mv_log'].dropna(), plot_df.loc[plot_df['mv_log'].notna(), TARGET], 1)
    p = np.poly1d(z)
    x_line = np.linspace(plot_df['mv_log'].min(), plot_df['mv_log'].max(), 100)
    ax.plot(x_line, p(x_line), 'k--', linewidth=2, alpha=0.7, label='Trend')

ax.set_xlabel('Market Value (log10, €)', fontsize=12)
ax.set_ylabel('WAR Rating (0-100)', fontsize=12)
ax.set_title('WAR Rating vs Market Value (Test Set 2023-2025)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "rating_vs_market_value.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  rating_vs_market_value.png 저장 완료.")


# ─────────────────────────────────────────────
# 11. 히든 젬 발굴 (Hidden Gems)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("HIDDEN GEMS - 고평점 저시장가치 선수 발굴")
print("=" * 60)

# 기준: 시장 가치 데이터가 있는 선수만 대상
# 테스트셋 (2023-2025) 기준
gems_df = test_df[
    test_df['market_value'].notna() &
    (test_df['market_value'] > 0)
].copy()

if len(gems_df) > 0:
    # 평점과 시장가치를 각각 정규화
    gems_df['rating_norm'] = (gems_df[TARGET] - gems_df[TARGET].min()) / (
        gems_df[TARGET].max() - gems_df[TARGET].min() + 1e-9)
    gems_df['mv_norm'] = (gems_df['market_value'] - gems_df['market_value'].min()) / (
        gems_df['market_value'].max() - gems_df['market_value'].min() + 1e-9)

    # 히든 젬 점수: 높은 평점 + 낮은 시장 가치 (하위 40% 시장가치)
    mv_40pct = gems_df['market_value'].quantile(0.40)
    rating_60pct = gems_df[TARGET].quantile(0.60)

    hidden_gems = gems_df[
        (gems_df['market_value'] <= mv_40pct) &
        (gems_df[TARGET] >= rating_60pct)
    ].copy()

    # 젬 점수 = 평점 / log(시장가치+1)
    hidden_gems['gem_score'] = hidden_gems[TARGET] / np.log10(
        hidden_gems['market_value'].clip(lower=1) + 1)
    hidden_gems = hidden_gems.nlargest(20, 'gem_score')

    print(f"\n  기준: 시장가치 하위 40% (≤ €{mv_40pct/1e6:.1f}M) + 평점 상위 40%")
    print(f"  발굴된 히든 젬: {len(hidden_gems)}명\n")

    display_cols = ['player', 'season', 'team', 'pos_group', TARGET, 'market_value', 'gem_score']
    display_df = hidden_gems[display_cols].copy()
    display_df['market_value'] = (display_df['market_value'] / 1e6).round(2).astype(str) + 'M€'
    display_df[TARGET] = display_df[TARGET].round(2)
    display_df['gem_score'] = display_df['gem_score'].round(3)
    display_df.columns = ['선수', '시즌', '팀', '포지션', 'WAR평점', '시장가치', '젬점수']

    print(display_df.to_string(index=False))

    # 히든 젬 저장
    hidden_gems.to_parquet(os.path.join(SCOUT_DIR, "hidden_gems.parquet"), index=False)
    print(f"\n  hidden_gems.parquet 저장 완료.")
else:
    print("  시장가치 데이터 없음 - 히든 젬 분석 불가.")


# ─────────────────────────────────────────────
# 12. 피처 중요도 출력 (XGBoost)
# ─────────────────────────────────────────────
print("\n[12] XGBoost 피처 중요도:")
importance_df = pd.DataFrame({
    'feature': FEATURES,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance_df.to_string(index=False))


# ─────────────────────────────────────────────
# 최종 요약
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("S1 완료 요약")
print("=" * 60)
print(f"  총 평가 선수-시즌: {df.shape[0]:,}")
print(f"  모델 저장 위치: {MODEL_DIR}")
print(f"  평점 데이터: {os.path.join(SCOUT_DIR, 'scout_ratings.parquet')}")
print(f"  결과 요약: {os.path.join(MODEL_DIR, 'results_summary.json')}")
print(f"  시각화: {FIG_DIR}")
print(f"\n  모델 성능 (Test set):")
for model_name, m in metrics.items():
    if 'test' in m:
        t = m['test']
        print(f"    {model_name:10s}: RMSE={t['rmse']:.3f}, MAE={t['mae']:.3f}, R²={t['r2']:.3f}")
print("=" * 60)

"""
S6: 선수 하락세 / 부상 위험 감지 모델 (Scout용)
Player Decline & Injury Risk Detection for EPL Scouts
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
from scipy import stats

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE_DIR = Path("C:/Users/xcv54/workspace/EPL project")
DATA_DIR = BASE_DIR / "data" / "processed"
SCOUT_DIR = BASE_DIR / "data" / "scout"
MODEL_DIR = BASE_DIR / "models" / "s6_decline"
FIG_DIR = MODEL_DIR / "figures"

for d in [SCOUT_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────
print("=" * 60)
print("[S6] 선수 하락세 감지 모델 학습 시작")
print("=" * 60)

print("\n[1] 데이터 로드 중...")
season_df = pd.read_parquet(DATA_DIR / "player_season_stats.parquet")
match_df = pd.read_parquet(DATA_DIR / "player_match_logs.parquet")
team_df = pd.read_parquet(DATA_DIR / "team_season_summary.parquet")

print(f"  선수 시즌 통계: {season_df.shape}")
print(f"  경기 로그: {match_df.shape}")
print(f"  팀 시즌 요약: {team_df.shape}")

# ─────────────────────────────────────────────
# 2. 시즌 연도 파싱 (예: '2020/21' → 2020)
# ─────────────────────────────────────────────
def parse_season_year(s):
    """시즌 문자열에서 시작 연도 추출 ('2020/21' → 2020)"""
    try:
        return int(str(s).split('/')[0])
    except:
        return np.nan

season_df['season_year'] = season_df['season'].apply(parse_season_year)
team_df['season_year'] = team_df['Season'].apply(parse_season_year)

# ─────────────────────────────────────────────
# 3. 포지션 그룹화 (수비수 / 미드필더 / 공격수 / GK)
# ─────────────────────────────────────────────
def map_position_group(pos):
    """세부 포지션 → 포지션 그룹 매핑"""
    if pd.isna(pos):
        return 'Unknown'
    pos = str(pos).lower()
    if 'goalkeeper' in pos or 'gk' in pos:
        return 'GK'
    elif 'back' in pos or 'defend' in pos or pos in ['cb', 'rb', 'lb', 'rwb', 'lwb']:
        return 'DEF'
    elif 'midfield' in pos or pos in ['cm', 'cdm', 'cam', 'dm', 'am']:
        return 'MID'
    elif 'forward' in pos or 'winger' in pos or 'striker' in pos or pos in ['cf', 'st', 'lw', 'rw', 'ss']:
        return 'FWD'
    else:
        return 'MID'  # 기본값

season_df['pos_group'] = season_df['position'].apply(map_position_group)

# ─────────────────────────────────────────────
# 4. 포지션별 복합 성과 지수 계산 (per-90 기반 Z-score)
# ─────────────────────────────────────────────
print("\n[2] 포지션별 복합 성과 지수 계산 중...")

# per-90 기준 안전한 분모 처리
season_df['90s_safe'] = season_df['90s'].clip(lower=0.1)

# 포지션별 핵심 지표 정의
POS_METRICS = {
    'FWD': ['gls_1', 'ast_1', 'g_a_1'],      # 골, 어시스트 (per-90)
    'MID': ['ast_1', 'gls_1', 'g_a_1'],       # 어시스트, 골 기여
    'DEF': ['gls_1', 'ast_1'],                  # 수비수는 단순 지표 사용
    'GK':  ['gls_1', 'ast_1'],                  # GK는 실점 등 없으므로 단순 사용
    'Unknown': ['gls_1', 'ast_1'],
}

def compute_composite_score(df):
    """포지션별 Z-score 기반 복합 성과 지수 계산"""
    scores = np.zeros(len(df))
    for pos_g, metrics in POS_METRICS.items():
        mask = df['pos_group'] == pos_g
        if mask.sum() == 0:
            continue
        sub = df.loc[mask, metrics].copy()
        # Z-score 정규화 (분산 0 방지)
        z = sub.apply(lambda col: stats.zscore(col.fillna(0), nan_policy='omit'), axis=0)
        scores[mask.values] = z.mean(axis=1).values
    return scores

season_df['perf_score'] = compute_composite_score(season_df)

# ─────────────────────────────────────────────
# 5. 팀 품질 지수 (포인트 기반 정규화)
# ─────────────────────────────────────────────
team_quality = team_df.groupby(['team', 'season_year'])['points'].sum().reset_index()
team_quality.rename(columns={'points': 'team_quality'}, inplace=True)
# 시즌별 정규화 (0~1)
team_quality['team_quality'] = team_quality.groupby('season_year')['team_quality'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
)

season_df = season_df.merge(
    team_quality, left_on=['team', 'season_year'], right_on=['team', 'season_year'], how='left'
)

# ─────────────────────────────────────────────
# 6. 경기 일관성 점수 계산 (match_logs에서 추출)
# ─────────────────────────────────────────────
print("[3] 경기별 일관성 지표 계산 중...")

match_df['season_year'] = match_df['season'].apply(parse_season_year)

# 경기당 G+A 계산
match_df['g_a_match'] = match_df['gls'].fillna(0) + match_df['ast'].fillna(0)

# 선수별 시즌별 표준편차 (일관성의 역수)
consistency = match_df.groupby(['player', 'season_year']).agg(
    consistency_std=('g_a_match', 'std'),     # 낮을수록 일관성 높음
    match_count=('g_a_match', 'count'),
    avg_min_match=('min', 'mean'),             # 경기당 평균 출전 시간
    total_min_log=('min', 'sum'),              # 로그 기준 총 출전 시간
).reset_index()
consistency['consistency_score'] = 1.0 / (1.0 + consistency['consistency_std'].fillna(0))

season_df = season_df.merge(
    consistency[['player', 'season_year', 'consistency_score', 'match_count',
                  'avg_min_match', 'total_min_log']],
    on=['player', 'season_year'], how='left'
)

# ─────────────────────────────────────────────
# 7. 다음 시즌 성과 병합 → 하락 레이블 생성
# ─────────────────────────────────────────────
print("[4] 시즌 N → N+1 하락 레이블 생성 중...")

# player_id가 없는 경우 player 이름 + team 조합으로 식별
season_df['player_key'] = season_df['player']

# 현재 시즌과 다음 시즌 병합
current = season_df[['player_key', 'season_year', 'perf_score', 'min', 'mp',
                       'market_value', 'age', 'pos_group', 'team', 'team_quality']].copy()
next_season = current[['player_key', 'season_year', 'perf_score', 'min', 'mp', 'market_value']].copy()
next_season.columns = ['player_key', 'next_year', 'next_perf_score', 'next_min',
                        'next_mp', 'next_market_value']
next_season['season_year'] = next_season['next_year'] - 1

merged = current.merge(next_season, on=['player_key', 'season_year'], how='inner')

# 성과 하락 레이블: 다음 시즌 성과가 0.5 표준편차 이상 하락
perf_diff = merged['next_perf_score'] - merged['perf_score']
perf_std = perf_diff.std()
merged['decline_perf'] = (perf_diff < -0.5 * perf_std).astype(int)

# 가용성 하락 레이블: 출전 시간 30% 이상 감소 (부상/form 하락)
merged['decline_avail'] = (
    (merged['next_min'] < merged['min'] * 0.7) & (merged['min'] >= 450)
).astype(int)

# 종합 하락 레이블: 둘 중 하나라도 해당
merged['decline'] = ((merged['decline_perf'] == 1) | (merged['decline_avail'] == 1)).astype(int)

print(f"  전체 샘플: {len(merged)}")
print(f"  성과 하락: {merged['decline_perf'].sum()} ({merged['decline_perf'].mean():.1%})")
print(f"  가용성 하락: {merged['decline_avail'].sum()} ({merged['decline_avail'].mean():.1%})")
print(f"  종합 하락: {merged['decline'].sum()} ({merged['decline'].mean():.1%})")

# ─────────────────────────────────────────────
# 8. 과거 성과 궤적 피처 계산 (최근 2-3 시즌 기울기)
# ─────────────────────────────────────────────
print("[5] 역사적 성과 궤적 피처 계산 중...")

def compute_trajectory(df, player_col='player_key', year_col='season_year', score_col='perf_score', n_seasons=3):
    """최근 n_seasons에 걸친 성과 기울기 계산 (선형 회귀)"""
    results = []
    for (player,), grp in df.groupby([player_col]):
        grp = grp.sort_values(year_col)
        for idx, row in grp.iterrows():
            cur_year = row[year_col]
            # 현재 시즌 포함 과거 n_seasons
            hist = grp[grp[year_col] <= cur_year].tail(n_seasons)
            if len(hist) < 2:
                slope = np.nan
                peak_minus_current = np.nan
            else:
                x = hist[year_col].values.astype(float)
                y = hist[score_col].values.astype(float)
                # nan 제거
                valid = ~np.isnan(y)
                if valid.sum() >= 2:
                    slope = np.polyfit(x[valid], y[valid], 1)[0]
                else:
                    slope = np.nan
                peak_score = grp[grp[year_col] <= cur_year][score_col].max()
                peak_minus_current = peak_score - row[score_col]
            results.append({player_col: player, year_col: cur_year,
                             'perf_slope': slope, 'peak_minus_current': peak_minus_current})
    return pd.DataFrame(results)

trajectory_df = compute_trajectory(
    season_df[['player_key', 'season_year', 'perf_score']].drop_duplicates()
)

merged = merged.merge(trajectory_df, on=['player_key', 'season_year'], how='left')

# ─────────────────────────────────────────────
# 9. 시장 가치 변화 트렌드 (최근 2시즌 기울기)
# ─────────────────────────────────────────────
mv_traj = []
for player, grp in season_df.groupby('player_key'):
    grp = grp.sort_values('season_year')
    for idx, row in grp.iterrows():
        cur_year = row['season_year']
        hist = grp[grp['season_year'] <= cur_year].tail(3)
        if len(hist) < 2:
            mv_slope = np.nan
        else:
            x = hist['season_year'].values.astype(float)
            y = hist['market_value'].fillna(method='ffill').fillna(0).values.astype(float)
            valid = ~np.isnan(y)
            if valid.sum() >= 2:
                mv_slope = np.polyfit(x[valid], y[valid], 1)[0]
            else:
                mv_slope = np.nan
        mv_traj.append({'player_key': player, 'season_year': cur_year, 'mv_slope': mv_slope})

mv_traj_df = pd.DataFrame(mv_traj)
merged = merged.merge(mv_traj_df, on=['player_key', 'season_year'], how='left')

# ─────────────────────────────────────────────
# 10. 워크로드 피처 (최근 2시즌 누적 출전 시간)
# ─────────────────────────────────────────────
workload = []
for player, grp in season_df.groupby('player_key'):
    grp = grp.sort_values('season_year')
    for idx, row in grp.iterrows():
        cur_year = row['season_year']
        hist = grp[(grp['season_year'] >= cur_year - 2) & (grp['season_year'] <= cur_year)]
        workload.append({
            'player_key': player,
            'season_year': cur_year,
            'workload_2y_min': hist['min'].sum(),       # 최근 2시즌 총 출전 분
            'workload_2y_mp': hist['mp'].sum(),          # 최근 2시즌 총 경기 수
        })

workload_df = pd.DataFrame(workload)
merged = merged.merge(workload_df, on=['player_key', 'season_year'], how='left')

# ─────────────────────────────────────────────
# 11. consistency, avg_min_match 병합
# ─────────────────────────────────────────────
cons_features = season_df[['player_key', 'season_year', 'consistency_score',
                             'match_count', 'avg_min_match', 'total_min_log']].copy()
# 중복 제거 (player_key + season_year 기준 평균)
cons_features = cons_features.groupby(['player_key', 'season_year']).mean().reset_index()
merged = merged.merge(cons_features, on=['player_key', 'season_year'], how='left')

# ─────────────────────────────────────────────
# 12. 포지션 인코딩
# ─────────────────────────────────────────────
pos_dummies = pd.get_dummies(merged['pos_group'], prefix='pos')
merged = pd.concat([merged, pos_dummies], axis=1)

# ─────────────────────────────────────────────
# 13. 피처 목록 정의 (25+)
# ─────────────────────────────────────────────
FEATURE_COLS = [
    # 나이 관련
    'age',
    'age_sq',          # age²
    # 현재 시즌 성과
    'perf_score',
    'min',
    'mp',
    'consistency_score',
    'match_count',
    'avg_min_match',
    'total_min_log',
    # 역사적 궤적
    'perf_slope',
    'peak_minus_current',
    # 워크로드
    'workload_2y_min',
    'workload_2y_mp',
    # 시장 가치
    'market_value',
    'mv_slope',
    # 팀 컨텍스트
    'team_quality',
    # 포지션 원-핫 인코딩
    'pos_DEF',
    'pos_FWD',
    'pos_GK',
    'pos_MID',
    'pos_Unknown',
]

# age² 추가
merged['age_sq'] = merged['age'] ** 2

# 없는 포지션 더미 열 보완
for col in ['pos_DEF', 'pos_FWD', 'pos_GK', 'pos_MID', 'pos_Unknown']:
    if col not in merged.columns:
        merged[col] = 0

print(f"\n[6] 최종 피처 수: {len(FEATURE_COLS)}")
print(f"  피처 목록: {FEATURE_COLS}")
print(f"  사용 가능 샘플 수: {len(merged)}")

# ─────────────────────────────────────────────
# 14. 시간 기반 학습/검증/테스트 분리
# ─────────────────────────────────────────────
print("\n[7] Train/Val/Test 분리 (시간 기반)...")

train_mask = merged['season_year'] < 2021
val_mask = (merged['season_year'] >= 2021) & (merged['season_year'] <= 2022)
test_mask = merged['season_year'] >= 2023

X_train = merged.loc[train_mask, FEATURE_COLS]
y_train = merged.loc[train_mask, 'decline']
X_val = merged.loc[val_mask, FEATURE_COLS]
y_val = merged.loc[val_mask, 'decline']
X_test = merged.loc[test_mask, FEATURE_COLS]
y_test = merged.loc[test_mask, 'decline']

print(f"  Train: {len(X_train)} (decline rate: {y_train.mean():.1%})")
print(f"  Val:   {len(X_val)} (decline rate: {y_val.mean():.1%})")
print(f"  Test:  {len(X_test)} (decline rate: {y_test.mean():.1%})")

# ─────────────────────────────────────────────
# 15. 결측치 처리 (중앙값 대체)
# ─────────────────────────────────────────────
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_val_imp = imputer.transform(X_val)
X_test_imp = imputer.transform(X_test)

X_train_df = pd.DataFrame(X_train_imp, columns=FEATURE_COLS)
X_val_df = pd.DataFrame(X_val_imp, columns=FEATURE_COLS)
X_test_df = pd.DataFrame(X_test_imp, columns=FEATURE_COLS)

# ─────────────────────────────────────────────
# 16. 클래스 불균형 처리 (SMOTE)
# ─────────────────────────────────────────────
print("\n[8] SMOTE 클래스 균형 조정...")
smote = SMOTE(random_state=42, k_neighbors=min(5, int(y_train.sum()) - 1))
try:
    X_train_res, y_train_res = smote.fit_resample(X_train_df, y_train)
    print(f"  SMOTE 후 훈련 샘플: {len(X_train_res)} (decline: {y_train_res.mean():.1%})")
except Exception as e:
    print(f"  SMOTE 실패 ({e}), 원본 데이터 사용")
    X_train_res, y_train_res = X_train_df, y_train

# ─────────────────────────────────────────────
# 17. 모델 정의 및 학습
# ─────────────────────────────────────────────
print("\n[9] 모델 학습 중...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_val_scaled = scaler.transform(X_val_df)
X_test_scaled = scaler.transform(X_test_df)

# (a) XGBoost
print("  XGBoost 학습 중...")
scale_pos = (y_train_res == 0).sum() / (y_train_res == 1).sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
)
xgb_model.fit(X_train_res, y_train_res,
              eval_set=[(X_val_df, y_val)],
              verbose=False)

# (b) Random Forest
print("  Random Forest 학습 중...")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)
rf_model.fit(X_train_res, y_train_res)

# (c) Logistic Regression
print("  Logistic Regression 학습 중...")
lr_model = LogisticRegression(
    C=0.1,
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
)
lr_model.fit(X_train_scaled, y_train_res)

# (d) MLP
print("  MLP 학습 중...")
mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
)
mlp_model.fit(X_train_scaled, y_train_res)

# ─────────────────────────────────────────────
# 18. 모델 평가
# ─────────────────────────────────────────────
print("\n[10] 모델 평가...")

models = {
    'XGBoost': (xgb_model, X_test_df),
    'RandomForest': (rf_model, X_test_df),
    'LogisticRegression': (lr_model, X_test_scaled),
    'MLP': (mlp_model, X_test_scaled),
}

results = {}
for name, (model, X_eval) in models.items():
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = {
        'auc_roc': round(auc, 4),
        'avg_precision': round(ap, 4),
        'precision_decline': round(report.get('1', {}).get('precision', 0), 4),
        'recall_decline': round(report.get('1', {}).get('recall', 0), 4),
        'f1_decline': round(report.get('1', {}).get('f1-score', 0), 4),
    }
    print(f"  {name}: AUC={auc:.4f}, AP={ap:.4f}, "
          f"F1(decline)={results[name]['f1_decline']:.4f}")

# 최고 성능 모델 선택 (AUC 기준)
best_model_name = max(results, key=lambda k: results[k]['auc_roc'])
best_model, best_X = models[best_model_name]
print(f"\n  최고 모델: {best_model_name} (AUC={results[best_model_name]['auc_roc']:.4f})")

# ─────────────────────────────────────────────
# 19. 전체 데이터 예측 (Scout 출력용)
# ─────────────────────────────────────────────
print("\n[11] 전체 데이터 하락 확률 예측...")

all_features = merged[FEATURE_COLS].copy()
all_features_imp = imputer.transform(all_features)
all_features_df = pd.DataFrame(all_features_imp, columns=FEATURE_COLS)

# XGBoost 예측 (전체 데이터)
xgb_probs_all = xgb_model.predict_proba(all_features_df)[:, 1]
rf_probs_all = rf_model.predict_proba(all_features_df)[:, 1]
lr_probs_all = lr_model.predict_proba(scaler.transform(all_features_df))[:, 1]
mlp_probs_all = mlp_model.predict_proba(scaler.transform(all_features_df))[:, 1]

# 앙상블 (평균)
ensemble_probs = (xgb_probs_all + rf_probs_all + lr_probs_all + mlp_probs_all) / 4.0

merged_output = merged.copy()
merged_output['decline_prob_xgb'] = xgb_probs_all
merged_output['decline_prob_rf'] = rf_probs_all
merged_output['decline_prob_lr'] = lr_probs_all
merged_output['decline_prob_mlp'] = mlp_probs_all
merged_output['decline_prob_ensemble'] = ensemble_probs

# ─────────────────────────────────────────────
# 20. Scout 출력: 하락 감시 리스트 (최신 시즌)
# ─────────────────────────────────────────────
print("\n[12] Scout 출력 생성 중...")

latest_year = merged_output['season_year'].max()
latest_data = merged_output[merged_output['season_year'] == latest_year].copy()
# 동일 선수 중복 제거 (가장 높은 하락 확률 행 유지)
latest_data = latest_data.sort_values('decline_prob_ensemble', ascending=False).drop_duplicates(
    subset='player_key', keep='first'
)

# 하락 감시 리스트 (확률 상위 30명)
decline_watch = latest_data.nlargest(30, 'decline_prob_ensemble')[
    ['player_key', 'team', 'pos_group', 'age', 'season_year',
     'decline_prob_ensemble', 'decline_prob_xgb', 'decline_perf', 'decline_avail',
     'perf_score', 'min', 'perf_slope', 'peak_minus_current', 'market_value']
].reset_index(drop=True)

# 아이언맨 리스트 (확률 하위 30명, 출전 시간 1000분 이상)
iron_men = latest_data[latest_data['min'] >= 1000].nsmallest(30, 'decline_prob_ensemble')[
    ['player_key', 'team', 'pos_group', 'age', 'season_year',
     'decline_prob_ensemble', 'perf_score', 'min', 'consistency_score', 'market_value']
].reset_index(drop=True)

print(f"\n  [하락 감시 리스트 TOP 10]")
print(decline_watch[['player_key', 'team', 'age', 'pos_group',
                       'decline_prob_ensemble']].head(10).to_string(index=False))
print(f"\n  [아이언맨 리스트 TOP 10]")
print(iron_men[['player_key', 'team', 'age', 'pos_group',
                 'decline_prob_ensemble']].head(10).to_string(index=False))

# ─────────────────────────────────────────────
# 21. 나이별·포지션별 하락 곡선 데이터
# ─────────────────────────────────────────────
age_decline = merged_output.groupby(['age', 'pos_group'])['decline_prob_ensemble'].mean().reset_index()

# ─────────────────────────────────────────────
# 22. 피처 중요도 (XGBoost 기준)
# ─────────────────────────────────────────────
feature_importance = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance_xgb': xgb_model.feature_importances_,
    'importance_rf': rf_model.feature_importances_,
}).sort_values('importance_xgb', ascending=False)

print(f"\n  [주요 하락 예측 피처 TOP 10]")
print(feature_importance.head(10).to_string(index=False))

# ─────────────────────────────────────────────
# 23. 시각화
# ─────────────────────────────────────────────
print("\n[13] 시각화 생성 중...")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

COLOR_PALETTE = {
    'DEF': '#2196F3',
    'MID': '#4CAF50',
    'FWD': '#F44336',
    'GK': '#FF9800',
    'Unknown': '#9E9E9E',
}

# ── 그림 1: 포지션별 나이-하락 확률 곡선 ──────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

for pos_g in ['DEF', 'MID', 'FWD', 'GK']:
    sub = age_decline[age_decline['pos_group'] == pos_g]
    sub = sub[(sub['age'] >= 18) & (sub['age'] <= 38)]
    if len(sub) < 3:
        continue
    sub_sorted = sub.sort_values('age')
    # 스무딩 (rolling average)
    smoothed = sub_sorted.set_index('age')['decline_prob_ensemble'].rolling(
        window=3, center=True, min_periods=1
    ).mean()
    ax.plot(smoothed.index, smoothed.values,
            color=COLOR_PALETTE.get(pos_g, 'gray'),
            linewidth=2.5, label=pos_g, marker='o', markersize=4)

ax.axvline(x=30, color='black', linestyle='--', alpha=0.4, label='Age 30')
ax.axvline(x=33, color='red', linestyle='--', alpha=0.4, label='Age 33 (high risk)')
ax.set_xlabel('Age', fontsize=13)
ax.set_ylabel('Decline Probability', fontsize=13)
ax.set_title('Decline Probability by Age and Position\n(EPL Player Decline Risk)', fontsize=15)
ax.legend(fontsize=11)
ax.set_xlim(18, 40)
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / 'decline_prob_by_age.png', dpi=150, bbox_inches='tight')
plt.close()
print("  decline_prob_by_age.png 저장 완료")

# ── 그림 2: 피처 중요도 ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for ax_idx, (ax, imp_col, title) in enumerate(zip(
    axes,
    ['importance_xgb', 'importance_rf'],
    ['XGBoost Feature Importance', 'Random Forest Feature Importance']
)):
    top15 = feature_importance.nlargest(15, imp_col)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top15)))
    bars = ax.barh(range(len(top15)), top15[imp_col].values, color=colors)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15['feature'].values, fontsize=10)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

plt.suptitle('Feature Importance for Player Decline Prediction', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  feature_importance.png 저장 완료")

# ── 그림 3: 하락 감시 리스트 시각화 ─────────────────────────────
fig, ax = plt.subplots(figsize=(14, 10))

watch_top20 = decline_watch.head(20).sort_values('decline_prob_ensemble')
colors = [COLOR_PALETTE.get(p, '#9E9E9E') for p in watch_top20['pos_group']]
bars = ax.barh(range(len(watch_top20)), watch_top20['decline_prob_ensemble'].values,
               color=colors, edgecolor='white', linewidth=0.5)

# 바 오른쪽에 나이 표시
for i, (_, row) in enumerate(watch_top20.iterrows()):
    ax.text(row['decline_prob_ensemble'] + 0.005, i,
            f" Age {int(row['age']) if not np.isnan(row['age']) else '?'} | {row['team'][:15]}",
            va='center', fontsize=9)

ax.set_yticks(range(len(watch_top20)))
ax.set_yticklabels([f"{row['player_key'][:25]} ({row['pos_group']})"
                     for _, row in watch_top20.iterrows()], fontsize=9)
ax.set_xlabel('Decline Probability (Ensemble)', fontsize=12)
ax.set_title('Decline Watch List - Top 20 Players at Risk\n(Latest Season)', fontsize=14)
ax.set_xlim(0, 1.15)
ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
ax.axvline(x=0.7, color='darkred', linestyle='--', alpha=0.5, label='70% high risk')

# 포지션 범례
legend_patches = [mpatches.Patch(color=c, label=p) for p, c in COLOR_PALETTE.items() if p != 'Unknown']
ax.legend(handles=legend_patches, loc='lower right', fontsize=9)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / 'decline_watch_list.png', dpi=150, bbox_inches='tight')
plt.close()
print("  decline_watch_list.png 저장 완료")

# ── 그림 4: ROC 커브 (모든 모델) ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC 커브
ax = axes[0]
model_probs = {
    'XGBoost': xgb_model.predict_proba(X_test_df)[:, 1],
    'RandomForest': rf_model.predict_proba(X_test_df)[:, 1],
    'LogisticRegression': lr_model.predict_proba(X_test_scaled)[:, 1],
    'MLP': mlp_model.predict_proba(X_test_scaled)[:, 1],
}
model_colors = ['#F44336', '#2196F3', '#4CAF50', '#FF9800']

for (mname, probs), color in zip(model_probs.items(), model_colors):
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{mname} (AUC={auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - All Models', fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Precision-Recall 커브
ax = axes[1]
for (mname, probs), color in zip(model_probs.items(), model_colors):
    prec, rec, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    ax.plot(rec, prec, color=color, linewidth=2, label=f'{mname} (AP={ap:.3f})')

baseline = y_test.mean()
ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.3f})')
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves - All Models', fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.suptitle('Model Performance Comparison for Decline Prediction', fontsize=15)
plt.tight_layout()
plt.savefig(FIG_DIR / 'roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  roc_curves.png 저장 완료")

# ── 그림 5: 아이언맨 vs 하락 위험 산점도 ────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))

# 최신 시즌 데이터에서 출전 1000분 이상만
scatter_data = latest_data[latest_data['min'] >= 900].copy()
colors_scatter = [COLOR_PALETTE.get(p, '#9E9E9E') for p in scatter_data['pos_group']]

scatter = ax.scatter(
    scatter_data['age'],
    scatter_data['decline_prob_ensemble'],
    c=scatter_data['min'],
    cmap='YlOrRd',
    s=scatter_data['min'] / 30,
    alpha=0.7,
    edgecolors='white',
    linewidth=0.3,
)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Minutes Played', fontsize=11)

# 하락 감시 선수 라벨
top_risk = scatter_data.nlargest(8, 'decline_prob_ensemble')
for _, row in top_risk.iterrows():
    ax.annotate(
        row['player_key'][:15],
        (row['age'], row['decline_prob_ensemble']),
        textcoords='offset points',
        xytext=(5, 5),
        fontsize=7,
        color='darkred',
    )

ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.4, label='High Risk Threshold (0.6)')
ax.axhline(y=0.3, color='green', linestyle='--', alpha=0.4, label='Iron Men Threshold (0.3)')
ax.set_xlabel('Age', fontsize=12)
ax.set_ylabel('Decline Probability', fontsize=12)
ax.set_title('Player Decline Risk: Age vs. Probability (Latest Season)\nBubble size = Minutes Played',
             fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / 'age_vs_decline_risk.png', dpi=150, bbox_inches='tight')
plt.close()
print("  age_vs_decline_risk.png 저장 완료")

# ─────────────────────────────────────────────
# 24. 결과 저장
# ─────────────────────────────────────────────
print("\n[14] 결과 저장 중...")

# decline_predictions.parquet
output_cols = ['player_key', 'team', 'pos_group', 'age', 'season_year',
               'decline', 'decline_perf', 'decline_avail',
               'decline_prob_xgb', 'decline_prob_rf', 'decline_prob_lr',
               'decline_prob_mlp', 'decline_prob_ensemble',
               'perf_score', 'min', 'mp', 'market_value',
               'perf_slope', 'peak_minus_current', 'consistency_score']
merged_output[output_cols].to_parquet(SCOUT_DIR / 'decline_predictions.parquet', index=False)
print(f"  decline_predictions.parquet 저장 완료 ({len(merged_output)} rows)")

# results_summary.json
summary = {
    'model_performance': results,
    'best_model': best_model_name,
    'best_auc': results[best_model_name]['auc_roc'],
    'dataset': {
        'total_samples': int(len(merged)),
        'train_samples': int(train_mask.sum()),
        'val_samples': int(val_mask.sum()),
        'test_samples': int(test_mask.sum()),
        'decline_rate_train': round(float(y_train.mean()), 4),
        'decline_rate_test': round(float(y_test.mean()), 4),
        'n_features': len(FEATURE_COLS),
        'feature_list': FEATURE_COLS,
    },
    'scout_outputs': {
        'decline_watch_list': decline_watch[
            ['player_key', 'team', 'pos_group', 'age', 'decline_prob_ensemble']
        ].assign(age=lambda df: df['age'].fillna(0).astype(int)).to_dict(orient='records'),
        'iron_men_list': iron_men[
            ['player_key', 'team', 'pos_group', 'age', 'decline_prob_ensemble']
        ].assign(age=lambda df: df['age'].fillna(0).astype(int)).to_dict(orient='records'),
    },
    'feature_importance_top10': feature_importance.head(10)[
        ['feature', 'importance_xgb', 'importance_rf']
    ].to_dict(orient='records'),
    'early_warning_signals': feature_importance.head(5)['feature'].tolist(),
    'decline_definition': {
        'performance_decline': '성과 지수가 0.5 표준편차 이상 하락',
        'availability_decline': '출전 시간 30% 이상 감소 (부상/form 하락)',
        'combined': '둘 중 하나 이상 해당',
    },
    'time_split': {
        'train': 'season_year < 2021',
        'val': '2021 <= season_year <= 2022',
        'test': 'season_year >= 2023',
    },
}

with open(SCOUT_DIR / 'results_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
print("  results_summary.json 저장 완료")

# ─────────────────────────────────────────────
# 25. 최종 요약 출력
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("[S6] 선수 하락세 감지 모델 완료")
print("=" * 60)
print(f"\n최고 모델: {best_model_name}")
print(f"  AUC-ROC: {results[best_model_name]['auc_roc']:.4f}")
print(f"  Avg Precision: {results[best_model_name]['avg_precision']:.4f}")
print(f"  F1 (decline): {results[best_model_name]['f1_decline']:.4f}")

print(f"\n모든 모델 성능:")
for name, res in results.items():
    print(f"  {name}: AUC={res['auc_roc']:.4f}, F1={res['f1_decline']:.4f}")

print(f"\n주요 하락 예측 신호 (Early Warning):")
for i, feat in enumerate(summary['early_warning_signals'], 1):
    print(f"  {i}. {feat}")

print(f"\n저장 파일:")
print(f"  데이터: {SCOUT_DIR}/decline_predictions.parquet")
print(f"  요약:   {SCOUT_DIR}/results_summary.json")
print(f"  그림:   {FIG_DIR}/")

print(f"\n하락 감시 리스트 TOP 5:")
print(decline_watch[['player_key', 'team', 'pos_group', 'age', 'decline_prob_ensemble']].head(5).to_string(index=False))
print(f"\n아이언맨 TOP 5:")
print(iron_men[['player_key', 'team', 'pos_group', 'age', 'decline_prob_ensemble']].head(5).to_string(index=False))

"""
S6 하락세 감지 모델 결과 복원 스크립트
decline_predictions.parquet을 읽고, 필요 시 모델을 재학습하여
results_summary.json을 생성한다.
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
    roc_curve, precision_recall_curve, average_precision_score, f1_score
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
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
SCOUT_DIR = BASE_DIR / "data" / "scout"
MODEL_DIR = BASE_DIR / "models" / "s6_decline"
FIG_DIR = MODEL_DIR / "figures"

for d in [SCOUT_DIR, FIG_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("[S6-FIX] 선수 하락세 감지 모델 결과 복원 시작")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. decline_predictions.parquet 로드
# ─────────────────────────────────────────────
pred_path = SCOUT_DIR / "decline_predictions.parquet"
print(f"\n[1] decline_predictions.parquet 로드: {pred_path}")

predictions_df = pd.read_parquet(pred_path)
print(f"  로드 완료: {predictions_df.shape}")
print(f"  컬럼: {list(predictions_df.columns)}")

# ─────────────────────────────────────────────
# 2. 원본 데이터 재로드 및 피처 재구성
# ─────────────────────────────────────────────
print("\n[2] 원본 데이터 로드 및 모델 재학습...")

season_df = pd.read_parquet(DATA_DIR / "player_season_stats.parquet")
match_df = pd.read_parquet(DATA_DIR / "player_match_logs.parquet")
team_df = pd.read_parquet(DATA_DIR / "team_season_summary.parquet")

print(f"  선수 시즌 통계: {season_df.shape}")
print(f"  경기 로그: {match_df.shape}")
print(f"  팀 요약: {team_df.shape}")

# 시즌 연도 파싱
def parse_season_year(s):
    try:
        return int(str(s).split('/')[0])
    except:
        return np.nan

season_df['season_year'] = season_df['season'].apply(parse_season_year)
team_df['season_year'] = team_df['Season'].apply(parse_season_year)

# 포지션 그룹화
def map_position_group(pos):
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
        return 'MID'

season_df['pos_group'] = season_df['position'].apply(map_position_group)

# 포지션별 복합 성과 지수
season_df['90s_safe'] = season_df['90s'].clip(lower=0.1)
POS_METRICS = {
    'FWD': ['gls_1', 'ast_1', 'g_a_1'],
    'MID': ['ast_1', 'gls_1', 'g_a_1'],
    'DEF': ['gls_1', 'ast_1'],
    'GK':  ['gls_1', 'ast_1'],
    'Unknown': ['gls_1', 'ast_1'],
}

def compute_composite_score(df):
    scores = np.zeros(len(df))
    for pos_g, metrics in POS_METRICS.items():
        mask = df['pos_group'] == pos_g
        if mask.sum() == 0:
            continue
        avail_metrics = [m for m in metrics if m in df.columns]
        if not avail_metrics:
            continue
        sub = df.loc[mask, avail_metrics].copy()
        z = sub.apply(lambda col: stats.zscore(col.fillna(0), nan_policy='omit'), axis=0)
        scores[mask.values] = z.mean(axis=1).values
    return scores

season_df['perf_score'] = compute_composite_score(season_df)

# 팀 품질
team_quality = team_df.groupby(['team', 'season_year'])['points'].sum().reset_index()
team_quality.rename(columns={'points': 'team_quality'}, inplace=True)
team_quality['team_quality'] = team_quality.groupby('season_year')['team_quality'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
)
season_df = season_df.merge(team_quality, on=['team', 'season_year'], how='left')

# 경기 일관성
match_df['season_year'] = match_df['season'].apply(parse_season_year)
match_df['g_a_match'] = match_df['gls'].fillna(0) + match_df['ast'].fillna(0)
consistency = match_df.groupby(['player', 'season_year']).agg(
    consistency_std=('g_a_match', 'std'),
    match_count=('g_a_match', 'count'),
    avg_min_match=('min', 'mean'),
    total_min_log=('min', 'sum'),
).reset_index()
consistency['consistency_score'] = 1.0 / (1.0 + consistency['consistency_std'].fillna(0))

season_df = season_df.merge(
    consistency[['player', 'season_year', 'consistency_score', 'match_count',
                  'avg_min_match', 'total_min_log']],
    on=['player', 'season_year'], how='left'
)

# 시즌 N → N+1 하락 레이블
season_df['player_key'] = season_df['player']
current = season_df[['player_key', 'season_year', 'perf_score', 'min', 'mp',
                       'market_value', 'age', 'pos_group', 'team', 'team_quality']].copy()
next_season = current[['player_key', 'season_year', 'perf_score', 'min', 'mp', 'market_value']].copy()
next_season.columns = ['player_key', 'next_year', 'next_perf_score', 'next_min',
                        'next_mp', 'next_market_value']
next_season['season_year'] = next_season['next_year'] - 1
merged = current.merge(next_season, on=['player_key', 'season_year'], how='inner')

perf_diff = merged['next_perf_score'] - merged['perf_score']
perf_std = perf_diff.std()
merged['decline_perf'] = (perf_diff < -0.5 * perf_std).astype(int)
merged['decline_avail'] = (
    (merged['next_min'] < merged['min'] * 0.7) & (merged['min'] >= 450)
).astype(int)
merged['decline'] = ((merged['decline_perf'] == 1) | (merged['decline_avail'] == 1)).astype(int)

print(f"  전체 샘플: {len(merged)}")
print(f"  하락 레이블 비율: {merged['decline'].mean():.1%}")

# 궤적 피처
def compute_trajectory(df, player_col='player_key', year_col='season_year', score_col='perf_score', n_seasons=3):
    results = []
    for (player,), grp in df.groupby([player_col]):
        grp = grp.sort_values(year_col)
        for idx, row in grp.iterrows():
            cur_year = row[year_col]
            hist = grp[grp[year_col] <= cur_year].tail(n_seasons)
            if len(hist) < 2:
                slope = np.nan
                peak_minus_current = np.nan
            else:
                x = hist[year_col].values.astype(float)
                y = hist[score_col].values.astype(float)
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

# 시장가치 궤적
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
            y = hist['market_value'].ffill().fillna(0).values.astype(float)
            valid = ~np.isnan(y)
            mv_slope = np.polyfit(x[valid], y[valid], 1)[0] if valid.sum() >= 2 else np.nan
        mv_traj.append({'player_key': player, 'season_year': cur_year, 'mv_slope': mv_slope})
mv_traj_df = pd.DataFrame(mv_traj)
merged = merged.merge(mv_traj_df, on=['player_key', 'season_year'], how='left')

# 워크로드
workload = []
for player, grp in season_df.groupby('player_key'):
    grp = grp.sort_values('season_year')
    for idx, row in grp.iterrows():
        cur_year = row['season_year']
        hist = grp[(grp['season_year'] >= cur_year - 2) & (grp['season_year'] <= cur_year)]
        workload.append({
            'player_key': player,
            'season_year': cur_year,
            'workload_2y_min': hist['min'].sum(),
            'workload_2y_mp': hist['mp'].sum(),
        })
workload_df = pd.DataFrame(workload)
merged = merged.merge(workload_df, on=['player_key', 'season_year'], how='left')

# 일관성 피처 병합
cons_features = season_df[['player_key', 'season_year', 'consistency_score',
                             'match_count', 'avg_min_match', 'total_min_log']].copy()
cons_features = cons_features.groupby(['player_key', 'season_year']).mean().reset_index()
merged = merged.merge(cons_features, on=['player_key', 'season_year'], how='left')

# 포지션 인코딩
pos_dummies = pd.get_dummies(merged['pos_group'], prefix='pos')
merged = pd.concat([merged, pos_dummies], axis=1)
merged['age_sq'] = merged['age'] ** 2
for col in ['pos_DEF', 'pos_FWD', 'pos_GK', 'pos_MID', 'pos_Unknown']:
    if col not in merged.columns:
        merged[col] = 0

FEATURE_COLS = [
    'age', 'age_sq',
    'perf_score', 'min', 'mp',
    'consistency_score', 'match_count', 'avg_min_match', 'total_min_log',
    'perf_slope', 'peak_minus_current',
    'workload_2y_min', 'workload_2y_mp',
    'market_value', 'mv_slope',
    'team_quality',
    'pos_DEF', 'pos_FWD', 'pos_GK', 'pos_MID', 'pos_Unknown',
]

# ─────────────────────────────────────────────
# 3. Train/Val/Test 분리
# ─────────────────────────────────────────────
print("\n[3] Train/Val/Test 분리...")
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

# 결측치 처리
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_val_imp = imputer.transform(X_val)
X_test_imp = imputer.transform(X_test)
X_train_df = pd.DataFrame(X_train_imp, columns=FEATURE_COLS)
X_val_df = pd.DataFrame(X_val_imp, columns=FEATURE_COLS)
X_test_df = pd.DataFrame(X_test_imp, columns=FEATURE_COLS)

# SMOTE
if HAS_SMOTE and int(y_train.sum()) > 5:
    try:
        smote = SMOTE(random_state=42, k_neighbors=min(5, int(y_train.sum()) - 1))
        X_train_res, y_train_res = smote.fit_resample(X_train_df, y_train)
        print(f"  SMOTE 후 훈련 샘플: {len(X_train_res)} (decline: {y_train_res.mean():.1%})")
    except Exception as e:
        print(f"  SMOTE 실패 ({e}), 원본 사용")
        X_train_res, y_train_res = X_train_df, y_train
else:
    X_train_res, y_train_res = X_train_df, y_train

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_val_scaled = scaler.transform(X_val_df)
X_test_scaled = scaler.transform(X_test_df)

# ─────────────────────────────────────────────
# 4. 모델 학습
# ─────────────────────────────────────────────
print("\n[4] 모델 학습...")

# XGBoost
print("  XGBoost...")
scale_pos = max(1.0, (y_train_res == 0).sum() / max(1, (y_train_res == 1).sum()))
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    eval_metric='logloss', random_state=42, n_jobs=-1,
)
xgb_model.fit(X_train_res, y_train_res,
              eval_set=[(X_val_df, y_val)], verbose=False)

# Random Forest
print("  Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=300, max_depth=8, class_weight='balanced',
    random_state=42, n_jobs=-1,
)
rf_model.fit(X_train_res, y_train_res)

# Logistic Regression
print("  Logistic Regression...")
lr_model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train_res)

# MLP
print("  MLP...")
mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32), activation='relu',
    max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1,
)
mlp_model.fit(X_train_scaled, y_train_res)

# ─────────────────────────────────────────────
# 5. 모델 평가 (Test set)
# ─────────────────────────────────────────────
print("\n[5] 모델 평가...")

models_eval = {
    'XGBoost':            (xgb_model, X_test_df),
    'RandomForest':       (rf_model, X_test_df),
    'LogisticRegression': (lr_model, X_test_scaled),
    'MLP':                (mlp_model, X_test_scaled),
}

results = {}
confusion_matrices = {}
for name, (model, X_eval) in models_eval.items():
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]
    auc = roc_auc_score(y_test, y_prob) if len(y_test.unique()) > 1 else 0.5
    ap  = average_precision_score(y_test, y_prob) if len(y_test.unique()) > 1 else float(y_test.mean())
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    results[name] = {
        'auc_roc':           round(float(auc), 4),
        'avg_precision':     round(float(ap), 4),
        'precision_decline': round(float(report.get('1', {}).get('precision', 0)), 4),
        'recall_decline':    round(float(report.get('1', {}).get('recall', 0)), 4),
        'f1_decline':        round(float(report.get('1', {}).get('f1-score', 0)), 4),
        'accuracy':          round(float(report.get('accuracy', 0)), 4),
    }
    confusion_matrices[name] = cm
    print(f"  {name}: AUC={auc:.4f}, F1(decline)={results[name]['f1_decline']:.4f}")

best_model_name = max(results, key=lambda k: results[k]['auc_roc'])
print(f"\n  최고 모델: {best_model_name} (AUC={results[best_model_name]['auc_roc']:.4f})")

# ─────────────────────────────────────────────
# 6. 전체 데이터 예측 (Scout 출력)
# ─────────────────────────────────────────────
print("\n[6] 전체 데이터 하락 확률 예측...")

all_features = merged[FEATURE_COLS].copy()
all_features_imp = imputer.transform(all_features)
all_features_df = pd.DataFrame(all_features_imp, columns=FEATURE_COLS)
all_features_scaled = scaler.transform(all_features_df)

xgb_probs_all  = xgb_model.predict_proba(all_features_df)[:, 1]
rf_probs_all   = rf_model.predict_proba(all_features_df)[:, 1]
lr_probs_all   = lr_model.predict_proba(all_features_scaled)[:, 1]
mlp_probs_all  = mlp_model.predict_proba(all_features_scaled)[:, 1]
ensemble_probs = (xgb_probs_all + rf_probs_all + lr_probs_all + mlp_probs_all) / 4.0

merged_output = merged.copy()
merged_output['decline_prob_xgb']      = xgb_probs_all
merged_output['decline_prob_rf']       = rf_probs_all
merged_output['decline_prob_lr']       = lr_probs_all
merged_output['decline_prob_mlp']      = mlp_probs_all
merged_output['decline_prob_ensemble'] = ensemble_probs

# ─────────────────────────────────────────────
# 7. 최신 시즌 Scout 리스트 생성
# ─────────────────────────────────────────────
print("\n[7] Scout 리스트 생성...")

latest_year = merged_output['season_year'].max()
latest_data = merged_output[merged_output['season_year'] == latest_year].copy()
latest_data = (
    latest_data
    .sort_values('decline_prob_ensemble', ascending=False)
    .drop_duplicates(subset='player_key', keep='first')
)
print(f"  최신 시즌: {latest_year}, 선수 수: {len(latest_data)}")

# 하락 감시 리스트 TOP 30
decline_watch = latest_data.nlargest(30, 'decline_prob_ensemble')[
    ['player_key', 'team', 'pos_group', 'age', 'season_year',
     'decline_prob_ensemble', 'decline_prob_xgb', 'decline_perf', 'decline_avail',
     'perf_score', 'min', 'perf_slope', 'peak_minus_current', 'market_value']
].reset_index(drop=True)

# 아이언맨 리스트 (출전 1000분 이상, 가장 낮은 하락 확률)
iron_men_pool = latest_data[latest_data['min'] >= 1000]
if len(iron_men_pool) < 20:
    iron_men_pool = latest_data[latest_data['min'] >= 500]
iron_men = iron_men_pool.nsmallest(30, 'decline_prob_ensemble')[
    ['player_key', 'team', 'pos_group', 'age', 'season_year',
     'decline_prob_ensemble', 'perf_score', 'min', 'consistency_score', 'market_value']
].reset_index(drop=True)

# 연령 타당성 검증
print("\n  [하락 감시 리스트 TOP 10]")
print(decline_watch[['player_key', 'team', 'age', 'pos_group', 'decline_prob_ensemble']].head(10).to_string(index=False))
print(f"\n  [아이언맨 리스트 TOP 10]")
print(iron_men[['player_key', 'team', 'age', 'pos_group', 'decline_prob_ensemble']].head(10).to_string(index=False))

# 감시 리스트 연령 분포 확인
watch_ages = decline_watch['age'].dropna()
iron_ages  = iron_men['age'].dropna()
print(f"\n  하락 감시 리스트 평균 나이: {watch_ages.mean():.1f} (합리적이면 30+)")
print(f"  아이언맨 리스트 평균 나이: {iron_ages.mean():.1f} (합리적이면 20s-mid-30s)")

# ─────────────────────────────────────────────
# 8. 피처 중요도
# ─────────────────────────────────────────────
feature_importance = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance_xgb': xgb_model.feature_importances_,
    'importance_rf':  rf_model.feature_importances_,
}).sort_values('importance_xgb', ascending=False)

print("\n  [피처 중요도 TOP 10 - XGBoost]")
print(feature_importance.head(10)[['feature', 'importance_xgb', 'importance_rf']].to_string(index=False))

# ─────────────────────────────────────────────
# 9. 나이/포지션별 하락율
# ─────────────────────────────────────────────
age_decline_table = (
    merged_output
    .groupby(['age', 'pos_group'])['decline_prob_ensemble']
    .mean()
    .reset_index()
)

# 포지션별 연령 구간 하락 요약
age_bins = [(18, 22), (23, 26), (27, 30), (31, 35), (36, 40)]
age_pos_decline = {}
for pos in ['DEF', 'MID', 'FWD', 'GK']:
    sub = merged_output[merged_output['pos_group'] == pos].copy()
    bin_data = {}
    for lo, hi in age_bins:
        seg = sub[(sub['age'] >= lo) & (sub['age'] <= hi)]
        if len(seg) > 0:
            bin_data[f"{lo}-{hi}"] = {
                'avg_decline_prob': round(float(seg['decline_prob_ensemble'].mean()), 4),
                'actual_decline_rate': round(float(seg['decline'].mean()), 4),
                'n_players': int(len(seg)),
            }
    age_pos_decline[pos] = bin_data

# ─────────────────────────────────────────────
# 10. parquet 저장 (기존 갱신)
# ─────────────────────────────────────────────
print("\n[8] decline_predictions.parquet 갱신 저장...")
output_cols = ['player_key', 'team', 'pos_group', 'age', 'season_year',
               'decline', 'decline_perf', 'decline_avail',
               'decline_prob_xgb', 'decline_prob_rf', 'decline_prob_lr',
               'decline_prob_mlp', 'decline_prob_ensemble',
               'perf_score', 'min', 'mp', 'market_value',
               'perf_slope', 'peak_minus_current', 'consistency_score']
merged_output[output_cols].to_parquet(SCOUT_DIR / 'decline_predictions.parquet', index=False)
print(f"  저장 완료 ({len(merged_output)} rows)")

# ─────────────────────────────────────────────
# 11. results_summary.json 생성
# ─────────────────────────────────────────────
print("\n[9] results_summary.json 생성...")

def safe_records(df, cols):
    """DataFrame을 JSON-safe dict list로 변환"""
    subset = df[cols].copy()
    for col in subset.select_dtypes(include='float').columns:
        subset[col] = subset[col].round(4)
    # int 변환 가능한 나이 컬럼
    if 'age' in subset.columns:
        subset['age'] = subset['age'].fillna(0).astype(int)
    return [
        {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
         for k, v in row.items()}
        for row in subset.to_dict(orient='records')
    ]

summary = {
    'model': 'S6 Player Decline Detection',
    'description': 'EPL 선수 하락세 및 부상 위험 감지 모델 (이진 분류)',
    'latest_season': int(latest_year),

    # ── 모델 성능 ──────────────────────────────────
    'model_performance': results,
    'best_model': best_model_name,
    'best_auc': results[best_model_name]['auc_roc'],

    # ── 혼동 행렬 ──────────────────────────────────
    'confusion_matrices': confusion_matrices,

    # ── 데이터셋 정보 ──────────────────────────────
    'dataset': {
        'total_samples':      int(len(merged)),
        'train_samples':      int(train_mask.sum()),
        'val_samples':        int(val_mask.sum()),
        'test_samples':       int(test_mask.sum()),
        'decline_rate_train': round(float(y_train.mean()), 4),
        'decline_rate_val':   round(float(y_val.mean()), 4),
        'decline_rate_test':  round(float(y_test.mean()), 4),
        'n_features':         len(FEATURE_COLS),
        'feature_list':       FEATURE_COLS,
    },
    'time_split': {
        'train': 'season_year < 2021',
        'val':   '2021 <= season_year <= 2022',
        'test':  'season_year >= 2023',
    },

    # ── 피처 중요도 ────────────────────────────────
    'feature_importance_top10': safe_records(
        feature_importance.head(10),
        ['feature', 'importance_xgb', 'importance_rf']
    ),
    'early_warning_signals': feature_importance.head(5)['feature'].tolist(),

    # ── Scout 리스트 ───────────────────────────────
    'scout_outputs': {
        'decline_watch_list': safe_records(
            decline_watch.head(20),
            ['player_key', 'team', 'pos_group', 'age',
             'decline_prob_ensemble', 'decline_prob_xgb',
             'perf_score', 'perf_slope', 'peak_minus_current', 'market_value']
        ),
        'iron_men_list': safe_records(
            iron_men.head(20),
            ['player_key', 'team', 'pos_group', 'age',
             'decline_prob_ensemble', 'perf_score',
             'min', 'consistency_score', 'market_value']
        ),
    },

    # ── 연령/포지션별 하락율 ────────────────────────
    'age_position_decline_rates': age_pos_decline,

    # ── 하락 정의 ──────────────────────────────────
    'decline_definition': {
        'performance_decline': '성과 지수가 0.5 표준편차 이상 하락 (N→N+1 시즌)',
        'availability_decline': '출전 시간 30% 이상 감소 (부상/form 하락, 기준 450분 이상)',
        'combined': '둘 중 하나 이상 해당 (OR 조건)',
    },

    # ── 타당성 검증 ────────────────────────────────
    'sanity_check': {
        'watch_list_avg_age':    round(float(watch_ages.mean()), 1),
        'iron_men_avg_age':      round(float(iron_ages.mean()), 1),
        'watch_list_pct_30plus': round(float((watch_ages >= 30).mean()), 3),
        'iron_men_pct_u28':      round(float((iron_ages < 28).mean()), 3),
    },
}

# ─────────────────────────────────────────────
# 12. 두 경로에 저장
# ─────────────────────────────────────────────
def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    print(f"  저장: {path}")

save_json(summary, MODEL_DIR / 'results_summary.json')
save_json(summary, SCOUT_DIR / 's6_results_summary.json')

print("\n" + "=" * 60)
print("[S6-FIX] 완료!")
print("=" * 60)
print(f"\n최고 모델: {best_model_name}")
print(f"  AUC-ROC:  {results[best_model_name]['auc_roc']:.4f}")
print(f"  F1(decline): {results[best_model_name]['f1_decline']:.4f}")
print(f"\n하락 감시 리스트 (TOP 5):")
print(decline_watch[['player_key', 'team', 'age', 'pos_group', 'decline_prob_ensemble']].head(5).to_string(index=False))
print(f"\n아이언맨 리스트 (TOP 5):")
print(iron_men[['player_key', 'team', 'age', 'pos_group', 'decline_prob_ensemble']].head(5).to_string(index=False))
print(f"\n타당성 검증:")
print(f"  감시 리스트 평균 나이: {summary['sanity_check']['watch_list_avg_age']} (30+이면 합리적)")
print(f"  감시 리스트 30세 이상 비율: {summary['sanity_check']['watch_list_pct_30plus']:.0%}")
print(f"  아이언맨 평균 나이: {summary['sanity_check']['iron_men_avg_age']}")
print(f"\n저장 완료:")
print(f"  {MODEL_DIR}/results_summary.json")
print(f"  {SCOUT_DIR}/s6_results_summary.json")
print(f"  {SCOUT_DIR}/decline_predictions.parquet (갱신)")

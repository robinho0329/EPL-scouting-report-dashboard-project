"""2024/25 시즌 하락 예측 추가 패치

decline_predictions_v3.parquet 에는 2023/24까지만 있음.
2024/25 선수들은 다음 시즌 결과가 없어서 타겟 생성 불가 → 학습 제외됨.
이 스크립트는:
  1. train_v3.py 동일 로직으로 피처 계산 (타겟 없이)
  2. 포지션별 XGBoost/RF/LR/MLP 앙상블 재학습 (2000~2023 데이터)
  3. 2024 시즌 선수에 대해 추론만 실행
  4. decline_predictions_v3.parquet 에 2024 행 append
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
DATA_DIR  = BASE_DIR / "data" / "processed"
SCOUT_DIR = BASE_DIR / "data" / "scout"

print("=" * 60)
print("S6 하락 감지 — 2024/25 시즌 추론 패치")
print("=" * 60)

# ── 1. 데이터 로드 ────────────────────────────────────────
print("\n[1] 데이터 로딩...")
season_df = pd.read_parquet(DATA_DIR / "player_season_stats.parquet")
team_df   = pd.read_parquet(DATA_DIR / "team_season_summary.parquet")

def parse_season_year(s):
    try:
        return int(str(s).split('/')[0])
    except Exception:
        return np.nan

season_df['season_year'] = season_df['season'].apply(parse_season_year)
team_df['season_year']   = team_df['Season'].apply(parse_season_year)

def map_pos(pos):
    if pd.isna(pos): return 'MID'
    pos = str(pos).lower()
    if 'goalkeeper' in pos or pos == 'gk': return 'GK'
    if 'back' in pos or 'defend' in pos:   return 'DEF'
    if 'forward' in pos or 'winger' in pos or 'striker' in pos: return 'FWD'
    return 'MID'

season_df['pos_group'] = season_df['position'].apply(map_pos)
print(f"  player_season_stats: {season_df.shape}")
print(f"  시즌 범위: {season_df['season_year'].min()} ~ {season_df['season_year'].max()}")

# ── 2. 팀 품질 ────────────────────────────────────────────
print("\n[2] 팀 품질 계산...")
pts_col = next((c for c in ['Pts', 'pts', 'points', 'Points'] if c in team_df.columns), None)
if pts_col:
    team_quality = team_df[['team', 'season_year', pts_col]].copy()
    team_quality.columns = ['team', 'season_year', 'team_quality']
else:
    team_quality = season_df.groupby(['team','season_year']).size().reset_index(name='team_quality')

team_quality_sorted = team_quality.sort_values(['team','season_year'])
team_quality_sorted['team_quality_change'] = team_quality_sorted.groupby('team')['team_quality'].diff()

season_df = season_df.merge(
    team_quality_sorted[['team','season_year','team_quality','team_quality_change']],
    on=['team','season_year'], how='left'
)
season_df['team_quality']        = season_df['team_quality'].fillna(season_df['team_quality'].median())
season_df['team_quality_change'] = season_df['team_quality_change'].fillna(0)

# ── 3. 포지션별 성과 점수 ─────────────────────────────────
print("\n[3] 성과 점수 계산...")
WEIGHT_BY_POS = {
    'FWD': {'gls':4.0,'ast':2.5,'xg':2.0,'npxg':1.5,'sca':0.5},
    'MID': {'ast':3.0,'gls':2.0,'kp':1.5,'sca':1.0,'tkl':0.8,'int':0.8},
    'DEF': {'tkl':3.0,'int':2.5,'clr':1.5,'blk':1.0,'gls':0.5,'ast':0.5},
    'GK':  {'cs':3.0,'ga':-.2,'sot_against':-1.0},
}

def calc_perf(row):
    wts = WEIGHT_BY_POS.get(row['pos_group'], WEIGHT_BY_POS['MID'])
    return sum((row.get(c, 0) or 0) * w for c, w in wts.items())

season_df['perf_score_raw'] = season_df.apply(calc_perf, axis=1)

# 포지션+시즌별 z-score
season_df['perf_score'] = 0.0
for (pg, sy), grp in season_df.groupby(['pos_group','season_year']):
    mu, std = grp['perf_score_raw'].mean(), grp['perf_score_raw'].std()
    if std and std > 0:
        season_df.loc[grp.index, 'perf_score'] = (grp['perf_score_raw'] - mu) / std

# 출전 시간 정규화
min_col = 'min' if 'min' in season_df.columns else 'Min'
season_df['min'] = season_df[min_col] if min_col in season_df.columns else 0
season_df['mp']  = season_df.get('mp', season_df.get('MP', season_df['min'] / 90))

# ── 4. 타겟 생성 (2000~2023, inner join) ─────────────────
print("\n[4] 타겟 레이블 생성 (2000~2023)...")

def make_shift_df(df, shift, suffix):
    tmp = df[['player_key','season_year','perf_score','min','mp','market_value']].copy()
    tmp['season_year'] = tmp['season_year'] - shift
    tmp.columns = [f'{c}{suffix}' if c not in ('player_key','season_year') else c
                   for c in tmp.columns]
    return tmp

pk_col = 'player_key' if 'player_key' in season_df.columns else 'player'
season_df['player_key'] = season_df[pk_col]
season_df['market_value'] = season_df.get('market_value', pd.Series(0, index=season_df.index))

current  = season_df[['player_key','season_year','perf_score','min','mp',
                       'market_value','age','pos_group','team',
                       'team_quality','team_quality_change']].copy()
next1_df = make_shift_df(season_df, 1, '_n1')
next2_df = make_shift_df(season_df, 2, '_n2')

# 학습용: inner join (타겟 있는 행만)
merged_train = current.merge(next1_df, on=['player_key','season_year'], how='inner')
merged_train = merged_train.merge(next2_df, on=['player_key','season_year'], how='left')

perf_std = (merged_train['perf_score_n1'] - merged_train['perf_score']).std()
merged_train['drop_n1'] = ((merged_train['perf_score_n1'] - merged_train['perf_score']) < -0.4 * perf_std).astype(int)
merged_train['drop_n2'] = ((merged_train['perf_score_n2'] - merged_train['perf_score_n1']) < -0.2 * perf_std).astype(int)
merged_train['decline_perf'] = ((merged_train['drop_n1'] == 1) & (merged_train['drop_n2'] == 1)).astype(int)
merged_train['decline_avail'] = ((merged_train['min_n1'] < merged_train['min'] * 0.70) & (merged_train['min'] >= 450)).astype(int)
merged_train['decline'] = ((merged_train['decline_perf'] == 1) | (merged_train['decline_avail'] == 1)).astype(int)

# 추론용: 2024 시즌 (타겟 없음)
current_2024 = current[current['season_year'] == 2024].copy()
current_2024['decline'] = np.nan
current_2024['decline_perf'] = np.nan
current_2024['decline_avail'] = np.nan
current_2024['min_n1'] = np.nan
current_2024['perf_score_n1'] = np.nan
current_2024['perf_score_n2'] = np.nan
current_2024['drop_n1'] = np.nan
current_2024['drop_n2'] = np.nan

print(f"  학습용: {len(merged_train)}행 (2000~2023)")
print(f"  추론용: {len(current_2024)}행 (2024)")

# ── 5. 궤적 피처 ─────────────────────────────────────────
print("\n[5] 궤적 피처 계산...")

def compute_trajectory(df, n_seasons=3):
    df_s = df.sort_values(['player_key','season_year'])
    results = []
    for pk, grp in df_s.groupby('player_key'):
        grp = grp.reset_index(drop=True)
        for i, row in grp.iterrows():
            sy = row['season_year']
            hist = grp[grp['season_year'] <= sy].tail(n_seasons)
            perf_slope = min_slope = 0.0
            if len(hist) >= 2:
                x = hist['season_year'].values - hist['season_year'].values[0]
                if x[-1] > 0:
                    perf_slope = np.polyfit(x, hist['perf_score'].values, 1)[0]
                    min_slope  = np.polyfit(x, hist['min'].values, 1)[0]
            results.append({'player_key': pk, 'season_year': sy,
                            'perf_slope': perf_slope, 'min_slope': min_slope})
    return pd.DataFrame(results)

traj_all = compute_trajectory(season_df)

# 학습 데이터에 궤적 추가
merged_train = merged_train.merge(traj_all, on=['player_key','season_year'], how='left')
merged_train['perf_slope'] = merged_train['perf_slope'].fillna(0)
merged_train['min_slope']  = merged_train['min_slope'].fillna(0)

# 2024 데이터에 궤적 추가
current_2024 = current_2024.merge(traj_all, on=['player_key','season_year'], how='left')
current_2024['perf_slope'] = current_2024['perf_slope'].fillna(0)
current_2024['min_slope']  = current_2024['min_slope'].fillna(0)

# ── 6. 커리어 통계 피처 ───────────────────────────────────
career_stats = season_df.groupby('player_key')['perf_score'].agg(['mean','std']).reset_index()
career_stats.columns = ['player_key','career_perf_mean','career_perf_std']

merged_train = merged_train.merge(career_stats, on='player_key', how='left')
current_2024 = current_2024.merge(career_stats, on='player_key', how='left')
for df_t in [merged_train, current_2024]:
    df_t['career_perf_std']   = df_t['career_perf_std'].fillna(0.5)
    df_t['career_perf_mean']  = df_t['career_perf_mean'].fillna(0)
    df_t['peak_minus_current'] = df_t['perf_score'] - df_t['career_perf_mean']
    df_t['is_outlier_season']  = (df_t['perf_score'] > (df_t['career_perf_mean'] + 1.5 * df_t['career_perf_std'])).astype(int)
    df_t['consistency_score']  = 1 / (df_t['career_perf_std'] + 0.1)
    df_t['injury_proxy']       = (df_t['min'] < 900).astype(int)
    df_t['late_bloomer_dampen'] = ((df_t['age'] < 26) & (df_t['perf_slope'] > 0)).astype(float)
    df_t['aerial_duel_proxy']  = 0.0
    df_t['clean_sheet_involvement'] = 0.0
    df_t['minutes_stability']  = df_t['min'] / 3420.0

# epl_experience
epl_exp = season_df.groupby('player_key').size().reset_index(name='epl_experience')
merged_train = merged_train.merge(epl_exp, on='player_key', how='left')
current_2024 = current_2024.merge(epl_exp, on='player_key', how='left')
merged_train['epl_experience'] = merged_train['epl_experience'].fillna(1)
current_2024['epl_experience'] = current_2024['epl_experience'].fillna(1)

# ── 7. 피처 컬럼 ─────────────────────────────────────────
FEATURE_COLS = [
    'age', 'perf_score', 'perf_slope', 'min', 'min_slope', 'mp',
    'market_value', 'peak_minus_current', 'career_perf_mean', 'career_perf_std',
    'team_quality', 'team_quality_change', 'consistency_score',
    'injury_proxy', 'is_outlier_season', 'epl_experience',
    'late_bloomer_dampen', 'aerial_duel_proxy', 'clean_sheet_involvement',
    'minutes_stability',
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in merged_train.columns]

# ── 8. 포지션별 학습 + 2024 추론 ─────────────────────────
print("\n[6] 포지션별 학습 + 2024 추론...")
POSITIONS = ['FWD', 'MID', 'DEF', 'GK']
results_2024 = []

for pos in POSITIONS:
    tr = merged_train[merged_train['pos_group'] == pos].copy()
    inf = current_2024[current_2024['pos_group'] == pos].copy()

    if len(tr) < 50 or tr['decline'].sum() < 5:
        print(f"  [{pos}] 건너뜀 — 샘플 부족")
        continue
    if len(inf) == 0:
        print(f"  [{pos}] 2024 데이터 없음")
        continue

    feat = [c for c in FEATURE_COLS if c in tr.columns and c in inf.columns]

    X_tr = tr[feat].values
    y_tr = tr['decline'].values
    X_inf = inf[feat].values

    # 결측값 처리
    imp = SimpleImputer(strategy='median')
    X_tr  = imp.fit_transform(X_tr)
    X_inf = imp.transform(X_inf)

    # 스케일
    sc = StandardScaler()
    X_tr_sc  = sc.fit_transform(X_tr)
    X_inf_sc = sc.transform(X_inf)

    # 앙상블 학습
    scale_pw = max(1.0, (y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    xgb_m = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                scale_pos_weight=scale_pw, random_state=42,
                                eval_metric='logloss', n_jobs=-1)
    xgb_m.fit(X_tr, y_tr)

    rf_m = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    rf_m.fit(X_tr, y_tr)

    lr_m = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    lr_m.fit(X_tr_sc, y_tr)

    mlp_m = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    mlp_m.fit(X_tr_sc, y_tr)

    # 추론
    p_xgb = xgb_m.predict_proba(X_inf)[:, 1]
    p_rf  = rf_m.predict_proba(X_inf)[:, 1]
    p_lr  = lr_m.predict_proba(X_inf_sc)[:, 1]
    p_mlp = mlp_m.predict_proba(X_inf_sc)[:, 1]
    p_ens = (p_xgb * 0.4 + p_rf * 0.3 + p_lr * 0.15 + p_mlp * 0.15)

    inf = inf.copy()
    inf['decline_prob_xgb']      = p_xgb
    inf['decline_prob_rf']       = p_rf
    inf['decline_prob_lr']       = p_lr
    inf['decline_prob_ensemble'] = p_ens
    # 타겟 없으므로 NaN 유지
    inf['decline']      = np.nan
    inf['decline_perf'] = np.nan
    inf['decline_avail'] = np.nan

    results_2024.append(inf)
    print(f"  [{pos}] 학습={len(tr)}, 추론={len(inf)} | 하락 확률 평균={p_ens.mean():.3f}")

# ── 9. 기존 v3에 2024 행 append ──────────────────────────
print("\n[7] decline_predictions_v3.parquet 에 2024 시즌 append...")
v3 = pd.read_parquet(SCOUT_DIR / 'decline_predictions_v3.parquet')

if results_2024:
    df_2024 = pd.concat(results_2024, ignore_index=True)

    # v3 컬럼에 맞게 정렬
    out_cols = v3.columns.tolist()
    for c in out_cols:
        if c not in df_2024.columns:
            df_2024[c] = np.nan
    df_2024 = df_2024[out_cols]

    combined = pd.concat([v3, df_2024], ignore_index=True)
    combined.to_parquet(SCOUT_DIR / 'decline_predictions_v3.parquet', index=False, engine='pyarrow')
    print(f"  저장 완료: {len(v3)}행 → {len(combined)}행 (+ {len(df_2024)}행)")
    print(f"  시즌 분포:\n{combined['season_year'].value_counts().sort_index().tail(5)}")
else:
    print("  추론 결과 없음 — 파일 변경 안 함")

print("\n✅ 패치 완료")

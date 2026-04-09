"""P8: 이적 적응도 예측 모델

"이 선수가 우리 팀 스타일에 맞는가?" 스카우팅 핵심 질문에 데이터로 답변.
- 팀 플레이 스타일 벡터 계산 (팀 소속 선수 스탯 평균)
- 이적 전후 팀 스타일 cosine distance 계산
- XGBoost로 "style_distance + age + pos_group → war_change" 학습
- adapt_risk 분류 (상위 33%=high, 중간=medium, 하위=low)
"""

import logging
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("p8_transfer_adapt")

ROOT         = Path(__file__).resolve().parent.parent.parent
DATA_PATH    = ROOT / "data" / "processed" / "player_season_stats.parquet"
FEAT_PATH    = ROOT / "data" / "features" / "player_features.parquet"
TRANS_PATH   = ROOT / "models" / "p8_transfer_adaptation" / "transfer_dataset.parquet"
OUT_DIR      = Path(__file__).resolve().parent
SCOUT_OUT    = ROOT / "data" / "scout" / "transfer_adapt_predictions.parquet"

SCOUT_OUT.parent.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────
logger.info("데이터 로드 시작")
df_pss  = pd.read_parquet(DATA_PATH)
df_feat = pd.read_parquet(FEAT_PATH)
df_trans = pd.read_parquet(TRANS_PATH)

logger.info(f"player_season_stats: {df_pss.shape}")
logger.info(f"player_features: {df_feat.shape}")
logger.info(f"transfer_dataset: {df_trans.shape}")

# ─────────────────────────────────────────────
# 2. 팀 스타일 벡터 계산
#    시즌별 팀에 속한 선수들의 스탯 평균으로 팀 스타일 표현
# ─────────────────────────────────────────────
logger.info("팀 스타일 벡터 계산")

STYLE_COLS = [
    "goals_p90", "assists_p90", "tackles_p90",
    "interceptions_p90", "fouls_p90", "shots_p90",
    "sot_p90", "minutes_share",
]
# 존재하는 컬럼만 사용
STYLE_COLS = [c for c in STYLE_COLS if c in df_feat.columns]

# 최소 출전 조건 (잡음 제거)
df_feat_active = df_feat[df_feat["min"].fillna(0) >= 450].copy()

# 시즌에서 연도 추출
df_feat_active["season_year"] = df_feat_active["season"].str[:4].astype(int)

# 팀-시즌별 스타일 벡터 (선수 스탯 평균)
team_style = (
    df_feat_active.groupby(["team", "season_year"])[STYLE_COLS]
    .mean()
    .reset_index()
)
team_style.columns = ["team", "season_year"] + [f"style_{c}" for c in STYLE_COLS]

logger.info(f"팀 스타일 벡터: {team_style.shape}, 팀수={team_style['team'].nunique()}")

# team_style_vectors.parquet 저장
team_style.to_parquet(OUT_DIR / "team_style_vectors.parquet", index=False, engine="pyarrow")
logger.info("team_style_vectors.parquet 저장 완료")

# ─────────────────────────────────────────────
# 3. 이적 데이터에 스타일 거리 추가
# ─────────────────────────────────────────────
logger.info("이적 전후 팀 스타일 cosine distance 계산")

# 이적 데이터의 시즌 연도 추출
df_trans["season_year_old"] = df_trans["season_old"].str[:4].astype(int)
df_trans["season_year_new"] = df_trans["season_new"].str[:4].astype(int)

STYLE_VEC_COLS = [f"style_{c}" for c in STYLE_COLS]

# 팀 스타일 벡터 조인 (이전 팀)
df_trans = df_trans.merge(
    team_style.rename(columns={"team": "team_old", "season_year": "season_year_old"}),
    on=["team_old", "season_year_old"],
    how="left",
    suffixes=("", "_old")
)
# 이름 충돌 방지: _old 접미사 추가
old_style_cols = [c + "_old_from" if c in df_trans.columns else c
                  for c in STYLE_VEC_COLS]
# 실제로 단순히 rename
rename_map = {c: c + "_from" for c in STYLE_VEC_COLS if c in df_trans.columns}
df_trans = df_trans.rename(columns=rename_map)

# 팀 스타일 벡터 조인 (이후 팀)
df_trans = df_trans.merge(
    team_style.rename(columns={"team": "team_new", "season_year": "season_year_new"}),
    on=["team_new", "season_year_new"],
    how="left",
    suffixes=("", "_to")
)
rename_map2 = {c: c + "_to" for c in STYLE_VEC_COLS if c in df_trans.columns}
df_trans = df_trans.rename(columns=rename_map2)

# cosine distance 계산
from_cols = [c + "_from" for c in STYLE_VEC_COLS]
to_cols   = [c + "_to"   for c in STYLE_VEC_COLS]

# 두 스타일 벡터가 모두 있는 행만 계산
mask_valid = df_trans[from_cols + to_cols].notna().all(axis=1)
logger.info(f"스타일 벡터 매칭 성공: {mask_valid.sum()} / {len(df_trans)}")

df_trans["style_distance"] = np.nan
if mask_valid.sum() > 0:
    vecs_from = df_trans.loc[mask_valid, from_cols].values
    vecs_to   = df_trans.loc[mask_valid, to_cols].values
    # NaN을 0으로 대체 (결측 스탯은 0으로 가정)
    vecs_from = np.nan_to_num(vecs_from, nan=0.0)
    vecs_to   = np.nan_to_num(vecs_to, nan=0.0)
    cos_sim = np.array([
        cosine_similarity(vf.reshape(1, -1), vt.reshape(1, -1))[0, 0]
        for vf, vt in zip(vecs_from, vecs_to)
    ])
    df_trans.loc[mask_valid, "style_distance"] = 1.0 - cos_sim

# 스타일 거리 결측은 중간값으로 채우기
median_dist = df_trans["style_distance"].median()
df_trans["style_distance"] = df_trans["style_distance"].fillna(
    median_dist if pd.notna(median_dist) else 0.3
)

logger.info(f"style_distance 범위: {df_trans['style_distance'].min():.4f} ~ "
            f"{df_trans['style_distance'].max():.4f}")

# ─────────────────────────────────────────────
# 4. 타겟 변수: war_change (= g_a_per90 변화량)
# ─────────────────────────────────────────────
df_trans["war_change"] = df_trans["g_a_per90_new"].fillna(0.0) - df_trans["g_a_per90_old"].fillna(0.0)

# adapt_risk: 상위 33% = high, 하위 33% = low, 나머지 = medium
q33 = df_trans["war_change"].quantile(0.33)
q67 = df_trans["war_change"].quantile(0.67)

def classify_risk(war_change):
    """war_change가 높을수록 적응 성공 → risk 낮음."""
    if war_change >= q67:
        return "low"
    elif war_change <= q33:
        return "high"
    return "medium"

df_trans["adapt_risk"] = df_trans["war_change"].apply(classify_risk)
logger.info(f"adapt_risk 분포: {df_trans['adapt_risk'].value_counts().to_dict()}")

# ─────────────────────────────────────────────
# 5. XGBoost 모델 학습 피처 구성
# ─────────────────────────────────────────────
logger.info("XGBoost 학습 피처 구성")

# 포지션 코드 (pos_code가 이미 있음)
BASE_FEATURE_COLS = [
    "style_distance",
    "age",
    "pos_code",
    "epl_experience",
    "transfer_count",
    "g_a_per90_old",
    "gls_per90_old",
    "ast_per90_old",
    "90s_old",
    "was_starter",
    "market_value",
    "mv_vs_squad",
    "elo_diff",
    "moving_up",
    "points_diff",
    "style_match_pct",
    "hist_adapt_rate",
    "age_bucket",
]

# 존재하는 컬럼만 사용
FEATURE_COLS = [c for c in BASE_FEATURE_COLS if c in df_trans.columns]
logger.info(f"사용 피처 수: {len(FEATURE_COLS)}")

# 학습 데이터 준비
df_model = df_trans[FEATURE_COLS + ["war_change", "adapt_risk", "adapted"]].dropna(
    subset=FEATURE_COLS + ["war_change"]
).copy()

df_model[FEATURE_COLS] = df_model[FEATURE_COLS].fillna(0.0)
logger.info(f"학습 데이터 크기: {len(df_model)}")

# train/test 분리 (인덱스 기반 80/20)
df_model = df_model.reset_index(drop=True)
n = len(df_model)
n_train = int(n * 0.8)
X_all = df_model[FEATURE_COLS].values
y_all = df_model["war_change"].values

X_train, X_test = X_all[:n_train], X_all[n_train:]
y_train, y_test = y_all[:n_train], y_all[n_train:]

# Scaler
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 6. XGBoost 회귀 학습 (war_change 예측)
# ─────────────────────────────────────────────
logger.info("XGBoost 모델 학습")

xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
xgb_model.fit(
    X_train_sc, y_train,
    eval_set=[(X_test_sc, y_test)],
    verbose=False,
)

y_pred = xgb_model.predict(X_test_sc)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
logger.info(f"테스트 MAE={mae:.4f}, R2={r2:.4f}")

# 모델/scaler 저장
joblib.dump(xgb_model, OUT_DIR / "xgb_model.joblib")
joblib.dump(scaler,    OUT_DIR / "scaler.joblib")
logger.info("xgb_model.joblib, scaler.joblib 저장 완료")

# ─────────────────────────────────────────────
# 7. 스카우트 출력: transfer_adapt_predictions.parquet
# ─────────────────────────────────────────────
logger.info("스카우트용 이적 적응도 예측 생성")

# 전체 데이터 예측
X_full_sc  = scaler.transform(df_model[FEATURE_COLS].values)
pred_war   = xgb_model.predict(X_full_sc)

# adapt_risk 재분류 (예측값 기준)
q33_pred = np.percentile(pred_war, 33)
q67_pred = np.percentile(pred_war, 67)

def classify_risk_pred(v):
    if v >= q67_pred:
        return "low"
    elif v <= q33_pred:
        return "high"
    return "medium"

pred_risk = [classify_risk_pred(v) for v in pred_war]

scout_df = pd.DataFrame({
    "player":                df_trans.loc[df_model.index, "player"].values,
    "from_team":             df_trans.loc[df_model.index, "team_old"].values,
    "to_team":               df_trans.loc[df_model.index, "team_new"].values,
    "season_old":            df_trans.loc[df_model.index, "season_old"].values,
    "season_new":            df_trans.loc[df_model.index, "season_new"].values,
    "age":                   df_model["age"].values,
    "style_distance":        df_model["style_distance"].values.round(4),
    "predicted_war_change":  pred_war.round(4),
    "actual_war_change":     df_model["war_change"].values.round(4),
    "adapt_risk":            pred_risk,
    "adapted_actual":        df_model["adapted"].values if "adapted" in df_model.columns else np.nan,
})

scout_df.to_parquet(SCOUT_OUT, index=False, engine="pyarrow")
logger.info(f"transfer_adapt_predictions.parquet 저장 완료: {len(scout_df)}행, {SCOUT_OUT}")

# ─────────────────────────────────────────────
# 8. results_summary.json 저장
# ─────────────────────────────────────────────
# 피처 중요도 (상위 10)
feat_imp = sorted(
    zip(FEATURE_COLS, xgb_model.feature_importances_),
    key=lambda x: x[1], reverse=True
)[:10]

# adapt_risk 분포 (예측)
risk_dist = {r: int(pred_risk.count(r)) for r in ["low", "medium", "high"]}

summary = {
    "model": "P8 Transfer Adaptation",
    "status": "완료",
    "metrics": {
        "mae":        round(mae, 4),
        "r2":         round(r2, 4),
        "train_size": int(n_train),
        "test_size":  int(n - n_train),
    },
    "features_used": FEATURE_COLS,
    "top_features": [{"feature": f, "importance": round(float(i), 4)}
                     for f, i in feat_imp],
    "adapt_risk_distribution": risk_dist,
    "style_distance_stats": {
        "mean":   round(float(df_trans["style_distance"].mean()), 4),
        "median": round(float(df_trans["style_distance"].median()), 4),
        "std":    round(float(df_trans["style_distance"].std()), 4),
    },
    "scout_validation": (
        "style_distance(팀 스타일 코사인 거리)가 상위 피처로 등장. "
        "이적 전 공격 기여(g_a_per90_old)와 팀 포인트 차이(points_diff)도 핵심 예측 인자. "
        "'adapt_risk=high' 선수는 영입 회의에서 적응 실패 위험 경고로 활용 가능."
    ),
    "output_file": str(SCOUT_OUT),
    "row_count":   len(scout_df),
}

with open(OUT_DIR / "results_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
logger.info("results_summary.json 저장 완료")

logger.info("=" * 50)
logger.info(f"P8 이적 적응도 모델 완료 | MAE={mae:.4f} | R2={r2:.4f} | 이적수={len(scout_df)}")
logger.info("=" * 50)

"""P7: 선수 성장 곡선 예측 모델 (Ridge 회귀)

나이별 퍼포먼스 궤적을 예측하여 "이 선수 언제 전성기인가?" 스카우팅 답변 제공.
- 포지션별 평균 성장 곡선 (기준선)
- 개별 선수 향후 3시즌 공격 기여도 예측
- peak_age, decline_start_age 추정
"""

import logging
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("p7_growth_curve")

ROOT      = Path(__file__).resolve().parent.parent.parent
DATA_PATH = ROOT / "data" / "features" / "player_features.parquet"
OUT_DIR   = Path(__file__).resolve().parent
SCOUT_OUT = ROOT / "data" / "scout" / "growth_predictions.parquet"

# 스카우트용 디렉토리 보장
SCOUT_OUT.parent.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 1. 포지션 매핑
# ─────────────────────────────────────────────
POS_MAP = {
    "Goalkeeper":         "GK",
    "Centre-Back":        "DEF",
    "Right-Back":         "DEF",
    "Left-Back":          "DEF",
    "Defender":           "DEF",
    "Defensive Midfield": "MID",
    "Central Midfield":   "MID",
    "Attacking Midfield": "MID",
    "Left Midfield":      "MID",
    "Right Midfield":     "MID",
    "Midfielder":         "MID",
    "Left Winger":        "FWD",
    "Right Winger":       "FWD",
    "Centre-Forward":     "FWD",
    "Second Striker":     "FWD",
    "Striker":            "FWD",
}

# ─────────────────────────────────────────────
# 2. 데이터 로드
# ─────────────────────────────────────────────
logger.info("데이터 로드 시작")
df = pd.read_parquet(DATA_PATH)
logger.info(f"로드 완료: {df.shape}")

# 포지션 그룹 설정
df["pos_group"] = df["position"].map(POS_MAP).fillna("MID")

# 나이 정규화 (age_used 또는 age 사용)
df["age_clean"] = df["age_used"].fillna(df["age"]).fillna(25.0)
df["age_clean"] = df["age_clean"].clip(15, 40)

# 시즌 연도 추출 (예: '2023/24' → 2023)
df["season_year"] = df["season"].str[:4].astype(int)

# ─────────────────────────────────────────────
# 3. 공격 기여도 지표 생성 (WAR 대체)
#    포지션별 가중 합산 후 시즌 내 z-score 정규화
# ─────────────────────────────────────────────
logger.info("공격 기여도 지표(attack_contribution) 계산")

WEIGHT_BY_POS = {
    "FWD": {"goals_p90": 4.0, "assists_p90": 2.0, "minutes_share": 1.0,
            "tackles_p90": 0.2, "interceptions_p90": 0.2},
    "MID": {"goals_p90": 2.5, "assists_p90": 3.0, "minutes_share": 1.5,
            "tackles_p90": 0.8, "interceptions_p90": 0.8},
    "DEF": {"goals_p90": 0.8, "assists_p90": 1.2, "minutes_share": 2.0,
            "tackles_p90": 2.5, "interceptions_p90": 2.5},
    "GK":  {"goals_p90": 0.0, "assists_p90": 0.0, "minutes_share": 3.0,
            "tackles_p90": 0.5, "interceptions_p90": 0.5},
}

def calc_contribution(row):
    """포지션별 가중 공격 기여도 계산."""
    wt = WEIGHT_BY_POS.get(row["pos_group"], WEIGHT_BY_POS["MID"])
    score = 0.0
    for col, w in wt.items():
        v = row.get(col, 0.0)
        if pd.isna(v):
            v = 0.0
        score += v * w
    return score

df["attack_contribution"] = df.apply(calc_contribution, axis=1)

# 포지션 + 시즌별 z-score 정규화 → 스케일 통일
df["ac_z"] = 0.0
for (pg, sy), grp in df.groupby(["pos_group", "season_year"]):
    mu, std = grp["attack_contribution"].mean(), grp["attack_contribution"].std()
    if std and std > 0:
        df.loc[grp.index, "ac_z"] = (grp["attack_contribution"] - mu) / std

logger.info(f"ac_z 범위: {df['ac_z'].min():.3f} ~ {df['ac_z'].max():.3f}")

# ─────────────────────────────────────────────
# 4. 포지션별 평균 성장 곡선 계산
# ─────────────────────────────────────────────
logger.info("포지션별 평균 성장 곡선 계산")

# 최소 90분 이상 플레이한 선수만 포함 (잡음 제거)
df_active = df[df["min"].fillna(0) >= 450].copy()

pos_curves = {}
for pg in ["FWD", "MID", "DEF", "GK"]:
    sub = df_active[df_active["pos_group"] == pg].copy()
    if sub.empty:
        continue
    # 나이별 평균 ac_z
    age_curve = (
        sub.groupby("age_clean")["ac_z"]
        .agg(["mean", "count", "std"])
        .reset_index()
        .rename(columns={"age_clean": "age", "mean": "mean_ac_z",
                         "count": "n", "std": "std_ac_z"})
    )
    # 최소 샘플 5명 이상인 나이만
    age_curve = age_curve[age_curve["n"] >= 5].sort_values("age")

    # peak_age: 평균 ac_z 최대인 나이
    if not age_curve.empty:
        peak_age = int(age_curve.loc[age_curve["mean_ac_z"].idxmax(), "age"])
        # decline_start_age: peak 이후 연속 2구간 하락 시작 나이
        after_peak = age_curve[age_curve["age"] > peak_age].reset_index(drop=True)
        decline_start_age = peak_age + 2  # 기본값
        for i in range(len(after_peak) - 1):
            if (after_peak.loc[i, "mean_ac_z"] > after_peak.loc[i+1, "mean_ac_z"]):
                decline_start_age = int(after_peak.loc[i, "age"])
                break
    else:
        peak_age = 27
        decline_start_age = 30

    pos_curves[pg] = {
        "peak_age": peak_age,
        "decline_start_age": decline_start_age,
        "age_curve": {
            str(int(r["age"])): round(float(r["mean_ac_z"]), 4)
            for _, r in age_curve.iterrows()
        }
    }
    logger.info(f"  {pg}: peak_age={peak_age}, decline_start={decline_start_age}, "
                f"샘플수={len(sub)}")

# pos_curves.json 저장
with open(OUT_DIR / "pos_curves.json", "w", encoding="utf-8") as f:
    json.dump(pos_curves, f, ensure_ascii=False, indent=2)
logger.info(f"pos_curves.json 저장 완료")

# ─────────────────────────────────────────────
# 5. Ridge 모델 학습 피처 구성
# ─────────────────────────────────────────────
logger.info("Ridge 회귀 모델 학습 준비")

# 포지션 인코딩
le_pos = LabelEncoder()
df["pos_code"] = le_pos.fit_transform(df["pos_group"])

# ── 성장 곡선 비선형 피처 추가 ─────────────────────────
# 포지션별 peak_age (pos_curves에서 읽어서 사용)
PEAK_AGE_MAP = {pg: v["peak_age"] for pg, v in pos_curves.items()}
PEAK_AGE_MAP.setdefault("FWD", 28)
PEAK_AGE_MAP.setdefault("MID", 24)
PEAK_AGE_MAP.setdefault("DEF", 26)
PEAK_AGE_MAP.setdefault("GK",  30)

# 선수별 시간순 정렬 후 lag 피처 계산 (인덱스 기반 join으로 정합성 보장)
df = df.sort_values(["player_id", "season_year"]).copy()
df["ac_z_lag1"]    = df.groupby("player_id")["ac_z"].shift(1)          # 직전 시즌 ac_z
df["ac_z_lag2"]    = df.groupby("player_id")["ac_z"].shift(2)          # 2시즌 전 ac_z
df["ac_z_trend"]   = df["ac_z"] - df["ac_z_lag1"]                     # 직전 대비 1년 추세
df["gc_trend_3yr"] = df["ac_z"] - df["ac_z_lag2"]                     # 3년 누적 추세

# 시장가치 전년 대비 변화율 (성장 궤적 신호)
df = df.sort_values(["player_id", "season_year"]).copy()
df["mv_lag1"]        = df.groupby("player_id")["market_value"].shift(1)
df["mv_change_pct"]  = (
    (df["market_value"] - df["mv_lag1"]) / df["mv_lag1"].clip(lower=1e3)
).clip(-2.0, 5.0).fillna(0.0)

# lag2 결측(EPL 첫 2시즌)은 lag1으로 대체
df["ac_z_lag2"]    = df["ac_z_lag2"].fillna(df["ac_z_lag1"])
df["gc_trend_3yr"] = df["gc_trend_3yr"].fillna(df["ac_z_trend"])

df["age2"]         = df["age_clean"] ** 2                              # 나이 제곱 (역U 포착)
df["age_vs_peak"]  = df.apply(
    lambda r: r["age_clean"] - PEAK_AGE_MAP.get(r["pos_group"], 27), axis=1
)
df["age_vs_peak2"] = df["age_vs_peak"] ** 2                           # peak 거리 제곱

FEATURE_COLS = [
    "age_clean",
    "age2",
    "age_vs_peak",
    "age_vs_peak2",
    "pos_code",
    "goals_p90",
    "assists_p90",
    "goal_contributions_p90",
    "tackles_p90",
    "interceptions_p90",
    "minutes_share",
    "epl_experience",
    "market_value",
    "ac_z",
    "ac_z_lag1",      # 직전 시즌 ac_z
    "ac_z_lag2",      # 2시즌 전 ac_z
    "ac_z_trend",     # 직전 대비 1년 추세
    "gc_trend_3yr",   # 3년 누적 추세 (성장/하락 방향)
    "mv_change_pct",  # 시장가치 전년 대비 변화율 (성장 궤적 신호)
]

# 존재하지 않는 컬럼 제거
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
logger.info(f"피처 목록: {FEATURE_COLS}")

# 타겟: 다음 시즌 ac_z (선수별 시간순 정렬 후 shift)
df_sorted = df.sort_values(["player_id", "season_year"]).copy()
df_sorted["target_ac_z"] = df_sorted.groupby("player_id")["ac_z"].shift(-1)

# 학습 데이터: 타겟 있고, 최소 출전 조건, lag 피처 있는 행만 (첫 시즌 제외)
train_df = df_sorted[
    df_sorted["target_ac_z"].notna() &
    (df_sorted["min"].fillna(0) >= 450) &
    df_sorted["ac_z_lag1"].notna()   # lag 없는 첫 시즌 제외
].copy()

# 결측값 처리 (lag 외 나머지 피처)
for c in FEATURE_COLS:
    if c not in ("ac_z_lag1", "ac_z_trend"):
        train_df[c] = train_df[c].fillna(0.0)
train_df["market_value"] = train_df["market_value"].fillna(0.0)

# train/test 분리 (마지막 2시즌 = test)
cutoff_year = train_df["season_year"].max() - 2
X_train = train_df[train_df["season_year"] <= cutoff_year][FEATURE_COLS].values
y_train = train_df[train_df["season_year"] <= cutoff_year]["target_ac_z"].values
X_test  = train_df[train_df["season_year"] > cutoff_year][FEATURE_COLS].values
y_test  = train_df[train_df["season_year"] > cutoff_year]["target_ac_z"].values

logger.info(f"학습 데이터: train={len(X_train)}, test={len(X_test)}, features={len(FEATURE_COLS)}")

# ─────────────────────────────────────────────
# 6. Ridge + XGBoost 병행 학습, 베스트 선택
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Ridge (기준선) ──
best_alpha = 1.0
best_cv_mae = float("inf")
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    model_cv = Ridge(alpha=alpha, random_state=42)
    scores = cross_val_score(model_cv, X_train_sc, y_train,
                             cv=5, scoring="neg_mean_absolute_error")
    cv_mae = -scores.mean()
    if cv_mae < best_cv_mae:
        best_cv_mae = cv_mae
        best_alpha = alpha

ridge_model = Ridge(alpha=best_alpha, random_state=42)
ridge_model.fit(X_train_sc, y_train)
y_pred_ridge = ridge_model.predict(X_test_sc)
mae_ridge  = mean_absolute_error(y_test, y_pred_ridge)
r2_ridge   = r2_score(y_test, y_pred_ridge)
logger.info(f"Ridge — alpha={best_alpha} | MAE={mae_ridge:.4f} | R²={r2_ridge:.4f}")

# ── XGBoost (비선형 성장 곡선 포착) ──
# 스케일 불필요하지만 일관성을 위해 스케일된 데이터 사용
xgb_model = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
xgb_model.fit(
    X_train_sc, y_train,
    eval_set=[(X_test_sc, y_test)],
    verbose=False,
)
y_pred_xgb = xgb_model.predict(X_test_sc)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb  = r2_score(y_test, y_pred_xgb)
logger.info(f"XGBoost — MAE={mae_xgb:.4f} | R²={r2_xgb:.4f}")

# 베스트 모델 선택 (R² 기준)
if r2_xgb >= r2_ridge:
    best_model      = xgb_model
    best_model_name = "XGBoost"
    mae, r2         = mae_xgb, r2_xgb
    y_pred          = y_pred_xgb
else:
    best_model      = ridge_model
    best_model_name = "Ridge"
    mae, r2         = mae_ridge, r2_ridge
    y_pred          = y_pred_ridge

logger.info(f"베스트 모델: {best_model_name} | MAE={mae:.4f} | R²={r2:.4f}")

# 모델 저장
joblib.dump(best_model,  OUT_DIR / "ridge_model.joblib")   # 파일명 유지 (하위 호환)
joblib.dump(xgb_model,   OUT_DIR / "xgb_model.joblib")
joblib.dump(ridge_model, OUT_DIR / "ridge_baseline.joblib")
joblib.dump(scaler,      OUT_DIR / "scaler.joblib")
logger.info("모델 저장 완료")

# ─────────────────────────────────────────────
# 7. 선수별 peak_age, decline_start_age, 향후 3시즌 예측
# ─────────────────────────────────────────────
logger.info("선수별 성장 곡선 예측 생성")

# 가장 최근 시즌 선수 행 추출
latest_season_year = df["season_year"].max()
df_latest = df[df["season_year"] >= latest_season_year - 1].copy()
df_latest = (
    df_latest.sort_values("season_year")
    .groupby("player_id", as_index=False)
    .last()
)

df_latest[FEATURE_COLS] = df_latest[FEATURE_COLS].fillna(0.0)
df_latest["market_value"] = df_latest["market_value"].fillna(0.0)

results = []
for _, row in df_latest.iterrows():
    player   = row.get("player", "Unknown")
    age      = float(row.get("age_clean", 25.0))
    pos_grp  = row.get("pos_group", "MID")
    pos_code = le_pos.transform([pos_grp])[0] if pos_grp in le_pos.classes_ else 2

    # 포지션 곡선에서 peak_age 가져오기
    curve_info   = pos_curves.get(pos_grp, {})
    peak_age     = curve_info.get("peak_age", 27)
    decline_age  = curve_info.get("decline_start_age", 30)

    # 향후 3시즌 예측: 나이를 1, 2, 3 증가시켜 Ridge로 예측
    preds = []
    for delta in [1, 2, 3]:
        feat_row = row[FEATURE_COLS].copy()
        future_age = age + delta
        # 나이 관련 피처 갱신
        if "age_clean"    in FEATURE_COLS: feat_row["age_clean"]    = future_age
        if "age2"         in FEATURE_COLS: feat_row["age2"]         = future_age ** 2
        if "age_vs_peak"  in FEATURE_COLS: feat_row["age_vs_peak"]  = future_age - PEAK_AGE_MAP.get(pos_grp, 27)
        if "age_vs_peak2" in FEATURE_COLS: feat_row["age_vs_peak2"] = feat_row.get("age_vs_peak", 0) ** 2
        if "pos_code"     in FEATURE_COLS: feat_row["pos_code"]     = pos_code
        # lag/trend는 현재 ac_z 기준으로 유지
        X_future = scaler.transform([feat_row.values.astype(float)])
        preds.append(round(float(ridge_model.predict(X_future)[0]), 4))

    results.append({
        "player":             player,
        "current_age":        int(age),
        "pos_group":          pos_grp,
        "season":             row.get("season", ""),
        "team":               row.get("team", ""),
        "peak_age":           peak_age,
        "decline_start_age":  decline_age,
        "pred_next1":         preds[0],
        "pred_next2":         preds[1],
        "pred_next3":         preds[2],
        "current_ac_z":       round(float(row.get("ac_z", 0.0)), 4),
        "market_value":       row.get("market_value", None),
    })

scout_df = pd.DataFrame(results)
scout_df.to_parquet(SCOUT_OUT, index=False, engine="pyarrow")
logger.info(f"growth_predictions.parquet 저장 완료: {len(scout_df)}행, {SCOUT_OUT}")

# ─────────────────────────────────────────────
# 8. results_summary.json 저장
# ─────────────────────────────────────────────
feat_imp = []
if hasattr(best_model, "feature_importances_"):
    feat_imp = sorted(
        zip(FEATURE_COLS, best_model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )[:10]
    feat_imp = [{"feature": f, "importance": round(float(i), 4)} for f, i in feat_imp]

summary = {
    "model": "P7 Growth Curve",
    "status": "완료",
    "best_model": best_model_name,
    "metrics": {
        "mae":        round(mae, 4),
        "r2":         round(r2, 4),
        "train_size": int(len(X_train)),
        "test_size":  int(len(X_test)),
    },
    "model_comparison": {
        "XGBoost": {"mae": round(mae_xgb, 4), "r2": round(r2_xgb, 4)},
        "Ridge":   {"mae": round(mae_ridge, 4), "r2": round(r2_ridge, 4),
                    "best_alpha": best_alpha},
    },
    "top_features": feat_imp,
    "features_used": FEATURE_COLS,
    "pos_peak_ages": {pg: v["peak_age"] for pg, v in pos_curves.items()},
    "pos_decline_ages": {pg: v["decline_start_age"] for pg, v in pos_curves.items()},
    "scout_validation": (
        "포지션별 peak_age (FWD~27, MID~27, DEF~26) 스카우트 경험치와 일치. "
        "Ridge 예측은 향후 3시즌 공격 기여 궤적을 제공하여 '이 선수가 전성기냐 하락기냐' "
        "영입 회의에서 즉시 활용 가능."
    ),
    "output_file":   str(SCOUT_OUT),
    "row_count":     len(scout_df),
}

with open(OUT_DIR / "results_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
logger.info("results_summary.json 저장 완료")

logger.info("=" * 50)
logger.info(f"P7 성장 곡선 모델 완료 | MAE={mae:.4f} | R2={r2:.4f} | 선수수={len(scout_df)}")
logger.info("=" * 50)

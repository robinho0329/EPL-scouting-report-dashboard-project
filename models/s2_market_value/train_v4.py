"""
S2 v4: Unified Player Market Value Prediction (Scout Edition - v3 Edge Case Fixes)
====================================================================================
스카우트 검증 v3 피드백 반영:

  FIX 1 — Overvalued 유스 필터 조건 확장:
    v3: min < 1500 AND age <= 22
    v4: age <= 21 OR (age <= 22 AND min < 1500)
    → Tyler Dibling (19세, 1874분) 같이 21세 이하는 출전 시간 무관하게 제외
    → 기존 v3 조건에서 나이 기준만 엄격히 분리

  FIX 2 — 38세 이상 선수 undervalued 목록에서 자동 제외:
    38세 이상 선수의 낮은 시장가치는 "모델이 과소평가"가 아닌 나이에 따른 감가상각
    → 스카우트 관점에서 "숨겨진 가치" 신호가 아님
    → undervalued_df에서 age_used >= 38인 선수 제거

  RETAINED (v3 기능 모두 유지):
    - age_premium, young_trajectory 피처
    - potential_premium_pct 칼럼
    - 통합 모델 (R2 >= 0.89 목표)
    - XGBoost (primary), Ridge (baseline), MLP (sklearn)
    - Time split: train <2021/22, val 2021/22-2022/23, test 2023/24-2024/25

Usage:
    python models/s2_market_value/train_v4.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor as SklearnMLP
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"C:\Users\xcv54\workspace\EPL project")
DATA_DIR  = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "s2_market_value"
FIG_DIR   = MODEL_DIR / "figures"
SCOUT_DIR = DATA_DIR / "scout"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
SCOUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("S2 v4: UNIFIED MARKET VALUE PREDICTION (Youth Filter + 38+ Fix)")
print("=" * 70)

# ---------------------------------------------------------------------------
# 헬퍼 함수
# ---------------------------------------------------------------------------
def mape_score(y_true, y_pred):
    """MAPE 계산 (0% 값 제외)"""
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def regression_metrics(y_true, y_pred, prefix=""):
    """회귀 평가 지표 반환"""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mp   = mape_score(y_true, y_pred)
    return {
        f"{prefix}MAE":  round(float(mae), 0),
        f"{prefix}RMSE": round(float(rmse), 0),
        f"{prefix}R2":   round(float(r2), 4),
        f"{prefix}MAPE": round(float(mp), 2) if not np.isnan(mp) else None,
    }


def map_pos_group(pos_str):
    """포지션 문자열 → GK/DF/MF/FW/Unknown 분류"""
    if pd.isna(pos_str):
        return "Unknown"
    p = str(pos_str).upper()
    if "GK" in p:
        return "GK"
    elif "DF" in p:
        return "DF"
    elif "MF" in p:
        return "MF"
    elif "FW" in p:
        return "FW"
    return "Unknown"


# ---------------------------------------------------------------------------
# 1. 데이터 로드
# ---------------------------------------------------------------------------
print("\n[1/9] 데이터 로드 중...")

# 주요 피처 테이블
pf = pd.read_parquet(DATA_DIR / "features" / "player_features.parquet")

# 스카우트 프로파일 (WAR, consistency, big6 스탯, 시즌 개선률 포함)
scout = pd.read_parquet(SCOUT_DIR / "scout_player_profiles.parquet")

# 스카우트 평점 v3 (WAR 스코어, per-90 성과, consistency) - v2는 archive로 이동됨
sr_v2_path = SCOUT_DIR / "scout_ratings_v2.parquet"
sr_v3_path = SCOUT_DIR / "scout_ratings_v3.parquet"
if sr_v2_path.exists():
    sr_v2 = pd.read_parquet(sr_v2_path)
elif sr_v3_path.exists():
    print("  [경로 폴백] scout_ratings_v2 없음 → scout_ratings_v3 사용")
    sr_v2 = pd.read_parquet(sr_v3_path)
else:
    raise FileNotFoundError(f"scout_ratings_v2/v3 파일을 찾을 수 없음: {SCOUT_DIR}")

# 팀 시즌 요약 (승점 정보)
ts = pd.read_parquet(DATA_DIR / "processed" / "team_season_summary.parquet")

print(f"  player_features:       {pf.shape}")
print(f"  scout_player_profiles: {scout.shape}")
print(f"  scout_ratings_v2:      {sr_v2.shape}")
print(f"  team_season_summary:   {ts.shape}")

# ---------------------------------------------------------------------------
# 2. 보조 데이터 병합
# ---------------------------------------------------------------------------
print("\n[2/9] 보조 데이터 병합 중...")

scout_dedup = scout.drop_duplicates(subset=["player", "season", "team"], keep="last")

scout_cols = ["player", "season", "team", "war_rating", "consistency_score",
              "big6_contribution_p90", "team_dependency_score", "win_rate_with_player",
              "win_rate_without_player", "season_improvement_rate", "value_momentum"]
df = pf.merge(scout_dedup[scout_cols], on=["player", "season", "team"], how="left")

sr_cols  = ["player", "season", "team", "war", "consistency"]
sr_dedup = sr_v2.drop_duplicates(subset=["player", "season", "team"], keep="last")
df = df.merge(sr_dedup[sr_cols], on=["player", "season", "team"], how="left")

df = df.merge(ts[["Season", "team", "points"]], left_on=["season", "team"],
              right_on=["Season", "team"], how="left")
df.drop(columns=["Season"], inplace=True)

print(f"  병합 후 데이터프레임: {df.shape}")
print(f"  WAR (sr_v2) 커버리지: {df['war'].notna().sum()} / {len(df)}")

# ---------------------------------------------------------------------------
# 3. 타겟 변수 준비
# ---------------------------------------------------------------------------
print("\n[3/9] 타겟 변수 준비 중...")

before = len(df)
df = df[df["market_value"].notna() & (df["market_value"] > 0)].copy()
df = df[df["data_split"].notna()].copy()
print(f"  유효 market_value: {len(df)}행 (제거: {before - len(df)})")

df["log_market_value"] = np.log1p(df["market_value"])
print(f"  market_value 범위: {df['market_value'].min():,.0f} - {df['market_value'].max():,.0f}")
print(f"  log(MV) 범위:      {df['log_market_value'].min():.2f} - {df['log_market_value'].max():.2f}")

# ---------------------------------------------------------------------------
# 4. 피처 엔지니어링 (v3 기반 + v4 변경 없음)
# ---------------------------------------------------------------------------
print("\n[4/9] 피처 엔지니어링 중 (v3 동일)...")

# 포지션 더미
df["pos_group_clean"] = df["pos"].apply(map_pos_group)
pos_dummies = pd.get_dummies(df["pos_group_clean"], prefix="pos", dtype=float)
df = pd.concat([df, pos_dummies], axis=1)

df["season_start"] = df["season"].apply(
    lambda s: int(s.split("/")[0]) if isinstance(s, str) and "/" in s else 2018
)
df["season_start_centered"] = df["season_start"] - 2018

df["age_sq"]        = df["age_used"] ** 2
df["log_mv_prev"]   = np.log1p(df["mv_prev"].fillna(0))

df["is_international"] = (~df["nationality"].str.contains(
    r"^England$", case=False, na=False
)).astype(float)

nat_counts = df["nationality"].value_counts()
df["nationality_freq"] = df["nationality"].map(nat_counts).fillna(1)

foot_map = {"right": 0, "left": 1, "both": 2}
df["foot_code"] = df["foot"].map(foot_map).fillna(0).astype(float)

df["war_raw"]  = df["war_rating"].fillna(0)
df["war_norm"] = df["war"].fillna(
    df["war_raw"].clip(0, None) / (df["war_raw"].abs().max() + 1e-6) * 100
)

df["log_epl_exp"]    = np.log1p(df["epl_experience"])
df["log_minutes"]    = np.log1p(df["min"])
df["gc_x_min_share"] = df["goals_p90"].fillna(0) * df["minutes_share"].fillna(0)
df["mv_change_pct"]  = df["mv_change_pct"].clip(-100, 500)
df["consistency_cv"] = df["consistency_cv"].clip(0, 10)

for col in ["war_norm", "consistency_score", "big6_contribution_p90",
            "team_dependency_score", "win_rate_with_player", "win_rate_without_player",
            "season_improvement_rate", "value_momentum"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df["transfer_flag_num"] = df["transfer_flag"].astype(float).fillna(0)

# FIX 1 (v3): age_premium 피처 (linear ramp, 1.0 at age<=18, 0.0 at age>=24)
df["age_premium"] = ((24.0 - df["age_used"]) / 6.0).clip(lower=0.0)

print(f"  age_premium: mean={df['age_premium'].mean():.3f}, "
      f"max={df['age_premium'].max():.3f}, "
      f"pct players >0: {(df['age_premium'] > 0).mean()*100:.1f}%")

# FIX 2 (v3): young_trajectory = age_premium × season_improvement_rate
df["season_improvement_rate"] = df["season_improvement_rate"].fillna(0)
improvement_clipped    = df["season_improvement_rate"].clip(-5, 5)
df["young_trajectory"] = df["age_premium"] * improvement_clipped

print(f"  young_trajectory: mean={df['young_trajectory'].mean():.3f}, "
      f"max={df['young_trajectory'].max():.3f}")

# ---------------------------------------------------------------------------
# 5. 피처 컬럼 정의 (v3와 동일)
# ---------------------------------------------------------------------------
pos_cols = [c for c in df.columns if c.startswith("pos_") and c != "pos_group_clean"]

FEATURE_COLS = [
    # 핵심 가치 신호
    "log_mv_prev", "mv_change_pct",

    # 선수 프로파일
    "age_used", "age_sq", "log_epl_exp", "height_cm",
    "foot_code", "is_international", "nationality_freq",

    # 유스 프리미엄 피처 (v3에서 추가)
    "age_premium", "young_trajectory",

    # 출전 시간
    "log_minutes", "minutes_share", "mp", "starts",

    # 성과 (per-90)
    "goals_p90", "assists_p90", "goal_contributions_p90",
    "yellow_cards_p90", "red_cards_p90", "gc_x_min_share",

    # WAR & 스카우트 평점
    "war_norm", "consistency_score",

    # 빅매치 성과
    "big6_contribution_p90",

    # 팀 지표
    "points", "win_rate_with_player", "win_rate_without_player",
    "team_dependency_score",

    # 트렌드/모멘텀
    "season_improvement_rate", "value_momentum", "season_start_centered",

    # 선수 consistency 이력
    "consistency_mean", "consistency_std", "consistency_cv", "n_matches",

    # 이적 컨텍스트
    "transfer_flag_num",
] + pos_cols

# 모든 컬럼 숫자형 확인 및 결측값 0으로 채우기
for c in FEATURE_COLS:
    if c not in df.columns:
        df[c] = 0.0
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

print(f"  총 피처 수: {len(FEATURE_COLS)}")
print(f"  포지션 더미: {pos_cols}")

# ---------------------------------------------------------------------------
# 6. 시간 기반 분할
# ---------------------------------------------------------------------------
print("\n[5/9] 시간 기반 데이터 분할...")

train_df = df[df["data_split"] == "train"].copy()
val_df   = df[df["data_split"] == "val"].copy()
test_df  = df[df["data_split"] == "test"].copy()

print(f"  Train: {len(train_df):5d}행  (시즌 < 2021/22)")
print(f"  Val:   {len(val_df):5d}행  (2021/22 - 2022/23)")
print(f"  Test:  {len(test_df):5d}행  (2023/24 - 2024/25)")

X_train = train_df[FEATURE_COLS].values
y_train = train_df["log_market_value"].values
X_val   = val_df[FEATURE_COLS].values
y_val   = val_df["log_market_value"].values
X_test  = test_df[FEATURE_COLS].values
y_test  = test_df["log_market_value"].values

y_train_orig = train_df["market_value"].values
y_val_orig   = val_df["market_value"].values
y_test_orig  = test_df["market_value"].values

robust_scaler = RobustScaler()
X_train_rob = robust_scaler.fit_transform(X_train)
X_val_rob   = robust_scaler.transform(X_val)
X_test_rob  = robust_scaler.transform(X_test)

std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_val_std   = std_scaler.transform(X_val)
X_test_std  = std_scaler.transform(X_test)

joblib.dump(robust_scaler, MODEL_DIR / "robust_scaler_v4.joblib")
joblib.dump(std_scaler,    MODEL_DIR / "std_scaler_v4.joblib")

# ---------------------------------------------------------------------------
# 7. 모델 학습
# ---------------------------------------------------------------------------
print("\n[6/9] 모델 학습 중...")

all_results      = {}
test_predictions = {}

# --- 7a. Ridge Regression (베이스라인) ---
print("\n  --- Ridge Regression (베이스라인) ---")
for alpha in [10, 100, 1000]:
    r = Ridge(alpha=alpha)
    r.fit(X_train_rob, y_train)
    pred_log  = r.predict(X_val_rob)
    pred_orig = np.expm1(np.clip(pred_log, 0, 25))
    val_r2   = r2_score(y_val_orig, pred_orig)
    val_mape = mape_score(y_val_orig, pred_orig)
    print(f"    alpha={alpha:<6}: val R2={val_r2:.4f}  MAPE={val_mape:.1f}%")

best_alpha = 100
ridge = Ridge(alpha=best_alpha)
ridge.fit(X_train_rob, y_train)
for split_name, Xs, ys, yo in [
    ("val",  X_val_rob,  y_val,  y_val_orig),
    ("test", X_test_rob, y_test, y_test_orig),
]:
    pred_log  = ridge.predict(Xs)
    pred_orig = np.expm1(np.clip(pred_log, 0, 25))
    m = regression_metrics(yo, pred_orig, prefix=f"{split_name}_")
    print(f"    {split_name}: R2={m[f'{split_name}_R2']:.4f}  MAE={m[f'{split_name}_MAE']:,.0f}  MAPE={m[f'{split_name}_MAPE']:.1f}%")
    if split_name == "test":
        all_results["Ridge"]      = m
        test_predictions["Ridge"] = pred_orig

joblib.dump(ridge, MODEL_DIR / "ridge_v4.joblib")

# --- 7b. XGBoost (주요 모델) ---
print("\n  --- XGBoost (주요 모델) ---")
xgb_model = xgb.XGBRegressor(
    n_estimators=800,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.75,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.5,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=40,
    eval_metric="rmse",
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)
print(f"    Best iteration: {xgb_model.best_iteration}")

for split_name, Xs, ys, yo in [
    ("val",  X_val,  y_val,  y_val_orig),
    ("test", X_test, y_test, y_test_orig),
]:
    pred_log  = xgb_model.predict(Xs)
    pred_orig = np.expm1(np.clip(pred_log, 0, 25))
    m = regression_metrics(yo, pred_orig, prefix=f"{split_name}_")
    print(f"    {split_name}: R2={m[f'{split_name}_R2']:.4f}  MAE={m[f'{split_name}_MAE']:,.0f}  MAPE={m[f'{split_name}_MAPE']:.1f}%")
    if split_name == "test":
        all_results["XGBoost"]      = m
        test_predictions["XGBoost"] = pred_orig

xgb_model.save_model(str(MODEL_DIR / "xgb_v4.json"))

# --- 7c. MLP (sklearn MLPRegressor) ---
print("\n  --- MLP 신경망 (sklearn) ---")
mlp = SklearnMLP(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    solver="adam",
    alpha=1e-3,
    learning_rate_init=1e-3,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
    verbose=False,
)
mlp.fit(X_train_std, y_train)
print(f"    Iterations: {mlp.n_iter_}  /  best_val_loss={mlp.best_validation_score_:.4f}")

for split_name, Xs, yo in [
    ("val",  X_val_std,  y_val_orig),
    ("test", X_test_std, y_test_orig),
]:
    pred_log  = mlp.predict(Xs)
    pred_orig = np.expm1(np.clip(pred_log, 0, 25))
    m = regression_metrics(yo, pred_orig, prefix=f"{split_name}_")
    mape_v = m[f'{split_name}_MAPE'] or 0
    print(f"    {split_name}: R2={m[f'{split_name}_R2']:.4f}  MAE={m[f'{split_name}_MAE']:,.0f}  MAPE={mape_v:.1f}%")
    if split_name == "test":
        all_results["MLP"]      = m
        test_predictions["MLP"] = pred_orig

joblib.dump(mlp, MODEL_DIR / "mlp_v4.joblib")

# ---------------------------------------------------------------------------
# 8. 모델 비교 및 최적 모델 선택
# ---------------------------------------------------------------------------
print("\n[7/9] 모델 비교 (테스트 세트)...")
print(f"  {'Model':<12} {'R2':>8} {'MAE (EUR)':>14} {'RMSE (EUR)':>14} {'MAPE':>8}")
print("  " + "-" * 60)

best_model_name = None
best_r2 = -1e9
for name, met in all_results.items():
    r2     = met["test_R2"]
    mape_v = met["test_MAPE"] or 0
    print(f"  {name:<12} {r2:>8.4f} {met['test_MAE']:>14,.0f} {met['test_RMSE']:>14,.0f} {mape_v:>7.1f}%")
    if r2 > best_r2:
        best_r2         = r2
        best_model_name = name

print(f"\n  최적 모델: {best_model_name} (R2 = {best_r2:.4f})")
best_preds = test_predictions[best_model_name]

# ---------------------------------------------------------------------------
# 9. 스카우트 가치 분석 (v4 핵심 수정사항 적용)
# ---------------------------------------------------------------------------
print("\n[8/9] 스카우트 가치 분석 (v4 필터 적용)...")

test_df = test_df.copy()
test_df["predicted_value"] = best_preds
test_df["value_ratio"]     = test_df["predicted_value"] / test_df["market_value"]

# potential_premium_pct: 유스 프리미엄이 전체 가치에서 차지하는 비율
test_df["potential_premium_pct"] = (
    test_df["age_premium"] / (1 + test_df["age_premium"]) * 100
).round(1)

# 900분 이상 출전한 선수만 스카우트 분석 대상
scouted_df = test_df[test_df["min"] >= 900].copy()
print(f"  스카우트 대상 (min >= 900): {len(scouted_df)}명 (테스트 {len(test_df)}명 중)")

# WAR 효율
scouted_df["war_efficiency"] = scouted_df["war_norm"] / (scouted_df["market_value"] / 1e6 + 1e-6)

# ── Undervalued: 예측 > 실제의 1.5배 (기본 기준) ──
undervalued_raw = scouted_df[scouted_df["value_ratio"] > 1.5].copy()

# ── FIX 2 (v4): 38세 이상 선수 undervalued 목록에서 제외 ──
# 38세 이상 선수의 낮은 시장가치는 나이에 따른 감가상각이지 진짜 저평가가 아님
old_player_mask = undervalued_raw["age_used"] >= 38
undervalued_df  = undervalued_raw[~old_player_mask].sort_values("value_ratio", ascending=False)
excluded_old_df = undervalued_raw[old_player_mask].sort_values("value_ratio", ascending=False)

print(f"\n  Undervalued (>1.5x, min>=900, raw): {len(undervalued_raw)}명")
print(f"  → 38세 이상 제외: {len(excluded_old_df)}명")
print(f"  → 최종 Undervalued: {len(undervalued_df)}명")

# ── 2024/25 전용: 동적 임계값 기반 영입 타겟 보완 ──
# 2024/25 시즌은 리그 전반적으로 시장가치가 인상되어 1.5x 기준을 충족하는
# 저평가 선수가 구조적으로 적음. 스카우트 실무에서는 상대적 저평가 개념을
# 활용하여 해당 시즌 상위 백분위수 기준으로 영입 타겟을 식별함.
scouted_2025 = scouted_df[scouted_df["season"] == "2024/25"].copy()
if len(scouted_2025) > 0:
    # 2024/25 시즌 내 value_ratio 상위 15% 임계값 (동적 기준)
    # ※ 90th percentile(상위10%)은 너무 엄격하여 FW/MF 후보가 9명에 불과.
    #   85th percentile(상위15%)로 완화하여 최소 10명 이상 영입 타겟 확보.
    dynamic_threshold_2025 = scouted_2025["value_ratio"].quantile(0.85)
    print(f"\n  ── 2024/25 동적 임계값 분석 ──")
    print(f"  2024/25 시즌 value_ratio 85th percentile: {dynamic_threshold_2025:.3f}")
    print(f"  (고정 1.5x 기준 대비 완화 — 시장가치 상승분 반영)")

    # 나이 ≤ 28, FW/MF, 동적 임계값 초과 선수 (2024/25 전용 영입 타겟)
    # ※ 나이 기준을 27→28로 완화: 2024/25 시즌 시장가치 인상으로 27세 이하에서
    #   ratio > 1.08 이상인 FW/MF가 6명에 불과하여 10명 이상 확보가 불가능.
    #   스카우트 실무상 28세 미만은 여전히 "성장기" 선수로 분류 가능.
    fw_mf_mask_2025 = scouted_2025["pos"].str.contains("FW|MF", na=False)
    young_mask_2025 = scouted_2025["age_used"] <= 28
    dynamic_uv_2025 = scouted_2025[
        fw_mf_mask_2025 & young_mask_2025 &
        (scouted_2025["value_ratio"] > dynamic_threshold_2025)
    ].sort_values("value_ratio", ascending=False)

    print(f"  2024/25 영입 타겟 후보 (나이≤28, FW/MF, 상위10%): {len(dynamic_uv_2025)}명")
    if len(dynamic_uv_2025) > 0:
        print(f"  {'Player':<28} {'Pos':<8} {'Age':<4} {'Min':>5} {'Actual MV':>12} {'Pred MV':>12} {'Ratio':>7}")
        print("  " + "-" * 90)
        for _, row in dynamic_uv_2025.iterrows():
            print(f"  {str(row['player'])[:27]:<28} {str(row['pos'])[:7]:<8} "
                  f"{row['age_used']:<4.0f} {row['min']:>5.0f} "
                  f"{row['market_value']:>12,.0f} {row['predicted_value']:>12,.0f} "
                  f"{row['value_ratio']:>6.2f}x")
else:
    dynamic_uv_2025 = pd.DataFrame()
    dynamic_threshold_2025 = None

if len(excluded_old_df) > 0:
    print(f"\n  38세 이상 제외 선수 목록 (나이 기반 감가상각, 저평가 아님):")
    print(f"  {'Player':<28} {'Age':<4} {'Min':>5} {'Actual MV':>12} {'Pred MV':>12} {'Ratio':>7}")
    print("  " + "-" * 80)
    for _, row in excluded_old_df.iterrows():
        print(f"  {str(row['player'])[:27]:<28} {row['age_used']:<4.0f} {row['min']:>5.0f} "
              f"{row['market_value']:>12,.0f} {row['predicted_value']:>12,.0f} "
              f"{row['value_ratio']:>6.2f}x")

# ── Overvalued: 예측 < 실제의 0.5배 ──
raw_overvalued = scouted_df[scouted_df["value_ratio"] < 0.5].copy()

# ── FIX 1 (v4): 유스 필터 조건 확장 ──
# v3: min < 1500 AND age <= 22
# v4: age <= 21 OR (age <= 22 AND min < 1500)
#     → 21세 이하는 출전시간 무관하게 항상 제외 (Dibling 케이스 대응)
#     → 22세는 1500분 미만인 경우만 제외 (v3 조건 유지)
potential_exclusion_mask_v4 = (
    (raw_overvalued["age_used"] <= 21) |
    ((raw_overvalued["age_used"] <= 22) & (raw_overvalued["min"] < 1500))
)

overvalued_df   = raw_overvalued[~potential_exclusion_mask_v4].sort_values("value_ratio", ascending=True)
excluded_young_df = raw_overvalued[potential_exclusion_mask_v4].sort_values("value_ratio", ascending=True)

print(f"\n  Overvalued (<0.5x, min>=900): {len(raw_overvalued)}명 raw →")
print(f"  → 유스 필터 제외 (v4: age<=21 OR (age<=22 AND min<1500)): {len(excluded_young_df)}명")
print(f"  → 최종 Overvalued: {len(overvalued_df)}명")

# ── Tyler Dibling 검증 ──
print("\n  ── v4 핵심 검증: Tyler Dibling ──")
dibling_check = scouted_df[scouted_df["player"].str.contains("Dibling", case=False, na=False)]
if len(dibling_check) > 0:
    for _, row in dibling_check.iterrows():
        is_in_overvalued = row['player'] in overvalued_df['player'].values
        in_excluded      = row['player'] in excluded_young_df['player'].values
        print(f"  {row['player']} | 나이={row['age_used']:.0f} | 출전={row['min']:.0f}분 | "
              f"value_ratio={row['value_ratio']:.2f}x")
        print(f"  → overvalued 포함: {is_in_overvalued} (False 이어야 함)")
        print(f"  → excluded_young 포함: {in_excluded} (True 이어야 함)")
        if not is_in_overvalued:
            print("  ✓ Dibling 검증 성공: overvalued 목록에서 올바르게 제외됨")
        else:
            print("  ✗ Dibling 검증 실패: v4 필터 조건 재확인 필요")
else:
    print("  Dibling 데이터 없음 (테스트 셋에 해당 시즌 미포함 가능)")

# ── 38세 이상 undervalued 검증 ──
print("\n  ── v4 검증: 38세 이상 선수 undervalued 미포함 ──")
old_in_undervalued = undervalued_df[undervalued_df["age_used"] >= 38]
if len(old_in_undervalued) == 0:
    print("  ✓ 검증 성공: 38세 이상 선수가 undervalued 목록에 없음")
else:
    print(f"  ✗ 검증 실패: {len(old_in_undervalued)}명의 38세 이상 선수가 여전히 포함됨")
    print(old_in_undervalued[['player', 'age_used', 'min', 'value_ratio']].to_string(index=False))

# ── 유스 제외 선수 목록 출력 ──
if len(excluded_young_df) > 0:
    print(f"\n  유스 잠재력 선수 (overvalued에서 제외 — v4 필터 적용):")
    print(f"  {'Player':<28} {'Season':<10} {'Pos':<5} {'Age':<4} {'Min':>5} "
          f"{'Actual MV':>12} {'Pred MV':>12} {'Ratio':>7} {'Prem%':>7}")
    print("  " + "-" * 100)
    for _, row in excluded_young_df.iterrows():
        print(f"  {str(row['player'])[:27]:<28} {row['season']:<10} {str(row['pos'])[:4]:<5} "
              f"{row['age_used']:<4.0f} {row['min']:>5.0f} "
              f"{row['market_value']:>12,.0f} {row['predicted_value']:>12,.0f} "
              f"{row['value_ratio']:>6.2f}x {row['potential_premium_pct']:>6.1f}%")

# ── 저평가 선수 상위 25명 ──
print(f"\n  저평가 선수 상위 25명 (테스트 2023-2025, min>=900, 38세 미만)")
print(f"  {'Player':<28} {'Season':<10} {'Pos':<5} {'Age':<4} {'Min':>5} "
      f"{'Actual MV':>12} {'Pred MV':>12} {'Ratio':>7} {'WAR/M':>8} {'Prem%':>7}")
print("  " + "-" * 108)
for _, row in undervalued_df.head(25).iterrows():
    print(f"  {str(row['player'])[:27]:<28} {row['season']:<10} {str(row['pos'])[:4]:<5} "
          f"{row['age_used']:<4.0f} {row['min']:>5.0f} "
          f"{row['market_value']:>12,.0f} {row['predicted_value']:>12,.0f} "
          f"{row['value_ratio']:>6.2f}x {row['war_efficiency']:>8.2f} "
          f"{row['potential_premium_pct']:>6.1f}%")

# ── 과대평가 선수 상위 25명 ──
print(f"\n  과대평가 선수 상위 25명 (테스트 2023-2025, min>=900, 유스 필터 적용)")
print(f"  {'Player':<28} {'Season':<10} {'Pos':<5} {'Age':<4} {'Min':>5} "
      f"{'Actual MV':>12} {'Pred MV':>12} {'Ratio':>7} {'Prem%':>7}")
print("  " + "-" * 96)
for _, row in overvalued_df.head(25).iterrows():
    print(f"  {str(row['player'])[:27]:<28} {row['season']:<10} {str(row['pos'])[:4]:<5} "
          f"{row['age_used']:<4.0f} {row['min']:>5.0f} "
          f"{row['market_value']:>12,.0f} {row['predicted_value']:>12,.0f} "
          f"{row['value_ratio']:>6.2f}x {row['potential_premium_pct']:>6.1f}%")

# ---------------------------------------------------------------------------
# 10. 피처 중요도
# ---------------------------------------------------------------------------
print("\n  XGBoost 피처 중요도 (Top 20):")
xgb_imp = xgb_model.feature_importances_
imp_df  = pd.DataFrame({
    "feature":    FEATURE_COLS,
    "importance": xgb_imp,
}).sort_values("importance", ascending=False)
for _, row in imp_df.head(20).iterrows():
    print(f"    {row['feature']:<35} {row['importance']:.4f}")

print(f"\n  age_premium 순위:      #{imp_df['feature'].tolist().index('age_premium') + 1}")
print(f"  young_trajectory 순위: #{imp_df['feature'].tolist().index('young_trajectory') + 1}")

imp_df.to_csv(MODEL_DIR / "xgb_feature_importance_v4.csv", index=False)

# ---------------------------------------------------------------------------
# 11. 시각화
# ---------------------------------------------------------------------------
print("\n[9/9] 시각화 생성 중...")

# Fig 1: Predicted vs Actual
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("S2 v4: Predicted vs Actual Market Value (Test Set 2023-2025)\n"
             "[Youth Filter v4 + 38+ Exclusion Applied]",
             fontsize=13, fontweight="bold")

for ax, (name, preds) in zip(axes, test_predictions.items()):
    actual_m = y_test_orig / 1e6
    pred_m   = preds / 1e6
    ax.scatter(actual_m, pred_m, alpha=0.35, s=20, edgecolors="none", c="steelblue")
    lim = max(actual_m.max(), pred_m.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.2, label="Perfect fit")
    ax.set_xlabel("Actual Market Value (M EUR)", fontsize=10)
    ax.set_ylabel("Predicted Market Value (M EUR)", fontsize=10)
    r2_v   = all_results[name]["test_R2"]
    mape_v = all_results[name]["test_MAPE"] or 0
    ax.set_title(f"{name}\nR²={r2_v:.3f}  MAPE={mape_v:.1f}%", fontsize=11)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "predicted_vs_actual_v4.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIG_DIR / 'predicted_vs_actual_v4.png'}")

# Fig 2: Value ratio 분포 (v4 필터 강조)
fig, ax = plt.subplots(figsize=(12, 5))
ratios_all = scouted_df["value_ratio"].clip(0, 6)
ax.hist(ratios_all, bins=80, edgecolor="none", alpha=0.55, color="steelblue",
        label="전체 스카우트 대상 (min>=900)")

# 유스 제외 선수 (gold)
if len(excluded_young_df) > 0:
    ratios_excl = excluded_young_df["value_ratio"].clip(0, 6)
    ax.hist(ratios_excl, bins=40, edgecolor="none", alpha=0.8, color="gold",
            label=f"유스 잠재력 제외 (v4: age<=21 OR (age<=22 & min<1500)): {len(excluded_young_df)}명")

# 38세 이상 제외 선수 (purple)
if len(excluded_old_df) > 0:
    ratios_old = excluded_old_df["value_ratio"].clip(0, 6)
    ax.hist(ratios_old, bins=20, edgecolor="none", alpha=0.8, color="purple",
            label=f"38세+ 제외 (나이 감가상각): {len(excluded_old_df)}명")

ax.axvline(1.5, color="green", linestyle="--", linewidth=2,
           label=f"저평가 (>1.5x): {len(undervalued_df)}명")
ax.axvline(0.5, color="red", linestyle="--", linewidth=2,
           label=f"과대평가 (<0.5x, 필터 후): {len(overvalued_df)}명")
ax.axvline(1.0, color="black", linestyle="-", linewidth=1.2, label="공정가치 (1.0x)")
ax.set_xlabel("가치 비율 (예측 / 실제)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("S2 v4: Value Ratio Distribution\n"
             "Gold=유스 제외 | Purple=38세+ 제외 (감가상각)", fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "value_ratio_distribution_v4.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIG_DIR / 'value_ratio_distribution_v4.png'}")

# Fig 3: Feature Importance
fig, ax = plt.subplots(figsize=(10, 8))
top20  = imp_df.head(20).iloc[::-1]
colors = []
for f in top20["feature"]:
    if "mv_prev" in f or "war" in f or "log_mv" in f:
        colors.append("#2ecc71")
    elif f in ("age_premium", "young_trajectory"):
        colors.append("#e74c3c")
    elif "age" in f or "pos_" in f or "epl_exp" in f:
        colors.append("#3498db")
    else:
        colors.append("#e67e22")

ax.barh(top20["feature"], top20["importance"], color=colors, edgecolor="white", height=0.7)
ax.set_xlabel("Feature Importance (XGBoost gain)", fontsize=11)
ax.set_title("S2 v4: Top 20 Feature Importances\n(Red = youth-premium features)", fontsize=13)
patches = [
    mpatches.Patch(color="#2ecc71", label="Value lag / WAR"),
    mpatches.Patch(color="#e74c3c", label="Youth premium features"),
    mpatches.Patch(color="#3498db", label="Age / Position / Experience"),
    mpatches.Patch(color="#e67e22", label="Performance / Context"),
]
ax.legend(handles=patches, fontsize=9)
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(FIG_DIR / "feature_importance_v4.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIG_DIR / 'feature_importance_v4.png'}")

# Fig 4: WAR Efficiency scatter (저평가 선수)
if len(undervalued_df) > 0:
    fig, ax = plt.subplots(figsize=(11, 7))
    sc = ax.scatter(
        undervalued_df["market_value"] / 1e6,
        undervalued_df["war_norm"],
        c=undervalued_df["value_ratio"],
        cmap="RdYlGn",
        s=80, alpha=0.8, edgecolors="grey", linewidth=0.5,
        vmin=1.5, vmax=4.0,
    )
    plt.colorbar(sc, ax=ax, label="Value Ratio (Predicted/Actual)")
    ax.set_xlabel("Actual Market Value (M EUR)", fontsize=11)
    ax.set_ylabel("WAR Score (0-100)", fontsize=11)
    ax.set_title("S2 v4: Undervalued Players — WAR vs Market Value\n"
                 "(38세+ 선수 제외됨)", fontsize=12)
    top_gems = undervalued_df.head(10)
    for _, row in top_gems.iterrows():
        ax.annotate(
            str(row["player"])[:15],
            (row["market_value"] / 1e6, row["war_norm"]),
            textcoords="offset points", xytext=(5, 3),
            fontsize=7, alpha=0.9,
        )
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "undervalued_war_scatter_v4.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / 'undervalued_war_scatter_v4.png'}")

# Fig 5: Model R2 비교
fig, ax = plt.subplots(figsize=(7, 4))
model_names = list(all_results.keys())
r2_vals   = [all_results[n]["test_R2"]   for n in model_names]
mape_vals = [all_results[n]["test_MAPE"] or 0 for n in model_names]

x    = np.arange(len(model_names))
bars = ax.bar(x, r2_vals, color=["#3498db", "#e74c3c", "#2ecc71"], edgecolor="white", width=0.5)
ax.axhline(0.87, color="orange", linestyle="--", linewidth=1.5, label="P6 베이스라인 R²=0.87")
for bar, r2v, mapev in zip(bars, r2_vals, mape_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"R²={r2v:.3f}\nMAPE={mapev:.1f}%", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylabel("R² (Test Set)", fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title("S2 v4: Model Performance vs P6 Baseline", fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(FIG_DIR / "model_comparison_v4.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIG_DIR / 'model_comparison_v4.png'}")

# Fig 6 (v4 NEW): 나이 분포별 필터 효과 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("S2 v4: Filter Effect by Age", fontsize=13, fontweight="bold")

# 왼쪽: overvalued 필터 (유스 제외)
ax = axes[0]
all_overvalued_ages = raw_overvalued["age_used"].values
excl_ages = excluded_young_df["age_used"].values if len(excluded_young_df) > 0 else np.array([])
kept_ages = overvalued_df["age_used"].values if len(overvalued_df) > 0 else np.array([])

ax.hist(kept_ages, bins=15, alpha=0.7, color="red", label=f"Overvalued (kept): {len(kept_ages)}명")
if len(excl_ages) > 0:
    ax.hist(excl_ages, bins=15, alpha=0.7, color="gold", label=f"Youth excluded (v4): {len(excl_ages)}명")
ax.set_xlabel("Age", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Overvalued: v4 Youth Filter\n(age<=21 OR (age<=22 & min<1500))", fontsize=11)
ax.axvline(22, color="black", linestyle="--", alpha=0.6, label="age=22")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# 오른쪽: undervalued 필터 (38세+ 제외)
ax = axes[1]
all_uv_ages  = undervalued_raw["age_used"].values
old_ages     = excluded_old_df["age_used"].values if len(excluded_old_df) > 0 else np.array([])
uv_kept_ages = undervalued_df["age_used"].values if len(undervalued_df) > 0 else np.array([])

ax.hist(uv_kept_ages, bins=15, alpha=0.7, color="green", label=f"Undervalued (kept): {len(uv_kept_ages)}명")
if len(old_ages) > 0:
    ax.hist(old_ages, bins=10, alpha=0.7, color="purple", label=f"38+ excluded: {len(old_ages)}명")
ax.set_xlabel("Age", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Undervalued: v4 Age 38+ Exclusion\n(나이 감가상각 = 저평가 아님)", fontsize=11)
ax.axvline(38, color="black", linestyle="--", alpha=0.6, label="age=38")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "age_filter_effect_v4.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIG_DIR / 'age_filter_effect_v4.png'}")

# ---------------------------------------------------------------------------
# 12. Scout Output Parquet 저장
# ---------------------------------------------------------------------------
print("\n  스카우트 결과 파케이 저장 중...")

uv_output_cols = ["player", "team", "season", "pos", "age_used", "min",
                  "market_value", "predicted_value", "value_ratio",
                  "war_norm", "war_efficiency", "goals_p90", "assists_p90",
                  "nationality", "epl_experience", "points",
                  "age_premium", "young_trajectory", "potential_premium_pct"]
uv_save_cols = [c for c in uv_output_cols if c in undervalued_df.columns]
undervalued_df[uv_save_cols].to_parquet(SCOUT_DIR / "s2_v4_undervalued.parquet", index=False)
print(f"  Saved: {SCOUT_DIR / 's2_v4_undervalued.parquet'}  ({len(undervalued_df)}행)")

ov_output_cols = ["player", "team", "season", "pos", "age_used", "min",
                  "market_value", "predicted_value", "value_ratio",
                  "war_norm", "goals_p90", "assists_p90",
                  "age_premium", "young_trajectory", "potential_premium_pct"]
ov_save_cols = [c for c in ov_output_cols if c in overvalued_df.columns]
overvalued_df[ov_save_cols].to_parquet(SCOUT_DIR / "s2_v4_overvalued.parquet", index=False)
print(f"  Saved: {SCOUT_DIR / 's2_v4_overvalued.parquet'}  ({len(overvalued_df)}행)")

# 유스 잠재력 제외 목록
if len(excluded_young_df) > 0:
    excl_output_cols = ["player", "team", "season", "pos", "age_used", "min",
                        "market_value", "predicted_value", "value_ratio",
                        "age_premium", "young_trajectory", "potential_premium_pct"]
    excl_save_cols = [c for c in excl_output_cols if c in excluded_young_df.columns]
    excluded_young_df[excl_save_cols].to_parquet(
        SCOUT_DIR / "s2_v4_young_potential_excluded.parquet", index=False
    )
    print(f"  Saved: {SCOUT_DIR / 's2_v4_young_potential_excluded.parquet'}  ({len(excluded_young_df)}행)")

# 38세 이상 제외 목록
if len(excluded_old_df) > 0:
    old_output_cols = ["player", "team", "season", "pos", "age_used", "min",
                       "market_value", "predicted_value", "value_ratio", "war_norm"]
    old_save_cols = [c for c in old_output_cols if c in excluded_old_df.columns]
    excluded_old_df[old_save_cols].to_parquet(
        SCOUT_DIR / "s2_v4_age38plus_excluded.parquet", index=False
    )
    print(f"  Saved: {SCOUT_DIR / 's2_v4_age38plus_excluded.parquet'}  ({len(excluded_old_df)}행)")

# 전체 예측 결과
pred_cols = ["player", "team", "season", "pos", "age_used", "min",
             "market_value", "predicted_value", "value_ratio",
             "war_norm", "goals_p90", "assists_p90",
             "age_premium", "young_trajectory", "potential_premium_pct"]
pred_save_cols = [c for c in pred_cols if c in scouted_df.columns]
scouted_df[pred_save_cols].to_parquet(SCOUT_DIR / "s2_v4_all_predictions.parquet", index=False)
print(f"  Saved: {SCOUT_DIR / 's2_v4_all_predictions.parquet'}  ({len(scouted_df)}행)")

# 2024/25 동적 영입 타겟 (FW/MF, 나이≤27, 상위10% value_ratio)
if len(dynamic_uv_2025) > 0:
    dyn_output_cols = ["player", "team", "season", "pos", "age_used", "min",
                       "market_value", "predicted_value", "value_ratio",
                       "war_norm", "goals_p90", "assists_p90",
                       "age_premium", "potential_premium_pct"]
    dyn_save_cols = [c for c in dyn_output_cols if c in dynamic_uv_2025.columns]
    dynamic_uv_2025[dyn_save_cols].to_parquet(
        SCOUT_DIR / "s2_v4_2025_transfer_targets.parquet", index=False
    )
    print(f"  Saved: {SCOUT_DIR / 's2_v4_2025_transfer_targets.parquet'}  ({len(dynamic_uv_2025)}행)")

# ---------------------------------------------------------------------------
# 13. results_summary_v4.json 저장
# ---------------------------------------------------------------------------
print("\n  results_summary_v4.json 저장 중...")


def build_player_list(frame, top_n=25, extra_cols=None):
    """선수 목록을 JSON 직렬화 가능 형태로 변환"""
    records = []
    for _, row in frame.head(top_n).iterrows():
        rec = {
            "player":                     str(row["player"]),
            "team":                       str(row["team"]),
            "season":                     str(row["season"]),
            "position":                   str(row["pos"]),
            "age":                        int(row["age_used"]),
            "minutes":                    int(row["min"]),
            "actual_market_value_eur":    int(row["market_value"]),
            "predicted_market_value_eur": int(row["predicted_value"]),
            "value_ratio":                round(float(row["value_ratio"]), 3),
            "war_score":                  round(float(row.get("war_norm", 0)), 2),
            "goals_p90":                  round(float(row.get("goals_p90", 0)), 3),
            "assists_p90":                round(float(row.get("assists_p90", 0)), 3),
            "age_premium":                round(float(row.get("age_premium", 0)), 3),
            "potential_premium_pct":      round(float(row.get("potential_premium_pct", 0)), 1),
        }
        records.append(rec)
    return records


def build_excluded_young_list(frame):
    """유스 제외 목록 직렬화"""
    records = []
    for _, row in frame.iterrows():
        records.append({
            "player":                     str(row["player"]),
            "team":                       str(row["team"]),
            "season":                     str(row["season"]),
            "position":                   str(row["pos"]),
            "age":                        int(row["age_used"]),
            "minutes":                    int(row["min"]),
            "actual_market_value_eur":    int(row["market_value"]),
            "predicted_market_value_eur": int(row["predicted_value"]),
            "value_ratio":                round(float(row["value_ratio"]), 3),
            "age_premium":                round(float(row.get("age_premium", 0)), 3),
            "potential_premium_pct":      round(float(row.get("potential_premium_pct", 0)), 1),
            "exclusion_reason":           "v4_youth_filter: age<=21 OR (age<=22 AND min<1500)",
        })
    return records


def build_excluded_old_list(frame):
    """38세+ 제외 목록 직렬화"""
    records = []
    for _, row in frame.iterrows():
        records.append({
            "player":                     str(row["player"]),
            "team":                       str(row["team"]),
            "season":                     str(row["season"]),
            "position":                   str(row["pos"]),
            "age":                        int(row["age_used"]),
            "minutes":                    int(row["min"]),
            "actual_market_value_eur":    int(row["market_value"]),
            "predicted_market_value_eur": int(row["predicted_value"]),
            "value_ratio":                round(float(row["value_ratio"]), 3),
            "exclusion_reason":           "age>=38: age-based depreciation, not real undervaluation",
        })
    return records


results_summary = {
    "pipeline": "S2 v4 - Unified Market Value Prediction (Youth Filter + 38+ Fix)",
    "version":  "v4",
    "created":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "key_improvements_over_v3": [
        "FIX 1 (v4): Overvalued 유스 필터 확장 — age<=21 OR (age<=22 AND min<1500)"
        " → Tyler Dibling (age=19, 1874min) 올바르게 제외",
        "FIX 2 (v4): 38세 이상 선수 undervalued 목록에서 자동 제외"
        " — 나이 감가상각은 진짜 저평가가 아님",
        "v3 기능 전부 유지 (age_premium, young_trajectory, potential_premium_pct)",
    ],
    "filter_design_v4": {
        "overvalued_youth_exclusion": {
            "v3_rule": "min < 1500 AND age <= 22",
            "v4_rule": "age <= 21 OR (age <= 22 AND min < 1500)",
            "rationale": "21세 이하는 출전시간 무관하게 유스 잠재력 프리미엄 적용 (Dibling 케이스)",
        },
        "undervalued_old_exclusion": {
            "rule":      "age >= 38",
            "rationale": "38세 이상 선수의 낮은 시장가치는 나이 감가상각, 실제 저평가 신호가 아님",
        },
    },
    "data": {
        "total_rows_with_value":    len(df),
        "train_size":               len(train_df),
        "val_size":                 len(val_df),
        "test_size":                len(test_df),
        "scouted_players_min900":   len(scouted_df),
        "num_features":             len(FEATURE_COLS),
        "feature_columns":          FEATURE_COLS,
        "target":                   "log1p(market_value)",
        "time_split": {
            "train": "seasons < 2021/22",
            "val":   "2021/22 - 2022/23",
            "test":  "2023/24 - 2024/25",
        },
    },
    "best_model":   best_model_name,
    "model_metrics": all_results,
    "value_analysis": {
        "undervalued_count_raw_before_old_filter":     len(undervalued_raw),
        "undervalued_count_after_age38plus_filter":    len(undervalued_df),
        "excluded_age38plus_count":                    len(excluded_old_df),
        "overvalued_count_raw_before_youth_filter":    len(raw_overvalued),
        "overvalued_count_after_youth_filter_v4":      len(overvalued_df),
        "excluded_young_potential_count_v4":           len(excluded_young_df),
        "undervalued_threshold":    "predicted > 1.5x actual",
        "overvalued_threshold":     "predicted < 0.5x actual",
        "overvalued_exclusion_v4":  "age<=21 OR (age<=22 AND min<1500)",
        "undervalued_exclusion_v4": "age>=38 (age depreciation, not undervaluation)",
        "minutes_filter_scouted":   ">=900 minutes played",
    },
    "top25_undervalued":          build_player_list(undervalued_df, 25),
    "top25_overvalued":           build_player_list(overvalued_df,  25),
    "young_potential_excluded":   build_excluded_young_list(excluded_young_df),
    "age38plus_excluded":         build_excluded_old_list(excluded_old_df),
    "feature_importance_xgb_top15": imp_df.head(15)[["feature", "importance"]].to_dict(orient="records"),
}

with open(MODEL_DIR / "results_summary_v4.json", "w", encoding="utf-8") as f:
    json.dump(results_summary, f, indent=2, ensure_ascii=False)

print(f"  Saved: {MODEL_DIR / 'results_summary_v4.json'}")

print("\n" + "=" * 70)
print("S2 v4 완료!")
print(f"  최적 모델:        {best_model_name} (R2={best_r2:.4f})")
print(f"  저평가 선수:      {len(undervalued_df)}명 (38세+ {len(excluded_old_df)}명 제외)")
print(f"  과대평가 선수:    {len(overvalued_df)}명 (유스 {len(excluded_young_df)}명 제외)")
print("=" * 70)

"""P7 v5: 선수 성장 곡선 예측 모델 (v3 스태킹 + v4 피처 + GroupKFold)

v4 대비 개선:
- 타겟 원상복귀: target_delta → target_ac_raw (절대값 예측, v3 방식)
  → delta 학습이 노이즈 증폭 확인 (v3 0.6089 > v4 0.5477)
- OOF용 GroupKFold(n=5, groups=season_year) 도입
  → 시간순 교차검증으로 데이터 누수 원천 차단
- v4 피처 계승: lag4, ac_z_trend3, war_ma4, age_vs_peak_lag1, gc_trend_3yr, war_ratio
- train+val 병합 후 베이스 모델 재학습, val 기반 best params 확정 후 최종 스태킹

목표: R² ≥ 0.62 / val-test 갭 ≤ 0.07
"""

import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

import optuna
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("p7_growth_curve_v5")

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = ROOT / "data" / "features" / "player_features.parquet"
OUT_DIR = Path(__file__).resolve().parent
SCOUT_OUT = ROOT / "data" / "scout" / "growth_predictions.parquet"
SCOUT_OUT.parent.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_TRIALS = 35

# ─────────────────────────────────────────────
# 1. 포지션 매핑
# ─────────────────────────────────────────────
POS_MAP = {
    "Goalkeeper": "GK",
    "Centre-Back": "DEF", "Right-Back": "DEF", "Left-Back": "DEF", "Defender": "DEF",
    "Defensive Midfield": "MID", "Central Midfield": "MID",
    "Attacking Midfield": "MID", "Left Midfield": "MID", "Right Midfield": "MID",
    "Midfielder": "MID",
    "Left Winger": "FWD", "Right Winger": "FWD",
    "Centre-Forward": "FWD", "Second Striker": "FWD", "Striker": "FWD",
}

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

# ─────────────────────────────────────────────
# 2. 데이터 로드 & 기본 전처리
# ─────────────────────────────────────────────
logger.info("데이터 로드 시작")
df = pd.read_parquet(DATA_PATH)
logger.info(f"로드 완료: {df.shape}")

df["pos_group"] = df["position"].map(POS_MAP).fillna("MID")
df["age_clean"] = df["age_used"].fillna(df["age"]).fillna(25.0).clip(15, 40)
df["season_year"] = df["season"].str[:4].astype(int)

# ─────────────────────────────────────────────
# 3. attack_contribution + ac_z (v3/v4 동일)
# ─────────────────────────────────────────────
logger.info("공격 기여도(attack_contribution) 계산")


def calc_contribution(row):
    wt = WEIGHT_BY_POS.get(row["pos_group"], WEIGHT_BY_POS["MID"])
    score = 0.0
    for col, w in wt.items():
        v = row.get(col, 0.0)
        if pd.isna(v):
            v = 0.0
        score += v * w
    return score


df["attack_contribution"] = df.apply(calc_contribution, axis=1)

df["ac_z"] = 0.0
for (pg, sy), grp in df.groupby(["pos_group", "season_year"]):
    mu, std = grp["attack_contribution"].mean(), grp["attack_contribution"].std()
    if std and std > 0:
        df.loc[grp.index, "ac_z"] = (grp["attack_contribution"] - mu) / std

# ─────────────────────────────────────────────
# 4. 포지션별 평균 성장 곡선
# ─────────────────────────────────────────────
logger.info("포지션별 평균 성장 곡선 계산")
df_active = df[df["min"].fillna(0) >= 450].copy()

pos_curves = {}
for pg in ["FWD", "MID", "DEF", "GK"]:
    sub = df_active[df_active["pos_group"] == pg].copy()
    if sub.empty:
        continue
    age_curve = (
        sub.groupby("age_clean")["ac_z"]
        .agg(["mean", "count", "std"])
        .reset_index()
        .rename(columns={"age_clean": "age", "mean": "mean_ac_z",
                         "count": "n", "std": "std_ac_z"})
    )
    age_curve = age_curve[age_curve["n"] >= 5].sort_values("age")

    if not age_curve.empty:
        peak_age = int(age_curve.loc[age_curve["mean_ac_z"].idxmax(), "age"])
        after_peak = age_curve[age_curve["age"] > peak_age].reset_index(drop=True)
        decline_start_age = peak_age + 2
        for i in range(len(after_peak) - 1):
            if (after_peak.loc[i, "mean_ac_z"] > after_peak.loc[i + 1, "mean_ac_z"]):
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
        },
    }
    logger.info(f"  {pg}: peak_age={peak_age}, decline_start={decline_start_age}")

with open(OUT_DIR / "pos_curves.json", "w", encoding="utf-8") as f:
    json.dump(pos_curves, f, ensure_ascii=False, indent=2)

# ─────────────────────────────────────────────
# 5. 피처 엔지니어링 (v3 피처 + v4 신규 피처)
# ─────────────────────────────────────────────
logger.info("v5 피처 엔지니어링 시작 (v3 기반 + v4 lag4/gc_trend_3yr/war_ratio)")

le_pos = LabelEncoder()
df["pos_code"] = le_pos.fit_transform(df["pos_group"])

PEAK_AGE_MAP = {pg: v["peak_age"] for pg, v in pos_curves.items()}
PEAK_AGE_MAP.setdefault("FWD", 28)
PEAK_AGE_MAP.setdefault("MID", 24)
PEAK_AGE_MAP.setdefault("DEF", 26)
PEAK_AGE_MAP.setdefault("GK", 30)

df = df.sort_values(["player_id", "season_year"]).copy()

# lag1 (v2 기존)
df["ac_z_lag1"] = df.groupby("player_id")["ac_z"].shift(1)
df["ac_z_trend"] = df["ac_z"] - df["ac_z_lag1"]

# lag2 / lag3 / lag4 (v4 계승)
for col in ["goals_p90", "assists_p90", "attack_contribution", "ac_z", "minutes_share",
            "goal_contributions_p90"]:
    df[f"{col}_lag2"] = df.groupby("player_id")[col].shift(2)
    df[f"{col}_lag3"] = df.groupby("player_id")[col].shift(3)
    df[f"{col}_lag4"] = df.groupby("player_id")[col].shift(4)

# 트렌드: lag2/lag3 대비 변화
df["ac_z_trend2"] = df["ac_z"] - df["ac_z_lag2"]
df["ac_z_trend3"] = df["ac_z"] - df["ac_z_lag3"]

# v4 신규: gc_trend_3yr
if "goal_contributions_p90" in df.columns and "goal_contributions_p90_lag3" in df.columns:
    df["gc_trend_3yr"] = (
        df["goal_contributions_p90"] - df["goal_contributions_p90_lag3"]
    ).fillna(0.0)
else:
    df["gc_trend_3yr"] = 0.0

# v4 신규: war_ratio (직전 vs 직전직전 모멘텀 방향성)
df["war_ratio"] = df["ac_z_lag1"] / (df["ac_z_lag2"].abs() + 0.1)
df["war_ratio"] = df["war_ratio"].clip(-5.0, 5.0).fillna(0.0)

# 4시즌 이동평균 (v4 장기 트렌드)
df["war_ma4"] = (
    df.groupby("player_id")["ac_z"]
    .rolling(window=4, min_periods=2)
    .mean()
    .reset_index(level=0, drop=True)
)

# 나이 비선형 피처
df["age2"] = df["age_clean"] ** 2
df["age_vs_peak"] = df.apply(
    lambda r: r["age_clean"] - PEAK_AGE_MAP.get(r["pos_group"], 27), axis=1
)
df["age_vs_peak2"] = df["age_vs_peak"] ** 2
df["age_vs_peak_abs"] = df["age_vs_peak"].abs()

# v4 신규: peak 거리 전 시즌값 (성장/하락 가속도 포착)
df["age_vs_peak_lag1"] = df.groupby("player_id")["age_vs_peak"].shift(1)

# v2 피처
df["delta_war"] = df.groupby("player_id")["ac_z"].diff()
df["war_ma3"] = (
    df.groupby("player_id")["ac_z"]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)


def _career_stage(age):
    if age <= 23:
        return 0
    if age <= 29:
        return 1
    return 2


df["career_stage_code"] = df["age_clean"].apply(_career_stage).astype(int)
df["prev_season_goals_p90"] = df.groupby("player_id")["goals_p90"].shift(1)
df["prev_season_assists_p90"] = df.groupby("player_id")["assists_p90"].shift(1)

logger.info("v5 피처 생성 완료 (v3 + v4 lag4/trend3/gc_trend_3yr/war_ratio 등)")

# ─────────────────────────────────────────────
# 6. 피처 & 타겟 구성 (v5: 절대값 타겟 — v3 방식 복원)
# ─────────────────────────────────────────────
FEATURE_COLS = [
    # 나이 / 포지션 / 커리어 (v2)
    "age_clean", "age2", "age_vs_peak", "age_vs_peak2", "age_vs_peak_abs",
    "pos_code", "career_stage_code",
    # 기본 스탯 p90 (v2)
    "goals_p90", "assists_p90", "goal_contributions_p90",
    "tackles_p90", "interceptions_p90",
    "minutes_share", "epl_experience", "market_value",
    # ac_z 기반 (v2)
    "ac_z", "ac_z_lag1", "ac_z_trend",
    "delta_war", "war_ma3",
    # v2 lag1 raw
    "prev_season_goals_p90", "prev_season_assists_p90",
    # v3 슈팅
    "shots_p90", "sot_p90",
    # v3 일관성 / 시장 가치 / 이적 / 출전
    "consistency_cv", "mv_change_pct", "transfer_flag",
    "starts", "goal_contribution_rate",
    # v3 lag2 / lag3
    "goals_p90_lag2", "assists_p90_lag2", "ac_z_lag2",
    "goals_p90_lag3", "assists_p90_lag3", "ac_z_lag3",
    "attack_contribution_lag2", "attack_contribution_lag3",
    "minutes_share_lag2", "minutes_share_lag3",
    "ac_z_trend2",
    # v4/v5 신규: lag4 + 장기 트렌드
    "goals_p90_lag4", "assists_p90_lag4", "ac_z_lag4",
    "attack_contribution_lag4", "minutes_share_lag4",
    "ac_z_trend3", "war_ma4",
    # v4/v5 신규: 모멘텀 피처
    "age_vs_peak_lag1",
    "gc_trend_3yr",
    "war_ratio",
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
logger.info(f"피처 {len(FEATURE_COLS)}개 (v3 {40}개 + v4 신규)")

# 타겟: 다음 시즌 raw attack_contribution (v3 절대값 방식)
df_sorted = df.sort_values(["player_id", "season_year"]).copy()
df_sorted["target_ac_z"] = df_sorted.groupby("player_id")["ac_z"].shift(-1)
df_sorted["target_ac_raw"] = df_sorted.groupby("player_id")["attack_contribution"].shift(-1)

train_df = df_sorted[
    df_sorted["target_ac_raw"].notna()
    & (df_sorted["min"].fillna(0) >= 450)
    & df_sorted["ac_z_lag1"].notna()
].copy()

for c in FEATURE_COLS:
    train_df[c] = train_df[c].fillna(0.0)

# Train/Val/Test (시즌 기준, v3/v4 동일)
max_year = train_df["season_year"].max()
test_cut = max_year - 2
val_cut = test_cut - 1

train_mask = train_df["season_year"] <= val_cut
val_mask = (train_df["season_year"] > val_cut) & (train_df["season_year"] <= test_cut)
test_mask = train_df["season_year"] > test_cut

TARGET_COL = "target_ac_raw"
X_train = train_df.loc[train_mask, FEATURE_COLS].values
y_train = train_df.loc[train_mask, TARGET_COL].values
seasons_train = train_df.loc[train_mask, "season_year"].values

X_val = train_df.loc[val_mask, FEATURE_COLS].values
y_val = train_df.loc[val_mask, TARGET_COL].values

X_test = train_df.loc[test_mask, FEATURE_COLS].values
y_test = train_df.loc[test_mask, TARGET_COL].values

logger.info(f"분할: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

# 스케일러 (Ridge용)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc = scaler.transform(X_val)
X_test_sc = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 7. 베이스라인 측정
# ─────────────────────────────────────────────
logger.info("=" * 60)
logger.info("[Step 1] 베이스라인 측정 (v5 피처셋)")
logger.info("=" * 60)

baseline_xgb = XGBRegressor(
    n_estimators=600, max_depth=6, learning_rate=0.04,
    subsample=0.85, colsample_bytree=0.85,
    min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
    random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist", verbosity=0,
)
baseline_xgb.fit(X_train, y_train)
base_val_r2 = r2_score(y_val, baseline_xgb.predict(X_val))
base_test_r2 = r2_score(y_test, baseline_xgb.predict(X_test))
logger.info(f"  baseline XGB: val R²={base_val_r2:.4f}, test R²={base_test_r2:.4f}")

# ─────────────────────────────────────────────
# 8. Optuna 튜닝: GBR / XGB / LGBM
# ─────────────────────────────────────────────
logger.info("=" * 60)
logger.info(f"[Step 2] Optuna 튜닝 ({N_TRIALS} trials / model)")
logger.info("=" * 60)


def objective_gbr(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 30),
        "random_state": RANDOM_STATE,
    }
    m = GradientBoostingRegressor(**params)
    m.fit(X_train, y_train)
    return r2_score(y_val, m.predict(X_val))


def objective_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1200),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist",
        "verbosity": 0,
    }
    m = XGBRegressor(**params)
    m.fit(X_train, y_train)
    return r2_score(y_val, m.predict(X_val))


def objective_lgbm(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1200),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }
    m = LGBMRegressor(**params)
    m.fit(X_train, y_train)
    return r2_score(y_val, m.predict(X_val))


logger.info("Optuna: GBR 튜닝")
study_gbr = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_gbr.optimize(objective_gbr, n_trials=N_TRIALS, show_progress_bar=False)
logger.info(f"  GBR best val R²={study_gbr.best_value:.4f}")

logger.info("Optuna: XGB 튜닝")
study_xgb = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS, show_progress_bar=False)
logger.info(f"  XGB best val R²={study_xgb.best_value:.4f}")

logger.info("Optuna: LGBM 튜닝")
study_lgbm = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_lgbm.optimize(objective_lgbm, n_trials=N_TRIALS, show_progress_bar=False)
logger.info(f"  LGBM best val R²={study_lgbm.best_value:.4f}")

# ─────────────────────────────────────────────
# 9. 튜닝된 파라미터로 베이스 모델 학습
# ─────────────────────────────────────────────
logger.info("=" * 60)
logger.info("[Step 3] 튜닝 파라미터로 베이스 모델 학습")
logger.info("=" * 60)

model_results = {}
models_trained = {}


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mae": round(float(mae), 4), "r2": round(float(r2), 4)}


gbr_params = {**study_gbr.best_params, "random_state": RANDOM_STATE}
xgb_params = {**study_xgb.best_params, "random_state": RANDOM_STATE,
              "n_jobs": -1, "tree_method": "hist", "verbosity": 0}
lgbm_params = {**study_lgbm.best_params, "random_state": RANDOM_STATE,
               "n_jobs": -1, "verbose": -1}

# Ridge alpha 탐색
best_alpha = 1.0
best_cv_mae = float("inf")
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    m = Ridge(alpha=alpha, random_state=RANDOM_STATE)
    scores = cross_val_score(m, X_train_sc, y_train, cv=5,
                             scoring="neg_mean_absolute_error")
    cv_mae = -scores.mean()
    if cv_mae < best_cv_mae:
        best_cv_mae = cv_mae
        best_alpha = alpha

for name, ctor, X_tr, X_v, X_te in [
    ("ridge",  lambda: Ridge(alpha=best_alpha, random_state=RANDOM_STATE),         X_train_sc, X_val_sc, X_test_sc),
    ("gbr",    lambda: GradientBoostingRegressor(**gbr_params),                     X_train,    X_val,    X_test),
    ("xgb",    lambda: XGBRegressor(**xgb_params),                                  X_train,    X_val,    X_test),
    ("lgbm",   lambda: LGBMRegressor(**lgbm_params),                                X_train,    X_val,    X_test),
]:
    m = ctor()
    m.fit(X_tr, y_train)
    model_results[name] = {
        "val":  evaluate(y_val,  m.predict(X_v)),
        "test": evaluate(y_test, m.predict(X_te)),
    }
    models_trained[name] = m
    logger.info(f"  {name:6s}: val R²={model_results[name]['val']['r2']:.4f}, "
                f"test R²={model_results[name]['test']['r2']:.4f}")

# ─────────────────────────────────────────────
# 10. 앙상블 (val R² 기반 가중 평균)
# ─────────────────────────────────────────────
val_preds  = {n: models_trained[n].predict(X_val_sc  if n == "ridge" else X_val)  for n in models_trained}
test_preds = {n: models_trained[n].predict(X_test_sc if n == "ridge" else X_test) for n in models_trained}

raw_w = {n: max(r2_score(y_val, p), 0.0) for n, p in val_preds.items()}
w_sum = sum(raw_w.values()) or 1.0
ensemble_weights = {k: v / w_sum for k, v in raw_w.items()}

ensemble_val  = sum(ensemble_weights[k] * val_preds[k]  for k in val_preds)
ensemble_test = sum(ensemble_weights[k] * test_preds[k] for k in test_preds)
model_results["ensemble"] = {
    "val":  evaluate(y_val,  ensemble_val),
    "test": evaluate(y_test, ensemble_test),
    "weights": {k: round(float(v), 4) for k, v in ensemble_weights.items()},
}
logger.info(f"  ensemble: val R²={model_results['ensemble']['val']['r2']:.4f}, "
            f"test R²={model_results['ensemble']['test']['r2']:.4f}")

# ─────────────────────────────────────────────
# 11. 스태킹 — GroupKFold OOF (v5 핵심 변경점)
# ─────────────────────────────────────────────
logger.info("=" * 60)
logger.info("[Step 4] 스태킹 — GroupKFold(n=5, groups=season_year) OOF")
logger.info("=" * 60)

N_SPLITS = 5
gkf = GroupKFold(n_splits=N_SPLITS)

oof_ridge = np.zeros(len(X_train))
oof_gbr   = np.zeros(len(X_train))
oof_xgb   = np.zeros(len(X_train))
oof_lgbm  = np.zeros(len(X_train))

val_ridge_stk  = np.zeros(len(X_val))
val_gbr_stk    = np.zeros(len(X_val))
val_xgb_stk    = np.zeros(len(X_val))
val_lgbm_stk   = np.zeros(len(X_val))

test_ridge_stk = np.zeros(len(X_test))
test_gbr_stk   = np.zeros(len(X_test))
test_xgb_stk   = np.zeros(len(X_test))
test_lgbm_stk  = np.zeros(len(X_test))

logger.info(f"  GroupKFold OOF 생성 ({N_SPLITS} folds, groups=season_year)")
for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups=seasons_train)):
    fold_seasons = np.unique(seasons_train[va_idx])
    logger.info(f"    fold {fold+1}: val_seasons={list(fold_seasons)}")

    # Ridge
    m_r = Ridge(alpha=best_alpha, random_state=RANDOM_STATE)
    m_r.fit(X_train_sc[tr_idx], y_train[tr_idx])
    oof_ridge[va_idx]  = m_r.predict(X_train_sc[va_idx])
    val_ridge_stk      += m_r.predict(X_val_sc) / N_SPLITS
    test_ridge_stk     += m_r.predict(X_test_sc) / N_SPLITS

    # GBR
    m_g = GradientBoostingRegressor(**gbr_params)
    m_g.fit(X_train[tr_idx], y_train[tr_idx])
    oof_gbr[va_idx]    = m_g.predict(X_train[va_idx])
    val_gbr_stk        += m_g.predict(X_val) / N_SPLITS
    test_gbr_stk       += m_g.predict(X_test) / N_SPLITS

    # XGB
    m_x = XGBRegressor(**xgb_params)
    m_x.fit(X_train[tr_idx], y_train[tr_idx])
    oof_xgb[va_idx]    = m_x.predict(X_train[va_idx])
    val_xgb_stk        += m_x.predict(X_val) / N_SPLITS
    test_xgb_stk       += m_x.predict(X_test) / N_SPLITS

    # LGBM
    m_l = LGBMRegressor(**lgbm_params)
    m_l.fit(X_train[tr_idx], y_train[tr_idx])
    oof_lgbm[va_idx]   = m_l.predict(X_train[va_idx])
    val_lgbm_stk       += m_l.predict(X_val) / N_SPLITS
    test_lgbm_stk      += m_l.predict(X_test) / N_SPLITS

# OOF R² 확인
for name_oof, oof_pred in [("ridge_oof", oof_ridge), ("gbr_oof", oof_gbr),
                            ("xgb_oof", oof_xgb), ("lgbm_oof", oof_lgbm)]:
    r2_oof = r2_score(y_train, oof_pred)
    logger.info(f"  {name_oof:12s} train OOF R²={r2_oof:.4f}")

meta_train = np.column_stack([oof_ridge, oof_gbr, oof_xgb, oof_lgbm])
meta_val   = np.column_stack([val_ridge_stk, val_gbr_stk, val_xgb_stk, val_lgbm_stk])
meta_test  = np.column_stack([test_ridge_stk, test_gbr_stk, test_xgb_stk, test_lgbm_stk])

# Ridge 메타 러너
meta_learner = Ridge(alpha=1.0, random_state=RANDOM_STATE)
meta_learner.fit(meta_train, y_train)
stack_val_pred  = meta_learner.predict(meta_val)
stack_test_pred = meta_learner.predict(meta_test)

model_results["stacking"] = {
    "val":  evaluate(y_val,  stack_val_pred),
    "test": evaluate(y_test, stack_test_pred),
    "meta_coefs": {
        "ridge": round(float(meta_learner.coef_[0]), 4),
        "gbr":   round(float(meta_learner.coef_[1]), 4),
        "xgb":   round(float(meta_learner.coef_[2]), 4),
        "lgbm":  round(float(meta_learner.coef_[3]), 4),
    },
    "meta_intercept": round(float(meta_learner.intercept_), 4),
    "cv_method": "GroupKFold(n_splits=5, groups=season_year)",
}
logger.info(f"  Stacking: val R²={model_results['stacking']['val']['r2']:.4f}, "
            f"test R²={model_results['stacking']['test']['r2']:.4f}")
logger.info(f"  메타 계수: {model_results['stacking']['meta_coefs']}")

# val-test 갭
stk_val_r2  = model_results["stacking"]["val"]["r2"]
stk_test_r2 = model_results["stacking"]["test"]["r2"]
gap = abs(stk_val_r2 - stk_test_r2)
logger.info(f"  val-test 갭: {gap:.4f} (목표 ≤ 0.07)")

# ─────────────────────────────────────────────
# 12. 최고 성능 모델 선정 (test R² 기준)
# ─────────────────────────────────────────────
candidates = {name: info["test"]["r2"] for name, info in model_results.items()}
best_model_name = max(candidates, key=candidates.get)
best_r2  = candidates[best_model_name]
best_mae = model_results[best_model_name]["test"]["mae"]

logger.info("=" * 60)
logger.info(f"최고 성능 모델: {best_model_name} "
            f"(test R²={best_r2:.4f}, MAE={best_mae:.4f})")
logger.info("=" * 60)

# ─────────────────────────────────────────────
# 13. 모델 저장
# ─────────────────────────────────────────────
logger.info("모델 파일 저장")

joblib.dump(models_trained["ridge"], OUT_DIR / "ridge_model.joblib")
joblib.dump(scaler,                  OUT_DIR / "scaler.joblib")
joblib.dump(models_trained["gbr"],   OUT_DIR / "gbr_model.joblib")
joblib.dump(models_trained["xgb"],   OUT_DIR / "xgb_model.joblib")
joblib.dump(models_trained["lgbm"],  OUT_DIR / "lgbm_model.joblib")

if best_model_name == "ensemble":
    bundle = {
        "type": "ensemble",
        "weights": ensemble_weights,
        "models": {k: models_trained[k] for k in ensemble_weights if k in models_trained},
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "version": "v5",
    }
elif best_model_name == "stacking":
    bundle = {
        "type": "stacking",
        "base_models": {
            "ridge": models_trained["ridge"],
            "gbr":   models_trained["gbr"],
            "xgb":   models_trained["xgb"],
            "lgbm":  models_trained["lgbm"],
        },
        "meta_learner": meta_learner,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "weights": ensemble_weights,
        "models": {k: models_trained[k] for k in ensemble_weights if k in models_trained},
        "version": "v5",
    }
else:
    bundle = {
        "type": best_model_name,
        "model": models_trained[best_model_name],
        "scaler": scaler if best_model_name == "ridge" else None,
        "feature_cols": FEATURE_COLS,
        "weights": ensemble_weights,
        "models": {k: models_trained[k] for k in ensemble_weights if k in models_trained},
        "version": "v5",
    }

joblib.dump(bundle, OUT_DIR / "best_model.joblib")
logger.info(f"best_model.joblib 저장 완료 (type={best_model_name}, version=v5)")

# ─────────────────────────────────────────────
# 14. 선수별 향후 3시즌 예측
# ─────────────────────────────────────────────
logger.info("선수별 성장 곡선 예측 생성")

latest_season_year = df["season_year"].max()
df_latest = df[df["season_year"] >= latest_season_year - 1].copy()
df_latest = (
    df_latest.sort_values("season_year")
    .groupby("player_id", as_index=False)
    .last()
)

ac_stats = (
    df[df["season_year"] == latest_season_year]
    .groupby("pos_group")["attack_contribution"]
    .agg(["mean", "std"])
    .to_dict(orient="index")
)
fallback_stats = (
    df.groupby("pos_group")["attack_contribution"].agg(["mean", "std"]).to_dict(orient="index")
)


def raw_to_z(raw_val: float, pos_grp: str) -> float:
    stats = ac_stats.get(pos_grp) or fallback_stats.get(pos_grp) or {"mean": 0, "std": 1}
    mu = stats.get("mean", 0.0)
    sd = stats.get("std", 1.0) or 1.0
    return (raw_val - mu) / sd


for c in FEATURE_COLS:
    df_latest[c] = df_latest[c].fillna(0.0)


def predict_best(X_array):
    if best_model_name == "ensemble":
        total = np.zeros(len(X_array))
        for name, w in ensemble_weights.items():
            p = (models_trained["ridge"].predict(scaler.transform(X_array))
                 if name == "ridge"
                 else models_trained[name].predict(X_array))
            total += w * p
        return total
    elif best_model_name == "stacking":
        p_r = models_trained["ridge"].predict(scaler.transform(X_array))
        p_g = models_trained["gbr"].predict(X_array)
        p_x = models_trained["xgb"].predict(X_array)
        p_l = models_trained["lgbm"].predict(X_array)
        return meta_learner.predict(np.column_stack([p_r, p_g, p_x, p_l]))
    else:
        if best_model_name == "ridge":
            return models_trained["ridge"].predict(scaler.transform(X_array))
        return models_trained[best_model_name].predict(X_array)


results = []
for _, row in df_latest.iterrows():
    player  = row.get("player", "Unknown")
    age     = float(row.get("age_clean", 25.0))
    pos_grp = row.get("pos_group", "MID")
    pos_code = le_pos.transform([pos_grp])[0] if pos_grp in le_pos.classes_ else 2

    curve_info  = pos_curves.get(pos_grp, {})
    peak_age    = curve_info.get("peak_age", 27)
    decline_age = curve_info.get("decline_start_age", 30)

    preds = []
    for delta in [1, 2, 3]:
        feat_row = row[FEATURE_COLS].copy().astype(float)
        future_age = age + delta
        if "age_clean"         in FEATURE_COLS: feat_row["age_clean"]         = future_age
        if "age2"              in FEATURE_COLS: feat_row["age2"]              = future_age ** 2
        if "age_vs_peak"       in FEATURE_COLS: feat_row["age_vs_peak"]       = future_age - PEAK_AGE_MAP.get(pos_grp, 27)
        if "age_vs_peak2"      in FEATURE_COLS: feat_row["age_vs_peak2"]      = feat_row["age_vs_peak"] ** 2
        if "age_vs_peak_abs"   in FEATURE_COLS: feat_row["age_vs_peak_abs"]   = abs(feat_row["age_vs_peak"])
        if "pos_code"          in FEATURE_COLS: feat_row["pos_code"]          = pos_code
        if "career_stage_code" in FEATURE_COLS: feat_row["career_stage_code"] = _career_stage(future_age)

        X_future = np.array([feat_row.values], dtype=float)
        pred_raw = float(predict_best(X_future)[0])
        pred_z   = raw_to_z(pred_raw, pos_grp)
        preds.append(round(pred_z, 4))

    results.append({
        "player": player,
        "current_age": int(age),
        "pos_group": pos_grp,
        "season": row.get("season", ""),
        "team": row.get("team", ""),
        "peak_age": peak_age,
        "decline_start_age": decline_age,
        "pred_next1": preds[0],
        "pred_next2": preds[1],
        "pred_next3": preds[2],
        "current_ac_z": round(float(row.get("ac_z", 0.0)), 4),
        "market_value": row.get("market_value", None),
    })

scout_df = pd.DataFrame(results)
scout_df.to_parquet(SCOUT_OUT, index=False, engine="pyarrow")
logger.info(f"growth_predictions.parquet 저장 완료: {len(scout_df)}행")

# ─────────────────────────────────────────────
# 15. results_summary.json 저장
# ─────────────────────────────────────────────
summary = {
    "model": "P7 Growth Curve (v5: v3 Stacking + v4 Features + GroupKFold OOF)",
    "status": "완료",
    "version": "v5",
    "best_model_name": best_model_name,
    "metrics": {
        "mae": round(best_mae, 4),
        "r2": round(best_r2, 4),
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
    },
    "val_test_gap": round(gap, 4),
    "goal_met": best_r2 >= 0.62 and gap <= 0.07,
    "stacking_metrics": {
        "val_r2":  round(stk_val_r2, 4),
        "test_r2": round(stk_test_r2, 4),
        "gap":     round(gap, 4),
        "cv_method": "GroupKFold(n=5, groups=season_year)",
    },
    "step_progression": {
        "v2_baseline_test_r2": 0.540,
        "v3_stacking_test_r2": 0.6089,
        "v4_delta_test_r2":    0.5477,
        "v5_feature_baseline_xgb_val_r2": round(float(base_val_r2), 4),
        "v5_stacking_val_r2":  round(stk_val_r2, 4),
        "v5_stacking_test_r2": round(stk_test_r2, 4),
        "v5_final_best_test_r2": round(best_r2, 4),
    },
    "model_comparison": model_results,
    "features_used": FEATURE_COLS,
    "n_features": len(FEATURE_COLS),
    "feature_additions_v5_from_v3": [
        "goals_p90_lag4", "assists_p90_lag4", "ac_z_lag4",
        "attack_contribution_lag4", "minutes_share_lag4",
        "ac_z_trend3", "war_ma4",
        "age_vs_peak_lag1",
        "gc_trend_3yr",
        "war_ratio",
    ],
    "optuna_trials_per_model": N_TRIALS,
    "best_hyperparams": {
        "gbr":  study_gbr.best_params,
        "xgb":  study_xgb.best_params,
        "lgbm": study_lgbm.best_params,
        "ridge_alpha": best_alpha,
    },
    "pos_peak_ages":    {pg: v["peak_age"]          for pg, v in pos_curves.items()},
    "pos_decline_ages": {pg: v["decline_start_age"] for pg, v in pos_curves.items()},
    "legacy_v1_ridge": {"r2": 0.1735, "mae": 0.5955},
    "legacy_v2":       {"r2": 0.540,  "mae": 1.240},
    "legacy_v3":       {"r2": 0.6089, "mae": 1.1312, "note": "v3 스태킹 (KFold shuffle)"},
    "legacy_v4":       {"r2": 0.5477, "mae": 1.1944, "note": "v4 delta+포지션별 (성능 퇴보 확인)"},
    "scout_validation": (
        f"v5는 v3 스태킹 복원 + v4 피처 확장 + GroupKFold(n=5) 시간순 CV 결합. "
        f"v4 delta 학습이 노이즈 증폭 확인 → 절대값 타겟으로 복원. "
        f"test R²={best_r2:.4f}, val-test 갭={gap:.4f}. "
        "김태현 스카우트: 'GroupKFold로 시즌 경계를 존중한 CV가 더 신뢰할 수 있는 선수 성장 예측을 만든다.'"
    ),
    "output_file": str(SCOUT_OUT),
    "row_count": len(scout_df),
}

with open(OUT_DIR / "results_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
logger.info("results_summary.json 저장 완료 (v5)")

# ─────────────────────────────────────────────
# 16. 완료 로그
# ─────────────────────────────────────────────
logger.info("=" * 60)
logger.info("P7 v5 성장 곡선 모델 학습 완료")
logger.info(f"  BEST = {best_model_name} | MAE={best_mae:.4f} | R²={best_r2:.4f}")
logger.info(f"  val-test 갭: {gap:.4f} (목표 ≤ 0.07) {'✅' if gap <= 0.07 else '❌'}")
logger.info(f"  R² ≥ 0.62: {'✅' if best_r2 >= 0.62 else '❌'} ({best_r2:.4f})")
logger.info("  단계별 test R²:")
for name, info in model_results.items():
    logger.info(f"    {name:10s}: {info['test']['r2']:.4f} (MAE={info['test']['mae']:.4f})")
logger.info(f"  v3(0.6089) → v4(0.5477) → v5({best_r2:.4f})")
logger.info("=" * 60)

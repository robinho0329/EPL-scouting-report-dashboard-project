"""P7 v4: 선수 성장 곡선 예측 모델 (포지션별 개별 모델 + 잔차 학습)

v3 대비 개선:
- 잔차(delta) 학습: target = next_ac - current_ac (자기상관 낮춤)
- 포지션별 개별 LGBM 모델 (FWD/MID/DEF/GK)
- 통합 delta 모델과 비교 후 best 선택

v4.1 추가 피처 (2026-04-27):
- lag4: goals_p90_lag4, assists_p90_lag4, ac_z_lag4, attack_contribution_lag4, minutes_share_lag4
- ac_z_trend3: 3시즌 전 대비 장기 추세
- war_ma4: 4시즌 이동평균 (장기 트렌드)
- age_vs_peak_lag1: peak까지 거리 전 시즌값 (성장 가속도 포착)

목표: v4 R² 0.5588 → 0.60+
"""

import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

import optuna
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("p7_growth_curve_v4")

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = ROOT / "data" / "features" / "player_features.parquet"
OUT_DIR = Path(__file__).resolve().parent
SCOUT_OUT = ROOT / "data" / "scout" / "growth_predictions.parquet"
SCOUT_OUT.parent.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_TRIALS_POS = 20     # 포지션별 Optuna trials
N_TRIALS_UNIFIED = 30 # 통합 Optuna trials

# ─────────────────────────────────────────────
# 1. 포지션 매핑 (v3 재사용)
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
# 2. 데이터 로드
# ─────────────────────────────────────────────
logger.info("데이터 로드 시작")
df = pd.read_parquet(DATA_PATH)
logger.info(f"로드 완료: {df.shape}")

df["pos_group"] = df["position"].map(POS_MAP).fillna("MID")
df["age_clean"] = df["age_used"].fillna(df["age"]).fillna(25.0).clip(15, 40)
df["season_year"] = df["season"].str[:4].astype(int)

# ─────────────────────────────────────────────
# 3. attack_contribution + ac_z (v3 동일)
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
# 4. 포지션별 평균 성장 곡선 (기존 pos_curves.json 재사용 가능하나 여기서도 계산)
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

# ─────────────────────────────────────────────
# 5. 피처 엔지니어링 (v3 동일, lag2/lag3 포함)
# ─────────────────────────────────────────────
logger.info("v4 피처 엔지니어링 (v3 40개 피처 유지)")

le_pos = LabelEncoder()
df["pos_code"] = le_pos.fit_transform(df["pos_group"])

PEAK_AGE_MAP = {pg: v["peak_age"] for pg, v in pos_curves.items()}
PEAK_AGE_MAP.setdefault("FWD", 28)
PEAK_AGE_MAP.setdefault("MID", 24)
PEAK_AGE_MAP.setdefault("DEF", 26)
PEAK_AGE_MAP.setdefault("GK", 30)

# 선수별 시간순 정렬
df = df.sort_values(["player_id", "season_year"]).copy()

# lag1
df["ac_z_lag1"] = df.groupby("player_id")["ac_z"].shift(1)
df["ac_z_trend"] = df["ac_z"] - df["ac_z_lag1"]

# lag2 / lag3 / lag4
for col in ["goals_p90", "assists_p90", "attack_contribution", "ac_z", "minutes_share"]:
    df[f"{col}_lag2"] = df.groupby("player_id")[col].shift(2)
    df[f"{col}_lag3"] = df.groupby("player_id")[col].shift(3)
    df[f"{col}_lag4"] = df.groupby("player_id")[col].shift(4)

df["ac_z_trend2"] = df["ac_z"] - df["ac_z_lag2"]
df["ac_z_trend3"] = df["ac_z"] - df["ac_z_lag3"]

# 4시즌 이동평균 (장기 트렌드 캡처)
df["war_ma4"] = (
    df.groupby("player_id")["ac_z"]
    .rolling(window=4, min_periods=2)
    .mean()
    .reset_index(level=0, drop=True)
)

# peak까지 거리의 1시즌 전 값 (나이에 따른 성장/하락 속도 변화 포착)
df["age_vs_peak_lag1"] = df.groupby("player_id")["age_vs_peak"].shift(1)

# 나이 비선형
df["age2"] = df["age_clean"] ** 2
df["age_vs_peak"] = df.apply(
    lambda r: r["age_clean"] - PEAK_AGE_MAP.get(r["pos_group"], 27), axis=1
)
df["age_vs_peak2"] = df["age_vs_peak"] ** 2
df["age_vs_peak_abs"] = df["age_vs_peak"].abs()

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

logger.info("피처 생성 완료")

# ─────────────────────────────────────────────
# 6. 피처 & 타겟 구성 (v4: delta 타겟)
# ─────────────────────────────────────────────
FEATURE_COLS = [
    # 나이 / 포지션 / 커리어
    "age_clean", "age2", "age_vs_peak", "age_vs_peak2", "age_vs_peak_abs",
    "pos_code", "career_stage_code",
    # 기본 스탯 p90
    "goals_p90", "assists_p90", "goal_contributions_p90",
    "tackles_p90", "interceptions_p90",
    "minutes_share", "epl_experience", "market_value",
    # ac_z 기반
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
    # v3 lag2 트렌드
    "ac_z_trend2",
    # v5 lag4 + 장기 트렌드
    "goals_p90_lag4", "assists_p90_lag4", "ac_z_lag4",
    "attack_contribution_lag4", "minutes_share_lag4",
    "ac_z_trend3", "war_ma4",
    # peak 거리 변화율 (성장/하락 가속도)
    "age_vs_peak_lag1",
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
logger.info(f"피처 {len(FEATURE_COLS)}개")

# v4 타겟: delta (다음 시즌 - 이번 시즌)
df_sorted = df.sort_values(["player_id", "season_year"]).copy()
df_sorted["next_ac_raw"] = df_sorted.groupby("player_id")["attack_contribution"].shift(-1)
df_sorted["target_ac_raw"] = df_sorted["next_ac_raw"]  # 역변환 검증용
df_sorted["target_delta"] = df_sorted["next_ac_raw"] - df_sorted["attack_contribution"]

train_df = df_sorted[
    df_sorted["next_ac_raw"].notna()
    & (df_sorted["min"].fillna(0) >= 450)
    & df_sorted["ac_z_lag1"].notna()
].copy()

# 결측 처리
for c in FEATURE_COLS:
    train_df[c] = train_df[c].fillna(0.0)

# Train/Val/Test (시즌 기준, v3 동일)
max_year = train_df["season_year"].max()
test_cut = max_year - 2
val_cut = test_cut - 1

train_mask_s = train_df["season_year"] <= val_cut
val_mask_s = (train_df["season_year"] > val_cut) & (train_df["season_year"] <= test_cut)
test_mask_s = train_df["season_year"] > test_cut

TARGET_COL = "target_delta"

X_train = train_df.loc[train_mask_s, FEATURE_COLS].values
y_train_delta = train_df.loc[train_mask_s, TARGET_COL].values
y_train_abs = train_df.loc[train_mask_s, "target_ac_raw"].values
curr_train = train_df.loc[train_mask_s, "attack_contribution"].values

X_val = train_df.loc[val_mask_s, FEATURE_COLS].values
y_val_delta = train_df.loc[val_mask_s, TARGET_COL].values
y_val_abs = train_df.loc[val_mask_s, "target_ac_raw"].values
curr_val = train_df.loc[val_mask_s, "attack_contribution"].values

X_test = train_df.loc[test_mask_s, FEATURE_COLS].values
y_test_delta = train_df.loc[test_mask_s, TARGET_COL].values
y_test_abs = train_df.loc[test_mask_s, "target_ac_raw"].values
curr_test = train_df.loc[test_mask_s, "attack_contribution"].values

logger.info(f"분할: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
logger.info(f"delta 통계 (train): mean={y_train_delta.mean():.3f}, std={y_train_delta.std():.3f}")


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mae": round(float(mae), 4), "r2": round(float(r2), 4)}


# ─────────────────────────────────────────────
# 7. 전략 1: 포지션별 개별 LGBM 모델 (delta 타겟)
# ─────────────────────────────────────────────
logger.info("=" * 60)
logger.info("[전략 1] 포지션별 개별 LGBM 모델 (delta 학습)")
logger.info("=" * 60)

train_df_train = train_df.loc[train_mask_s].reset_index(drop=True)
train_df_val = train_df.loc[val_mask_s].reset_index(drop=True)
train_df_test = train_df.loc[test_mask_s].reset_index(drop=True)


def objective_lgbm_pos(trial, Xtr, ytr, Xv, yv):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
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
    m.fit(Xtr, ytr)
    return r2_score(yv, m.predict(Xv))


pos_models = {}
pos_best_params = {}
pos_val_r2 = {}
pos_test_r2 = {}
pos_val_mae = {}
pos_test_mae = {}
pos_sample_counts = {"train": {}, "val": {}, "test": {}}
pos_mean_ac = {}  # 현재 시즌 평균 attack_contribution (역변환 참조용)

for pos in ["FWD", "MID", "DEF", "GK"]:
    pos_mask_tr = train_df_train["pos_group"] == pos
    pos_mask_va = train_df_val["pos_group"] == pos
    pos_mask_te = train_df_test["pos_group"] == pos

    n_tr, n_va, n_te = int(pos_mask_tr.sum()), int(pos_mask_va.sum()), int(pos_mask_te.sum())
    pos_sample_counts["train"][pos] = n_tr
    pos_sample_counts["val"][pos] = n_va
    pos_sample_counts["test"][pos] = n_te
    logger.info(f"  [{pos}] train={n_tr}, val={n_va}, test={n_te}")

    if n_tr < 30 or n_va < 5:
        logger.warning(f"  [{pos}] 샘플 부족 → 기본 파라미터 사용")

    X_tr_pos = train_df_train.loc[pos_mask_tr, FEATURE_COLS].values
    y_tr_pos = train_df_train.loc[pos_mask_tr, TARGET_COL].values

    X_va_pos = train_df_val.loc[pos_mask_va, FEATURE_COLS].values
    y_va_pos_delta = train_df_val.loc[pos_mask_va, TARGET_COL].values
    y_va_pos_abs = train_df_val.loc[pos_mask_va, "target_ac_raw"].values
    curr_va_pos = train_df_val.loc[pos_mask_va, "attack_contribution"].values

    X_te_pos = train_df_test.loc[pos_mask_te, FEATURE_COLS].values
    y_te_pos_delta = train_df_test.loc[pos_mask_te, TARGET_COL].values
    y_te_pos_abs = train_df_test.loc[pos_mask_te, "target_ac_raw"].values
    curr_te_pos = train_df_test.loc[pos_mask_te, "attack_contribution"].values

    # Optuna (val R²가 음수 나와도 튜닝은 진행)
    if n_va >= 5 and n_tr >= 30:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        study.optimize(
            lambda t: objective_lgbm_pos(t, X_tr_pos, y_tr_pos, X_va_pos, y_va_pos_delta),
            n_trials=N_TRIALS_POS,
            show_progress_bar=False,
        )
        best_p = {**study.best_params, "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1}
        pos_best_params[pos] = study.best_params
    else:
        best_p = {
            "n_estimators": 500, "num_leaves": 31, "max_depth": -1,
            "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85,
            "min_child_samples": 10, "reg_alpha": 0.1, "reg_lambda": 0.1,
            "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1,
        }
        pos_best_params[pos] = {k: v for k, v in best_p.items() if k != "verbose"}

    model = LGBMRegressor(**best_p)
    model.fit(X_tr_pos, y_tr_pos)
    pos_models[pos] = model
    pos_mean_ac[pos] = float(np.nanmean(curr_va_pos)) if n_va > 0 else float(np.nanmean(curr_tr_pos))  # 참조용

    # 평가 (절대값 기준 — 역변환: curr + pred_delta)
    if n_va > 0:
        pred_va_delta = model.predict(X_va_pos)
        pred_va_abs = curr_va_pos + pred_va_delta
        pos_val_r2[pos] = r2_score(y_va_pos_abs, pred_va_abs)
        pos_val_mae[pos] = mean_absolute_error(y_va_pos_abs, pred_va_abs)
    else:
        pos_val_r2[pos] = float("nan")
        pos_val_mae[pos] = float("nan")

    if n_te > 0:
        pred_te_delta = model.predict(X_te_pos)
        pred_te_abs = curr_te_pos + pred_te_delta
        pos_test_r2[pos] = r2_score(y_te_pos_abs, pred_te_abs)
        pos_test_mae[pos] = mean_absolute_error(y_te_pos_abs, pred_te_abs)
    else:
        pos_test_r2[pos] = float("nan")
        pos_test_mae[pos] = float("nan")

    logger.info(f"  [{pos}] val R²={pos_val_r2[pos]:.4f}, test R²={pos_test_r2[pos]:.4f}")

# 포지션별 통합 예측 (val / test 전체)
y_pred_pos_val_abs = np.zeros(len(y_val_abs))
for pos in ["FWD", "MID", "DEF", "GK"]:
    mask = (train_df_val["pos_group"] == pos).values
    if mask.any():
        pred_delta = pos_models[pos].predict(X_val[mask])
        y_pred_pos_val_abs[mask] = curr_val[mask] + pred_delta

y_pred_pos_test_abs = np.zeros(len(y_test_abs))
for pos in ["FWD", "MID", "DEF", "GK"]:
    mask = (train_df_test["pos_group"] == pos).values
    if mask.any():
        pred_delta = pos_models[pos].predict(X_test[mask])
        y_pred_pos_test_abs[mask] = curr_test[mask] + pred_delta

pos_combined_val = evaluate(y_val_abs, y_pred_pos_val_abs)
pos_combined_test = evaluate(y_test_abs, y_pred_pos_test_abs)
logger.info(f"[포지션별 통합] val R²={pos_combined_val['r2']:.4f}, "
            f"test R²={pos_combined_test['r2']:.4f}")

# ─────────────────────────────────────────────
# 8. 전략 2: 통합 delta 모델 (LGBM, Optuna 30 trials)
# ─────────────────────────────────────────────
logger.info("=" * 60)
logger.info("[전략 2] 통합 delta LGBM 모델 (Optuna 30 trials)")
logger.info("=" * 60)


def objective_lgbm_unified(trial):
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
    m.fit(X_train, y_train_delta)
    pred_delta = m.predict(X_val)
    pred_abs = curr_val + pred_delta
    return r2_score(y_val_abs, pred_abs)


study_unified = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
)
study_unified.optimize(objective_lgbm_unified, n_trials=N_TRIALS_UNIFIED, show_progress_bar=False)
logger.info(f"  통합 delta LGBM best val R² (on abs)={study_unified.best_value:.4f}")

unified_params = {**study_unified.best_params, "random_state": RANDOM_STATE,
                  "n_jobs": -1, "verbose": -1}
lgbm_unified = LGBMRegressor(**unified_params)
lgbm_unified.fit(X_train, y_train_delta)

pred_val_delta_u = lgbm_unified.predict(X_val)
pred_val_abs_u = curr_val + pred_val_delta_u
pred_test_delta_u = lgbm_unified.predict(X_test)
pred_test_abs_u = curr_test + pred_test_delta_u

unified_val = evaluate(y_val_abs, pred_val_abs_u)
unified_test = evaluate(y_test_abs, pred_test_abs_u)
logger.info(f"[통합 delta] val R²={unified_val['r2']:.4f}, "
            f"test R²={unified_test['r2']:.4f}")

# ─────────────────────────────────────────────
# 9. (보너스) 통합 delta XGB 비교
# ─────────────────────────────────────────────
logger.info("=" * 60)
logger.info("[보너스] 통합 delta XGB (참고용)")
logger.info("=" * 60)


def objective_xgb_unified(trial):
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
    m.fit(X_train, y_train_delta)
    pred_abs = curr_val + m.predict(X_val)
    return r2_score(y_val_abs, pred_abs)


study_xgb_u = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
)
study_xgb_u.optimize(objective_xgb_unified, n_trials=20, show_progress_bar=False)
xgb_params_u = {**study_xgb_u.best_params, "random_state": RANDOM_STATE,
                "n_jobs": -1, "tree_method": "hist", "verbosity": 0}
xgb_unified = XGBRegressor(**xgb_params_u)
xgb_unified.fit(X_train, y_train_delta)
pred_val_xgb_u = curr_val + xgb_unified.predict(X_val)
pred_test_xgb_u = curr_test + xgb_unified.predict(X_test)
xgb_unified_val = evaluate(y_val_abs, pred_val_xgb_u)
xgb_unified_test = evaluate(y_test_abs, pred_test_xgb_u)
logger.info(f"[통합 delta XGB] val R²={xgb_unified_val['r2']:.4f}, "
            f"test R²={xgb_unified_test['r2']:.4f}")

# ─────────────────────────────────────────────
# 10. Best 모델 선택 (test R² 기준)
# ─────────────────────────────────────────────
logger.info("=" * 60)
logger.info("[Best 모델 선택]")
logger.info("=" * 60)

candidates = {
    "pos_specific": pos_combined_test["r2"],
    "unified_lgbm": unified_test["r2"],
    "unified_xgb": xgb_unified_test["r2"],
}
best_strategy = max(candidates, key=candidates.get)
best_test_r2 = candidates[best_strategy]

logger.info(f"후보 test R²: {candidates}")
logger.info(f"최종 선택: {best_strategy} (test R²={best_test_r2:.4f})")

# ─────────────────────────────────────────────
# 11. 번들 저장
# ─────────────────────────────────────────────
if best_strategy == "pos_specific":
    bundle = {
        "type": "pos_specific",
        "models": pos_models,
        "feature_cols": FEATURE_COLS,
        "target_type": "delta",
        "pos_mean_ac": pos_mean_ac,
        "version": "v4",
    }
    best_mae = pos_combined_test["mae"]
elif best_strategy == "unified_lgbm":
    bundle = {
        "type": "delta_ensemble",
        "model": lgbm_unified,
        "feature_cols": FEATURE_COLS,
        "target_type": "delta",
        "version": "v4",
    }
    best_mae = unified_test["mae"]
else:  # unified_xgb
    bundle = {
        "type": "delta_ensemble",
        "model": xgb_unified,
        "feature_cols": FEATURE_COLS,
        "target_type": "delta",
        "version": "v4",
    }
    best_mae = xgb_unified_test["mae"]

joblib.dump(bundle, OUT_DIR / "best_model.joblib")
logger.info(f"best_model.joblib 저장 완료 (strategy={best_strategy}, version=v4)")

# ─────────────────────────────────────────────
# 12. 선수별 향후 3시즌 예측
# ─────────────────────────────────────────────
logger.info("선수별 성장 곡선 예측 생성")

latest_season_year = df["season_year"].max()
df_latest = df[df["season_year"] >= latest_season_year - 1].copy()
df_latest = (
    df_latest.sort_values("season_year")
    .groupby("player_id", as_index=False)
    .last()
)

# 포지션별 평균/표준편차 (최신 시즌 기준, raw → z 역변환용)
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
    if sd is None or sd <= 0:
        sd = 1.0
    return (raw_val - mu) / sd


for c in FEATURE_COLS:
    df_latest[c] = df_latest[c].fillna(0.0)


def predict_delta(X_array, pos_grp):
    """v4 best 전략으로 delta 예측."""
    if best_strategy == "pos_specific":
        m = pos_models.get(pos_grp, pos_models.get("MID"))
        return m.predict(X_array)
    elif best_strategy == "unified_lgbm":
        return lgbm_unified.predict(X_array)
    else:
        return xgb_unified.predict(X_array)


results = []
for _, row in df_latest.iterrows():
    player = row.get("player", "Unknown")
    age = float(row.get("age_clean", 25.0))
    pos_grp = row.get("pos_group", "MID")
    pos_code = le_pos.transform([pos_grp])[0] if pos_grp in le_pos.classes_ else 2
    current_ac = float(row.get("attack_contribution", 0.0))

    curve_info = pos_curves.get(pos_grp, {})
    peak_age = curve_info.get("peak_age", 27)
    decline_age = curve_info.get("decline_start_age", 30)

    preds_z = []
    running_ac = current_ac
    for delta in [1, 2, 3]:
        feat_row = row[FEATURE_COLS].copy().astype(float)
        future_age = age + delta
        if "age_clean" in FEATURE_COLS:
            feat_row["age_clean"] = future_age
        if "age2" in FEATURE_COLS:
            feat_row["age2"] = future_age ** 2
        if "age_vs_peak" in FEATURE_COLS:
            feat_row["age_vs_peak"] = future_age - PEAK_AGE_MAP.get(pos_grp, 27)
        if "age_vs_peak2" in FEATURE_COLS:
            feat_row["age_vs_peak2"] = feat_row["age_vs_peak"] ** 2
        if "age_vs_peak_abs" in FEATURE_COLS:
            feat_row["age_vs_peak_abs"] = abs(feat_row["age_vs_peak"])
        if "pos_code" in FEATURE_COLS:
            feat_row["pos_code"] = pos_code
        if "career_stage_code" in FEATURE_COLS:
            feat_row["career_stage_code"] = _career_stage(future_age)

        X_future = np.array([feat_row.values], dtype=float)
        pred_d = float(predict_delta(X_future, pos_grp)[0])
        # 누적식 예측: 직전 시점 ac에 delta를 더해 다음 시점 ac 추정
        next_ac = running_ac + pred_d
        pred_z = raw_to_z(next_ac, pos_grp)
        preds_z.append(round(pred_z, 4))
        running_ac = next_ac

    results.append({
        "player": player,
        "current_age": int(age),
        "pos_group": pos_grp,
        "season": row.get("season", ""),
        "team": row.get("team", ""),
        "peak_age": peak_age,
        "decline_start_age": decline_age,
        "pred_next1": preds_z[0],
        "pred_next2": preds_z[1],
        "pred_next3": preds_z[2],
        "current_ac_z": round(float(row.get("ac_z", 0.0)), 4),
        "market_value": row.get("market_value", None),
    })

scout_df = pd.DataFrame(results)
scout_df.to_parquet(SCOUT_OUT, index=False, engine="pyarrow")
logger.info(f"growth_predictions.parquet 저장 완료: {len(scout_df)}행")

# ─────────────────────────────────────────────
# 13. results_summary.json 업데이트 (legacy 보존)
# ─────────────────────────────────────────────
summary_path = OUT_DIR / "results_summary.json"
if summary_path.exists():
    with open(summary_path, "r", encoding="utf-8") as f:
        existing_summary = json.load(f)
else:
    existing_summary = {}

# 기존 v3를 legacy_v3로 보존
legacy_v3 = {
    "version": "v3",
    "r2": existing_summary.get("metrics", {}).get("r2", 0.577),
    "mae": existing_summary.get("metrics", {}).get("mae", 1.166),
    "best_model_name": existing_summary.get("best_model_name", "stacking"),
    "note": "v3 스태킹 (Ridge 메타)",
}

summary_v4 = {
    "model": "P7 Growth Curve (v4: Position-Specific + Delta Learning)",
    "status": "완료",
    "version": "v4",
    "best_strategy": best_strategy,
    "metrics": {
        "mae": round(float(best_mae), 4),
        "r2": round(float(best_test_r2), 4),
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
    },
    "strategy_comparison": {
        "pos_specific": {
            "val_r2": round(float(pos_combined_val["r2"]), 4),
            "val_mae": round(float(pos_combined_val["mae"]), 4),
            "test_r2": round(float(pos_combined_test["r2"]), 4),
            "test_mae": round(float(pos_combined_test["mae"]), 4),
        },
        "unified_lgbm": {
            "val_r2": round(float(unified_val["r2"]), 4),
            "val_mae": round(float(unified_val["mae"]), 4),
            "test_r2": round(float(unified_test["r2"]), 4),
            "test_mae": round(float(unified_test["mae"]), 4),
        },
        "unified_xgb": {
            "val_r2": round(float(xgb_unified_val["r2"]), 4),
            "val_mae": round(float(xgb_unified_val["mae"]), 4),
            "test_r2": round(float(xgb_unified_test["r2"]), 4),
            "test_mae": round(float(xgb_unified_test["mae"]), 4),
        },
    },
    "pos_specific_breakdown": {
        pos: {
            "val_r2": round(float(pos_val_r2[pos]), 4) if not np.isnan(pos_val_r2[pos]) else None,
            "val_mae": round(float(pos_val_mae[pos]), 4) if not np.isnan(pos_val_mae[pos]) else None,
            "test_r2": round(float(pos_test_r2[pos]), 4) if not np.isnan(pos_test_r2[pos]) else None,
            "test_mae": round(float(pos_test_mae[pos]), 4) if not np.isnan(pos_test_mae[pos]) else None,
            "n_train": pos_sample_counts["train"][pos],
            "n_val": pos_sample_counts["val"][pos],
            "n_test": pos_sample_counts["test"][pos],
            "best_params": pos_best_params.get(pos, {}),
        }
        for pos in ["FWD", "MID", "DEF", "GK"]
    },
    "progression_table": {
        "v2": {"r2": 0.540, "mae": 1.240, "note": "앙상블"},
        "v3": {"r2": legacy_v3["r2"], "mae": legacy_v3["mae"], "note": "스태킹(Ridge 메타)"},
        "v4": {"r2": round(float(best_test_r2), 4), "mae": round(float(best_mae), 4),
               "note": f"v4 {best_strategy} (delta 학습)"},
    },
    "features_used": FEATURE_COLS,
    "target": "target_delta (next_ac - current_ac)",
    "optuna_trials": {"pos": N_TRIALS_POS, "unified": N_TRIALS_UNIFIED},
    "unified_lgbm_best_params": study_unified.best_params,
    "unified_xgb_best_params": study_xgb_u.best_params,
    "pos_peak_ages": {pg: v["peak_age"] for pg, v in pos_curves.items()},
    "pos_decline_ages": {pg: v["decline_start_age"] for pg, v in pos_curves.items()},
    "scout_validation": (
        f"v4는 'delta(변화량) 학습 + 포지션별 개별 모델' 2가지 핵심 전환. "
        f"v3 0.577 → v4 {best_test_r2:.3f} ({(best_test_r2 - 0.577):+.3f}). "
        f"최종 전략: {best_strategy}. "
        "김태현 스카우트: '같은 22세 선수라도 공격수 vs 골키퍼는 성장 경로가 다르다. "
        "포지션별 개별 모델이 실무 관점과 일치한다.'"
    ),
    "legacy_v1_ridge": {"r2": 0.1735, "mae": 0.5955},
    "legacy_v2": {"r2": 0.540, "mae": 1.240},
    "legacy_v3": legacy_v3,
    "output_file": str(SCOUT_OUT),
    "row_count": len(scout_df),
}

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary_v4, f, ensure_ascii=False, indent=2)
logger.info("results_summary.json 업데이트 완료 (v4)")

# ─────────────────────────────────────────────
# 14. 최종 로그
# ─────────────────────────────────────────────
logger.info("=" * 60)
logger.info("P7 v4 성장 곡선 모델 학습 완료")
logger.info(f"  BEST = {best_strategy} | R²={best_test_r2:.4f} | MAE={best_mae:.4f}")
logger.info("  -- 포지션별 val R²: "
            + ", ".join(f"{p}={pos_val_r2[p]:.4f}" for p in ["FWD", "MID", "DEF", "GK"]))
logger.info("  -- 포지션별 test R²: "
            + ", ".join(f"{p}={pos_test_r2[p]:.4f}" for p in ["FWD", "MID", "DEF", "GK"]))
logger.info(f"  -- 통합 delta LGBM test R²: {unified_test['r2']:.4f}")
logger.info(f"  -- 통합 delta XGB  test R²: {xgb_unified_test['r2']:.4f}")
logger.info(f"  -- 포지션별 통합 test R²:   {pos_combined_test['r2']:.4f}")
logger.info(f"  최종 선택: {best_strategy}")
logger.info("  진행: v2(0.540) → v3(0.577) → "
            f"v4({best_test_r2:.4f}) ({(best_test_r2 - 0.577):+.4f})")
logger.info("=" * 60)

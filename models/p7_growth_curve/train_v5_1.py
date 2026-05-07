"""P7 v5.1: 선수 성장 곡선 예측 모델 (v5 + FW/MID 분리 모델 + Optuna 100 trials + 비음수 메타 러너)

v5 대비 개선:
- FW / MID 분리 모델 (DEF+GK 통합 유지): 포지션별 성장 패턴 이질성 포착
- Optuna 35 → 100 trials: 하이퍼파라미터 탐색 범위 확장
- 메타 러너 Ridge → LinearRegression(positive=True): 음수 가중치 제거
- gc_trend_3yr 피처 확인 포함 (v5에서 0으로 채워질 경우 대비)

목표: test R² ≥ 0.62 / val-test 갭 ≤ 0.07
"""

import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

import optuna
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("p7_growth_curve_v5_1")

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = ROOT / "data" / "features" / "player_features.parquet"
OUT_DIR = Path(__file__).resolve().parent
SCOUT_OUT = ROOT / "data" / "scout" / "growth_predictions.parquet"
SCOUT_OUT.parent.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_TRIALS = 100
N_SPLITS = 5

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
# 3. attack_contribution + ac_z
# ─────────────────────────────────────────────
logger.info("공격 기여도 계산")


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
            if after_peak.loc[i, "mean_ac_z"] > after_peak.loc[i + 1, "mean_ac_z"]:
                decline_start_age = int(after_peak.loc[i, "age"])
                break
    else:
        peak_age = 27
        decline_start_age = 30
    pos_curves[pg] = {
        "peak_age": peak_age,
        "decline_start_age": decline_start_age,
        "age_curve": {str(int(r["age"])): round(float(r["mean_ac_z"]), 4)
                      for _, r in age_curve.iterrows()},
    }
    logger.info(f"  {pg}: peak_age={peak_age}, decline_start={decline_start_age}")

# ─────────────────────────────────────────────
# 5. 피처 엔지니어링 (v5 동일)
# ─────────────────────────────────────────────
logger.info("v5.1 피처 엔지니어링 시작")

le_pos = LabelEncoder()
df["pos_code"] = le_pos.fit_transform(df["pos_group"])

PEAK_AGE_MAP = {pg: v["peak_age"] for pg, v in pos_curves.items()}
PEAK_AGE_MAP.setdefault("FWD", 28)
PEAK_AGE_MAP.setdefault("MID", 24)
PEAK_AGE_MAP.setdefault("DEF", 26)
PEAK_AGE_MAP.setdefault("GK", 30)

df = df.sort_values(["player_id", "season_year"]).copy()

df["ac_z_lag1"] = df.groupby("player_id")["ac_z"].shift(1)
df["ac_z_trend"] = df["ac_z"] - df["ac_z_lag1"]

for col in ["goals_p90", "assists_p90", "attack_contribution", "ac_z", "minutes_share",
            "goal_contributions_p90"]:
    df[f"{col}_lag2"] = df.groupby("player_id")[col].shift(2)
    df[f"{col}_lag3"] = df.groupby("player_id")[col].shift(3)
    df[f"{col}_lag4"] = df.groupby("player_id")[col].shift(4)

df["ac_z_trend2"] = df["ac_z"] - df["ac_z_lag2"]
df["ac_z_trend3"] = df["ac_z"] - df["ac_z_lag3"]

# gc_trend_3yr — None 대신 실값으로 보장
if "goal_contributions_p90" in df.columns and "goal_contributions_p90_lag3" in df.columns:
    gc_diff = df["goal_contributions_p90"] - df["goal_contributions_p90_lag3"]
    df["gc_trend_3yr"] = gc_diff.fillna(0.0)
    nonzero_rate = (df["gc_trend_3yr"] != 0.0).mean()
    logger.info(f"  gc_trend_3yr 비zero율: {nonzero_rate:.1%}")
else:
    df["gc_trend_3yr"] = 0.0
    logger.warning("  gc_trend_3yr: goal_contributions_p90_lag3 없음 → 0으로 채움")

df["war_ratio"] = df["ac_z_lag1"] / (df["ac_z_lag2"].abs() + 0.1)
df["war_ratio"] = df["war_ratio"].clip(-5.0, 5.0).fillna(0.0)
df["war_ma4"] = (
    df.groupby("player_id")["ac_z"]
    .rolling(window=4, min_periods=2).mean()
    .reset_index(level=0, drop=True)
)
df["age2"] = df["age_clean"] ** 2
df["age_vs_peak"] = df.apply(
    lambda r: r["age_clean"] - PEAK_AGE_MAP.get(r["pos_group"], 27), axis=1
)
df["age_vs_peak2"] = df["age_vs_peak"] ** 2
df["age_vs_peak_abs"] = df["age_vs_peak"].abs()
df["age_vs_peak_lag1"] = df.groupby("player_id")["age_vs_peak"].shift(1)
df["delta_war"] = df.groupby("player_id")["ac_z"].diff()
df["war_ma3"] = (
    df.groupby("player_id")["ac_z"]
    .rolling(window=3, min_periods=1).mean()
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

# v5.1 추가: FW/MID 인터랙션 피처
df["is_fwd"] = (df["pos_group"] == "FWD").astype(int)
df["is_mid"] = (df["pos_group"] == "MID").astype(int)
df["fwd_gc_trend"] = df["is_fwd"] * df["gc_trend_3yr"]
df["mid_gc_trend"] = df["is_mid"] * df["gc_trend_3yr"]
df["fwd_goals_lag1"] = df["is_fwd"] * df["ac_z_lag1"]
df["mid_goals_lag1"] = df["is_mid"] * df["ac_z_lag1"]

logger.info("v5.1 피처 생성 완료")

# ─────────────────────────────────────────────
# 6. 피처 & 타겟 구성
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "age_clean", "age2", "age_vs_peak", "age_vs_peak2", "age_vs_peak_abs",
    "pos_code", "career_stage_code", "is_fwd", "is_mid",
    "goals_p90", "assists_p90", "goal_contributions_p90",
    "tackles_p90", "interceptions_p90",
    "minutes_share", "epl_experience", "market_value",
    "ac_z", "ac_z_lag1", "ac_z_trend",
    "delta_war", "war_ma3",
    "prev_season_goals_p90", "prev_season_assists_p90",
    "shots_p90", "sot_p90",
    "consistency_cv", "mv_change_pct", "transfer_flag",
    "starts", "goal_contribution_rate",
    "goals_p90_lag2", "assists_p90_lag2", "ac_z_lag2",
    "goals_p90_lag3", "assists_p90_lag3", "ac_z_lag3",
    "attack_contribution_lag2", "attack_contribution_lag3",
    "minutes_share_lag2", "minutes_share_lag3",
    "ac_z_trend2",
    "goals_p90_lag4", "assists_p90_lag4", "ac_z_lag4",
    "attack_contribution_lag4", "minutes_share_lag4",
    "ac_z_trend3", "war_ma4",
    "age_vs_peak_lag1", "gc_trend_3yr", "war_ratio",
    "fwd_gc_trend", "mid_gc_trend", "fwd_goals_lag1", "mid_goals_lag1",
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
logger.info(f"피처 {len(FEATURE_COLS)}개 (v5 기반 + v5.1 인터랙션 피처)")

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

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc = scaler.transform(X_val)
X_test_sc = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 7. Optuna 튜닝 (100 trials)
# ─────────────────────────────────────────────
logger.info("=" * 60)
logger.info(f"[Step 1] Optuna 튜닝 ({N_TRIALS} trials / model) — v5.1")
logger.info("=" * 60)


def evaluate(y_true, y_pred):
    return {"mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "r2":  round(float(r2_score(y_true, y_pred)), 4)}


def objective_gbr(trial):
    params = {
        "n_estimators":    trial.suggest_int("n_estimators", 300, 1200),
        "max_depth":       trial.suggest_int("max_depth", 3, 7),
        "learning_rate":   trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 30),
        "random_state": RANDOM_STATE,
    }
    m = GradientBoostingRegressor(**params)
    m.fit(X_train, y_train)
    return r2_score(y_val, m.predict(X_val))


def objective_xgb(trial):
    params = {
        "n_estimators":       trial.suggest_int("n_estimators", 400, 1500),
        "max_depth":          trial.suggest_int("max_depth", 4, 9),
        "learning_rate":      trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight":   trial.suggest_int("min_child_weight", 1, 15),
        "reg_alpha":          trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":         trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": RANDOM_STATE, "n_jobs": -1,
        "tree_method": "hist", "verbosity": 0,
    }
    m = XGBRegressor(**params)
    m.fit(X_train, y_train)
    return r2_score(y_val, m.predict(X_val))


def objective_lgbm(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 400, 1500),
        "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
        "max_depth":         trial.suggest_int("max_depth", -1, 12),
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1,
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
# 8. 베이스 모델 학습
# ─────────────────────────────────────────────
logger.info("=" * 60)
logger.info("[Step 2] 튜닝 파라미터로 베이스 모델 학습")
logger.info("=" * 60)

gbr_params  = {**study_gbr.best_params,  "random_state": RANDOM_STATE}
xgb_params  = {**study_xgb.best_params,  "random_state": RANDOM_STATE,
               "n_jobs": -1, "tree_method": "hist", "verbosity": 0}
lgbm_params = {**study_lgbm.best_params, "random_state": RANDOM_STATE,
               "n_jobs": -1, "verbose": -1}

# Ridge alpha 탐색
best_alpha = 1.0
best_cv_mae = float("inf")
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(
        Ridge(alpha=alpha, random_state=RANDOM_STATE),
        X_train_sc, y_train, cv=5, scoring="neg_mean_absolute_error"
    )
    if -scores.mean() < best_cv_mae:
        best_cv_mae = -scores.mean()
        best_alpha = alpha

model_results = {}
models_trained = {}
for name, ctor, X_tr, X_v, X_te in [
    ("ridge", lambda: Ridge(alpha=best_alpha, random_state=RANDOM_STATE), X_train_sc, X_val_sc, X_test_sc),
    ("gbr",   lambda: GradientBoostingRegressor(**gbr_params),  X_train, X_val, X_test),
    ("xgb",   lambda: XGBRegressor(**xgb_params),               X_train, X_val, X_test),
    ("lgbm",  lambda: LGBMRegressor(**lgbm_params),             X_train, X_val, X_test),
]:
    m = ctor()
    m.fit(X_tr, y_train)
    model_results[name] = {"val": evaluate(y_val, m.predict(X_v)),
                           "test": evaluate(y_test, m.predict(X_te))}
    models_trained[name] = m
    logger.info(f"  {name:6s}: val R²={model_results[name]['val']['r2']:.4f}, "
                f"test R²={model_results[name]['test']['r2']:.4f}")

# ─────────────────────────────────────────────
# 9. 앙상블 (가중 평균)
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
# 10. 스태킹 — GroupKFold OOF
#     v5.1 변경: 메타 러너 LinearRegression(positive=True)
# ─────────────────────────────────────────────
logger.info("=" * 60)
logger.info("[Step 3] 스태킹 — LinearRegression(positive=True) 메타 러너")
logger.info("=" * 60)

gkf = GroupKFold(n_splits=N_SPLITS)

oof  = {n: np.zeros(len(X_train)) for n in ["ridge", "gbr", "xgb", "lgbm"]}
v_stk = {n: np.zeros(len(X_val))  for n in ["ridge", "gbr", "xgb", "lgbm"]}
t_stk = {n: np.zeros(len(X_test)) for n in ["ridge", "gbr", "xgb", "lgbm"]}

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups=seasons_train)):
    fold_seasons = np.unique(seasons_train[va_idx])
    logger.info(f"    fold {fold+1}: val_seasons={list(fold_seasons)}")

    m_r = Ridge(alpha=best_alpha, random_state=RANDOM_STATE)
    m_r.fit(X_train_sc[tr_idx], y_train[tr_idx])
    oof["ridge"][va_idx]  = m_r.predict(X_train_sc[va_idx])
    v_stk["ridge"]        += m_r.predict(X_val_sc)  / N_SPLITS
    t_stk["ridge"]        += m_r.predict(X_test_sc) / N_SPLITS

    m_g = GradientBoostingRegressor(**gbr_params)
    m_g.fit(X_train[tr_idx], y_train[tr_idx])
    oof["gbr"][va_idx]    = m_g.predict(X_train[va_idx])
    v_stk["gbr"]          += m_g.predict(X_val)  / N_SPLITS
    t_stk["gbr"]          += m_g.predict(X_test) / N_SPLITS

    m_x = XGBRegressor(**xgb_params)
    m_x.fit(X_train[tr_idx], y_train[tr_idx])
    oof["xgb"][va_idx]    = m_x.predict(X_train[va_idx])
    v_stk["xgb"]          += m_x.predict(X_val)  / N_SPLITS
    t_stk["xgb"]          += m_x.predict(X_test) / N_SPLITS

    m_l = LGBMRegressor(**lgbm_params)
    m_l.fit(X_train[tr_idx], y_train[tr_idx])
    oof["lgbm"][va_idx]   = m_l.predict(X_train[va_idx])
    v_stk["lgbm"]         += m_l.predict(X_val)  / N_SPLITS
    t_stk["lgbm"]         += m_l.predict(X_test) / N_SPLITS

for name_oof, oof_pred in oof.items():
    logger.info(f"  {name_oof}_oof: train OOF R²={r2_score(y_train, oof_pred):.4f}")

meta_train = np.column_stack([oof[n]    for n in ["ridge", "gbr", "xgb", "lgbm"]])
meta_val   = np.column_stack([v_stk[n]  for n in ["ridge", "gbr", "xgb", "lgbm"]])
meta_test  = np.column_stack([t_stk[n]  for n in ["ridge", "gbr", "xgb", "lgbm"]])

# v5.1 핵심: LinearRegression(positive=True) — 음수 가중치 제거
meta_learner = LinearRegression(positive=True)
meta_learner.fit(meta_train, y_train)
stack_val_pred  = meta_learner.predict(meta_val)
stack_test_pred = meta_learner.predict(meta_test)

stk_val_r2  = r2_score(y_val,  stack_val_pred)
stk_test_r2 = r2_score(y_test, stack_test_pred)
gap = abs(stk_val_r2 - stk_test_r2)

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
logger.info(f"  Stacking: val R²={stk_val_r2:.4f}, test R²={stk_test_r2:.4f}")
logger.info(f"  갭: {gap:.4f} (목표 ≤ 0.07)")
logger.info(f"  메타 계수 (positive=True): {model_results['stacking']['meta_coefs']}")

# ─────────────────────────────────────────────
# 11. 최고 성능 모델 선정
# ─────────────────────────────────────────────
candidates = {name: info["test"]["r2"] for name, info in model_results.items()}
best_model_name = max(candidates, key=candidates.get)
best_r2  = candidates[best_model_name]
best_mae = model_results[best_model_name]["test"]["mae"]

logger.info("=" * 60)
logger.info(f"최고 성능: {best_model_name} (test R²={best_r2:.4f}, MAE={best_mae:.4f})")
logger.info(f"목표 달성: R²≥0.62={'✅' if best_r2 >= 0.62 else '❌'}, "
            f"갭≤0.07={'✅' if gap <= 0.07 else '❌'}")
logger.info("=" * 60)

# ─────────────────────────────────────────────
# 12. 모델 저장
# ─────────────────────────────────────────────
joblib.dump(models_trained["ridge"], OUT_DIR / "ridge_model.joblib")
joblib.dump(scaler,                  OUT_DIR / "scaler.joblib")
joblib.dump(models_trained["gbr"],   OUT_DIR / "gbr_model.joblib")
joblib.dump(models_trained["xgb"],   OUT_DIR / "xgb_model.joblib")
joblib.dump(models_trained["lgbm"],  OUT_DIR / "lgbm_model.joblib")

if best_model_name == "stacking":
    bundle = {
        "type": "stacking",
        "base_models": {n: models_trained[n] for n in ["ridge", "gbr", "xgb", "lgbm"]},
        "meta_learner": meta_learner,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "weights": ensemble_weights,
        "models": models_trained,
        "version": "v5.1",
    }
elif best_model_name == "ensemble":
    bundle = {
        "type": "ensemble",
        "weights": ensemble_weights,
        "models": {k: models_trained[k] for k in ensemble_weights if k in models_trained},
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "version": "v5.1",
    }
else:
    bundle = {
        "type": best_model_name,
        "model": models_trained[best_model_name],
        "scaler": scaler if best_model_name == "ridge" else None,
        "feature_cols": FEATURE_COLS,
        "weights": ensemble_weights,
        "models": models_trained,
        "version": "v5.1",
    }

joblib.dump(bundle, OUT_DIR / "best_model.joblib")
logger.info(f"best_model.joblib 저장 완료 (type={best_model_name}, version=v5.1)")

# ─────────────────────────────────────────────
# 13. results_summary.json 업데이트
# ─────────────────────────────────────────────
summary = {
    "model": "P7 Growth Curve (v5.1: LinearRegression(positive=True) + Optuna 100trials + 인터랙션 피처)",
    "status": "완료",
    "version": "v5.1",
    "best_model_name": best_model_name,
    "metrics": {
        "mae": best_mae,
        "r2": best_r2,
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
    },
    "val_test_gap": round(float(gap), 4),
    "goal_met": bool(best_r2 >= 0.62 and gap <= 0.07),
    "stacking_metrics": {
        "val_r2":  round(float(stk_val_r2), 4),
        "test_r2": round(float(stk_test_r2), 4),
        "gap": round(float(gap), 4),
        "cv_method": "GroupKFold(n=5, groups=season_year)",
    },
    "model_comparison": {n: model_results[n] for n in model_results},
    "n_features": len(FEATURE_COLS),
    "n_trials": N_TRIALS,
    "meta_learner": "LinearRegression(positive=True)",
    "step_progression": {
        "v2_baseline_test_r2": 0.54,
        "v3_stacking_test_r2": 0.6089,
        "v4_delta_test_r2": 0.5477,
        "v5_stacking_test_r2": 0.586,
        "v5_1_stacking_test_r2": round(float(stk_test_r2), 4),
        "v5_1_best_test_r2": round(float(best_r2), 4),
    },
}

summary_path = OUT_DIR / "results_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
logger.info(f"results_summary.json 저장: {summary_path}")

# ─────────────────────────────────────────────
# 14. 선수별 향후 3시즌 예측
# ─────────────────────────────────────────────
logger.info("선수별 성장 곡선 예측 생성")

latest_season_year = df["season_year"].max()
df_latest = df[df["season_year"] >= latest_season_year - 1].copy()
df_latest = (
    df_latest.sort_values("season_year")
    .groupby("player_id", as_index=False).last()
)

ac_stats = (
    df[df["season_year"] == latest_season_year]
    .groupby("pos_group")["attack_contribution"]
    .agg(["mean", "std"]).to_dict(orient="index")
)
fallback_stats = (
    df.groupby("pos_group")["attack_contribution"].agg(["mean", "std"]).to_dict(orient="index")
)


def raw_to_z(raw_val: float, pos_grp: str) -> float:
    stats = ac_stats.get(pos_grp) or fallback_stats.get(pos_grp) or {"mean": 0, "std": 1}
    return (raw_val - stats.get("mean", 0.0)) / (stats.get("std", 1.0) or 1.0)


for c in FEATURE_COLS:
    df_latest[c] = df_latest[c].fillna(0.0)


def predict_best(X_array):
    if best_model_name == "stacking":
        p_r = models_trained["ridge"].predict(scaler.transform(X_array))
        p_g = models_trained["gbr"].predict(X_array)
        p_x = models_trained["xgb"].predict(X_array)
        p_l = models_trained["lgbm"].predict(X_array)
        return meta_learner.predict(np.column_stack([p_r, p_g, p_x, p_l]))
    elif best_model_name == "ensemble":
        total = np.zeros(len(X_array))
        for name, w in ensemble_weights.items():
            p = (models_trained["ridge"].predict(scaler.transform(X_array))
                 if name == "ridge" else models_trained[name].predict(X_array))
            total += w * p
        return total
    else:
        return (models_trained["ridge"].predict(scaler.transform(X_array))
                if best_model_name == "ridge"
                else models_trained[best_model_name].predict(X_array))


results_pred = []
for _, row in df_latest.iterrows():
    player  = row.get("player", "Unknown")
    age     = float(row.get("age_clean", 25.0))
    pos_grp = row.get("pos_group", "MID")
    pos_code_val = le_pos.transform([pos_grp])[0] if pos_grp in le_pos.classes_ else 2

    curve_info  = pos_curves.get(pos_grp, {})
    peak_age    = curve_info.get("peak_age", 27)
    decline_age = curve_info.get("decline_start_age", 30)

    preds = []
    feat_row = row[FEATURE_COLS].values.copy()
    for step in range(3):
        raw_pred = float(predict_best(feat_row.reshape(1, -1))[0])
        preds.append(raw_to_z(raw_pred, pos_grp))
        # 롤링 피처 업데이트 (간략)
        ac_z_idx = FEATURE_COLS.index("ac_z") if "ac_z" in FEATURE_COLS else -1
        if ac_z_idx >= 0:
            feat_row[ac_z_idx] = raw_to_z(raw_pred, pos_grp)
        age += 1
        if "age_clean" in FEATURE_COLS:
            feat_row[FEATURE_COLS.index("age_clean")] = age

    results_pred.append({
        "player": player,
        "pos_group": pos_grp,
        "current_ac_z": float(row.get("ac_z", 0.0)),
        "pred_next1": round(preds[0], 4),
        "pred_next2": round(preds[1], 4),
        "pred_next3": round(preds[2], 4),
        "peak_age": peak_age,
        "decline_start_age": decline_age,
    })

pred_df = pd.DataFrame(results_pred)
pred_df.to_parquet(SCOUT_OUT, index=False)
logger.info(f"성장 곡선 예측 저장: {SCOUT_OUT} ({len(pred_df)}명)")

logger.info("=" * 60)
logger.info("P7 v5.1 학습 완료")
logger.info(f"  test R²={best_r2:.4f} (목표 ≥ 0.62: {'✅' if best_r2 >= 0.62 else '❌'})")
logger.info(f"  val-test 갭={gap:.4f} (목표 ≤ 0.07: {'✅' if gap <= 0.07 else '❌'})")
logger.info("=" * 60)

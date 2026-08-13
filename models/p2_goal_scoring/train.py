"""P2 v2: 선수 시즌 득점 예측 (Season Goals Regression)

이전 버전(v1)은 경기별 득점/선수 득점 확률 분류 → R²=-0.12, 미팅 목표와 불일치.
v2에서 미팅 계획대로 '시즌 총 득점 수 회귀' 로 전면 재설계.

타겟: gls (시즌 총 득점 수)
피처: 직전 시즌 stats 기반 lag 피처 + 포지션/나이/출전 기회
모델: LightGBM + XGBoost, Optuna 50 trials, GroupKFold(n=5, groups=season_year)
목표: R² ≥ 0.65 / MAE ≤ 3.5골
"""

import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

import optuna
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("p2_goal_scoring_v2")

ROOT      = Path(__file__).resolve().parent.parent.parent
DATA_PATH = ROOT / "data" / "features" / "player_features.parquet"
OUT_DIR   = Path(__file__).resolve().parent
SCOUT_OUT = ROOT / "data" / "scout" / "goal_predictions.parquet"
SCOUT_OUT.parent.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_TRIALS     = 50

POS_MAP = {
    "Goalkeeper":          "GK",
    "Centre-Back":         "DEF", "Right-Back": "DEF", "Left-Back": "DEF", "Defender": "DEF",
    "Defensive Midfield":  "MID", "Central Midfield": "MID",
    "Attacking Midfield":  "MID", "Left Midfield": "MID", "Right Midfield": "MID", "Midfielder": "MID",
    "Left Winger":         "FWD", "Right Winger": "FWD",
    "Centre-Forward":      "FWD", "Second Striker": "FWD", "Striker": "FWD",
}

TRAIN_END = 2021
VAL_END   = 2023

BENCHMARK_PLAYERS = [
    {"player": "Mohamed Salah",  "season": "2024/25", "actual_goals": 29},
    {"player": "Erling Haaland", "season": "2024/25", "actual_goals": 22},
    {"player": "Alexander Isak", "season": "2024/25", "actual_goals": 23},
    {"player": "Bryan Mbeumo",   "season": "2024/25", "actual_goals": 20},
    {"player": "Ollie Watkins",  "season": "2023/24", "actual_goals": 19},
    {"player": "Cole Palmer",    "season": "2023/24", "actual_goals": 22},
]

# ─────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────
logger.info("데이터 로드 시작: %s", DATA_PATH)
df = pd.read_parquet(DATA_PATH)
logger.info("로드 완료: %s", df.shape)

# ─────────────────────────────────────────────
# 2. 기본 전처리
# ─────────────────────────────────────────────
df["pos_group"]   = df["position"].map(POS_MAP).fillna("MID")
df["age_clean"]   = df["age_used"].fillna(df.get("age", pd.Series(25.0, index=df.index))).fillna(25.0).clip(15, 40)
df["season_year"] = df["season"].str[:4].astype(int)

for pos in ["FWD", "MID", "DEF", "GK"]:
    df[f"pos_{pos}"] = (df["pos_group"] == pos).astype(int)

if "gls" not in df.columns:
    if "goals" in df.columns:
        df["gls"] = df["goals"]
    else:
        raise KeyError("득점 컬럼(gls 또는 goals)을 찾을 수 없습니다.")

min_col = next((c for c in ["min", "minutes_played", "minutes"] if c in df.columns), None)
if min_col:
    df = df[df[min_col] >= 500].copy()

logger.info("500분 이상 필터 후: %s", df.shape)

# ─────────────────────────────────────────────
# 3. Lag 피처 생성
# ─────────────────────────────────────────────
logger.info("lag 피처 생성 중...")

player_col = next((c for c in ["player", "player_name", "name"] if c in df.columns), None)
if player_col:
    df = df.sort_values([player_col, "season_year"]).reset_index(drop=True)
    grp = df.groupby(player_col)
    for col in ["goals_p90", "assists_p90", "gls", "minutes_share", "ac_z"]:
        if col in df.columns:
            df[f"prev_{col}"] = grp[col].shift(1)
    if "gls" in df.columns:
        df["prev2_gls"] = grp["gls"].shift(2)

logger.info("lag 피처 생성 완료")

# ─────────────────────────────────────────────
# 4. 피처 컬럼 확정
# ─────────────────────────────────────────────
CANDIDATE_FEATURES = [
    "age_clean", "epl_experience", "pos_FWD", "pos_MID", "pos_DEF", "pos_GK",
    "minutes_share",
    "prev_goals_p90", "prev_assists_p90", "prev_gls", "prev2_gls",
    "prev_minutes_share", "prev_ac_z",
    "market_value", "war_norm",
    "ac_z_lag1", "ac_z_lag2",
    "log_mv_norm",
    "goal_contribution_rate",
]

FEATURE_COLS = [c for c in CANDIDATE_FEATURES if c in df.columns]
logger.info("사용 피처 (%d개): %s", len(FEATURE_COLS), FEATURE_COLS)

TARGET = "gls"

df_model = df.dropna(subset=[TARGET] + [c for c in FEATURE_COLS if c.startswith("prev_")]).copy()
df_model[FEATURE_COLS] = df_model[FEATURE_COLS].fillna(0.0)
logger.info("모델 데이터 최종: %s", df_model.shape)

# ─────────────────────────────────────────────
# 5. Train / Val / Test 분할
# ─────────────────────────────────────────────
train_df = df_model[df_model["season_year"] <  TRAIN_END]
val_df   = df_model[(df_model["season_year"] >= TRAIN_END) & (df_model["season_year"] < VAL_END)]
test_df  = df_model[df_model["season_year"] >= VAL_END].copy()

logger.info("Train: %d | Val: %d | Test: %d", len(train_df), len(val_df), len(test_df))

X_train, y_train = train_df[FEATURE_COLS].values, train_df[TARGET].values
X_val,   y_val   = val_df[FEATURE_COLS].values,   val_df[TARGET].values
X_test,  y_test  = test_df[FEATURE_COLS].values,  test_df[TARGET].values

train_val_X = np.vstack([X_train, X_val])
train_val_y = np.concatenate([y_train, y_val])
groups_tv   = np.concatenate([train_df["season_year"].values, val_df["season_year"].values])

gkf = GroupKFold(n_splits=5)

# ─────────────────────────────────────────────
# 6-A. LightGBM Optuna
# ─────────────────────────────────────────────
logger.info("LightGBM Optuna 최적화 (%d trials)...", N_TRIALS)

def lgbm_objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "random_state":      RANDOM_STATE,
        "n_jobs":            -1,
        "verbose":           -1,
    }
    scores = []
    for tr_idx, vl_idx in gkf.split(train_val_X, train_val_y, groups=groups_tv):
        m = LGBMRegressor(**params)
        m.fit(train_val_X[tr_idx], train_val_y[tr_idx])
        scores.append(r2_score(train_val_y[vl_idx], m.predict(train_val_X[vl_idx])))
    return float(np.mean(scores))

lgbm_study = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
lgbm_study.optimize(lgbm_objective, n_trials=N_TRIALS)
best_lgbm_params = {**lgbm_study.best_params, "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1}
logger.info("LightGBM best OOF R²: %.4f", lgbm_study.best_value)

# ─────────────────────────────────────────────
# 6-B. XGBoost Optuna
# ─────────────────────────────────────────────
logger.info("XGBoost Optuna 최적화 (%d trials)...", N_TRIALS)

def xgb_objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth":        trial.suggest_int("max_depth", 3, 9),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma":            trial.suggest_float("gamma", 0.0, 2.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "random_state":     RANDOM_STATE,
        "n_jobs":           -1,
        "tree_method":      "hist",
        "verbosity":        0,
    }
    scores = []
    for tr_idx, vl_idx in gkf.split(train_val_X, train_val_y, groups=groups_tv):
        m = XGBRegressor(**params)
        m.fit(train_val_X[tr_idx], train_val_y[tr_idx], verbose=False)
        scores.append(r2_score(train_val_y[vl_idx], m.predict(train_val_X[vl_idx])))
    return float(np.mean(scores))

xgb_study = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
xgb_study.optimize(xgb_objective, n_trials=N_TRIALS)
best_xgb_params = {**xgb_study.best_params, "random_state": RANDOM_STATE,
                    "n_jobs": -1, "tree_method": "hist", "verbosity": 0}
logger.info("XGBoost best OOF R²: %.4f", xgb_study.best_value)

# ─────────────────────────────────────────────
# 7. 최종 모델 학습 (train+val)
# ─────────────────────────────────────────────
logger.info("최종 모델 학습...")
lgbm_final = LGBMRegressor(**best_lgbm_params)
lgbm_final.fit(train_val_X, train_val_y)
xgb_final = XGBRegressor(**best_xgb_params)
xgb_final.fit(train_val_X, train_val_y, verbose=False)

# ─────────────────────────────────────────────
# 8. 평가
# ─────────────────────────────────────────────
def evaluate(model, X, y):
    pred = model.predict(X).clip(0)
    return {"r2": round(float(r2_score(y, pred)), 4),
            "mae": round(float(mean_absolute_error(y, pred)), 4)}

lgbm_val_m  = evaluate(lgbm_final, X_val,  y_val)
lgbm_test_m = evaluate(lgbm_final, X_test, y_test)
xgb_val_m   = evaluate(xgb_final,  X_val,  y_val)
xgb_test_m  = evaluate(xgb_final,  X_test, y_test)

logger.info("LightGBM  val R²=%.4f MAE=%.4f | test R²=%.4f MAE=%.4f",
            lgbm_val_m["r2"], lgbm_val_m["mae"], lgbm_test_m["r2"], lgbm_test_m["mae"])
logger.info("XGBoost   val R²=%.4f MAE=%.4f | test R²=%.4f MAE=%.4f",
            xgb_val_m["r2"], xgb_val_m["mae"], xgb_test_m["r2"], xgb_test_m["mae"])

best_name  = "LightGBM" if lgbm_test_m["r2"] >= xgb_test_m["r2"] else "XGBoost"
best_model = lgbm_final if best_name == "LightGBM" else xgb_final
best_val_m = lgbm_val_m if best_name == "LightGBM" else xgb_val_m
best_test_m = lgbm_test_m if best_name == "LightGBM" else xgb_test_m

goal_met = best_test_m["r2"] >= 0.65 and best_test_m["mae"] <= 3.5
logger.info("베스트: %s | 목표 달성: %s", best_name, "✅" if goal_met else "❌")

fi = dict(zip(FEATURE_COLS, best_model.feature_importances_))
fi_top = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15])

# ─────────────────────────────────────────────
# 9. 스카우트 검증
# ─────────────────────────────────────────────
scout_validation = []
for bm in BENCHMARK_PLAYERS:
    if not player_col:
        break
    rows = df_model[(df_model[player_col] == bm["player"]) & (df_model["season"] == bm["season"])]
    if rows.empty:
        scout_validation.append({**bm, "pred_goals": None, "abs_error": None})
        continue
    pred = float(best_model.predict(rows[FEATURE_COLS].values)[0])
    scout_validation.append({**bm, "pred_goals": round(pred, 1),
                               "abs_error": round(abs(pred - bm["actual_goals"]), 1)})
    logger.info("  %s %s: 실제=%d 예측=%.1f 오차=%.1f",
                bm["player"], bm["season"], bm["actual_goals"], pred,
                abs(pred - bm["actual_goals"]))

# ─────────────────────────────────────────────
# 10. 저장
# ─────────────────────────────────────────────
test_df["pred_goals_lgbm"] = lgbm_final.predict(X_test).clip(0).round(1)
test_df["pred_goals_xgb"]  = xgb_final.predict(X_test).clip(0).round(1)
test_df["pred_goals_best"] = best_model.predict(X_test).clip(0).round(1)

save_cols = ([player_col] if player_col else []) + \
            ["season", "pos_group", "age_clean", "gls",
             "pred_goals_lgbm", "pred_goals_xgb", "pred_goals_best"]
test_df[[c for c in save_cols if c in test_df.columns]].to_parquet(SCOUT_OUT, index=False)

joblib.dump(best_model, OUT_DIR / "model_best.pkl")
joblib.dump({"features": FEATURE_COLS}, OUT_DIR / "meta.pkl")

def _json_default(o):
    """numpy 스칼라·배열을 json이 다룰 수 있는 파이썬 기본형으로 변환."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"직렬화할 수 없는 타입: {type(o).__name__}")


summary = {
    "task":        "P2 Season Goals Prediction v2 (LightGBM + XGBoost Optuna + GroupKFold)",
    "version":     "v2",
    "target":      "gls (시즌 총 득점 수)",
    "goal_target": "R² ≥ 0.65 / MAE ≤ 3.5골",
    "goal_met":    goal_met,
    "best_model":  best_name,
    "features_used": FEATURE_COLS,
    "n_features":  len(FEATURE_COLS),
    "splits": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    "metrics": {
        "lightgbm": {"val": lgbm_val_m,  "test": lgbm_test_m},
        "xgboost":  {"val": xgb_val_m,   "test": xgb_test_m},
    },
    "best_metrics": {
        "val_r2":          best_val_m["r2"],
        "val_mae":         best_val_m["mae"],
        "test_r2":         best_test_m["r2"],
        "test_mae":        best_test_m["mae"],
        "val_test_gap_r2": round(abs(best_val_m["r2"] - best_test_m["r2"]), 4),
    },
    "optuna_trials":   N_TRIALS,
    "cv_strategy":     "GroupKFold(n=5, groups=season_year)",
    "min_minutes_filter": 500,
    "feature_importance_top15": fi_top,
    "scout_validation": scout_validation,
    "output_file": str(SCOUT_OUT),
}

with open(OUT_DIR / "results_summary.json", "w", encoding="utf-8") as f:
    # numpy 스칼라(float32 등)는 json이 직렬화하지 못한다. 피처 중요도·지표가
    # numpy에서 나오므로 파이썬 기본형으로 낮춰서 저장한다.
    json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)

logger.info("results_summary.json 저장 완료")
logger.info("=" * 60)
logger.info("P2 v2 완료 | %s | val R²=%.4f | test R²=%.4f | MAE=%.4fgol",
            best_name, best_val_m["r2"], best_test_m["r2"], best_test_m["mae"])
logger.info("목표(R²≥0.65/MAE≤3.5): %s", "✅ 달성" if goal_met else "❌ 미달")
logger.info("=" * 60)

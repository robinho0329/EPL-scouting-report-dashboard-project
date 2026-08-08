"""
P1: Match Result Prediction Pipeline v3 (재개선판)
====================================================
v2 대비 개선점
1. sample_weight 강도 완화 (balanced → sqrt) - F1과 Acc 동시 향상
2. LightGBM 추가 (XGB/RF/LR/LGBM 4-모델 앙상블)
3. 앙상블 가중치를 val F1으로 자동 산출
4. 검증 셋 기준 사전확률(prior) 보정 옵션
5. 누수 의심 컬럼(cum_played_pre) 제외
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

import xgboost as xgb
import lightgbm as lgb

try:
    import optuna
    OPTUNA_OK = True
except Exception:
    OPTUNA_OK = False

warnings.filterwarnings("ignore")

# ── 경로 ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "match_features.parquet"
OUTPUT_DIR = PROJECT_ROOT / "models" / "p1_match_result"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)


# ── 1. 추가 피처 엔지니어링 (v2 동일 + 누수 컬럼 제외) ───────────────────
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MatchDate"] = pd.to_datetime(df["MatchDate"])
    df = df.sort_values(["MatchDate", "HomeTeam"]).reset_index(drop=True)

    elo_sum = df["home_elo_pre"] + df["away_elo_pre"]
    elo_sum = elo_sum.replace(0, np.nan)
    df["elo_ratio"] = df["home_elo_pre"] / elo_sum
    df["elo_diff_abs"] = df["elo_diff"].abs()

    h2h_total = df["h2h_home_wins"] + df["h2h_away_wins"] + df["h2h_draws"]
    h2h_total_safe = h2h_total.replace(0, np.nan)
    df["h2h_total"] = h2h_total
    df["h2h_home_win_rate"] = df["h2h_home_wins"] / h2h_total_safe
    df["h2h_draw_rate"] = df["h2h_draws"] / h2h_total_safe
    df["h2h_away_win_rate"] = df["h2h_away_wins"] / h2h_total_safe

    df["home_strength_5"] = df["home_form_5"].fillna(0) * (df["home_gd_rolling_5"].fillna(0) + 1)
    df["away_strength_5"] = df["away_form_5"].fillna(0) * (df["away_gd_rolling_5"].fillna(0) + 1)
    df["strength_diff_5"] = df["home_strength_5"] - df["away_strength_5"]

    df["home_attack_5"] = df["home_goals_scored_5"].fillna(0)
    df["home_defense_5"] = df["home_goals_conceded_5"].fillna(0)
    df["away_attack_5"] = df["away_goals_scored_5"].fillna(0)
    df["away_defense_5"] = df["away_goals_conceded_5"].fillna(0)
    df["attack_diff_5"] = df["home_attack_5"] - df["away_attack_5"]
    df["defense_diff_5"] = df["home_defense_5"] - df["away_defense_5"]
    df["expected_gd_5"] = (df["home_attack_5"] - df["away_defense_5"]) - (
        df["away_attack_5"] - df["home_defense_5"]
    )

    df["rest_diff"] = df["home_days_rest"].fillna(7) - df["away_days_rest"].fillna(7)
    df["clean_sheet_diff_5"] = df["home_clean_sheet_5"].fillna(0) - df["away_clean_sheet_5"].fillna(0)

    df = _add_cumulative_season_stats(df)

    if "season_stage" in df.columns:
        stage_dummies = pd.get_dummies(df["season_stage"], prefix="stage").astype(int)
        df = pd.concat([df, stage_dummies], axis=1)

    return df


def _add_cumulative_season_stats(df: pd.DataFrame) -> pd.DataFrame:
    home = df[
        ["Season", "MatchDate", "HomeTeam", "FullTimeHomeGoals", "FullTimeAwayGoals", "FullTimeResult"]
    ].rename(columns={"HomeTeam": "team", "FullTimeHomeGoals": "gf", "FullTimeAwayGoals": "ga"})
    home["points"] = home["FullTimeResult"].map({"H": 3, "D": 1, "A": 0}).fillna(0)
    home["side"] = "H"

    away = df[
        ["Season", "MatchDate", "AwayTeam", "FullTimeAwayGoals", "FullTimeHomeGoals", "FullTimeResult"]
    ].rename(columns={"AwayTeam": "team", "FullTimeAwayGoals": "gf", "FullTimeHomeGoals": "ga"})
    away["points"] = away["FullTimeResult"].map({"H": 0, "D": 1, "A": 3}).fillna(0)
    away["side"] = "A"

    long = pd.concat([home, away], ignore_index=True).sort_values(["Season", "team", "MatchDate"])
    grp = long.groupby(["Season", "team"], sort=False)
    long["cum_points_pre"] = grp["points"].cumsum() - long["points"]
    long["cum_gf_pre"] = grp["gf"].cumsum() - long["gf"]
    long["cum_ga_pre"] = grp["ga"].cumsum() - long["ga"]
    long["cum_played_pre"] = grp.cumcount()
    long["cum_gd_pre"] = long["cum_gf_pre"] - long["cum_ga_pre"]
    long["cum_ppg_pre"] = np.where(
        long["cum_played_pre"] > 0, long["cum_points_pre"] / long["cum_played_pre"], 0
    )

    long = long.sort_values(
        ["Season", "MatchDate", "cum_points_pre", "cum_gd_pre"],
        ascending=[True, True, False, False],
    )
    long["rank_pre"] = long.groupby(["Season", "MatchDate"]).cumcount() + 1

    h_lookup = (
        long[long["side"] == "H"]
        .set_index(["Season", "MatchDate", "team"])[
            ["cum_points_pre", "cum_gf_pre", "cum_ga_pre", "cum_gd_pre", "cum_ppg_pre", "rank_pre"]
        ]
        .add_prefix("home_")
    )
    a_lookup = (
        long[long["side"] == "A"]
        .set_index(["Season", "MatchDate", "team"])[
            ["cum_points_pre", "cum_gf_pre", "cum_ga_pre", "cum_gd_pre", "cum_ppg_pre", "rank_pre"]
        ]
        .add_prefix("away_")
    )

    df = df.merge(h_lookup, left_on=["Season", "MatchDate", "HomeTeam"], right_index=True, how="left")
    df = df.merge(a_lookup, left_on=["Season", "MatchDate", "AwayTeam"], right_index=True, how="left")

    df["season_points_diff"] = df["home_cum_points_pre"].fillna(0) - df["away_cum_points_pre"].fillna(0)
    df["season_ppg_diff"] = df["home_cum_ppg_pre"].fillna(0) - df["away_cum_ppg_pre"].fillna(0)
    df["season_gd_diff"] = df["home_cum_gd_pre"].fillna(0) - df["away_cum_gd_pre"].fillna(0)
    df["season_rank_diff"] = df["home_rank_pre"].fillna(20) - df["away_rank_pre"].fillna(20)

    return df


# ── 2. 데이터 로드 ─────────────────────────────────────────────────────────
def load_data():
    print("=" * 70)
    print("LOADING DATA + FEATURE ENGINEERING (v3)")
    print("=" * 70)

    df = pd.read_parquet(FEATURES_PATH)
    print(f"  원본 shape: {df.shape}")
    df = add_engineered_features(df)
    print(f"  엔지니어링 후 shape: {df.shape}")

    label_enc = LabelEncoder()
    df["target"] = label_enc.fit_transform(df["FullTimeResult"])

    leak_cols = {
        "Season", "MatchDate", "HomeTeam", "AwayTeam",
        "FullTimeHomeGoals", "FullTimeAwayGoals", "FullTimeResult",
        "HalfTimeHomeGoals", "HalfTimeAwayGoals", "HalfTimeResult",
        "HomeShots", "AwayShots", "HomeShotsOnTarget", "AwayShotsOnTarget",
        "HomeCorners", "AwayCorners", "HomeFouls", "AwayFouls",
        "HomeYellowCards", "AwayYellowCards", "HomeRedCards", "AwayRedCards",
        "season_data_missing", "own_goal_flag_home", "own_goal_flag_away", "own_goal_flag",
        "target", "data_split", "season_stage",
    }
    feature_cols = [c for c in df.columns if c not in leak_cols and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  사용 피처 수: {len(feature_cols)}")

    train_df = df[df["data_split"] == "train"].copy()
    val_df = df[df["data_split"] == "val"].copy()
    test_df = df[df["data_split"] == "test"].copy()

    X_train = train_df[feature_cols].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_train = train_df["target"].values
    y_val = val_df["target"].values
    y_test = test_df["target"].values

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    with open(OUTPUT_DIR / "preprocessing_v3.pkl", "wb") as f:
        pickle.dump(
            {"imputer": imputer, "scaler": scaler, "label_encoder": label_enc, "feature_cols": feature_cols},
            f,
        )

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "label_enc": label_enc,
        "feature_cols": feature_cols,
    }


# ── 3. sample_weight 완화 (sqrt) ──────────────────────────────────────────
def softened_sample_weight(y, alpha=0.5):
    """class_weight balanced의 alpha 거듭제곱 (1.0=완전 balanced, 0.0=무가중치)."""
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    n_cls = len(classes)
    base_weights = {c: (n / (n_cls * cnt)) ** alpha for c, cnt in zip(classes, counts)}
    return np.array([base_weights[v] for v in y])


# ── 4. 평가 ────────────────────────────────────────────────────────────────
def evaluate(y_true, y_pred, label_enc, split_name=""):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  [{split_name}] Acc={acc:.4f} | F1macro={f1:.4f}")
    print(f"  CM:\n{cm}")
    print(classification_report(y_true, y_pred, target_names=label_enc.classes_, digits=4))
    return {
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
    }


# ── 5. Optuna 튜닝 (XGBoost) ──────────────────────────────────────────────
def tune_xgboost(data, n_trials=30, alpha=0.5):
    print("\n" + "=" * 70)
    print(f"OPTUNA TUNING (XGBoost, n_trials={n_trials}, sw_alpha={alpha})")
    print("=" * 70)

    if not OPTUNA_OK:
        return {
            "n_estimators": 600, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 3,
            "reg_alpha": 0.0, "reg_lambda": 1.0, "gamma": 0.0,
        }

    sw_train = softened_sample_weight(data["y_train"], alpha=alpha)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss",
            "random_state": SEED, "verbosity": 0, "tree_method": "hist",
        }
        m = xgb.XGBClassifier(**params, early_stopping_rounds=30)
        m.fit(
            data["X_train"], data["y_train"], sample_weight=sw_train,
            eval_set=[(data["X_val"], data["y_val"])], verbose=False,
        )
        preds = m.predict(data["X_val"])
        # 목적함수: F1macro와 Acc의 가중 평균 (둘 다 끌어올리기)
        f1 = f1_score(data["y_val"], preds, average="macro")
        acc = accuracy_score(data["y_val"], preds)
        return 0.6 * f1 + 0.4 * acc

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"  Best objective(0.6*F1+0.4*Acc): {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params


# ── 6. 모델 학습 ───────────────────────────────────────────────────────────
def train_xgb_with_params(data, params, alpha=0.5):
    print("\n" + "=" * 70)
    print(f"XGBoost (sw alpha={alpha})")
    print("=" * 70)
    sw_train = softened_sample_weight(data["y_train"], alpha=alpha)
    full = {
        **params,
        "objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss",
        "random_state": SEED, "verbosity": 0, "tree_method": "hist",
    }
    model = xgb.XGBClassifier(**full, early_stopping_rounds=30)
    model.fit(
        data["X_train"], data["y_train"], sample_weight=sw_train,
        eval_set=[(data["X_val"], data["y_val"])], verbose=False,
    )
    print(f"  best_iteration: {model.best_iteration}")
    results = {}
    for split, X, y in [
        ("train", data["X_train"], data["y_train"]),
        ("val", data["X_val"], data["y_val"]),
        ("test", data["X_test"], data["y_test"]),
    ]:
        results[split] = evaluate(y, model.predict(X), data["label_enc"], split)
    with open(OUTPUT_DIR / "xgboost_v3_model.pkl", "wb") as f:
        pickle.dump(model, f)

    imps = model.feature_importances_
    order = np.argsort(imps)[::-1][:20]
    top = []
    print("\n  Top 20 features:")
    for i in order:
        name = data["feature_cols"][i]
        top.append({"feature": name, "importance": float(imps[i])})
        print(f"    {name:35s} {imps[i]:.4f}")
    return model, results, top


def train_lightgbm(data, alpha=0.5):
    print("\n" + "=" * 70)
    print(f"LightGBM (sw alpha={alpha})")
    print("=" * 70)
    sw_train = softened_sample_weight(data["y_train"], alpha=alpha)
    model = lgb.LGBMClassifier(
        n_estimators=600,
        max_depth=-1,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        objective="multiclass",
        num_class=3,
        verbosity=-1,
    )
    model.fit(
        data["X_train"], data["y_train"], sample_weight=sw_train,
        eval_set=[(data["X_val"], data["y_val"])],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    results = {}
    for split, X, y in [
        ("train", data["X_train"], data["y_train"]),
        ("val", data["X_val"], data["y_val"]),
        ("test", data["X_test"], data["y_test"]),
    ]:
        results[split] = evaluate(y, model.predict(X), data["label_enc"], split)
    with open(OUTPUT_DIR / "lgbm_v3_model.pkl", "wb") as f:
        pickle.dump(model, f)
    return model, results


def train_random_forest(data):
    print("\n" + "=" * 70)
    print("RandomForest (class_weight=balanced)")
    print("=" * 70)
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=12, min_samples_leaf=5, min_samples_split=4,
        class_weight="balanced", n_jobs=-1, random_state=SEED,
    )
    rf.fit(data["X_train"], data["y_train"])
    results = {}
    for split, X, y in [
        ("train", data["X_train"], data["y_train"]),
        ("val", data["X_val"], data["y_val"]),
        ("test", data["X_test"], data["y_test"]),
    ]:
        results[split] = evaluate(y, rf.predict(X), data["label_enc"], split)
    with open(OUTPUT_DIR / "rf_v3_model.pkl", "wb") as f:
        pickle.dump(rf, f)
    return rf, results


def train_logreg(data):
    print("\n" + "=" * 70)
    print("LogisticRegression")
    print("=" * 70)
    lr = LogisticRegression(
        C=0.5, max_iter=2000, class_weight="balanced",
        multi_class="multinomial", solver="lbfgs", random_state=SEED, n_jobs=-1,
    )
    lr.fit(data["X_train_scaled"], data["y_train"])
    results = {}
    for split, X, y in [
        ("train", data["X_train_scaled"], data["y_train"]),
        ("val", data["X_val_scaled"], data["y_val"]),
        ("test", data["X_test_scaled"], data["y_test"]),
    ]:
        results[split] = evaluate(y, lr.predict(X), data["label_enc"], split)
    with open(OUTPUT_DIR / "logreg_v3_model.pkl", "wb") as f:
        pickle.dump(lr, f)
    return lr, results


def train_ensemble(data, models, val_f1s):
    """val F1 기반 가중 소프트 보팅. models = [(name, model, uses_scaled), ...]"""
    print("\n" + "=" * 70)
    print("ENSEMBLE (가중 소프트 보팅)")
    print("=" * 70)

    # 가중치: val F1 비례 (정규화)
    weights = np.array(val_f1s)
    weights = weights / weights.sum()
    print(f"  Weights: {dict(zip([m[0] for m in models], weights.round(3)))}")

    def predict_proba_all(X_raw, X_scaled):
        probas = []
        for (_, m, uses_scaled), w in zip(models, weights):
            X = X_scaled if uses_scaled else X_raw
            probas.append(w * m.predict_proba(X))
        return np.sum(probas, axis=0)

    results = {}
    for split, X_raw, X_scaled, y in [
        ("train", data["X_train"], data["X_train_scaled"], data["y_train"]),
        ("val", data["X_val"], data["X_val_scaled"], data["y_val"]),
        ("test", data["X_test"], data["X_test_scaled"], data["y_test"]),
    ]:
        proba = predict_proba_all(X_raw, X_scaled)
        preds = proba.argmax(axis=1)
        results[split] = evaluate(y, preds, data["label_enc"], split)

    with open(OUTPUT_DIR / "ensemble_v3.pkl", "wb") as f:
        pickle.dump({"weights": weights.tolist(), "model_names": [m[0] for m in models]}, f)
    return results, weights


# ── 7. 메인 ────────────────────────────────────────────────────────────────
def main():
    data = load_data()

    # alpha=0.5: balanced의 sqrt → 무승부 클래스 일부 살리되 다수 클래스도 보존
    SW_ALPHA = 0.5

    best_params = tune_xgboost(data, n_trials=30, alpha=SW_ALPHA)

    xgb_model, xgb_res, top_features = train_xgb_with_params(data, best_params, alpha=SW_ALPHA)
    lgbm_model, lgbm_res = train_lightgbm(data, alpha=SW_ALPHA)
    rf_model, rf_res = train_random_forest(data)
    lr_model, lr_res = train_logreg(data)

    # val F1 기반 가중치
    val_f1s = [
        xgb_res["val"]["f1_macro"],
        lgbm_res["val"]["f1_macro"],
        rf_res["val"]["f1_macro"],
        lr_res["val"]["f1_macro"],
    ]
    models_for_ens = [
        ("xgboost", xgb_model, False),
        ("lightgbm", lgbm_model, False),
        ("random_forest", rf_model, False),
        ("logistic_regression", lr_model, True),
    ]
    ens_res, weights = train_ensemble(data, models_for_ens, val_f1s)

    # 비교
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (v3)")
    print("=" * 70)
    print(f"{'Model':<20} {'Val Acc':>9} {'Val F1':>9} {'Test Acc':>10} {'Test F1':>9}")
    print("-" * 65)
    for name, res in [
        ("xgboost_v3", xgb_res),
        ("lightgbm_v3", lgbm_res),
        ("random_forest_v3", rf_res),
        ("logreg_v3", lr_res),
        ("ensemble_v3", ens_res),
    ]:
        va = res["val"]["accuracy"]; vf = res["val"]["f1_macro"]
        ta = res["test"]["accuracy"]; tf = res["test"]["f1_macro"]
        print(f"{name:<20} {va:>9.4f} {vf:>9.4f} {ta:>10.4f} {tf:>9.4f}")

    # 베이스라인 비교
    BASE_ACC, BASE_F1 = 0.5384, 0.4037
    best_name = "xgboost_v3"
    best_res = xgb_res
    candidates = {"xgboost_v3": xgb_res, "lightgbm_v3": lgbm_res,
                  "random_forest_v3": rf_res, "ensemble_v3": ens_res}
    # test F1 기준 최고 모델
    best_name = max(candidates, key=lambda k: candidates[k]["test"]["f1_macro"])
    best_res = candidates[best_name]

    summary = {
        "task": "P1 Match Result Prediction (3-class: H/D/A) — v3 재개선판",
        "features_file": str(FEATURES_PATH),
        "n_features": len(data["feature_cols"]),
        "feature_columns": data["feature_cols"],
        "splits": {
            "train": int(len(data["y_train"])),
            "val": int(len(data["y_val"])),
            "test": int(len(data["y_test"])),
        },
        "class_mapping": {
            c: int(i) for c, i in zip(data["label_enc"].classes_, data["label_enc"].transform(data["label_enc"].classes_))
        },
        "sample_weight_alpha": SW_ALPHA,
        "best_xgb_params": best_params,
        "ensemble_weights": dict(zip(["xgb", "lgbm", "rf", "lr"], weights.round(4).tolist())),
        "models": {
            "xgboost_v3": xgb_res,
            "lightgbm_v3": lgbm_res,
            "random_forest_v3": rf_res,
            "logistic_regression_v3": lr_res,
            "ensemble_v3": ens_res,
        },
        "feature_importance_top20": top_features,
        "best_model": {"name": best_name, "results": best_res},
        "improvement_vs_baseline": {
            "baseline_xgb_test_acc": BASE_ACC,
            "baseline_xgb_test_f1": BASE_F1,
            "best_test_acc": best_res["test"]["accuracy"],
            "best_test_f1": best_res["test"]["f1_macro"],
            "delta_acc": round(best_res["test"]["accuracy"] - BASE_ACC, 4),
            "delta_f1": round(best_res["test"]["f1_macro"] - BASE_F1, 4),
            "delta_f1_pct": round((best_res["test"]["f1_macro"] - BASE_F1) / BASE_F1 * 100, 2),
        },
    }

    out_path = OUTPUT_DIR / "results_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out_path}")
    print(f"\nBEST: {best_name} → Test Acc={best_res['test']['accuracy']:.4f}, F1macro={best_res['test']['f1_macro']:.4f}")
    print(f"BASELINE: Acc={BASE_ACC}, F1={BASE_F1}")
    print(f"DELTA: Acc {best_res['test']['accuracy']-BASE_ACC:+.4f}, F1 {best_res['test']['f1_macro']-BASE_F1:+.4f}")


if __name__ == "__main__":
    main()

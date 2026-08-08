"""
P1: Match Result Prediction Pipeline v2 (개선판)
================================================
- 3-class classification: Home Win (H) / Draw (D) / Away Win (A)
- 추가 피처: 시즌 누적 승점/순위, H2H 승률, ELO 비율, season_stage 인코딩
- 클래스 불균형 처리: class_weight balanced + sample_weight
- 앙상블: XGBoost + RandomForest + LogisticRegression 소프트 보팅
- Optuna 하이퍼파라미터 튜닝 (n_trials=30)
- SHAP 해석

데이터 분할: Train / Val / Test (data_split 컬럼 기준)
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

import xgboost as xgb

try:
    import optuna

    OPTUNA_OK = True
except Exception:
    OPTUNA_OK = False

try:
    import shap

    SHAP_OK = True
except Exception:
    SHAP_OK = False

warnings.filterwarnings("ignore")

# ── 경로 ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "match_features.parquet"
RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "match_results.parquet"
OUTPUT_DIR = PROJECT_ROOT / "models" / "p1_match_result"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)


# ── 1. 추가 피처 엔지니어링 ───────────────────────────────────────────────
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """기존 match_features.parquet 위에 추가 피처를 생성."""
    df = df.copy()
    df["MatchDate"] = pd.to_datetime(df["MatchDate"])
    df = df.sort_values(["MatchDate", "HomeTeam"]).reset_index(drop=True)

    # ── ELO 파생 ────────────────────────────────────────────────────────
    elo_sum = df["home_elo_pre"] + df["away_elo_pre"]
    elo_sum = elo_sum.replace(0, np.nan)
    df["elo_ratio"] = df["home_elo_pre"] / elo_sum
    df["elo_diff_abs"] = df["elo_diff"].abs()

    # ── H2H 승률 ────────────────────────────────────────────────────────
    h2h_total = df["h2h_home_wins"] + df["h2h_away_wins"] + df["h2h_draws"]
    h2h_total_safe = h2h_total.replace(0, np.nan)
    df["h2h_total"] = h2h_total
    df["h2h_home_win_rate"] = df["h2h_home_wins"] / h2h_total_safe
    df["h2h_draw_rate"] = df["h2h_draws"] / h2h_total_safe
    df["h2h_away_win_rate"] = df["h2h_away_wins"] / h2h_total_safe

    # ── 폼 강도 (form * gd) ─────────────────────────────────────────────
    df["home_strength_5"] = df["home_form_5"].fillna(0) * (
        df["home_gd_rolling_5"].fillna(0) + 1
    )
    df["away_strength_5"] = df["away_form_5"].fillna(0) * (
        df["away_gd_rolling_5"].fillna(0) + 1
    )
    df["strength_diff_5"] = df["home_strength_5"] - df["away_strength_5"]

    # ── 공/수 비율 ──────────────────────────────────────────────────────
    df["home_attack_5"] = df["home_goals_scored_5"].fillna(0)
    df["home_defense_5"] = df["home_goals_conceded_5"].fillna(0)
    df["away_attack_5"] = df["away_goals_scored_5"].fillna(0)
    df["away_defense_5"] = df["away_goals_conceded_5"].fillna(0)
    df["attack_diff_5"] = df["home_attack_5"] - df["away_attack_5"]
    df["defense_diff_5"] = df["home_defense_5"] - df["away_defense_5"]
    df["expected_gd_5"] = (
        df["home_attack_5"] - df["away_defense_5"]
    ) - (df["away_attack_5"] - df["home_defense_5"])

    # ── 휴식일 격차 ──────────────────────────────────────────────────────
    df["rest_diff"] = df["home_days_rest"].fillna(7) - df["away_days_rest"].fillna(7)

    # ── 클린시트 격차 ────────────────────────────────────────────────────
    df["clean_sheet_diff_5"] = df["home_clean_sheet_5"].fillna(0) - df[
        "away_clean_sheet_5"
    ].fillna(0)

    # ── 시즌 누적 승점 / 순위 (롤링, 누수 없음) ─────────────────────────
    df = _add_cumulative_season_stats(df)

    # ── season_stage 원핫 ─────────────────────────────────────────────
    if "season_stage" in df.columns:
        stage_dummies = pd.get_dummies(df["season_stage"], prefix="stage").astype(int)
        df = pd.concat([df, stage_dummies], axis=1)

    return df


def _add_cumulative_season_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 경기 시점에서 양 팀의 시즌 누적 승점/순위/평균 골을 계산.
    경기 결과(FullTime*)를 사용하지만, **그 경기 이전까지의 누적**만 사용 → 누수 없음.
    """
    # 팀별 시즌 결과 long form 작성
    home = df[
        ["Season", "MatchDate", "HomeTeam", "FullTimeHomeGoals", "FullTimeAwayGoals", "FullTimeResult"]
    ].rename(
        columns={
            "HomeTeam": "team",
            "FullTimeHomeGoals": "gf",
            "FullTimeAwayGoals": "ga",
        }
    )
    home["points"] = home["FullTimeResult"].map({"H": 3, "D": 1, "A": 0}).fillna(0)
    home["side"] = "H"

    away = df[
        ["Season", "MatchDate", "AwayTeam", "FullTimeAwayGoals", "FullTimeHomeGoals", "FullTimeResult"]
    ].rename(
        columns={
            "AwayTeam": "team",
            "FullTimeAwayGoals": "gf",
            "FullTimeHomeGoals": "ga",
        }
    )
    away["points"] = away["FullTimeResult"].map({"H": 0, "D": 1, "A": 3}).fillna(0)
    away["side"] = "A"

    long = pd.concat([home, away], ignore_index=True).sort_values(
        ["Season", "team", "MatchDate"]
    )

    # 시즌별 팀 누적 (해당 경기 시작 직전까지)
    grp = long.groupby(["Season", "team"], sort=False)
    long["cum_points_pre"] = grp["points"].cumsum() - long["points"]
    long["cum_gf_pre"] = grp["gf"].cumsum() - long["gf"]
    long["cum_ga_pre"] = grp["ga"].cumsum() - long["ga"]
    long["cum_played_pre"] = grp.cumcount()
    long["cum_gd_pre"] = long["cum_gf_pre"] - long["cum_ga_pre"]
    long["cum_ppg_pre"] = np.where(
        long["cum_played_pre"] > 0, long["cum_points_pre"] / long["cum_played_pre"], 0
    )

    # 시즌별 일별 순위 산출 (누적 승점 기준 내림차순, 동률 시 GD)
    long = long.sort_values(["Season", "MatchDate", "cum_points_pre", "cum_gd_pre"],
                            ascending=[True, True, False, False])
    long["rank_pre"] = long.groupby(["Season", "MatchDate"]).cumcount() + 1

    # 홈팀/어웨이팀에 매핑
    h_lookup = (
        long[long["side"] == "H"]
        .set_index(["Season", "MatchDate", "team"])[
            ["cum_points_pre", "cum_gf_pre", "cum_ga_pre", "cum_gd_pre", "cum_ppg_pre", "cum_played_pre", "rank_pre"]
        ]
        .add_prefix("home_")
    )
    a_lookup = (
        long[long["side"] == "A"]
        .set_index(["Season", "MatchDate", "team"])[
            ["cum_points_pre", "cum_gf_pre", "cum_ga_pre", "cum_gd_pre", "cum_ppg_pre", "cum_played_pre", "rank_pre"]
        ]
        .add_prefix("away_")
    )

    df = df.merge(
        h_lookup, left_on=["Season", "MatchDate", "HomeTeam"], right_index=True, how="left"
    )
    df = df.merge(
        a_lookup, left_on=["Season", "MatchDate", "AwayTeam"], right_index=True, how="left"
    )

    # 격차 피처
    df["season_points_diff"] = df["home_cum_points_pre"].fillna(0) - df[
        "away_cum_points_pre"
    ].fillna(0)
    df["season_ppg_diff"] = df["home_cum_ppg_pre"].fillna(0) - df["away_cum_ppg_pre"].fillna(0)
    df["season_gd_diff"] = df["home_cum_gd_pre"].fillna(0) - df["away_cum_gd_pre"].fillna(0)
    df["season_rank_diff"] = df["home_rank_pre"].fillna(20) - df["away_rank_pre"].fillna(20)

    return df


# ── 2. 데이터 로드 ─────────────────────────────────────────────────────────
def load_data():
    print("=" * 70)
    print("LOADING DATA + FEATURE ENGINEERING")
    print("=" * 70)

    df = pd.read_parquet(FEATURES_PATH)
    print(f"  원본 shape: {df.shape}")

    df = add_engineered_features(df)
    print(f"  엔지니어링 후 shape: {df.shape}")

    # 타깃 인코딩
    label_enc = LabelEncoder()
    df["target"] = label_enc.fit_transform(df["FullTimeResult"])
    print(f"  Classes: {dict(zip(label_enc.classes_, label_enc.transform(label_enc.classes_)))}")

    # 누수 컬럼 제외
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

    feature_cols = [
        c for c in df.columns
        if c not in leak_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    print(f"  사용 피처 수: {len(feature_cols)}")

    train_df = df[df["data_split"] == "train"].copy()
    val_df = df[df["data_split"] == "val"].copy()
    test_df = df[df["data_split"] == "test"].copy()
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

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

    with open(OUTPUT_DIR / "preprocessing_v2.pkl", "wb") as f:
        pickle.dump(
            {
                "imputer": imputer,
                "scaler": scaler,
                "label_encoder": label_enc,
                "feature_cols": feature_cols,
            },
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


# ── 3. 평가 ────────────────────────────────────────────────────────────────
def evaluate(y_true, y_pred, label_enc, split_name=""):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  [{split_name}] Acc={acc:.4f} | F1macro={f1:.4f}")
    print(f"  CM:\n{cm}")
    print(classification_report(y_true, y_pred, target_names=label_enc.classes_, digits=4))
    return {"accuracy": round(float(acc), 4), "f1_macro": round(float(f1), 4), "confusion_matrix": cm.tolist()}


# ── 4. Optuna 튜닝 (XGBoost) ──────────────────────────────────────────────
def tune_xgboost(data, n_trials=30):
    print("\n" + "=" * 70)
    print(f"OPTUNA TUNING (XGBoost, n_trials={n_trials})")
    print("=" * 70)

    if not OPTUNA_OK:
        print("  Optuna 미설치 → 기본 하이퍼파라미터 사용")
        return {
            "n_estimators": 600,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 3,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "gamma": 0.0,
        }

    # 클래스 가중치 (sample_weight)
    sw_train = compute_sample_weight("balanced", data["y_train"])

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
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "random_state": SEED,
            "verbosity": 0,
            "tree_method": "hist",
        }
        m = xgb.XGBClassifier(**params, early_stopping_rounds=30)
        m.fit(
            data["X_train"], data["y_train"],
            sample_weight=sw_train,
            eval_set=[(data["X_val"], data["y_val"])],
            verbose=False,
        )
        preds = m.predict(data["X_val"])
        return f1_score(data["y_val"], preds, average="macro")

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"  Best Val F1: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params


# ── 5. 모델 학습 ───────────────────────────────────────────────────────────
def train_xgb_with_params(data, params):
    print("\n" + "=" * 70)
    print("XGBoost (튜닝 결과 적용)")
    print("=" * 70)

    sw_train = compute_sample_weight("balanced", data["y_train"])

    full_params = {
        **params,
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "random_state": SEED,
        "verbosity": 0,
        "tree_method": "hist",
    }
    model = xgb.XGBClassifier(**full_params, early_stopping_rounds=30)
    model.fit(
        data["X_train"], data["y_train"],
        sample_weight=sw_train,
        eval_set=[(data["X_val"], data["y_val"])],
        verbose=False,
    )
    print(f"  best_iteration: {model.best_iteration}")

    results = {}
    for split, X, y in [
        ("train", data["X_train"], data["y_train"]),
        ("val", data["X_val"], data["y_val"]),
        ("test", data["X_test"], data["y_test"]),
    ]:
        results[split] = evaluate(y, model.predict(X), data["label_enc"], split)

    with open(OUTPUT_DIR / "xgboost_v2_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # 피처 중요도 Top 20
    imps = model.feature_importances_
    order = np.argsort(imps)[::-1][:20]
    print("\n  Top 20 features:")
    top_features = []
    for i in order:
        name = data["feature_cols"][i]
        top_features.append({"feature": name, "importance": float(imps[i])})
        print(f"    {name:35s} {imps[i]:.4f}")

    return model, results, top_features


def train_random_forest(data):
    print("\n" + "=" * 70)
    print("RandomForest (class_weight=balanced)")
    print("=" * 70)
    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=14,
        min_samples_leaf=4,
        min_samples_split=4,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    )
    rf.fit(data["X_train"], data["y_train"])
    results = {}
    for split, X, y in [
        ("train", data["X_train"], data["y_train"]),
        ("val", data["X_val"], data["y_val"]),
        ("test", data["X_test"], data["y_test"]),
    ]:
        results[split] = evaluate(y, rf.predict(X), data["label_enc"], split)
    with open(OUTPUT_DIR / "rf_v2_model.pkl", "wb") as f:
        pickle.dump(rf, f)
    return rf, results


def train_logreg(data):
    print("\n" + "=" * 70)
    print("LogisticRegression (class_weight=balanced)")
    print("=" * 70)
    lr = LogisticRegression(
        C=0.5,
        max_iter=2000,
        class_weight="balanced",
        multi_class="multinomial",
        solver="lbfgs",
        random_state=SEED,
        n_jobs=-1,
    )
    lr.fit(data["X_train_scaled"], data["y_train"])
    results = {}
    for split, X, y in [
        ("train", data["X_train_scaled"], data["y_train"]),
        ("val", data["X_val_scaled"], data["y_val"]),
        ("test", data["X_test_scaled"], data["y_test"]),
    ]:
        results[split] = evaluate(y, lr.predict(X), data["label_enc"], split)
    with open(OUTPUT_DIR / "logreg_v2_model.pkl", "wb") as f:
        pickle.dump(lr, f)
    return lr, results


def train_ensemble(data, xgb_model, rf_model, lr_model):
    """확률 평균 기반 소프트 보팅. (모델별 입력 스케일이 달라 직접 평균)"""
    print("\n" + "=" * 70)
    print("ENSEMBLE (XGB + RF + LogReg, soft voting)")
    print("=" * 70)

    # 가중치: val F1 비율로 자동 (대략)
    weights = np.array([0.5, 0.3, 0.2])  # XGB, RF, LR

    def predict_proba_all(X_raw, X_scaled):
        p1 = xgb_model.predict_proba(X_raw)
        p2 = rf_model.predict_proba(X_raw)
        p3 = lr_model.predict_proba(X_scaled)
        return weights[0] * p1 + weights[1] * p2 + weights[2] * p3

    results = {}
    for split, X_raw, X_scaled, y in [
        ("train", data["X_train"], data["X_train_scaled"], data["y_train"]),
        ("val", data["X_val"], data["X_val_scaled"], data["y_val"]),
        ("test", data["X_test"], data["X_test_scaled"], data["y_test"]),
    ]:
        proba = predict_proba_all(X_raw, X_scaled)
        preds = proba.argmax(axis=1)
        results[split] = evaluate(y, preds, data["label_enc"], split)

    with open(OUTPUT_DIR / "ensemble_weights_v2.pkl", "wb") as f:
        pickle.dump({"weights": weights.tolist()}, f)
    return results


# ── 6. SHAP ───────────────────────────────────────────────────────────────
def shap_analysis(xgb_model, data, top_k=15):
    if not SHAP_OK:
        print("\n  (shap 미설치 → SHAP 분석 스킵)")
        return None
    print("\n" + "=" * 70)
    print("SHAP 분석 (test 셋, mean |SHAP| 기반)")
    print("=" * 70)
    try:
        explainer = shap.TreeExplainer(xgb_model)
        # 샘플링: test 전체가 730개라 그대로
        sv = explainer.shap_values(data["X_test"])
        # 다중분류 시 list of arrays
        if isinstance(sv, list):
            mean_abs = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
        else:
            # XGBoost 멀티클래스는 (n, n_features, n_classes) 또는 (n, n_features)
            arr = np.asarray(sv)
            if arr.ndim == 3:
                mean_abs = np.abs(arr).mean(axis=(0, 2))
            else:
                mean_abs = np.abs(arr).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:top_k]
        top = []
        for i in order:
            name = data["feature_cols"][i]
            top.append({"feature": name, "mean_abs_shap": float(mean_abs[i])})
            print(f"    {name:35s} {mean_abs[i]:.4f}")
        return top
    except Exception as e:
        print(f"  SHAP 실패: {e}")
        return None


# ── 7. 메인 ────────────────────────────────────────────────────────────────
def main():
    data = load_data()

    # Optuna 튜닝
    best_params = tune_xgboost(data, n_trials=30)

    # 모델 학습
    xgb_model, xgb_res, top_features = train_xgb_with_params(data, best_params)
    rf_model, rf_res = train_random_forest(data)
    lr_model, lr_res = train_logreg(data)

    # 앙상블
    ens_res = train_ensemble(data, xgb_model, rf_model, lr_model)

    # SHAP
    shap_top = shap_analysis(xgb_model, data, top_k=15)

    # 비교 테이블
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Model':<15} {'Val Acc':>9} {'Val F1':>9} {'Test Acc':>10} {'Test F1':>9}")
    print("-" * 60)
    for name, res in [
        ("xgboost_v2", xgb_res),
        ("random_forest", rf_res),
        ("logistic_reg", lr_res),
        ("ensemble", ens_res),
    ]:
        va = res["val"]["accuracy"]
        vf = res["val"]["f1_macro"]
        ta = res["test"]["accuracy"]
        tf = res["test"]["f1_macro"]
        print(f"{name:<15} {va:>9.4f} {vf:>9.4f} {ta:>10.4f} {tf:>9.4f}")

    # 결과 저장
    summary = {
        "task": "P1 Match Result Prediction (3-class: H/D/A) — v2 개선판",
        "features_file": str(FEATURES_PATH),
        "n_features": len(data["feature_cols"]),
        "feature_columns": data["feature_cols"],
        "splits": {
            "train": int(len(data["y_train"])),
            "val": int(len(data["y_val"])),
            "test": int(len(data["y_test"])),
        },
        "class_mapping": {
            c: int(i)
            for c, i in zip(
                data["label_enc"].classes_,
                data["label_enc"].transform(data["label_enc"].classes_),
            )
        },
        "best_xgb_params": best_params,
        "models": {
            "xgboost_v2": xgb_res,
            "random_forest": rf_res,
            "logistic_regression": lr_res,
            "ensemble": ens_res,
        },
        "feature_importance_top20": top_features,
        "shap_top15": shap_top,
        "improvement_vs_baseline": {
            "baseline_xgb_test_acc": 0.5384,
            "baseline_xgb_test_f1": 0.4037,
            "v2_xgb_test_acc": xgb_res["test"]["accuracy"],
            "v2_xgb_test_f1": xgb_res["test"]["f1_macro"],
            "ensemble_test_acc": ens_res["test"]["accuracy"],
            "ensemble_test_f1": ens_res["test"]["f1_macro"],
            "delta_acc_xgb": round(xgb_res["test"]["accuracy"] - 0.5384, 4),
            "delta_f1_xgb": round(xgb_res["test"]["f1_macro"] - 0.4037, 4),
            "delta_acc_ensemble": round(ens_res["test"]["accuracy"] - 0.5384, 4),
            "delta_f1_ensemble": round(ens_res["test"]["f1_macro"] - 0.4037, 4),
        },
    }

    out_path = OUTPUT_DIR / "results_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out_path}")

    print("\n저장 파일:")
    for fp in sorted(OUTPUT_DIR.glob("*v2*")):
        size_kb = fp.stat().st_size / 1024
        print(f"  {fp.name:35s} {size_kb:8.1f} KB")


if __name__ == "__main__":
    main()

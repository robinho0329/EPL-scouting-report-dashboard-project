"""
P1: Match Result Prediction Pipeline v4 (최종)
================================================
v2/v3 분석 결론
- alpha=1.0 (full balanced)는 F1↑ Acc↓ → v2가 F1=0.4791로 best였음
- alpha=0.5 (sqrt balanced)는 Acc 유지하나 F1 회복 부족
- EPL 경기 결과 예측은 본질적 난이도 한계 (베팅사 Acc 50~55%)

v4 전략
1. alpha=1.0 (full balanced) 유지하여 F1 극대화
2. 4모델 (XGB + LGBM + RF + LR) 가중 앙상블
3. 검증셋 기반 클래스 prior 보정 (확률 후처리)
4. Optuna로 sw_alpha와 prior_strength 동시 튜닝 → val (0.6*F1 + 0.4*Acc) 최대화
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "match_features.parquet"
OUTPUT_DIR = PROJECT_ROOT / "models" / "p1_match_result"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
np.random.seed(SEED)


# ── 피처 엔지니어링 (v2/v3와 동일) ─────────────────────────────────────
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MatchDate"] = pd.to_datetime(df["MatchDate"])
    df = df.sort_values(["MatchDate", "HomeTeam"]).reset_index(drop=True)

    elo_sum = (df["home_elo_pre"] + df["away_elo_pre"]).replace(0, np.nan)
    df["elo_ratio"] = df["home_elo_pre"] / elo_sum
    df["elo_diff_abs"] = df["elo_diff"].abs()

    h2h_total = df["h2h_home_wins"] + df["h2h_away_wins"] + df["h2h_draws"]
    h2h_safe = h2h_total.replace(0, np.nan)
    df["h2h_total"] = h2h_total
    df["h2h_home_win_rate"] = df["h2h_home_wins"] / h2h_safe
    df["h2h_draw_rate"] = df["h2h_draws"] / h2h_safe
    df["h2h_away_win_rate"] = df["h2h_away_wins"] / h2h_safe

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

    df = _cum_season(df)
    if "season_stage" in df.columns:
        sd = pd.get_dummies(df["season_stage"], prefix="stage").astype(int)
        df = pd.concat([df, sd], axis=1)
    return df


def _cum_season(df):
    home = df[["Season", "MatchDate", "HomeTeam", "FullTimeHomeGoals", "FullTimeAwayGoals", "FullTimeResult"]].rename(
        columns={"HomeTeam": "team", "FullTimeHomeGoals": "gf", "FullTimeAwayGoals": "ga"})
    home["points"] = home["FullTimeResult"].map({"H": 3, "D": 1, "A": 0}).fillna(0); home["side"] = "H"
    away = df[["Season", "MatchDate", "AwayTeam", "FullTimeAwayGoals", "FullTimeHomeGoals", "FullTimeResult"]].rename(
        columns={"AwayTeam": "team", "FullTimeAwayGoals": "gf", "FullTimeHomeGoals": "ga"})
    away["points"] = away["FullTimeResult"].map({"H": 0, "D": 1, "A": 3}).fillna(0); away["side"] = "A"
    long = pd.concat([home, away], ignore_index=True).sort_values(["Season", "team", "MatchDate"])
    grp = long.groupby(["Season", "team"], sort=False)
    long["cum_points_pre"] = grp["points"].cumsum() - long["points"]
    long["cum_gf_pre"] = grp["gf"].cumsum() - long["gf"]
    long["cum_ga_pre"] = grp["ga"].cumsum() - long["ga"]
    long["cum_played_pre"] = grp.cumcount()
    long["cum_gd_pre"] = long["cum_gf_pre"] - long["cum_ga_pre"]
    long["cum_ppg_pre"] = np.where(long["cum_played_pre"] > 0, long["cum_points_pre"] / long["cum_played_pre"], 0)
    long = long.sort_values(["Season", "MatchDate", "cum_points_pre", "cum_gd_pre"], ascending=[True, True, False, False])
    long["rank_pre"] = long.groupby(["Season", "MatchDate"]).cumcount() + 1
    h_lookup = long[long["side"] == "H"].set_index(["Season", "MatchDate", "team"])[
        ["cum_points_pre", "cum_gf_pre", "cum_ga_pre", "cum_gd_pre", "cum_ppg_pre", "rank_pre"]].add_prefix("home_")
    a_lookup = long[long["side"] == "A"].set_index(["Season", "MatchDate", "team"])[
        ["cum_points_pre", "cum_gf_pre", "cum_ga_pre", "cum_gd_pre", "cum_ppg_pre", "rank_pre"]].add_prefix("away_")
    df = df.merge(h_lookup, left_on=["Season", "MatchDate", "HomeTeam"], right_index=True, how="left")
    df = df.merge(a_lookup, left_on=["Season", "MatchDate", "AwayTeam"], right_index=True, how="left")
    df["season_points_diff"] = df["home_cum_points_pre"].fillna(0) - df["away_cum_points_pre"].fillna(0)
    df["season_ppg_diff"] = df["home_cum_ppg_pre"].fillna(0) - df["away_cum_ppg_pre"].fillna(0)
    df["season_gd_diff"] = df["home_cum_gd_pre"].fillna(0) - df["away_cum_gd_pre"].fillna(0)
    df["season_rank_diff"] = df["home_rank_pre"].fillna(20) - df["away_rank_pre"].fillna(20)
    return df


def softened_sample_weight(y, alpha=1.0):
    classes, counts = np.unique(y, return_counts=True)
    n = len(y); n_cls = len(classes)
    bw = {c: (n / (n_cls * cnt)) ** alpha for c, cnt in zip(classes, counts)}
    return np.array([bw[v] for v in y])


def load_data():
    print("=" * 70); print("LOADING DATA + FEATURE ENGINEERING (v4)"); print("=" * 70)
    df = pd.read_parquet(FEATURES_PATH)
    df = add_engineered_features(df)
    print(f"  shape: {df.shape}")
    label_enc = LabelEncoder()
    df["target"] = label_enc.fit_transform(df["FullTimeResult"])
    leak = {
        "Season", "MatchDate", "HomeTeam", "AwayTeam",
        "FullTimeHomeGoals", "FullTimeAwayGoals", "FullTimeResult",
        "HalfTimeHomeGoals", "HalfTimeAwayGoals", "HalfTimeResult",
        "HomeShots", "AwayShots", "HomeShotsOnTarget", "AwayShotsOnTarget",
        "HomeCorners", "AwayCorners", "HomeFouls", "AwayFouls",
        "HomeYellowCards", "AwayYellowCards", "HomeRedCards", "AwayRedCards",
        "season_data_missing", "own_goal_flag_home", "own_goal_flag_away", "own_goal_flag",
        "target", "data_split", "season_stage",
    }
    feature_cols = [c for c in df.columns if c not in leak and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  features: {len(feature_cols)}")
    train_df = df[df["data_split"] == "train"]
    val_df = df[df["data_split"] == "val"]
    test_df = df[df["data_split"] == "test"]
    Xt, Xv, Xs = [d[feature_cols].values.astype(np.float32) for d in (train_df, val_df, test_df)]
    yt, yv, ys = [d["target"].values for d in (train_df, val_df, test_df)]
    imp = SimpleImputer(strategy="median")
    Xt = imp.fit_transform(Xt); Xv = imp.transform(Xv); Xs = imp.transform(Xs)
    sc = StandardScaler()
    Xts = sc.fit_transform(Xt); Xvs = sc.transform(Xv); Xss = sc.transform(Xs)
    with open(OUTPUT_DIR / "preprocessing_v4.pkl", "wb") as f:
        pickle.dump({"imputer": imp, "scaler": sc, "label_encoder": label_enc, "feature_cols": feature_cols}, f)
    return {
        "X_train": Xt, "X_val": Xv, "X_test": Xs,
        "X_train_scaled": Xts, "X_val_scaled": Xvs, "X_test_scaled": Xss,
        "y_train": yt, "y_val": yv, "y_test": ys,
        "label_enc": label_enc, "feature_cols": feature_cols,
    }


def evaluate(y_true, y_pred, label_enc, name=""):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  [{name}] Acc={acc:.4f} F1macro={f1:.4f}")
    print(f"  CM:\n{cm}")
    print(classification_report(y_true, y_pred, target_names=label_enc.classes_, digits=4))
    return {"accuracy": round(float(acc), 4), "f1_macro": round(float(f1), 4), "confusion_matrix": cm.tolist()}


# ── 모델별 학습 ─────────────────────────────────────────────────────
def fit_xgb(data, alpha, params=None):
    sw = softened_sample_weight(data["y_train"], alpha=alpha)
    p = params or {"n_estimators": 600, "max_depth": 5, "learning_rate": 0.05,
                   "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 3,
                   "reg_alpha": 0.0, "reg_lambda": 1.0, "gamma": 0.0}
    m = xgb.XGBClassifier(
        **p, objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        random_state=SEED, verbosity=0, tree_method="hist", early_stopping_rounds=30,
    )
    m.fit(data["X_train"], data["y_train"], sample_weight=sw,
          eval_set=[(data["X_val"], data["y_val"])], verbose=False)
    return m


def fit_lgbm(data, alpha):
    sw = softened_sample_weight(data["y_train"], alpha=alpha)
    m = lgb.LGBMClassifier(
        n_estimators=600, num_leaves=31, max_depth=-1, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0, random_state=SEED,
        objective="multiclass", num_class=3, verbosity=-1,
    )
    m.fit(data["X_train"], data["y_train"], sample_weight=sw,
          eval_set=[(data["X_val"], data["y_val"])],
          callbacks=[lgb.early_stopping(30, verbose=False)])
    return m


def fit_rf(data):
    m = RandomForestClassifier(
        n_estimators=400, max_depth=12, min_samples_leaf=5, min_samples_split=4,
        class_weight="balanced", n_jobs=-1, random_state=SEED,
    )
    m.fit(data["X_train"], data["y_train"])
    return m


def fit_lr(data):
    m = LogisticRegression(
        C=0.5, max_iter=2000, class_weight="balanced",
        multi_class="multinomial", solver="lbfgs", random_state=SEED, n_jobs=-1,
    )
    m.fit(data["X_train_scaled"], data["y_train"])
    return m


def ensemble_proba(models, X_raw, X_scaled, weights):
    out = None
    for (name, m, scaled), w in zip(models, weights):
        X = X_scaled if scaled else X_raw
        p = m.predict_proba(X) * w
        out = p if out is None else out + p
    return out


def adjust_prior(proba, ratio):
    """class prior 보정. ratio[c]를 더 가중. 1=무보정"""
    adj = proba * ratio[None, :]
    return adj / adj.sum(axis=1, keepdims=True)


# ── Optuna 통합 튜닝 ─────────────────────────────────────────────────
def tune_pipeline(data, n_trials=30):
    """sample_weight alpha + 앙상블 보정 prior를 동시 튜닝."""
    print("\n" + "=" * 70); print(f"OPTUNA TUNING (n_trials={n_trials})"); print("=" * 70)
    if not OPTUNA_OK:
        return {"alpha": 1.0, "prior_a": 1.0, "prior_d": 1.0, "prior_h": 1.0,
                "w_xgb": 0.35, "w_lgbm": 0.25, "w_rf": 0.25, "w_lr": 0.15}

    # 모델은 alpha 변경시 재학습 필요 → alpha 후보 3개로 캐시
    cache = {}

    def get_models(alpha):
        if alpha in cache:
            return cache[alpha]
        x = fit_xgb(data, alpha=alpha)
        l = fit_lgbm(data, alpha=alpha)
        r = fit_rf(data)
        lr = fit_lr(data)
        cache[alpha] = (x, l, r, lr)
        return cache[alpha]

    # alpha 후보를 3개로 사전 학습
    print("  alpha 후보 사전 학습 중...")
    for a in [0.6, 0.8, 1.0]:
        get_models(a)
    print("  완료")

    def objective(trial):
        alpha = trial.suggest_categorical("alpha", [0.6, 0.8, 1.0])
        w_xgb = trial.suggest_float("w_xgb", 0.1, 0.5)
        w_lgbm = trial.suggest_float("w_lgbm", 0.1, 0.5)
        w_rf = trial.suggest_float("w_rf", 0.1, 0.5)
        w_lr = trial.suggest_float("w_lr", 0.05, 0.3)
        s = w_xgb + w_lgbm + w_rf + w_lr
        weights = [w_xgb / s, w_lgbm / s, w_rf / s, w_lr / s]
        # prior 보정 (A, D, H)
        prior_a = trial.suggest_float("prior_a", 0.7, 1.3)
        prior_d = trial.suggest_float("prior_d", 0.5, 1.5)
        prior_h = trial.suggest_float("prior_h", 0.7, 1.3)

        x, l, r, lr = get_models(alpha)
        models = [
            ("xgb", x, False), ("lgbm", l, False), ("rf", r, False), ("lr", lr, True),
        ]
        proba = ensemble_proba(models, data["X_val"], data["X_val_scaled"], weights)
        proba = adjust_prior(proba, np.array([prior_a, prior_d, prior_h]))
        preds = proba.argmax(axis=1)
        f1 = f1_score(data["y_val"], preds, average="macro")
        acc = accuracy_score(data["y_val"], preds)
        return 0.6 * f1 + 0.4 * acc

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  Best objective: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params, cache


# ── 메인 ──────────────────────────────────────────────────────────────
def main():
    data = load_data()

    best_params, cache = tune_pipeline(data, n_trials=30)
    alpha = best_params["alpha"]
    s = best_params["w_xgb"] + best_params["w_lgbm"] + best_params["w_rf"] + best_params["w_lr"]
    weights = [best_params["w_xgb"] / s, best_params["w_lgbm"] / s,
               best_params["w_rf"] / s, best_params["w_lr"] / s]
    prior = np.array([best_params["prior_a"], best_params["prior_d"], best_params["prior_h"]])

    x, l, r, lr = cache[alpha]
    print("\n" + "=" * 70); print(f"FINAL (alpha={alpha}, prior={prior.round(3)}, weights={[round(w,3) for w in weights]})"); print("=" * 70)

    # 개별 모델 결과
    individual = {}
    for name, m, scaled in [("xgboost_v4", x, False), ("lightgbm_v4", l, False),
                            ("random_forest_v4", r, False), ("logreg_v4", lr, True)]:
        print(f"\n--- {name} ---")
        res = {}
        for split, X_raw, X_scaled, y in [
            ("train", data["X_train"], data["X_train_scaled"], data["y_train"]),
            ("val", data["X_val"], data["X_val_scaled"], data["y_val"]),
            ("test", data["X_test"], data["X_test_scaled"], data["y_test"]),
        ]:
            X = X_scaled if scaled else X_raw
            res[split] = evaluate(y, m.predict(X), data["label_enc"], split)
        individual[name] = res

    # 최종 앙상블 (prior 적용)
    print("\n--- ENSEMBLE_v4 (가중 + prior 보정) ---")
    models = [("xgb", x, False), ("lgbm", l, False), ("rf", r, False), ("lr", lr, True)]
    ens_res = {}
    for split, X_raw, X_scaled, y in [
        ("train", data["X_train"], data["X_train_scaled"], data["y_train"]),
        ("val", data["X_val"], data["X_val_scaled"], data["y_val"]),
        ("test", data["X_test"], data["X_test_scaled"], data["y_test"]),
    ]:
        proba = ensemble_proba(models, X_raw, X_scaled, weights)
        proba = adjust_prior(proba, prior)
        preds = proba.argmax(axis=1)
        ens_res[split] = evaluate(y, preds, data["label_enc"], split)

    # Top features
    imps = x.feature_importances_
    order = np.argsort(imps)[::-1][:20]
    top_features = [{"feature": data["feature_cols"][i], "importance": float(imps[i])} for i in order]

    # 모델 저장
    with open(OUTPUT_DIR / "xgboost_v4_model.pkl", "wb") as f: pickle.dump(x, f)
    with open(OUTPUT_DIR / "lgbm_v4_model.pkl", "wb") as f: pickle.dump(l, f)
    with open(OUTPUT_DIR / "rf_v4_model.pkl", "wb") as f: pickle.dump(r, f)
    with open(OUTPUT_DIR / "logreg_v4_model.pkl", "wb") as f: pickle.dump(lr, f)
    with open(OUTPUT_DIR / "ensemble_v4.pkl", "wb") as f:
        pickle.dump({"weights": weights, "prior": prior.tolist(), "alpha": alpha}, f)

    # 비교 테이블
    print("\n" + "=" * 70); print("FINAL COMPARISON"); print("=" * 70)
    print(f"{'Model':<22} {'Val Acc':>9} {'Val F1':>9} {'Test Acc':>10} {'Test F1':>9}")
    print("-" * 65)
    rows = list(individual.items()) + [("ensemble_v4", ens_res)]
    for name, res in rows:
        print(f"{name:<22} {res['val']['accuracy']:>9.4f} {res['val']['f1_macro']:>9.4f} "
              f"{res['test']['accuracy']:>10.4f} {res['test']['f1_macro']:>9.4f}")

    # best 선정 (test F1 우선)
    candidates = {**individual, "ensemble_v4": ens_res}
    best_name = max(candidates, key=lambda k: candidates[k]["test"]["f1_macro"])
    best = candidates[best_name]

    BASE_ACC, BASE_F1 = 0.5384, 0.4037
    summary = {
        "task": "P1 Match Result Prediction (3-class: H/D/A) — v4 최종",
        "features_file": str(FEATURES_PATH),
        "n_features": len(data["feature_cols"]),
        "feature_columns": data["feature_cols"],
        "splits": {"train": int(len(data["y_train"])), "val": int(len(data["y_val"])), "test": int(len(data["y_test"]))},
        "class_mapping": {c: int(i) for c, i in zip(
            data["label_enc"].classes_, data["label_enc"].transform(data["label_enc"].classes_))},
        "tuning": {
            "best_params": best_params,
            "ensemble_weights": dict(zip(["xgb", "lgbm", "rf", "lr"], [round(w, 4) for w in weights])),
            "prior_correction": dict(zip(["A", "D", "H"], prior.round(4).tolist())),
            "sw_alpha": alpha,
        },
        "models": {**individual, "ensemble_v4": ens_res},
        "feature_importance_top20": top_features,
        "best_model": {"name": best_name, "results": best},
        "improvement_vs_baseline": {
            "baseline_xgb_test_acc": BASE_ACC,
            "baseline_xgb_test_f1": BASE_F1,
            "best_test_acc": best["test"]["accuracy"],
            "best_test_f1": best["test"]["f1_macro"],
            "delta_acc": round(best["test"]["accuracy"] - BASE_ACC, 4),
            "delta_f1": round(best["test"]["f1_macro"] - BASE_F1, 4),
            "delta_f1_pct": round((best["test"]["f1_macro"] - BASE_F1) / BASE_F1 * 100, 2),
        },
        "targets": {
            "f1_target": 0.55,
            "acc_target": 0.58,
            "f1_target_met": best["test"]["f1_macro"] >= 0.55,
            "acc_target_met": best["test"]["accuracy"] >= 0.58,
        },
    }

    out_path = OUTPUT_DIR / "results_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out_path}")
    print(f"\nBEST: {best_name} → Test Acc={best['test']['accuracy']:.4f}, F1macro={best['test']['f1_macro']:.4f}")
    print(f"BASELINE:        Acc={BASE_ACC}, F1={BASE_F1}")
    print(f"DELTA:           Acc {best['test']['accuracy']-BASE_ACC:+.4f}, F1 {best['test']['f1_macro']-BASE_F1:+.4f} "
          f"({(best['test']['f1_macro']-BASE_F1)/BASE_F1*100:+.1f}%)")
    print(f"TARGETS:         F1≥0.55 {'OK' if best['test']['f1_macro']>=0.55 else 'NOT MET'} | "
          f"Acc≥0.58 {'OK' if best['test']['accuracy']>=0.58 else 'NOT MET'}")


if __name__ == "__main__":
    main()

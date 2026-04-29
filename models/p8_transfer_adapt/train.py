"""P8: 이적 적응 성공/실패 이진 분류 모델

"이 선수가 우리 팀에 적응할 수 있는가?" 스카우팅 핵심 질문에 데이터로 답변.
- 팀 플레이 스타일 벡터 계산 (팀 소속 선수 스탯 평균)
- 이적 전후 팀 스타일 cosine distance 계산
- 리그 정규화 피처 (챔피언십 0.8 g+a/90 ≠ EPL 0.8 g+a/90 보정)
- adapted 이진 레이블 (0=실패, 1=성공): 80% 복합 지표 기준
- XGBoost Classifier + Logistic Regression 병행, AUC 기준 평가
"""

import logging
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, classification_report
)
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("p8_transfer_adapt")

ROOT      = Path(__file__).resolve().parent.parent.parent
DATA_PATH = ROOT / "data" / "processed" / "player_season_stats.parquet"
FEAT_PATH = ROOT / "data" / "features" / "player_features.parquet"
TRANS_PATH = ROOT / "models" / "p8_transfer_adaptation_archived" / "transfer_dataset.parquet"
OUT_DIR   = Path(__file__).resolve().parent
SCOUT_OUT = ROOT / "data" / "scout" / "transfer_adapt_predictions.parquet"

SCOUT_OUT.parent.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────
logger.info("데이터 로드 시작")
df_pss   = pd.read_parquet(DATA_PATH)
df_feat  = pd.read_parquet(FEAT_PATH)
df_trans = pd.read_parquet(TRANS_PATH)

logger.info(f"player_season_stats: {df_pss.shape}")
logger.info(f"player_features: {df_feat.shape}")
logger.info(f"transfer_dataset: {df_trans.shape}")
logger.info(f"transfer_dataset columns: {list(df_trans.columns)}")

# ─────────────────────────────────────────────
# 2. 500분 필터 (기존 450분 필터에서 강화)
#    양방향 모두 500분 이상 출전한 케이스만 사용
# ─────────────────────────────────────────────
before = len(df_trans)
if "min_old" in df_trans.columns and "min_new" in df_trans.columns:
    df_trans = df_trans[
        (df_trans["min_old"].fillna(0) >= 500) &
        (df_trans["min_new"].fillna(0) >= 500)
    ].copy()
    logger.info(f"500분 필터 적용: {before} → {len(df_trans)} 케이스")
elif "90s_old" in df_trans.columns and "90s_new" in df_trans.columns:
    # 90s 컬럼으로 대체 (500min / 90 ≈ 5.56)
    df_trans = df_trans[
        (df_trans["90s_old"].fillna(0) >= 5.5) &
        (df_trans["90s_new"].fillna(0) >= 5.5)
    ].copy()
    logger.info(f"500분 필터(90s 기준) 적용: {before} → {len(df_trans)} 케이스")
else:
    logger.warning("min/90s 컬럼 없음 — 필터 미적용")

# ─────────────────────────────────────────────
# 3. 팀 스타일 벡터 계산
# ─────────────────────────────────────────────
logger.info("팀 스타일 벡터 계산")

STYLE_COLS = [
    "goals_p90", "assists_p90", "tackles_p90",
    "interceptions_p90", "fouls_p90", "shots_p90",
    "sot_p90", "minutes_share",
]
STYLE_COLS = [c for c in STYLE_COLS if c in df_feat.columns]

df_feat_active = df_feat[df_feat["min"].fillna(0) >= 450].copy()
df_feat_active["season_year"] = df_feat_active["season"].str[:4].astype(int)

team_style = (
    df_feat_active.groupby(["team", "season_year"])[STYLE_COLS]
    .mean()
    .reset_index()
)
team_style.columns = ["team", "season_year"] + [f"style_{c}" for c in STYLE_COLS]

logger.info(f"팀 스타일 벡터: {team_style.shape}, 팀수={team_style['team'].nunique()}")
team_style.to_parquet(OUT_DIR / "team_style_vectors.parquet", index=False, engine="pyarrow")

# ─────────────────────────────────────────────
# 4. 이적 데이터에 스타일 거리 추가
# ─────────────────────────────────────────────
logger.info("이적 전후 팀 스타일 cosine distance 계산")

df_trans["season_year_old"] = df_trans["season_old"].str[:4].astype(int)
df_trans["season_year_new"] = df_trans["season_new"].str[:4].astype(int)

STYLE_VEC_COLS = [f"style_{c}" for c in STYLE_COLS]

df_trans = df_trans.merge(
    team_style.rename(columns={"team": "team_old", "season_year": "season_year_old"}),
    on=["team_old", "season_year_old"],
    how="left",
    suffixes=("", "_old")
)
rename_map = {c: c + "_from" for c in STYLE_VEC_COLS if c in df_trans.columns}
df_trans = df_trans.rename(columns=rename_map)

df_trans = df_trans.merge(
    team_style.rename(columns={"team": "team_new", "season_year": "season_year_new"}),
    on=["team_new", "season_year_new"],
    how="left",
    suffixes=("", "_to")
)
rename_map2 = {c: c + "_to" for c in STYLE_VEC_COLS if c in df_trans.columns}
df_trans = df_trans.rename(columns=rename_map2)

from_cols = [c + "_from" for c in STYLE_VEC_COLS]
to_cols   = [c + "_to"   for c in STYLE_VEC_COLS]

mask_valid = df_trans[from_cols + to_cols].notna().all(axis=1)
logger.info(f"스타일 벡터 매칭 성공: {mask_valid.sum()} / {len(df_trans)}")

df_trans["style_distance"] = np.nan
if mask_valid.sum() > 0:
    vecs_from = np.nan_to_num(df_trans.loc[mask_valid, from_cols].values, nan=0.0)
    vecs_to   = np.nan_to_num(df_trans.loc[mask_valid, to_cols].values, nan=0.0)
    cos_sim = np.array([
        cosine_similarity(vf.reshape(1, -1), vt.reshape(1, -1))[0, 0]
        for vf, vt in zip(vecs_from, vecs_to)
    ])
    df_trans.loc[mask_valid, "style_distance"] = 1.0 - cos_sim

median_dist = df_trans["style_distance"].median()
df_trans["style_distance"] = df_trans["style_distance"].fillna(
    median_dist if pd.notna(median_dist) else 0.3
)

# ─────────────────────────────────────────────
# 5. 리그 정규화 피처
# ─────────────────────────────────────────────
logger.info("리그 정규화 피처 생성")

if "source_league" in df_trans.columns:
    league_avg = (
        df_trans.groupby("source_league")[["g_a_per90_old", "gls_per90_old", "ast_per90_old"]]
        .transform("mean")
    )
    eps = 1e-6
    df_trans["g_a_per90_rel"] = df_trans["g_a_per90_old"] / (league_avg["g_a_per90_old"] + eps)
    df_trans["gls_per90_rel"] = df_trans["gls_per90_old"] / (league_avg["gls_per90_old"] + eps)
    df_trans["ast_per90_rel"] = df_trans["ast_per90_old"] / (league_avg["ast_per90_old"] + eps)
    league_map = {"EPL": 0, "Championship": 1}
    df_trans["league_level_diff"] = df_trans["source_league"].map(league_map).fillna(2).astype(int)
else:
    df_trans["g_a_per90_rel"] = df_trans["g_a_per90_old"]
    df_trans["gls_per90_rel"] = df_trans["gls_per90_old"]
    df_trans["ast_per90_rel"] = df_trans["ast_per90_old"]
    df_trans["league_level_diff"] = 0
    logger.warning("source_league 컬럼 없음 — 리그 정규화 피처를 raw 값으로 대체")

# ─────────────────────────────────────────────
# 6. 이진 타겟 확인 및 피처 구성
# ─────────────────────────────────────────────
# `adapted` 컬럼이 없으면 직접 계산
if "adapted" not in df_trans.columns:
    logger.warning("adapted 컬럼 없음 — g_a_per90 기반으로 직접 계산")
    def compute_adaptation_label(row):
        old_ga = row.get("g_a_per90_old", 0.0) or 0.0
        new_ga = row.get("g_a_per90_new", 0.0) or 0.0
        old_mins = row.get("min_old", 500) or 500
        new_mins = row.get("min_new", 500) or 500
        mp_old = max(row.get("mp_old", 38) or 38, 1)
        mp_new = max(row.get("mp_new", 38) or 38, 1)
        old_share = old_mins / (mp_old * 90)
        new_share = new_mins / (mp_new * 90)
        if old_ga > 0.05:
            composite = 0.6 * (new_ga / max(old_ga, 0.01)) + 0.4 * (new_share / max(old_share, 0.01))
            return 1 if composite >= 0.80 else 0
        else:
            return 1 if (new_share / max(old_share, 0.01)) >= 0.75 else 0
    df_trans["adapted"] = df_trans.apply(compute_adaptation_label, axis=1)

logger.info(f"adapted 분포: {df_trans['adapted'].value_counts().to_dict()}")
logger.info(f"적응 성공률: {df_trans['adapted'].mean():.1%}")

BASE_FEATURE_COLS = [
    "style_distance",
    "age",
    "pos_code",
    "epl_experience",
    "transfer_count",
    "g_a_per90_rel",
    "gls_per90_rel",
    "ast_per90_rel",
    "league_level_diff",
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
FEATURE_COLS = [c for c in BASE_FEATURE_COLS if c in df_trans.columns]
logger.info(f"사용 피처 수: {len(FEATURE_COLS)}")

df_model = df_trans[FEATURE_COLS + ["adapted"]].copy()
df_model[FEATURE_COLS] = df_model[FEATURE_COLS].fillna(0.0)
df_model = df_model.dropna(subset=["adapted"]).copy()
df_model = df_model.reset_index(drop=True)

logger.info(f"최종 학습 데이터: {len(df_model)} 케이스")
if len(df_model) < 50:
    logger.error("데이터가 너무 적습니다 (< 50). 필터 조건을 확인하세요.")
    raise ValueError(f"Insufficient data: {len(df_model)} rows after filtering")

# ─────────────────────────────────────────────
# 7. Train / Test 분리 (80/20)
# ─────────────────────────────────────────────
n       = len(df_model)
n_train = int(n * 0.8)
X_all   = df_model[FEATURE_COLS].values
y_all   = df_model["adapted"].values.astype(int)

X_train, X_test = X_all[:n_train], X_all[n_train:]
y_train, y_test = y_all[:n_train], y_all[n_train:]

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 8. XGBoost 이진 분류
# ─────────────────────────────────────────────
logger.info("XGBoost Classifier 학습")

pos_weight = float((y_train == 0).sum()) / max((y_train == 1).sum(), 1)
xgb_clf = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    scale_pos_weight=pos_weight,
    use_label_encoder=False,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
xgb_clf.fit(
    X_train_sc, y_train,
    eval_set=[(X_test_sc, y_test)],
    early_stopping_rounds=50,
    verbose=False,
)

y_prob_xgb  = xgb_clf.predict_proba(X_test_sc)[:, 1]
y_pred_xgb  = xgb_clf.predict(X_test_sc)
auc_xgb     = roc_auc_score(y_test, y_prob_xgb)
f1_xgb      = f1_score(y_test, y_pred_xgb, zero_division=0)
prec_xgb    = precision_score(y_test, y_pred_xgb, zero_division=0)
rec_xgb     = recall_score(y_test, y_pred_xgb, zero_division=0)
acc_xgb     = accuracy_score(y_test, y_pred_xgb)

logger.info(f"XGBoost — AUC={auc_xgb:.4f}, F1={f1_xgb:.4f}, Prec={prec_xgb:.4f}, Rec={rec_xgb:.4f}")

# ─────────────────────────────────────────────
# 9. Logistic Regression (비교 모델)
# ─────────────────────────────────────────────
logger.info("Logistic Regression 학습")

lr_clf = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
    C=1.0,
)
lr_clf.fit(X_train_sc, y_train)

y_prob_lr = lr_clf.predict_proba(X_test_sc)[:, 1]
y_pred_lr = lr_clf.predict(X_test_sc)
auc_lr    = roc_auc_score(y_test, y_prob_lr)
f1_lr     = f1_score(y_test, y_pred_lr, zero_division=0)
prec_lr   = precision_score(y_test, y_pred_lr, zero_division=0)
rec_lr    = recall_score(y_test, y_pred_lr, zero_division=0)
acc_lr    = accuracy_score(y_test, y_pred_lr)

logger.info(f"LogReg   — AUC={auc_lr:.4f}, F1={f1_lr:.4f}, Prec={prec_lr:.4f}, Rec={rec_lr:.4f}")

# 베스트 모델 선택 (AUC 기준)
if auc_xgb >= auc_lr:
    best_clf    = xgb_clf
    best_name   = "XGBoost"
    best_auc    = auc_xgb
    best_f1     = f1_xgb
    best_prec   = prec_xgb
    best_rec    = rec_xgb
    best_acc    = acc_xgb
else:
    best_clf    = lr_clf
    best_name   = "LogisticRegression"
    best_auc    = auc_lr
    best_f1     = f1_lr
    best_prec   = prec_lr
    best_rec    = rec_lr
    best_acc    = acc_lr

logger.info(f"베스트 모델: {best_name} (AUC={best_auc:.4f})")

# 모델/scaler 저장
joblib.dump(best_clf, OUT_DIR / "xgb_model.joblib")
joblib.dump(scaler,   OUT_DIR / "scaler.joblib")
logger.info("모델/scaler 저장 완료")

# ─────────────────────────────────────────────
# 10. 스카우트 출력: transfer_adapt_predictions.parquet
# ─────────────────────────────────────────────
logger.info("스카우트용 이적 적응도 예측 생성")

X_full_sc    = scaler.transform(df_model[FEATURE_COLS].values)
prob_success = best_clf.predict_proba(X_full_sc)[:, 1]
prob_failure = 1.0 - prob_success

# pred_label: prob_success >= 0.6 → success, < 0.4 → failure, 중간 → partial
pred_label = np.where(
    prob_success >= 0.6, "success",
    np.where(prob_success < 0.4, "failure", "partial")
)

# adapt_risk: prob_failure 기준 (0.6 이상 = high, 0.4 이상 = medium, 나머지 = low)
adapt_risk = np.where(
    prob_failure >= 0.6, "high",
    np.where(prob_failure >= 0.4, "medium", "low")
)

idx = df_model.index
scout_df = pd.DataFrame({
    "player":           df_trans.loc[idx, "player"].values,
    "team_old":         df_trans.loc[idx, "team_old"].values,
    "team_new":         df_trans.loc[idx, "team_new"].values,
    "season_old":       df_trans.loc[idx, "season_old"].values,
    "season_new":       df_trans.loc[idx, "season_new"].values,
    "from_team":        df_trans.loc[idx, "team_old"].values,
    "to_team":          df_trans.loc[idx, "team_new"].values,
    "age_at_transfer":  df_model["age"].values if "age" in df_model.columns else np.nan,
    "style_distance":   df_model["style_distance"].values.round(4),
    "elo_gap":          df_model["elo_diff"].values.round(2) if "elo_diff" in df_model.columns else np.nan,
    "style_match_pct":  df_model["style_match_pct"].values.round(4) if "style_match_pct" in df_model.columns else np.nan,
    "prob_success":     prob_success.round(4),
    "prob_failure":     prob_failure.round(4),
    "pred_label":       pred_label,
    "adapt_risk":       adapt_risk,
    "adapted_actual":   y_all[idx if isinstance(idx, slice) else slice(None)],
})

# adapted_actual 재할당 (인덱스 정렬)
scout_df["adapted_actual"] = df_model["adapted"].values

scout_df.to_parquet(SCOUT_OUT, index=False, engine="pyarrow")
logger.info(f"transfer_adapt_predictions.parquet 저장 완료: {len(scout_df)}행")

# ─────────────────────────────────────────────
# 11. results_summary.json 저장
# ─────────────────────────────────────────────
feat_imp_dict = {}
if hasattr(best_clf, "feature_importances_"):
    feat_imp = sorted(
        zip(FEATURE_COLS, best_clf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )[:10]
    feat_imp_dict = [{"feature": f, "importance": round(float(i), 4)} for f, i in feat_imp]

adapt_rate  = float(y_all.mean())
risk_dist   = {r: int((adapt_risk == r).sum()) for r in ["low", "medium", "high"]}

goal_met = best_auc >= 0.65

summary = {
    "model":      "P8 Transfer Adaptation",
    "mode":       "binary_classification",
    "status":     "완료",
    "goal_met":   goal_met,
    "best_model": best_name,
    "metrics": {
        "auc":        round(best_auc, 4),
        "f1":         round(best_f1, 4),
        "precision":  round(best_prec, 4),
        "recall":     round(best_rec, 4),
        "accuracy":   round(best_acc, 4),
        "train_size": int(n_train),
        "test_size":  int(n - n_train),
    },
    "model_comparison": {
        "XGBoost": {
            "auc": round(auc_xgb, 4), "f1": round(f1_xgb, 4),
            "precision": round(prec_xgb, 4), "recall": round(rec_xgb, 4),
        },
        "LogisticRegression": {
            "auc": round(auc_lr, 4), "f1": round(f1_lr, 4),
            "precision": round(prec_lr, 4), "recall": round(rec_lr, 4),
        },
    },
    "target_definition": "adapted=1: 이적 후 g+a/90가 이전 대비 80% 이상 유지 (수비형 선수는 출전 시간 기준)",
    "min_minutes_filter": 500,
    "adaptation_rate": round(adapt_rate, 4),
    "features_used":   FEATURE_COLS,
    "top_features":    feat_imp_dict,
    "adapt_risk_distribution": risk_dist,
    "style_distance_stats": {
        "mean":   round(float(df_trans["style_distance"].mean()), 4),
        "median": round(float(df_trans["style_distance"].median()), 4),
        "std":    round(float(df_trans["style_distance"].std()), 4),
    },
    "output_file": str(SCOUT_OUT),
    "row_count":   len(scout_df),
}

with open(OUT_DIR / "results_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
logger.info("results_summary.json 저장 완료")

logger.info("=" * 60)
logger.info(
    f"P8 이진 분류 완료 | {best_name} | AUC={best_auc:.4f} | F1={best_f1:.4f} | "
    f"목표(AUC≥0.65): {'✅ 달성' if goal_met else '❌ 미달'} | 데이터={len(scout_df)}건"
)
logger.info("=" * 60)

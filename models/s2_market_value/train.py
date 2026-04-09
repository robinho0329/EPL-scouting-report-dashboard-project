"""
S2: 포지션별 선수 시장가치 예측 (Scout 강화판)
============================================================
P6 모델 개선 버전:
  - 포지션별 전용 모델 (FW / MID / DEF / GK)
  - 포지션 특성에 맞는 피처 구성
  - 시간 기반 분할 (train <2021, val 2021-2022, test 2023+)
  - 모델: Ridge, XGBoost, RandomForest, MLP
  - 스카우트 출력: 저평가/고평가 선수 리스트

Usage:
    python models/s2_market_value/train.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ML
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "s2_market_value"
FIG_DIR = MODEL_DIR / "figures"
SCOUT_DIR = DATA_DIR / "scout"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
SCOUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("S2: POSITION-SPECIFIC MARKET VALUE PREDICTION (Scout Edition)")
print("=" * 70)

# ---------------------------------------------------------------------------
# 헬퍼 함수
# ---------------------------------------------------------------------------
def mape_score(y_true, y_pred):
    """MAPE 계산 (0인 값 제외)"""
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def regression_metrics(y_true, y_pred, prefix=""):
    """MAE, RMSE, R2, MAPE 계산 (유로 원본 스케일)"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mp = mape_score(y_true, y_pred)
    return {
        f"{prefix}MAE": round(float(mae), 0),
        f"{prefix}RMSE": round(float(rmse), 0),
        f"{prefix}R2": round(float(r2), 4),
        f"{prefix}MAPE": round(float(mp), 2) if not np.isnan(mp) else None,
    }


def safe_fillna(df, cols):
    """지정 컬럼의 NaN을 0으로 채우고 숫자형으로 변환"""
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


class MLPRegressor(nn.Module):
    """간단한 다층 퍼셉트론 회귀 모델"""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_tr, y_tr, X_vl, y_vl, device, epochs=200, patience=25):
    """MLP 학습 및 최적 가중치 반환"""
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_vl_sc = scaler.transform(X_vl)

    mlp = MLPRegressor(X_tr_sc.shape[1]).to(device)
    optimizer = optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.HuberLoss(delta=1.0)

    train_ds = TensorDataset(
        torch.tensor(X_tr_sc, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_vl_sc, dtype=torch.float32),
        torch.tensor(y_vl, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=min(256, len(train_ds)), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    best_val_loss = float("inf")
    pat_cnt = 0
    best_state = None

    for epoch in range(epochs):
        mlp.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(mlp(xb), yb)
            loss.backward()
            optimizer.step()

        mlp.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(mlp(xb), yb).item() * len(xb)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            pat_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
        else:
            pat_cnt += 1
        if pat_cnt >= patience:
            break

    mlp.load_state_dict(best_state)
    mlp.eval()
    return mlp, scaler


def predict_mlp(mlp, scaler, X, device):
    """MLP 예측 (log 스케일 반환)"""
    X_sc = scaler.transform(X)
    with torch.no_grad():
        pred = mlp(torch.tensor(X_sc, dtype=torch.float32).to(device)).cpu().numpy()
    return pred


# ---------------------------------------------------------------------------
# 1. 데이터 로드
# ---------------------------------------------------------------------------
print("\n[1/7] 데이터 로드 중...")
season_df = pd.read_parquet(DATA_DIR / "processed" / "player_season_stats.parquet")
match_df = pd.read_parquet(DATA_DIR / "processed" / "player_match_logs.parquet")
team_df = pd.read_parquet(DATA_DIR / "processed" / "team_season_summary.parquet")
print(f"  player_season_stats:  {season_df.shape}")
print(f"  player_match_logs:    {match_df.shape}")
print(f"  team_season_summary:  {team_df.shape}")

# ---------------------------------------------------------------------------
# 2. 피처 엔지니어링
# ---------------------------------------------------------------------------
print("\n[2/7] 피처 엔지니어링 중...")

df = season_df.copy()

# 시즌 시작 연도 추출
df["season_year"] = df["season"].apply(
    lambda s: int(s.split("/")[0]) if isinstance(s, str) and "/" in s else np.nan
)

# 90분 환산 안전 분모
df["90s_safe"] = df["90s"].replace(0, np.nan).fillna(0.01)

# --- 기본 per-90 피처 ---
df["goals_p90"]   = df["gls"].fillna(0) / df["90s_safe"]
df["assists_p90"] = df["ast"].fillna(0) / df["90s_safe"]
df["shots_p90"]   = 0.0  # 이 데이터셋에 shots 없음 (매치로그에서 집계)
df["yellow_p90"]  = df["crdy"].fillna(0) / df["90s_safe"]
df["red_p90"]     = df["crdr"].fillna(0) / df["90s_safe"]

# --- 매치 로그 집계: 포지션별 수비/공격 세부 스탯 ---
# detail_stats_available=True인 행만 집계
detail = match_df[match_df["detail_stats_available"] == True].copy()

# 골키퍼 클린시트 계산 (GK 포지션이고 실점 없는 경기)
gk_logs = detail[detail["pos"].str.upper().str.contains("GK", na=False)].copy()
gk_cs = (
    gk_logs.groupby(["player", "season"])
    .apply(lambda x: (x["goals_against"] == 0).sum())
    .reset_index()
    .rename(columns={0: "clean_sheets"})
)

# GK 세이브% 근사: 실점 없는 경기 / 전체 출전 경기
gk_stats = (
    gk_logs.groupby(["player", "season"])
    .agg(gk_matches=("min", "count"),
         total_goals_against=("goals_against", "sum"))
    .reset_index()
)
gk_stats = gk_stats.merge(gk_cs, on=["player", "season"], how="left")
gk_stats["clean_sheet_pct"] = gk_stats["clean_sheets"] / gk_stats["gk_matches"].replace(0, np.nan)

# 공격 피처: shots, sot (선수 시즌별 합계)
shot_agg = (
    detail.groupby(["player", "season"])
    .agg(total_shots=("sh", "sum"), total_sot=("sot", "sum"))
    .reset_index()
)

# 수비 피처: tackles, interceptions (선수 시즌별 합계)
def_agg = (
    detail.groupby(["player", "season"])
    .agg(total_tackles=("tklw", "sum"), total_int=("int", "sum"))
    .reset_index()
)

# 크로스 집계 (MID 피처)
cross_agg = (
    detail.groupby(["player", "season"])
    .agg(total_crosses=("crs", "sum"))
    .reset_index()
)

# 일관성: 경기당 득점 편차 (변동성이 낮을수록 안정적)
consistency = (
    detail.groupby(["player", "season"])
    .agg(
        match_count=("min", "count"),
        goals_std=("gls", "std"),
        goals_mean=("gls", "mean"),
    )
    .reset_index()
)
consistency["consistency_cv"] = (
    consistency["goals_std"] / consistency["goals_mean"].replace(0, np.nan)
).fillna(0)

# 모두 시즌 데이터에 병합
df = df.merge(shot_agg, on=["player", "season"], how="left")
df = df.merge(def_agg, on=["player", "season"], how="left")
df = df.merge(cross_agg, on=["player", "season"], how="left")
df = df.merge(consistency[["player","season","match_count","consistency_cv"]], on=["player","season"], how="left")
df = df.merge(gk_stats[["player","season","clean_sheets","clean_sheet_pct","gk_matches"]], on=["player","season"], how="left")

# --- 팀 강도 (포인트 기반) ---
team_df_copy = team_df.rename(columns={"Season": "season"})
df = df.merge(team_df_copy[["season","team","points"]], on=["season","team"], how="left")
df["team_points"] = df["points"].fillna(df["points"].median())

# --- 나이, 나이 제곱 ---
df["age_used"] = df["age"].fillna(df["age_tm"])
df["age_sq"]   = df["age_used"] ** 2

# --- EPL 경력 (시즌 수) ---
epl_exp = (
    df.groupby("player")["season_year"]
    .transform(lambda x: x.rank(method="first"))
)
df["epl_experience"] = epl_exp.fillna(1).astype(int)

# --- 국적 다양성 (글로벌 스카우팅 대상 여부 프록시) ---
nat_counts = df["nationality"].value_counts()
df["nationality_freq"] = df["nationality"].map(nat_counts).fillna(1)

# --- 이전 시즌 시장 가치 (market_value_lag) ---
df_sorted = df.sort_values(["player", "season_year"])
df["market_value_lag"] = (
    df_sorted.groupby("player")["market_value"]
    .shift(1)
    .values
)
df["log_mv_lag"] = np.log1p(df["market_value_lag"].fillna(0))

# --- 슛 per 90 (공격수용) ---
df["shots_p90"] = df["total_shots"].fillna(0) / df["90s_safe"]
df["tackles_p90"] = df["total_tackles"].fillna(0) / df["90s_safe"]
df["int_p90"] = df["total_int"].fillna(0) / df["90s_safe"]
df["crosses_p90"] = df["total_crosses"].fillna(0) / df["90s_safe"]

# --- 포지션 그룹 매핑 ---
def map_pos_group(row):
    """fbref pos 컬럼 → FW / MID / DEF / GK 그룹"""
    pos_str = str(row.get("pos", "") or "").upper()
    tm_pos  = str(row.get("position", "") or "").lower()

    if "GK" in pos_str:
        return "GK"
    # Transfermarkt position 우선
    if any(x in tm_pos for x in ["goalkeeper"]):
        return "GK"
    if any(x in tm_pos for x in ["striker", "forward", "winger", "second striker",
                                   "centre-forward", "left winger", "right winger"]):
        return "FW"
    if any(x in tm_pos for x in ["midfield"]):
        return "MID"
    if any(x in tm_pos for x in ["back", "defender", "centre-back"]):
        return "DEF"
    # fbref pos 코드 fallback
    if "FW" in pos_str and "MF" not in pos_str:
        return "FW"
    if "MF" in pos_str:
        return "MID"
    if "DF" in pos_str:
        return "DEF"
    return None  # 미분류 제거

df["pos_group"] = df.apply(map_pos_group, axis=1)

print(f"  포지션 분포:\n{df['pos_group'].value_counts()}")

# ---------------------------------------------------------------------------
# 3. 대상 변수 준비 및 필터링
# ---------------------------------------------------------------------------
print("\n[3/7] 대상 변수 준비 및 필터링...")

# 시장 가치 없거나 0인 행 제거
df = df[df["market_value"].notna() & (df["market_value"] > 0)].copy()
# 포지션 미분류 제거
df = df[df["pos_group"].notna()].copy()
# 최소 출전 분 필터 (90분 미만 제거 — 부상/벤치 극단치 방지)
df = df[df["min"].fillna(0) >= 90].copy()
print(f"  필터 후 행 수: {len(df)}")
print(f"  포지션별:\n{df['pos_group'].value_counts()}")

# 로그 변환 타겟
df["log_mv"] = np.log1p(df["market_value"])

# ---------------------------------------------------------------------------
# 4. 포지션별 피처 정의
# ---------------------------------------------------------------------------
# 공통 피처
COMMON_FEATURES = [
    "age_used", "age_sq",
    "epl_experience",
    "team_points",
    "nationality_freq",
    "min",
    "log_mv_lag",
]

POSITION_FEATURES = {
    "FW": COMMON_FEATURES + [
        "goals_p90",
        "assists_p90",
        "shots_p90",
        "consistency_cv",
        "yellow_p90",
    ],
    "MID": COMMON_FEATURES + [
        "assists_p90",
        "goals_p90",
        "crosses_p90",    # key_passes 프록시 (크로스)
        "tackles_p90",    # 압박 참여
        "consistency_cv",
    ],
    "DEF": COMMON_FEATURES + [
        "tackles_p90",
        "int_p90",
        "yellow_p90",     # 파울 경향 (리스크)
        "goals_p90",      # 세트피스 득점 능력
        "consistency_cv",
    ],
    "GK": COMMON_FEATURES + [
        "clean_sheet_pct",
        "clean_sheets",
        "gk_matches",
    ],
}

# ---------------------------------------------------------------------------
# 5. 시간 기반 분할
# ---------------------------------------------------------------------------
print("\n[4/7] 시간 기반 데이터 분할...")

def assign_split(year):
    if pd.isna(year):
        return "train"
    if year < 2021:
        return "train"
    elif year <= 2022:
        return "val"
    else:
        return "test"

df["data_split"] = df["season_year"].apply(assign_split)
print(f"  Train (<2021):       {(df['data_split']=='train').sum()}")
print(f"  Val   (2021-2022):   {(df['data_split']=='val').sum()}")
print(f"  Test  (2023-2025):   {(df['data_split']=='test').sum()}")

# ---------------------------------------------------------------------------
# 6. 포지션별 모델 학습
# ---------------------------------------------------------------------------
print("\n[5/7] 포지션별 모델 학습...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

all_results = {}         # { pos: { model_name: metrics } }
all_test_dfs = []        # 포지션별 테스트 예측 결과
best_preds_all = {}      # { pos: best_pred_array }

POSITIONS = ["FW", "MID", "DEF", "GK"]

for pos in POSITIONS:
    print(f"\n  ===== {pos} =====")
    feat_cols = POSITION_FEATURES[pos]

    pos_df = df[df["pos_group"] == pos].copy()
    pos_df = safe_fillna(pos_df, feat_cols)

    # NaN 클리핑 (이상치)
    pos_df["consistency_cv"] = pos_df.get("consistency_cv", pd.Series(0)).clip(0, 10)

    train_d = pos_df[pos_df["data_split"] == "train"]
    val_d   = pos_df[pos_df["data_split"] == "val"]
    test_d  = pos_df[pos_df["data_split"] == "test"]
    print(f"    train={len(train_d)}, val={len(val_d)}, test={len(test_d)}")

    if len(train_d) < 10 or len(val_d) < 5 or len(test_d) < 5:
        print(f"    [경고] {pos} 데이터 부족, 건너뜀")
        continue

    X_tr = train_d[feat_cols].values.astype(np.float32)
    y_tr = train_d["log_mv"].values.astype(np.float32)
    X_vl = val_d[feat_cols].values.astype(np.float32)
    y_vl = val_d["log_mv"].values.astype(np.float32)
    X_te = test_d[feat_cols].values.astype(np.float32)
    y_te = test_d["log_mv"].values.astype(np.float32)

    y_tr_orig = train_d["market_value"].values
    y_vl_orig = val_d["market_value"].values
    y_te_orig = test_d["market_value"].values

    pos_results = {}
    pos_test_preds = {}

    # ---- Ridge ----
    ridge_sc = StandardScaler()
    Xtr_r = ridge_sc.fit_transform(X_tr)
    Xvl_r = ridge_sc.transform(X_vl)
    Xte_r = ridge_sc.transform(X_te)
    ridge = Ridge(alpha=100.0)
    ridge.fit(Xtr_r, y_tr)
    pred_log = ridge.predict(Xte_r)
    pred_orig = np.expm1(np.clip(pred_log, 0, 25))
    m = regression_metrics(y_te_orig, pred_orig, prefix="test_")
    pos_results["Ridge"] = m
    pos_test_preds["Ridge"] = pred_orig
    print(f"    Ridge:  R2={m['test_R2']:.4f}  MAE={m['test_MAE']:,.0f}  MAPE={m['test_MAPE']}%")
    joblib.dump(ridge, MODEL_DIR / f"ridge_{pos}.joblib")
    joblib.dump(ridge_sc, MODEL_DIR / f"scaler_ridge_{pos}.joblib")

    # ---- XGBoost ----
    xgb_m = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
        verbosity=0,
    )
    xgb_m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
    pred_log = xgb_m.predict(X_te)
    pred_orig = np.expm1(np.clip(pred_log, 0, 25))
    m = regression_metrics(y_te_orig, pred_orig, prefix="test_")
    pos_results["XGBoost"] = m
    pos_test_preds["XGBoost"] = pred_orig
    print(f"    XGBoost: R2={m['test_R2']:.4f}  MAE={m['test_MAE']:,.0f}  MAPE={m['test_MAPE']}%  (iter={xgb_m.best_iteration})")
    xgb_m.save_model(str(MODEL_DIR / f"xgb_{pos}.json"))

    # XGBoost 피처 중요도 저장
    imp_df = pd.DataFrame({
        "feature": feat_cols,
        "importance": xgb_m.feature_importances_,
    }).sort_values("importance", ascending=False)
    imp_df.to_csv(MODEL_DIR / f"xgb_importance_{pos}.csv", index=False)

    # ---- Random Forest ----
    rf_m = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=4,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    rf_m.fit(X_tr, y_tr)
    pred_log = rf_m.predict(X_te)
    pred_orig = np.expm1(np.clip(pred_log, 0, 25))
    m = regression_metrics(y_te_orig, pred_orig, prefix="test_")
    pos_results["RandomForest"] = m
    pos_test_preds["RandomForest"] = pred_orig
    print(f"    RF:     R2={m['test_R2']:.4f}  MAE={m['test_MAE']:,.0f}  MAPE={m['test_MAPE']}%")
    joblib.dump(rf_m, MODEL_DIR / f"rf_{pos}.joblib")

    # ---- MLP ----
    mlp_m, mlp_sc = train_mlp(X_tr, y_tr, X_vl, y_vl, device)
    pred_log = predict_mlp(mlp_m, mlp_sc, X_te, device)
    pred_orig = np.expm1(np.clip(pred_log, 0, 25))
    m = regression_metrics(y_te_orig, pred_orig, prefix="test_")
    pos_results["MLP"] = m
    pos_test_preds["MLP"] = pred_orig
    print(f"    MLP:    R2={m['test_R2']:.4f}  MAE={m['test_MAE']:,.0f}  MAPE={m['test_MAPE']}%")
    torch.save(mlp_m.state_dict(), MODEL_DIR / f"mlp_{pos}.pt")
    joblib.dump(mlp_sc, MODEL_DIR / f"scaler_mlp_{pos}.joblib")

    # ---- 최적 모델 선택 (R2 기준) ----
    best_name = max(pos_results, key=lambda k: pos_results[k]["test_R2"])
    best_pred = pos_test_preds[best_name]
    print(f"    → 최적 모델: {best_name} (R2={pos_results[best_name]['test_R2']:.4f})")

    # ---- 테스트 DF에 예측값 추가 ----
    test_copy = test_d.copy()
    test_copy["predicted_value"] = best_pred
    test_copy["best_model"] = best_name
    # 가치 점수: 예측가 / 실제가 (>1.5 → 저평가, <0.7 → 고평가)
    test_copy["value_score"] = test_copy["predicted_value"] / test_copy["market_value"]

    all_results[pos] = {
        "model_metrics": pos_results,
        "best_model": best_name,
        "feature_cols": feat_cols,
        "test_size": len(test_d),
    }
    all_test_dfs.append(test_copy)
    best_preds_all[pos] = best_pred

# ---------------------------------------------------------------------------
# 7. 스카우트 분석: 저평가 / 고평가 선수
# ---------------------------------------------------------------------------
print("\n[6/7] 스카우트 분석 중...")

# 전체 테스트셋 합치기
full_test = pd.concat(all_test_dfs, ignore_index=True)

# value_score 이상치 클리핑 (표시용)
full_test["value_score_clipped"] = full_test["value_score"].clip(0, 5)

# 저평가: value_score > 1.5 (모델이 실제보다 높게 평가)
undervalued = (
    full_test[full_test["value_score"] > 1.5]
    .sort_values("value_score", ascending=False)
    .head(20)
)
# 고평가: value_score < 0.7 (모델이 실제보다 낮게 평가)
overvalued = (
    full_test[full_test["value_score"] < 0.7]
    .sort_values("value_score", ascending=True)
    .head(20)
)

sig_under = (full_test["value_score"] > 1.5).sum()
sig_over  = (full_test["value_score"] < 0.7).sum()

print(f"\n  저평가 선수 수 (>1.5x): {sig_under}")
print(f"  고평가 선수 수 (<0.7x): {sig_over}")

print("\n  TOP 20 저평가 선수 (스카우트 바겐 리스트)")
print(f"  {'선수':<28} {'시즌':<10} {'포지션':<6} {'나이':<5} {'실제(M)':<10} {'예측(M)':<10} {'점수':>8}")
print("  " + "-" * 82)
for _, row in undervalued.iterrows():
    print(f"  {str(row['player'])[:27]:<28} {row['season']:<10} {str(row['pos_group']):<6} "
          f"{row['age_used']:<5.0f} {row['market_value']/1e6:>8.2f}M  {row['predicted_value']/1e6:>8.2f}M  "
          f"{row['value_score']:>7.2f}x")

print("\n  TOP 20 고평가 선수 (회피 리스트)")
print(f"  {'선수':<28} {'시즌':<10} {'포지션':<6} {'나이':<5} {'실제(M)':<10} {'예측(M)':<10} {'점수':>8}")
print("  " + "-" * 82)
for _, row in overvalued.iterrows():
    print(f"  {str(row['player'])[:27]:<28} {row['season']:<10} {str(row['pos_group']):<6} "
          f"{row['age_used']:<5.0f} {row['market_value']/1e6:>8.2f}M  {row['predicted_value']/1e6:>8.2f}M  "
          f"{row['value_score']:>7.2f}x")

# ---------------------------------------------------------------------------
# 8. 시각화
# ---------------------------------------------------------------------------
print("\n[7/7] 시각화 생성 중...")

POS_COLORS = {"FW": "#e63946", "MID": "#457b9d", "DEF": "#2d6a4f", "GK": "#f4a261"}

# --- (a) 포지션별 Predicted vs Actual 산점도 ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("S2: Position-Specific Market Value Prediction (Test Set)", fontsize=14, fontweight="bold")

for ax, pos in zip(axes.flatten(), POSITIONS):
    if pos not in best_preds_all:
        ax.set_visible(False)
        continue
    pos_test = full_test[full_test["pos_group"] == pos]
    actual    = pos_test["market_value"].values / 1e6
    predicted = pos_test["predicted_value"].values / 1e6
    r2_val    = all_results[pos]["model_metrics"][all_results[pos]["best_model"]]["test_R2"]

    ax.scatter(actual, predicted, alpha=0.35, s=18, color=POS_COLORS[pos], edgecolors="none")
    mx = max(actual.max(), predicted.max()) * 1.05
    ax.plot([0, mx], [0, mx], "k--", linewidth=1, label="Perfect fit")
    ax.set_xlabel("Actual MV (M€)")
    ax.set_ylabel("Predicted MV (M€)")
    ax.set_title(f"{pos}  |  {all_results[pos]['best_model']}  R²={r2_val:.3f}")
    ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(FIG_DIR / "predicted_vs_actual_by_position.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  저장: {FIG_DIR / 'predicted_vs_actual_by_position.png'}")

# --- (b) 포지션별 피처 중요도 ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("S2: Feature Importance by Position (XGBoost)", fontsize=14, fontweight="bold")

for ax, pos in zip(axes.flatten(), POSITIONS):
    imp_path = MODEL_DIR / f"xgb_importance_{pos}.csv"
    if not imp_path.exists():
        ax.set_visible(False)
        continue
    imp = pd.read_csv(imp_path).head(10).iloc[::-1]
    bars = ax.barh(imp["feature"], imp["importance"],
                   color=POS_COLORS[pos], edgecolor="black", alpha=0.85)
    ax.set_xlabel("Importance")
    ax.set_title(f"{pos} - Top Features")

plt.tight_layout()
fig.savefig(FIG_DIR / "feature_importance_by_position.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  저장: {FIG_DIR / 'feature_importance_by_position.png'}")

# --- (c) 저평가 선수 바 차트 ---
fig, ax = plt.subplots(figsize=(12, 8))
uv_plot = undervalued.head(15).copy()
uv_plot["label"] = uv_plot["player"].str[:20] + " (" + uv_plot["season"] + ")"
colors_uv = [POS_COLORS.get(p, "gray") for p in uv_plot["pos_group"]]
bars = ax.barh(range(len(uv_plot)), uv_plot["value_score"], color=colors_uv, edgecolor="black", alpha=0.85)
ax.set_yticks(range(len(uv_plot)))
ax.set_yticklabels(uv_plot["label"], fontsize=9)
ax.axvline(1.5, color="green", linestyle="--", linewidth=1.5, label="Undervalued threshold (1.5x)")
ax.set_xlabel("Value Score (Predicted / Actual)")
ax.set_title("TOP 15 Undervalued Players (Scout Bargain List)")
# 범례 패치
patches = [mpatches.Patch(color=POS_COLORS[p], label=p) for p in POSITIONS if p in uv_plot["pos_group"].values]
patches.append(mpatches.Patch(color="white", label=""))
patches.append(plt.Line2D([0], [0], color="green", linestyle="--", label="1.5x threshold"))
ax.legend(handles=patches, fontsize=8, loc="lower right")
plt.tight_layout()
fig.savefig(FIG_DIR / "undervalued_bargain_list.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  저장: {FIG_DIR / 'undervalued_bargain_list.png'}")

# --- (d) 고평가 선수 바 차트 ---
fig, ax = plt.subplots(figsize=(12, 8))
ov_plot = overvalued.head(15).copy()
ov_plot["label"] = ov_plot["player"].str[:20] + " (" + ov_plot["season"] + ")"
colors_ov = [POS_COLORS.get(p, "gray") for p in ov_plot["pos_group"]]
ax.barh(range(len(ov_plot)), ov_plot["value_score"], color=colors_ov, edgecolor="black", alpha=0.85)
ax.set_yticks(range(len(ov_plot)))
ax.set_yticklabels(ov_plot["label"], fontsize=9)
ax.axvline(0.7, color="red", linestyle="--", linewidth=1.5, label="Overvalued threshold (0.7x)")
ax.set_xlabel("Value Score (Predicted / Actual)")
ax.set_title("TOP 15 Overvalued Players (Scout Avoid List)")
patches2 = [mpatches.Patch(color=POS_COLORS[p], label=p) for p in POSITIONS if p in ov_plot["pos_group"].values]
patches2.append(plt.Line2D([0], [0], color="red", linestyle="--", label="0.7x threshold"))
ax.legend(handles=patches2, fontsize=8, loc="lower right")
plt.tight_layout()
fig.savefig(FIG_DIR / "overvalued_avoid_list.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  저장: {FIG_DIR / 'overvalued_avoid_list.png'}")

# --- (e) 포지션별 R2 비교 막대 그래프 ---
fig, ax = plt.subplots(figsize=(10, 5))
model_names = ["Ridge", "XGBoost", "RandomForest", "MLP"]
bar_width = 0.18
x = np.arange(len(POSITIONS))
for i, mname in enumerate(model_names):
    r2_vals = []
    for pos in POSITIONS:
        if pos in all_results and mname in all_results[pos]["model_metrics"]:
            r2_vals.append(all_results[pos]["model_metrics"][mname]["test_R2"])
        else:
            r2_vals.append(0)
    ax.bar(x + i * bar_width, r2_vals, bar_width, label=mname, alpha=0.85)
ax.set_xticks(x + bar_width * 1.5)
ax.set_xticklabels(POSITIONS)
ax.set_ylabel("R² Score (Test Set)")
ax.set_title("S2: Model Performance by Position")
ax.legend()
ax.set_ylim(0, 1)
ax.axhline(0, color="black", linewidth=0.5)
plt.tight_layout()
fig.savefig(FIG_DIR / "model_comparison_by_position.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  저장: {FIG_DIR / 'model_comparison_by_position.png'}")

# ---------------------------------------------------------------------------
# 9. 결과 저장
# ---------------------------------------------------------------------------
# ---- undervalued_players.parquet → data/scout/ ----
scout_cols = [
    "player", "season", "team", "pos_group", "age_used",
    "market_value", "predicted_value", "value_score",
    "goals_p90", "assists_p90", "min", "epl_experience",
]
scout_cols_available = [c for c in scout_cols if c in full_test.columns]
undervalued_full = full_test[full_test["value_score"] > 1.5].sort_values("value_score", ascending=False)
undervalued_full[scout_cols_available].to_parquet(SCOUT_DIR / "undervalued_players.parquet", index=False)
print(f"\n  저장: {SCOUT_DIR / 'undervalued_players.parquet'}  ({len(undervalued_full)} rows)")

# ---- results_summary.json ----
def make_player_dict(row):
    return {
        "player":                  str(row["player"]),
        "season":                  str(row["season"]),
        "team":                    str(row.get("team", "")),
        "position":                str(row["pos_group"]),
        "age":                     float(row["age_used"]) if not pd.isna(row["age_used"]) else None,
        "actual_market_value_eur": int(row["market_value"]),
        "predicted_market_value_eur": int(row["predicted_value"]),
        "value_score":             round(float(row["value_score"]), 3),
    }

undervalued_list = [make_player_dict(r) for _, r in undervalued.iterrows()]
overvalued_list  = [make_player_dict(r) for _, r in overvalued.iterrows()]

# 포지션별 최고 모델 요약
pos_summary = {}
for pos, res in all_results.items():
    best = res["best_model"]
    pos_summary[pos] = {
        "best_model":    best,
        "test_R2":       res["model_metrics"][best]["test_R2"],
        "test_MAE":      res["model_metrics"][best]["test_MAE"],
        "test_MAPE":     res["model_metrics"][best]["test_MAPE"],
        "test_size":     res["test_size"],
        "feature_cols":  res["feature_cols"],
        "all_model_metrics": res["model_metrics"],
    }

results_summary = {
    "pipeline":         "S2 - Position-Specific Market Value Prediction",
    "created":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "improvements_over_p6": [
        "포지션별 전용 모델 (FW/MID/DEF/GK)",
        "포지션 특화 피처 구성",
        "매치 로그 기반 수비/공격 세부 스탯 집계",
        "GK 클린시트/세이브% 피처",
        "팀 강도(포인트) 피처 추가",
        "시장가치 래그(이전 시즌) 피처 추가",
    ],
    "data": {
        "total_rows_after_filter": int(len(df)),
        "train_size": int((df["data_split"]=="train").sum()),
        "val_size":   int((df["data_split"]=="val").sum()),
        "test_size":  int((df["data_split"]=="test").sum()),
    },
    "position_models":  pos_summary,
    "value_analysis": {
        "significantly_undervalued_count": int(sig_under),
        "significantly_overvalued_count":  int(sig_over),
        "undervalued_threshold": ">1.5x",
        "overvalued_threshold":  "<0.7x",
    },
    "top20_undervalued": undervalued_list,
    "top20_overvalued":  overvalued_list,
}

class NumpyEncoder(json.JSONEncoder):
    """numpy 타입 JSON 직렬화 처리"""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

with open(MODEL_DIR / "results_summary.json", "w", encoding="utf-8") as f:
    json.dump(results_summary, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
print(f"  저장: {MODEL_DIR / 'results_summary.json'}")

print("\n" + "=" * 70)
print("S2 파이프라인 완료!")
for pos, res in pos_summary.items():
    print(f"  {pos}: {res['best_model']}  R2={res['test_R2']:.4f}  MAE={res['test_MAE']:,.0f}€  MAPE={res['test_MAPE']}%")
print("=" * 70)

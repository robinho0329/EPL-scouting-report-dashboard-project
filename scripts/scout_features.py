"""
scout_features.py
스카우트 / 트랜스퍼 매니저 전용 피처 엔지니어링 파이프라인
=============================================================
입력:  data/processed/*.parquet
출력:  data/scout/scout_player_profiles.parquet
       data/scout/scout_team_profiles.parquet
       data/scout/scout_transfers.parquet
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# 0. 경로 설정 및 데이터 로드
# ─────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
SCOUT_DIR  = os.path.join(BASE_DIR, "data", "scout")
os.makedirs(SCOUT_DIR, exist_ok=True)

print("=" * 60)
print("EPL Scout Feature Engineering Pipeline")
print("=" * 60)

print("\n[1/3] 원본 데이터 로드 중...")
match_results      = pd.read_parquet(os.path.join(PROC_DIR, "match_results.parquet"))
team_season_summary = pd.read_parquet(os.path.join(PROC_DIR, "team_season_summary.parquet"))
player_season_stats = pd.read_parquet(os.path.join(PROC_DIR, "player_season_stats.parquet"))
player_match_logs   = pd.read_parquet(os.path.join(PROC_DIR, "player_match_logs.parquet"))

print(f"  match_results:       {match_results.shape}")
print(f"  team_season_summary: {team_season_summary.shape}")
print(f"  player_season_stats: {player_season_stats.shape}")
print(f"  player_match_logs:   {player_match_logs.shape}")


# ─────────────────────────────────────────
# 공통 유틸리티
# ─────────────────────────────────────────

# 시즌 문자열을 정수 순서로 변환 (예: '2000/01' → 2000)
def season_to_int(s):
    try:
        return int(str(s)[:4])
    except Exception:
        return np.nan

# 변동계수(CV) 계산 - 일관성 측정에 사용
def coef_variation(series):
    """낮을수록 일관성 높음. NaN이 많을 경우 NaN 반환."""
    clean = series.dropna()
    if len(clean) < 3:
        return np.nan
    mu = clean.mean()
    if mu == 0:
        return np.nan
    return clean.std() / mu

# 빅6 팀 목록 (EPL 전통적 빅6)
TOP6 = {"Arsenal", "Chelsea", "Liverpool", "Man City", "Man United", "Tottenham"}

# 포지션 그룹 매핑 (FBref pos 기준)
def map_pos_group(pos_str):
    """pos 컬럼(FBref)을 GK/DF/MF/FW 4개 그룹으로 단순화"""
    if pd.isna(pos_str):
        return "Unknown"
    p = str(pos_str).upper()
    if "GK" in p:
        return "GK"
    elif p.startswith("DF"):
        return "DF"
    elif p.startswith("MF"):
        return "MF"
    elif p.startswith("FW"):
        return "FW"
    else:
        return "Unknown"

# 나이 카테고리 분류
def age_category(age):
    if pd.isna(age):
        return "Unknown"
    if age < 21:
        return "youth"
    elif age < 24:
        return "developing"
    elif age < 30:
        return "prime"
    else:
        return "veteran"


###############################################################################
# ══════════════════════════════════════════════════════════════════════════════
# PART 1: scout_player_profiles
# ══════════════════════════════════════════════════════════════════════════════
###############################################################################
print("\n" + "=" * 60)
print("[PART 1] scout_player_profiles 생성")
print("=" * 60)

# ── 기본 복사 ──────────────────────────────────────────────────────────────
pp = player_season_stats.copy()
pp["season_int"] = pp["season"].apply(season_to_int)

# ── 포지션 그룹 ────────────────────────────────────────────────────────────
pp["pos_group"] = pp["pos"].apply(map_pos_group)

# ── 나이 카테고리 ──────────────────────────────────────────────────────────
pp["age_category"] = pp["age"].apply(age_category)


# ─────────────────────────────────────────
# 1-A. Per-90 정규화 스탯
# ─────────────────────────────────────────
print("  1-A: Per-90 정규화 스탯 계산...")

# 90s 컬럼이 0이거나 NaN인 경우 min/90으로 추정
pp["90s_safe"] = pp["90s"].fillna(pp["min"].fillna(0) / 90.0)
pp["90s_safe"] = pp["90s_safe"].replace(0, np.nan)

for stat in ["gls", "ast", "g_a", "crdy", "crdr"]:
    pp[f"{stat}_p90"] = pp[stat] / pp["90s_safe"]


# ─────────────────────────────────────────
# 1-B. WAR 스타일 종합 레이팅 (포지션 가중치)
# ─────────────────────────────────────────
print("  1-B: WAR 스타일 종합 레이팅 계산...")

# 포지션별 가중치 딕셔너리
# (gls, ast, g_a, crdy_penalty) — crdy는 페널티
WAR_WEIGHTS = {
    "GK":      {"gls_p90": 0.5, "ast_p90": 0.3, "crdy_p90": -0.3, "crdr_p90": -1.0},
    "DF":      {"gls_p90": 1.0, "ast_p90": 1.0, "crdy_p90": -0.5, "crdr_p90": -2.0},
    "MF":      {"gls_p90": 2.0, "ast_p90": 2.5, "crdy_p90": -0.4, "crdr_p90": -1.5},
    "FW":      {"gls_p90": 3.5, "ast_p90": 2.0, "crdy_p90": -0.3, "crdr_p90": -1.2},
    "Unknown": {"gls_p90": 2.0, "ast_p90": 2.0, "crdy_p90": -0.4, "crdr_p90": -1.5},
}

def compute_war_rating(row):
    w = WAR_WEIGHTS.get(row["pos_group"], WAR_WEIGHTS["Unknown"])
    score = 0.0
    for col, weight in w.items():
        val = row.get(col, np.nan)
        if pd.notna(val):
            score += val * weight
    return round(score, 4)

pp["war_rating"] = pp.apply(compute_war_rating, axis=1)


# ─────────────────────────────────────────
# 1-C. 일관성 점수 (매치로그 CV 기반)
# ─────────────────────────────────────────
print("  1-C: 일관성 점수 계산 (매치로그 기반)...")

# 매치로그에서 선수-시즌별 출전 경기 통계 집계
pml_active = player_match_logs[player_match_logs["min"] > 0].copy()

# 기여 지표: 골+어시스트
pml_active["contribution"] = pml_active["gls"].fillna(0) + pml_active["ast"].fillna(0)

# 선수-시즌 그룹별 일관성 계산
print("    매치로그 선수-시즌별 일관성 집계 중...")
# pandas 2.0 호환: groupby 후 컬럼 직접 집계
consistency_df = (
    pml_active.groupby(["player", "season"])["contribution"]
    .apply(coef_variation)
    .reset_index()
    .rename(columns={"contribution": "consistency_cv"})
)

# 일관성 점수 = 1 - 정규화 CV (높을수록 일관적)
# CV가 NaN인 경우 NaN 유지
max_cv = consistency_df["consistency_cv"].quantile(0.95)
consistency_df["consistency_score"] = (
    1 - (consistency_df["consistency_cv"] / max_cv).clip(0, 1)
)

pp = pp.merge(consistency_df[["player", "season", "consistency_score"]], on=["player", "season"], how="left")


# ─────────────────────────────────────────
# 1-D. 미닛 셰어 (팀 총 출전 시간 대비 %)
# ─────────────────────────────────────────
print("  1-D: 미닛 셰어 계산...")

# 팀-시즌별 총 출전 시간 (이론상 38경기 × 11선수 × 90분 = 37,620분이지만 실제 데이터 기반)
team_total_min = (
    pp.groupby(["team", "season"])["min"]
    .sum()
    .reset_index()
    .rename(columns={"min": "team_total_min"})
)
pp = pp.merge(team_total_min, on=["team", "season"], how="left")
pp["minutes_share"] = (pp["min"] / pp["team_total_min"] * 100).round(2)


# ─────────────────────────────────────────
# 1-E. 마켓 밸류 효율성
# ─────────────────────────────────────────
print("  1-E: 마켓 밸류 효율성 계산...")

# 성과 점수 = WAR 레이팅 (단, 최소 450분 이상만 의미 있는 값으로 처리)
pp["performance_score"] = np.where(
    pp["min"].fillna(0) >= 450,
    pp["war_rating"],
    np.nan
)
# 시장가치 > 0인 경우만 계산
pp["market_value_efficiency"] = np.where(
    (pp["market_value"].notna()) & (pp["market_value"] > 0),
    pp["performance_score"] / (pp["market_value"] / 1_000_000),  # 백만 유로 단위
    np.nan
)
pp["market_value_efficiency"] = pp["market_value_efficiency"].round(4)


# ─────────────────────────────────────────
# 1-F. 커리어 궤적 및 시즌 간 개선율
# ─────────────────────────────────────────
print("  1-F: 커리어 궤적 및 시즌 간 개선율 계산...")

# 선수-시즌 정렬
pp_sorted = pp.sort_values(["player", "season_int"])

# 시즌 간 WAR 개선율 계산
pp_sorted["prev_war_rating"] = pp_sorted.groupby("player")["war_rating"].shift(1)
pp_sorted["war_improvement"] = pp_sorted["war_rating"] - pp_sorted["prev_war_rating"]

def assign_trajectory(improvements):
    """
    최근 2시즌 이상 데이터가 있는 경우:
      - improving: 평균 WAR 개선율 > 0.1
      - declining: 평균 WAR 개선율 < -0.1
      - stable: 그 외
    pandas 2.0 호환: Series(war_improvement)를 직접 받음
    """
    clean = improvements.dropna()
    if len(clean) < 1:
        return "unknown"
    avg_imp = clean.mean()
    if avg_imp > 0.1:
        return "improving"
    elif avg_imp < -0.1:
        return "declining"
    else:
        return "stable"

trajectory_df = (
    pp_sorted.groupby("player")["war_improvement"]
    .apply(assign_trajectory)
    .reset_index()
    .rename(columns={"war_improvement": "career_trajectory"})
)

pp_sorted = pp_sorted.merge(trajectory_df, on="player", how="left")

# 시즌 간 개선율 (백분율)
pp_sorted["season_improvement_rate"] = np.where(
    pp_sorted["prev_war_rating"].notna() & (pp_sorted["prev_war_rating"] != 0),
    (pp_sorted["war_improvement"] / pp_sorted["prev_war_rating"].abs() * 100).round(2),
    np.nan
)

pp = pp_sorted.copy()


# ─────────────────────────────────────────
# 1-G. 밸류 모멘텀 (시장가치 변화율)
# ─────────────────────────────────────────
print("  1-G: 밸류 모멘텀 계산...")

pp = pp.sort_values(["player", "season_int"])
pp["prev_market_value"] = pp.groupby("player")["market_value"].shift(1)
pp["value_momentum"] = np.where(
    (pp["prev_market_value"].notna()) & (pp["prev_market_value"] > 0),
    ((pp["market_value"] - pp["prev_market_value"]) / pp["prev_market_value"] * 100).round(2),
    np.nan
)


# ─────────────────────────────────────────
# 1-H. 다재다능함 점수 (포지션 다양성)
# ─────────────────────────────────────────
print("  1-H: 다재다능함(Versatility) 점수 계산...")

# 매치로그에서 선수가 플레이한 고유 포지션 수 계산
pos_variety = (
    player_match_logs[player_match_logs["pos"].notna() & (player_match_logs["min"] > 0)]
    .groupby(["player", "season"])["pos"]
    .nunique()
    .reset_index()
    .rename(columns={"pos": "n_positions_played"})
)
pp = pp.merge(pos_variety, on=["player", "season"], how="left")

# 1~3+ 포지션 → 다재다능함 점수 (1=전문화, 3+=다재다능)
pp["versatility_score"] = pp["n_positions_played"].fillna(1).clip(1, 5) / 5.0
pp["versatility_score"] = pp["versatility_score"].round(3)


# ─────────────────────────────────────────
# 1-I. 빅매치 퍼포먼스 (빅6 상대 성과)
# ─────────────────────────────────────────
print("  1-I: 빅매치 퍼포먼스 계산...")

# 빅6 상대 매치로그 필터
pml_big6 = player_match_logs[
    (player_match_logs["opponent"].isin(TOP6)) &
    (player_match_logs["min"].fillna(0) > 0)
].copy()
pml_big6["contribution"] = pml_big6["gls"].fillna(0) + pml_big6["ast"].fillna(0)

big_match_perf = (
    pml_big6.groupby(["player", "season"])
    .agg(
        big6_matches=("min", "count"),
        big6_goals=("gls", "sum"),
        big6_assists=("ast", "sum"),
        big6_contribution=("contribution", "sum"),
        big6_min=("min", "sum"),
    )
    .reset_index()
)
big_match_perf["big6_contribution_p90"] = (
    big_match_perf["big6_contribution"] / (big_match_perf["big6_min"] / 90)
).replace([np.inf, -np.inf], np.nan).round(4)

pp = pp.merge(big_match_perf, on=["player", "season"], how="left")


# ─────────────────────────────────────────
# 1-J. 팀 의존도 (선수 출전 여부에 따른 팀 성과 차이)
# ─────────────────────────────────────────
print("  1-J: 팀 의존도 계산...")

# 매치로그에서 경기별 선수 출전 여부 + 팀 결과 연계
pml_team = player_match_logs[["player", "season", "team", "min", "outcome"]].copy()
pml_team["played"] = pml_team["min"].fillna(0) > 0

# 팀-시즌별 총 매치 수 (outcome 기준)
# outcome: 'W', 'D', 'L'
pml_team["is_win"] = pml_team["outcome"] == "W"

# 선수 출전 경기 승률
with_player = (
    pml_team[pml_team["played"]]
    .groupby(["player", "season", "team"])["is_win"]
    .mean()
    .reset_index()
    .rename(columns={"is_win": "win_rate_with_player"})
)
# 선수 불출전 경기 승률 (같은 팀 매치에서)
# pandas 2.0 호환: groupby apply 대신 피벗 방식 사용
# (팀-시즌-경기일 기준으로 불출전 경기 승률 계산)
def win_rate_without(group):
    """불출전 경기의 평균 승률 계산"""
    absent = group[~group["played"]]
    if len(absent) == 0:
        return np.nan
    return absent["is_win"].mean()

team_match_win = (
    pml_team.groupby(["team", "season", "player"])
    .apply(win_rate_without)
    .reset_index()
    .rename(columns={0: "win_rate_without_player"})
)

team_dep = with_player.merge(team_match_win, on=["player", "season", "team"], how="left")
team_dep["team_dependency_score"] = (
    team_dep["win_rate_with_player"] - team_dep["win_rate_without_player"]
).round(4)

pp = pp.merge(
    team_dep[["player", "season", "team_dependency_score",
              "win_rate_with_player", "win_rate_without_player"]],
    on=["player", "season"],
    how="left"
)


# ─────────────────────────────────────────
# 1-K. 최종 컬럼 정리 및 저장
# ─────────────────────────────────────────
print("  1-K: 최종 컬럼 정리 및 저장...")

PLAYER_PROFILE_COLS = [
    # 식별자
    "player", "team", "season", "season_int",
    # 기본 스탯
    "pos", "pos_group", "position", "age", "age_category",
    "mp", "starts", "min", "90s_safe",
    "gls", "ast", "g_a", "g_pk", "pk", "pkatt",
    "crdy", "crdr",
    # Per-90 스탯
    "gls_p90", "ast_p90", "g_a_p90", "crdy_p90", "crdr_p90",
    # 종합 레이팅
    "war_rating",
    # 일관성
    "consistency_score",
    # 분출전
    "minutes_share",
    # 마켓 밸류
    "market_value", "market_value_efficiency",
    # 나이·궤적
    "career_trajectory", "season_improvement_rate",
    # 밸류 모멘텀
    "value_momentum",
    # 다재다능함
    "n_positions_played", "versatility_score",
    # 팀 의존도
    "win_rate_with_player", "win_rate_without_player", "team_dependency_score",
    # 빅매치
    "big6_matches", "big6_goals", "big6_assists",
    "big6_contribution_p90",
    # 이적 플래그
    "transfer_flag",
    # 추가 메타
    "nationality", "foot", "height_cm", "birth_year",
    "player_id",
]

# 실제 존재하는 컬럼만 선택
available_cols = [c for c in PLAYER_PROFILE_COLS if c in pp.columns]
scout_player = pp[available_cols].copy()

# 무한값 처리
scout_player.replace([np.inf, -np.inf], np.nan, inplace=True)

# 저장 (Parquet)
out_path_player = os.path.join(SCOUT_DIR, "scout_player_profiles.parquet")
scout_player.to_parquet(out_path_player, index=False)

# CSV 저장 (UTF-8 BOM)
scout_player.to_csv(
    out_path_player.replace(".parquet", ".csv"),
    index=False, encoding="utf-8-sig"
)

print(f"\n  → 저장 완료: {out_path_player}")
print(f"  Shape: {scout_player.shape}")
print("\n  [scout_player_profiles 요약 통계]")
summary_cols = ["war_rating", "gls_p90", "ast_p90", "consistency_score",
                "minutes_share", "market_value_efficiency",
                "versatility_score", "team_dependency_score", "big6_contribution_p90"]
available_summary = [c for c in summary_cols if c in scout_player.columns]
print(scout_player[available_summary].describe().round(4).to_string())


###############################################################################
# ══════════════════════════════════════════════════════════════════════════════
# PART 2: scout_team_profiles
# ══════════════════════════════════════════════════════════════════════════════
###############################################################################
print("\n" + "=" * 60)
print("[PART 2] scout_team_profiles 생성")
print("=" * 60)

tp = team_season_summary.copy()
tp.rename(columns={"Season": "season"}, inplace=True)
tp["season_int"] = tp["season"].apply(season_to_int)


# ─────────────────────────────────────────
# 2-A. 스쿼드 뎁스 지표
# ─────────────────────────────────────────
print("  2-A: 스쿼드 뎁스 지표 계산...")

# 900분 이상 출전 선수 수
squad_depth = (
    player_season_stats[player_season_stats["min"].fillna(0) >= 900]
    .groupby(["team", "season"])["player"]
    .nunique()
    .reset_index()
    .rename(columns={"player": "squad_depth_900min"})
)
tp = tp.merge(squad_depth, on=["team", "season"], how="left")

# 300분 이상 출전 선수 수 (로테이션 뎁스)
squad_depth_300 = (
    player_season_stats[player_season_stats["min"].fillna(0) >= 300]
    .groupby(["team", "season"])["player"]
    .nunique()
    .reset_index()
    .rename(columns={"player": "squad_depth_300min"})
)
tp = tp.merge(squad_depth_300, on=["team", "season"], how="left")


# ─────────────────────────────────────────
# 2-B. 평균 스쿼드 나이 & 시장가치
# ─────────────────────────────────────────
print("  2-B: 평균 스쿼드 나이 및 시장가치 계산...")

# 900분 이상 출전 선수만 사용 (주전 스쿼드 기준)
squad_age_val = (
    player_season_stats[player_season_stats["min"].fillna(0) >= 900]
    .groupby(["team", "season"])
    .agg(
        avg_squad_age=("age", "mean"),
        avg_squad_market_value=("market_value", "mean"),
        total_squad_market_value=("market_value", "sum"),
        n_players_with_value=("market_value", lambda x: x.notna().sum()),
    )
    .reset_index()
)
tp = tp.merge(squad_age_val, on=["team", "season"], how="left")


# ─────────────────────────────────────────
# 2-C. 플레이 스타일 지표
# ─────────────────────────────────────────
print("  2-C: 플레이 스타일 지표 계산...")

# 공격/수비 균형
tp["attack_defense_ratio"] = (tp["total_goals_for"] / tp["total_goals_against"].replace(0, np.nan)).round(3)

# 홈/어웨이 강점 (승점 기준)
tp["home_points"] = tp["home_wins"] * 3 + tp["home_draws"]
tp["away_points"] = tp["away_wins"] * 3 + tp["away_draws"]
tp["home_strength"] = (tp["home_points"] / (tp["home_played"] * 3)).round(3)
tp["away_strength"] = (tp["away_points"] / (tp["away_played"] * 3)).round(3)
tp["home_away_ratio"] = (tp["home_strength"] / tp["away_strength"].replace(0, np.nan)).round(3)

# 득점 효율성
tp["goals_per_game"] = (tp["total_goals_for"] / tp["total_played"]).round(3)
tp["goals_against_per_game"] = (tp["total_goals_against"] / tp["total_played"]).round(3)
tp["clean_sheet_proxy"] = (1 - tp["goals_against_per_game"] / tp["goals_against_per_game"].max()).round(3)


# ─────────────────────────────────────────
# 2-D. 유스 개발 점수 (U21 선수 출전 시간)
# ─────────────────────────────────────────
print("  2-D: 유스 개발 점수 계산...")

# U21 선수 = 나이 < 21 (FBref age 기준)
u21_min = (
    player_season_stats[player_season_stats["age"].fillna(99) < 21]
    .groupby(["team", "season"])
    .agg(
        u21_players=("player", "nunique"),
        u21_total_min=("min", "sum"),
    )
    .reset_index()
)

# 팀-시즌별 총 출전 시간
team_total = (
    player_season_stats.groupby(["team", "season"])["min"]
    .sum()
    .reset_index()
    .rename(columns={"min": "team_total_min"})
)
u21_min = u21_min.merge(team_total, on=["team", "season"], how="left")
u21_min["youth_development_score"] = (
    u21_min["u21_total_min"] / u21_min["team_total_min"] * 100
).round(2)

tp = tp.merge(
    u21_min[["team", "season", "u21_players", "u21_total_min", "youth_development_score"]],
    on=["team", "season"], how="left"
)


# ─────────────────────────────────────────
# 2-E. 스쿼드 턴오버율 (시즌별 신규 선수 비율)
# ─────────────────────────────────────────
print("  2-E: 스쿼드 턴오버율 계산...")

# 시즌별 팀 선수 목록
squad_by_season = (
    player_season_stats[player_season_stats["min"].fillna(0) >= 300]
    .groupby(["team", "season"])["player"]
    .apply(set)
    .reset_index()
    .rename(columns={"player": "player_set"})
)
squad_by_season = squad_by_season.sort_values(["team", "season"])

# 이전 시즌 선수 목록 (같은 팀)
squad_by_season["prev_player_set"] = squad_by_season.groupby("team")["player_set"].shift(1)

def calc_turnover(row):
    """새로운 선수 수 / 현재 시즌 선수 수"""
    if pd.isna(row["prev_player_set"]) or not isinstance(row["prev_player_set"], set):
        return np.nan
    current = row["player_set"]
    prev    = row["prev_player_set"]
    new_players = len(current - prev)
    return round(new_players / len(current) * 100, 2) if current else np.nan

squad_by_season["squad_turnover_rate"] = squad_by_season.apply(calc_turnover, axis=1)
squad_by_season["new_players_count"] = squad_by_season.apply(
    lambda r: len(r["player_set"] - r["prev_player_set"])
    if isinstance(r["prev_player_set"], set) else np.nan,
    axis=1
)

tp = tp.merge(
    squad_by_season[["team", "season", "squad_turnover_rate", "new_players_count"]],
    on=["team", "season"], how="left"
)


# ─────────────────────────────────────────
# 2-F. 임금 효율성 프록시 (승점 / 총 시장가치)
# ─────────────────────────────────────────
print("  2-F: 임금 효율성 프록시 계산...")

# points / total_squad_market_value (백만 유로 단위)
tp["wage_efficiency_proxy"] = np.where(
    (tp["total_squad_market_value"].notna()) & (tp["total_squad_market_value"] > 0),
    (tp["points"] / (tp["total_squad_market_value"] / 1_000_000)).round(4),
    np.nan
)


# ─────────────────────────────────────────
# 2-G. 최종 정리 및 저장
# ─────────────────────────────────────────
print("  2-G: 최종 정리 및 저장...")

tp.replace([np.inf, -np.inf], np.nan, inplace=True)

out_path_team = os.path.join(SCOUT_DIR, "scout_team_profiles.parquet")
tp.to_parquet(out_path_team, index=False)
tp.to_csv(out_path_team.replace(".parquet", ".csv"), index=False, encoding="utf-8-sig")

print(f"\n  → 저장 완료: {out_path_team}")
print(f"  Shape: {tp.shape}")
print("\n  [scout_team_profiles 요약 통계]")
team_summary_cols = [
    "squad_depth_900min", "avg_squad_age", "total_squad_market_value",
    "attack_defense_ratio", "home_strength", "away_strength",
    "youth_development_score", "squad_turnover_rate", "wage_efficiency_proxy"
]
available_team_cols = [c for c in team_summary_cols if c in tp.columns]
print(tp[available_team_cols].describe().round(4).to_string())


###############################################################################
# ══════════════════════════════════════════════════════════════════════════════
# PART 3: scout_transfers
# ══════════════════════════════════════════════════════════════════════════════
###############################################################################
print("\n" + "=" * 60)
print("[PART 3] scout_transfers 생성")
print("=" * 60)

# ─────────────────────────────────────────
# 3-A. 이적 이벤트 감지
#      - 동일 선수가 연속 시즌에 다른 팀 소속 = 이적 이벤트
#      - transfer_flag가 True이거나 팀이 바뀐 경우
# ─────────────────────────────────────────
print("  3-A: 이적 이벤트 감지...")

pss_sorted = player_season_stats.sort_values(["player", "season"]).copy()
pss_sorted["season_int"] = pss_sorted["season"].apply(season_to_int)
pss_sorted = pss_sorted.sort_values(["player", "season_int"])

# 이전 시즌 팀 컬럼 추가
pss_sorted["prev_team"]    = pss_sorted.groupby("player")["team"].shift(1)
pss_sorted["prev_season"]  = pss_sorted.groupby("player")["season"].shift(1)
pss_sorted["prev_season_int"] = pss_sorted.groupby("player")["season_int"].shift(1)

# 이적 조건: 팀이 바뀌고 연속 시즌 (1년 차이)
transfer_mask = (
    pss_sorted["prev_team"].notna() &
    (pss_sorted["team"] != pss_sorted["prev_team"]) &
    ((pss_sorted["season_int"] - pss_sorted["prev_season_int"]) == 1)
)

transfers = pss_sorted[transfer_mask].copy()
print(f"    감지된 이적 이벤트: {len(transfers)}건")


# ─────────────────────────────────────────
# 3-B. 이전/신규 팀 스탯 결합
# ─────────────────────────────────────────
print("  3-B: 이전/신규 팀 스탯 결합...")

# 팀 프로필에서 필요한 컬럼 추출
team_stats_for_transfer = tp[[
    "team", "season", "points", "total_goals_for", "total_goals_against",
    "attack_defense_ratio", "home_strength", "away_strength",
    "avg_squad_age", "total_squad_market_value",
    "youth_development_score"
]].copy()

# 이전 팀 스탯 (이전 시즌)
transfers = transfers.merge(
    team_stats_for_transfer.rename(columns={
        c: f"prev_team_{c}" for c in team_stats_for_transfer.columns
        if c not in ["team", "season"]
    }).rename(columns={"team": "prev_team_name", "season": "prev_season_name"}),
    left_on=["prev_team", "prev_season"],
    right_on=["prev_team_name", "prev_season_name"],
    how="left"
)
transfers.drop(columns=["prev_team_name", "prev_season_name"], errors="ignore", inplace=True)

# 신규 팀 스탯 (이적 후 시즌)
transfers = transfers.merge(
    team_stats_for_transfer.rename(columns={
        c: f"new_team_{c}" for c in team_stats_for_transfer.columns
        if c not in ["team", "season"]
    }).rename(columns={"team": "new_team_name", "season": "new_season_name"}),
    left_on=["team", "season"],
    right_on=["new_team_name", "new_season_name"],
    how="left"
)
transfers.drop(columns=["new_team_name", "new_season_name"], errors="ignore", inplace=True)


# ─────────────────────────────────────────
# 3-C. 적응 피처
# ─────────────────────────────────────────
print("  3-C: 적응 피처 계산...")

# 이전 시즌 선수 스탯 가져오기 (WAR 레이팅, per90 등)
scout_player_temp = scout_player[
    ["player", "team", "season", "war_rating", "gls_p90", "ast_p90",
     "minutes_share", "pos_group", "market_value", "age"]
].copy()

# 이전 시즌 선수 스탯
transfers = transfers.merge(
    scout_player_temp.rename(columns={
        c: f"prev_{c}" for c in scout_player_temp.columns
        if c not in ["player"]
    }).rename(columns={
        "prev_team": "prev_team_stat",
        "prev_season": "prev_season_stat",
    }),
    left_on=["player", "prev_team", "prev_season"],
    right_on=["player", "prev_team_stat", "prev_season_stat"],
    how="left"
)
transfers.drop(columns=["prev_team_stat", "prev_season_stat"], errors="ignore", inplace=True)

# 이적 후 시즌 선수 스탯
transfers = transfers.merge(
    scout_player_temp.rename(columns={
        c: f"new_{c}" for c in scout_player_temp.columns
        if c not in ["player"]
    }).rename(columns={
        "new_team": "new_team_stat",
        "new_season": "new_season_stat",
    }),
    left_on=["player", "team", "season"],
    right_on=["player", "new_team_stat", "new_season_stat"],
    how="left"
)
transfers.drop(columns=["new_team_stat", "new_season_stat"], errors="ignore", inplace=True)

# 적응 지표: 이적 후 WAR 변화
transfers["adaptation_war_delta"] = (
    transfers["new_war_rating"] - transfers["prev_war_rating"]
).round(4)

# 적응 지표: 출전 시간 비율 변화
transfers["adaptation_minutes_share_delta"] = (
    transfers["new_minutes_share"] - transfers["prev_minutes_share"]
).round(2)

# 적응 성공 여부 (WAR 개선 + 출전시간 유지)
def adaptation_success(row):
    war_ok   = row.get("adaptation_war_delta", np.nan)
    min_ok   = row.get("adaptation_minutes_share_delta", np.nan)
    if pd.isna(war_ok) or pd.isna(min_ok):
        return "unknown"
    if war_ok > 0.1 and min_ok >= -5:
        return "successful"
    elif war_ok < -0.2 or min_ok < -15:
        return "struggling"
    else:
        return "neutral"

transfers["adaptation_outcome"] = transfers.apply(adaptation_success, axis=1)


# ─────────────────────────────────────────
# 3-D. 스타일 매치 점수
# ─────────────────────────────────────────
print("  3-D: 스타일 매치 점수 계산...")

# 스타일 매치: 이전 팀과 신규 팀의 플레이 스타일 유사도
# attack_defense_ratio, home_strength, away_strength 기반 유클리드 거리
def style_match_score(row):
    """
    이전 팀과 신규 팀의 스타일 거리. 낮을수록 스타일이 비슷함.
    0~1 스케일로 정규화 (0=완전 다름, 1=완전 같음)
    """
    metrics = ["attack_defense_ratio", "home_strength", "away_strength"]
    diffs = []
    for m in metrics:
        prev_val = row.get(f"prev_team_{m}", np.nan)
        new_val  = row.get(f"new_team_{m}", np.nan)
        if pd.notna(prev_val) and pd.notna(new_val):
            diffs.append((prev_val - new_val) ** 2)
    if not diffs:
        return np.nan
    dist = np.sqrt(sum(diffs) / len(diffs))
    # 거리 → 유사도 (0~1 정규화, 거리 2.0 이상이면 0)
    score = max(0, 1 - dist / 2.0)
    return round(score, 4)

transfers["style_match_score"] = transfers.apply(style_match_score, axis=1)

# 팀 레벨 업그레이드 여부 (승점 기준)
transfers["team_level_upgrade"] = (
    transfers["new_team_points"] - transfers["prev_team_points"]
).round(0)
transfers["is_step_up"] = transfers["team_level_upgrade"] > 0
transfers["is_step_down"] = transfers["team_level_upgrade"] < 0


# ─────────────────────────────────────────
# 3-E. 최종 컬럼 정리 및 저장
# ─────────────────────────────────────────
print("  3-E: 최종 정리 및 저장...")

TRANSFER_COLS = [
    # 식별자
    "player", "prev_team", "team", "prev_season", "season",
    "season_int", "prev_season_int",
    # 선수 정보
    "pos_group", "age", "age_category", "market_value",
    # 이전 팀 스탯
    "prev_team_points", "prev_team_total_goals_for", "prev_team_total_goals_against",
    "prev_team_attack_defense_ratio", "prev_team_home_strength", "prev_team_away_strength",
    "prev_team_avg_squad_age", "prev_team_total_squad_market_value",
    # 신규 팀 스탯
    "new_team_points", "new_team_total_goals_for", "new_team_total_goals_against",
    "new_team_attack_defense_ratio", "new_team_home_strength", "new_team_away_strength",
    "new_team_avg_squad_age", "new_team_total_squad_market_value",
    # 선수 이전 시즌 성과
    "prev_war_rating", "prev_gls_p90", "prev_ast_p90", "prev_minutes_share",
    # 선수 이적 후 성과
    "new_war_rating", "new_gls_p90", "new_ast_p90", "new_minutes_share",
    # 적응 지표
    "adaptation_war_delta", "adaptation_minutes_share_delta", "adaptation_outcome",
    # 스타일 매치
    "style_match_score",
    # 팀 레벨
    "team_level_upgrade", "is_step_up", "is_step_down",
    # 이적 플래그
    "transfer_flag",
]

available_transfer_cols = [c for c in TRANSFER_COLS if c in transfers.columns]
scout_transfers = transfers[available_transfer_cols].copy()
scout_transfers.replace([np.inf, -np.inf], np.nan, inplace=True)

out_path_transfer = os.path.join(SCOUT_DIR, "scout_transfers.parquet")
scout_transfers.to_parquet(out_path_transfer, index=False)
scout_transfers.to_csv(
    out_path_transfer.replace(".parquet", ".csv"),
    index=False, encoding="utf-8-sig"
)

print(f"\n  → 저장 완료: {out_path_transfer}")
print(f"  Shape: {scout_transfers.shape}")
print("\n  [scout_transfers 요약 통계]")
transfer_summary_cols = [
    "adaptation_war_delta", "adaptation_minutes_share_delta",
    "style_match_score", "team_level_upgrade"
]
available_transfer_sum = [c for c in transfer_summary_cols if c in scout_transfers.columns]
print(scout_transfers[available_transfer_sum].describe().round(4).to_string())

print("\n  [적응 결과 분포]")
if "adaptation_outcome" in scout_transfers.columns:
    print(scout_transfers["adaptation_outcome"].value_counts())

print("\n  [스텝업/다운 분포]")
if "is_step_up" in scout_transfers.columns:
    print(scout_transfers["is_step_up"].value_counts())


###############################################################################
# ══════════════════════════════════════════════════════════════════════════════
# 최종 완료 메시지
# ══════════════════════════════════════════════════════════════════════════════
###############################################################################
print("\n" + "=" * 60)
print("전체 파이프라인 완료")
print("=" * 60)
print(f"  scout_player_profiles : {scout_player.shape[0]:,}행 × {scout_player.shape[1]}열")
print(f"  scout_team_profiles   : {tp.shape[0]:,}행 × {tp.shape[1]}열")
print(f"  scout_transfers       : {scout_transfers.shape[0]:,}행 × {scout_transfers.shape[1]}열")
print(f"\n  저장 위치: {SCOUT_DIR}")
print("=" * 60)

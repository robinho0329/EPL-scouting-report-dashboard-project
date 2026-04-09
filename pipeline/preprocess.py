"""
EPL 전처리 파이프라인
5개 에이전트 분석 결과를 바탕으로 전체 전처리를 수행합니다.

출력:
  - data/processed/match_results.parquet
  - data/processed/player_season_stats.parquet
  - data/processed/player_match_logs.parquet
  - config/team_name_mapping.json
"""

import os
import sys
import json
import re
import warnings
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ── 프로젝트 루트 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CONFIG_DIR = PROJECT_ROOT / "config"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# Step 1: 팀명 매핑 테이블
# ================================================================
def load_team_mapping() -> dict:
    """config/team_name_mapping.json 로드 → {variant: standard} 역매핑 생성"""
    mapping_path = CONFIG_DIR / "team_name_mapping.json"
    with open(mapping_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # 역매핑: variant → standard
    reverse = {}
    for standard, variants in raw.items():
        for v in variants:
            reverse[v] = standard
        reverse[standard] = standard  # 자기 자신도 매핑
    return reverse


def standardize_team(name: str, mapping: dict) -> str:
    """팀명을 표준명으로 변환. 매핑에 없으면 원본 반환."""
    if pd.isna(name):
        return name
    name = str(name).strip()
    return mapping.get(name, name)


# ================================================================
# Step 2: epl_final.csv 클리닝
# ================================================================
def clean_epl_final(team_map: dict) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Step 2: epl_final.csv 클리닝")
    print("=" * 60)

    path = PROJECT_ROOT / "epl_final.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"  원본 shape: {df.shape}")

    # MatchDate → datetime
    df["MatchDate"] = pd.to_datetime(df["MatchDate"], errors="coerce")

    # 2003/04, 2004/05 시즌 누락 플래그
    df["season_data_missing"] = df["Season"].isin(["2003/04", "2004/05"])
    missing_count = df["season_data_missing"].sum()
    print(f"  2003/04, 2004/05 시즌 누락 플래그: {missing_count}건")

    # SOT=0인데 골 있는 건 → own_goal_flag
    # HomeShotsOnTarget=0 이면서 FullTimeHomeGoals>0 이거나,
    # AwayShotsOnTarget=0 이면서 FullTimeAwayGoals>0
    home_og = (df["HomeShotsOnTarget"] == 0) & (df["FullTimeHomeGoals"] > 0)
    away_og = (df["AwayShotsOnTarget"] == 0) & (df["FullTimeAwayGoals"] > 0)
    df["own_goal_flag_home"] = home_og
    df["own_goal_flag_away"] = away_og
    df["own_goal_flag"] = home_og | away_og
    og_count = df["own_goal_flag"].sum()
    print(f"  SOT=0인데 골 있는 건 (own_goal_flag): {og_count}건")

    # 팀명은 이미 epl_final 기준이므로 별도 매핑 불필요
    # (HomeTeam, AwayTeam 이미 표준명)

    print(f"  클리닝 후 shape: {df.shape}")
    return df


# ================================================================
# Step 3: FBref squad_stats 클리닝
# ================================================================
def clean_fbref_squad_stats(team_map: dict) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Step 3: FBref squad_stats 클리닝")
    print("=" * 60)

    fbref_root = DATA_RAW / "fbref"
    frames = []
    file_count = 0

    for season_dir in sorted(fbref_root.iterdir()):
        if not season_dir.is_dir():
            continue
        for team_dir in sorted(season_dir.iterdir()):
            if not team_dir.is_dir():
                continue
            csv_path = team_dir / "squad_stats.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path, encoding="utf-8-sig")
                frames.append(df)
                file_count += 1

    print(f"  로드된 squad_stats CSV 수: {file_count}")
    df = pd.concat(frames, ignore_index=True)
    print(f"  concat 후 shape: {df.shape}")

    # player=="Player" 헤더 혼입 제거
    header_mask = df["player"] == "Player"
    header_count = header_mask.sum()
    df = df[~header_mask].copy()
    print(f"  헤더 혼입 제거: {header_count}행")

    # 숫자 컬럼 타입 변환
    int_cols = ["mp", "crdy", "crdr"]
    float_cols = ["age", "starts", "min", "90s", "gls", "ast", "g_a", "g_pk",
                  "pk", "pkatt", "gls_1", "ast_1", "g_a_1", "g_pk_1", "g_a_pk"]

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # mp=0 선수 NaN → 0 채우기
    mp0_mask = df["mp"] == 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df.loc[mp0_mask, numeric_cols] = df.loc[mp0_mask, numeric_cols].fillna(0)
    print(f"  mp=0 선수 NaN→0 채우기: {mp0_mask.sum()}건")

    # 이적 선수 처리: 같은 시즌 2팀 이상 → transfer_flag
    player_season_teams = df.groupby(["player", "season"])["team"].nunique()
    transfer_players = player_season_teams[player_season_teams > 1].reset_index()
    transfer_players.columns = ["player", "season", "team_count"]
    df = df.merge(
        transfer_players[["player", "season"]].assign(transfer_flag=True),
        on=["player", "season"],
        how="left",
    )
    df["transfer_flag"] = df["transfer_flag"].fillna(False)
    print(f"  이적 선수 (같은 시즌 2+팀): {transfer_players.shape[0]}건")

    # 팀명 표준화 (FBref squad_stats의 team 컬럼)
    df["team"] = df["team"].apply(lambda x: standardize_team(x, team_map))

    # Unicode NFC 정규화 (선수명)
    df["player"] = df["player"].apply(
        lambda x: unicodedata.normalize("NFC", str(x)) if pd.notna(x) else x
    )

    # matches 컬럼 제거 (불필요)
    if "matches" in df.columns:
        df.drop(columns=["matches"], inplace=True)

    print(f"  최종 shape: {df.shape}")
    return df


# ================================================================
# Step 4: FBref matchlogs 클리닝
# ================================================================
def clean_fbref_matchlogs(team_map: dict) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Step 4: FBref matchlogs 클리닝")
    print("=" * 60)

    fbref_root = DATA_RAW / "fbref"
    frames = []
    file_count = 0

    for season_dir in sorted(fbref_root.iterdir()):
        if not season_dir.is_dir():
            continue
        for team_dir in sorted(season_dir.iterdir()):
            if not team_dir.is_dir():
                continue
            ml_dir = team_dir / "matchlogs"
            if not ml_dir.is_dir():
                continue
            for csv_file in ml_dir.iterdir():
                if csv_file.suffix.lower() == ".csv":
                    try:
                        df = pd.read_csv(csv_file, encoding="utf-8-sig")
                        frames.append(df)
                        file_count += 1
                    except Exception as e:
                        print(f"  [WARN] 읽기 실패: {csv_file}: {e}")

    print(f"  로드된 matchlog CSV 수: {file_count}")
    df = pd.concat(frames, ignore_index=True)
    print(f"  concat 후 shape: {df.shape}")

    # date 컬럼 BOM 제거 + datetime 변환
    if "date" in df.columns:
        df["date"] = df["date"].astype(str).str.replace("\ufeff", "", regex=False)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # result 파싱: en-dash(U+2013) 등으로 구분
    def parse_result(val):
        if pd.isna(val):
            return pd.Series({"outcome": np.nan, "goals_for": np.nan, "goals_against": np.nan})
        val = str(val).strip()
        # 패턴: "W 2–1", "D 1–1", "L 0–3" 등
        # en-dash(–) 또는 hyphen(-) 사용 가능
        m = re.match(r"([WDL])\s+(\d+)\s*[\u2013\-]\s*(\d+)", val)
        if m:
            return pd.Series({
                "outcome": m.group(1),
                "goals_for": int(m.group(2)),
                "goals_against": int(m.group(3)),
            })
        return pd.Series({"outcome": np.nan, "goals_for": np.nan, "goals_against": np.nan})

    result_parsed = df["result"].apply(parse_result)
    df["outcome"] = result_parsed["outcome"]
    df["goals_for"] = result_parsed["goals_for"]
    df["goals_against"] = result_parsed["goals_against"]
    print(f"  result 파싱 완료 - outcome 분포: {df['outcome'].value_counts().to_dict()}")

    # start 컬럼: Y/Y* → True, N → False
    if "start" in df.columns:
        df["started"] = df["start"].astype(str).str.strip().str.startswith("Y")
        df["started"] = df["started"].where(df["start"].notna(), other=np.nan)
        df.drop(columns=["start"], inplace=True)

    # pkwon/pkcon 컬럼 제거
    drop_cols = [c for c in ["pkwon", "pkcon"] if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        print(f"  제거된 컬럼: {drop_cols}")

    # opponent 팀명 표준화
    df["opponent"] = df["opponent"].apply(lambda x: standardize_team(x, team_map))

    # squad/team 팀명 표준화
    if "squad" in df.columns:
        df["squad"] = df["squad"].apply(lambda x: standardize_team(x, team_map))
    if "team" in df.columns:
        df["team"] = df["team"].apply(lambda x: standardize_team(x, team_map))

    # Unicode NFC 정규화 (선수명)
    if "player" in df.columns:
        df["player"] = df["player"].apply(
            lambda x: unicodedata.normalize("NFC", str(x)) if pd.notna(x) else x
        )

    # 숫자 컬럼 타입 변환
    num_cols = ["min", "gls", "ast", "pk", "pkatt", "sh", "sot",
                "crdy", "crdr", "fls", "fld", "off", "crs", "tklw", "int", "og"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2000~2009 상세스탯 NaN은 그대로 유지
    # 시즌별 가용 여부 플래그
    detail_stat_cols = ["sh", "sot", "fls", "fld", "off", "crs", "tklw", "int"]
    available_cols = [c for c in detail_stat_cols if c in df.columns]
    if available_cols:
        df["detail_stats_available"] = ~df[available_cols].isna().all(axis=1)
        avail_pct = df["detail_stats_available"].mean() * 100
        print(f"  상세스탯 가용률: {avail_pct:.1f}%")

    # match_report 컬럼 제거 (불필요)
    if "match_report" in df.columns:
        df.drop(columns=["match_report"], inplace=True)

    print(f"  최종 shape: {df.shape}")
    return df


# ================================================================
# Step 5: Transfermarkt squad_values 클리닝
# ================================================================
def clean_transfermarkt(team_map: dict) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Step 5: Transfermarkt squad_values 클리닝")
    print("=" * 60)

    tm_root = DATA_RAW / "transfermarkt"
    frames = []
    file_count = 0

    for season_dir in sorted(tm_root.iterdir()):
        if not season_dir.is_dir():
            continue
        for team_dir in sorted(season_dir.iterdir()):
            if not team_dir.is_dir():
                continue
            csv_path = team_dir / "squad_values.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path, encoding="utf-8-sig")
                    frames.append(df)
                    file_count += 1
                except Exception as e:
                    print(f"  [WARN] 읽기 실패: {csv_path}: {e}")

    print(f"  로드된 TM squad_values CSV 수: {file_count}")
    df = pd.concat(frames, ignore_index=True)
    print(f"  concat 후 shape: {df.shape}")

    # contract_until 컬럼 삭제
    if "contract_until" in df.columns:
        df.drop(columns=["contract_until"], inplace=True)
        print("  contract_until 컬럼 삭제")

    # DOB: 있으면 datetime, 없으면 age로 역산
    if "dob" in df.columns:
        df["dob"] = pd.to_datetime(df["dob"], errors="coerce", dayfirst=True)

    # season에서 시작 연도 추출: "2000/01" → 2000
    def extract_season_start_year(season_str):
        if pd.isna(season_str):
            return np.nan
        m = re.match(r"(\d{4})", str(season_str))
        return int(m.group(1)) if m else np.nan

    df["_season_start_year"] = df["season"].apply(extract_season_start_year)

    # DOB 없는 경우 age로 birth_year 역산
    dob_missing = df["dob"].isna()
    df["birth_year"] = df["dob"].dt.year
    df.loc[dob_missing & df["age"].notna(), "birth_year"] = (
        df.loc[dob_missing & df["age"].notna(), "_season_start_year"]
        - df.loc[dob_missing & df["age"].notna(), "age"].astype(int)
    )
    dob_filled = dob_missing.sum()
    print(f"  DOB 없는 행: {dob_filled}건 → birth_year 역산 적용")

    df.drop(columns=["_season_start_year"], inplace=True)

    # height: "1,83m" → 183 (int, cm)
    def parse_height(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip()
        m = re.match(r"(\d+),(\d+)\s*m?", val)
        if m:
            return int(m.group(1)) * 100 + int(m.group(2))
        return np.nan

    if "height" in df.columns:
        df["height_cm"] = df["height"].apply(parse_height)
        df.drop(columns=["height"], inplace=True)
        valid_h = df["height_cm"].notna().sum()
        print(f"  height 변환 완료: {valid_h}건 유효")

    # 2000~2003 몸값=0 → no_value_data=True 플래그
    def season_to_start_year(s):
        if pd.isna(s):
            return np.nan
        m = re.match(r"(\d{4})", str(s))
        return int(m.group(1)) if m else np.nan

    start_years = df["season"].apply(season_to_start_year)
    early_season = start_years.between(2000, 2003, inclusive="both")
    zero_value = df["market_value"] == 0
    df["no_value_data"] = early_season & zero_value
    no_val_count = df["no_value_data"].sum()
    print(f"  2000~2003 몸값=0 플래그: {no_val_count}건")

    # TM 팀명 → 표준 팀명 변환
    df["team"] = df["team"].apply(lambda x: standardize_team(x, team_map))

    # Unicode NFC 정규화 (선수명)
    df["player"] = df["player"].apply(
        lambda x: unicodedata.normalize("NFC", str(x)) if pd.notna(x) else x
    )

    # market_value를 float으로 확실히
    df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce")

    print(f"  최종 shape: {df.shape}")
    return df


# ================================================================
# Step 6: 통합 테이블 생성
# ================================================================
def build_match_results(epl_df: pd.DataFrame) -> pd.DataFrame:
    """match_results.parquet: epl_final + 팀 시즌 집계"""
    print("\n" + "-" * 40)
    print("Step 6-1: match_results 생성")
    print("-" * 40)

    # 기본 경기 결과 테이블
    result = epl_df.copy()

    # 팀 시즌 집계 (홈/원정 합산)
    home_agg = (
        epl_df.groupby(["Season", "HomeTeam"])
        .agg(
            home_played=("FullTimeResult", "count"),
            home_wins=("FullTimeResult", lambda x: (x == "H").sum()),
            home_draws=("FullTimeResult", lambda x: (x == "D").sum()),
            home_losses=("FullTimeResult", lambda x: (x == "A").sum()),
            home_goals_for=("FullTimeHomeGoals", "sum"),
            home_goals_against=("FullTimeAwayGoals", "sum"),
        )
        .reset_index()
        .rename(columns={"HomeTeam": "team"})
    )

    away_agg = (
        epl_df.groupby(["Season", "AwayTeam"])
        .agg(
            away_played=("FullTimeResult", "count"),
            away_wins=("FullTimeResult", lambda x: (x == "A").sum()),
            away_draws=("FullTimeResult", lambda x: (x == "D").sum()),
            away_losses=("FullTimeResult", lambda x: (x == "H").sum()),
            away_goals_for=("FullTimeAwayGoals", "sum"),
            away_goals_against=("FullTimeHomeGoals", "sum"),
        )
        .reset_index()
        .rename(columns={"AwayTeam": "team"})
    )

    team_season = home_agg.merge(away_agg, on=["Season", "team"], how="outer")
    team_season["total_played"] = team_season["home_played"].fillna(0) + team_season["away_played"].fillna(0)
    team_season["total_wins"] = team_season["home_wins"].fillna(0) + team_season["away_wins"].fillna(0)
    team_season["total_draws"] = team_season["home_draws"].fillna(0) + team_season["away_draws"].fillna(0)
    team_season["total_losses"] = team_season["home_losses"].fillna(0) + team_season["away_losses"].fillna(0)
    team_season["total_goals_for"] = team_season["home_goals_for"].fillna(0) + team_season["away_goals_for"].fillna(0)
    team_season["total_goals_against"] = team_season["home_goals_against"].fillna(0) + team_season["away_goals_against"].fillna(0)
    team_season["goal_diff"] = team_season["total_goals_for"] - team_season["total_goals_against"]
    team_season["points"] = team_season["total_wins"] * 3 + team_season["total_draws"]

    # int 변환
    int_agg_cols = [c for c in team_season.columns if c not in ["Season", "team"]]
    for c in int_agg_cols:
        team_season[c] = team_season[c].astype(int)

    print(f"  match_results (경기): {result.shape}")
    print(f"  team_season_summary: {team_season.shape}")

    return result, team_season


def build_player_season_stats(
    fbref_squad: pd.DataFrame, tm_df: pd.DataFrame
) -> pd.DataFrame:
    """player_season_stats.parquet: FBref squad_stats + TM squad_values 매칭"""
    print("\n" + "-" * 40)
    print("Step 6-2: player_season_stats 생성")
    print("-" * 40)

    # 선수명 NFC 정규화는 이미 적용됨
    # exact match 우선: player + season + team
    merged = fbref_squad.merge(
        tm_df,
        on=["player", "season", "team"],
        how="left",
        suffixes=("", "_tm"),
    )

    # 매칭 결과
    matched = merged["player_id"].notna().sum()
    total = len(merged)
    print(f"  exact match (player+season+team): {matched}/{total} ({matched/total*100:.1f}%)")

    # 못 매칭된 건은 player + season 만으로 재시도 (팀이 다를 수 있음 - 이적 등)
    unmatched_mask = merged["player_id"].isna()
    if unmatched_mask.sum() > 0:
        unmatched = merged.loc[unmatched_mask, fbref_squad.columns].copy()
        merged = merged.loc[~unmatched_mask].copy()

        # player + season 매칭
        rematched = unmatched.merge(
            tm_df,
            on=["player", "season"],
            how="left",
            suffixes=("", "_tm"),
        )
        # team_tm가 있을 경우 제거 (team은 fbref 기준 유지)
        if "team_tm" in rematched.columns:
            rematched.drop(columns=["team_tm"], inplace=True)

        merged = pd.concat([merged, rematched], ignore_index=True)
        rematched_count = rematched["player_id"].notna().sum()
        print(f"  player+season 재매칭: {rematched_count}건 추가")

    final_matched = merged["player_id"].notna().sum()
    print(f"  최종 매칭: {final_matched}/{len(merged)} ({final_matched/len(merged)*100:.1f}%)")
    print(f"  player_season_stats shape: {merged.shape}")

    return merged


def build_player_match_logs(matchlog_df: pd.DataFrame) -> pd.DataFrame:
    """player_match_logs.parquet: 클리닝된 matchlog"""
    print("\n" + "-" * 40)
    print("Step 6-3: player_match_logs 생성")
    print("-" * 40)
    print(f"  player_match_logs shape: {matchlog_df.shape}")
    return matchlog_df


# ================================================================
# 결측치 요약 출력
# ================================================================
def print_summary(name: str, df: pd.DataFrame):
    print(f"\n{'─' * 50}")
    print(f"[{name}] shape: {df.shape}")
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(1)
    summary = pd.DataFrame({"null_count": null_counts, "null_pct": null_pct})
    summary = summary[summary["null_count"] > 0].sort_values("null_count", ascending=False)
    if len(summary) > 0:
        print(f"  결측치 있는 컬럼 ({len(summary)}개):")
        for col, row in summary.head(15).iterrows():
            print(f"    {col:30s} : {int(row['null_count']):>8,}  ({row['null_pct']:.1f}%)")
        if len(summary) > 15:
            print(f"    ... 외 {len(summary) - 15}개 컬럼")
    else:
        print("  결측치 없음")
    print(f"  dtypes: {dict(df.dtypes.value_counts())}")


# ================================================================
# 메인 실행
# ================================================================
def main():
    print("=" * 60)
    print("EPL 전처리 파이프라인 시작")
    print("=" * 60)

    # Step 1: 팀명 매핑
    print("\nStep 1: 팀명 매핑 테이블 로드")
    team_map = load_team_mapping()
    print(f"  매핑 항목 수: {len(team_map)} (variant → standard)")

    # Step 2: epl_final
    epl_df = clean_epl_final(team_map)

    # Step 3: FBref squad_stats
    fbref_squad = clean_fbref_squad_stats(team_map)

    # Step 4: FBref matchlogs
    matchlog_df = clean_fbref_matchlogs(team_map)

    # Step 5: Transfermarkt
    tm_df = clean_transfermarkt(team_map)

    # Step 6: 통합 테이블 생성
    print("\n" + "=" * 60)
    print("Step 6: 통합 테이블 생성")
    print("=" * 60)

    # 6-1: match_results
    match_results, team_season = build_match_results(epl_df)

    # 6-2: player_season_stats
    player_season = build_player_season_stats(fbref_squad, tm_df)

    # 6-3: player_match_logs
    player_matchlogs = build_player_match_logs(matchlog_df)

    # ── 저장 ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("저장 중...")
    print("=" * 60)

    # match_results: 경기별 + 팀시즌 집계를 하나의 parquet에 두기보다,
    # 경기별 결과를 메인으로 저장하고, 팀시즌 집계는 별도 저장
    match_results.to_parquet(DATA_PROCESSED / "match_results.parquet", index=False)
    team_season.to_parquet(DATA_PROCESSED / "team_season_summary.parquet", index=False)
    print(f"  match_results.parquet: {match_results.shape}")
    print(f"  team_season_summary.parquet: {team_season.shape}")

    player_season.to_parquet(DATA_PROCESSED / "player_season_stats.parquet", index=False)
    print(f"  player_season_stats.parquet: {player_season.shape}")

    player_matchlogs.to_parquet(DATA_PROCESSED / "player_match_logs.parquet", index=False)
    print(f"  player_match_logs.parquet: {player_matchlogs.shape}")

    # ── 최종 요약 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("최종 요약")
    print("=" * 60)
    print_summary("match_results", match_results)
    print_summary("team_season_summary", team_season)
    print_summary("player_season_stats", player_season)
    print_summary("player_match_logs", player_matchlogs)

    print("\n" + "=" * 60)
    print("전처리 파이프라인 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()

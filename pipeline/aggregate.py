"""Data aggregation pipeline.

Reads raw crawled CSV files and produces:
1. player_match_stats.parquet  - All match-level player stats (unified)
2. player_season_stats.parquet - Season-level aggregation per player
3. player_alltime_stats.parquet - All-time career aggregation per player
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import logging

from config.settings import FBREF_RAW_DIR, PROCESSED_DIR, DASHBOARD_DIR
from crawlers.utils.name_normalizer import normalize_team_name, normalize_player_name

logger = logging.getLogger(__name__)

# Columns we expect to aggregate (FBref column names after cleaning)
NUMERIC_COLS = [
    "mp", "starts", "min", "90s",
    "gls", "ast", "g_a", "g_pk", "pk", "pkatt",
    "crdy", "crdr",
    "xg", "npxg", "xag",
    "prgc", "prgp", "prgr",
    "gls_1", "ast_1", "g_a_1",  # per-90 duplicates
]


def load_squad_stats() -> pd.DataFrame:
    """Load all squad_stats.csv files into a single DataFrame."""
    files = list(FBREF_RAW_DIR.rglob("squad_stats.csv"))
    if not files:
        logger.warning("No squad_stats.csv files found")
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error reading {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined)} player-season rows from {len(files)} files")
    return combined


def load_match_logs() -> pd.DataFrame:
    """Load all individual match log CSVs into a single DataFrame."""
    files = list(FBREF_RAW_DIR.rglob("matchlogs/*.csv"))
    if not files:
        logger.warning("No match log files found")
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error reading {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined)} match-log rows from {len(files)} files")
    return combined


def clean_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize a DataFrame of player stats."""
    if df.empty:
        return df

    # Normalize team names
    if "team" in df.columns:
        df["team"] = df["team"].apply(normalize_team_name)

    # Normalize player names (keep original too)
    if "player" in df.columns:
        df["player_normalized"] = df["player"].apply(normalize_player_name)

    # Convert numeric columns (handle non-numeric gracefully)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df


def build_season_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to player-season level."""
    if df.empty:
        return df

    group_cols = ["player", "player_normalized", "season", "team"]
    existing_groups = [c for c in group_cols if c in df.columns]

    agg_cols = {c: "sum" for c in NUMERIC_COLS if c in df.columns}

    # Add position (most common)
    if "pos" in df.columns or "position" in df.columns:
        pos_col = "pos" if "pos" in df.columns else "position"
        agg_cols[pos_col] = lambda x: x.mode().iloc[0] if not x.mode().empty else ""

    season_stats = df.groupby(existing_groups, as_index=False).agg(agg_cols)
    return season_stats


def build_alltime_stats(season_stats: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to all-time career level."""
    if season_stats.empty:
        return season_stats

    group_cols = ["player", "player_normalized"]
    existing_groups = [c for c in group_cols if c in season_stats.columns]

    if not existing_groups:
        # player 컬럼이라도 있으면 사용
        if "player" in season_stats.columns:
            existing_groups = ["player"]
        else:
            logger.warning("No player column found for alltime stats")
            return pd.DataFrame()

    agg_cols = {c: "sum" for c in NUMERIC_COLS if c in season_stats.columns}

    # Number of seasons
    if "season" in season_stats.columns:
        agg_cols["season"] = "count"

    alltime = season_stats.groupby(existing_groups, as_index=False).agg(agg_cols)

    if "season" in alltime.columns:
        alltime = alltime.rename(columns={"season": "num_seasons"})

    # Sort by goals descending
    if "goals" in alltime.columns:
        alltime = alltime.sort_values("goals", ascending=False)

    return alltime


def run_pipeline():
    """Execute the full aggregation pipeline."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading squad stats...")
    squad_df = load_squad_stats()
    squad_df = clean_and_normalize(squad_df)

    print("Loading match logs...")
    matchlog_df = load_match_logs()
    matchlog_df = clean_and_normalize(matchlog_df)

    # Save match-level data
    if not matchlog_df.empty:
        # 중복 컬럼 제거 + mixed type 처리
        matchlog_df = matchlog_df.loc[:, ~matchlog_df.columns.duplicated()]
        for col in matchlog_df.columns:
            if matchlog_df[col].dtype == object:
                matchlog_df[col] = matchlog_df[col].astype(str)
        matchlog_df.to_parquet(PROCESSED_DIR / "player_match_stats.parquet", index=False)
        print(f"Saved player_match_stats.parquet: {len(matchlog_df)} rows")

    # Build season stats (from squad stats if available, else from match logs)
    if not squad_df.empty:
        print("Building season stats from squad data...")
        season_stats = squad_df
    elif not matchlog_df.empty:
        print("Building season stats from match logs...")
        season_stats = build_season_stats(matchlog_df)
    else:
        print("No data available to aggregate.")
        return

    # "vs " prefix 팀 데이터 제거
    if "team" in season_stats.columns:
        before = len(season_stats)
        season_stats = season_stats[~season_stats["team"].str.startswith("vs ", na=False)]
        removed = before - len(season_stats)
        if removed > 0:
            print(f"Removed {removed} 'vs' opponent rows")

    # 중복 컬럼 제거
    season_stats = season_stats.loc[:, ~season_stats.columns.duplicated()]

    # 안전하게 parquet 저장 (mixed type 컬럼 처리)
    for col in season_stats.columns:
        if season_stats[col].dtype == object:
            season_stats[col] = season_stats[col].astype(str)

    season_stats.to_parquet(DASHBOARD_DIR / "player_season_stats.parquet", index=False)
    print(f"Saved player_season_stats.parquet: {len(season_stats)} rows")

    # Build all-time stats
    print("Building all-time stats...")
    alltime_stats = build_alltime_stats(season_stats)

    # 중복 컬럼 제거 및 타입 처리
    alltime_stats = alltime_stats.loc[:, ~alltime_stats.columns.duplicated()]
    for col in alltime_stats.columns:
        if alltime_stats[col].dtype == object:
            alltime_stats[col] = alltime_stats[col].astype(str)

    alltime_stats.to_parquet(DASHBOARD_DIR / "player_alltime_stats.parquet", index=False)
    print(f"Saved player_alltime_stats.parquet: {len(alltime_stats)} rows")

    print("\nPipeline complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pipeline()

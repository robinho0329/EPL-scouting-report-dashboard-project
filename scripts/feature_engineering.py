"""
EPL Feature Engineering Pipeline
==================================
Reads processed parquet files and produces enriched feature datasets.

Outputs:
  - data/features/match_features.parquet     (match-level, rolling team features)
  - data/features/player_features.parquet    (player-season derived stats)
  - data/features/feature_summary.json       (metadata / basic stats for all features)

Design principles:
  - Strict temporal ordering – no future data leakage.
  - Handles the 2000-2012 limited-stats era gracefully.
  - NaN for early matches where rolling windows are insufficient.

Data split reference:
  Train 2000-2021  |  Val 2021-2023  |  Test 2023-2025
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("C:/Users/xcv54/workspace/EPL project")
DATA_PROC = BASE / "data" / "processed"
DATA_FEAT = BASE / "data" / "features"
DATA_FEAT.mkdir(parents=True, exist_ok=True)

MATCH_FILE      = DATA_PROC / "match_results.parquet"
TEAM_SUMM_FILE  = DATA_PROC / "team_season_summary.parquet"
PLAYER_SEAS_FILE= DATA_PROC / "player_season_stats.parquet"
PLAYER_LOG_FILE = DATA_PROC / "player_match_logs.parquet"

OUT_MATCH       = DATA_FEAT / "match_features.parquet"
OUT_PLAYER      = DATA_FEAT / "player_features.parquet"
OUT_SUMMARY     = DATA_FEAT / "feature_summary.json"

# ---------------------------------------------------------------------------
# Known EPL rivalries (derby pairs – unordered)
# ---------------------------------------------------------------------------
DERBY_PAIRS = {
    frozenset({"Man United", "Man City"}),
    frozenset({"Man United", "Liverpool"}),
    frozenset({"Arsenal",    "Tottenham"}),
    frozenset({"Chelsea",    "Arsenal"}),
    frozenset({"Liverpool",  "Everton"}),
    frozenset({"Aston Villa","Birmingham"}),
    frozenset({"Newcastle",  "Sunderland"}),
    frozenset({"West Ham",   "Millwall"}),
    frozenset({"Chelsea",    "Tottenham"}),
    frozenset({"Leeds",      "Man United"}),
    frozenset({"Wolves",     "Aston Villa"}),
    frozenset({"Norwich",    "Ipswich"}),
    frozenset({"Sheffield United", "Leeds"}),
}

# ---------------------------------------------------------------------------
# Season order helper
# ---------------------------------------------------------------------------
def season_sort_key(s: str) -> int:
    """Convert '2000/01' -> 2000 for ordering."""
    return int(str(s).split("/")[0])


# ===========================================================================
#  SECTION 1 – MATCH-LEVEL FEATURES
# ===========================================================================

def load_match_data() -> pd.DataFrame:
    print("\n[1/3] Loading match_results.parquet ...")
    df = pd.read_parquet(MATCH_FILE)
    df = df.sort_values(["MatchDate", "HomeTeam"]).reset_index(drop=True)
    df["season_year"] = df["Season"].apply(season_sort_key)
    print(f"    {len(df):,} matches  |  {df['Season'].nunique()} seasons  |  "
          f"{df['HomeTeam'].nunique()} teams")
    return df


# ---------------------------------------------------------------------------
# ELO rating system
# ---------------------------------------------------------------------------
def compute_elo(df: pd.DataFrame, K: int = 20, initial: int = 1500) -> pd.DataFrame:
    """
    Walk matches in strict chronological order and compute pre-match ELO ratings
    for every team.  Returns df with new columns:
        home_elo_pre, away_elo_pre, home_elo_post, away_elo_post, elo_diff
    """
    print("    Computing ELO ratings (K={}, initial={}) ...".format(K, initial))
    from collections import defaultdict
    elo: dict = defaultdict(lambda: float(initial))

    home_pre, away_pre = [], []

    for _, row in df.iterrows():
        ht, at = row["HomeTeam"], row["AwayTeam"]
        he, ae = elo[ht], elo[at]
        home_pre.append(he)
        away_pre.append(ae)

        # Expected scores
        exp_h = 1.0 / (1.0 + 10 ** ((ae - he) / 400.0))
        exp_a = 1.0 - exp_h

        # Actual scores
        res = row["FullTimeResult"]
        if res == "H":
            act_h, act_a = 1.0, 0.0
        elif res == "A":
            act_h, act_a = 0.0, 1.0
        else:
            act_h, act_a = 0.5, 0.5

        # Goal-difference multiplier (enhances sensitivity to large wins)
        gd = abs(row["FullTimeHomeGoals"] - row["FullTimeAwayGoals"])
        if gd <= 1:
            gd_mult = 1.0
        elif gd == 2:
            gd_mult = 1.5
        else:
            gd_mult = (11.0 + gd) / 8.0

        elo[ht] = he + K * gd_mult * (act_h - exp_h)
        elo[at] = ae + K * gd_mult * (act_a - exp_a)

    df = df.copy()
    df["home_elo_pre"]  = home_pre
    df["away_elo_pre"]  = away_pre
    df["elo_diff"]      = df["home_elo_pre"] - df["away_elo_pre"]
    return df


# ---------------------------------------------------------------------------
# Rolling team-level stats helper
# ---------------------------------------------------------------------------
def rolling_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each team, compute rolling statistics using only past matches.
    Adds many *_home and *_away columns to df (indexed on match row).
    No data leakage: shift(1) ensures the current match is excluded.
    """
    print("    Building rolling team statistics ...")

    # ------------------------------------------------------------------
    # Step 1: create a long-format 'team match' frame (one row per team
    #         per match) so that rolling is straightforward.
    # ------------------------------------------------------------------
    home_df = df[["MatchDate", "Season", "HomeTeam", "AwayTeam",
                  "FullTimeHomeGoals", "FullTimeAwayGoals",
                  "FullTimeResult",
                  "HomeShotsOnTarget", "AwayShotsOnTarget",
                  "HomeShots", "AwayShots"]].copy()
    home_df.rename(columns={
        "HomeTeam":             "team",
        "AwayTeam":             "opponent",
        "FullTimeHomeGoals":    "gf",
        "FullTimeAwayGoals":    "gc",
        "HomeShotsOnTarget":    "sot_for",
        "AwayShotsOnTarget":    "sot_against",
        "HomeShots":            "shots_for",
        "AwayShots":            "shots_against",
    }, inplace=True)
    home_df["is_home"] = True
    home_df["result_code"] = home_df["FullTimeResult"].map({"H": "W", "A": "L", "D": "D"})

    away_df = df[["MatchDate", "Season", "AwayTeam", "HomeTeam",
                  "FullTimeHomeGoals", "FullTimeAwayGoals",
                  "FullTimeResult",
                  "HomeShotsOnTarget", "AwayShotsOnTarget",
                  "HomeShots", "AwayShots"]].copy()
    away_df.rename(columns={
        "AwayTeam":             "team",
        "HomeTeam":             "opponent",
        "FullTimeAwayGoals":    "gf",
        "FullTimeHomeGoals":    "gc",
        "AwayShotsOnTarget":    "sot_for",
        "HomeShotsOnTarget":    "sot_against",
        "AwayShots":            "shots_for",
        "HomeShots":            "shots_against",
    }, inplace=True)
    away_df["is_home"] = False
    away_df["result_code"] = away_df["FullTimeResult"].map({"A": "W", "H": "L", "D": "D"})

    long = pd.concat([home_df, away_df], ignore_index=True)
    long = long.sort_values(["team", "MatchDate"]).reset_index(drop=True)

    # Numeric encodings
    long["pts"]     = long["result_code"].map({"W": 3, "D": 1, "L": 0})
    long["win"]     = (long["result_code"] == "W").astype(int)
    long["clean"]   = (long["gc"] == 0).astype(int)
    long["gd"]      = long["gf"] - long["gc"]

    # Shots on target – treat 0 as missing for early era (season_data_missing)
    # We rely on the season_data_missing flag from match_results rather than
    # checking zeros, so we just leave them as-is; NaN propagation handles gaps.

    # ------------------------------------------------------------------
    # Step 2: compute rolling features per team (using .shift(1) to
    #         exclude the current match from its own window).
    # ------------------------------------------------------------------
    def shifted_rolling(series: pd.Series, window: int, func="mean"):
        """Exclude current row, roll over previous `window` rows."""
        s = series.shift(1)
        if func == "mean":
            return s.rolling(window, min_periods=1).mean()
        elif func == "sum":
            return s.rolling(window, min_periods=1).sum()
        elif func == "std":
            return s.rolling(window, min_periods=2).std()

    records = []
    for team, grp in long.groupby("team", sort=False):
        grp = grp.sort_values("MatchDate").reset_index(drop=True)

        r = grp[["MatchDate", "team", "is_home"]].copy()

        # Form: rolling points avg (W=3, D=1, L=0)
        r["form_5"]  = shifted_rolling(grp["pts"],   5)
        r["form_10"] = shifted_rolling(grp["pts"],  10)

        # Win rate rolling
        r["win_rate_5"] = shifted_rolling(grp["win"], 5)

        # Goals scored / conceded
        for w in [3, 5, 10]:
            r[f"goals_scored_{w}"]   = shifted_rolling(grp["gf"],    w)
            r[f"goals_conceded_{w}"] = shifted_rolling(grp["gc"],    w)
            r[f"gd_rolling_{w}"]     = shifted_rolling(grp["gd"],    w)

        # Clean sheet ratio rolling 5
        r["clean_sheet_5"] = shifted_rolling(grp["clean"], 5)

        # Shots on target rolling 5 (NaN where unavailable)
        r["sot_rolling_5"] = shifted_rolling(grp["sot_for"], 5)

        # Days since last match
        r["days_rest"] = grp["MatchDate"].diff().dt.days  # NaN for first match

        # Home / Away form separately (rolling 5 within that venue type)
        home_grp = grp[grp["is_home"] == True].sort_values("MatchDate")
        away_grp = grp[grp["is_home"] == False].sort_values("MatchDate")

        home_form = home_grp["pts"].shift(1).rolling(5, min_periods=1).mean()
        home_form.index = home_grp.index
        away_form = away_grp["pts"].shift(1).rolling(5, min_periods=1).mean()
        away_form.index = away_grp.index

        venue_form = pd.concat([home_form, away_form]).sort_index()
        r["venue_form_5"] = venue_form.values  # aligned because we kept original index order

        # Season momentum: rolling-5 pts vs season avg pts (up to that match)
        grp["season_pts_cumavg"] = (
            grp.groupby("Season")["pts"]
               .apply(lambda s: s.shift(1).expanding().mean())
               .reset_index(level=0, drop=True)
        )
        r["momentum_5"] = r["form_5"] - grp["season_pts_cumavg"]

        records.append(r)

    rolling_long = pd.concat(records, ignore_index=True)

    # ------------------------------------------------------------------
    # Step 3: merge back to the original match frame (home + away sides)
    # ------------------------------------------------------------------
    home_roll = rolling_long[rolling_long["is_home"] == True].drop(columns="is_home")
    away_roll = rolling_long[rolling_long["is_home"] == False].drop(columns="is_home")

    roll_cols = [c for c in home_roll.columns if c not in ("MatchDate", "team")]

    home_roll = home_roll.rename(columns={c: f"home_{c}" for c in roll_cols})
    home_roll = home_roll.rename(columns={"team": "HomeTeam"})

    away_roll = away_roll.rename(columns={c: f"away_{c}" for c in roll_cols})
    away_roll = away_roll.rename(columns={"team": "AwayTeam"})

    df = df.merge(home_roll, on=["MatchDate", "HomeTeam"], how="left")
    df = df.merge(away_roll, on=["MatchDate", "AwayTeam"], how="left")

    return df


# ---------------------------------------------------------------------------
# Head-to-head record
# ---------------------------------------------------------------------------
def compute_h2h(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each match, compute results of the last 5 H2H meetings (strict past).
    Adds: h2h_home_wins, h2h_away_wins, h2h_draws (from home team perspective)
    """
    print("    Computing head-to-head records (last 5 meetings) ...")
    df = df.copy()
    df["h2h_home_wins"] = np.nan
    df["h2h_away_wins"] = np.nan
    df["h2h_draws"]     = np.nan

    # group by sorted team pair
    df["_pair"] = df.apply(
        lambda r: tuple(sorted([r["HomeTeam"], r["AwayTeam"]])), axis=1
    )
    df = df.sort_values("MatchDate").reset_index(drop=True)

    h2h_history: dict = {}   # pair -> list of (date, home_team, result)

    for idx, row in df.iterrows():
        ht, at = row["HomeTeam"], row["AwayTeam"]
        pair = row["_pair"]

        past = h2h_history.get(pair, [])
        last5 = past[-5:]

        if last5:
            hw = sum(1 for _, home, res in last5 if home == ht and res == "H") + \
                 sum(1 for _, home, res in last5 if home == at and res == "A")
            aw = sum(1 for _, home, res in last5 if home == at and res == "H") + \
                 sum(1 for _, home, res in last5 if home == ht and res == "A")
            dd = sum(1 for _, _, res in last5 if res == "D")
            df.at[idx, "h2h_home_wins"] = hw
            df.at[idx, "h2h_away_wins"] = aw
            df.at[idx, "h2h_draws"]     = dd
        # else: leave NaN (no prior meetings in dataset)

        h2h_history.setdefault(pair, []).append(
            (row["MatchDate"], ht, row["FullTimeResult"])
        )

    df.drop(columns="_pair", inplace=True)
    return df


# ---------------------------------------------------------------------------
# Promoted team detection
# ---------------------------------------------------------------------------
def compute_promoted_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    A team is 'promoted' if it did not appear in the previous season's matches.
    """
    print("    Computing promoted team flags ...")
    df = df.copy()
    seasons_sorted = sorted(df["Season"].unique(), key=season_sort_key)
    prev_teams: dict = {}
    all_prev: set = set()

    for i, s in enumerate(seasons_sorted):
        if i == 0:
            prev_teams[s] = set()
        else:
            prev_s = seasons_sorted[i - 1]
            mask = df["Season"] == prev_s
            prev_teams[s] = set(df[mask]["HomeTeam"]) | set(df[mask]["AwayTeam"])

    df["home_promoted"] = df.apply(
        lambda r: r["HomeTeam"] not in prev_teams.get(r["Season"], set()), axis=1
    ).astype(int)
    df["away_promoted"] = df.apply(
        lambda r: r["AwayTeam"] not in prev_teams.get(r["Season"], set()), axis=1
    ).astype(int)
    return df


# ---------------------------------------------------------------------------
# Season stage
# ---------------------------------------------------------------------------
def compute_season_stage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tag each match with its matchweek within the season, then bucket:
      early (1-12), mid (13-25), late (26-38)
    """
    print("    Computing season stage ...")
    df = df.copy()

    def _stage(mw):
        if mw <= 12:
            return "early"
        elif mw <= 25:
            return "mid"
        else:
            return "late"

    # Rank matches within each season by date → matchweek proxy
    df["matchweek"] = (
        df.groupby("Season")["MatchDate"]
          .rank(method="dense")
          .astype(int)
    )
    df["season_stage"] = df["matchweek"].apply(_stage)
    return df


# ---------------------------------------------------------------------------
# Derive match-level composite features
# ---------------------------------------------------------------------------
def compute_match_derived(df: pd.DataFrame) -> pd.DataFrame:
    print("    Computing match-level derived features ...")
    df = df.copy()

    # Pre-match ELO difference already computed; add form difference
    df["form_diff_5"]  = df["home_form_5"]  - df["away_form_5"]
    df["form_diff_10"] = df["home_form_10"] - df["away_form_10"]
    df["gd_trend_diff"] = df["home_gd_rolling_5"] - df["away_gd_rolling_5"]

    # Is derby
    df["is_derby"] = df.apply(
        lambda r: int(frozenset({r["HomeTeam"], r["AwayTeam"]}) in DERBY_PAIRS), axis=1
    )

    # Is weekend (Saturday=5 or Sunday=6)
    df["is_weekend"] = df["MatchDate"].dt.dayofweek.isin([5, 6]).astype(int)

    # Split label
    def _split(season):
        yr = season_sort_key(season)
        if yr <= 2020:
            return "train"
        elif yr <= 2022:
            return "val"
        else:
            return "test"

    df["data_split"] = df["Season"].apply(_split)

    return df


# ---------------------------------------------------------------------------
# Master match feature builder
# ---------------------------------------------------------------------------
def build_match_features() -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("BUILDING MATCH FEATURES")
    print("=" * 60)

    df = load_match_data()
    df = compute_elo(df)
    df = rolling_team_stats(df)
    df = compute_h2h(df)
    df = compute_promoted_flag(df)
    df = compute_season_stage(df)
    df = compute_match_derived(df)

    print(f"\n    Final match feature frame: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


# ===========================================================================
#  SECTION 2 – PLAYER-LEVEL FEATURES
# ===========================================================================

def load_player_data():
    print("\n[2/3] Loading player data ...")
    ps  = pd.read_parquet(PLAYER_SEAS_FILE)
    pml = pd.read_parquet(PLAYER_LOG_FILE)
    print(f"    player_season_stats: {ps.shape[0]:,} rows x {ps.shape[1]} cols")
    print(f"    player_match_logs:   {pml.shape[0]:,} rows x {pml.shape[1]} cols")
    return ps, pml


def compute_per90(ps: pd.DataFrame) -> pd.DataFrame:
    """Per-90-minute stats."""
    print("    Computing per-90 stats ...")
    ps = ps.copy()

    # Guard: avoid division by zero / NaN
    ninety = ps["90s"].replace(0, np.nan)

    stat_map = {
        "gls":   "goals_p90",
        "ast":   "assists_p90",
        "g_a":   "goal_contributions_p90",
        "crdy":  "yellow_cards_p90",
        "crdr":  "red_cards_p90",
        "pk":    "penalties_p90",
    }
    for src, dst in stat_map.items():
        if src in ps.columns:
            ps[dst] = ps[src] / ninety

    return ps


def compute_player_per90_from_logs(ps: pd.DataFrame, pml: pd.DataFrame) -> pd.DataFrame:
    """
    For seasons with detail_stats_available, aggregate per-90 stats from logs
    (shots, shots on target, tackles won, interceptions).
    """
    print("    Computing detail per-90 stats from match logs ...")
    detail = pml[pml["detail_stats_available"] == True].copy()

    agg = (
        detail.groupby(["player", "season"])
              .agg(
                  log_shots=("sh", "sum"),
                  log_sot=("sot", "sum"),
                  log_tklw=("tklw", "sum"),
                  log_interceptions=("int", "sum"),
                  log_fouls_committed=("fls", "sum"),
                  log_crosses=("crs", "sum"),
                  log_offside=("off", "sum"),
                  log_minutes=("min", "sum"),
              )
              .reset_index()
    )
    agg["log_ninety"] = agg["log_minutes"].replace(0, np.nan) / 90.0
    agg["shots_p90"]         = agg["log_shots"]         / agg["log_ninety"]
    agg["sot_p90"]           = agg["log_sot"]           / agg["log_ninety"]
    agg["tackles_p90"]       = agg["log_tklw"]          / agg["log_ninety"]
    agg["interceptions_p90"] = agg["log_interceptions"] / agg["log_ninety"]
    agg["fouls_p90"]         = agg["log_fouls_committed"] / agg["log_ninety"]

    ps = ps.merge(
        agg[["player", "season",
             "shots_p90", "sot_p90", "tackles_p90",
             "interceptions_p90", "fouls_p90",
             "log_minutes"]],
        on=["player", "season"],
        how="left"
    )
    return ps


def compute_goal_contribution_rate(ps: pd.DataFrame) -> pd.DataFrame:
    ps = ps.copy()
    mp = ps["mp"].replace(0, np.nan)
    ps["goal_contribution_rate"] = (
        (ps["gls"].fillna(0) + ps["ast"].fillna(0)) / mp
    )
    return ps


def compute_minutes_share(ps: pd.DataFrame) -> pd.DataFrame:
    """
    Minutes share = player minutes / (team_total_matches * 90).
    team_total_matches taken from season_summary aggregation.
    """
    print("    Computing minutes share ...")
    ps = ps.copy()

    team_mp = (
        ps.groupby(["team", "season"])["mp"]
          .max()   # proxy: max matches played by any player = team matches
          .reset_index()
          .rename(columns={"mp": "team_max_mp"})
    )
    ps = ps.merge(team_mp, on=["team", "season"], how="left")
    ps["minutes_share"] = ps["min"] / (ps["team_max_mp"].replace(0, np.nan) * 90.0)
    return ps


def compute_consistency_score(pml: pd.DataFrame) -> pd.DataFrame:
    """
    Consistency score based on std-dev of per-match goal contributions.
    Lower std = more consistent.  Uses match logs.
    """
    print("    Computing consistency scores from match logs ...")
    pml = pml.copy()
    pml["match_gc"] = pml["gls"].fillna(0) + pml["ast"].fillna(0)

    cons = (
        pml.groupby(["player", "season"])["match_gc"]
           .agg(consistency_std="std", consistency_mean="mean", n_matches="count")
           .reset_index()
    )
    # Coefficient of variation (relative consistency): lower is more consistent
    cons["consistency_cv"] = (
        cons["consistency_std"] / cons["consistency_mean"].replace(0, np.nan)
    )
    return cons[["player", "season", "consistency_std",
                 "consistency_mean", "consistency_cv", "n_matches"]]


def compute_age_bracket(ps: pd.DataFrame) -> pd.DataFrame:
    ps = ps.copy()

    def _bracket(age):
        if pd.isna(age):
            return "unknown"
        elif age < 21:
            return "youth"
        elif age <= 29:
            return "prime"
        else:
            return "veteran"

    age_col = ps["age_tm"].fillna(ps["age"])
    ps["age_bracket"] = age_col.apply(_bracket)
    ps["age_used"]    = age_col
    return ps


def compute_market_value_features(ps: pd.DataFrame) -> pd.DataFrame:
    """
    Season-over-season market value % change and value per goal contribution.
    Market value available from 2004/05 onward.
    """
    print("    Computing market value features ...")
    ps = ps.copy()

    # Sort by player and season chronologically
    ps["_season_yr"] = ps["season"].apply(season_sort_key)
    ps = ps.sort_values(["player", "_season_yr"]).reset_index(drop=True)

    # MV change YoY (within same player career)
    ps["mv_prev"] = ps.groupby("player")["market_value"].shift(1)
    ps["mv_change_pct"] = np.where(
        (ps["mv_prev"].notna()) & (ps["mv_prev"] > 0),
        (ps["market_value"] - ps["mv_prev"]) / ps["mv_prev"] * 100.0,
        np.nan,
    )

    # Market value per goal contribution
    gc_total = ps["gls"].fillna(0) + ps["ast"].fillna(0)
    ps["mv_per_goal_contribution"] = np.where(
        gc_total > 0,
        ps["market_value"] / gc_total,
        np.nan,
    )

    ps.drop(columns="_season_yr", inplace=True)
    return ps


def compute_experience(ps: pd.DataFrame) -> pd.DataFrame:
    """
    Cumulative EPL seasons played (prior to current season).
    Uses player name as key since player_id has many NaNs.
    """
    print("    Computing EPL experience ...")
    ps = ps.copy()
    ps["_season_yr"] = ps["season"].apply(season_sort_key)
    ps = ps.sort_values(["player", "_season_yr"]).reset_index(drop=True)

    # Rank within player's career seasons
    ps["epl_experience"] = ps.groupby("player").cumcount()  # 0-indexed: seasons before current

    ps.drop(columns="_season_yr", inplace=True)
    return ps


def compute_versatility(pml: pd.DataFrame) -> pd.DataFrame:
    """Number of distinct positions played per player-season."""
    print("    Computing positional versatility from match logs ...")
    pml = pml.copy()
    pml_pos = pml[pml["pos"].notna()]

    vers = (
        pml_pos.groupby(["player", "season"])["pos"]
               .nunique()
               .reset_index()
               .rename(columns={"pos": "versatility_positions"})
    )
    return vers


def build_player_features() -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("BUILDING PLAYER FEATURES")
    print("=" * 60)

    ps, pml = load_player_data()

    ps = compute_per90(ps)
    ps = compute_player_per90_from_logs(ps, pml)
    ps = compute_goal_contribution_rate(ps)
    ps = compute_minutes_share(ps)
    ps = compute_age_bracket(ps)
    ps = compute_market_value_features(ps)
    ps = compute_experience(ps)

    # Merge consistency scores
    cons = compute_consistency_score(pml)
    ps = ps.merge(cons, on=["player", "season"], how="left")

    # Merge versatility
    vers = compute_versatility(pml)
    ps = ps.merge(vers, on=["player", "season"], how="left")

    # Data split label
    def _split(s):
        yr = season_sort_key(s)
        if yr <= 2020:
            return "train"
        elif yr <= 2022:
            return "val"
        else:
            return "test"

    ps["data_split"] = ps["season"].apply(_split)

    print(f"\n    Final player feature frame: {ps.shape[0]:,} rows x {ps.shape[1]} cols")
    return ps


# ===========================================================================
#  SECTION 3 – FEATURE SUMMARY JSON
# ===========================================================================

def build_feature_summary(match_df: pd.DataFrame,
                           player_df: pd.DataFrame) -> dict:
    print("\n[3/3] Building feature summary ...")

    def _col_stats(df, col):
        s = df[col]
        info = {
            "dtype": str(s.dtype),
            "null_count": int(s.isna().sum()),
            "null_pct": round(float(s.isna().mean() * 100), 2),
        }
        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe()
            info.update({
                "mean":  round(float(desc["mean"]),  4) if "mean"  in desc else None,
                "std":   round(float(desc["std"]),   4) if "std"   in desc else None,
                "min":   round(float(desc["min"]),   4) if "min"   in desc else None,
                "max":   round(float(desc["max"]),   4) if "max"   in desc else None,
                "p25":   round(float(desc["25%"]),   4) if "25%"   in desc else None,
                "p50":   round(float(desc["50%"]),   4) if "50%"   in desc else None,
                "p75":   round(float(desc["75%"]),   4) if "75%"   in desc else None,
            })
        elif pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
            vc = s.value_counts()
            info["top_values"] = vc.head(5).to_dict()
        return info

    # ---- match features catalogue ----
    match_feature_descriptions = {
        # ELO
        "home_elo_pre":        "Pre-match ELO rating for home team",
        "away_elo_pre":        "Pre-match ELO rating for away team",
        "elo_diff":            "Pre-match ELO difference (home - away)",
        # Rolling goals
        "home_goals_scored_3":   "Home team goals scored rolling avg (last 3)",
        "home_goals_scored_5":   "Home team goals scored rolling avg (last 5)",
        "home_goals_scored_10":  "Home team goals scored rolling avg (last 10)",
        "home_goals_conceded_3": "Home team goals conceded rolling avg (last 3)",
        "home_goals_conceded_5": "Home team goals conceded rolling avg (last 5)",
        "home_goals_conceded_10":"Home team goals conceded rolling avg (last 10)",
        "away_goals_scored_3":   "Away team goals scored rolling avg (last 3)",
        "away_goals_scored_5":   "Away team goals scored rolling avg (last 5)",
        "away_goals_scored_10":  "Away team goals scored rolling avg (last 10)",
        "away_goals_conceded_3": "Away team goals conceded rolling avg (last 3)",
        "away_goals_conceded_5": "Away team goals conceded rolling avg (last 5)",
        "away_goals_conceded_10":"Away team goals conceded rolling avg (last 10)",
        # Form
        "home_form_5":   "Home team rolling 5-match points average",
        "home_form_10":  "Home team rolling 10-match points average",
        "away_form_5":   "Away team rolling 5-match points average",
        "away_form_10":  "Away team rolling 10-match points average",
        "home_win_rate_5": "Home team rolling 5-match win rate",
        "away_win_rate_5": "Away team rolling 5-match win rate",
        # Venue form
        "home_venue_form_5": "Home team rolling 5-match form at home venues",
        "away_venue_form_5": "Away team rolling 5-match form at away venues",
        # Momentum
        "home_momentum_5": "Home rolling-5 pts minus season-to-date avg pts",
        "away_momentum_5": "Away rolling-5 pts minus season-to-date avg pts",
        # GD
        "home_gd_rolling_3":  "Home team goal difference rolling avg (last 3)",
        "home_gd_rolling_5":  "Home team goal difference rolling avg (last 5)",
        "home_gd_rolling_10": "Home team goal difference rolling avg (last 10)",
        "away_gd_rolling_3":  "Away team goal difference rolling avg (last 3)",
        "away_gd_rolling_5":  "Away team goal difference rolling avg (last 5)",
        "away_gd_rolling_10": "Away team goal difference rolling avg (last 10)",
        # Clean sheets
        "home_clean_sheet_5": "Home team clean sheet ratio (rolling 5)",
        "away_clean_sheet_5": "Away team clean sheet ratio (rolling 5)",
        # Shots on target
        "home_sot_rolling_5": "Home team shots on target rolling avg (last 5; NaN pre-2001)",
        "away_sot_rolling_5": "Away team shots on target rolling avg (last 5; NaN pre-2001)",
        # Rest
        "home_days_rest": "Days since home team's last match",
        "away_days_rest": "Days since away team's last match",
        # H2H
        "h2h_home_wins": "Home team wins in last 5 H2H meetings",
        "h2h_away_wins": "Away team wins in last 5 H2H meetings",
        "h2h_draws":     "Draws in last 5 H2H meetings",
        # Promoted
        "home_promoted": "1 if home team is newly promoted (not in prev season)",
        "away_promoted": "1 if away team is newly promoted (not in prev season)",
        # Season stage / calendar
        "matchweek":     "Approximate matchweek number within season",
        "season_stage":  "Season segment: early(1-12), mid(13-25), late(26-38)",
        "is_weekend":    "1 if match played on Saturday or Sunday",
        "is_derby":      "1 if match is a known EPL rivalry derby",
        # Derived differences
        "form_diff_5":   "Pre-match form difference (home_form_5 - away_form_5)",
        "form_diff_10":  "Pre-match form difference (home_form_10 - away_form_10)",
        "gd_trend_diff": "Goal difference trend difference (home - away, rolling 5)",
        # Split
        "data_split": "Train / Val / Test split assignment",
    }

    player_feature_descriptions = {
        # Per-90
        "goals_p90":               "Goals per 90 minutes",
        "assists_p90":             "Assists per 90 minutes",
        "goal_contributions_p90":  "Goals + assists per 90 minutes",
        "yellow_cards_p90":        "Yellow cards per 90 minutes",
        "red_cards_p90":           "Red cards per 90 minutes",
        "shots_p90":               "Shots per 90 minutes (detail era)",
        "sot_p90":                 "Shots on target per 90 minutes (detail era)",
        "tackles_p90":             "Tackles won per 90 minutes (detail era)",
        "interceptions_p90":       "Interceptions per 90 minutes (detail era)",
        "fouls_p90":               "Fouls committed per 90 minutes (detail era)",
        # Rates
        "goal_contribution_rate":  "(Goals + assists) / matches played",
        "minutes_share":           "Player minutes / (team matches * 90)",
        # Age
        "age_bracket":             "youth (<21), prime (21-29), veteran (30+)",
        "age_used":                "Age used for bracket (age_tm if available, else age)",
        # Market value
        "market_value":            "Market value in EUR (0 for early seasons)",
        "mv_prev":                 "Market value in previous season",
        "mv_change_pct":           "Season-over-season market value % change",
        "mv_per_goal_contribution":"Market value per (goals + assists) in the season",
        # Experience
        "epl_experience":          "Cumulative prior EPL seasons played",
        # Consistency
        "consistency_std":         "Std dev of per-match goal contributions",
        "consistency_mean":        "Mean per-match goal contribution",
        "consistency_cv":          "Coefficient of variation of match goal contributions",
        # Versatility
        "versatility_positions":   "Number of distinct positions played in season",
        # Split
        "data_split": "Train / Val / Test split assignment",
    }

    match_stats = {}
    for col in match_df.columns:
        match_stats[col] = _col_stats(match_df, col)

    player_stats = {}
    for col in player_df.columns:
        player_stats[col] = _col_stats(player_df, col)

    summary = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "data_split_reference": {
            "train": "seasons 2000/01 – 2020/21",
            "val":   "seasons 2021/22 – 2022/23",
            "test":  "seasons 2023/24 – 2024/25",
        },
        "match_features": {
            "shape":        list(match_df.shape),
            "row_count":    len(match_df),
            "col_count":    match_df.shape[1],
            "descriptions": match_feature_descriptions,
            "column_stats": match_stats,
        },
        "player_features": {
            "shape":        list(player_df.shape),
            "row_count":    len(player_df),
            "col_count":    player_df.shape[1],
            "descriptions": player_feature_descriptions,
            "column_stats": player_stats,
        },
    }
    return summary


# ===========================================================================
#  SECTION 4 – MAIN
# ===========================================================================

def print_section_summary(df: pd.DataFrame, label: str):
    print(f"\n  {label} summary")
    print(f"  {'─'*50}")
    print(f"  Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    null_cols = df.columns[df.isna().any()].tolist()
    print(f"  Cols with NaN  : {len(null_cols)}")
    if null_cols:
        top_null = df[null_cols].isna().mean().sort_values(ascending=False).head(10)
        for c, pct in top_null.items():
            print(f"    {c:<45} {pct*100:5.1f}% null")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"  Numeric cols   : {len(numeric_cols)}")
    cat_cols     = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    print(f"  Categorical cols: {len(cat_cols)}")


def main():
    print("\n" + "=" * 60)
    print("EPL FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    # ---- Match features ----
    match_df = build_match_features()
    print_section_summary(match_df, "match_features")

    # ---- Player features ----
    player_df = build_player_features()
    print_section_summary(player_df, "player_features")

    # ---- Save outputs ----
    print(f"\n  Saving outputs to {DATA_FEAT} ...")
    match_df.to_parquet(OUT_MATCH,  index=False)
    player_df.to_parquet(OUT_PLAYER, index=False)
    print(f"  match_features.parquet  → {OUT_MATCH.stat().st_size / 1e6:.2f} MB")
    print(f"  player_features.parquet → {OUT_PLAYER.stat().st_size / 1e6:.2f} MB")

    # ---- Feature summary ----
    summary = build_feature_summary(match_df, player_df)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  feature_summary.json    → {OUT_SUMMARY.stat().st_size / 1e3:.1f} KB")

    # ---- Final verification ----
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    for path, label in [(OUT_MATCH, "match_features"), (OUT_PLAYER, "player_features")]:
        vdf = pd.read_parquet(path)
        print(f"\n  {label}.parquet re-loaded: {vdf.shape[0]:,} rows × {vdf.shape[1]} cols")

        if "data_split" in vdf.columns:
            split_col = "data_split"
        elif "Season" in vdf.columns:
            split_col = "Season"
        else:
            split_col = None

        if split_col == "data_split":
            print("  Split breakdown:")
            print(vdf["data_split"].value_counts().to_string(header=False))

    # ---- ELO sanity check ----
    print("\n  ELO top-5 teams at final season (2024/25):")
    last = match_df[match_df["Season"] == "2024/25"].copy()
    # approximate final ELO from post-match ratings
    elo_cols = match_df[["HomeTeam", "home_elo_pre"]].rename(
        columns={"HomeTeam": "team", "home_elo_pre": "elo"}
    )
    elo_a = match_df[["AwayTeam", "away_elo_pre"]].rename(
        columns={"AwayTeam": "team", "away_elo_pre": "elo"}
    )
    final_elo = (
        pd.concat([elo_cols, elo_a], ignore_index=True)
          .groupby("team")["elo"]
          .last()
          .sort_values(ascending=False)
          .head(5)
    )
    for t, e in final_elo.items():
        print(f"    {t:<20} ELO = {e:.1f}")

    # ---- Player sanity check ----
    print("\n  Player feature sample (top scorers 2023/24):")
    p2324 = player_df[player_df["season"] == "2023/24"].sort_values("gls", ascending=False)
    show_cols = ["player", "team", "gls", "goals_p90", "goal_contribution_rate",
                 "age_bracket", "epl_experience", "data_split"]
    show_cols = [c for c in show_cols if c in p2324.columns]
    print(p2324[show_cols].head(5).to_string(index=False))

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Outputs in: {DATA_FEAT}")
    for f in DATA_FEAT.iterdir():
        print(f"    {f.name:<40} {f.stat().st_size / 1e3:>8.1f} KB")


if __name__ == "__main__":
    main()

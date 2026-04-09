"""
Scout-Specific Feature Engineering Pipeline
============================================
Reads all processed EPL data and generates enriched datasets for scouting:
  1. scout_player_profiles.parquet  — one row per player-season
  2. scout_team_profiles.parquet    — one row per team-season
  3. scout_transfer_history.parquet — one row per transfer event
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
BASE = r"C:\Users\xcv54\workspace\EPL project"
PROCESSED = os.path.join(BASE, "data", "processed")
SCOUT = os.path.join(BASE, "data", "scout")
os.makedirs(SCOUT, exist_ok=True)


def load_data():
    """Load all processed parquet files."""
    pss = pd.read_parquet(os.path.join(PROCESSED, "player_season_stats.parquet"))
    pml = pd.read_parquet(os.path.join(PROCESSED, "player_match_logs.parquet"))
    mr = pd.read_parquet(os.path.join(PROCESSED, "match_results.parquet"))
    tss = pd.read_parquet(os.path.join(PROCESSED, "team_season_summary.parquet"))
    print(f"Loaded player_season_stats  : {pss.shape}")
    print(f"Loaded player_match_logs    : {pml.shape}")
    print(f"Loaded match_results        : {mr.shape}")
    print(f"Loaded team_season_summary  : {tss.shape}")
    return pss, pml, mr, tss


# ── helpers ──────────────────────────────────────────────────────────────────
TOP6 = {"Arsenal", "Chelsea", "Liverpool", "Man City", "Man United", "Tottenham"}

POSITION_PEAK = {"FW": 27, "MF": 27, "DF": 28, "GK": 30}


def primary_position(pos_str):
    """Extract primary position group from composite position string."""
    if pd.isna(pos_str):
        return "UNK"
    first = str(pos_str).split(",")[0].strip()
    return first if first in ("FW", "MF", "DF", "GK") else "UNK"


def season_sort_key(s):
    """Convert '2000/01' to int 2000 for sorting."""
    try:
        return int(str(s).split("/")[0])
    except Exception:
        return 0


def safe_div(a, b, fill=np.nan):
    """Safe division returning fill on zero/nan denominator."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where((b == 0) | b.isna() if hasattr(b, "isna") else (b == 0), fill, a / b)
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  1. SCOUT PLAYER PROFILES
# ═══════════════════════════════════════════════════════════════════════════

def build_player_profiles(pss, pml, mr, tss):
    print("\n" + "=" * 70)
    print("Building scout_player_profiles ...")
    print("=" * 70)

    df = pss.copy()
    df["primary_pos"] = df["pos"].apply(primary_position)

    # ── per-90 stats ─────────────────────────────────────────────────────
    nineties = df["90s"].replace(0, np.nan)
    df["goals_per90"] = df["gls"] / nineties
    df["assists_per90"] = df["ast"] / nineties
    df["g_plus_a_per90"] = df["g_a"] / nineties

    # Position-specific per90 — these come from match logs aggregation
    ml_season = (
        pml.groupby(["player", "team", "season"])
        .agg(
            total_min=("min", "sum"),
            total_gls=("gls", "sum"),
            total_ast=("ast", "sum"),
            total_sh=("sh", "sum"),
            total_sot=("sot", "sum"),
            total_tklw=("tklw", "sum"),
            total_int=("int", "sum"),
            total_crs=("crs", "sum"),
            total_fls=("fls", "sum"),
            match_count=("min", "count"),
        )
        .reset_index()
    )
    ml_nineties = ml_season["total_min"] / 90.0
    ml_nineties = ml_nineties.replace(0, np.nan)

    ml_season["tackles_per90"] = ml_season["total_tklw"] / ml_nineties
    ml_season["interceptions_per90"] = ml_season["total_int"] / ml_nineties
    # clearances not available — use tackles + interceptions as defensive proxy
    ml_season["clearances_per90"] = np.nan  # placeholder
    ml_season["key_passes_per90"] = ml_season["total_crs"] / ml_nineties  # crosses as proxy
    ml_season["progressive_passes_per90"] = np.nan  # not available
    ml_season["shots_per90"] = ml_season["total_sh"] / ml_nineties
    ml_season["shot_accuracy"] = np.where(
        ml_season["total_sh"] > 0,
        ml_season["total_sot"] / ml_season["total_sh"],
        np.nan,
    )

    df = df.merge(
        ml_season[
            [
                "player", "team", "season",
                "tackles_per90", "interceptions_per90", "clearances_per90",
                "key_passes_per90", "progressive_passes_per90",
                "shots_per90", "shot_accuracy",
                "total_min", "match_count",
            ]
        ],
        on=["player", "team", "season"],
        how="left",
    )

    # ── WAR (Wins Above Replacement) ────────────────────────────────────
    # composite raw = (goals + assists*0.7 + defensive_actions*0.3) / 90min * minutes_played
    def_actions = pml.groupby(["player", "team", "season"])[["tklw", "int"]].sum().sum(axis=1).reset_index(name="def_actions")
    df = df.merge(def_actions, on=["player", "team", "season"], how="left")
    df["def_actions"] = df["def_actions"].fillna(0)

    raw_war = (df["gls"].fillna(0) + df["ast"].fillna(0) * 0.7 + df["def_actions"] * 0.3)
    df["war_raw"] = np.where(df["min"] > 0, raw_war / 90.0 * df["min"], 0)

    # Normalize within position-season
    df["war"] = df.groupby(["season", "primary_pos"])["war_raw"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

    # ── Consistency Index ────────────────────────────────────────────────
    # 1 - (std / mean) of match-level goals+assists for each player-season
    pml_copy = pml.copy()
    pml_copy["ga"] = pml_copy["gls"].fillna(0) + pml_copy["ast"].fillna(0)
    consistency = (
        pml_copy.groupby(["player", "team", "season"])["ga"]
        .agg(["mean", "std"])
        .reset_index()
    )
    consistency["consistency_index"] = np.where(
        consistency["mean"] > 0,
        1 - (consistency["std"] / consistency["mean"]),
        np.nan,
    )
    consistency["consistency_index"] = consistency["consistency_index"].clip(-2, 1)
    df = df.merge(
        consistency[["player", "team", "season", "consistency_index"]],
        on=["player", "team", "season"],
        how="left",
    )

    # ── Versatility Score ────────────────────────────────────────────────
    all_positions = pml["pos"].dropna().nunique()
    positions_played = (
        pml.groupby(["player", "team", "season"])["pos"]
        .nunique()
        .reset_index(name="n_positions_played")
    )
    positions_played["versatility_score"] = positions_played["n_positions_played"] / max(all_positions, 1)
    df = df.merge(
        positions_played[["player", "team", "season", "versatility_score"]],
        on=["player", "team", "season"],
        how="left",
    )

    # ── Team Dependency ──────────────────────────────────────────────────
    team_goals = tss[["Season", "team", "total_goals_for"]].rename(columns={"Season": "season"})
    df = df.merge(team_goals, on=["team", "season"], how="left")
    df["team_dependency"] = np.where(
        df["total_goals_for"] > 0,
        df["g_a"].fillna(0) / df["total_goals_for"],
        np.nan,
    )

    # ── Market Value Momentum ────────────────────────────────────────────
    df = df.sort_values(["player", "team", "season"])
    df["prev_market_value"] = df.groupby("player")["market_value"].shift(1)
    df["market_value_momentum"] = np.where(
        (df["prev_market_value"] > 0) & df["prev_market_value"].notna(),
        (df["market_value"] - df["prev_market_value"]) / df["prev_market_value"],
        np.nan,
    )

    # ── Minutes Trend ────────────────────────────────────────────────────
    df["prev_min"] = df.groupby("player")["min"].shift(1)
    df["minutes_trend"] = np.where(
        (df["prev_min"] > 0) & df["prev_min"].notna(),
        (df["min"] - df["prev_min"]) / df["prev_min"],
        np.nan,
    )

    # ── Big Game Performance ─────────────────────────────────────────────
    pml_copy["is_big_game"] = pml_copy["opponent"].isin(TOP6)
    big_game = (
        pml_copy[pml_copy["is_big_game"]]
        .groupby(["player", "team", "season"])
        .agg(big_gls=("gls", "sum"), big_ast=("ast", "sum"), big_min=("min", "sum"))
        .reset_index()
    )
    all_game = (
        pml_copy.groupby(["player", "team", "season"])
        .agg(all_gls=("gls", "sum"), all_ast=("ast", "sum"), all_min=("min", "sum"))
        .reset_index()
    )
    bg = big_game.merge(all_game, on=["player", "team", "season"], how="left")
    bg["big_game_ga_per90"] = np.where(
        bg["big_min"] > 0,
        (bg["big_gls"] + bg["big_ast"]) / (bg["big_min"] / 90.0),
        np.nan,
    )
    bg["all_ga_per90"] = np.where(
        bg["all_min"] > 0,
        (bg["all_gls"] + bg["all_ast"]) / (bg["all_min"] / 90.0),
        np.nan,
    )
    bg["big_game_performance"] = np.where(
        (bg["all_ga_per90"] > 0) & ~np.isnan(bg["big_game_ga_per90"]),
        bg["big_game_ga_per90"] / bg["all_ga_per90"],
        np.nan,
    )
    df = df.merge(
        bg[["player", "team", "season", "big_game_performance"]],
        on=["player", "team", "season"],
        how="left",
    )

    # ── Form Index (last 5 matches, exponential decay) ───────────────────
    def compute_form(group):
        group = group.sort_values("date")
        last5 = group.tail(5)
        if len(last5) == 0:
            return np.nan
        ga = (last5["gls"].fillna(0) + last5["ast"].fillna(0)).values
        n = len(ga)
        weights = np.array([0.5 ** (n - 1 - i) for i in range(n)])
        weights = weights / weights.sum()
        return float(np.dot(ga, weights))

    form = (
        pml_copy.groupby(["player", "team", "season"])
        .apply(compute_form)
        .reset_index(name="form_index")
    )
    df = df.merge(form, on=["player", "team", "season"], how="left")

    # ── Experience Score ─────────────────────────────────────────────────
    career = (
        pss.groupby("player")
        .agg(total_seasons=("season", "nunique"), total_appearances=("mp", "sum"))
        .reset_index()
    )
    # Cumulative up to each season
    pss_sorted = pss.sort_values(["player", "season"])
    pss_sorted["cum_seasons"] = pss_sorted.groupby("player").cumcount() + 1
    pss_sorted["cum_appearances"] = pss_sorted.groupby("player")["mp"].cumsum()
    pss_sorted["experience_score"] = pss_sorted["cum_seasons"] + pss_sorted["cum_appearances"] / 100.0
    df = df.merge(
        pss_sorted[["player", "team", "season", "experience_score"]],
        on=["player", "team", "season"],
        how="left",
    )

    # ── Peak Distance ────────────────────────────────────────────────────
    df["peak_age"] = df["primary_pos"].map(POSITION_PEAK).fillna(28)
    df["peak_distance"] = (df["age"] - df["peak_age"]).abs()

    # ── International Frequency (rarity of nationality) ──────────────────
    nation_counts = pss.groupby("nationality")["player"].nunique().reset_index(name="nation_player_count")
    total_players = pss["player"].nunique()
    nation_counts["international_frequency"] = 1 - (nation_counts["nation_player_count"] / total_players)
    df = df.merge(nation_counts[["nationality", "international_frequency"]], on="nationality", how="left")

    # ── Season Stage Performance ─────────────────────────────────────────
    pml_copy["match_num"] = pml_copy.groupby(["player", "team", "season"]).cumcount() + 1
    pml_copy["total_matches"] = pml_copy.groupby(["player", "team", "season"])["match_num"].transform("max")
    pml_copy["is_first_half"] = pml_copy["match_num"] <= (pml_copy["total_matches"] / 2)

    stage = (
        pml_copy.groupby(["player", "team", "season", "is_first_half"])
        .agg(stage_ga=("ga", "sum"), stage_min=("min", "sum"))
        .reset_index()
    )
    stage["stage_ga_per90"] = np.where(
        stage["stage_min"] > 0,
        stage["stage_ga"] / (stage["stage_min"] / 90.0),
        0,
    )
    first_half = stage[stage["is_first_half"]][["player", "team", "season", "stage_ga_per90"]].rename(
        columns={"stage_ga_per90": "first_half_ga_per90"}
    )
    second_half = stage[~stage["is_first_half"]][["player", "team", "season", "stage_ga_per90"]].rename(
        columns={"stage_ga_per90": "second_half_ga_per90"}
    )
    df = df.merge(first_half, on=["player", "team", "season"], how="left")
    df = df.merge(second_half, on=["player", "team", "season"], how="left")
    df["season_stage_ratio"] = np.where(
        df["first_half_ga_per90"] > 0,
        df["second_half_ga_per90"] / df["first_half_ga_per90"],
        np.nan,
    )

    # ── Goals Above Expected (xG not available — placeholder) ────────────
    df["goals_above_expected"] = np.nan

    # ── Career Tracking: prev season stats ───────────────────────────────
    for col in ["gls", "ast", "g_a", "min", "mp"]:
        df[f"prev_{col}"] = df.groupby("player")[col].shift(1)

    # ── Career trajectory (slope of g+a per90 over last 3 seasons) ───────
    def career_slope(group):
        group = group.sort_values("season")
        vals = group["g_plus_a_per90"].dropna().tail(3).values
        if len(vals) < 2:
            return np.nan
        x = np.arange(len(vals))
        slope, _, _, _, _ = sp_stats.linregress(x, vals)
        return slope

    trajectory = (
        df.groupby("player")
        .apply(career_slope)
        .reset_index(name="career_trajectory")
    )
    df = df.merge(trajectory, on="player", how="left")

    # ── Seasons at club ──────────────────────────────────────────────────
    df = df.sort_values(["player", "team", "season"])
    # Detect runs at the same club
    df["team_change"] = (df.groupby("player")["team"].shift(1) != df["team"]).astype(int)
    df["team_spell"] = df.groupby("player")["team_change"].cumsum()
    df["seasons_at_club"] = df.groupby(["player", "team_spell"]).cumcount() + 1

    # ── Is new signing ───────────────────────────────────────────────────
    df["is_new_signing"] = df["seasons_at_club"] == 1

    # ── Clean up ─────────────────────────────────────────────────────────
    drop_cols = [
        "def_actions", "war_raw", "prev_market_value", "prev_min",
        "total_goals_for", "team_change", "team_spell",
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Rename basic columns for scout readability
    df.rename(columns={"height_cm": "height", "90s": "nineties"}, inplace=True)

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  2. SCOUT TEAM PROFILES
# ═══════════════════════════════════════════════════════════════════════════

def build_team_profiles(pss, pml, tss, player_profiles):
    print("\n" + "=" * 70)
    print("Building scout_team_profiles ...")
    print("=" * 70)

    tf = tss.copy().rename(columns={"Season": "season"})

    # ── Squad-level aggregations from player_season_stats ────────────────
    squad_agg = (
        pss.groupby(["team", "season"])
        .agg(
            avg_age=("age", "mean"),
            avg_market_value=("market_value", "mean"),
            total_market_value=("market_value", "sum"),
            squad_size=("player", "nunique"),
            avg_minutes=("min", "mean"),
        )
        .reset_index()
    )
    tf = tf.merge(squad_agg, on=["team", "season"], how="left")

    # ── Average experience from player profiles ──────────────────────────
    exp_agg = (
        player_profiles.groupby(["team", "season"])
        .agg(
            avg_experience=("experience_score", "mean"),
            avg_war=("war", "mean"),
        )
        .reset_index()
    )
    tf = tf.merge(exp_agg, on=["team", "season"], how="left")

    # ── Playing style indicators ─────────────────────────────────────────
    tf["goals_per_game"] = np.where(
        tf["total_played"] > 0,
        tf["total_goals_for"] / tf["total_played"],
        np.nan,
    )
    tf["goals_conceded_per_game"] = np.where(
        tf["total_played"] > 0,
        tf["total_goals_against"] / tf["total_played"],
        np.nan,
    )
    tf["defensive_strength"] = 1 - (
        tf["goals_conceded_per_game"]
        / tf.groupby("season")["goals_conceded_per_game"].transform("max").replace(0, np.nan)
    )

    # ── Possession proxy: shots + corners ────────────────────────────────
    # Merge home/away shot data from match_results per team-season
    # (not directly in tss, compute from match_results)
    # For simplicity: use shots ratio from match results
    # This is a proxy — higher shots ratio ~ higher possession

    # ── Squad depth: players with 900+ minutes ───────────────────────────
    depth = (
        pss[pss["min"] >= 900]
        .groupby(["team", "season"])["player"]
        .nunique()
        .reset_index(name="squad_depth_900plus")
    )
    tf = tf.merge(depth, on=["team", "season"], how="left")
    tf["squad_depth_900plus"] = tf["squad_depth_900plus"].fillna(0).astype(int)

    # ── ELO rating (simple cumulative points-based proxy) ────────────────
    tf = tf.sort_values(["team", "season"])
    tf["elo_rating"] = 1500.0  # starting
    # Compute a rolling ELO based on points
    for team in tf["team"].unique():
        mask = tf["team"] == team
        team_df = tf.loc[mask].sort_values("season")
        elo = 1500.0
        elos = []
        for _, row in team_df.iterrows():
            expected_pts = (elo - 1500) / 10 + 50  # expected points
            actual_pts = row["points"] if pd.notna(row["points"]) else 50
            elo = elo + 0.3 * (actual_pts - expected_pts)
            elos.append(elo)
        tf.loc[team_df.index, "elo_rating"] = elos

    # ── League position ──────────────────────────────────────────────────
    tf["league_position"] = (
        tf.groupby("season")["points"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # ── Position cluster distribution (by primary position %) ────────────
    pos_dist = (
        pss.assign(primary_pos=pss["pos"].apply(primary_position))
        .groupby(["team", "season", "primary_pos"])["player"]
        .nunique()
        .unstack(fill_value=0)
    )
    pos_total = pos_dist.sum(axis=1)
    for col in pos_dist.columns:
        pos_dist[f"pct_{col}"] = pos_dist[col] / pos_total
    pos_pct = pos_dist[[c for c in pos_dist.columns if c.startswith("pct_")]].reset_index()
    tf = tf.merge(pos_pct, on=["team", "season"], how="left")

    # ── Transfer activity ────────────────────────────────────────────────
    new_signings = player_profiles[player_profiles["is_new_signing"]]
    transfer_agg = (
        new_signings.groupby(["team", "season"])
        .agg(
            num_new_signings=("player", "nunique"),
            avg_new_signing_mv=("market_value", "mean"),
        )
        .reset_index()
    )
    tf = tf.merge(transfer_agg, on=["team", "season"], how="left")
    tf["num_new_signings"] = tf["num_new_signings"].fillna(0).astype(int)

    return tf


# ═══════════════════════════════════════════════════════════════════════════
#  3. SCOUT TRANSFER HISTORY
# ═══════════════════════════════════════════════════════════════════════════

def build_transfer_history(pss, player_profiles):
    print("\n" + "=" * 70)
    print("Building scout_transfer_history ...")
    print("=" * 70)

    df = player_profiles.sort_values(["player", "season"]).copy()

    # Detect transfers: team changed between consecutive seasons
    df["prev_team"] = df.groupby("player")["team"].shift(1)
    df["prev_season"] = df.groupby("player")["season"].shift(1)

    transfers = df[
        (df["prev_team"].notna()) & (df["prev_team"] != df["team"])
    ].copy()

    transfers = transfers.rename(columns={"prev_team": "from_team", "team": "to_team"})

    # Performance before (prev season g+a per90)
    transfers["performance_before"] = transfers["prev_g_a"]
    # Performance after (current season g+a)
    transfers["performance_after"] = transfers["g_a"]

    # Adaptation success: maintained or improved per90 output
    transfers["adaptation_success"] = (
        transfers["g_plus_a_per90"] >= transfers.groupby("player")["g_plus_a_per90"].shift(1)
    ).astype(float)
    # For first transfer detected, compare with prev_g_a
    mask_first = transfers["adaptation_success"].isna()
    transfers.loc[mask_first, "adaptation_success"] = np.where(
        transfers.loc[mask_first, "prev_g_a"].notna() & (transfers.loc[mask_first, "prev_g_a"] > 0),
        (transfers.loc[mask_first, "g_a"] >= transfers.loc[mask_first, "prev_g_a"]).astype(float),
        np.nan,
    )

    # Value change
    transfers["value_change_pct"] = transfers["market_value_momentum"]

    # Style match score: compare player's primary position with the new team's
    # position distribution — higher if team has many players in same position
    # (proxy for "fits the system")
    pos_pct_map = (
        player_profiles.assign(pp=player_profiles["primary_pos"])
        .groupby(["team", "season", "pp"])["player"]
        .nunique()
        .reset_index(name="pos_count")
    )
    team_size = (
        player_profiles.groupby(["team", "season"])["player"]
        .nunique()
        .reset_index(name="team_total")
    )
    pos_pct_map = pos_pct_map.merge(team_size, on=["team", "season"])
    pos_pct_map["pos_share"] = pos_pct_map["pos_count"] / pos_pct_map["team_total"]

    transfers = transfers.merge(
        pos_pct_map.rename(columns={"team": "to_team", "pp": "primary_pos", "pos_share": "style_match_score"})[
            ["to_team", "season", "primary_pos", "style_match_score"]
        ],
        on=["to_team", "season", "primary_pos"],
        how="left",
    )

    # Select final columns
    keep_cols = [
        "player", "from_team", "to_team", "season", "prev_season",
        "primary_pos", "age",
        "performance_before", "performance_after",
        "adaptation_success", "value_change_pct", "style_match_score",
        "market_value", "war", "experience_score",
    ]
    transfers = transfers[[c for c in keep_cols if c in transfers.columns]].reset_index(drop=True)

    return transfers


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    pss, pml, mr, tss = load_data()

    # 1. Player profiles
    player_profiles = build_player_profiles(pss, pml, mr, tss)
    out1 = os.path.join(SCOUT, "scout_player_profiles.parquet")
    player_profiles.to_parquet(out1, index=False, engine="pyarrow")
    print(f"\nSaved: {out1}")
    print(f"  Shape: {player_profiles.shape}")
    print(f"  Columns: {list(player_profiles.columns)}")
    print(f"  Seasons: {sorted(player_profiles['season'].unique())[:3]} ... {sorted(player_profiles['season'].unique())[-3:]}")
    print(f"  Players: {player_profiles['player'].nunique()}")
    print(f"  Non-null WAR: {player_profiles['war'].notna().sum()}")
    print(f"  Non-null consistency: {player_profiles['consistency_index'].notna().sum()}")
    print(f"  Non-null market_value_momentum: {player_profiles['market_value_momentum'].notna().sum()}")

    # 2. Team profiles
    team_profiles = build_team_profiles(pss, pml, tss, player_profiles)
    out2 = os.path.join(SCOUT, "scout_team_profiles.parquet")
    team_profiles.to_parquet(out2, index=False, engine="pyarrow")
    print(f"\nSaved: {out2}")
    print(f"  Shape: {team_profiles.shape}")
    print(f"  Columns: {list(team_profiles.columns)}")
    print(f"  Teams: {team_profiles['team'].nunique()}")
    print(f"  Avg ELO range: {team_profiles['elo_rating'].min():.1f} — {team_profiles['elo_rating'].max():.1f}")

    # 3. Transfer history
    transfer_history = build_transfer_history(pss, player_profiles)
    out3 = os.path.join(SCOUT, "scout_transfer_history.parquet")
    transfer_history.to_parquet(out3, index=False, engine="pyarrow")
    print(f"\nSaved: {out3}")
    print(f"  Shape: {transfer_history.shape}")
    print(f"  Columns: {list(transfer_history.columns)}")
    print(f"  Unique players transferred: {transfer_history['player'].nunique()}")
    print(f"  Adaptation success rate: {transfer_history['adaptation_success'].mean():.2%}" if transfer_history['adaptation_success'].notna().any() else "  Adaptation success: no data")

    print("\n" + "=" * 70)
    print("Scout feature pipeline complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

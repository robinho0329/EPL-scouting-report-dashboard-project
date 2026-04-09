"""
EPL Team Season Summary - Comprehensive Statistical Analysis
Produces structured JSON report for Korean preprocessing-EDA report
"""

import pandas as pd
import numpy as np
import json
import os
from collections import defaultdict

# ── paths ──────────────────────────────────────────────────────────────────────
PARQUET_PATH = "C:/Users/xcv54/workspace/EPL project/data/processed/team_season_summary.parquet"
OUTPUT_PATH  = "C:/Users/xcv54/workspace/EPL project/reports/analysis_team_season.json"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

df = pd.read_parquet(PARQUET_PATH)

report = {}

# ══════════════════════════════════════════════════════════════════════════════
# 1. BASIC INFO
# ══════════════════════════════════════════════════════════════════════════════
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
object_cols  = df.select_dtypes(include=["object"]).columns.tolist()

report["1_basic_info"] = {
    "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
    "columns": df.columns.tolist(),
    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    "numeric_columns": numeric_cols,
    "categorical_columns": object_cols,
    "memory_usage_bytes": {
        "total": int(df.memory_usage(deep=True).sum()),
        "per_column": {col: int(v) for col, v in df.memory_usage(deep=True).items()},
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# 2. MISSING VALUES
# ══════════════════════════════════════════════════════════════════════════════
missing_count = df.isnull().sum()
missing_pct   = (df.isnull().mean() * 100).round(4)

report["2_missing_values"] = {
    "total_missing_cells": int(missing_count.sum()),
    "total_missing_pct": round(float(missing_count.sum() / df.size * 100), 4),
    "per_column": {
        col: {
            "count": int(missing_count[col]),
            "percentage": float(missing_pct[col]),
        }
        for col in df.columns
    },
    "columns_with_missing": [col for col in df.columns if missing_count[col] > 0],
}

# ══════════════════════════════════════════════════════════════════════════════
# 3. DESCRIPTIVE STATISTICS FOR NUMERIC COLUMNS
# ══════════════════════════════════════════════════════════════════════════════
desc = df[numeric_cols].describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90])

def safe_float(val):
    if isinstance(val, (np.integer,)):  return int(val)
    if isinstance(val, (np.floating,)): return round(float(val), 4)
    return val

desc_dict = {}
for col in numeric_cols:
    col_stats = desc[col].to_dict()
    col_stats["skewness"]  = round(float(df[col].skew()), 4)
    col_stats["kurtosis"]  = round(float(df[col].kurt()), 4)
    col_stats["variance"]  = round(float(df[col].var()), 4)
    col_stats["sum"]       = int(df[col].sum())
    desc_dict[col] = {k: safe_float(v) for k, v in col_stats.items()}

report["3_descriptive_statistics"] = desc_dict

# ══════════════════════════════════════════════════════════════════════════════
# 4. SEASON COVERAGE
# ══════════════════════════════════════════════════════════════════════════════
teams_per_season = df.groupby("Season")["team"].count().to_dict()
unique_teams_all = df["team"].unique().tolist()

season_list = sorted(df["Season"].unique().tolist())

report["4_season_coverage"] = {
    "total_seasons": int(df["Season"].nunique()),
    "seasons": season_list,
    "teams_per_season": {s: int(c) for s, c in teams_per_season.items()},
    "teams_per_season_stats": {
        "min":  int(min(teams_per_season.values())),
        "max":  int(max(teams_per_season.values())),
        "mean": round(float(np.mean(list(teams_per_season.values()))), 2),
    },
    "unique_teams_ever": sorted(unique_teams_all),
    "total_unique_teams": len(unique_teams_all),
}

# ══════════════════════════════════════════════════════════════════════════════
# 5. TOP TEAMS BY GOALS SCORED (all-time aggregate)
# ══════════════════════════════════════════════════════════════════════════════
goals_agg = (
    df.groupby("team")
      .agg(
          total_goals_for=("total_goals_for", "sum"),
          total_goals_against=("total_goals_against", "sum"),
          seasons_played=("Season", "nunique"),
      )
      .sort_values("total_goals_for", ascending=False)
      .reset_index()
)
goals_agg["goals_per_season"] = (
    goals_agg["total_goals_for"] / goals_agg["seasons_played"]
).round(2)

report["5_top_teams_goals"] = {
    "top_20_total_goals": goals_agg.head(20)[
        ["team", "total_goals_for", "total_goals_against", "seasons_played", "goals_per_season"]
    ].to_dict(orient="records"),
    "top_10_goals_per_season": goals_agg.sort_values("goals_per_season", ascending=False)
      .head(10)[["team", "goals_per_season", "seasons_played"]].to_dict(orient="records"),
    "all_teams_goal_summary": goals_agg[
        ["team", "total_goals_for", "total_goals_against", "seasons_played", "goals_per_season"]
    ].to_dict(orient="records"),
}

# ══════════════════════════════════════════════════════════════════════════════
# 6. TOP TEAMS BY WINS
# ══════════════════════════════════════════════════════════════════════════════
wins_agg = (
    df.groupby("team")
      .agg(
          total_wins=("total_wins", "sum"),
          total_draws=("total_draws", "sum"),
          total_losses=("total_losses", "sum"),
          total_points=("points", "sum"),
          seasons_played=("Season", "nunique"),
      )
      .reset_index()
)
wins_agg["win_rate"]      = (wins_agg["total_wins"] / (wins_agg["total_wins"] + wins_agg["total_draws"] + wins_agg["total_losses"]) * 100).round(2)
wins_agg["points_per_season"] = (wins_agg["total_points"] / wins_agg["seasons_played"]).round(2)
wins_agg = wins_agg.sort_values("total_wins", ascending=False).reset_index(drop=True)

report["6_top_teams_wins"] = {
    "top_20_total_wins": wins_agg.head(20)[
        ["team", "total_wins", "total_draws", "total_losses", "win_rate", "total_points", "points_per_season", "seasons_played"]
    ].to_dict(orient="records"),
    "top_10_win_rate_min5_seasons": wins_agg[wins_agg["seasons_played"] >= 5]
      .sort_values("win_rate", ascending=False)
      .head(10)[["team", "win_rate", "seasons_played"]].to_dict(orient="records"),
}

# ══════════════════════════════════════════════════════════════════════════════
# 7. DISTRIBUTION OF NUMERIC STATS
# ══════════════════════════════════════════════════════════════════════════════
dist_report = {}
for col in numeric_cols:
    series = df[col].dropna()
    # histogram bins (10 equal-width)
    counts, bin_edges = np.histogram(series, bins=10)
    dist_report[col] = {
        "histogram": {
            "counts": counts.tolist(),
            "bin_edges": [round(float(e), 4) for e in bin_edges.tolist()],
        },
        "quartiles": {
            "Q1":     round(float(series.quantile(0.25)), 4),
            "median": round(float(series.quantile(0.50)), 4),
            "Q3":     round(float(series.quantile(0.75)), 4),
            "IQR":    round(float(series.quantile(0.75) - series.quantile(0.25)), 4),
        },
        "outliers_iqr_method": {
            "lower_fence": round(float(series.quantile(0.25) - 1.5 * (series.quantile(0.75) - series.quantile(0.25))), 4),
            "upper_fence": round(float(series.quantile(0.75) + 1.5 * (series.quantile(0.75) - series.quantile(0.25))), 4),
            "outlier_count": int(((series < series.quantile(0.25) - 1.5 * (series.quantile(0.75) - series.quantile(0.25))) |
                                   (series > series.quantile(0.75) + 1.5 * (series.quantile(0.75) - series.quantile(0.25)))).sum()),
        },
    }

report["7_distribution_stats"] = dist_report

# ══════════════════════════════════════════════════════════════════════════════
# 8. ERA COMPARISON: 2000-2012 vs 2013-2025
# ══════════════════════════════════════════════════════════════════════════════
def season_start_year(s):
    return int(str(s).split("/")[0])

df["_season_start"] = df["Season"].apply(season_start_year)

era_early = df[df["_season_start"].between(2000, 2012)]
era_late  = df[df["_season_start"].between(2013, 2025)]

era_report = {}
for col in numeric_cols:
    era_report[col] = {
        "era_2000_2012": {
            "mean":   round(float(era_early[col].mean()), 4),
            "median": round(float(era_early[col].median()), 4),
            "std":    round(float(era_early[col].std()), 4),
            "n":      int(era_early[col].count()),
        },
        "era_2013_2025": {
            "mean":   round(float(era_late[col].mean()), 4),
            "median": round(float(era_late[col].median()), 4),
            "std":    round(float(era_late[col].std()), 4),
            "n":      int(era_late[col].count()),
        },
        "mean_diff_late_minus_early": round(float(era_late[col].mean() - era_early[col].mean()), 4),
        "pct_change": round(float(
            (era_late[col].mean() - era_early[col].mean()) / era_early[col].mean() * 100
        ), 2) if era_early[col].mean() != 0 else None,
    }

report["8_era_comparison"] = {
    "era_2000_2012_seasons": sorted(era_early["Season"].unique().tolist()),
    "era_2013_2025_seasons": sorted(era_late["Season"].unique().tolist()),
    "era_2000_2012_team_entries": int(len(era_early)),
    "era_2013_2025_team_entries": int(len(era_late)),
    "per_column": era_report,
}

# ══════════════════════════════════════════════════════════════════════════════
# 9. PROMOTED / RELEGATED TEAMS FREQUENCY
# ══════════════════════════════════════════════════════════════════════════════
team_seasons = df.groupby("team")["_season_start"].apply(sorted).reset_index()

promotion_events  = defaultdict(int)
relegation_events = defaultdict(int)
total_seasons_per_team = {}

all_seasons_sorted = sorted(df["_season_start"].unique().tolist())

for _, row in team_seasons.iterrows():
    team   = row["team"]
    yrs    = list(row["_season_start"])
    total_seasons_per_team[team] = len(yrs)
    for i in range(1, len(yrs)):
        if yrs[i] - yrs[i-1] > 1:
            # disappeared → relegated after prev season
            relegation_events[team] += 1
        # reappeared → promoted (if they were absent)
    # if team first appeared after 2000 → promoted at some point (infer from gap)
    for i in range(1, len(yrs)):
        if yrs[i] - yrs[i-1] > 1:
            promotion_events[team] += 1

# Consecutive seasons fully present = never relegated
always_present = [t for t, cnt in total_seasons_per_team.items() if cnt == len(all_seasons_sorted)]
promoted_teams_ranked  = sorted(promotion_events.items(),  key=lambda x: x[1], reverse=True)
relegated_teams_ranked = sorted(relegation_events.items(), key=lambda x: x[1], reverse=True)

report["9_promotion_relegation"] = {
    "always_present_teams": sorted(always_present),
    "always_present_count": len(always_present),
    "teams_with_promotion_events": promoted_teams_ranked,
    "teams_with_relegation_events": relegated_teams_ranked,
    "seasons_per_team": dict(sorted(total_seasons_per_team.items(), key=lambda x: x[1], reverse=True)),
    "teams_with_only_1_season": [t for t, c in total_seasons_per_team.items() if c == 1],
    "teams_with_10_plus_seasons": [t for t, c in total_seasons_per_team.items() if c >= 10],
}

# ══════════════════════════════════════════════════════════════════════════════
# 10. TRANSFER-RELATED COLUMNS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
transfer_keywords = ["transfer", "spend", "fee", "buy", "sell", "squad", "wage",
                     "market", "value", "budget", "loan", "sign"]
transfer_cols = [c for c in df.columns if any(kw in c.lower() for kw in transfer_keywords)]

report["10_transfer_analysis"] = {
    "transfer_related_columns_found": transfer_cols,
    "note": (
        "No dedicated transfer columns found in this dataset. "
        "The dataset contains match-result statistics only "
        "(played, wins, draws, losses, goals for/against, goal diff, points). "
        "Transfer data would need to be sourced from a separate dataset."
        if not transfer_cols else f"Found {len(transfer_cols)} transfer-related columns."
    ),
}

# ══════════════════════════════════════════════════════════════════════════════
# 11. DATA QUALITY CHECKS
# ══════════════════════════════════════════════════════════════════════════════
# Duplicates
dup_full   = int(df.drop(columns=["_season_start"]).duplicated().sum())
dup_key    = int(df.duplicated(subset=["Season", "team"]).sum())

# Logical checks
issues = {}

# games played should equal wins + draws + losses
df["_calc_played"] = df["home_wins"] + df["home_draws"] + df["home_losses"]
home_mismatch = int((df["_calc_played"] != df["home_played"]).sum())

df["_calc_played_away"] = df["away_wins"] + df["away_draws"] + df["away_losses"]
away_mismatch = int((df["_calc_played_away"] != df["away_played"]).sum())

df["_calc_total_wins"]   = df["home_wins"]   + df["away_wins"]
df["_calc_total_draws"]  = df["home_draws"]  + df["away_draws"]
df["_calc_total_losses"] = df["home_losses"] + df["away_losses"]
total_wins_mismatch   = int((df["_calc_total_wins"]   != df["total_wins"]).sum())
total_draws_mismatch  = int((df["_calc_total_draws"]  != df["total_draws"]).sum())
total_losses_mismatch = int((df["_calc_total_losses"] != df["total_losses"]).sum())

# goals consistency
df["_calc_gf"] = df["home_goals_for"] + df["away_goals_for"]
df["_calc_ga"] = df["home_goals_against"] + df["away_goals_against"]
gf_mismatch = int((df["_calc_gf"] != df["total_goals_for"]).sum())
ga_mismatch = int((df["_calc_ga"] != df["total_goals_against"]).sum())

# goal diff
df["_calc_gd"] = df["total_goals_for"] - df["total_goals_against"]
gd_mismatch = int((df["_calc_gd"] != df["goal_diff"]).sum())

# points = 3*wins + draws
df["_calc_pts"] = 3 * df["total_wins"] + df["total_draws"]
pts_mismatch = int((df["_calc_pts"] != df["points"]).sum())

# negative values check
neg_checks = {}
for col in numeric_cols:
    if col == "goal_diff":
        continue  # can be negative legitimately
    neg_count = int((df[col] < 0).sum())
    if neg_count > 0:
        neg_checks[col] = neg_count

# range checks
range_checks = {
    "total_played_range":         {"min": int(df["total_played"].min()), "max": int(df["total_played"].max())},
    "total_wins_range":           {"min": int(df["total_wins"].min()),   "max": int(df["total_wins"].max())},
    "total_goals_for_range":      {"min": int(df["total_goals_for"].min()), "max": int(df["total_goals_for"].max())},
    "points_range":               {"min": int(df["points"].min()),       "max": int(df["points"].max())},
    "goal_diff_range":            {"min": int(df["goal_diff"].min()),    "max": int(df["goal_diff"].max())},
}

# Extreme single-season records
best_season  = df.loc[df["points"].idxmax(),  ["Season","team","points","total_wins","total_goals_for"]].to_dict()
worst_season = df.loc[df["points"].idxmin(),  ["Season","team","points","total_wins","total_goals_for"]].to_dict()
most_goals   = df.loc[df["total_goals_for"].idxmax(), ["Season","team","total_goals_for","total_goals_against"]].to_dict()
most_conceded= df.loc[df["total_goals_against"].idxmax(), ["Season","team","total_goals_for","total_goals_against"]].to_dict()

report["11_data_quality"] = {
    "duplicate_rows_full": dup_full,
    "duplicate_season_team_keys": dup_key,
    "logical_consistency_checks": {
        "home_played_vs_w_d_l_mismatch":  home_mismatch,
        "away_played_vs_w_d_l_mismatch":  away_mismatch,
        "total_wins_sum_mismatch":        total_wins_mismatch,
        "total_draws_sum_mismatch":       total_draws_mismatch,
        "total_losses_sum_mismatch":      total_losses_mismatch,
        "total_goals_for_sum_mismatch":   gf_mismatch,
        "total_goals_against_sum_mismatch": ga_mismatch,
        "goal_diff_calculation_mismatch": gd_mismatch,
        "points_3w_plus_d_mismatch":      pts_mismatch,
        "all_checks_passed": all(v == 0 for v in [
            home_mismatch, away_mismatch, total_wins_mismatch,
            total_draws_mismatch, total_losses_mismatch,
            gf_mismatch, ga_mismatch, gd_mismatch, pts_mismatch
        ]),
    },
    "negative_values_in_non_goal_diff_cols": neg_checks if neg_checks else "없음 (none found)",
    "value_range_checks": range_checks,
    "notable_single_season_records": {
        "highest_points":   {k: (int(v) if isinstance(v, (np.integer,)) else str(v)) for k, v in best_season.items()},
        "lowest_points":    {k: (int(v) if isinstance(v, (np.integer,)) else str(v)) for k, v in worst_season.items()},
        "most_goals_scored":{k: (int(v) if isinstance(v, (np.integer,)) else str(v)) for k, v in most_goals.items()},
        "most_goals_conceded":{k: (int(v) if isinstance(v, (np.integer,)) else str(v)) for k, v in most_conceded.items()},
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# METADATA
# ══════════════════════════════════════════════════════════════════════════════
report["_metadata"] = {
    "source_file": PARQUET_PATH,
    "analysis_date": "2026-03-21",
    "total_sections": 11,
    "description": "EPL team season summary comprehensive statistical analysis",
}

# ══════════════════════════════════════════════════════════════════════════════
# SAVE JSON
# ══════════════════════════════════════════════════════════════════════════════
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=str)

print(f"[OK] JSON saved to: {OUTPUT_PATH}")
print(f"     Sections: {list(report.keys())}")

# ── quick sanity print ─────────────────────────────────────────────────────────
print("\n=== QUICK SUMMARY ===")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} cols")
print(f"Seasons: {df['Season'].nunique()} | Teams ever: {df['team'].nunique()}")
print(f"Missing cells: {int(df.isnull().sum().sum())}")
print(f"Duplicate (Season+team) keys: {dup_key}")
print(f"All logic checks passed: {report['11_data_quality']['logical_consistency_checks']['all_checks_passed']}")
print(f"Era 2000-2012 entries: {len(era_early)} | Era 2013-2025 entries: {len(era_late)}")
print(f"Always-present teams: {len(always_present)}")

"""
EPL FBref Match Log Data Analysis Script
Korean preprocessing-EDA report
"""

import os
import json
import random
import warnings
import traceback
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

BASE_DIR = "C:/Users/xcv54/workspace/EPL project/data/raw/fbref"
PARQUET_PATH = "C:/Users/xcv54/workspace/EPL project/data/processed/player_match_logs.parquet"
REPORTS_DIR = "C:/Users/xcv54/workspace/EPL project/reports"
OUTPUT_PATH = f"{REPORTS_DIR}/analysis_fbref_matchlogs.json"

os.makedirs(REPORTS_DIR, exist_ok=True)


def to_native(obj):
    """Recursively convert numpy/pandas types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    elif pd.isna(obj) if not isinstance(obj, (list, dict, str)) else False:
        return None
    else:
        return obj


# ─────────────────────────────────────────────
# 1. Collect all CSV file paths
# ─────────────────────────────────────────────
print("Step 1: Collecting all CSV files...")

all_files = []  # list of (season, team, filepath)
files_per_season = defaultdict(int)
files_per_team = defaultdict(int)

seasons = sorted(os.listdir(BASE_DIR))
for season in seasons:
    season_path = os.path.join(BASE_DIR, season)
    if not os.path.isdir(season_path):
        continue
    try:
        teams = sorted(os.listdir(season_path))
    except Exception:
        continue
    for team in teams:
        ml_path = os.path.join(season_path, team, "matchlogs")
        if not os.path.isdir(ml_path):
            continue
        try:
            csvs = [f for f in os.listdir(ml_path) if f.endswith(".csv")]
        except Exception:
            continue
        for csv in csvs:
            full_path = os.path.join(ml_path, csv)
            all_files.append((season, team, full_path))
            files_per_season[season] += 1
            files_per_team[team] += 1

total_files = len(all_files)
print(f"  Total CSV files found: {total_files}")


# ─────────────────────────────────────────────
# 2. Sample files across seasons/teams
# ─────────────────────────────────────────────
print("Step 2: Sampling files for analysis...")

# Stratified sample: pick files from each season proportionally (min 2 per season)
sample_set = []
random.seed(42)
per_season_groups = defaultdict(list)
for item in all_files:
    per_season_groups[item[0]].append(item)

for season, items in per_season_groups.items():
    n = max(4, min(len(items), int(len(items) * 0.15)))  # at least 4, up to 15%
    sample_set.extend(random.sample(items, min(n, len(items))))

# Make sure we have at least 100
if len(sample_set) < 100:
    remaining = [f for f in all_files if f not in set(map(tuple, sample_set))]
    extra = random.sample(remaining, min(100 - len(sample_set), len(remaining)))
    sample_set.extend(extra)

print(f"  Sample size: {len(sample_set)} files")


# ─────────────────────────────────────────────
# 3. Read sampled files
# ─────────────────────────────────────────────
print("Step 3: Reading sampled files...")

dfs = []
load_errors = []
header_contaminations = []
encoding_issues = []

for season, team, fpath in sample_set:
    try:
        df = pd.read_csv(fpath, encoding="utf-8-sig", low_memory=False)
        df["_season"] = season
        df["_team"] = team
        df["_file"] = fpath
        # Detect header contamination (rows where numeric cols contain column names)
        if len(df) > 0 and "Date" in df.columns:
            mask = df["Date"] == "Date"
            if mask.any():
                header_contaminations.append(fpath)
                df = df[~mask].copy()
        dfs.append(df)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(fpath, encoding="latin-1", low_memory=False)
            df["_season"] = season
            df["_team"] = team
            df["_file"] = fpath
            dfs.append(df)
            encoding_issues.append(fpath)
        except Exception as e2:
            load_errors.append({"file": fpath, "error": str(e2)})
    except Exception as e:
        load_errors.append({"file": fpath, "error": str(e)})

print(f"  Successfully loaded: {len(dfs)} files")
print(f"  Load errors: {len(load_errors)}")


# ─────────────────────────────────────────────
# 4. Analyze column presence per era
# ─────────────────────────────────────────────
print("Step 4: Analyzing columns per era...")

early_cols = Counter()   # 2000-2005
recent_cols = Counter()  # 2020-2025
mid_cols = Counter()     # 2013-2025

for season, team, fpath in sample_set:
    # Find the df for this file
    matching = [d for d in dfs if d["_file"].iloc[0] == fpath] if dfs else []
    # We'll do this differently - iterate dfs directly
    pass

# Reset: iterate dfs list
early_col_sets = []
recent_col_sets = []
mid_col_sets = []
all_col_counter = Counter()

for df in dfs:
    season = df["_season"].iloc[0]
    year_start = int(season.split("-")[0])
    meta_cols = {"_season", "_team", "_file"}
    data_cols = [c for c in df.columns if c not in meta_cols]
    all_col_counter.update(data_cols)
    if year_start <= 2005:
        early_col_sets.append(set(data_cols))
    if year_start >= 2020:
        recent_col_sets.append(set(data_cols))
    if year_start >= 2013:
        mid_col_sets.append(set(data_cols))

# Common columns in early vs recent
if early_col_sets:
    early_common = set.intersection(*early_col_sets) if len(early_col_sets) > 1 else early_col_sets[0]
    early_all = set.union(*early_col_sets) if early_col_sets else set()
else:
    early_common = set()
    early_all = set()

if recent_col_sets:
    recent_common = set.intersection(*recent_col_sets) if len(recent_col_sets) > 1 else recent_col_sets[0]
    recent_all = set.union(*recent_col_sets) if recent_col_sets else set()
else:
    recent_common = set()
    recent_all = set()

if mid_col_sets:
    mid_common = set.intersection(*mid_col_sets) if len(mid_col_sets) > 1 else mid_col_sets[0]
    mid_all = set.union(*mid_col_sets) if mid_col_sets else set()
else:
    mid_common = set()
    mid_all = set()

cols_only_in_recent = recent_all - early_all
cols_only_in_early = early_all - recent_all
cols_in_both = early_all & recent_all

print(f"  Early columns (union): {len(early_all)}, Recent columns (union): {len(recent_all)}")


# ─────────────────────────────────────────────
# 5. Combine all sampled data for global analysis
# ─────────────────────────────────────────────
print("Step 5: Combining data for analysis...")

combined = pd.concat(dfs, ignore_index=True, sort=False)
print(f"  Combined shape: {combined.shape}")

meta_cols = {"_season", "_team", "_file"}
data_cols = [c for c in combined.columns if c not in meta_cols]


# ─────────────────────────────────────────────
# 6. Missing value analysis
# ─────────────────────────────────────────────
print("Step 6: Missing value analysis...")

missing_counts = combined[data_cols].isnull().sum()
missing_pct = (missing_counts / len(combined) * 100).round(2)
missing_report = {
    col: {"count": int(missing_counts[col]), "pct": float(missing_pct[col])}
    for col in data_cols
}
top_missing = sorted(missing_report.items(), key=lambda x: x[1]["pct"], reverse=True)[:20]


# ─────────────────────────────────────────────
# 7. Per-player match count distribution
# ─────────────────────────────────────────────
print("Step 7: Player match count distribution...")

if "_file" in combined.columns:
    # Each file = one player-season
    per_player_counts = combined.groupby("_file").size()
    match_count_stats = {
        "min": int(per_player_counts.min()),
        "max": int(per_player_counts.max()),
        "mean": round(float(per_player_counts.mean()), 2),
        "median": round(float(per_player_counts.median()), 2),
        "std": round(float(per_player_counts.std()), 2),
        "percentiles": {
            "10": round(float(per_player_counts.quantile(0.10)), 2),
            "25": round(float(per_player_counts.quantile(0.25)), 2),
            "75": round(float(per_player_counts.quantile(0.75)), 2),
            "90": round(float(per_player_counts.quantile(0.90)), 2),
        },
        "total_player_season_files_in_sample": int(per_player_counts.count()),
    }
else:
    match_count_stats = {}


# ─────────────────────────────────────────────
# 8. Minutes played distribution
# ─────────────────────────────────────────────
print("Step 8: Minutes played distribution...")

minutes_col = None
for col in ["Min", "Minutes", "MP", "min"]:
    if col in combined.columns:
        minutes_col = col
        break

minutes_stats = {}
if minutes_col:
    mins = pd.to_numeric(combined[minutes_col], errors="coerce").dropna()
    if len(mins) > 0:
        minutes_stats = {
            "column_used": minutes_col,
            "count": int(mins.count()),
            "min": float(mins.min()),
            "max": float(mins.max()),
            "mean": round(float(mins.mean()), 2),
            "median": round(float(mins.median()), 2),
            "std": round(float(mins.std()), 2),
            "pct_90_mins": round(float((mins == 90).sum() / len(mins) * 100), 2),
            "pct_zero_mins": round(float((mins == 0).sum() / len(mins) * 100), 2),
            "value_counts_common": mins.value_counts().head(10).to_dict(),
        }


# ─────────────────────────────────────────────
# 9. Goals, assists, shots distributions
# ─────────────────────────────────────────────
print("Step 9: Goals/assists/shots distributions...")

stat_cols_map = {
    "goals": ["gls", "Gls", "Goals", "G"],
    "assists": ["ast", "Ast", "Assists", "A"],
    "shots": ["sh", "Sh", "Shots"],
    "shots_on_target": ["sot", "SoT", "SoT%"],
    "penalty_goals": ["pk", "PK"],
    "penalty_attempts": ["pkatt", "PKatt"],
    "yellow_cards": ["crdy", "CrdY"],
    "red_cards": ["crdr", "CrdR"],
    "fouls_committed": ["fls", "Fls"],
    "fouls_drawn": ["fld", "Fld"],
    "offsides": ["off", "Off"],
    "crosses": ["crs", "Crs"],
    "tackles_won": ["tklw", "TklW"],
    "interceptions": ["int", "Int"],
    "own_goals": ["og", "OG"],
    "xg": ["xG"],
    "xa": ["xA", "xAG"],
}

stat_distributions = {}
for stat_name, candidates in stat_cols_map.items():
    for col in candidates:
        if col in combined.columns:
            vals = pd.to_numeric(combined[col], errors="coerce").dropna()
            if len(vals) > 0:
                stat_distributions[stat_name] = {
                    "column": col,
                    "count": int(vals.count()),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "mean": round(float(vals.mean()), 4),
                    "median": round(float(vals.median()), 4),
                    "std": round(float(vals.std()), 4),
                    "total_sum": float(vals.sum()),
                    "pct_zero": round(float((vals == 0).sum() / len(vals) * 100), 2),
                    "value_counts": {str(k): int(v) for k, v in vals.value_counts().head(15).items()},
                }
            break


# ─────────────────────────────────────────────
# 10. Start vs Sub ratio
# ─────────────────────────────────────────────
print("Step 10: Start vs Sub ratio...")

start_sub_analysis = {}
for col in ["start", "Start", "Starts", "GS", "started"]:
    if col in combined.columns:
        vc = combined[col].value_counts(dropna=False).to_dict()
        start_sub_analysis = {
            "column": col,
            "value_counts": {str(k): int(v) for k, v in vc.items()},
        }
        # Y = started, N = sub, Y* = captain
        y_count = int(combined[col].isin(["Y", "Y*"]).sum())
        n_count = int((combined[col] == "N").sum())
        total_app = y_count + n_count
        start_sub_analysis["starter_count"] = y_count
        start_sub_analysis["sub_count"] = n_count
        start_sub_analysis["total_appearances"] = total_app
        start_sub_analysis["start_pct"] = round(y_count / total_app * 100, 2) if total_app > 0 else None
        start_sub_analysis["sub_pct"] = round(n_count / total_app * 100, 2) if total_app > 0 else None
        break

# Also check if there's a position column for sub detection
for col in ["pos", "Pos", "Position"]:
    if col in combined.columns:
        pos_vc = combined[col].value_counts(dropna=False).head(20).to_dict()
        start_sub_analysis["position_col"] = col
        start_sub_analysis["position_values"] = {str(k): int(v) for k, v in pos_vc.items()}
        break


# ─────────────────────────────────────────────
# 11. Result column format analysis
# ─────────────────────────────────────────────
print("Step 11: Result column format analysis...")

result_analysis = {}
for col in ["result", "Result", "Res"]:
    if col in combined.columns:
        sample_vals = combined[col].dropna().unique()[:50]
        result_analysis["column"] = col
        result_analysis["unique_count"] = int(combined[col].nunique())
        result_analysis["sample_values"] = [str(v) for v in sample_vals[:20]]
        result_analysis["null_count"] = int(combined[col].isnull().sum())

        # Check for W/D/L
        wins = combined[col].str.startswith("W", na=False).sum()
        draws = combined[col].str.startswith("D", na=False).sum()
        losses = combined[col].str.startswith("L", na=False).sum()
        result_analysis["outcome_counts"] = {
            "W": int(wins), "D": int(draws), "L": int(losses)
        }

        # Check dash format: hyphen "-" (0x2D) vs en-dash "–" (0x2013)
        hyphen_count = combined[col].dropna().str.contains("-", regex=False).sum()
        endash_count = combined[col].dropna().str.contains("–", regex=False).sum()
        result_analysis["dash_format"] = {
            "hyphen_(-_0x2D)": int(hyphen_count),
            "en_dash_(–_0x2013)": int(endash_count),
        }
        break


# ─────────────────────────────────────────────
# 12. Position values
# ─────────────────────────────────────────────
print("Step 12: Position values...")

position_analysis = {}
for col in ["Pos", "Position", "pos"]:
    if col in combined.columns:
        vc = combined[col].value_counts(dropna=False).head(30)
        position_analysis = {
            "column": col,
            "unique_values": int(combined[col].nunique()),
            "value_counts": {str(k): int(v) for k, v in vc.items()},
        }
        break


# ─────────────────────────────────────────────
# 13. Date range coverage
# ─────────────────────────────────────────────
print("Step 13: Date range coverage...")

date_analysis = {}
for col in ["Date", "date"]:
    if col in combined.columns:
        dates = pd.to_datetime(combined[col], errors="coerce")
        valid_dates = dates.dropna()
        if len(valid_dates) > 0:
            date_analysis = {
                "column": col,
                "min_date": str(valid_dates.min().date()),
                "max_date": str(valid_dates.max().date()),
                "total_valid_dates": int(len(valid_dates)),
                "null_date_count": int(dates.isnull().sum()),
                "years_covered": sorted(valid_dates.dt.year.unique().tolist()),
            }
        break


# ─────────────────────────────────────────────
# 14. Data quality: encoding issues, header contamination, duplicates
# ─────────────────────────────────────────────
print("Step 14: Data quality checks...")

# Duplicate rows - within each file (player-season), check for duplicated match dates
# True duplicates = same player file has two rows with identical date+round
dup_count = combined.duplicated(subset=[c for c in ["date", "_file", "round"] if c in combined.columns]).sum()

# Check for numeric cols with string contamination
contamination_report = {}
numeric_candidates = ["gls", "ast", "min", "sh", "sot", "Gls", "Ast", "Min", "Sh", "SoT", "xG", "xA"]
for col in numeric_candidates:
    if col in combined.columns:
        raw = combined[col].dropna()
        coerced = pd.to_numeric(raw, errors="coerce")
        bad = coerced.isnull().sum()
        if bad > 0:
            bad_vals = raw[coerced.isnull()].unique()[:10]
            contamination_report[col] = {
                "non_numeric_count": int(bad),
                "sample_bad_values": [str(v) for v in bad_vals],
            }

# Data type summary
dtype_summary = {}
for col in data_cols:
    if col in combined.columns:
        dtype_summary[col] = str(combined[col].dtype)


# ─────────────────────────────────────────────
# 15. Column by column stats summary
# ─────────────────────────────────────────────
print("Step 15: Column stats summary...")

col_stats = {}
for col in data_cols:
    if col not in combined.columns:
        continue
    s = combined[col]
    info = {
        "dtype": str(s.dtype),
        "null_count": int(s.isnull().sum()),
        "null_pct": round(float(s.isnull().mean() * 100), 2),
        "unique_count": int(s.nunique(dropna=True)),
    }
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().sum() > s.notna().sum() * 0.5:  # mostly numeric
        info["numeric_stats"] = {
            "min": to_native(numeric.min()),
            "max": to_native(numeric.max()),
            "mean": round(float(numeric.mean()), 4) if numeric.notna().any() else None,
            "std": round(float(numeric.std()), 4) if numeric.notna().any() else None,
        }
    else:
        vc = s.value_counts(dropna=False).head(10)
        info["top_values"] = {str(k): int(v) for k, v in vc.items()}
    col_stats[col] = info


# ─────────────────────────────────────────────
# 16. Era comparison: 2000-2012 vs 2013-2025
# ─────────────────────────────────────────────
print("Step 16: Era comparison...")

era_old_dfs = [df for df in dfs if int(df["_season"].iloc[0].split("-")[0]) <= 2012]
era_new_dfs = [df for df in dfs if int(df["_season"].iloc[0].split("-")[0]) >= 2013]

def get_era_col_info(dfs_list):
    if not dfs_list:
        return {}
    combined_era = pd.concat(dfs_list, ignore_index=True, sort=False)
    era_data_cols = [c for c in combined_era.columns if c not in meta_cols]
    col_union = set()
    col_sets_list = []
    for df in dfs_list:
        dc = [c for c in df.columns if c not in meta_cols]
        col_sets_list.append(set(dc))
        col_union |= set(dc)
    if col_sets_list:
        col_intersection = set.intersection(*col_sets_list)
    else:
        col_intersection = set()
    missing_pcts = {}
    for col in era_data_cols:
        mp = float(combined_era[col].isnull().mean() * 100)
        missing_pcts[col] = round(mp, 2)
    return {
        "file_count": len(dfs_list),
        "row_count": len(combined_era),
        "columns_union": sorted(col_union),
        "columns_always_present": sorted(col_intersection),
        "column_count_union": len(col_union),
        "missing_pcts_top20": dict(sorted(missing_pcts.items(), key=lambda x: x[1], reverse=True)[:20]),
    }

era_comparison = {
    "era_2000_2012": get_era_col_info(era_old_dfs),
    "era_2013_2025": get_era_col_info(era_new_dfs),
    "columns_only_in_old_era": sorted(
        set(get_era_col_info(era_old_dfs).get("columns_union", [])) -
        set(get_era_col_info(era_new_dfs).get("columns_union", []))
    ) if era_old_dfs and era_new_dfs else [],
    "columns_only_in_new_era": sorted(
        set(get_era_col_info(era_new_dfs).get("columns_union", [])) -
        set(get_era_col_info(era_old_dfs).get("columns_union", []))
    ) if era_old_dfs and era_new_dfs else [],
}


# ─────────────────────────────────────────────
# 17. Parquet analysis
# ─────────────────────────────────────────────
print("Step 17: Parquet analysis...")

parquet_analysis = {"exists": False}
if os.path.exists(PARQUET_PATH):
    try:
        pq = pd.read_parquet(PARQUET_PATH)
        parquet_analysis["exists"] = True
        parquet_analysis["shape"] = list(pq.shape)
        parquet_analysis["columns"] = list(pq.columns)
        parquet_analysis["column_count"] = len(pq.columns)
        parquet_analysis["row_count"] = len(pq)
        parquet_analysis["dtypes"] = {col: str(dtype) for col, dtype in pq.dtypes.items()}
        parquet_analysis["missing_pcts"] = {
            col: round(float(pq[col].isnull().mean() * 100), 2) for col in pq.columns
        }
        parquet_analysis["memory_mb"] = round(float(pq.memory_usage(deep=True).sum() / 1e6), 2)

        # Compare with raw CSV columns
        raw_cols = set(all_col_counter.keys())
        pq_cols = set(pq.columns)
        parquet_analysis["cols_in_parquet_not_raw"] = sorted(pq_cols - raw_cols)
        parquet_analysis["cols_in_raw_not_parquet"] = sorted(raw_cols - pq_cols)

        # Quality comparison
        if "Date" in pq.columns:
            pq_dates = pd.to_datetime(pq["Date"], errors="coerce")
            parquet_analysis["date_range"] = {
                "min": str(pq_dates.min()),
                "max": str(pq_dates.max()),
            }

        # Sample missing rates for key columns
        key_cols = ["Gls", "Ast", "Min", "xG", "xA", "Pos", "Result", "Date"]
        parquet_analysis["key_col_missing"] = {}
        for kc in key_cols:
            if kc in pq.columns:
                parquet_analysis["key_col_missing"][kc] = round(
                    float(pq[kc].isnull().mean() * 100), 2
                )

        # CSV vs Parquet shape comparison
        parquet_analysis["comparison_with_csv_sample"] = {
            "csv_total_files": total_files,
            "csv_sample_rows": len(combined),
            "parquet_rows": len(pq),
            "parquet_cols": len(pq.columns),
            "csv_sample_cols": len(data_cols),
        }

    except Exception as e:
        parquet_analysis["error"] = str(e)
        parquet_analysis["traceback"] = traceback.format_exc()


# ─────────────────────────────────────────────
# 18. Sample file structure (first 3 rows of 2 files)
# ─────────────────────────────────────────────
print("Step 18: Sample file structures...")

sample_structures = []
for season, team, fpath in sample_set[:5]:
    try:
        df_s = pd.read_csv(fpath, encoding="utf-8-sig", low_memory=False, nrows=3)
        sample_structures.append({
            "season": season,
            "team": team,
            "file": os.path.basename(fpath),
            "columns": list(df_s.columns),
            "shape": list(df_s.shape),
            "first_row": {str(k): str(v) for k, v in df_s.iloc[0].to_dict().items()} if len(df_s) > 0 else {},
        })
    except Exception as e:
        sample_structures.append({"file": fpath, "error": str(e)})


# ─────────────────────────────────────────────
# 19. Compile final report
# ─────────────────────────────────────────────
print("Step 19: Compiling report...")

report = {
    "meta": {
        "analysis_date": "2026-03-21",
        "base_dir": BASE_DIR,
        "parquet_path": PARQUET_PATH,
        "total_seasons": len(seasons),
        "seasons_list": seasons,
    },
    "file_inventory": {
        "total_matchlog_csv_files": total_files,
        "files_per_season": dict(sorted(files_per_season.items())),
        "files_per_team": dict(sorted(files_per_team.items(), key=lambda x: -x[1])),
        "top10_teams_by_file_count": dict(
            sorted(files_per_team.items(), key=lambda x: -x[1])[:10]
        ),
        "sample_size_analyzed": len(sample_set),
        "sample_load_errors": len(load_errors),
        "sample_encoding_issues": len(encoding_issues),
        "header_contaminated_files": len(header_contaminations),
    },
    "column_analysis": {
        "all_columns_found": dict(all_col_counter.most_common()),
        "total_unique_columns": len(all_col_counter),
        "early_era_columns_union_2000_2005": sorted(early_all),
        "recent_era_columns_union_2020_2025": sorted(recent_all),
        "columns_only_in_early_era": sorted(cols_only_in_early),
        "columns_only_in_recent_era": sorted(cols_only_in_recent),
        "columns_in_both_eras": sorted(cols_in_both),
        "early_always_present": sorted(early_common),
        "recent_always_present": sorted(recent_common),
    },
    "missing_value_analysis": {
        "top_20_missing_columns": {k: v for k, v in top_missing},
        "full_missing_report": missing_report,
    },
    "match_count_per_player": match_count_stats,
    "minutes_played_distribution": minutes_stats,
    "stat_distributions": stat_distributions,
    "start_vs_sub_analysis": start_sub_analysis,
    "result_column_analysis": result_analysis,
    "position_analysis": position_analysis,
    "date_coverage": date_analysis,
    "data_quality": {
        "duplicate_rows_in_sample": int(dup_count),
        "header_contaminated_files": header_contaminations[:20],
        "encoding_issue_files": encoding_issues[:20],
        "load_errors": load_errors[:20],
        "numeric_column_contamination": contamination_report,
        "dtype_summary": dtype_summary,
    },
    "column_stats": col_stats,
    "era_comparison": era_comparison,
    "parquet_analysis": parquet_analysis,
    "sample_file_structures": sample_structures,
}

# ─────────────────────────────────────────────
# 20. Save to JSON
# ─────────────────────────────────────────────
print("Step 20: Saving report to JSON...")
report = to_native(report)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, default=str, indent=2)

print(f"\n✓ Report saved to: {OUTPUT_PATH}")
print(f"  Total CSV files: {total_files}")
print(f"  Sample analyzed: {len(sample_set)} files")
print(f"  Combined rows: {len(combined)}")
print(f"  Unique columns found: {len(all_col_counter)}")
print(f"  Parquet exists: {parquet_analysis['exists']}")

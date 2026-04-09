"""
EPL Transfermarkt Data Preprocessing-EDA Analysis
Korean preprocessing-EDA report for squad_values CSVs
"""

import os
import glob
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Helper: convert numpy/pandas types to native Python
# ─────────────────────────────────────────────
def to_python(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    if isinstance(obj, (pd.Series,)):
        return obj.tolist()
    return str(obj)


def safe_json(obj):
    """Recursively convert a nested structure to JSON-safe types."""
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return safe_json(obj.tolist())
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


# ─────────────────────────────────────────────
# 1. Discover all squad_values.csv files
# ─────────────────────────────────────────────
BASE = r"C:/Users/xcv54/workspace/EPL project/data/raw/transfermarkt"
PARQUET = r"C:/Users/xcv54/workspace/EPL project/data/processed/player_season_stats.parquet"
REPORT = r"C:/Users/xcv54/workspace/EPL project/reports/analysis_transfermarkt.json"

all_csvs = sorted(glob.glob(os.path.join(BASE, "**", "squad_values.csv"), recursive=True))

print(f"Found {len(all_csvs)} squad_values.csv files")

# ─────────────────────────────────────────────
# 2. Parse season & team from path, read all CSVs
# ─────────────────────────────────────────────
dfs = []
file_errors = []
files_per_season = {}

for path in all_csvs:
    parts = Path(path).parts
    # path: .../transfermarkt/<season>/<team>/squad_values.csv
    try:
        season_idx = parts.index("transfermarkt") + 1
        season = parts[season_idx]
        team_folder = parts[season_idx + 1]
    except (ValueError, IndexError):
        season = "unknown"
        team_folder = "unknown"

    files_per_season[season] = files_per_season.get(season, 0) + 1

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["_source_season"] = season
        df["_source_team_folder"] = team_folder
        df["_source_file"] = path
        dfs.append(df)
    except Exception as e:
        file_errors.append({"file": path, "error": str(e)})

print(f"Successfully read: {len(dfs)} files | Errors: {len(file_errors)}")

# ─────────────────────────────────────────────
# 3. Combine into master DataFrame
# ─────────────────────────────────────────────
master = pd.concat(dfs, ignore_index=True)
print(f"Master shape: {master.shape}")

# ─────────────────────────────────────────────
# 4. Column names and dtypes
# ─────────────────────────────────────────────
columns_info = {col: str(master[col].dtype) for col in master.columns}

# ─────────────────────────────────────────────
# 5. Missing values
# ─────────────────────────────────────────────
total_rows = len(master)
missing_info = {}
for col in master.columns:
    cnt = int(master[col].isna().sum())
    missing_info[col] = {
        "missing_count": cnt,
        "missing_pct": round(cnt / total_rows * 100, 2) if total_rows else 0.0,
    }

# ─────────────────────────────────────────────
# 6. Market value analysis
# ─────────────────────────────────────────────
mv_col = "market_value"

# Separate zero vs NaN
mv_zero_count = int((master[mv_col] == 0).sum()) if mv_col in master.columns else 0
mv_null_count = int(master[mv_col].isna().sum()) if mv_col in master.columns else 0
mv_nonzero = master[master[mv_col] > 0][mv_col] if mv_col in master.columns else pd.Series(dtype=float)

# Overall distribution
mv_dist = {
    "total_records": total_rows,
    "zero_market_value_count": mv_zero_count,
    "zero_market_value_pct": round(mv_zero_count / total_rows * 100, 2),
    "null_market_value_count": mv_null_count,
    "nonzero_records": int(len(mv_nonzero)),
    "min": float(mv_nonzero.min()) if len(mv_nonzero) else None,
    "max": float(mv_nonzero.max()) if len(mv_nonzero) else None,
    "mean": float(mv_nonzero.mean()) if len(mv_nonzero) else None,
    "median": float(mv_nonzero.median()) if len(mv_nonzero) else None,
    "std": float(mv_nonzero.std()) if len(mv_nonzero) else None,
    "percentiles": {
        "25th": float(mv_nonzero.quantile(0.25)) if len(mv_nonzero) else None,
        "75th": float(mv_nonzero.quantile(0.75)) if len(mv_nonzero) else None,
        "90th": float(mv_nonzero.quantile(0.90)) if len(mv_nonzero) else None,
        "95th": float(mv_nonzero.quantile(0.95)) if len(mv_nonzero) else None,
        "99th": float(mv_nonzero.quantile(0.99)) if len(mv_nonzero) else None,
    },
}

# Per-season market value stats
season_col = "season" if "season" in master.columns else "_source_season"
mv_per_season = {}
for s, grp in master.groupby(season_col):
    nz = grp[grp[mv_col] > 0][mv_col] if mv_col in grp.columns else pd.Series(dtype=float)
    mv_per_season[str(s)] = {
        "players": int(len(grp)),
        "nonzero_mv_players": int(len(nz)),
        "mean_mv": float(nz.mean()) if len(nz) else None,
        "median_mv": float(nz.median()) if len(nz) else None,
        "total_mv": float(nz.sum()) if len(nz) else None,
        "max_mv": float(nz.max()) if len(nz) else None,
        "min_mv": float(nz.min()) if len(nz) else None,
    }

# Top 20 most valuable players ever
if mv_col in master.columns:
    top20_mv = (
        master.nlargest(20, mv_col)[
            [c for c in ["player", "team", "season", mv_col, "position", "nationality"] if c in master.columns]
        ]
        .reset_index(drop=True)
        .to_dict(orient="records")
    )
else:
    top20_mv = []

# Market value trend (average & total per season for nonzero)
mv_trend = {}
for s, stats in mv_per_season.items():
    mv_trend[s] = {"avg_mv": stats["mean_mv"], "total_mv": stats["total_mv"]}

# ─────────────────────────────────────────────
# 7. DOB / Age analysis
# ─────────────────────────────────────────────
age_analysis = {}
if "age" in master.columns:
    ages = pd.to_numeric(master["age"], errors="coerce").dropna()
    age_analysis["age_stats"] = {
        "min": float(ages.min()) if len(ages) else None,
        "max": float(ages.max()) if len(ages) else None,
        "mean": round(float(ages.mean()), 2) if len(ages) else None,
        "median": float(ages.median()) if len(ages) else None,
        "std": round(float(ages.std()), 2) if len(ages) else None,
    }
    age_bins = pd.cut(ages, bins=[0, 18, 21, 24, 27, 30, 33, 36, 100],
                      labels=["U18", "18-20", "21-23", "24-26", "27-29", "30-32", "33-35", "36+"])
    age_dist = age_bins.value_counts().sort_index().to_dict()
    age_analysis["age_distribution_bins"] = {str(k): int(v) for k, v in age_dist.items()}

    missing_age = int(master["age"].isna().sum())
    age_analysis["missing_age_count"] = missing_age

if "dob" in master.columns:
    missing_dob = int(master["dob"].isna().sum())
    age_analysis["missing_dob_count"] = missing_dob

    # Parse DOB, find youngest/oldest
    master["_dob_parsed"] = pd.to_datetime(master["dob"], format="%d/%m/%Y", errors="coerce")
    valid_dob = master[master["_dob_parsed"].notna()]

    if len(valid_dob):
        # Oldest = earliest DOB
        oldest_idx = valid_dob["_dob_parsed"].idxmin()
        youngest_idx = valid_dob["_dob_parsed"].idxmax()
        oldest_row = valid_dob.loc[oldest_idx]
        youngest_row = valid_dob.loc[youngest_idx]

        age_analysis["oldest_player"] = {
            "name": str(oldest_row.get("player", "")),
            "dob": str(oldest_row.get("dob", "")),
            "team": str(oldest_row.get("team", "")),
            "season": str(oldest_row.get(season_col, "")),
        }
        age_analysis["youngest_player"] = {
            "name": str(youngest_row.get("player", "")),
            "dob": str(youngest_row.get("dob", "")),
            "team": str(youngest_row.get("team", "")),
            "season": str(youngest_row.get(season_col, "")),
        }

# ─────────────────────────────────────────────
# 8. Nationality distribution (top 20)
# ─────────────────────────────────────────────
nat_analysis = {}
if "nationality" in master.columns:
    nat_counts = master["nationality"].value_counts()
    nat_analysis["top_20"] = {str(k): int(v) for k, v in nat_counts.head(20).items()}
    nat_analysis["unique_nationalities"] = int(nat_counts.nunique())
    nat_analysis["missing_nationality"] = int(master["nationality"].isna().sum())

# ─────────────────────────────────────────────
# 9. Position distribution
# ─────────────────────────────────────────────
pos_analysis = {}
if "position" in master.columns:
    pos_counts = master["position"].value_counts()
    pos_analysis["distribution"] = {str(k): int(v) for k, v in pos_counts.items()}
    pos_analysis["unique_positions"] = int(pos_counts.nunique())
    pos_analysis["missing_position"] = int(master["position"].isna().sum())

# ─────────────────────────────────────────────
# 10. Height analysis
# ─────────────────────────────────────────────
height_analysis = {}
if "height" in master.columns:
    # Height stored as "1,85m" format
    def parse_height(h):
        if pd.isna(h):
            return np.nan
        h = str(h).replace(",", ".").replace("m", "").strip()
        try:
            return float(h)
        except ValueError:
            return np.nan

    master["_height_m"] = master["height"].apply(parse_height)
    h_valid = master["_height_m"].dropna()

    height_analysis["records_with_height"] = int(len(h_valid))
    height_analysis["missing_height_count"] = int(master["_height_m"].isna().sum())
    height_analysis["missing_height_pct"] = round(
        master["_height_m"].isna().sum() / total_rows * 100, 2
    )
    if len(h_valid):
        height_analysis["min_m"] = round(float(h_valid.min()), 2)
        height_analysis["max_m"] = round(float(h_valid.max()), 2)
        height_analysis["mean_m"] = round(float(h_valid.mean()), 3)
        height_analysis["median_m"] = round(float(h_valid.median()), 2)
        height_analysis["std_m"] = round(float(h_valid.std()), 3)

        # Bins
        h_bins = pd.cut(h_valid, bins=[1.5, 1.65, 1.70, 1.75, 1.80, 1.85, 1.90, 1.95, 2.10],
                        labels=["≤165", "166-170", "171-175", "176-180",
                                "181-185", "186-190", "191-195", ">195"])
        h_dist = h_bins.value_counts().sort_index().to_dict()
        height_analysis["height_distribution_bins_cm"] = {str(k): int(v) for k, v in h_dist.items()}

# ─────────────────────────────────────────────
# 11. Joined date analysis
# ─────────────────────────────────────────────
joined_analysis = {}
if "joined" in master.columns:
    master["_joined_parsed"] = pd.to_datetime(master["joined"], format="%d/%m/%Y", errors="coerce")
    j_valid = master["_joined_parsed"].dropna()

    joined_analysis["missing_joined_count"] = int(master["_joined_parsed"].isna().sum())
    joined_analysis["missing_joined_pct"] = round(
        master["_joined_parsed"].isna().sum() / total_rows * 100, 2
    )
    if len(j_valid):
        joined_analysis["earliest_join_date"] = str(j_valid.min().date())
        joined_analysis["latest_join_date"] = str(j_valid.max().date())

        # Year distribution
        year_dist = j_valid.dt.year.value_counts().sort_index().to_dict()
        joined_analysis["join_year_distribution"] = {str(k): int(v) for k, v in year_dist.items()}

# ─────────────────────────────────────────────
# 12. Foot preference distribution
# ─────────────────────────────────────────────
foot_analysis = {}
if "foot" in master.columns:
    foot_counts = master["foot"].value_counts(dropna=False)
    foot_analysis["distribution"] = {str(k): int(v) for k, v in foot_counts.items()}
    foot_analysis["missing_foot_count"] = int(master["foot"].isna().sum())
    foot_analysis["missing_foot_pct"] = round(
        master["foot"].isna().sum() / total_rows * 100, 2
    )

# ─────────────────────────────────────────────
# 13. Players per team per season
# ─────────────────────────────────────────────
team_season_counts = {}
if "team" in master.columns:
    ts_grp = master.groupby(["team", season_col]).size()
    for (team, season), count in ts_grp.items():
        if str(season) not in team_season_counts:
            team_season_counts[str(season)] = {}
        team_season_counts[str(season)][str(team)] = int(count)

# ─────────────────────────────────────────────
# 14. Data quality
# ─────────────────────────────────────────────
# Duplicates
dup_subset = [c for c in ["player", "team", "season"] if c in master.columns]
dup_count = int(master.duplicated(subset=dup_subset).sum()) if dup_subset else 0

# Encoding issues: check for typical garbled characters
encoding_issues = int(master.select_dtypes(include="object").apply(
    lambda col: col.str.contains(r"[ï¿½â€]", regex=True, na=False).sum()
).sum())

dq = {
    "total_rows": total_rows,
    "total_columns": len(master.columns),
    "duplicate_player_team_season": dup_count,
    "encoding_issue_cells_approx": encoding_issues,
    "zero_market_value_count": mv_zero_count,
    "null_market_value_count": mv_null_count,
    "missing_dob_count": int(master["dob"].isna().sum()) if "dob" in master.columns else None,
    "missing_nationality_count": int(master["nationality"].isna().sum()) if "nationality" in master.columns else None,
    "missing_position_count": int(master["position"].isna().sum()) if "position" in master.columns else None,
    "file_read_errors": file_errors,
}

# ─────────────────────────────────────────────
# 15. Market value by position
# ─────────────────────────────────────────────
mv_by_pos = {}
if "position" in master.columns and mv_col in master.columns:
    nz_pos = master[master[mv_col] > 0]
    for pos, grp in nz_pos.groupby("position"):
        vals = grp[mv_col]
        mv_by_pos[str(pos)] = {
            "count": int(len(vals)),
            "mean_mv": round(float(vals.mean()), 2),
            "median_mv": float(vals.median()),
            "max_mv": float(vals.max()),
            "total_mv": float(vals.sum()),
        }
    # Sort by mean_mv descending
    mv_by_pos = dict(sorted(mv_by_pos.items(), key=lambda x: x[1]["mean_mv"], reverse=True))

# ─────────────────────────────────────────────
# 16. Most expensive teams per season
# ─────────────────────────────────────────────
expensive_teams_per_season = {}
if "team" in master.columns and mv_col in master.columns:
    for s, grp in master.groupby(season_col):
        nz = grp[grp[mv_col] > 0]
        if len(nz) == 0:
            continue
        team_totals = nz.groupby("team")[mv_col].sum().sort_values(ascending=False)
        top5 = team_totals.head(5)
        expensive_teams_per_season[str(s)] = {
            str(team): float(val) for team, val in top5.items()
        }

# ─────────────────────────────────────────────
# 17. Processed parquet analysis (if exists)
# ─────────────────────────────────────────────
parquet_analysis = {}
parquet_exists = os.path.exists(PARQUET)
parquet_analysis["file_exists"] = parquet_exists

if parquet_exists:
    try:
        pq = pd.read_parquet(PARQUET)
        parquet_analysis["shape"] = list(pq.shape)
        parquet_analysis["columns"] = list(pq.columns)
        parquet_analysis["dtypes"] = {col: str(pq[col].dtype) for col in pq.columns}

        # Missing values in parquet
        pq_missing = {}
        for col in pq.columns:
            cnt = int(pq[col].isna().sum())
            pq_missing[col] = {
                "missing_count": cnt,
                "missing_pct": round(cnt / len(pq) * 100, 2),
            }
        parquet_analysis["missing_values"] = pq_missing

        # TM columns detection
        tm_cols = [c for c in pq.columns if any(x in c.lower() for x in
                   ["market_value", "mv", "nationality", "foot", "height_cm", "dob", "joined"])]
        fbref_cols = [c for c in pq.columns if any(x in c.lower() for x in
                      ["xg", "npxg", "kp", "progressive", "90s", "gca", "sca"])]
        parquet_analysis["detected_tm_columns"] = tm_cols
        parquet_analysis["detected_fbref_columns"] = fbref_cols[:20]

        # Match rate: rows where TM market value is not null vs total
        for mv_c in ["market_value", "market_value_eur", "market_value_gbp"]:
            if mv_c in pq.columns:
                matched = int(pq[mv_c].notna().sum())
                parquet_analysis[f"tm_match_rate_{mv_c}"] = {
                    "matched": matched,
                    "total": len(pq),
                    "match_pct": round(matched / len(pq) * 100, 2),
                }

        # Seasons in parquet
        for sc in ["season", "Season", "season_id"]:
            if sc in pq.columns:
                parquet_analysis["seasons_in_parquet"] = sorted(pq[sc].dropna().unique().astype(str).tolist())
                break

        # Sample of top market value players in parquet
        for mv_c in ["market_value", "market_value_eur"]:
            if mv_c in pq.columns:
                nz_pq = pq[pq[mv_c] > 0] if mv_c in pq.columns else pq
                if len(nz_pq):
                    name_col = next((c for c in ["player", "Player", "name"] if c in nz_pq.columns), None)
                    top_cols = [c for c in [name_col, "team", "season", mv_c] if c]
                    top5_pq = nz_pq.nlargest(5, mv_c)[top_cols].to_dict(orient="records")
                    parquet_analysis["top5_by_mv_in_parquet"] = top5_pq
                break

    except Exception as e:
        parquet_analysis["error"] = str(e)

# ─────────────────────────────────────────────
# 18. Summary stats
# ─────────────────────────────────────────────
seasons_list = sorted(files_per_season.keys())
summary = {
    "total_csv_files_found": len(all_csvs),
    "files_read_successfully": len(dfs),
    "files_with_read_errors": len(file_errors),
    "seasons_covered": seasons_list,
    "total_seasons": len(seasons_list),
    "files_per_season": {k: files_per_season[k] for k in sorted(files_per_season)},
    "total_rows_combined": total_rows,
    "total_columns": len(master.columns),
    "unique_teams": int(master["team"].nunique()) if "team" in master.columns else None,
    "unique_players": int(master["player"].nunique()) if "player" in master.columns else None,
}

# ─────────────────────────────────────────────
# Compile final report
# ─────────────────────────────────────────────
report = {
    "report_generated_at": datetime.now().isoformat(),
    "data_source": BASE,
    "summary": summary,
    "columns_and_dtypes": columns_info,
    "missing_values_per_column": missing_info,
    "market_value_analysis": {
        "overall_distribution": mv_dist,
        "per_season_stats": mv_per_season,
        "top_20_most_valuable_players_ever": top20_mv,
        "market_value_trend_by_season": mv_trend,
        "market_value_by_position": mv_by_pos,
    },
    "age_and_dob_analysis": age_analysis,
    "nationality_analysis": nat_analysis,
    "position_analysis": pos_analysis,
    "height_analysis": height_analysis,
    "joined_date_analysis": joined_analysis,
    "foot_preference_analysis": foot_analysis,
    "players_per_team_per_season": team_season_counts,
    "most_expensive_teams_per_season": expensive_teams_per_season,
    "data_quality": dq,
    "processed_parquet_analysis": parquet_analysis,
}

# ─────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────
os.makedirs(os.path.dirname(REPORT), exist_ok=True)
with open(REPORT, "w", encoding="utf-8") as f:
    json.dump(safe_json(report), f, ensure_ascii=False, indent=2, default=str)

print(f"\nReport saved to: {REPORT}")
print("Done.")

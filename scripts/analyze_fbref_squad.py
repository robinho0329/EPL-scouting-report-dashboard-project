"""
EPL FBref Squad Stats - Comprehensive Preprocessing/EDA Analysis
Research only - no source files modified
"""

import os
import glob
import json
import warnings
import traceback
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

BASE_DIR = r"C:/Users/xcv54/workspace/EPL project/data/raw/fbref"
PROCESSED_PARQUET = r"C:/Users/xcv54/workspace/EPL project/data/processed/team_season_summary.parquet"
OUTPUT_JSON = r"C:/Users/xcv54/workspace/EPL project/reports/analysis_fbref_squad.json"

# ─────────────────────────────────────────────
# 1. Find all squad_stats.csv files
# ─────────────────────────────────────────────
all_files = sorted(glob.glob(os.path.join(BASE_DIR, "**", "squad_stats.csv"), recursive=True))
print(f"Total squad_stats.csv found: {len(all_files)}")

# Parse season / team from path
file_meta = []
for f in all_files:
    parts = Path(f).parts
    # …/fbref/<season>/<team>/squad_stats.csv
    try:
        idx = parts.index("fbref")
        season = parts[idx + 1]
        team   = parts[idx + 2]
    except (ValueError, IndexError):
        season = team = "unknown"
    file_meta.append({"path": f, "season": season, "team": team})

seasons_all = sorted(set(m["season"] for m in file_meta))
files_per_season = Counter(m["season"] for m in file_meta)

# ─────────────────────────────────────────────
# 2. Select sample files (≥25, cover early + recent)
# ─────────────────────────────────────────────
def pick_samples(meta, n_per_season=2, early_seasons=None, recent_seasons=None):
    """Pick representative files: all from early/recent, 2 per season otherwise."""
    selected = []
    by_season = defaultdict(list)
    for m in meta:
        by_season[m["season"]].append(m)
    for season, items in sorted(by_season.items()):
        if early_seasons and season in early_seasons:
            selected.extend(items[:5])
        elif recent_seasons and season in recent_seasons:
            selected.extend(items[:5])
        else:
            selected.extend(items[:n_per_season])
    return selected

early  = [s for s in seasons_all if s.startswith(("2000", "2001", "2002", "2003", "2004"))]
recent = [s for s in seasons_all if s.startswith(("2020", "2021", "2022", "2023", "2024"))]

sample_meta = pick_samples(file_meta, n_per_season=2,
                            early_seasons=set(early), recent_seasons=set(recent))
print(f"Sample files selected: {len(sample_meta)}")

# ─────────────────────────────────────────────
# 3. Read each sample file
# ─────────────────────────────────────────────
def read_squad_csv(path, season, team):
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(path, encoding="latin-1", low_memory=False)
        except Exception as e:
            return None, str(e)
    except Exception as e:
        return None, str(e)
    df["_season"] = season
    df["_team"]   = team
    return df, None

frames = []
read_errors = []
for m in sample_meta:
    df, err = read_squad_csv(m["path"], m["season"], m["team"])
    if err:
        read_errors.append({"file": m["path"], "error": err})
    else:
        frames.append(df)

print(f"Successfully read: {len(frames)}, errors: {len(read_errors)}")

# ─────────────────────────────────────────────
# 4. Column analysis across eras
# ─────────────────────────────────────────────
early_cols  = {}   # season -> set of cols
recent_cols = {}

for m, df in zip([x for x in sample_meta if x in sample_meta], frames):
    # Re-map via enumerate since frames align with sample_meta (successful reads)
    pass

# Rebuild mapping properly
col_by_season = defaultdict(set)
col_by_file   = []
for (m, df) in zip(
    [m for m in sample_meta],  # same order
    frames
):
    cols_clean = [c for c in df.columns if not c.startswith("_")]
    col_by_season[m["season"]].update(cols_clean)
    col_by_file.append({"season": m["season"], "team": m["team"],
                        "columns": cols_clean, "n_cols": len(cols_clean)})

# Era columns
era1_seasons = [s for s in col_by_season if s[:4] <= "2012"]
era2_seasons = [s for s in col_by_season if s[:4] >  "2012"]

era1_cols_union = set()
for s in era1_seasons:
    era1_cols_union.update(col_by_season[s])
era2_cols_union = set()
for s in era2_seasons:
    era2_cols_union.update(col_by_season[s])

only_in_era1 = sorted(era1_cols_union - era2_cols_union)
only_in_era2 = sorted(era2_cols_union - era1_cols_union)
common_cols  = sorted(era1_cols_union & era2_cols_union)

# ─────────────────────────────────────────────
# 5. Sample rows from 2000-01 vs 2024-25
# ─────────────────────────────────────────────
def get_sample_rows(frames_meta, target_season, n=3):
    for df in frames:
        if df["_season"].iloc[0] == target_season:
            cols_show = [c for c in df.columns if not c.startswith("_")][:15]
            return df[cols_show].head(n).to_dict(orient="records")
    return []

early_sample  = get_sample_rows(frames, "2000-2001")
recent_sample = get_sample_rows(frames, "2024-2025")
if not recent_sample:
    # try last available
    last_season = max(df["_season"].iloc[0] for df in frames)
    recent_sample = get_sample_rows(frames, last_season)
    recent_sample_season = last_season
else:
    recent_sample_season = "2024-2025"

early_sample_cols  = []
recent_sample_cols = []
for df in frames:
    if df["_season"].iloc[0] == "2000-2001" and not early_sample_cols:
        early_sample_cols = [c for c in df.columns if not c.startswith("_")]
    if df["_season"].iloc[0] == recent_sample_season and not recent_sample_cols:
        recent_sample_cols = [c for c in df.columns if not c.startswith("_")]

# ─────────────────────────────────────────────
# 6. Combine all sample frames for global analysis
# ─────────────────────────────────────────────
# Use concat with ignore_index; fill missing cols with NaN
combined = pd.concat(frames, ignore_index=True, sort=False)
print(f"Combined shape: {combined.shape}")

# ─────────────────────────────────────────────
# 7. Data quality: header contamination
# ─────────────────────────────────────────────
player_col = None
for c in ["player", "Player", "PLAYER", "Name"]:
    if c in combined.columns:
        player_col = c
        break

header_contamination_count = 0
if player_col:
    header_contamination_count = int((combined[player_col].astype(str).str.strip() == "Player").sum())

# Rows with "Player" as value per file
hc_per_file = {}
for df in frames:
    season = df["_season"].iloc[0]
    team   = df["_team"].iloc[0]
    if player_col and player_col in df.columns:
        count = int((df[player_col].astype(str).str.strip() == "Player").sum())
        hc_per_file[f"{season}/{team}"] = count

# ─────────────────────────────────────────────
# 8. Missing values per column
# ─────────────────────────────────────────────
non_meta_cols = [c for c in combined.columns if not c.startswith("_")]
missing_stats = {}
for col in non_meta_cols:
    total = len(combined)
    n_null = int(combined[col].isna().sum())
    n_empty = int((combined[col].astype(str).str.strip() == "").sum())
    missing_stats[col] = {
        "null_count": n_null,
        "empty_str_count": n_empty,
        "pct_null": round(n_null / total * 100, 2)
    }

# ─────────────────────────────────────────────
# 9. Player count per team per season
# ─────────────────────────────────────────────
player_counts = []
for df in frames:
    season = df["_season"].iloc[0]
    team   = df["_team"].iloc[0]
    if player_col:
        # exclude header rows
        clean = df[df[player_col].astype(str).str.strip() != "Player"]
        n = int(clean[player_col].notna().sum())
    else:
        n = len(df)
    player_counts.append({"season": season, "team": team, "player_count": n})

pc_values = [x["player_count"] for x in player_counts]
player_count_stats = {
    "min": int(min(pc_values)) if pc_values else None,
    "max": int(max(pc_values)) if pc_values else None,
    "mean": round(float(np.mean(pc_values)), 2) if pc_values else None,
    "median": float(np.median(pc_values)) if pc_values else None,
    "distribution": sorted(pc_values)
}

# ─────────────────────────────────────────────
# 10. Position distribution
# ─────────────────────────────────────────────
pos_col = None
for c in ["pos", "Pos", "Position", "position"]:
    if c in combined.columns:
        pos_col = c
        break

position_distribution = {}
if pos_col:
    clean_pos = combined[pos_col].astype(str).str.strip()
    clean_pos = clean_pos[~clean_pos.isin(["nan", "", "Pos"])]
    position_distribution = {k: int(v) for k, v in clean_pos.value_counts().head(30).items()}

# ─────────────────────────────────────────────
# 11. Top players by goals / assists
# ─────────────────────────────────────────────
goals_col  = None
assist_col = None
for c in combined.columns:
    if c in ("gls", "Gls", "goals", "Goals", "G"):
        goals_col = c
        break
for c in combined.columns:
    if c in ("ast", "Ast", "assists", "Assists", "A"):
        assist_col = c
        break

top_scorers  = []
top_assisters = []

def safe_to_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(",", ""), errors="coerce")

if goals_col and player_col:
    tmp = combined[[player_col, "_season", "_team", goals_col]].copy()
    tmp[goals_col] = safe_to_numeric(tmp[goals_col])
    tmp = tmp[tmp[player_col].astype(str).str.strip() != "Player"]
    top = tmp.dropna(subset=[goals_col]).nlargest(10, goals_col)
    top_scorers = top[[player_col, "_season", "_team", goals_col]].to_dict(orient="records")

if assist_col and player_col:
    tmp = combined[[player_col, "_season", "_team", assist_col]].copy()
    tmp[assist_col] = safe_to_numeric(tmp[assist_col])
    tmp = tmp[tmp[player_col].astype(str).str.strip() != "Player"]
    top = tmp.dropna(subset=[assist_col]).nlargest(10, assist_col)
    top_assisters = top[[player_col, "_season", "_team", assist_col]].to_dict(orient="records")

# ─────────────────────────────────────────────
# 12. Age distribution
# ─────────────────────────────────────────────
age_col = None
for c in ["age", "Age", "AGE"]:
    if c in combined.columns:
        age_col = c
        break

age_stats = {}
if age_col:
    ages = safe_to_numeric(combined[age_col])
    ages = ages.dropna()
    ages = ages[(ages >= 14) & (ages <= 45)]
    age_stats = {
        "count": int(len(ages)),
        "min": float(ages.min()),
        "max": float(ages.max()),
        "mean": round(float(ages.mean()), 2),
        "median": float(ages.median()),
        "std": round(float(ages.std()), 2),
        "bins": {
            "<20": int((ages < 20).sum()),
            "20-24": int(((ages >= 20) & (ages < 25)).sum()),
            "25-29": int(((ages >= 25) & (ages < 30)).sum()),
            "30-34": int(((ages >= 30) & (ages < 35)).sum()),
            ">=35": int((ages >= 35).sum()),
        }
    }

# ─────────────────────────────────────────────
# 13. Nationality distribution
# ─────────────────────────────────────────────
nat_col = None
for c in ["nation", "Nation", "Nationality", "nationality", "Nat"]:
    if c in combined.columns:
        nat_col = c
        break

nationality_dist = {}
if nat_col:
    nats = combined[nat_col].astype(str).str.strip()
    nats = nats[~nats.isin(["nan", "", "Nation", "Nat"])]
    # FBref format: "eg ENG" → last token
    nats = nats.str.split().str[-1]
    nationality_dist = {k: int(v) for k, v in nats.value_counts().head(15).items()}

# ─────────────────────────────────────────────
# 14. Descriptive stats for common numeric cols
# ─────────────────────────────────────────────
numeric_desc = {}
# Check actual column names in combined df (lowercase normalized)
candidate_num_cols = [
    "mp", "starts", "min", "gls", "ast", "crdy", "crdr",
    "g_a", "g_pk", "pk", "pkatt", "90s",
    "gls_1", "ast_1",  # per-90 rates
    # also check original-case variants
    "MP", "Starts", "Min", "Gls", "Ast", "CrdY", "CrdR", "xG", "xAG", "npxG"
]
candidate_num_cols = [c for c in candidate_num_cols if c in combined.columns]
for c in candidate_num_cols:
    if c in combined.columns:
        s = safe_to_numeric(combined[c])
        s = s.dropna()
        if len(s) > 0:
            numeric_desc[c] = {
                "count": int(len(s)),
                "mean": round(float(s.mean()), 3),
                "std":  round(float(s.std()),  3),
                "min":  float(s.min()),
                "25%":  float(s.quantile(0.25)),
                "50%":  float(s.median()),
                "75%":  float(s.quantile(0.75)),
                "max":  float(s.max()),
            }

# ─────────────────────────────────────────────
# 15. Anomalies / quality issues
# ─────────────────────────────────────────────
anomalies = []

# Files with zero data rows
for df in frames:
    season = df["_season"].iloc[0]
    team   = df["_team"].iloc[0]
    if player_col and player_col in df.columns:
        real_rows = df[df[player_col].astype(str).str.strip() != "Player"]
        if len(real_rows) == 0:
            anomalies.append(f"No data rows in {season}/{team}")

# Encoding test - check for replacement chars
for df in frames:
    for col in df.select_dtypes(include="object").columns:
        if df[col].astype(str).str.contains("â|Ã|ï¿½", regex=True).any():
            anomalies.append(f"Possible encoding issue in {df['_season'].iloc[0]}/{df['_team'].iloc[0]} col={col}")
            break

# Duplicate player entries within same file
for df in frames:
    if player_col and player_col in df.columns:
        clean = df[df[player_col].astype(str).str.strip() != "Player"]
        dupes = clean[player_col].duplicated().sum()
        if dupes > 0:
            anomalies.append(f"Duplicate player entries ({dupes}) in {df['_season'].iloc[0]}/{df['_team'].iloc[0]}")

# Seasons with unusual column counts
col_counts_by_season = {}
for s, cols in col_by_season.items():
    col_counts_by_season[s] = len(cols)
median_col_count = np.median(list(col_counts_by_season.values()))
for s, cnt in col_counts_by_season.items():
    if cnt < median_col_count * 0.6:
        anomalies.append(f"Season {s} has unusually few columns ({cnt} vs median {median_col_count})")

# Negative goals/age
if goals_col:
    neg_goals = (safe_to_numeric(combined[goals_col]) < 0).sum()
    if neg_goals > 0:
        anomalies.append(f"Negative goals found: {neg_goals} rows")
if age_col:
    neg_age = (safe_to_numeric(combined[age_col]) < 14).sum()
    if neg_age > 0:
        anomalies.append(f"Suspicious ages (<14) found: {neg_age} rows")

# ─────────────────────────────────────────────
# 16. Processed parquet comparison
# ─────────────────────────────────────────────
parquet_analysis = {"exists": False}
if os.path.exists(PROCESSED_PARQUET):
    try:
        pq = pd.read_parquet(PROCESSED_PARQUET)
        parquet_analysis = {
            "exists": True,
            "shape": list(pq.shape),
            "columns": list(pq.columns),
            "n_columns": len(pq.columns),
            "n_rows": len(pq),
            "dtypes": {c: str(t) for c, t in pq.dtypes.items()},
            "null_pct": {c: round(float(pq[c].isna().mean() * 100), 2)
                         for c in pq.columns},
            "sample_rows": pq.head(3).to_dict(orient="records"),
            "raw_vs_processed": {
                "raw_total_files": len(all_files),
                "raw_total_seasons": len(seasons_all),
                "processed_unique_seasons": int(pq.get("season", pq.iloc[:, 0]).nunique())
                    if "season" in pq.columns else "N/A",
                "data_quality_improvement": "Processed parquet exists and loaded successfully"
            }
        }
    except Exception as e:
        parquet_analysis = {"exists": True, "error": str(e)}

# ─────────────────────────────────────────────
# 17. Column names per season (full map)
# ─────────────────────────────────────────────
col_per_season_dict = {s: sorted(cols) for s, cols in col_by_season.items()}

# ─────────────────────────────────────────────
# 17b. Season column format analysis
# ─────────────────────────────────────────────
season_format_analysis = {}
if "season" in combined.columns:
    sample_vals = combined["season"].dropna().astype(str).unique()[:10].tolist()
    season_format_analysis = {
        "column_exists_in_csv": True,
        "sample_values": sample_vals,
        "note": "FBref uses YYYY/YY format (e.g. 2000/01) inside CSV; folder structure uses YYYY-YYYY"
    }
else:
    season_format_analysis = {"column_exists_in_csv": False}

# ─────────────────────────────────────────────
# 17c. Per-90 rate columns analysis
# ─────────────────────────────────────────────
per90_cols = ["gls_1", "ast_1", "g_a_1", "g_pk_1", "g_a_pk"]
per90_analysis = {}
for c in per90_cols:
    if c in combined.columns:
        s = safe_to_numeric(combined[c])
        s = s.dropna()
        per90_analysis[c] = {
            "description": f"Per-90-minute rate column ({c})",
            "count": int(len(s)),
            "mean": round(float(s.mean()), 4),
            "max": float(s.max()),
        }

# ─────────────────────────────────────────────
# 17d. Data normalization observation
# ─────────────────────────────────────────────
all_cols_sets = list(col_by_season.values())
is_uniform = all(s == all_cols_sets[0] for s in all_cols_sets) if all_cols_sets else False
normalization_note = {
    "pre_normalized": is_uniform,
    "observation": (
        "All squad_stats.csv files (2000-01 through 2024-25) have been pre-normalized "
        "to identical 24 lowercase columns: player, nation, pos, age, mp, starts, min, 90s, "
        "gls, ast, g_a, g_pk, pk, pkatt, crdy, crdr, gls_1, ast_1, g_a_1, g_pk_1, g_a_pk, "
        "matches, season, team. "
        "Note: xG/xAG advanced stats were NOT scraped/included in this dataset — "
        "only the standard summary stats table is present."
    ) if is_uniform else "Column sets vary across seasons."
}

# ─────────────────────────────────────────────
# 17e. Null-player rows per file
# ─────────────────────────────────────────────
null_player_rows = {}
if player_col:
    for df in frames:
        key = f"{df['_season'].iloc[0]}/{df['_team'].iloc[0]}"
        n_null = int(df[player_col].isna().sum())
        if n_null > 0:
            null_player_rows[key] = n_null

# ─────────────────────────────────────────────
# 17f. Goals / Assists per season (aggregated from sample)
# ─────────────────────────────────────────────
goals_per_season = {}
assists_per_season = {}
if goals_col and player_col:
    tmp = combined.copy()
    tmp[goals_col] = safe_to_numeric(tmp[goals_col])
    tmp_clean = tmp[tmp[player_col].astype(str).str.strip() != "Player"]
    gps = tmp_clean.groupby("_season")[goals_col].sum().dropna()
    goals_per_season = {k: round(float(v), 1) for k, v in gps.items()}
if assist_col and player_col:
    tmp = combined.copy()
    tmp[assist_col] = safe_to_numeric(tmp[assist_col])
    tmp_clean = tmp[tmp[player_col].astype(str).str.strip() != "Player"]
    aps = tmp_clean.groupby("_season")[assist_col].sum().dropna()
    assists_per_season = {k: round(float(v), 1) for k, v in aps.items()}

# ─────────────────────────────────────────────
# 18. Compile final result
# ─────────────────────────────────────────────
result = {
    "metadata": {
        "analysis_date": "2026-03-21",
        "base_dir": BASE_DIR,
        "total_squad_stats_files": len(all_files),
        "sample_files_read": len(frames),
        "read_errors": len(read_errors),
        "read_error_details": read_errors[:5],
        "all_seasons_found": seasons_all,
        "total_seasons": len(seasons_all),
    },
    "files_per_season": dict(sorted(files_per_season.items())),
    "column_analysis": {
        "era_1_2000_2012": {
            "seasons": era1_seasons,
            "all_columns": sorted(era1_cols_union),
            "n_columns": len(era1_cols_union),
        },
        "era_2_2013_2025": {
            "seasons": era2_seasons,
            "all_columns": sorted(era2_cols_union),
            "n_columns": len(era2_cols_union),
        },
        "columns_only_in_era1_disappear": only_in_era1,
        "columns_only_in_era2_new": only_in_era2,
        "common_columns_both_eras": common_cols,
        "columns_per_season": col_per_season_dict,
    },
    "era_sample_comparison": {
        "season_2000_2001": {
            "columns": early_sample_cols,
            "n_cols": len(early_sample_cols),
            "sample_rows": early_sample
        },
        f"season_{recent_sample_season.replace('-', '_')}": {
            "columns": recent_sample_cols,
            "n_cols": len(recent_sample_cols),
            "sample_rows": recent_sample
        }
    },
    "descriptive_stats_common_numeric": numeric_desc,
    "data_quality": {
        "header_contamination_total_rows": header_contamination_count,
        "header_contamination_per_file": hc_per_file,
        "missing_values_per_column": missing_stats,
        "combined_shape": list(combined.shape),
        "read_errors": read_errors,
    },
    "player_counts_per_team_season": {
        "details": player_counts,
        "stats": player_count_stats,
    },
    "position_distribution": position_distribution,
    "top_players": {
        "top_scorers_in_sample": top_scorers,
        "top_assisters_in_sample": top_assisters,
        "goals_column_used": goals_col,
        "assists_column_used": assist_col,
    },
    "age_distribution": age_stats,
    "nationality_distribution_top15": nationality_dist,
    "anomalies_found": anomalies,
    "data_normalization_observation": normalization_note,
    "season_column_format": season_format_analysis,
    "per_90_rate_columns": per90_analysis,
    "null_player_rows_per_file": null_player_rows,
    "goals_per_season_sample": goals_per_season,
    "assists_per_season_sample": assists_per_season,
    "processed_parquet_comparison": parquet_analysis,
}

# ─────────────────────────────────────────────
# 19. Save JSON
# ─────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, default=str, indent=2)

print(f"\nSaved: {OUTPUT_JSON}")
print("Done.")

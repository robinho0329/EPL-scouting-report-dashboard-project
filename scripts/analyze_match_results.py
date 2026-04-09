"""
EPL Match Results – Comprehensive Analysis Script
For Korean preprocessing-EDA report
RESEARCH ONLY – writes output JSON only
"""
import pandas as pd
import numpy as np
import json
import os
import warnings

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = "C:/Users/xcv54/workspace/EPL project"
# The spec says data/epl_final.csv but the file lives at root – check both
CSV_CANDIDATES = [
    BASE + "/data/epl_final.csv",
    BASE + "/epl_final.csv",
]
PARQUET    = BASE + "/data/processed/match_results.parquet"
REPORT_DIR = BASE + "/reports"
OUT_JSON   = REPORT_DIR + "/analysis_match_results.json"
os.makedirs(REPORT_DIR, exist_ok=True)

# ── Helper: convert numpy / pandas scalars to native Python ───────────────────
def to_py(obj):
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)):    return bool(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, pd.Series):      return obj.tolist()
    if isinstance(obj, pd.Timestamp):   return str(obj)
    return obj

def clean(d):
    if isinstance(d, dict):  return {k: clean(v) for k, v in d.items()}
    if isinstance(d, list):  return [clean(v) for v in d]
    return to_py(d)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD CSV
# ══════════════════════════════════════════════════════════════════════════════
CSV_PATH, csv_enc, df_csv = None, None, None
for cand in CSV_CANDIDATES:
    if not os.path.exists(cand):
        continue
    for enc in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            df_csv = pd.read_csv(cand, encoding=enc)
            CSV_PATH, csv_enc = cand, enc
            print(f"[CSV] Loaded {cand} (enc={enc}) shape={df_csv.shape}")
            break
        except Exception as e:
            print(f"  {enc} failed: {e}")
    if df_csv is not None:
        break
if df_csv is None:
    raise RuntimeError("Could not load CSV")

# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD PARQUET
# ══════════════════════════════════════════════════════════════════════════════
par_exists, df_par = os.path.exists(PARQUET), None
if par_exists:
    try:
        df_par = pd.read_parquet(PARQUET)
        print(f"[Parquet] Loaded shape={df_par.shape}")
    except Exception as e:
        print(f"[Parquet] failed: {e}")
        par_exists = False

# ── Working copy from PARQUET (richer, processed) ────────────────────────────
# We use parquet as primary df (it already went through preprocessing).
# Fallback to CSV if parquet unavailable.
df = df_par.copy() if df_par is not None else df_csv.copy()

result = {}

# ══════════════════════════════════════════════════════════════════════════════
# 3. DATE NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════
date_col = "MatchDate" if "MatchDate" in df.columns else None
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
# Also normalize in CSV copy for comparison
if df_csv is not None and date_col and date_col in df_csv.columns:
    df_csv = df_csv.copy()
    df_csv[date_col] = pd.to_datetime(df_csv[date_col], errors="coerce")

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – METADATA (shape, columns, dtypes, memory)
# ══════════════════════════════════════════════════════════════════════════════
result["1_metadata"] = clean({
    "primary_source":       "parquet" if df_par is not None else "csv",
    "csv_path":             CSV_PATH,
    "csv_encoding":         csv_enc,
    "csv_file_size_bytes":  int(os.path.getsize(CSV_PATH)) if CSV_PATH else None,
    "parquet_path":         PARQUET if par_exists else None,
    "parquet_file_size_bytes": int(os.path.getsize(PARQUET)) if par_exists else None,
    "shape":                {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
    "columns":              list(df.columns),
    "dtypes":               {col: str(dtype) for col, dtype in df.dtypes.items()},
    "memory_usage_bytes":   int(df.memory_usage(deep=True).sum()),
    "memory_usage_kb":      round(df.memory_usage(deep=True).sum() / 1024, 2),
})

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – MISSING VALUES
# ══════════════════════════════════════════════════════════════════════════════
miss_count = df.isnull().sum()
miss_pct   = (df.isnull().mean() * 100).round(4)
result["2_missing_values"] = clean({
    "per_column": {
        col: {"count": int(miss_count[col]), "percent": float(miss_pct[col])}
        for col in df.columns
    },
    "summary": {
        "total_null_cells":          int(df.isnull().sum().sum()),
        "columns_with_any_missing":  [c for c in df.columns if miss_count[c] > 0],
        "pct_complete_rows":         round(float(df.dropna().shape[0] / len(df) * 100), 2),
    },
})

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – DESCRIPTIVE STATISTICS (all numeric)
# ══════════════════════════════════════════════════════════════════════════════
desc_dict = {}
if num_cols:
    raw = df[num_cols].describe().to_dict()
    for col, stats in raw.items():
        desc_dict[col] = {k: to_py(v) for k, v in stats.items()}
    # Extra percentiles
    pct_df = df[num_cols].quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    for col in num_cols:
        desc_dict[col]["percentiles"] = {str(q): to_py(pct_df.loc[q, col]) for q in pct_df.index}

result["3_descriptive_statistics"] = clean(desc_dict)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – SEASON COVERAGE
# ══════════════════════════════════════════════════════════════════════════════
seasons_list  = sorted([str(s) for s in df["Season"].dropna().unique()])
season_counts = df["Season"].value_counts().sort_index()
result["4_season_coverage"] = clean({
    "n_seasons":          len(seasons_list),
    "seasons_list":       seasons_list,
    "matches_per_season": {str(k): int(v) for k, v in season_counts.items()},
    "min_matches_season": {"season": str(season_counts.idxmin()), "count": int(season_counts.min())},
    "max_matches_season": {"season": str(season_counts.idxmax()), "count": int(season_counts.max())},
})

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – DATE RANGE & TOTALS
# ══════════════════════════════════════════════════════════════════════════════
date_info = {"total_matches": int(len(df))}
if date_col and df[date_col].notna().any():
    date_info["date_min"]       = str(df[date_col].min().date())
    date_info["date_max"]       = str(df[date_col].max().date())
    date_info["date_span_days"] = int((df[date_col].max() - df[date_col].min()).days)
    date_info["null_dates"]     = int(df[date_col].isnull().sum())
result["5_date_and_match_info"] = clean(date_info)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – RESULT DISTRIBUTION (Home Win / Draw / Away Win)
# ══════════════════════════════════════════════════════════════════════════════
vc_ft    = df["FullTimeResult"].value_counts()
total    = int(df["FullTimeResult"].notna().sum())
hw_ct, dr_ct, aw_ct = int((df["FullTimeResult"]=="H").sum()), int((df["FullTimeResult"]=="D").sum()), int((df["FullTimeResult"]=="A").sum())

per_season_results = {}
for season, grp in df.groupby("Season"):
    n = int(grp["FullTimeResult"].notna().sum())
    hw = int((grp["FullTimeResult"]=="H").sum())
    dr = int((grp["FullTimeResult"]=="D").sum())
    aw = int((grp["FullTimeResult"]=="A").sum())
    per_season_results[str(season)] = {
        "total": n, "home_wins": hw, "draws": dr, "away_wins": aw,
        "home_win_pct": round(hw/n*100,2), "draw_pct": round(dr/n*100,2), "away_win_pct": round(aw/n*100,2),
    }

ht_dist = {}
if "HalfTimeResult" in df.columns:
    vc_ht = df["HalfTimeResult"].value_counts()
    ht_tot = int(df["HalfTimeResult"].notna().sum())
    ht_dist = {str(k): {"count": int(v), "percent": round(float(v/ht_tot*100),2)} for k, v in vc_ht.items()}

result["6_result_distribution"] = clean({
    "full_time": {
        "overall": {
            "home_wins": hw_ct, "draws": dr_ct, "away_wins": aw_ct, "total": total,
            "home_win_pct": round(hw_ct/total*100,2),
            "draw_pct":     round(dr_ct/total*100,2),
            "away_win_pct": round(aw_ct/total*100,2),
        },
        "per_season": per_season_results,
        "unique_values": sorted([str(v) for v in df["FullTimeResult"].dropna().unique()]),
    },
    "half_time": ht_dist if ht_dist else {"note": "HalfTimeResult column absent"},
})

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 – GOALS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
df["TotalGoals"] = df["FullTimeHomeGoals"] + df["FullTimeAwayGoals"]
df["GoalDiff"]   = df["FullTimeHomeGoals"] - df["FullTimeAwayGoals"]

# Overall stats
goals_overall = {
    "avg_home_goals":    round(float(df["FullTimeHomeGoals"].mean()), 4),
    "avg_away_goals":    round(float(df["FullTimeAwayGoals"].mean()), 4),
    "avg_total_goals":   round(float(df["TotalGoals"].mean()), 4),
    "median_total_goals":float(df["TotalGoals"].median()),
    "std_total_goals":   round(float(df["TotalGoals"].std()), 4),
    "max_total_goals":   int(df["TotalGoals"].max()),
    "min_total_goals":   int(df["TotalGoals"].min()),
    "total_goals_all":   int(df["TotalGoals"].sum()),
    "pct_0_0_draws":     round(float(((df["FullTimeHomeGoals"]==0)&(df["FullTimeAwayGoals"]==0)).sum()/len(df)*100), 2),
}

# Per season
goals_per_season = {}
for season, grp in df.groupby("Season"):
    goals_per_season[str(season)] = {
        "avg_home_goals":   round(float(grp["FullTimeHomeGoals"].mean()), 4),
        "avg_away_goals":   round(float(grp["FullTimeAwayGoals"].mean()), 4),
        "avg_total_goals":  round(float(grp["TotalGoals"].mean()), 4),
        "total_home_goals": int(grp["FullTimeHomeGoals"].sum()),
        "total_away_goals": int(grp["FullTimeAwayGoals"].sum()),
        "total_goals":      int(grp["TotalGoals"].sum()),
        "n_matches":        int(len(grp)),
    }

# Top 15 scoring matches
top_cols = [c for c in ["Season", date_col, "HomeTeam", "AwayTeam",
                         "FullTimeHomeGoals", "FullTimeAwayGoals", "TotalGoals", "FullTimeResult"] if c]
top15 = df.nlargest(15, "TotalGoals")[top_cols].reset_index(drop=True)
top15_list = []
for i, row in top15.iterrows():
    rec = {"rank": int(i)+1}
    for c in top_cols:
        val = row[c]
        if pd.api.types.is_integer_dtype(type(val)) or isinstance(val, (np.integer,)):
            rec[c] = int(val)
        elif isinstance(val, pd.Timestamp):
            rec[c] = str(val.date())
        else:
            rec[c] = to_py(val)
    top15_list.append(rec)

# Top 20 scorelines
score_freq = (
    df.groupby(["FullTimeHomeGoals", "FullTimeAwayGoals"])
      .size().reset_index(name="count")
      .sort_values("count", ascending=False).head(20)
)
scorelines = [
    {"home": int(r["FullTimeHomeGoals"]), "away": int(r["FullTimeAwayGoals"]), "count": int(r["count"])}
    for _, r in score_freq.iterrows()
]

result["7_goals_analysis"] = clean({
    "overall": goals_overall,
    "avg_goals_per_season_trend": goals_per_season,
    "top_15_highest_scoring_matches": top15_list,
    "top_20_most_common_scorelines":  scorelines,
})

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 – HOME ADVANTAGE
# ══════════════════════════════════════════════════════════════════════════════
ha_per_season = {}
for season, grp in df.groupby("Season"):
    n = int(grp["FullTimeResult"].notna().sum())
    if n == 0: continue
    hw = int((grp["FullTimeResult"]=="H").sum())
    dr = int((grp["FullTimeResult"]=="D").sum())
    aw = int((grp["FullTimeResult"]=="A").sum())
    avg_hg = round(float(grp["FullTimeHomeGoals"].mean()), 4)
    avg_ag = round(float(grp["FullTimeAwayGoals"].mean()), 4)
    ha_per_season[str(season)] = {
        "n_matches": n,
        "home_win_pct": round(hw/n*100, 2),
        "draw_pct":     round(dr/n*100, 2),
        "away_win_pct": round(aw/n*100, 2),
        "avg_home_goals": avg_hg,
        "avg_away_goals": avg_ag,
        "goal_diff_per_match": round(avg_hg - avg_ag, 4),
    }

result["8_home_advantage"] = clean({
    "overall": {
        "home_win_pct":  round(hw_ct/total*100, 2),
        "draw_pct":      round(dr_ct/total*100, 2),
        "away_win_pct":  round(aw_ct/total*100, 2),
        "avg_home_goals": round(float(df["FullTimeHomeGoals"].mean()), 4),
        "avg_away_goals": round(float(df["FullTimeAwayGoals"].mean()), 4),
        "avg_goal_diff":  round(float(df["GoalDiff"].mean()), 4),
    },
    "per_season": ha_per_season,
})

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 – TEAM FREQUENCY & RECORDS
# ══════════════════════════════════════════════════════════════════════════════
home_cnt  = df["HomeTeam"].value_counts()
away_cnt  = df["AwayTeam"].value_counts()
total_app = home_cnt.add(away_cnt, fill_value=0).astype(int).sort_values(ascending=False)

all_teams_set = set(df["HomeTeam"].dropna().unique()) | set(df["AwayTeam"].dropna().unique())

# Per-team win/draw/loss record
team_records = []
for team in total_app.index:
    hg = df[df["HomeTeam"] == team]
    ag = df[df["AwayTeam"] == team]
    n  = len(hg) + len(ag)
    wins   = int((hg["FullTimeResult"]=="H").sum()) + int((ag["FullTimeResult"]=="A").sum())
    draws  = int((hg["FullTimeResult"]=="D").sum()) + int((ag["FullTimeResult"]=="D").sum())
    losses = int((hg["FullTimeResult"]=="A").sum()) + int((ag["FullTimeResult"]=="H").sum())
    team_records.append({
        "team": str(team), "total_matches": n,
        "wins": wins, "draws": draws, "losses": losses,
        "win_pct":  round(wins/n*100,2) if n else None,
        "home_matches": int(len(hg)), "away_matches": int(len(ag)),
        "goals_scored": int(hg["FullTimeHomeGoals"].sum()) + int(ag["FullTimeAwayGoals"].sum()),
        "goals_conceded": int(hg["FullTimeAwayGoals"].sum()) + int(ag["FullTimeHomeGoals"].sum()),
    })
team_records_by_wins    = sorted(team_records, key=lambda x: x["wins"], reverse=True)
team_records_by_matches = sorted(team_records, key=lambda x: x["total_matches"], reverse=True)

result["9_team_analysis"] = clean({
    "total_unique_teams": len(all_teams_set),
    "all_teams_sorted":   sorted([str(t) for t in all_teams_set]),
    "top_20_most_appearances":  {str(k): int(v) for k, v in total_app.head(20).items()},
    "bottom_20_least_appearances": {str(k): int(v) for k, v in total_app.tail(20).items()},
    "top_20_teams_by_wins":    team_records_by_wins[:20],
    "bottom_20_teams_by_wins": team_records_by_wins[-20:],
    "top_20_most_matches":     team_records_by_matches[:20],
})

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 – CARD STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
card_present = all(c in df.columns for c in ["HomeYellowCards","AwayYellowCards","HomeRedCards","AwayRedCards"])
if card_present:
    df["TotalYellowCards"] = df["HomeYellowCards"] + df["AwayYellowCards"]
    df["TotalRedCards"]    = df["HomeRedCards"]    + df["AwayRedCards"]

    cards_per_season = {}
    for season, grp in df.groupby("Season"):
        cards_per_season[str(season)] = {
            "avg_home_yellow": round(float(grp["HomeYellowCards"].mean()), 4),
            "avg_away_yellow": round(float(grp["AwayYellowCards"].mean()), 4),
            "avg_total_yellow":round(float(grp["TotalYellowCards"].mean()),4),
            "avg_home_red":    round(float(grp["HomeRedCards"].mean()),    4),
            "avg_away_red":    round(float(grp["AwayRedCards"].mean()),    4),
            "avg_total_red":   round(float(grp["TotalRedCards"].mean()),   4),
            "total_reds":      int(grp["TotalRedCards"].sum()),
        }

    top_yellow_cols = [c for c in ["Season", date_col, "HomeTeam","AwayTeam",
                                    "HomeYellowCards","AwayYellowCards",
                                    "HomeRedCards","AwayRedCards","TotalYellowCards","TotalRedCards"] if c]
    top_yellow = df.nlargest(10, "TotalYellowCards")[top_yellow_cols].astype(str).to_dict(orient="records")

    result["10_card_statistics"] = clean({
        "overall": {
            "avg_home_yellow": round(float(df["HomeYellowCards"].mean()),4),
            "avg_away_yellow": round(float(df["AwayYellowCards"].mean()),4),
            "avg_total_yellow":round(float(df["TotalYellowCards"].mean()),4),
            "avg_home_red":    round(float(df["HomeRedCards"].mean()),4),
            "avg_away_red":    round(float(df["AwayRedCards"].mean()),4),
            "avg_total_red":   round(float(df["TotalRedCards"].mean()),4),
            "max_yellows_match":int(df["TotalYellowCards"].max()),
            "max_reds_match":   int(df["TotalRedCards"].max()),
            "matches_with_red": int((df["TotalRedCards"]>0).sum()),
            "red_card_match_pct": round(float((df["TotalRedCards"]>0).mean()*100),2),
        },
        "per_season_trend": cards_per_season,
        "top_10_most_yellow_card_matches": top_yellow,
    })
else:
    result["10_card_statistics"] = {"note": "카드 컬럼 없음"}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 – SHOTS, CORNERS, FOULS
# ══════════════════════════════════════════════════════════════════════════════
scf_cols = {
    "HomeShots":           "home_shots",
    "AwayShots":           "away_shots",
    "HomeShotsOnTarget":   "home_sot",
    "AwayShotsOnTarget":   "away_sot",
    "HomeCorners":         "home_corners",
    "AwayCorners":         "away_corners",
    "HomeFouls":           "home_fouls",
    "AwayFouls":           "away_fouls",
}
scf_stats = {}
for col, label in scf_cols.items():
    if col in df.columns:
        scf_stats[label] = {
            "mean":       round(float(df[col].mean()), 4),
            "median":     float(df[col].median()),
            "std":        round(float(df[col].std()), 4),
            "min":        to_py(df[col].min()),
            "max":        to_py(df[col].max()),
            "null_count": int(df[col].isnull().sum()),
        }

shots_present = all(c in df.columns for c in ["HomeShots","AwayShots","HomeShotsOnTarget","AwayShotsOnTarget"])
if shots_present:
    df["TotalShots"]         = df["HomeShots"]         + df["AwayShots"]
    df["TotalShotsOnTarget"] = df["HomeShotsOnTarget"] + df["AwayShotsOnTarget"]
    df["HomeSOT_pct"]        = df["HomeShotsOnTarget"] / df["HomeShots"].replace(0, np.nan) * 100
    df["AwaySOT_pct"]        = df["AwayShotsOnTarget"] / df["AwayShots"].replace(0, np.nan) * 100
    scf_stats["avg_home_sot_pct"] = round(float(df["HomeSOT_pct"].mean()),2)
    scf_stats["avg_away_sot_pct"] = round(float(df["AwaySOT_pct"].mean()),2)

    shots_per_season = {}
    for season, grp in df.groupby("Season"):
        shots_per_season[str(season)] = {
            "avg_home_shots":   round(float(grp["HomeShots"].mean()),4),
            "avg_away_shots":   round(float(grp["AwayShots"].mean()),4),
            "avg_total_shots":  round(float(grp["TotalShots"].mean()),4),
            "avg_home_sot":     round(float(grp["HomeShotsOnTarget"].mean()),4),
            "avg_away_sot":     round(float(grp["AwayShotsOnTarget"].mean()),4),
            "avg_total_sot":    round(float(grp["TotalShotsOnTarget"].mean()),4),
        }
    scf_stats["per_season_shots"] = shots_per_season

result["11_shots_corners_fouls"] = clean(scf_stats)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 – BOOLEAN / FLAG COLUMNS
# ══════════════════════════════════════════════════════════════════════════════
bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
bool_dist = {}
for col in bool_cols:
    vc = df[col].value_counts()
    bool_dist[col] = {str(k): int(v) for k, v in vc.items()}
result["12_boolean_flag_columns"] = clean(bool_dist) if bool_dist else {"note": "No boolean columns"}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 – DATA QUALITY
# ══════════════════════════════════════════════════════════════════════════════
dup_rows  = int(df.duplicated().sum())
match_dup = int(df.duplicated(subset=["HomeTeam","AwayTeam",date_col]).sum()) if date_col else None

# Result vs goals consistency check
df_valid  = df[["FullTimeHomeGoals","FullTimeAwayGoals","FullTimeResult"]].dropna()
computed  = np.where(df_valid["FullTimeHomeGoals"]>df_valid["FullTimeAwayGoals"],"H",
            np.where(df_valid["FullTimeHomeGoals"]<df_valid["FullTimeAwayGoals"],"A","D"))
mismatch  = int((computed != df_valid["FullTimeResult"]).sum())

result["13_data_quality"] = clean({
    "duplicate_rows":                  dup_rows,
    "duplicate_rows_pct":              round(dup_rows/len(df)*100,4),
    "duplicate_matches_teams_date":    match_dup,
    "result_vs_goals_mismatch":        mismatch,
    "result_vs_goals_mismatch_pct":    round(mismatch/len(df_valid)*100,4) if len(df_valid)>0 else None,
    "negative_home_goals":             int((df["FullTimeHomeGoals"]<0).sum()),
    "negative_away_goals":             int((df["FullTimeAwayGoals"]<0).sum()),
    "null_dates":                      int(df[date_col].isnull().sum()) if date_col else None,
    "unexpected_full_time_result_values": sorted([
        str(v) for v in df["FullTimeResult"].dropna().unique() if v not in {"H","D","A"}
    ]),
    "full_time_result_value_counts":   {str(k): int(v) for k,v in df["FullTimeResult"].value_counts().items()},
    "half_time_result_value_counts":   {str(k): int(v) for k,v in df["HalfTimeResult"].value_counts().items()} if "HalfTimeResult" in df.columns else {},
    "total_null_cells":                int(df.isnull().sum().sum()),
    "columns_with_any_missing":        [c for c in df.columns if df[c].isnull().any()],
    "pct_complete_rows":               round(float(df.dropna().shape[0]/len(df)*100),2),
    "unique_teams_combined":           len(all_teams_set),
    "unique_teams_list":               sorted([str(t) for t in all_teams_set]),
    "date_range_start":                str(df[date_col].min().date()) if date_col and df[date_col].notna().any() else None,
    "date_range_end":                  str(df[date_col].max().date()) if date_col and df[date_col].notna().any() else None,
    "total_seasons":                   int(df["Season"].nunique()),
    "sample_season_values":            [str(s) for s in df["Season"].dropna().unique()[:5]],
})

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 14 – PARQUET vs CSV COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
par_cmp = {"parquet_exists": par_exists}
if df_par is not None and df_csv is not None:
    par_only = [c for c in df_par.columns if c not in df_csv.columns]
    csv_only = [c for c in df_csv.columns if c not in df_par.columns]
    shared   = [c for c in df_par.columns if c in df_csv.columns]

    dtype_changes = {}
    for col in shared:
        ct = str(df_csv[col].dtype); pt = str(df_par[col].dtype)
        if ct != pt: dtype_changes[col] = {"csv": ct, "parquet": pt}

    miss_changes = {}
    for col in shared:
        cn = int(df_csv[col].isnull().sum()); pn = int(df_par[col].isnull().sum())
        if cn != pn: miss_changes[col] = {"csv_nulls": cn, "parquet_nulls": pn}

    new_col_stats = {}
    for col in par_only[:15]:
        if pd.api.types.is_numeric_dtype(df_par[col]):
            new_col_stats[col] = {
                "dtype": str(df_par[col].dtype),
                "mean":  to_py(df_par[col].mean()), "min": to_py(df_par[col].min()),
                "max":   to_py(df_par[col].max()),  "nulls": int(df_par[col].isnull().sum()),
            }
        else:
            vc = df_par[col].value_counts().head(5)
            new_col_stats[col] = {
                "dtype": str(df_par[col].dtype),
                "top5_values": {str(k): int(v) for k,v in vc.items()},
                "nulls": int(df_par[col].isnull().sum()),
            }

    par_num = df_par.select_dtypes(include=[np.number]).columns.tolist()
    par_describe = {}
    if par_num:
        raw_par = df_par[par_num].describe().to_dict()
        for col, stats in raw_par.items():
            par_describe[col] = {k: to_py(v) for k, v in stats.items()}

    par_cmp.update({
        "csv_shape":                  list(df_csv.shape),
        "parquet_shape":              list(df_par.shape),
        "row_diff":                   int(df_par.shape[0] - df_csv.shape[0]),
        "col_diff":                   int(df_par.shape[1] - df_csv.shape[1]),
        "csv_columns":                list(df_csv.columns),
        "parquet_columns":            list(df_par.columns),
        "columns_only_in_parquet":    par_only,
        "columns_only_in_csv":        csv_only,
        "shared_columns":             shared,
        "dtype_changes":              dtype_changes,
        "null_count_changes":         miss_changes,
        "csv_memory_bytes":           int(df_csv.memory_usage(deep=True).sum()),
        "parquet_memory_bytes":       int(df_par.memory_usage(deep=True).sum()),
        "csv_file_size_bytes":        int(os.path.getsize(CSV_PATH)),
        "parquet_file_size_bytes":    int(os.path.getsize(PARQUET)),
        "parquet_dtypes":             {k: str(v) for k,v in df_par.dtypes.items()},
        "parquet_missing_per_col":    {col: int(df_par[col].isnull().sum()) for col in df_par.columns if df_par[col].isnull().any()},
        "new_columns_in_parquet_stats": new_col_stats,
        "parquet_describe":           par_describe,
    })

result["14_parquet_vs_csv_comparison"] = clean(par_cmp)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 15 – CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
# Refresh num_cols to include any derived columns (TotalGoals, GoalDiff, etc.)
num_cols_final = df.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols_final) >= 2:
    corr_df = df[num_cols_final].corr().round(4)
    result["15_correlation_matrix"] = clean({
        col: {k: to_py(v) for k, v in row.items()}
        for col, row in corr_df.to_dict().items()
    })
    if "TotalGoals" in df.columns:
        corrs = corr_df["TotalGoals"].drop("TotalGoals", errors="ignore").sort_values(ascending=False)
        result["15_top_correlations_with_total_goals"] = {str(k): round(float(v),4) for k,v in corrs.items()}
else:
    result["15_correlation_matrix"] = {"note": "Not enough numeric columns"}

# ══════════════════════════════════════════════════════════════════════════════
# SAVE JSON
# ══════════════════════════════════════════════════════════════════════════════
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2, default=str)

print(f"\n[DONE] Saved → {OUT_JSON}")
print(f"       JSON size: {os.path.getsize(OUT_JSON):,} bytes")
print(f"       Top-level keys: {list(result.keys())}")
print("\n=== QUICK SUMMARY ===")
print(f"Primary source:   {'parquet' if df_par is not None else 'csv'}")
print(f"Shape:            {df.shape}")
print(f"Total matches:    {len(df):,}")
print(f"Seasons:          {len(seasons_list)}")
print(f"Date range:       {date_info.get('date_min','?')} → {date_info.get('date_max','?')}")
print(f"Unique teams:     {len(all_teams_set)}")
print(f"Duplicate rows:   {dup_rows}")
print(f"Total null cells: {int(df.isnull().sum().sum())}")
print(f"Home win %:       {round(hw_ct/total*100,2)}%")
print(f"Draw %:           {round(dr_ct/total*100,2)}%")
print(f"Away win %:       {round(aw_ct/total*100,2)}%")
print(f"Avg goals/match:  {round(float(df['TotalGoals'].mean()),4)}")

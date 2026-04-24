"""
EPL Processed Data Analysis Script
Generates a comprehensive analysis JSON for Korean preprocessing-EDA report.
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def to_native(obj):
    """Recursively convert numpy/pandas types to native Python."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return to_native(obj.tolist())
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def analyze_file(path: str) -> dict:
    """Perform per-file analysis."""
    df = pd.read_parquet(path)
    result = {}

    # ── Basic metadata ──
    result["file_path"] = path
    result["shape"] = {"rows": df.shape[0], "cols": df.shape[1]}
    result["columns"] = df.columns.tolist()
    result["dtypes"] = {col: str(df[col].dtype) for col in df.columns}
    result["memory_usage_bytes"] = {
        "total": int(df.memory_usage(deep=True).sum()),
        "per_column": to_native(df.memory_usage(deep=True).to_dict()),
    }

    # ── Missing values ──
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    result["missing_values"] = {
        col: {"count": int(missing_count[col]), "pct": float(missing_pct[col])}
        for col in df.columns
        if missing_count[col] > 0
    }

    # ── Descriptive stats for numeric columns ──
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        desc = df[numeric_cols].describe().round(4)
        result["numeric_stats"] = to_native(desc.to_dict())
    else:
        result["numeric_stats"] = {}

    # ── Unique value counts for categorical / object / bool columns ──
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    result["categorical_stats"] = {}
    for col in cat_cols:
        vc = df[col].value_counts(dropna=False)
        result["categorical_stats"][col] = {
            "unique_count": int(df[col].nunique(dropna=False)),
            "top10": to_native(vc.head(10).to_dict()),
        }

    # ── Sample rows ──
    sample = df.head(3).copy()
    for col in sample.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        sample[col] = sample[col].astype(str)
    result["sample_rows"] = to_native(sample.to_dict(orient="records"))

    return result


# ─────────────────────────────────────────
# Load all files
# ─────────────────────────────────────────

PROCESSED_DIR = "C:/Users/xcv54/workspace/EPL project/data/processed/"
DASHBOARD_DIR = "C:/Users/xcv54/workspace/EPL project/data/dashboard/"
REPORT_PATH   = "C:/Users/xcv54/workspace/EPL project/reports/analysis_processed.json"

os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

parquet_sources = {}
for d in [PROCESSED_DIR, DASHBOARD_DIR]:
    for f in sorted(Path(d).glob("*.parquet")):
        key = f"{Path(d).name}/{f.name}"
        parquet_sources[key] = str(f)

print(f"Found {len(parquet_sources)} parquet files:")
for k, v in parquet_sources.items():
    print(f"  {k}")

# ─────────────────────────────────────────
# Per-file analysis
# ─────────────────────────────────────────

file_analyses = {}
dataframes = {}

for key, path in parquet_sources.items():
    print(f"\nAnalysing: {key}")
    df = pd.read_parquet(path)
    dataframes[key] = df
    file_analyses[key] = analyze_file(path)
    print(f"  shape={df.shape}")


# ─────────────────────────────────────────
# Cross-file analysis
# ─────────────────────────────────────────

cross = {}

def get_df(name_fragment):
    for k, df in dataframes.items():
        if name_fragment in k:
            return k, df
    return None, None


# ── 1. match_results ↔ team_season_summary ──
mr_key, mr_df = get_df("processed/match_results")
ts_key, ts_df = get_df("processed/team_season_summary")

if mr_df is not None and ts_df is not None:
    mr_cols = set(mr_df.columns)
    ts_cols = set(ts_df.columns)
    common = sorted(mr_cols & ts_cols)

    # match_results uses HomeTeam/AwayTeam; team_season_summary uses 'team'
    # The Season column is shared
    mr_seasons = sorted(mr_df["Season"].dropna().unique().tolist())
    ts_seasons = sorted(ts_df["Season"].dropna().unique().tolist())

    mr_home_teams = sorted(mr_df["HomeTeam"].dropna().unique().tolist())
    mr_away_teams = sorted(mr_df["AwayTeam"].dropna().unique().tolist())
    mr_all_teams  = sorted(set(mr_home_teams) | set(mr_away_teams))
    ts_teams      = sorted(ts_df["team"].dropna().unique().tolist())

    teams_in_mr_not_ts = sorted(set(mr_all_teams) - set(ts_teams))
    teams_in_ts_not_mr = sorted(set(ts_teams) - set(mr_all_teams))

    # Per-season team counts derived from match_results
    mr_teams_by_season = (
        pd.concat([
            mr_df[["Season", "HomeTeam"]].rename(columns={"HomeTeam": "team"}),
            mr_df[["Season", "AwayTeam"]].rename(columns={"AwayTeam": "team"}),
        ])
        .drop_duplicates()
        .groupby("Season")["team"]
        .count()
        .sort_index()
        .to_dict()
    )
    ts_teams_by_season = (
        ts_df.groupby("Season")["team"]
        .count()
        .sort_index()
        .to_dict()
    )

    cross["match_results_x_team_season_summary"] = {
        "description": (
            "match_results는 개별 경기 데이터(9380행)이고 team_season_summary는 "
            "팀×시즌 집계(500행)입니다. 공통 조인 키는 Season + team(HomeTeam/AwayTeam)입니다."
        ),
        "join_keys": {
            "shared_season_col": "Season",
            "match_results_team_cols": ["HomeTeam", "AwayTeam"],
            "team_season_summary_team_col": "team",
            "note": "team_season_summary의 'team'은 match_results의 HomeTeam 또는 AwayTeam과 매핑됩니다.",
        },
        "common_columns": common,
        "shape_comparison": {
            "match_results": {"rows": int(mr_df.shape[0]), "cols": int(mr_df.shape[1])},
            "team_season_summary": {"rows": int(ts_df.shape[0]), "cols": int(ts_df.shape[1])},
        },
        "season_alignment": {
            "match_results_seasons": mr_seasons,
            "team_season_summary_seasons": ts_seasons,
            "seasons_in_mr_not_ts": sorted(set(mr_seasons) - set(ts_seasons)),
            "seasons_in_ts_not_mr": sorted(set(ts_seasons) - set(mr_seasons)),
            "fully_aligned": set(mr_seasons) == set(ts_seasons),
        },
        "team_alignment": {
            "mr_unique_teams": len(mr_all_teams),
            "ts_unique_teams": len(ts_teams),
            "mr_teams": mr_all_teams,
            "ts_teams": ts_teams,
            "teams_in_mr_not_ts": teams_in_mr_not_ts,
            "teams_in_ts_not_mr": teams_in_ts_not_mr,
            "fully_consistent": len(teams_in_mr_not_ts) == 0 and len(teams_in_ts_not_mr) == 0,
        },
        "teams_per_season_match_results": {str(k): int(v) for k, v in mr_teams_by_season.items()},
        "teams_per_season_summary": {str(k): int(v) for k, v in ts_teams_by_season.items()},
    }
else:
    cross["match_results_x_team_season_summary"] = {"error": "One or both files not found"}


# ── 2. player_season_stats ↔ player_match_logs ──
ps_key, ps_df = get_df("processed/player_season_stats")
pl_key, pl_df = get_df("processed/player_match_logs")
pm_key, pm_df = get_df("processed/player_match_stats")

if ps_df is not None and pl_df is not None:
    ps_cols = set(ps_df.columns)
    pl_cols = set(pl_df.columns)
    common_p = sorted(ps_cols & pl_cols)

    # player_match_logs/stats share more columns — compare them too
    pm_cols = set(pm_df.columns) if pm_df is not None else set()
    pl_pm_common = sorted(pl_cols & pm_cols)
    pl_pm_diff_in_logs  = sorted(pl_cols - pm_cols)
    pl_pm_diff_in_stats = sorted(pm_cols - pl_cols)

    # Players in season stats vs match logs
    ps_players = set(ps_df["player"].dropna().unique())
    pl_players = set(pl_df["player"].dropna().unique())
    pm_players = set(pm_df["player"].dropna().unique()) if pm_df is not None else set()

    ps_seasons = sorted(ps_df["season"].dropna().unique().tolist())
    pl_seasons = sorted(pl_df["season"].dropna().unique().tolist())

    cross["player_season_stats_x_player_match_logs"] = {
        "description": (
            "player_season_stats는 선수×시즌 집계(14980행)이고 "
            "player_match_logs는 선수×경기 로그(299517행)입니다. "
            "player + season + team을 조인 키로 사용합니다."
        ),
        "join_keys": ["player", "season", "team"],
        "common_columns_with_match_logs": common_p,
        "shape_comparison": {
            "player_season_stats": {"rows": int(ps_df.shape[0]), "cols": int(ps_df.shape[1])},
            "player_match_logs": {"rows": int(pl_df.shape[0]), "cols": int(pl_df.shape[1])},
            "player_match_stats": {"rows": int(pm_df.shape[0]), "cols": int(pm_df.shape[1])} if pm_df is not None else None,
        },
        "player_match_logs_vs_match_stats": {
            "common_columns": pl_pm_common,
            "only_in_match_logs": pl_pm_diff_in_logs,
            "only_in_match_stats": pl_pm_diff_in_stats,
            "note": (
                "player_match_logs는 'started'/'detail_stats_available'/'outcome'을 포함하고 "
                "player_match_stats는 'start'/'pkwon'/'pkcon'/'match_report'/'player_normalized'을 포함합니다."
            ),
        },
        "player_counts": {
            "player_season_stats": len(ps_players),
            "player_match_logs": len(pl_players),
            "player_match_stats": len(pm_players),
            "in_ps_not_pl": len(ps_players - pl_players),
            "in_pl_not_ps": len(pl_players - ps_players),
        },
        "season_alignment": {
            "player_season_stats_seasons": ps_seasons,
            "player_match_logs_seasons": pl_seasons,
            "fully_aligned": ps_seasons == pl_seasons,
        },
        "avg_matches_per_player_per_season": round(
            pl_df.groupby(["season", "player"]).size().mean(), 2
        ) if pl_df is not None else None,
    }
else:
    cross["player_season_stats_x_player_match_logs"] = {"error": "One or both files not found"}


# ── 3. Data coverage across all files ──
coverage = {}

for key, df in dataframes.items():
    entry = {}
    # Seasons
    for scol in ["season", "Season"]:
        if scol in df.columns:
            seasons = sorted([str(s) for s in df[scol].dropna().unique()])
            entry["seasons"] = seasons
            entry["season_count"] = len(seasons)
            break
    # Teams
    for tcol in ["team", "HomeTeam", "AwayTeam", "team_name", "squad"]:
        if tcol in df.columns:
            vals = df[tcol].dropna().unique().tolist()
            entry[f"{tcol}_count"] = len(vals)
            entry[f"{tcol}_sample"] = sorted([str(v) for v in vals])[:30]
    # Players
    for pcol in ["player", "player_id"]:
        if pcol in df.columns:
            vals = df[pcol].dropna().unique().tolist()
            entry[f"{pcol}_count"] = len(vals)
    coverage[key] = entry

cross["data_coverage"] = coverage


# ── 4. Consistency checks ──
consistency = {}

# 4a. player counts between processed and dashboard player_season_stats
ps_proc_key, ps_proc_df = get_df("processed/player_season_stats")
ps_dash_key, ps_dash_df = get_df("dashboard/player_season_stats")
pa_key, pa_df           = get_df("dashboard/player_alltime_stats")

if ps_proc_df is not None and ps_dash_df is not None:
    proc_players = set(ps_proc_df["player"].dropna().unique())
    dash_players = set(ps_dash_df["player"].dropna().unique())
    only_in_proc = sorted(proc_players - dash_players)[:20]
    only_in_dash = sorted(dash_players - proc_players)[:20]
    consistency["player_count_processed_vs_dashboard"] = {
        "processed_player_season_stats_unique_players": len(proc_players),
        "dashboard_player_season_stats_unique_players": len(dash_players),
        "only_in_processed": only_in_proc,
        "only_in_dashboard": only_in_dash,
        "match": proc_players == dash_players,
        "note": (
            "dashboard 버전에는 1명이 추가됩니다. "
            "이는 player_normalized 컬럼이 있어 이름 표준화 후 합산된 결과입니다."
        ),
    }

# 4b. player_alltime_stats vs player_season_stats
if pa_df is not None and ps_proc_df is not None:
    pa_players = set(pa_df["player"].dropna().unique())
    consistency["player_alltime_vs_season_stats"] = {
        "alltime_unique_players": len(pa_players),
        "season_stats_unique_players": len(proc_players),
        "in_alltime_not_season": len(pa_players - proc_players),
        "in_season_not_alltime": len(proc_players - pa_players),
        "note": "player_alltime_stats는 player_season_stats를 집계한 파일입니다.",
    }

# 4c. Team name consistency between match_results and team_season_summary
if mr_df is not None and ts_df is not None:
    mr_all_teams_set = set(mr_df["HomeTeam"].dropna().unique()) | set(mr_df["AwayTeam"].dropna().unique())
    ts_teams_set = set(ts_df["team"].dropna().unique())
    consistency["team_name_consistency_mr_vs_ts"] = {
        "mr_unique_teams": len(mr_all_teams_set),
        "ts_unique_teams": len(ts_teams_set),
        "in_mr_not_ts": sorted(mr_all_teams_set - ts_teams_set)[:20],
        "in_ts_not_mr": sorted(ts_teams_set - mr_all_teams_set)[:20],
        "fully_consistent": mr_all_teams_set == ts_teams_set,
    }

# 4d. Season coverage consistency
season_coverage = {}
for key, df in dataframes.items():
    for scol in ["season", "Season"]:
        if scol in df.columns:
            season_coverage[key] = sorted([str(s) for s in df[scol].dropna().unique()])
            break

all_seasons = set()
for v in season_coverage.values():
    all_seasons.update(v)

consistency["season_coverage"] = {
    "all_seasons_across_files": sorted(all_seasons),
    "total_seasons": len(all_seasons),
    "per_file": season_coverage,
    "files_with_all_seasons": [
        k for k, v in season_coverage.items()
        if set(v) == all_seasons
    ],
    "files_with_partial_coverage": [
        {"file": k, "seasons": v, "missing": sorted(all_seasons - set(v))}
        for k, v in season_coverage.items()
        if set(v) != all_seasons
    ],
}

# 4e. Row count consistency: match_results rows per season vs team_season_summary
if mr_df is not None and ts_df is not None:
    mr_rows_per_season = mr_df.groupby("Season").size().sort_index().to_dict()
    ts_rows_per_season = ts_df.groupby("Season").size().sort_index().to_dict()
    season_row_check = {}
    for s in sorted(all_seasons):
        mr_val = int(mr_rows_per_season.get(s, 0))
        ts_val = int(ts_rows_per_season.get(s, 0))
        expected_matches = mr_val  # one row per match
        expected_teams   = ts_val  # one row per team
        season_row_check[s] = {
            "match_results_rows": mr_val,
            "team_season_summary_rows": ts_val,
        }
    consistency["rows_per_season_check"] = season_row_check

cross["consistency_checks"] = consistency


# ── 5. Key derived columns ──
# Based on actual inspection of the files
derived_cols_analysis = {}

# match_results — flags added during preprocessing
mr_derived = [c for c in mr_df.columns if c in [
    "season_data_missing", "own_goal_flag_home", "own_goal_flag_away", "own_goal_flag"
]] if mr_df is not None else []
if mr_derived:
    derived_cols_analysis["processed/match_results.parquet"] = {
        "derived_columns": mr_derived,
        "description": {
            "season_data_missing": "해당 시즌에 반세부 통계(슛, 코너킥 등)가 없는 경기를 표시하는 플래그",
            "own_goal_flag_home": "홈팀 자책골 여부 플래그",
            "own_goal_flag_away": "원정팀 자책골 여부 플래그",
            "own_goal_flag": "경기 내 자책골 발생 여부 통합 플래그",
        },
    }

# team_season_summary — aggregated/computed columns
ts_derived = [c for c in ts_df.columns if c in [
    "goal_diff", "points",
    "home_played", "away_played", "total_played",
    "total_wins", "total_draws", "total_losses",
    "total_goals_for", "total_goals_against",
]] if ts_df is not None else []
if ts_derived:
    derived_cols_analysis["processed/team_season_summary.parquet"] = {
        "derived_columns": ts_derived,
        "description": {
            "goal_diff": "total_goals_for - total_goals_against 계산값",
            "points": "승리×3 + 무승부×1 계산값",
            "total_*": "홈/원정 통계를 합산한 시즌 합계 컬럼들",
        },
    }

# player_season_stats — derived per-90 and ratio columns
ps_derived = [c for c in ps_df.columns if c in [
    "gls_1", "ast_1", "g_a_1", "g_pk_1", "g_a_pk",
    "transfer_flag", "player_id", "position", "dob",
    "age_tm", "nationality", "foot", "joined",
    "market_value", "market_value_str", "birth_year",
    "height_cm", "no_value_data",
]] if ps_df is not None else []
if ps_derived:
    derived_cols_analysis["processed/player_season_stats.parquet"] = {
        "derived_columns": ps_derived,
        "description": {
            "gls_1": "90분당 골 (FBref 파생 컬럼)",
            "ast_1": "90분당 어시스트 (FBref 파생 컬럼)",
            "g_a_1": "90분당 (골+어시스트) (FBref 파생 컬럼)",
            "g_pk_1": "90분당 PK제외 골 (FBref 파생 컬럼)",
            "g_a_pk": "90분당 PK제외 (골+어시스트) (FBref 파생 컬럼)",
            "transfer_flag": "TransferMarkt 데이터 병합 여부 플래그",
            "player_id": "TransferMarkt 선수 고유 ID",
            "no_value_data": "TransferMarkt 시장 가치 데이터 부재 플래그",
            "position / nationality / foot / joined / market_value / birth_year / height_cm": (
                "TransferMarkt에서 병합된 추가 속성"
            ),
        },
    }

# player_match_logs — enrichment columns
pl_derived = [c for c in pl_df.columns if c in [
    "outcome", "goals_for", "goals_against", "started", "detail_stats_available"
]] if pl_df is not None else []
if pl_derived:
    derived_cols_analysis["processed/player_match_logs.parquet"] = {
        "derived_columns": pl_derived,
        "description": {
            "outcome": "선수 소속 팀의 경기 결과 (W/D/L)",
            "goals_for": "해당 경기 소속팀 득점",
            "goals_against": "해당 경기 소속팀 실점",
            "started": "선발 출전 여부 (bool)",
            "detail_stats_available": "세부 통계(슈팅, 파울 등) 수집 여부 플래그",
        },
    }

# player_match_stats — enrichment columns
pm_derived = [c for c in pm_df.columns if c in [
    "player_normalized", "pkwon", "pkcon", "match_report"
]] if pm_df is not None else []
if pm_derived:
    derived_cols_analysis["processed/player_match_stats.parquet"] = {
        "derived_columns": pm_derived,
        "description": {
            "player_normalized": "선수 이름 표준화 값 (대소문자/특수문자 통일)",
            "pkwon": "획득한 PK 수",
            "pkcon": "허용한 PK 수",
            "match_report": "FBref 세부 경기 리포트 URL",
        },
    }

# dashboard files
if pa_df is not None:
    pa_derived = [c for c in pa_df.columns if c in ["player_normalized", "num_seasons", "90s"]]
    if pa_derived:
        derived_cols_analysis["dashboard/player_alltime_stats.parquet"] = {
            "derived_columns": pa_derived,
            "description": {
                "player_normalized": "선수 이름 표준화 값",
                "num_seasons": "집계된 시즌 수 (파생 컬럼)",
                "90s": "누적 90분 수",
            },
        }

if ps_dash_df is not None:
    pd_derived = [c for c in ps_dash_df.columns if c in ["player_normalized", "matches", "g_a_pk"]]
    if pd_derived:
        derived_cols_analysis["dashboard/player_season_stats.parquet"] = {
            "derived_columns": pd_derived,
            "description": {
                "player_normalized": "선수 이름 표준화 값",
                "matches": "출전 경기 수 (mp 컬럼 별칭)",
            },
        }

cross["key_derived_columns"] = derived_cols_analysis


# ── 6. Pipeline summary ──
raw_dir = "C:/Users/xcv54/workspace/EPL project/data/raw/"
pipeline = {
    "raw_input": {"total_files": 0, "total_size_bytes": 0, "file_types": {}},
    "processed_output": [],
    "dashboard_output": [],
    "total_raw_size_bytes": 0,
    "total_processed_size_bytes": 0,
    "total_dashboard_size_bytes": 0,
}

if os.path.exists(raw_dir):
    raw_files = list(Path(raw_dir).rglob("*"))
    raw_file_entries = []
    for f in sorted(raw_files):
        if f.is_file():
            ext = f.suffix.lower()
            size = f.stat().st_size
            pipeline["raw_input"]["total_files"] += 1
            pipeline["raw_input"]["total_size_bytes"] += size
            pipeline["total_raw_size_bytes"] += size
            pipeline["raw_input"]["file_types"][ext] = (
                pipeline["raw_input"]["file_types"].get(ext, 0) + 1
            )
            raw_file_entries.append({"name": f.name, "size_bytes": size, "subdir": str(f.parent.name)})
    # Keep only summary + sample
    pipeline["raw_input"]["sample_files"] = raw_file_entries[:20]
    pipeline["raw_input"]["all_subdirs"] = sorted(set(
        str(f.parent.relative_to(raw_dir)) for f in raw_files if f.is_file()
    ))

for f in sorted(Path(PROCESSED_DIR).glob("*.parquet")):
    key = f"processed/{f.name}"
    df = dataframes.get(key)
    size = int(f.stat().st_size)
    pipeline["total_processed_size_bytes"] += size
    pipeline["processed_output"].append({
        "name": f.name,
        "size_bytes": size,
        "rows": int(df.shape[0]) if df is not None else None,
        "cols": int(df.shape[1]) if df is not None else None,
    })

for f in sorted(Path(DASHBOARD_DIR).glob("*.parquet")):
    key = f"dashboard/{f.name}"
    df = dataframes.get(key)
    size = int(f.stat().st_size)
    pipeline["total_dashboard_size_bytes"] += size
    pipeline["dashboard_output"].append({
        "name": f.name,
        "size_bytes": size,
        "rows": int(df.shape[0]) if df is not None else None,
        "cols": int(df.shape[1]) if df is not None else None,
    })

pipeline["summary_note"] = (
    "raw → processed: FBref CSV 파일과 TransferMarkt 데이터를 병합하여 "
    "parquet으로 변환. dashboard 파일은 processed 파일을 추가 집계/정제한 최종 출력물."
)

cross["pipeline_summary"] = pipeline


# ─────────────────────────────────────────
# Assemble final report
# ─────────────────────────────────────────

report = {
    "meta": {
        "generated_at": str(pd.Timestamp.now()),
        "total_files_analysed": len(parquet_sources),
        "files": list(parquet_sources.keys()),
        "purpose": "EPL 전처리 데이터 분석 보고서 (한국어 EDA 리포트용)",
    },
    "per_file_analysis": file_analyses,
    "cross_file_analysis": cross,
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, default=str, indent=2)

print(f"\n{'='*60}")
print(f"Report saved to: {REPORT_PATH}")
print(f"Total files analysed: {len(parquet_sources)}")
print(f"Report size: {os.path.getsize(REPORT_PATH):,} bytes")
print()
print("Pipeline summary:")
print(f"  Raw files: {pipeline['raw_input']['total_files']:,}")
print(f"  Raw total size: {pipeline['total_raw_size_bytes']:,} bytes")
print(f"  Processed total size: {pipeline['total_processed_size_bytes']:,} bytes")
print(f"  Dashboard total size: {pipeline['total_dashboard_size_bytes']:,} bytes")

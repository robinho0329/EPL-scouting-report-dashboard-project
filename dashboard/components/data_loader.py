"""Shared data loading functions for the dashboard."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd

from config.settings import DASHBOARD_DIR, PROCESSED_DIR, MATCH_CSV


@st.cache_data
def load_match_data() -> pd.DataFrame:
    """경기 데이터 로드. match_results.parquet 우선, 없으면 epl_final.csv 폴백."""
    parquet_path = PROCESSED_DIR / "match_results.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        df["MatchDate"] = pd.to_datetime(df["MatchDate"])
        return df
    if MATCH_CSV.exists():
        df = pd.read_csv(MATCH_CSV)
        df["MatchDate"] = pd.to_datetime(df["MatchDate"])
        return df
    return pd.DataFrame()


@st.cache_data
def load_player_season_stats() -> pd.DataFrame:
    """Load player season-level aggregated stats."""
    path = DASHBOARD_DIR / "player_season_stats.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data
def load_player_alltime_stats() -> pd.DataFrame:
    """Load player all-time career stats."""
    path = DASHBOARD_DIR / "player_alltime_stats.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data
def load_player_match_stats() -> pd.DataFrame:
    """Load player match-level stats (from crawled data)."""
    path = PROCESSED_DIR / "player_match_stats.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def get_seasons(df: pd.DataFrame) -> list[str]:
    """Extract sorted unique seasons from a DataFrame."""
    if "season" in df.columns:
        return sorted(df["season"].unique().tolist(), reverse=True)
    if "Season" in df.columns:
        return sorted(df["Season"].unique().tolist(), reverse=True)
    return []


# ============================================================
# Scout Data Loaders
# ============================================================
SCOUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "scout"
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


@st.cache_data
def load_scout_ratings() -> pd.DataFrame:
    """S1: PIS 기반 선수 평가."""
    path = SCOUT_DIR / "scout_ratings_v3.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data
def load_hidden_gems() -> pd.DataFrame:
    """S1: 숨은 보석 선수 목록."""
    path = SCOUT_DIR / "hidden_gems_v3.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data
def load_undervalued() -> pd.DataFrame:
    """S2: 저평가 선수."""
    path = SCOUT_DIR / "s2_v4_undervalued.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data
def load_overvalued() -> pd.DataFrame:
    """S2: 고평가 선수."""
    path = SCOUT_DIR / "s2_v4_overvalued.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data
def load_s2_transfer_targets() -> pd.DataFrame:
    """S2: 2025 이적 시장 저평가 영입 후보 (value_ratio > 1.0, 2024/25 시즌)."""
    path = SCOUT_DIR / "s2_v4_2025_transfer_targets.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data
def load_clusters() -> pd.DataFrame:
    """S3 v4: 포지션별 분리 클러스터링 아키타입 (FW/MID/DEF/GK 독립 군집)."""
    path = SCOUT_DIR / "cluster_assignments_v4.parquet"
    if path.exists():
        return pd.read_parquet(path)
    # v3 폴백
    path_v3 = SCOUT_DIR / "cluster_assignments_v3.parquet"
    if path_v3.exists():
        return pd.read_parquet(path_v3)
    return pd.DataFrame()


@st.cache_data
def load_similarity_matrix() -> pd.DataFrame:
    """S3 v4: 포지션 내 코사인 유사도 매트릭스 (최근 3시즌)."""
    path = SCOUT_DIR / "similarity_matrix_v4.parquet"
    if path.exists():
        return pd.read_parquet(path)
    # v3 폴백
    path_v3 = SCOUT_DIR / "similarity_matrix_v3.parquet"
    if path_v3.exists():
        return pd.read_parquet(path_v3)
    return pd.DataFrame()


@st.cache_data
def load_s4_reference() -> dict:
    """S4: 성장 참조 프로필."""
    import json
    path = SCOUT_DIR / "s4_reference_profiles.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_transfer_predictions() -> pd.DataFrame:
    """S5: 이적 적응 예측."""
    path = SCOUT_DIR / "transfer_predictions_v3.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data
def load_decline_predictions() -> pd.DataFrame:
    """S6: 하락세 예측."""
    path = SCOUT_DIR / "decline_predictions_v3.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data
def load_growth_predictions() -> pd.DataFrame:
    """P7 성장 곡선 예측 결과 로드."""
    path = SCOUT_DIR / "growth_predictions.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_growth_predictions_v4() -> pd.DataFrame:
    """P7 성장 곡선 v4 - XGBoost 분류 결과 (Improving/Stable/Declining + 확률)."""
    path = SCOUT_DIR / "growth_predictions_v4.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_player_profiles() -> pd.DataFrame:
    """통합 선수 프로파일 - big6_contribution, team_dependency 등 49개 피처."""
    path = SCOUT_DIR / "scout_player_profiles.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_transfer_adapt_predictions() -> pd.DataFrame:
    """P8 이적 적응도 예측 결과 로드."""
    path = SCOUT_DIR / "transfer_adapt_predictions.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_model_results(model_name: str) -> dict:
    """모델 결과 JSON 로드."""
    import json
    mapping = {
        "s1": MODELS_DIR / "s1_player_rating" / "results_v3.json",
        "s2": MODELS_DIR / "s2_market_value" / "results_summary_v4.json",
        "s3": MODELS_DIR / "s3_similarity" / "results_summary_v3.json",
        "s5": MODELS_DIR / "s5_transfer_adapt" / "results_summary_v3.json",
        "s6": MODELS_DIR / "s6_decline" / "results_summary_v3.json",
    }
    path = mapping.get(model_name)
    if path and path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_teams(df: pd.DataFrame, season: str = None) -> list[str]:
    """Extract sorted unique teams, optionally filtered by season."""
    if season and "season" in df.columns:
        df = df[df["season"] == season]

    teams = set()
    for col in ["team", "HomeTeam", "AwayTeam"]:
        if col in df.columns:
            teams.update(df[col].dropna().unique())
    return sorted(teams)

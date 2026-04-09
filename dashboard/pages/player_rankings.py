"""선수 통계 순위 페이지 - EPL 선수 통계 랭킹 대시보드.

시즌별/포지션별 선수 통계를 순위 형태로 보여주며,
다양한 필터 옵션을 제공한다.
"""

import streamlit as st
import pandas as pd

from dashboard.components.data_loader import (
    load_player_season_stats,
    load_player_alltime_stats,
    load_match_data,
    load_scout_ratings,
    get_seasons,
    get_teams,
)

EPL_PURPLE = "#37003c"
EPL_MAGENTA = "#e90052"
EPL_GREEN = "#00ff87"

# 실제 parquet 컬럼명에 맞춘 매핑 (한국어 표시명)
STAT_CATEGORIES = {
    "WAR (대체불가 가치)": "war",
    "골": "gls",
    "어시스트": "ast",
    "출전 경기": "mp",
    "출전 시간(분)": "min",
    "선발 출전": "starts",
    "골+어시스트": "g_a",
    "비PK 골": "g_pk",
    "골/90분": "gls_1",
    "어시스트/90분": "ast_1",
}


def _goto_scout_report(player: str):
    st.session_state["scout_report_player"] = player
    st.session_state["_nav_target"] = "🔍 선수 즉시 분석"
    st.rerun()


def render():
    st.title("선수 통계 순위")
    st.caption(
        "시즌별/포지션별 선수 통계를 탐색하고 비교 분석합니다. "
        "WAR 기준 정렬 시 실제 대체불가 가치 순위를 확인할 수 있습니다."
    )

    # 데이터 로드
    season_stats = load_player_season_stats()
    alltime_stats = load_player_alltime_stats()
    match_data = load_match_data()
    scout_ratings = load_scout_ratings()

    has_player_data = not season_stats.empty

    if not has_player_data:
        st.warning(
            "선수 통계 데이터가 아직 없습니다. "
            "크롤링 파이프라인을 먼저 실행하여 선수 데이터를 수집하세요."
        )
        if not match_data.empty:
            st.markdown("---")
            st.subheader("사용 가능한 경기 데이터 (epl_final.csv)")
            _show_match_data_summary(match_data)
        return

    # WAR 데이터 season_stats에 병합
    if not scout_ratings.empty and "player" in scout_ratings.columns and "war" in scout_ratings.columns:
        _war_cols = [c for c in ["player", "season", "war", "tier", "market_value"] if c in scout_ratings.columns]
        _rat_slim = scout_ratings[_war_cols].drop_duplicates(subset=["player", "season"]) if "season" in scout_ratings.columns else scout_ratings[_war_cols].drop_duplicates("player")
        if "season" in _rat_slim.columns and "season" in season_stats.columns:
            season_stats = season_stats.merge(_rat_slim, on=["player", "season"], how="left")
        # alltime에도 최신 시즌 WAR 병합
        if not alltime_stats.empty:
            _latest_war = _rat_slim.sort_values("season", ascending=False).drop_duplicates("player").drop(columns=["season"], errors="ignore")
            alltime_stats = alltime_stats.merge(_latest_war, on="player", how="left")

    # === 필터 ===
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        available_stats = {
            k: v for k, v in STAT_CATEGORIES.items()
            if v in season_stats.columns or (v == "war" and "war" in season_stats.columns)
        }
        if not available_stats:
            st.warning("데이터에서 통계 컬럼을 찾을 수 없습니다.")
            return
        stat_display = st.selectbox("정렬 기준", list(available_stats.keys()))
        stat_col = available_stats[stat_display]

    with col2:
        seasons = get_seasons(season_stats)
        season_options = ["전체 시즌"] + seasons
        selected_season = st.selectbox("시즌 선택", season_options)

    with col3:
        if selected_season == "전체 시즌":
            teams = get_teams(season_stats)
        else:
            teams = get_teams(season_stats, selected_season)
        team_options = ["전체 팀"] + teams
        selected_team = st.selectbox("팀 선택", team_options)

    with col4:
        pos_options = ["전체 포지션", "GK", "DF", "MF", "FW"]
        selected_pos = st.selectbox("포지션 선택", pos_options)

    # === 필터 적용 ===
    if selected_season == "전체 시즌":
        df = alltime_stats.copy() if not alltime_stats.empty else season_stats.copy()
    else:
        df = season_stats[season_stats["season"] == selected_season].copy()

    if selected_team != "전체 팀" and "team" in df.columns:
        df = df[df["team"] == selected_team]

    if selected_pos != "전체 포지션":
        pos_col = None
        for c in ["pos", "position"]:
            if c in df.columns:
                pos_col = c
                break
        if pos_col:
            df = df[df[pos_col].str.contains(selected_pos, case=False, na=False)]

    # === WAR Top 5 미니 카드 ===
    if "war" in df.columns and df["war"].notna().any():
        top5_war = df.dropna(subset=["war"]).nlargest(5, "war")
        st.markdown("---")
        st.markdown(f"#### 🏆 {selected_season if selected_season != '전체 시즌' else '전체'} WAR Top 5")
        war_cols = st.columns(5)
        for i, (_, row) in enumerate(top5_war.iterrows()):
            if i >= 5:
                break
            with war_cols[i]:
                p = row.get("player", "?")
                w = row.get("war", 0)
                t = row.get("team", "")
                tier = row.get("tier", "")
                st.markdown(
                    f"""<div style='background:{EPL_PURPLE};color:white;border-radius:8px;
                    padding:10px;text-align:center;border-top:3px solid {EPL_GREEN};'>
                    <div style='font-size:0.75em;color:#aaa;'>#{i+1}</div>
                    <div style='font-weight:700;font-size:0.9em;'>{p}</div>
                    <div style='font-size:0.75em;color:#ccc;'>{t}</div>
                    <div style='font-size:1.3em;font-weight:700;color:{EPL_GREEN};'>{w:.1f}</div>
                    <div style='font-size:0.7em;color:#aaa;'>{tier if tier else ""}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # === 순위 표시 ===
    if stat_col not in df.columns:
        st.info(f"'{stat_display}' 데이터가 현재 선택 조건에서 제공되지 않습니다.")
        return

    sort_df = df.dropna(subset=[stat_col]) if stat_col == "war" else df
    sort_df = sort_df.sort_values(stat_col, ascending=False).head(50)

    display_cols = ["player"]
    if "team" in sort_df.columns:
        display_cols.append("team")
    if "pos" in sort_df.columns:
        display_cols.append("pos")
    if selected_season == "전체 시즌" and "num_seasons" in sort_df.columns:
        display_cols.append("num_seasons")
    if "mp" in sort_df.columns and stat_col != "mp":
        display_cols.append("mp")
    # WAR 항상 표시 (정렬 기준 아닐 때도)
    if "war" in sort_df.columns and stat_col != "war":
        display_cols.append("war")
    if "tier" in sort_df.columns:
        display_cols.append("tier")
    display_cols.append(stat_col)

    existing_cols = [c for c in dict.fromkeys(display_cols) if c in sort_df.columns]
    display_df = sort_df[existing_cols].reset_index(drop=True)
    display_df.index = display_df.index + 1
    display_df.index.name = "순위"

    rename_map = {
        "player": "선수",
        "team": "팀",
        "pos": "포지션",
        "num_seasons": "시즌 수",
        "mp": "출전 경기",
        "war": "WAR",
        "tier": "등급",
    }
    rename_map[stat_col] = stat_display
    display_df = display_df.rename(columns=rename_map)

    # WAR 소수점 포맷
    if "WAR" in display_df.columns:
        display_df["WAR"] = display_df["WAR"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")

    st.dataframe(display_df, use_container_width=True, height=520)

    # ── 즉시 분석 연동 ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔍 선수 즉시 분석으로 이동")
    all_ranked_players = sort_df["player"].dropna().unique().tolist() if "player" in sort_df.columns else []
    link_c1, link_c2 = st.columns([3, 1])
    with link_c1:
        picked_player = st.selectbox(
            "선수 선택",
            [""] + sorted(all_ranked_players),
            key="rankings_player_link",
            label_visibility="collapsed",
        )
    with link_c2:
        if st.button("🔍 즉시 분석", use_container_width=True, key="rankings_goto_report"):
            if picked_player:
                _goto_scout_report(picked_player)
            else:
                st.warning("선수를 선택하세요.")

    # 요약 메트릭
    if not sort_df.empty:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("1위 선수", sort_df.iloc[0].get("player", "N/A"))
        with col_b:
            top_val = sort_df[stat_col].max()
            st.metric("기록", f"{top_val:.1f}" if isinstance(top_val, float) else int(top_val))
        with col_c:
            st.metric("총 선수", len(sort_df))


def _show_match_data_summary(df: pd.DataFrame):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("총 경기 수", len(df))
    with col2:
        st.metric("시즌 수", df["Season"].nunique())
    with col3:
        total_goals = df["FullTimeHomeGoals"].sum() + df["FullTimeAwayGoals"].sum()
        st.metric("총 골 수", f"{total_goals:,}")

    st.subheader("시즌별 골 수")
    season_goals = (
        df.groupby("Season")
        .apply(lambda x: x["FullTimeHomeGoals"].sum() + x["FullTimeAwayGoals"].sum())
        .reset_index(name="골")
    )
    st.bar_chart(season_goals.set_index("Season")["골"])

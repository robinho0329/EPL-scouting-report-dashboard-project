"""역대 기록 페이지 - EPL 역대 기록 및 마일스톤."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.components.data_loader import (
    load_player_alltime_stats,
    load_match_data,
    load_scout_ratings,
)

EPL_PURPLE = "#37003c"
EPL_MAGENTA = "#e90052"
EPL_GREEN = "#00ff87"
EPL_CYAN = "#04f5ff"


def _goto_scout_report(player: str):
    st.session_state["scout_report_player"] = player
    st.session_state["_nav_target"] = "🔍 선수 즉시 분석"
    st.rerun()


def render():
    st.title("역대 기록")
    st.caption("EPL 2000~2025 경기 데이터 기반 역대 기록 및 스카우팅 레퍼런스")

    alltime = load_player_alltime_stats()
    match_data = load_match_data()
    scout_ratings = load_scout_ratings()

    # ── 🏆 역대 PIS 명전 (스카우팅 핵심 섹션) ─────────────────────────────
    if not scout_ratings.empty and "war" in scout_ratings.columns and "player" in scout_ratings.columns:
        st.markdown("### 🏆 역대 PIS 명전")
        st.caption("전 시즌 PIS 기준 역대 최고 선수들 — 스카우팅 레퍼런스 기준")

        # 선수별 최고 WAR 시즌 기준 집계
        war_best = (
            scout_ratings.dropna(subset=["war"])
            .sort_values("war", ascending=False)
            .drop_duplicates("player")
        )
        cols_show = [c for c in ["player", "team", "pos_group", "season", "war", "tier", "market_value"] if c in war_best.columns]
        top_war_all = war_best[cols_show].head(20).reset_index(drop=True)
        top_war_all.index += 1
        top_war_all.index.name = "순위"

        # 포맷
        if "market_value" in top_war_all.columns:
            top_war_all["market_value"] = top_war_all["market_value"].apply(
                lambda x: f"€{x/1e6:.1f}M" if pd.notna(x) and x >= 1e6 else ("-" if pd.isna(x) else f"€{x/1e3:.0f}K")
            )
        if "war" in top_war_all.columns:
            top_war_all["war"] = top_war_all["war"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")

        rename_m = {"player": "선수", "team": "팀", "pos_group": "포지션", "season": "최고 시즌", "war": "PIS", "tier": "등급", "market_value": "시장가치"}
        top_war_all = top_war_all.rename(columns=rename_m)
        st.dataframe(top_war_all, use_container_width=True, height=460)

        # PIS 명전 바 차트 (Top 15)
        top15 = war_best.head(15).sort_values("war", ascending=True)
        if len(top15):
            colors = [EPL_MAGENTA if i >= len(top15) - 3 else EPL_PURPLE for i in range(len(top15))]
            fig = go.Figure(go.Bar(
                x=top15["war"].values,
                y=top15["player"].values,
                orientation="h",
                marker_color=colors,
                text=[f"{v:.1f}" for v in top15["war"].values],
                textposition="outside",
            ))
            fig.update_layout(
                xaxis=dict(title="PIS (최고 시즌 기준)"),
                yaxis=dict(automargin=True),
                height=460, margin=dict(l=0, r=80, t=10, b=10),
                plot_bgcolor="#1a1a2e", paper_bgcolor="#0d0d1a", font_color="#ffffff",
            )
            st.plotly_chart(fig, use_container_width=True, theme=None)
            st.caption("💡 붉은 막대 = 역대 Top 3. PIS는 0~100 백분위 (100 = 리그 절대 최고)")

        # 즉시 분석 연동
        st.markdown("---")
        war_players = war_best["player"].dropna().unique().tolist() if "player" in war_best.columns else []
        st.markdown("#### 🔍 역대 명전 선수 즉시 분석")
        wl1, wl2 = st.columns([3, 1])
        with wl1:
            war_picked = st.selectbox("선수 선택", [""] + sorted(war_players), key="records_war_player", label_visibility="collapsed")
        with wl2:
            if st.button("🔍 즉시 분석", key="records_war_goto", use_container_width=True):
                if war_picked:
                    _goto_scout_report(war_picked)
                else:
                    st.warning("선수를 선택하세요.")
        st.markdown("---")

    # ── 경기 기록 ──────────────────────────────────────────────────────────
    if not match_data.empty:
        st.markdown("### 📊 경기 기록 (2000-2025)")

        c1, c2, c3, c4 = st.columns(4)
        total_matches = len(match_data)
        total_goals = int(match_data["FullTimeHomeGoals"].sum() + match_data["FullTimeAwayGoals"].sum())
        match_data = match_data.copy()
        match_data["TotalGoals"] = match_data["FullTimeHomeGoals"] + match_data["FullTimeAwayGoals"]
        max_goals_match = match_data.loc[match_data["TotalGoals"].idxmax()]

        c1.metric("총 경기", f"{total_matches:,}")
        c2.metric("총 골", f"{total_goals:,}")
        c3.metric("경기당 평균 골", f"{total_goals/total_matches:.2f}")
        c4.metric(
            "최다골 경기",
            f"{int(max_goals_match['TotalGoals'])} goals",
            f"{max_goals_match['HomeTeam']} vs {max_goals_match['AwayTeam']}",
        )
        st.caption("💡 2000년부터 현재까지 EPL 전체 경기 데이터 기반 역대 기록")

        # 팀 기록
        st.markdown("### 🏟️ 팀 기록")
        home_season = (
            match_data.groupby(["Season", "HomeTeam"])["FullTimeHomeGoals"]
            .sum().reset_index().sort_values("FullTimeHomeGoals", ascending=False)
        )
        away_season = (
            match_data.groupby(["Season", "AwayTeam"])["FullTimeAwayGoals"]
            .sum().reset_index().sort_values("FullTimeAwayGoals", ascending=False)
        )

        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown("**시즌 최다 홈 득점**")
            top_home = home_season.head(10).rename(columns={"HomeTeam": "팀", "FullTimeHomeGoals": "골", "Season": "시즌"})
            top_home.index = range(1, len(top_home) + 1)
            st.dataframe(top_home, use_container_width=True)

        with tc2:
            st.markdown("**시즌 최다 원정 득점**")
            top_away = away_season.head(10).rename(columns={"AwayTeam": "팀", "FullTimeAwayGoals": "골", "Season": "시즌"})
            top_away.index = range(1, len(top_away) + 1)
            st.dataframe(top_away, use_container_width=True)

        # 최대 점수차 승리
        st.markdown("### 🎯 최대 점수차 승리")
        match_data["GoalDiff"] = abs(match_data["FullTimeHomeGoals"] - match_data["FullTimeAwayGoals"])
        biggest = match_data.nlargest(10, "GoalDiff")[
            ["Season", "MatchDate", "HomeTeam", "AwayTeam", "FullTimeHomeGoals", "FullTimeAwayGoals", "GoalDiff"]
        ].reset_index(drop=True)
        biggest.index += 1
        biggest.columns = ["시즌", "날짜", "홈팀", "원정팀", "홈골", "원정골", "점수차"]
        st.dataframe(biggest, use_container_width=True)

    # ── 선수 역대 기록 ─────────────────────────────────────────────────────
    if not alltime.empty:
        st.markdown("---")
        st.markdown("### 👤 선수 역대 기록 (TOP 10)")
        st.caption("💡 크롤링 데이터 기반 선수별 통산 기록. 이적/복귀 포함 전 시즌 합산")

        # WAR 병합 (최신 시즌 기준)
        if not scout_ratings.empty and "player" in scout_ratings.columns and "war" in scout_ratings.columns:
            _latest = scout_ratings.sort_values("season", ascending=False).drop_duplicates("player")[["player", "war", "tier"]] if "season" in scout_ratings.columns else scout_ratings[["player", "war", "tier"]].drop_duplicates("player")
            alltime = alltime.merge(_latest, on="player", how="left")

        record_stats = {
            "최다 득점": "gls",
            "최다 어시스트": "ast",
            "최다 출전": "mp",
            "최다 출전시간": "min",
        }

        cols = st.columns(2)
        for i, (title, col_name) in enumerate(record_stats.items()):
            if col_name not in alltime.columns:
                continue
            with cols[i % 2]:
                st.markdown(f"**{title}**")
                show_cols = [c for c in ["player", col_name, "war", "tier"] if c in alltime.columns]
                top = alltime.nlargest(10, col_name)[show_cols].copy()
                top.index = range(1, len(top) + 1)
                rename_r = {"player": "선수", col_name: title.split()[-1], "war": "PIS", "tier": "등급"}
                top = top.rename(columns=rename_r)
                if "PIS" in top.columns:
                    top["PIS"] = top["PIS"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
                st.dataframe(top, use_container_width=True)

        # 선수 역대 기록 → 즉시 분석 연동
        st.markdown("---")
        st.markdown("#### 🔍 선수 역대 기록 즉시 분석")
        alltime_players = alltime["player"].dropna().unique().tolist() if "player" in alltime.columns else []
        al1, al2 = st.columns([3, 1])
        with al1:
            alltime_picked = st.selectbox("선수 선택", [""] + sorted(alltime_players), key="records_alltime_player", label_visibility="collapsed")
        with al2:
            if st.button("🔍 즉시 분석", key="records_alltime_goto", use_container_width=True):
                if alltime_picked:
                    _goto_scout_report(alltime_picked)
                else:
                    st.warning("선수를 선택하세요.")
    else:
        if match_data.empty:
            st.warning("데이터가 없습니다.")

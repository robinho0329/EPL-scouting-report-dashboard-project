"""Season Overview Page - 시즌별 통계 및 트렌드."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.components.data_loader import (
    load_match_data,
    load_player_season_stats,
    load_scout_ratings,
    get_seasons,
)

EPL_PURPLE = "#37003c"
EPL_MAGENTA = "#e90052"
EPL_GREEN = "#00ff87"


def _goto_scout_report(player: str):
    st.session_state["scout_report_player"] = player
    st.session_state["_nav_target"] = "🔍 선수 즉시 분석"
    st.rerun()


def render():
    st.title("시즌 개요")
    st.caption("시즌별 리그 현황과 핵심 선수 성과를 한눈에 파악합니다.")

    match_data = load_match_data()
    player_stats = load_player_season_stats()
    scout_ratings = load_scout_ratings()

    if match_data.empty:
        st.warning("경기 데이터가 없습니다.")
        return

    # ── 시즌 선택 ──────────────────────────────────────────────────────────
    all_match_seasons = get_seasons(match_data)
    selected = st.selectbox("시즌 선택", all_match_seasons)
    season_df = match_data[match_data["Season"] == selected]

    # ── 핵심 지표 ──────────────────────────────────────────────────────────
    st.markdown("### 📊 핵심 지표")
    total_goals = int(season_df["FullTimeHomeGoals"].sum() + season_df["FullTimeAwayGoals"].sum())
    avg_goals = total_goals / len(season_df) if len(season_df) > 0 else 0
    home_wins = int((season_df["FullTimeResult"] == "H").sum())
    away_wins = int((season_df["FullTimeResult"] == "A").sum())
    draws = int((season_df["FullTimeResult"] == "D").sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("총 경기", len(season_df))
    c2.metric("총 골", total_goals)
    c3.metric("경기당 평균 골", f"{avg_goals:.2f}")
    c4.metric("홈승 / 원정승", f"{home_wins} / {away_wins}")
    c5.metric("무승부", draws)
    st.caption("💡 평균 골이 높을수록 공격적인 시즌. 영입 타겟 활약 배경 지표로 참고")

    # ── 리그 순위표 ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏆 리그 순위표")
    league_table = _build_league_table(season_df)
    st.dataframe(league_table, use_container_width=True, height=450)
    st.caption("💡 실제 승점 기준 정렬. 강팀 여부 파악 → 해당 팀 선수 영입 난이도 판단에 활용")

    # ── 시즌 우수 선수 + WAR 병합 ─────────────────────────────────────────
    if not player_stats.empty:
        ps = player_stats[player_stats["season"] == selected].copy()

        # WAR 병합
        if not scout_ratings.empty and "player" in scout_ratings.columns and "war" in scout_ratings.columns:
            rat_season = scout_ratings[scout_ratings["season"] == selected][["player", "war", "tier"]].drop_duplicates("player") if "season" in scout_ratings.columns else scout_ratings[["player", "war", "tier"]].drop_duplicates("player")
            ps = ps.merge(rat_season, on="player", how="left")

        if not ps.empty:
            st.markdown("---")
            st.markdown("### ⭐ 시즌 우수 선수")
            st.caption("원하는 선수를 선택하고 즉시 분석 버튼으로 상세 리포트를 확인하세요.")

            tc1, tc2, tc3 = st.columns(3)

            with tc1:
                st.markdown("**득점왕 Top 5**")
                if "gls" in ps.columns:
                    cols_show = [c for c in ["player", "team", "gls", "war"] if c in ps.columns]
                    top_scorers = ps.nlargest(5, "gls")[cols_show].copy()
                    top_scorers = top_scorers.rename(columns={"player": "선수", "team": "팀", "gls": "골", "war": "PIS"})
                    if "PIS" in top_scorers.columns:
                        top_scorers["PIS"] = top_scorers["PIS"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
                    st.dataframe(top_scorers, hide_index=True, use_container_width=True)

            with tc2:
                st.markdown("**어시스트왕 Top 5**")
                if "ast" in ps.columns:
                    cols_show = [c for c in ["player", "team", "ast", "war"] if c in ps.columns]
                    top_assists = ps.nlargest(5, "ast")[cols_show].copy()
                    top_assists = top_assists.rename(columns={"player": "선수", "team": "팀", "ast": "어시스트", "war": "PIS"})
                    if "PIS" in top_assists.columns:
                        top_assists["PIS"] = top_assists["PIS"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
                    st.dataframe(top_assists, hide_index=True, use_container_width=True)

            with tc3:
                st.markdown("**PIS 상위 Top 5**")
                if "war" in ps.columns:
                    cols_show = [c for c in ["player", "team", "war", "tier"] if c in ps.columns]
                    top_war = ps.dropna(subset=["war"]).nlargest(5, "war")[cols_show].copy()
                    top_war = top_war.rename(columns={"player": "선수", "team": "팀", "war": "PIS", "tier": "등급"})
                    top_war["PIS"] = top_war["PIS"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
                    st.dataframe(top_war, hide_index=True, use_container_width=True)
                else:
                    st.markdown("**다출전 선수 Top 5**")
                    if "mp" in ps.columns:
                        cols_show = [c for c in ["player", "team", "mp"] if c in ps.columns]
                        top_apps = ps.nlargest(5, "mp")[cols_show].copy()
                        top_apps = top_apps.rename(columns={"player": "선수", "team": "팀", "mp": "경기"})
                        st.dataframe(top_apps, hide_index=True, use_container_width=True)

            st.caption("💡 PIS(기여 점수) 높은 선수가 실제 팀 기여도가 높습니다. 실제 영입 판단은 PIS 기준으로 하세요.")

            # ── 즉시 분석 연동 ─────────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 🔍 선수 즉시 분석으로 이동")
            all_players_season = sorted(ps["player"].dropna().unique().tolist()) if "player" in ps.columns else []
            link_col1, link_col2 = st.columns([3, 1])
            with link_col1:
                selected_player_link = st.selectbox(
                    "선수 선택", [""] + all_players_season,
                    key="season_overview_player_link",
                    label_visibility="collapsed",
                    placeholder="선수를 선택하세요"
                )
            with link_col2:
                if st.button("🔍 즉시 분석", use_container_width=True, key="season_goto_report"):
                    if selected_player_link:
                        _goto_scout_report(selected_player_link)
                    else:
                        st.warning("선수를 선택하세요.")

    # ── 시즌별 평균 골 추이 차트 ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 시즌별 평균 골 추이")
    if len(match_data) > 0:
        season_trend = (
            match_data.groupby("Season")
            .apply(lambda x: (x["FullTimeHomeGoals"].sum() + x["FullTimeAwayGoals"].sum()) / len(x))
            .reset_index(name="경기당 평균 골")
            .sort_values("Season")
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=season_trend["Season"],
            y=season_trend["경기당 평균 골"],
            mode="lines+markers+text",
            text=[f"{v:.2f}" for v in season_trend["경기당 평균 골"]],
            textposition="top center",
            line=dict(color=EPL_MAGENTA, width=2),
            marker=dict(size=8, color=EPL_PURPLE),
            fill="tozeroy",
            fillcolor="rgba(233,0,82,0.08)",
        ))
        # 선택 시즌 강조
        sel_val = season_trend[season_trend["Season"] == selected]["경기당 평균 골"]
        if not sel_val.empty:
            fig.add_trace(go.Scatter(
                x=[selected], y=[sel_val.values[0]],
                mode="markers",
                marker=dict(size=14, color=EPL_GREEN, symbol="star"),
                name=f"선택 시즌: {selected}",
                showlegend=True,
            ))
        fig.update_layout(
            xaxis_title="시즌", yaxis_title="경기당 평균 골",
            height=320, margin=dict(t=20, b=40),
            plot_bgcolor="#1a1a2e", paper_bgcolor="#0d0d1a", font_color="#ffffff",
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)
        st.caption("💡 별표(★)가 현재 선택 시즌. 트렌드로 공격적/수비적 리그 흐름 파악 가능")


def _build_league_table(df: pd.DataFrame) -> pd.DataFrame:
    teams = set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique())
    records = []

    for team in teams:
        home = df[df["HomeTeam"] == team]
        away = df[df["AwayTeam"] == team]

        hw = int((home["FullTimeResult"] == "H").sum())
        hd = int((home["FullTimeResult"] == "D").sum())
        hl = int((home["FullTimeResult"] == "A").sum())
        aw = int((away["FullTimeResult"] == "A").sum())
        ad = int((away["FullTimeResult"] == "D").sum())
        al = int((away["FullTimeResult"] == "H").sum())

        gf = int(home["FullTimeHomeGoals"].sum() + away["FullTimeAwayGoals"].sum())
        ga = int(home["FullTimeAwayGoals"].sum() + away["FullTimeHomeGoals"].sum())

        played = len(home) + len(away)
        wins = hw + aw
        draws_total = hd + ad
        losses = hl + al
        points = wins * 3 + draws_total

        records.append({
            "팀": team, "경기": played, "승": wins, "무": draws_total, "패": losses,
            "득점": gf, "실점": ga, "골득실": gf - ga, "승점": points,
        })

    table = pd.DataFrame(records)
    table = table.sort_values(["승점", "골득실", "득점"], ascending=[False, False, False])
    table = table.reset_index(drop=True)
    table.index += 1
    table.index.name = "순위"
    return table

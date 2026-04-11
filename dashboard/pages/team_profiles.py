"""팀 프로파일 - 구단별 선수단 구성 및 WAR 분석.

스카우트가 영입 후보 팀의 선수단 구성, 포지션별 WAR, 세대 교체 필요성을
한눈에 파악할 수 있는 팀 분석 페이지.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.data_loader import load_scout_ratings
from dashboard.utils.image_utils import get_player_image_b64, get_team_logo_b64

EPL_PURPLE = "#37003c"
EPL_MAGENTA = "#e90052"
EPL_GREEN = "#00ff87"
EPL_CYAN = "#04f5ff"

# 현재 EPL 20개 팀 (2024/25 기준)
EPL_TEAMS_2425 = [
    "Arsenal", "Aston Villa", "Brentford", "Brighton", "Chelsea",
    "Crystal Palace", "Everton", "Fulham", "Ipswich", "Leicester",
    "Liverpool", "Man City", "Man United", "Newcastle", "Nott'm Forest",
    "Southampton", "Tottenham", "West Ham", "Wolves", "Bournemouth",
]

POS_ORDER = {"GK": 0, "DF": 1, "MF": 2, "FW": 3}
POS_COLORS = {
    "GK": EPL_CYAN,
    "DF": "#4a90d9",
    "MF": EPL_GREEN,
    "FW": EPL_MAGENTA,
}


@st.cache_data(ttl=3600)
def _load_team_profiles() -> pd.DataFrame:
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent.parent / "data" / "scout" / "scout_team_profiles.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _simplify_pos(pos: str) -> str:
    """복합 포지션을 주 포지션으로 단순화."""
    if pd.isna(pos):
        return "Unknown"
    p = str(pos).split(",")[0].strip()
    return p if p in ["GK", "DF", "MF", "FW"] else p[:2]


def render():
    st.title("🏟️ 팀 프로파일")
    st.caption("구단별 선수단 구성, 포지션별 WAR 강점/약점, 세대 교체 현황을 분석합니다.")

    ratings = load_scout_ratings()
    team_profiles = _load_team_profiles()

    if ratings.empty:
        st.error("scout_ratings 데이터를 불러올 수 없습니다.")
        return

    # 최신 시즌
    latest_season = ratings["season"].max() if "season" in ratings.columns else None

    # 팀 목록: 최신 시즌 팀 우선
    if latest_season:
        current_teams = sorted(ratings[ratings["season"] == latest_season]["team"].dropna().unique().tolist())
    else:
        current_teams = sorted(ratings["team"].dropna().unique().tolist())

    # 현재 EPL 팀 우선 정렬
    epl_first = [t for t in EPL_TEAMS_2425 if t in current_teams]
    others = [t for t in current_teams if t not in EPL_TEAMS_2425]
    team_list = epl_first + others

    # ── 팀 선택 ────────────────────────────────────────────────────────
    tp_c1, tp_c2 = st.columns([2, 3])
    with tp_c1:
        sel_team = st.selectbox("팀 선택", team_list, key="team_profile_select")
    with tp_c2:
        if latest_season:
            all_seasons = sorted(ratings["season"].dropna().unique().tolist(), reverse=True)
            sel_season = st.selectbox("시즌", all_seasons, key="team_profile_season")
        else:
            sel_season = None

    st.markdown("---")

    if not sel_team:
        st.info("팀을 선택하세요.")
        return

    # 해당 팀 + 시즌 데이터
    if sel_season:
        team_df = ratings[(ratings["team"] == sel_team) & (ratings["season"] == sel_season)].copy()
    else:
        team_df = ratings[ratings["team"] == sel_team].copy()

    if team_df.empty:
        st.warning(f"{sel_team} ({sel_season}) 데이터가 없습니다.")
        return

    # 포지션 단순화
    team_df["pos_simple"] = team_df["pos"].apply(_simplify_pos)

    # ── 팀 헤더 ────────────────────────────────────────────────────────
    logo_b64 = get_team_logo_b64(sel_team, size=(72, 72))
    avg_war = team_df["war"].mean() if "war" in team_df.columns else None
    avg_age = team_df["age"].mean() if "age" in team_df.columns else None
    total_mv = team_df["market_value"].sum() if "market_value" in team_df.columns else None
    n_players = len(team_df)

    header_cols = st.columns([1, 4])
    with header_cols[0]:
        if logo_b64:
            st.markdown(
                f'<img src="data:image/png;base64,{logo_b64}" style="width:72px;height:72px;object-fit:contain;">',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div style="font-size:3em;">🏟️</div>', unsafe_allow_html=True)
    with header_cols[1]:
        st.markdown(f"## {sel_team}")
        st.caption(f"{sel_season} 시즌 · {n_players}명")

    hm1, hm2, hm3, hm4 = st.columns(4)
    hm1.metric("평균 WAR", f"{avg_war:.1f}" if avg_war else "-")
    hm2.metric("평균 연령", f"{avg_age:.1f}" if avg_age else "-")
    hm3.metric("총 스쿼드 가치", f"€{total_mv/1_000_000:.0f}M" if total_mv else "-")
    hm4.metric("스쿼드 규모", n_players)

    st.markdown("---")

    # ── 포지션별 WAR 분석 ──────────────────────────────────────────────
    st.markdown("### 📊 포지션별 WAR 강점 분석")

    pos_war = (
        team_df.groupby("pos_simple")["war"]
        .agg(["mean", "max", "count"])
        .reset_index()
        .rename(columns={"pos_simple": "포지션", "mean": "평균 WAR", "max": "최고 WAR", "count": "선수 수"})
    )
    pos_war["정렬"] = pos_war["포지션"].map(POS_ORDER).fillna(9)
    pos_war = pos_war.sort_values("정렬")

    # 수평 막대 차트
    fig_pos = go.Figure()
    for _, prow in pos_war.iterrows():
        pos_name = prow["포지션"]
        color = POS_COLORS.get(pos_name, "#888")
        fig_pos.add_trace(go.Bar(
            x=[prow["평균 WAR"]],
            y=[pos_name],
            orientation="h",
            marker_color=color,
            name=pos_name,
            text=[f"{prow['평균 WAR']:.1f}"],
            textposition="outside",
            showlegend=False,
        ))

    # EPL 평균 참조선
    if latest_season and "war" in ratings.columns:
        epl_avg_war = ratings[ratings["season"] == sel_season]["war"].mean() if sel_season else ratings["war"].mean()
        fig_pos.add_vline(
            x=epl_avg_war, line_dash="dash", line_color="#888",
            annotation_text=f"EPL 평균 {epl_avg_war:.1f}",
            annotation_position="top right",
        )

    fig_pos.update_layout(
        paper_bgcolor="#0d0d1a",
        plot_bgcolor="#1a1a2e",
        font_color="#fff",
        margin=dict(t=20, b=20, l=60, r=60),
        height=220,
        xaxis_title="평균 WAR",
        yaxis_title="",
    )
    st.plotly_chart(fig_pos, use_container_width=True, theme=None)

    # ── TOP 5 선수 카드 ─────────────────────────────────────────────────
    st.markdown("### 🌟 PIS Top 5 선수")
    top5 = team_df.sort_values("war", ascending=False).head(5) if "war" in team_df.columns else team_df.head(5)

    t5_cols = st.columns(5)
    for i, (_, row) in enumerate(top5.iterrows()):
        player = row.get("player", "")
        war = row.get("war", None)
        pos = row.get("pos", "")
        tier = row.get("tier", "")
        age = row.get("age", None)
        mv = row.get("market_value", None)

        pos_simple = _simplify_pos(pos)
        card_color = POS_COLORS.get(pos_simple, "#888")
        img_b64 = get_player_image_b64(player, size=(80, 80))

        with t5_cols[i]:
            img_html = (
                f'<img src="data:image/jpeg;base64,{img_b64}" '
                f'style="width:64px;height:64px;object-fit:cover;border-radius:50%;'
                f'border:3px solid {card_color};">'
                if img_b64 else
                '<div style="width:64px;height:64px;border-radius:50%;background:#2a2a4a;'
                'display:flex;align-items:center;justify-content:center;font-size:28px;">👤</div>'
            )
            war_str = f"{war:.0f}" if war and not pd.isna(war) else "-"
            age_str = f"{int(age)}세" if age and not pd.isna(age) else ""
            mv_str = f"€{mv/1_000_000:.1f}M" if mv and not pd.isna(mv) else ""

            st.markdown(
                f"""<div style='text-align:center;background:#1a1a2e;border-radius:10px;
                padding:10px 6px;border-top:3px solid {card_color};'>
                <div style='display:flex;justify-content:center;margin-bottom:6px;'>{img_html}</div>
                <div style='font-weight:700;color:#fff;font-size:0.85em;word-break:break-word;'>{player}</div>
                <div style='color:{card_color};font-size:0.75em;'>{pos_simple} {f"· {tier}" if tier else ""}</div>
                <div style='color:#FFD700;font-size:1.1em;font-weight:700;'>WAR {war_str}</div>
                <div style='color:#aaa;font-size:0.72em;'>{age_str}{f" · {mv_str}" if mv_str else ""}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            if st.button("🔍", key=f"tp_goto_{player}_{i}", help=f"{player} 즉시 분석", use_container_width=True):
                st.session_state["scout_report_player"] = player
                st.session_state["_nav_target"] = "🔍 선수 즉시 분석"
                st.rerun()

    st.markdown("---")

    # ── 연령 분포 + WAR 버블 차트 ────────────────────────────────────────
    st.markdown("### 📈 연령-WAR 분포")
    if "age" in team_df.columns and "war" in team_df.columns:
        bubble_df = team_df[team_df["war"].notna() & team_df["age"].notna()].copy().reset_index(drop=True)
        bubble_df["pos_simple"] = bubble_df["pos"].apply(_simplify_pos)
        # NaN-safe market value: convert to plain float list (Plotly requires finite positive values)
        if "market_value" in bubble_df.columns:
            mv_raw = pd.to_numeric(bubble_df["market_value"], errors="coerce")
            mv_sizes = [max(float(v) / 1_000_000, 1.0) if pd.notna(v) else 5.0 for v in mv_raw]
        else:
            mv_sizes = [10.0] * len(bubble_df)
        bubble_df["mv_m"] = mv_sizes

        # go.Scatter로 직접 구성 (px.scatter size 파라미터 NaN 민감도 우회)
        fig_bubble = go.Figure()
        for pos, color in POS_COLORS.items():
            mask = bubble_df["pos_simple"] == pos
            sub = bubble_df[mask]
            if sub.empty:
                continue
            fig_bubble.add_trace(go.Scatter(
                x=sub["age"].tolist(),
                y=sub["war"].tolist(),
                mode="markers+text",
                name=pos,
                text=sub["player"].tolist(),
                textposition="top center",
                marker=dict(
                    size=[max(s * 1.5, 8) for s in sub["mv_m"].tolist()],
                    sizemode="area",
                    sizeref=max(mv_sizes) / 40 ** 2 if mv_sizes else 1,
                    color=color,
                    opacity=0.85,
                ),
                hovertemplate="<b>%{text}</b><br>나이: %{x}<br>WAR: %{y:.1f}<extra></extra>",
            ))
        fig_bubble.add_hline(y=50, line_dash="dot", line_color="#888", annotation_text="WAR 50 기준선")
        fig_bubble.update_layout(
            paper_bgcolor="#0d0d1a",
            plot_bgcolor="#1a1a2e",
            font_color="#fff",
            xaxis_title="나이",
            yaxis_title="PIS (포지션 내 기여 백분위)",
            margin=dict(t=20, b=20, l=20, r=20),
            height=400,
            legend_title="포지션",
        )
        st.plotly_chart(fig_bubble, use_container_width=True, theme=None)
        st.caption("💡 버블 크기 = 시장가치. 우측 상단(나이 젊고 PIS 높음) = 핵심 자산. 좌측 하단 = 개발 필요 자원.")

    # ── 전체 스쿼드 테이블 ──────────────────────────────────────────────
    with st.expander("📋 전체 스쿼드 테이블"):
        squad_cols = [c for c in ["player", "pos", "age", "war", "tier", "market_value", "total_min", "goals_p90", "assists_p90"] if c in team_df.columns]
        squad_disp = team_df[squad_cols].sort_values("war", ascending=False).copy()
        squad_rename = {
            "player": "선수", "pos": "포지션", "age": "나이",
            "war": "PIS", "tier": "등급", "market_value": "시장가치(€M)",
            "total_min": "출전 분", "goals_p90": "골/90분", "assists_p90": "어시/90분",
        }
        squad_disp = squad_disp.rename(columns={k: v for k, v in squad_rename.items() if k in squad_disp.columns})
        if "시장가치(€M)" in squad_disp.columns:
            squad_disp["시장가치(€M)"] = squad_disp["시장가치(€M)"].apply(lambda x: round(x / 1_000_000, 2) if pd.notna(x) else None)
        st.dataframe(squad_disp.reset_index(drop=True), use_container_width=True, hide_index=True)

    # ── 팀 통계 이력 (season_profile) ──────────────────────────────────
    if not team_profiles.empty and "team" in team_profiles.columns:
        team_hist = team_profiles[team_profiles["team"] == sel_team].copy()
        if not team_hist.empty and "season" in team_hist.columns:
            st.markdown("### 📅 시즌별 성적 추이")
            team_hist = team_hist.sort_values("season")

            fig_hist = go.Figure()
            if "points" in team_hist.columns:
                fig_hist.add_trace(go.Scatter(
                    x=team_hist["season"], y=team_hist["points"],
                    mode="lines+markers",
                    name="승점",
                    line=dict(color=EPL_GREEN, width=2),
                    marker=dict(size=6),
                ))
            if "total_goals_for" in team_hist.columns:
                fig_hist.add_trace(go.Scatter(
                    x=team_hist["season"], y=team_hist["total_goals_for"],
                    mode="lines+markers",
                    name="득점",
                    line=dict(color=EPL_MAGENTA, width=2),
                    marker=dict(size=6),
                    yaxis="y2",
                ))
            fig_hist.update_layout(
                paper_bgcolor="#0d0d1a",
                plot_bgcolor="#1a1a2e",
                font_color="#fff",
                margin=dict(t=20, b=40, l=20, r=40),
                height=280,
                xaxis=dict(tickangle=45),
                yaxis=dict(title="승점", color=EPL_GREEN),
                yaxis2=dict(title="득점", color=EPL_MAGENTA, overlaying="y", side="right"),
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig_hist, use_container_width=True, theme=None)

    # ── 포지션 공백 분석 ────────────────────────────────────────────────
    st.markdown("### ⚠️ 포지션 공백 & 보강 포인트")
    pos_gap = pos_war.copy()
    epl_pos_avg = {}
    if sel_season and "war" in ratings.columns:
        sel_s_ratings = ratings[ratings["season"] == sel_season].copy()
        sel_s_ratings["pos_simple"] = sel_s_ratings["pos"].apply(_simplify_pos)
        epl_pos_avg = sel_s_ratings.groupby("pos_simple")["war"].mean().to_dict()

    gap_rows = []
    for _, grow in pos_gap.iterrows():
        pos_name = grow["포지션"]
        team_avg = grow["평균 WAR"]
        epl_avg = epl_pos_avg.get(pos_name, None)
        if epl_avg:
            gap = team_avg - epl_avg
            badge = "🟢 강점" if gap > 5 else ("🟡 보통" if gap > -5 else "🔴 약점")
            gap_rows.append({
                "포지션": pos_name,
                "팀 평균 WAR": round(team_avg, 1),
                "EPL 평균 WAR": round(epl_avg, 1),
                "격차": f"{gap:+.1f}",
                "평가": badge,
            })

    if gap_rows:
        st.dataframe(pd.DataFrame(gap_rows), use_container_width=True, hide_index=True)
        st.caption("💡 🔴 약점 포지션 = 영입 우선순위. S2 탐색기에서 해당 포지션 저평가 선수를 찾아보세요.")

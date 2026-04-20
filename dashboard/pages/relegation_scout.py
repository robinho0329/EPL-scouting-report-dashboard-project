"""강등권 이적 기회 탐색기

강등 위험 구단의 선수 중 이적 타이밍이 좋은 선수를 발굴하는 스카우트 도구.
P3 강등 예측 결과 + S1 PIS 데이터를 결합.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.data_loader import load_scout_ratings

# EPL 다크테마 색상
EPL_PURPLE = "#37003c"
EPL_MAGENTA = "#e90052"
EPL_GREEN = "#00ff87"
EPL_CYAN = "#04f5ff"

# 2024/25 시즌 기준 강등 위험 팀 (강등 위험도 0~1)
RELEGATION_RISK_TEAMS = {
    "Southampton": 0.95,   # 강등 확정
    "Leicester": 0.88,
    "Ipswich": 0.72,
    "Wolves": 0.45,
    "Crystal Palace": 0.30,
    "Everton": 0.22,
    "Nottm Forest": 0.18,
}


def _risk_color(risk: float) -> str:
    """강등 위험도에 따른 색상 반환."""
    if risk >= 0.70:
        return EPL_MAGENTA
    elif risk >= 0.40:
        return "#ff6b35"
    else:
        return EPL_GREEN


def _risk_label(risk: float) -> str:
    """강등 위험도 레이블 반환."""
    if risk >= 0.70:
        return "위험"
    elif risk >= 0.40:
        return "주의"
    else:
        return "경계"


def render():
    st.title("강등권 이적 기회 탐색기")
    st.caption("강등 위험 구단의 이적 가치 높은 선수를 선제적으로 발굴합니다.")

    # 데이터 로드
    ratings = load_scout_ratings()

    if ratings.empty:
        st.error("선수 평가 데이터(scout_ratings_v3.parquet)를 불러올 수 없습니다.")
        return

    # 2024/25 시즌 필터링 (없으면 최신 시즌)
    if "season" in ratings.columns:
        available_seasons = sorted(ratings["season"].unique(), reverse=True)
        target_season = "2024/25" if "2024/25" in available_seasons else available_seasons[0]
        season_data = ratings[ratings["season"] == target_season].copy()
    else:
        season_data = ratings.copy()
        target_season = "전체"

    st.markdown(f"**분석 시즌:** {target_season}")

    # ── 섹션 1: 강등 위험 구단 현황 ──────────────────────────────────
    st.markdown("---")
    st.markdown("### 강등 위험 구단 현황")

    # 강등 위험 팀의 선수 데이터 집계
    team_summary_rows = []
    for team, risk in RELEGATION_RISK_TEAMS.items():
        # 팀명 변형 허용 (Nottm Forest 등)
        team_mask = season_data["team"].str.contains(
            team.split()[0], case=False, na=False
        )
        team_df = season_data[team_mask]
        avg_war = team_df["war"].mean() if len(team_df) > 0 and "war" in team_df.columns else None
        player_count = len(team_df)
        team_summary_rows.append({
            "team": team,
            "risk": risk,
            "avg_war": avg_war,
            "player_count": player_count,
        })

    # 카드 형태로 팀별 현황 표시 (3열)
    cols_per_row = 3
    teams_list = team_summary_rows
    for row_start in range(0, len(teams_list), cols_per_row):
        row_teams = teams_list[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, t in zip(cols, row_teams):
            risk = t["risk"]
            color = _risk_color(risk)
            label = _risk_label(risk)
            avg_war_str = f"{t['avg_war']:.1f}" if t["avg_war"] is not None and not pd.isna(t["avg_war"]) else "-"
            with col:
                st.markdown(
                    f"""<div style='background:#1a1a2e;border-radius:10px;padding:14px;
                    border-left:4px solid {color};margin-bottom:10px;'>
                    <div style='color:{color};font-weight:700;font-size:1.1em;'>{t["team"]}</div>
                    <div style='color:#aaa;font-size:0.82em;margin-top:4px;'>
                      평균 PIS: <span style='color:#fff;font-weight:600;'>{avg_war_str}</span>
                      &nbsp;|&nbsp; 선수 수: <span style='color:#fff;font-weight:600;'>{t["player_count"]}명</span>
                    </div>
                    <div style='margin-top:6px;'>
                      <span style='background:{color};color:#000;font-weight:700;
                      font-size:0.78em;padding:2px 8px;border-radius:10px;'>
                      강등위험 {risk*100:.0f}% — {label}</span>
                    </div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    # ── 섹션 2: 필터 ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 탐색 조건")

    fc1, fc2, fc3, fc4 = st.columns(4)

    with fc1:
        # 강등권 팀 multiselect — 기본 상위 5팀
        all_risk_teams = list(RELEGATION_RISK_TEAMS.keys())
        default_teams = all_risk_teams[:5]
        selected_teams = st.multiselect(
            "강등권 팀 선택",
            options=all_risk_teams,
            default=default_teams,
            key="rel_teams",
        )

    with fc2:
        age_range = st.slider(
            "나이 범위",
            min_value=16, max_value=38,
            value=(18, 30),
            key="rel_age",
        )

    with fc3:
        pos_options = ["전체", "FW", "MF", "DF", "GK"]
        selected_pos = st.selectbox("포지션", pos_options, key="rel_pos")

    with fc4:
        min_war = st.slider(
            "최소 PIS",
            min_value=0, max_value=100,
            value=50,
            key="rel_min_war",
        )

    # ── 필터 적용 ────────────────────────────────────────────────────
    if not selected_teams:
        st.info("강등권 팀을 1개 이상 선택하세요.")
        return

    # 선택된 팀 데이터 추출 (팀명 부분 매칭)
    filtered_parts = []
    for team in selected_teams:
        keyword = team.split()[0]  # 첫 단어로 매칭 (Nottm Forest → Nottm)
        mask = season_data["team"].str.contains(keyword, case=False, na=False)
        filtered_parts.append(season_data[mask])

    if filtered_parts:
        filtered = pd.concat(filtered_parts, ignore_index=True).drop_duplicates()
    else:
        filtered = pd.DataFrame()

    # 나이 필터
    age_col = "age" if "age" in filtered.columns else ("age_used" if "age_used" in filtered.columns else None)
    if age_col and len(filtered) > 0:
        filtered = filtered[
            (filtered[age_col] >= age_range[0]) &
            (filtered[age_col] <= age_range[1])
        ]

    # 포지션 필터
    if selected_pos != "전체" and "pos" in filtered.columns and len(filtered) > 0:
        filtered = filtered[filtered["pos"].str.contains(selected_pos, case=False, na=False)]

    # PIS 필터
    if "war" in filtered.columns and len(filtered) > 0:
        filtered = filtered[filtered["war"] >= min_war]

    # PIS 내림차순 정렬
    if "war" in filtered.columns and len(filtered) > 0:
        filtered = filtered.sort_values("war", ascending=False)

    # ── 섹션 3: 이적 기회 선수 목록 ─────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 이적 기회 선수 목록 ({len(filtered)}명)")

    if len(filtered) == 0:
        st.info("조건에 맞는 선수가 없습니다. 필터 조건을 완화해 보세요.")
    else:
        # 요약 메트릭
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("해당 선수", f"{len(filtered)}명")
        if "war" in filtered.columns:
            m2.metric("평균 PIS", f"{filtered['war'].mean():.1f}")
            m3.metric("최고 PIS", f"{filtered['war'].max():.1f}")
        if "market_value" in filtered.columns:
            valid_mv = filtered["market_value"].dropna()
            if len(valid_mv) > 0:
                m4.metric("평균 시장가치", f"€{valid_mv.mean()/1_000_000:.1f}M")

        st.markdown("")

        # 선수 목록 (Top 30)
        top_n = filtered.head(30)
        for rank, (_, row) in enumerate(top_n.iterrows(), 1):
            player = str(row.get("player", ""))
            team = str(row.get("team", ""))
            pos = str(row.get("pos", ""))
            age_val = row.get(age_col, None) if age_col else None
            war = row.get("war", None)
            mv = row.get("market_value", None)

            # 팀 강등 위험도로 카드 색상 결정
            team_risk = 0.0
            for t_name, t_risk in RELEGATION_RISK_TEAMS.items():
                if t_name.split()[0].lower() in team.lower():
                    team_risk = t_risk
                    break
            card_color = _risk_color(team_risk)

            war_str = f"PIS {war:.1f}" if war is not None and not pd.isna(war) else "-"
            age_str = f"{int(age_val)}세" if age_val is not None and not pd.isna(age_val) else ""
            mv_str = f"€{mv/1_000_000:.1f}M" if mv is not None and not pd.isna(mv) else "-"
            risk_str = f"강등위험 {team_risk*100:.0f}%"

            col_card, col_btn = st.columns([10, 1])
            with col_card:
                st.markdown(
                    f"""<div style='display:flex;align-items:center;gap:12px;
                    background:#1a1a2e;border-radius:10px;padding:10px 14px;
                    margin-bottom:6px;border-left:4px solid {card_color};'>
                    <div style='color:{card_color};font-weight:700;font-size:1.1em;min-width:28px;'>#{rank}</div>
                    <div style='flex:1;min-width:0;'>
                      <div style='font-weight:700;color:#fff;font-size:0.95em;'>{player}</div>
                      <div style='color:#aaa;font-size:0.82em;'>{team} · {pos} · {age_str}</div>
                      <div style='color:#ccc;font-size:0.82em;margin-top:2px;'>
                        {war_str} &nbsp;|&nbsp; 시장가치: <span style='color:{card_color};font-weight:600;'>{mv_str}</span>
                        &nbsp;|&nbsp; <span style='color:{card_color};'>{risk_str}</span>
                      </div>
                    </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with col_btn:
                if st.button("분석", key=f"rel_goto_{player}_{rank}", help=f"{player} 즉시 분석"):
                    st.session_state["scout_report_player"] = player
                    st.session_state["_nav_target"] = "선수 즉시 분석"
                    st.rerun()

    # ── PIS 분포 차트 ───────────────────────────────────────────────
    if len(filtered) > 0 and "war" in filtered.columns and "team" in filtered.columns:
        st.markdown("---")
        st.markdown("### 팀별 PIS 분포")

        fig = px.box(
            filtered,
            x="team",
            y="war",
            color="team",
            color_discrete_sequence=[EPL_MAGENTA, EPL_GREEN, EPL_CYAN, "#ff6b35", "#9b59b6", "#f39c12", "#1abc9c"],
            labels={"war": "PIS(WAR)", "team": "팀"},
        )
        fig.update_layout(
            paper_bgcolor="#0d0d1a",
            plot_bgcolor="#1a1a2e",
            font_color="#fff",
            margin=dict(t=20, b=20, l=20, r=20),
            height=350,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

    # ── 섹션 4: 스카우트 인사이트 박스 ──────────────────────────────
    st.markdown("---")
    st.info(
        "**강등팀 선수 영입 전략 — 김태현 스카우트 인사이트**\n\n"
        "- 계약 1년 미만 선수는 Free Agent로 영입 가능 — 이적료 없이 선점 기회\n"
        "- 강등 확정 후 이적 의지 높아짐 — 협상력 유리, 시장가치보다 낮은 금액 협상 가능\n"
        "- PIS 상위 선수라도 강등팀 잔류 의사 확인 필요 — 팀 충성도 변수 고려\n"
        "- 나이 25세 이하 + PIS 70 이상 선수는 강등팀 최고 이적 타이밍 자원\n"
        "- 1부 잔류 의지 강한 선수는 강등 직후 1-2주 내 이적 협상 착수 권장"
    )

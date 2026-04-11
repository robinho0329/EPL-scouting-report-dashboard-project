"""Scout Overview - 스카우트 분석 대시보드 메인 페이지

S1~S6 모델 요약 + 핵심 인사이트
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.components.data_loader import (
    load_scout_ratings, load_hidden_gems, load_undervalued, load_overvalued,
    load_decline_predictions, load_transfer_predictions, load_s4_reference,
    load_growth_predictions, load_growth_predictions_v4,
)

EPL_PURPLE = "#37003c"
EPL_MAGENTA = "#e90052"
EPL_GREEN = "#00ff87"
EPL_CYAN = "#04f5ff"


def render():
    st.title("스카우트 인텔리전스 허브")
    st.markdown("**김태현 스카우트** | 중위권 구단 (에버턴급)")
    st.info(
        "📋 **사용 가이드**: "
        "①PIS 순위로 고효율 선수 발굴 → ②시장가치로 저평가 확인 → "
        "③유사 선수로 대체재 탐색 → ④성장 곡선으로 피크 타이밍 → "
        "⑤이적 리스크로 적응 성공률 검증 → ⑥하락 감지로 계약 연장 판단"
    )

    # ── 🔧 맞춤 필터 (예산 + 포지션) ────────────────────────────────────────
    st.markdown("#### 🔧 맞춤 영입 조건 설정")
    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        budget = st.slider(
            "💰 이적료 예산 범위 (€M)",
            min_value=0, max_value=150, value=(5, 50), step=5,
            help="선수의 현재 시장가치가 이 범위 안에 있는 선수만 표시합니다."
        )
    with f2:
        pos_filter = st.multiselect(
            "📍 포지션 필터",
            options=["FW", "MID", "DEF", "GK"],
            default=["FW", "MID", "DEF"],
            help="복수 선택 가능"
        )
    with f3:
        age_max = st.number_input("최대 나이", min_value=18, max_value=40, value=28, step=1)

    budget_min_raw = budget[0] * 1_000_000
    budget_max_raw = budget[1] * 1_000_000
    st.markdown("---")

    # ── 핵심 지표 카드 ──
    ratings = load_scout_ratings()
    gems = load_hidden_gems()
    if not gems.empty and 'season' in gems.columns:
        recent_seasons = sorted(gems['season'].unique())[-2:]
        gems = gems[gems['season'].isin(recent_seasons)]
    underval = load_undervalued()
    decline = load_decline_predictions()
    transfers = load_transfer_predictions()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("평가 선수", f"{len(ratings):,}" if len(ratings) else "N/A")
    with col2:
        st.metric("숨은 보석", f"{len(gems):,}" if len(gems) else "N/A")
    with col3:
        st.metric("저평가 발굴", f"{len(underval):,}" if len(underval) else "N/A")
    with col4:
        latest_decline = decline[decline["season_year"] == decline["season_year"].max()] if len(decline) else pd.DataFrame()
        watch_count = len(latest_decline[(latest_decline["age"] >= 28) & (latest_decline["decline_prob_ensemble"] >= 0.6)]) if len(latest_decline) else 0
        st.metric("하락 주의보", f"{watch_count}명")
    with col5:
        st.metric("이적 분석", f"{len(transfers):,}건" if len(transfers) else "N/A")

    st.caption(
        "💡 WAR(0~100 백분위, 평균=50·살라급=99) | "
        "숨은 보석=PIS 상위+시장가치 하위 | "
        "저평가=예측가 1.5배 이상 | "
        "하락 주의보=28세++60%+"
    )

    # ── 🎯 맞춤 영입 추천 ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🎯 맞춤 영입 추천")
    st.caption(f"예산 €{budget[0]}M~€{budget[1]}M | 포지션: {', '.join(pos_filter) if pos_filter else '전체'} | 나이 ≤ {age_max}세")

    if not ratings.empty and "market_value" in ratings.columns:
        latest_season = ratings["season"].max() if "season" in ratings.columns else None
        rec = ratings[ratings["season"] == latest_season].copy() if latest_season else ratings.copy()

        # 필터 적용
        rec = rec[rec["market_value"].fillna(0).between(budget_min_raw, budget_max_raw)]
        if pos_filter:
            rec = rec[rec["pos_group"].isin(pos_filter)]
        if "age" in rec.columns:
            rec = rec[rec["age"].fillna(99) <= age_max]

        if rec.empty:
            st.info("조건에 맞는 선수가 없습니다. 예산 범위 또는 나이 조건을 조정해보세요.")
        else:
            # ── 종합 영입 추천 스코어 계산 ──────────────────────────────
            # PIS 점수 (0~40): WAR 백분위를 그대로 0.4 스케일로
            # S2 저평가 보너스 (0~20): 저평가 목록에 있으면 +20
            # 성장 예측 보너스 (0~25): Improving=25, Stable=12, Declining=0
            # 나이 보너스 (0~15): ≤22=15, 23-24=10, 25-26=5, ≥27=0

            undervalued_players = set()
            _uv = load_undervalued()
            if not _uv.empty and "player" in _uv.columns:
                undervalued_players = set(_uv["player"].dropna().tolist())

            growth_map = {}
            _gv4 = load_growth_predictions_v4()
            if not _gv4.empty and "player" in _gv4.columns:
                pred_col_g = "pred_xgb" if "pred_xgb" in _gv4.columns else "pred_ensemble"
                growth_map = dict(zip(_gv4["player"], _gv4[pred_col_g]))

            def _buy_score(row):
                war = row.get("war", 0) or 0
                age = row.get("age", 30) or 30
                player = row.get("player", "")
                war_pts = min(war, 100) * 0.40
                s2_pts = 20 if player in undervalued_players else 0
                growth = growth_map.get(player, "Stable")
                grow_pts = 25 if growth == "Improving" else (12 if growth == "Stable" else 0)
                age_pts = 15 if age <= 22 else (10 if age <= 24 else (5 if age <= 26 else 0))
                return round(war_pts + s2_pts + grow_pts + age_pts, 1)

            rec["종합점수"] = rec.apply(_buy_score, axis=1)
            # 종합점수 기준 정렬
            rec = rec.nlargest(15, "종합점수")

            # S2 저평가 여부 표시
            rec["S2"] = rec["player"].apply(lambda p: "💎" if p in undervalued_players else "")
            rec["성장"] = rec["player"].apply(lambda p: {"Improving": "🟢", "Stable": "🟡", "Declining": "🔴"}.get(growth_map.get(p, ""), "⚪"))

            show_cols = ["player", "team", "pos_group", "age", "war", "종합점수", "S2", "성장", "market_value"]
            show_cols = [c for c in show_cols if c in rec.columns]
            rec_disp = rec[show_cols].copy()
            if "market_value" in rec_disp.columns:
                rec_disp["market_value"] = rec_disp["market_value"].apply(
                    lambda x: f"€{x/1e6:.1f}M" if pd.notna(x) and x >= 1e6 else (f"€{x/1e3:.0f}K" if pd.notna(x) else "-")
                )
            if "war" in rec_disp.columns:
                rec_disp["war"] = rec_disp["war"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
            rec_disp = rec_disp.rename(columns={
                "player": "선수", "team": "팀", "pos_group": "포지션",
                "age": "나이", "war": "PIS", "market_value": "시장가치",
            })
            st.dataframe(rec_disp, use_container_width=True, hide_index=True)
            st.caption(
                f"💡 {len(rec)}명 조건 충족 | **종합점수** = PIS(40) + S2저평가(20) + 성장예측(25) + 나이보너스(15) "
                f"| 💎=S2저평가 🟢=Improving | '선수 즉시 분석'에서 개별 리포트 확인"
            )
    else:
        st.info("시장가치 데이터가 없습니다.")

    st.markdown("---")

    # ── Top WAR 선수 (최신 시즌) ──
    if len(ratings):
        st.subheader("Top WAR 선수 (맞춤 필터 적용)")
        latest = ratings[ratings["season"] == "2024/25"].copy()
        if len(latest) == 0:
            latest = ratings[ratings["season"] == ratings["season"].max()].copy()

        # 맞춤 필터 연동
        if pos_filter:
            latest = latest[latest["pos_group"].isin(pos_filter)]
        if "age" in latest.columns:
            latest = latest[latest["age"].fillna(99) <= age_max]
        if "market_value" in latest.columns:
            latest = latest[latest["market_value"].fillna(0).between(budget_min_raw, budget_max_raw)]

        if latest.empty:
            st.info("필터 조건에 맞는 선수가 없습니다. 조건을 조정해보세요.")
        else:
            st.caption(f"필터: 예산 €{budget[0]}M~€{budget[1]}M | 포지션 {', '.join(pos_filter) if pos_filter else '전체'} | 나이 ≤{age_max}세")

        active_pos = pos_filter if pos_filter else ["FW", "MID", "DEF", "GK"]
        tab_labels = []
        pos_list_filtered = []
        pos_label_map = {"FW": "공격수 (FW)", "MID": "미드필더 (MID)", "DEF": "수비수 (DEF)", "GK": "골키퍼 (GK)"}
        for p in ["FW", "MID", "DEF", "GK"]:
            if p in active_pos:
                tab_labels.append(pos_label_map[p])
                pos_list_filtered.append(p)

        tabs = st.tabs(tab_labels if tab_labels else ["공격수 (FW)", "미드필더 (MID)", "수비수 (DEF)", "골키퍼 (GK)"])
        pos_list = pos_list_filtered if pos_list_filtered else ["FW", "MID", "DEF", "GK"]
        for tab, pos in zip(tabs, pos_list):
            with tab:
                pos_df = latest[latest["pos_group"] == pos].nlargest(10, "war")
                if len(pos_df):
                    pos_df = pos_df.copy().sort_values("war", ascending=True)
                    colors = [EPL_PURPLE, EPL_MAGENTA, EPL_GREEN, EPL_CYAN,
                              "#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4",
                              "#feca57", "#ff9ff3"]
                    teams = pos_df["team"].unique().tolist()
                    team_color = {t: colors[i % len(colors)] for i, t in enumerate(teams)}
                    bar_colors = [team_color[t] for t in pos_df["team"]]

                    fig = go.Figure(go.Bar(
                        x=pos_df["war"].values,
                        y=pos_df["player"].values,
                        orientation="h",
                        marker_color=bar_colors,
                        text=[f"{v:.1f}" for v in pos_df["war"].values],
                        textposition="outside",
                        showlegend=False,
                    ))
                    # 범례용 더미 트레이스
                    for team in teams:
                        fig.add_trace(go.Bar(
                            x=[None], y=[None], name=team,
                            marker_color=team_color[team], showlegend=True,
                        ))
                    max_war = pos_df["war"].max()
                    fig.update_layout(
                        xaxis=dict(range=[0, max_war * 1.2], title="PIS (선수 임팩트 기여도)"),
                        height=400,
                        yaxis=dict(automargin=True),
                        margin=dict(l=0, r=80, t=10, b=10),
                        plot_bgcolor="#1a1a2e", paper_bgcolor="#0d0d1a", font_color="#ffffff", showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.3, title_text="소속 팀"),
                    )
                    st.plotly_chart(fig, use_container_width=True, theme=None)

        st.caption("💡 PIS(0~100 백분위) 상위 선수는 해당 포지션에서 가장 높은 대체불가 가치를 지닌 선수입니다. PIS 80+(상위 20%)은 리그 최상위급")

    st.markdown("---")

    # ── 저평가 선수 Quick View (상위 3명) ──
    if len(underval):
        st.subheader("저평가 발굴 Top 3 (S2 시장 가치 예측)")
        display_cols = ["player", "team", "pos", "age_used", "market_value", "predicted_value", "value_ratio"]
        available = [c for c in display_cols if c in underval.columns]
        uv = underval[available].head(3).copy()
        if "market_value" in uv.columns:
            uv["market_value"] = uv["market_value"].apply(lambda x: f"\u20ac{x/1e6:.1f}M" if x >= 1e6 else f"\u20ac{x/1e3:.0f}K")
        if "predicted_value" in uv.columns:
            uv["predicted_value"] = uv["predicted_value"].apply(lambda x: f"\u20ac{x/1e6:.1f}M" if x >= 1e6 else f"\u20ac{x/1e3:.0f}K")
        if "value_ratio" in uv.columns:
            uv["value_ratio"] = uv["value_ratio"].apply(lambda x: f"{x:.1f}x")
        uv = uv.rename(columns={
            "player": "선수", "team": "팀", "pos": "포지션",
            "age_used": "나이", "market_value": "시장 가치", "predicted_value": "예측 가치", "value_ratio": "가치 배율",
        })
        st.dataframe(uv, use_container_width=True, hide_index=True)
        st.caption("💡 가치 배율이 높을수록 저평가 폭이 큼. 전체 목록은 선수 분석 → 시장 가치 탭에서 확인")

    # ── 성장 잠재력 Top 3 (P7 모델) ──
    growth = load_growth_predictions()
    if len(growth):
        st.subheader("성장 잠재력 Top 3 (25세 이하, P7 성장 예측)")
        young_growth = growth[growth["current_age"] <= 25].copy() if "current_age" in growth.columns else pd.DataFrame()
        if len(young_growth) and "pred_next1" in young_growth.columns:
            top3 = young_growth.nlargest(3, "pred_next1")
            g_cols = st.columns(3)
            for i, (_, row) in enumerate(top3.iterrows()):
                with g_cols[i]:
                    player_name = row.get("player", "N/A")
                    cur_age = row.get("current_age", None)
                    pos = row.get("pos_group", "N/A")
                    peak_age = row.get("peak_age", None)
                    pred1 = row.get("pred_next1", None)
                    seasons_left = int(peak_age) - int(cur_age) if pd.notna(peak_age) and pd.notna(cur_age) else None
                    peak_label = f"전성기까지 {seasons_left}시즌" if seasons_left is not None and seasons_left > 0 else ("전성기" if seasons_left == 0 else "전성기 도달")
                    st.markdown(
                        f"""
<div style="border:1px solid #00ff87; border-radius:8px; padding:12px; background:#f8fff8;">
<b style="font-size:1.05em;">{player_name}</b><br>
나이: {int(cur_age)}세 | 포지션: {pos}<br>
다음 시즌 예측 지수: <b>{pred1:.2f}</b><br>
<span style="color:#37003c; font-size:0.9em;">{peak_label}</span>
</div>
""",
                        unsafe_allow_html=True,
                    )
        st.caption("💡 25세 이하 선수 중 P7 모델이 예측한 다음 시즌 성과 지수(AC Z-score) 상위 3명. 전성기 잔여 시즌이 많을수록 성장 프리미엄 기대 가능.")
        st.markdown("---")

    # ── 하락 주의보 Quick View (상위 5명) ──
    if len(decline):
        st.subheader("계약 연장 재검토 대상 Top 5 (S6 하락 감지)")
        latest_yr = decline["season_year"].max()
        watch = decline[
            (decline["season_year"] == latest_yr) &
            (decline["age"] >= 28) &
            (decline["decline_prob_ensemble"] >= 0.6)
        ].nlargest(5, "decline_prob_ensemble")
        if len(watch):
            display = watch[["player_key", "team", "pos_group", "age", "decline_prob_ensemble", "perf_slope"]].copy()
            display.columns = ["선수", "팀", "포지션", "나이", "하락 확률", "성과 추세"]
            display["하락 확률"] = display["하락 확률"].apply(lambda x: f"{x:.1%}")
            display["성과 추세"] = display["성과 추세"].apply(lambda x: f"{x:+.3f}" if pd.notna(x) else "N/A")
            st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption("💡 하락 확률 60%+ + 28세 이상. 성과 추세가 음수이면 이미 하락 진행 중. 전체 목록은 선수 분석 → 하락 주의보 탭에서 확인")

    # ── 피크 연령 참조 ──
    ref = load_s4_reference()
    if ref and "peak_ages" in ref:
        st.subheader("포지션별 피크 연령 (S4 성장 레퍼런스)")
        peaks = ref["peak_ages"]
        pcols = st.columns(4)
        for i, (pos, info) in enumerate(peaks.items()):
            with pcols[i]:
                peak_age = info.get("epl_data") or info.get("peak_age") or info.get("smoothed_peak_age", "?")
                pos_kor = info.get("pos_kor", pos)
                st.metric(f"{pos} ({pos_kor})", f"{peak_age}세")
        st.caption("💡 EPL 데이터 기반 포지션별 피크 연령. 피크 전 선수는 성장 잠재력, 피크 후 선수는 하락 리스크 고려")

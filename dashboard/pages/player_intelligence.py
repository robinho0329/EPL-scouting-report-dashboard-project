"""선수 종합 인텔리전스 — S1(레이팅) + S2(시장가치) + S6(하락세) 1페이지 보고서

김태현 스카우트 영입 회의용 핵심 자료:
선수 1명 선택 → 레이팅 / 시장가치 / 하락세 리스크를 한 화면에 출력
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from dashboard.components.data_loader import (
    load_scout_ratings,
    load_undervalued,
    load_overvalued,
    load_decline_predictions,
    load_growth_predictions,
)

EPL_PURPLE  = "#37003c"
EPL_MAGENTA = "#e90052"
EPL_GREEN   = "#00ff87"
EPL_CYAN    = "#04f5ff"
EPL_GOLD    = "#f5a623"


def _risk_badge(prob: float) -> str:
    if prob >= 0.6:
        return "🔴 HIGH"
    if prob >= 0.35:
        return "🟡 MEDIUM"
    return "🟢 LOW"


def _pis_bar(pis: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pis,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "PIS 백분위", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": EPL_MAGENTA},
            "steps": [
                {"range": [0,  40], "color": "#2a0030"},
                {"range": [40, 70], "color": "#5c0060"},
                {"range": [70, 90], "color": "#8b0068"},
                {"range": [90,100], "color": EPL_MAGENTA},
            ],
            "threshold": {"line": {"color": EPL_GREEN, "width": 3}, "value": 70},
        },
    ))
    fig.update_layout(height=220, margin=dict(t=30, b=10, l=20, r=20),
                      paper_bgcolor="rgba(0,0,0,0)", font_color="white")
    return fig


def render():
    st.title("🧠 선수 종합 인텔리전스")
    st.markdown(
        "**S1 레이팅 · S2 시장가치 · S6 하락세** — 영입 회의 1페이지 보고서  \n"
        "*김태현 스카우트 전용 | 30-50M £ 예산 기준*"
    )
    st.markdown("---")

    # ── 데이터 로드 ────────────────────────────────────────────────────────────
    ratings   = load_scout_ratings()
    underval  = load_undervalued()
    overval   = load_overvalued()
    decline   = load_decline_predictions()
    growth    = load_growth_predictions()

    if ratings.empty:
        st.warning("S1 레이팅 데이터가 없습니다. GitHub Actions 학습 후 재시도하세요.")
        return

    # 최신 시즌만
    if "season" in ratings.columns:
        latest_season = ratings["season"].max()
        ratings_latest = ratings[ratings["season"] == latest_season].copy()
    else:
        ratings_latest = ratings.copy()

    # ── 사이드바 필터 ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔍 선수 검색")
        pos_options = sorted(ratings_latest["pos_group"].dropna().unique().tolist()) if "pos_group" in ratings_latest.columns else []
        sel_pos = st.multiselect("포지션", pos_options, default=pos_options)

        age_min, age_max = st.slider("나이 범위", 17, 40, (18, 32))

        if "market_value" in ratings_latest.columns:
            budget = st.slider("시장가치 (£M)", 0, 150, (2, 60), step=2)
            budget_min = budget[0] * 1_000_000
            budget_max = budget[1] * 1_000_000
        else:
            budget_min, budget_max = 0, 999_999_999

        min_pis = st.slider("최소 PIS", 0, 100, 50)

    # ── 필터 적용 ──────────────────────────────────────────────────────────────
    df = ratings_latest.copy()
    if sel_pos and "pos_group" in df.columns:
        df = df[df["pos_group"].isin(sel_pos)]
    if "age" in df.columns:
        df = df[df["age"].fillna(99).between(age_min, age_max)]
    if "market_value" in df.columns:
        df = df[df["market_value"].fillna(0).between(budget_min, budget_max)]
    if "pis" in df.columns:
        df = df[df["pis"].fillna(0) >= min_pis]

    if df.empty:
        st.info("필터 조건에 맞는 선수가 없습니다. 조건을 완화해보세요.")
        return

    player_list = sorted(df["player"].dropna().unique().tolist())
    selected_player = st.selectbox(
        "선수 선택",
        player_list,
        index=0,
        help="이름으로 검색 가능합니다"
    )

    # ── 선수 데이터 수집 ───────────────────────────────────────────────────────
    p_row = df[df["player"] == selected_player].sort_values("season", ascending=False).iloc[0] if "season" in df.columns else df[df["player"] == selected_player].iloc[0]

    # S2: 시장가치 (undervalued / overvalued)
    s2_row_under = underval[underval["player"] == selected_player].head(1) if not underval.empty and "player" in underval.columns else pd.DataFrame()
    s2_row_over  = overval[overval["player"] == selected_player].head(1)  if not overval.empty  and "player" in overval.columns else pd.DataFrame()

    # S6: 하락세 예측
    s6_row = pd.DataFrame()
    if not decline.empty and "player" in decline.columns:
        s6_match = decline[decline["player"] == selected_player]
        if not s6_match.empty:
            if "season_year" in s6_match.columns:
                s6_row = s6_match.sort_values("season_year", ascending=False).head(1)
            else:
                s6_row = s6_match.head(1)

    # P7: 성장 곡선
    growth_row = pd.DataFrame()
    if not growth.empty and "player" in growth.columns:
        growth_row = growth[growth["player"] == selected_player].head(1)

    # ── 헤더: 선수 기본 정보 ──────────────────────────────────────────────────
    st.markdown(f"## {selected_player}")
    meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
    with meta_col1:
        pos = p_row.get("pos_group", p_row.get("position", "N/A"))
        st.metric("포지션", str(pos))
    with meta_col2:
        age_val = p_row.get("age", "N/A")
        st.metric("나이", f"{age_val:.0f}세" if isinstance(age_val, float) else str(age_val))
    with meta_col3:
        team_val = p_row.get("team", "N/A")
        st.metric("현 소속팀", str(team_val))
    with meta_col4:
        mv_val = p_row.get("market_value", None)
        if mv_val and mv_val > 0:
            st.metric("현 시장가치", f"£{mv_val/1_000_000:.1f}M")
        else:
            st.metric("현 시장가치", "N/A")

    st.markdown("---")

    # ── 3열 레이아웃: S1 / S2 / S6 ────────────────────────────────────────────
    col_s1, col_s2, col_s6 = st.columns(3)

    # ── S1: 선수 종합 레이팅 ──────────────────────────────────────────────────
    with col_s1:
        st.subheader("⭐ S1 종합 레이팅")
        pis_val = p_row.get("pis", None)
        war_val = p_row.get("war", p_row.get("ac_z", None))
        epl_exp = p_row.get("epl_experience", None)

        if pis_val is not None and not pd.isna(pis_val):
            st.plotly_chart(_pis_bar(float(pis_val)), use_container_width=True)
        else:
            st.info("PIS 데이터 없음")

        col_a, col_b = st.columns(2)
        with col_a:
            if war_val is not None and not pd.isna(war_val):
                st.metric("WAR (Z-score)", f"{float(war_val):+.2f}")
        with col_b:
            if epl_exp is not None and not pd.isna(epl_exp):
                st.metric("EPL 경력", f"{int(epl_exp)}시즌")

        # 시즌 기록 미니 테이블
        if "season" in ratings.columns:
            history = ratings[ratings["player"] == selected_player].sort_values("season", ascending=False)[
                ["season", "pis", "war"] if all(c in ratings.columns for c in ["season","pis","war"]) else ["season"]
            ].head(5)
            if not history.empty:
                st.markdown("**시즌별 PIS 추이**")
                st.dataframe(history.reset_index(drop=True), use_container_width=True, height=160)

    # ── S2: 시장가치 ──────────────────────────────────────────────────────────
    with col_s2:
        st.subheader("💰 S2 시장가치")

        # 저평가/고평가 여부
        is_underval = not s2_row_under.empty
        is_overval  = not s2_row_over.empty

        if is_underval:
            row_u = s2_row_under.iloc[0]
            actual_mv = row_u.get("market_value", None)
            pred_mv   = row_u.get("predicted_value", row_u.get("pred_value", None))
            ratio     = row_u.get("value_ratio", None)

            st.success("✅ **저평가 선수**")
            c1, c2 = st.columns(2)
            with c1:
                if actual_mv:
                    st.metric("현 시장가치", f"£{actual_mv/1e6:.1f}M")
            with c2:
                if pred_mv:
                    st.metric("모델 적정가", f"£{pred_mv/1e6:.1f}M", delta=f"+{(pred_mv-actual_mv)/1e6:.1f}M" if (actual_mv and pred_mv) else None)
            if ratio:
                st.progress(min(float(ratio)/3.0, 1.0), text=f"적정가 배율 {float(ratio):.1f}x")

        elif is_overval:
            row_o = s2_row_over.iloc[0]
            actual_mv = row_o.get("market_value", None)
            pred_mv   = row_o.get("predicted_value", row_o.get("pred_value", None))
            st.warning("⚠️ **고평가 주의**")
            c1, c2 = st.columns(2)
            with c1:
                if actual_mv:
                    st.metric("현 시장가치", f"£{actual_mv/1e6:.1f}M")
            with c2:
                if pred_mv:
                    st.metric("모델 적정가", f"£{pred_mv/1e6:.1f}M")
        else:
            st.info("시장가치 분석 데이터 없음")
            if mv_val:
                st.metric("현재 시장가치", f"£{mv_val/1e6:.1f}M")

        # S2 포지션 평균 비교
        if "pos_group" in df.columns and "market_value" in df.columns:
            pos_group = p_row.get("pos_group", "")
            pos_avg_mv = df[df["pos_group"] == pos_group]["market_value"].median()
            if pos_avg_mv and mv_val:
                ratio_to_avg = mv_val / pos_avg_mv
                st.markdown("**포지션 내 시장가치 위치**")
                st.progress(min(ratio_to_avg / 3.0, 1.0),
                            text=f"포지션 중앙값 대비 {ratio_to_avg:.1f}x")

    # ── S6: 하락세 감지 ───────────────────────────────────────────────────────
    with col_s6:
        st.subheader("📉 S6 하락세 리스크")

        if not s6_row.empty:
            row_s6 = s6_row.iloc[0]
            # decline_prob 컬럼명 탐색
            prob_col = next((c for c in ["decline_prob_ensemble", "decline_prob", "prob_decline"]
                             if c in row_s6.index), None)
            prob_val = float(row_s6[prob_col]) if prob_col else None

            if prob_val is not None:
                badge = _risk_badge(prob_val)
                st.markdown(f"### {badge}")
                st.metric("하락 확률", f"{prob_val*100:.1f}%")

                # 게이지
                fig_risk = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob_val * 100,
                    number={"suffix": "%"},
                    title={"text": "하락세 위험도"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": EPL_MAGENTA if prob_val >= 0.6 else EPL_GOLD if prob_val >= 0.35 else EPL_GREEN},
                        "steps": [
                            {"range": [0,  35], "color": "#0d3b1e"},
                            {"range": [35, 60], "color": "#3b2e00"},
                            {"range": [60,100], "color": "#3b0000"},
                        ],
                        "threshold": {"line": {"color": "white", "width": 2}, "value": 60},
                    },
                ))
                fig_risk.update_layout(height=220, margin=dict(t=30, b=10, l=20, r=20),
                                       paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig_risk, use_container_width=True)

            # 보조 지표
            for col_name, label in [("age", "나이"), ("season_year", "시즌"), ("min", "출전 분")]:
                if col_name in row_s6.index:
                    val = row_s6[col_name]
                    if pd.notna(val):
                        st.caption(f"{label}: {val}")
        else:
            st.info("하락세 예측 데이터 없음 (S6 미학습 또는 해당 선수 대상 외)")
            age_check = p_row.get("age", None)
            if age_check and age_check >= 28:
                st.warning(f"⚠️ {age_check:.0f}세 이상 — 다음 학습 사이클에서 하락세 모니터링 권장")

    # ── P7 성장 곡선 ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 P7 성장 곡선 예측 (향후 3시즌)")

    if not growth_row.empty:
        row_g = growth_row.iloc[0]
        cur_ac = row_g.get("current_ac_z", 0.0)
        p1 = row_g.get("pred_next1", None)
        p2 = row_g.get("pred_next2", None)
        p3 = row_g.get("pred_next3", None)
        peak_age = row_g.get("peak_age", None)
        decline_age = row_g.get("decline_start_age", None)

        if all(v is not None for v in [p1, p2, p3]):
            seasons_label = ["현재", "+1시즌", "+2시즌", "+3시즌"]
            values = [float(cur_ac), float(p1), float(p2), float(p3)]
            colors = [EPL_CYAN] + [
                EPL_GREEN if v >= cur_ac else EPL_MAGENTA for v in values[1:]
            ]

            fig_growth = go.Figure()
            fig_growth.add_trace(go.Scatter(
                x=seasons_label, y=values,
                mode="lines+markers+text",
                text=[f"{v:+.2f}" for v in values],
                textposition="top center",
                line=dict(color=EPL_CYAN, width=2),
                marker=dict(color=colors, size=12),
            ))
            fig_growth.add_hline(y=0, line_dash="dash", line_color="gray",
                                 annotation_text="리그 평균")
            fig_growth.update_layout(
                height=280,
                margin=dict(t=20, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(20,0,30,0.5)",
                font_color="white",
                yaxis_title="공격 기여도 Z-score",
                xaxis_title="시즌",
            )
            st.plotly_chart(fig_growth, use_container_width=True)

            gc1, gc2, gc3 = st.columns(3)
            with gc1:
                if peak_age:
                    st.metric("포지션 피크 나이", f"{peak_age}세")
            with gc2:
                if decline_age:
                    st.metric("하락세 시작 나이", f"{decline_age}세")
            with gc3:
                trend = values[-1] - values[0]
                st.metric("3시즌 트렌드", f"{trend:+.2f}", delta_color="normal")
    else:
        age_val_g = p_row.get("age", None)
        if age_val_g and age_val_g <= 26:
            st.info("성장 곡선 데이터 없음 — P7 학습 후 재확인 권장 (25세 이하 핵심 선수)")
        else:
            st.info("성장 곡선 데이터 없음")

    # ── 영입 종합 권고 ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 영입 종합 권고")

    signals = []
    risk_score = 0

    pis_v = p_row.get("pis", None)
    if pis_v and pis_v >= 70:
        signals.append(f"✅ PIS {pis_v:.0f} — 포지션 상위 30%")
    elif pis_v and pis_v >= 50:
        signals.append(f"➡️ PIS {pis_v:.0f} — 포지션 평균 수준")
    elif pis_v:
        signals.append(f"⚠️ PIS {pis_v:.0f} — 포지션 하위권")
        risk_score += 1

    if is_underval:
        signals.append("✅ 저평가 — 시장가치 대비 실력 우수")
    elif is_overval:
        signals.append("⚠️ 고평가 — 이적료 협상 신중 필요")
        risk_score += 1

    if not s6_row.empty and prob_col:
        prob_s6 = float(s6_row.iloc[0][prob_col])
        if prob_s6 >= 0.6:
            signals.append(f"🔴 하락세 위험 {prob_s6*100:.0f}% — 단기 계약 권장")
            risk_score += 2
        elif prob_s6 >= 0.35:
            signals.append(f"🟡 하락세 중간 위험 {prob_s6*100:.0f}%")
            risk_score += 1
        else:
            signals.append(f"✅ 하락세 위험 낮음 ({prob_s6*100:.0f}%)")

    age_v = p_row.get("age", None)
    if age_v:
        if age_v <= 23:
            signals.append(f"✅ {age_v:.0f}세 — 성장 여력 충분")
        elif age_v >= 32:
            signals.append(f"⚠️ {age_v:.0f}세 — 잔여 계약 기간 고려 필요")
            risk_score += 1

    for sig in signals:
        st.markdown(f"- {sig}")

    # 최종 권고
    if risk_score == 0:
        st.success("**✅ 영입 강력 추천** — 모든 지표 긍정적")
    elif risk_score == 1:
        st.info("**➡️ 영입 검토 가능** — 일부 주의 사항 확인 필요")
    elif risk_score == 2:
        st.warning("**⚠️ 신중 검토** — 리스크 요인 존재, 조건부 협상 권장")
    else:
        st.error("**🔴 영입 비권장** — 복수 리스크 요인 확인됨")

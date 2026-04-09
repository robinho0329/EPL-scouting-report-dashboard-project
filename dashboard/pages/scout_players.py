"""Scout Player Analysis - S1 WAR + S2 시장가치 + S6 하락세 통합 선수 분석

선수 개별 프로필을 WAR/시장가치/하락 리스크 관점에서 종합 분석.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.components.data_loader import (
    load_scout_ratings, load_hidden_gems, load_undervalued, load_overvalued,
    load_decline_predictions, load_growth_predictions, load_s2_transfer_targets,
)


@st.cache_data
def _load_p6():
    """P6 절대 시장가치 예측 결과 로드."""
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent.parent / "data" / "scout" / "p6_market_value_predictions.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)
from dashboard.utils.image_utils import render_player_card, get_team_logo_b64

EPL_PURPLE = "#37003c"
EPL_MAGENTA = "#e90052"
EPL_GREEN = "#00ff87"
EPL_CYAN = "#04f5ff"

# 컬럼명 한국어 매핑
COL_RENAME = {
    "player": "선수",
    "team": "팀",
    "pos_group": "포지션",
    "def_subpos": "세부포지션",
    "age": "나이",
    "war": "WAR",
    "total_min": "출전시간",
    "goals_p90": "골/90분",
    "assists_p90": "어시/90분",
    "tackles_p90": "태클/90분",
    "int_p90": "인터셉트/90분",
    "consistency": "일관성",
    "tier": "등급",
    "market_value": "시장가치",
}


def render():
    st.title("선수 분석")
    st.markdown(
        "**김태현 스카우트 관점** — 중위권 구단 예산(£30-50M/시즌) 기준, "
        "WAR·시장가치·하락 리스크를 통합해 가성비 영입 및 계약 연장 판단을 지원합니다."
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "WAR 순위", "숨은 보석", "시장 가치", "하락 주의보"
    ])

    # ══════════════════════════════════════════
    # TAB 1: WAR 순위 (S1)
    # ══════════════════════════════════════════
    with tab1:
        st.subheader("S1. 선수 WAR 순위")
        with st.expander("📖 WAR 지표 해석 가이드 (스카우터 필독)", expanded=False):
            st.markdown("""
**WAR (Wins Above Replacement)** 는 해당 선수를 EPL 리그 평균 대체 선수로 바꿨을 때 팀이 잃게 되는 승리 수입니다.

> ⚠️ **WAR 스케일 안내**: WAR은 **0~100 백분위(percentile)** 기준으로 산출됩니다.
> 리그 최고값 ≈ 99 (살라급), 리그 평균 = 50. 절댓값이 아닌 상대적 순위 지표입니다.

| WAR 구간 (백분위) | 해석 | 스카우팅 판단 |
|-----------------|------|------------|
| **80+ (상위 20%)** | 리그 최상위급 | 빅클럽 핵심 자원, 영입 불가 영역 |
| **65 ~ 80 (상위 20~35%)** | 리그 상위급 | 주전 자원, 높은 이적료 예상 |
| **50 ~ 65 (상위 35~50%)** | 평균 이상 | 중위권 구단 주전감 — **우리 예산 최적 타겟** |
| **35 ~ 50 (하위 50~65%)** | 평균 수준 | 로테이션 자원 |
| **35 미만 (하위 35%)** | 대체 선수 이하 | 방출/계약 연장 재검토 |

- **p90 기준**: 90분당 수치로 출전시간 편향을 제거. 주전/벤치 공정 비교 가능
            """)
        ratings = load_scout_ratings()
        if len(ratings) == 0:
            st.warning("scout_ratings_v3.parquet 데이터 없음")
            return

        # 필터
        col1, col2, col3 = st.columns(3)
        seasons = sorted(ratings["season"].unique(), reverse=True)
        with col1:
            sel_season = st.selectbox("시즌 선택", seasons, key="war_season")
        with col2:
            sel_pos = st.selectbox("포지션 선택", ["전체", "FW", "MID", "DEF", "GK"], key="war_pos")
        with col3:
            sel_tier = st.selectbox("팀 등급 선택", ["전체", "top6", "mid", "bottom6"], key="war_tier")

        df = ratings[ratings["season"] == sel_season].copy()
        if sel_pos != "전체":
            df = df[df["pos_group"] == sel_pos]
        if sel_tier != "전체":
            df = df[df["tier"] == sel_tier]

        df = df.sort_values("war", ascending=False).reset_index(drop=True)
        df.index = df.index + 1

        # 상위 20명 바 차트
        top20 = df.head(20).copy().sort_values("war", ascending=True)
        if len(top20):
            pos_color = {"FW": EPL_MAGENTA, "MID": EPL_CYAN, "DEF": EPL_GREEN, "GK": EPL_PURPLE}
            bar_colors = [pos_color.get(p, "#888") for p in top20["pos_group"]]
            fig = go.Figure(go.Bar(
                x=top20["war"].values,
                y=top20["player"].values,
                orientation="h",
                marker_color=bar_colors,
                text=[f"{v:.1f}" for v in top20["war"].values],
                textposition="outside",
                showlegend=False,
            ))
            for pos in top20["pos_group"].unique():
                fig.add_trace(go.Bar(
                    x=[None], y=[None], name=pos,
                    marker_color=pos_color.get(pos, "#888"), showlegend=True,
                ))
            max_war = top20["war"].max()
            fig.update_layout(
                height=500,
                xaxis=dict(range=[0, max_war * 1.2], title="WAR 점수 (높을수록 대체불가 가치↑)"),
                margin=dict(l=10, r=60, t=10, b=10), plot_bgcolor="#1a1a2e", paper_bgcolor="#0d0d1a", font_color="#ffffff",
            )
            st.plotly_chart(fig, use_container_width=True, theme=None)
        st.caption(
            "💡 **스카우터 포인트**: WAR 50~65 구간(상위 35~50%, 백분위 기준) 선수가 중위권 구단 예산으로 확보 가능한 최적 타겟입니다. "
            "bottom6 팀의 WAR 상위 선수는 '숨은 보석' 탭에서 더 자세히 확인하세요."
        )

        # ── WAR Top 10 선수 카드 ──────────────────────────────
        st.markdown("#### WAR Top 10 선수")
        top10 = df.head(10).copy()
        for rank, (_, row) in enumerate(top10.iterrows(), start=1):
            p_name  = row.get("player", "")
            p_team  = row.get("team", "")
            p_pos   = row.get("pos_group", "")
            p_age   = row.get("age", None)
            p_war   = row.get("war", None)
            p_min   = row.get("total_min", None)
            g90     = row.get("goals_p90", None)
            a90     = row.get("assists_p90", None)

            # 팀 로고 b64
            logo_b64 = get_team_logo_b64(p_team, size=(28, 28)) if p_team else None

            # 추가 정보 문자열 조립
            extra_parts = []
            if p_age is not None and pd.notna(p_age):
                extra_parts.append(f"나이 {int(p_age)}세")
            if p_min is not None and pd.notna(p_min):
                extra_parts.append(f"출전 {int(p_min)}분")
            if g90 is not None and pd.notna(g90):
                extra_parts.append(f"골 {g90:.2f}/90분")
            if a90 is not None and pd.notna(a90):
                extra_parts.append(f"어시 {a90:.2f}/90분")
            extra_str = " | ".join(extra_parts)

            # 순위 + 팀 로고 헤더 라인
            header_parts = [f"**{rank}위**"]
            if logo_b64:
                header_parts.append(
                    f'<img src="data:image/png;base64,{logo_b64}" '
                    f'style="vertical-align:middle; height:22px; margin-left:4px;">'
                )
            if p_war is not None and pd.notna(p_war):
                header_parts.append(f"WAR **{p_war:.1f}**")
            st.markdown(" ".join(header_parts), unsafe_allow_html=True)

            render_player_card(
                player_name=p_name,
                team=p_team,
                archetype=p_pos,
                extra_info=extra_str,
            )
            st.markdown('<hr style="margin:4px 0; border-color:#eee;">', unsafe_allow_html=True)

        # ── 상위 30명 전체 테이블 (Top10 이후 나머지 포함) ──
        st.markdown("#### 전체 순위 테이블 (Top 30)")
        show_cols = ["player", "team", "pos_group", "age", "war", "total_min",
                     "goals_p90", "assists_p90", "tackles_p90", "int_p90"]
        available = [c for c in show_cols if c in df.columns]
        disp = df[available].head(30).copy()
        disp = disp.rename(columns={c: COL_RENAME.get(c, c) for c in disp.columns})
        st.dataframe(
            disp.style.format({"WAR": "{:.1f}", "골/90분": "{:.2f}",
                "어시/90분": "{:.2f}", "태클/90분": "{:.2f}", "인터셉트/90분": "{:.2f}"}, na_rep="-"),
            use_container_width=True, height=400,
        )

        # ── 즉시 분석 연동 ─────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🔍 선수 즉시 분석으로 이동")
        ranked_players = df["player"].dropna().unique().tolist() if "player" in df.columns else []
        sp_c1, sp_c2 = st.columns([3, 1])
        with sp_c1:
            sp_picked = st.selectbox(
                "선수 선택", [""] + sorted(ranked_players),
                key="war_tab_player_link", label_visibility="collapsed",
            )
        with sp_c2:
            if st.button("🔍 즉시 분석", key="war_tab_goto_report", use_container_width=True):
                if sp_picked:
                    st.session_state["scout_report_player"] = sp_picked
                    st.session_state["_nav_target"] = "🔍 선수 즉시 분석"
                    st.rerun()
                else:
                    st.warning("선수를 선택하세요.")

    # ══════════════════════════════════════════
    # TAB 2: 숨은 보석 (S1)
    # ══════════════════════════════════════════
    with tab2:
        st.subheader("S1. 숨은 보석")
        with st.expander("📖 숨은 보석 발굴 기준 (스카우터 필독)", expanded=False):
            st.markdown("""
**숨은 보석** = WAR 상위 25% + 시장가치 하위 50% 교집합 선수

스카우팅 관점에서 가장 중요한 탭입니다. 실력은 상위권이지만 이름값·팀 인지도 부족으로 시장에서 저평가된 선수를 발굴합니다.

**차트 읽는 법:**
- **X축(시장가치)**: 왼쪽으로 갈수록 저렴
- **Y축(WAR)**: 위로 갈수록 팀 기여도 높음
- **점 크기**: 출전시간 — 클수록 많이 뛰고 있음 (안정적 주전)
- **최우선 타겟**: 좌측 상단 (저렴 + 고WAR + 큰 점)

**영입 시 추가 확인 사항:**
- 나이가 24세 이하면 성장 여지까지 있어 더욱 가치 있음
- bottom6 팀 선수는 강팀 이적 후 적응 리스크를 S5 탭에서 확인 필요
            """)
        gems = load_hidden_gems()
        if not gems.empty and 'season' in gems.columns:
            recent_seasons = sorted(gems['season'].unique())[-2:]
            gems = gems[gems['season'].isin(recent_seasons)]
        if len(gems) == 0:
            st.warning("hidden_gems_v3.parquet 데이터 없음")
        else:
            gcol1, gcol2 = st.columns(2)
            with gcol1:
                gem_season = st.selectbox("시즌 선택", sorted(gems["season"].unique(), reverse=True), key="gem_season")
            with gcol2:
                gem_pos = st.selectbox("포지션 선택", ["전체", "FW", "MID", "DEF", "GK"], key="gem_pos")

            gdf = gems[gems["season"] == gem_season].copy()
            if gem_pos != "전체":
                gdf = gdf[gdf["pos_group"] == gem_pos]

            gdf = gdf.sort_values("war", ascending=False)

            if len(gdf):
                pos_color = {"FW": EPL_MAGENTA, "MID": EPL_CYAN, "DEF": EPL_GREEN, "GK": EPL_PURPLE}
                fig = go.Figure()
                for pos in gdf["pos_group"].unique():
                    pdf = gdf[gdf["pos_group"] == pos]
                    sizes = pdf["total_min"].fillna(500).tolist()
                    min_s, max_s = min(sizes), max(sizes)
                    norm = [5 + 15*(s-min_s)/(max_s-min_s+1) for s in sizes]
                    fig.add_trace(go.Scatter(
                        x=pdf["market_value"].tolist(),
                        y=pdf["war"].tolist(),
                        mode="markers",
                        name=pos,
                        marker=dict(size=norm, color=pos_color.get(pos, "#888"), opacity=0.7),
                        text=pdf["player"].tolist(),
                        customdata=list(zip(pdf["team"].tolist(), pdf["age"].tolist(), pdf["total_min"].fillna(0).tolist())),
                        hovertemplate="<b>%{text}</b><br>팀: %{customdata[0]}<br>나이: %{customdata[1]}<br>출전시간: %{customdata[2]:.0f}분<br>시장가치: €%{x:,.0f}<br>WAR: %{y:.1f}<extra>%{fullData.name}</extra>",
                    ))
                fig.add_annotation(
                    x=0.02, y=0.98, xref="paper", yref="paper",
                    text="← 저렴 + 고WAR = 최우선 타겟", showarrow=False,
                    font=dict(size=11, color=EPL_MAGENTA), bgcolor="#1a1a2e", bordercolor=EPL_MAGENTA,
                )
                fig.update_layout(
                    xaxis_title="시장가치 (EUR)", yaxis_title="WAR 점수 (팀 기여도)",
                    height=450, plot_bgcolor="#1a1a2e", paper_bgcolor="#0d0d1a", font_color="#ffffff", margin=dict(l=10, r=10, t=30, b=10),
                    legend_title="포지션",
                )
                st.plotly_chart(fig, use_container_width=True, theme=None)

                show_cols = ["player", "team", "pos_group", "age", "war", "market_value", "total_min",
                             "goals_p90", "assists_p90"]
                available = [c for c in show_cols if c in gdf.columns]
                disp = gdf[available].copy()
                if "market_value" in disp.columns:
                    disp["market_value"] = disp["market_value"].apply(
                        lambda x: f"€{x/1e6:.1f}M" if pd.notna(x) and x >= 1e6 else f"€{x/1e3:.0f}K" if pd.notna(x) else "N/A"
                    )
                disp = disp.rename(columns={c: COL_RENAME.get(c, c) for c in disp.columns})
                st.dataframe(disp, use_container_width=True, hide_index=True)
                st.caption(
                    "💡 **스카우터 포인트**: 나이 24세 이하 + WAR 3.0+ + 시장가치 €5M 미만 선수가 최고의 가성비 타겟. "
                    "팀 tier가 bottom6이면 이적 후 성장 가능성이 더 큽니다."
                )

    # ══════════════════════════════════════════
    # TAB 3: 시장 가치 (S2)
    # ══════════════════════════════════════════
    with tab3:
        st.subheader("S2. 시장 가치 분석")
        with st.expander("📖 시장가치 모델 해석 가이드 (스카우터 필독)", expanded=False):
            st.markdown("""
**S2 시장가치 모델**: WAR·나이·포지션·리그 수준 등을 학습한 XGBoost 기반 가치 예측 모델

**저평가 선수 (Undervalued)**: 예측가 ÷ 실제가 ≥ 1.5배
- 시장이 아직 알아채지 못한 선수
- **영입 타이밍**: 지금이 최적 — 다음 시즌 주목받으면 가격 오름

**고평가 선수 (Overvalued)**: 예측가 ÷ 실제가 ≤ 0.5배
- 이름값·최근 폼으로 거품이 낀 선수
- **매도 타이밍**: 보유 중이라면 지금이 최고가 — 추가 영입 시 신중

**가치 비율(value_ratio)** 읽는 법:
- `2.0x` = 예측가가 실제가의 2배 → 강력한 저평가 신호
- `0.3x` = 예측가가 실제가의 30% → 강력한 고평가 신호
            """)

        sub1, sub2, sub3, sub4 = st.tabs(["저평가 선수 (S2)", "고평가 선수 (S2)", "🔬 P6 절대가치 예측", "🎯 WAR 기반 저평가"])

        with sub1:
            underval = load_undervalued()
            if len(underval) == 0:
                st.warning("s2_v4_undervalued.parquet 데이터 없음")
            else:
                st.markdown("**예측 시장가치 > 실제 시장가치 1.5배** 이상인 선수 — 지금 사면 이득")
                uv = underval.sort_values("value_ratio", ascending=False).copy()

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=uv["player"].head(15).tolist(),
                    x=(uv["market_value"].head(15)/1e6).tolist(),
                    name="실제 시장가치 (지금 사는 가격)", orientation="h", marker_color=EPL_MAGENTA,
                ))
                fig.add_trace(go.Bar(
                    y=uv["player"].head(15).tolist(),
                    x=(uv["predicted_value"].head(15)/1e6).tolist(),
                    name="예측 시장가치 (모델 추정 적정가)", orientation="h", marker_color=EPL_GREEN,
                ))
                fig.update_layout(
                    barmode="group", xaxis_title="시장가치 (백만 EUR)",
                    yaxis=dict(autorange="reversed"), height=450,
                    margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="#1a1a2e", paper_bgcolor="#0d0d1a", font_color="#ffffff",
                    legend=dict(orientation="h", yanchor="bottom", y=-0.25),
                )
                st.plotly_chart(fig, use_container_width=True, theme=None)
                st.caption(
                    "💡 **스카우터 포인트**: 초록 막대(예측가)가 빨간 막대(실제가)보다 길수록 저평가 폭이 큽니다. "
                    "가치 비율 2.0x 이상 선수는 즉시 접촉 검토 권장."
                )

                disp = uv[["player", "team", "season", "pos", "age_used", "market_value",
                           "predicted_value", "value_ratio"]].copy()
                disp["market_value"] = disp["market_value"].apply(lambda x: f"€{x/1e6:.1f}M" if x >= 1e6 else f"€{x/1e3:.0f}K")
                disp["predicted_value"] = disp["predicted_value"].apply(lambda x: f"€{x/1e6:.1f}M" if x >= 1e6 else f"€{x/1e3:.0f}K")
                disp["value_ratio"] = disp["value_ratio"].apply(lambda x: f"{x:.1f}x")
                disp.columns = ["선수", "팀", "시즌", "포지션", "나이", "실제 시장가치",
                                "예측 시장가치", "가치 비율"]
                st.dataframe(disp, use_container_width=True, hide_index=True)

            # ── 2024/25 영입 후보 (value_ratio > 1.0 완화 기준) ──
            st.markdown("---")
            st.markdown("#### 📋 2024/25 이적 시장 영입 후보 (S2 확장판, value_ratio > 1.0)")
            st.caption(
                "위 목록은 1.5배 이상 강력 저평가 기준입니다. "
                "2024/25 시즌은 모델 훈련 데이터가 최초 포함되어 임계값을 1.0배로 완화해 후보를 확장했습니다."
            )
            targets = load_s2_transfer_targets()
            if targets.empty:
                st.info("s2_v4_2025_transfer_targets.parquet 없음")
            else:
                t = targets.sort_values("value_ratio", ascending=False).copy()
                fig_t = go.Figure()
                fig_t.add_trace(go.Bar(
                    y=t["player"].head(16).tolist(),
                    x=(t["market_value"].head(16) / 1e6).tolist(),
                    name="실제 시장가치", orientation="h", marker_color=EPL_MAGENTA,
                ))
                fig_t.add_trace(go.Bar(
                    y=t["player"].head(16).tolist(),
                    x=(t["predicted_value"].head(16) / 1e6).tolist(),
                    name="예측 시장가치", orientation="h", marker_color=EPL_GREEN,
                ))
                fig_t.update_layout(
                    barmode="group", xaxis_title="시장가치 (백만 EUR)",
                    yaxis=dict(autorange="reversed"), height=450,
                    margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="#1a1a2e", paper_bgcolor="#0d0d1a", font_color="#ffffff",
                    legend=dict(orientation="h", yanchor="bottom", y=-0.25),
                )
                st.plotly_chart(fig_t, use_container_width=True, theme=None)

                disp_t = t[["player", "team", "pos", "age_used", "market_value",
                             "predicted_value", "value_ratio", "war_norm"]].copy()
                disp_t["market_value"] = disp_t["market_value"].apply(lambda x: f"€{x/1e6:.1f}M" if x >= 1e6 else f"€{x/1e3:.0f}K")
                disp_t["predicted_value"] = disp_t["predicted_value"].apply(lambda x: f"€{x/1e6:.1f}M" if x >= 1e6 else f"€{x/1e3:.0f}K")
                disp_t["value_ratio"] = disp_t["value_ratio"].apply(lambda x: f"{x:.2f}x")
                disp_t["war_norm"] = disp_t["war_norm"].apply(lambda x: f"{x:.1f}")
                disp_t.columns = ["선수", "팀", "포지션", "나이", "실제 시장가치", "예측 시장가치", "가치 비율", "WAR"]
                st.dataframe(disp_t, use_container_width=True, hide_index=True)
                st.caption("💡 WAR 90+ = 리그 최상위권. Neco Williams(91.5), Youri Tielemans(91.1), Yoane Wissa(90.0)가 고WAR-저가 대표 사례.")

        with sub2:
            overval = load_overvalued()
            if len(overval) == 0:
                st.warning("s2_v4_overvalued.parquet 데이터 없음")
            else:
                st.markdown("**예측 시장가치 < 실제 시장가치 0.5배** 미만 — 거품 주의, 계약 연장 재검토")
                ov = overval.sort_values("value_ratio", ascending=True).copy()

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=ov["player"].head(15).tolist(),
                    x=(ov["market_value"].head(15)/1e6).tolist(),
                    name="실제 시장가치 (현재 몸값)", orientation="h", marker_color=EPL_MAGENTA,
                ))
                fig.add_trace(go.Bar(
                    y=ov["player"].head(15).tolist(),
                    x=(ov["predicted_value"].head(15)/1e6).tolist(),
                    name="예측 시장가치 (실적 기반 적정가)", orientation="h", marker_color="#ff6b6b",
                ))
                fig.update_layout(
                    barmode="group", xaxis_title="시장가치 (백만 EUR)",
                    yaxis=dict(autorange="reversed"), height=450,
                    margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="#1a1a2e", paper_bgcolor="#0d0d1a", font_color="#ffffff",
                    legend=dict(orientation="h", yanchor="bottom", y=-0.25),
                )
                st.plotly_chart(fig, use_container_width=True, theme=None)
                st.caption(
                    "💡 **스카우터 포인트**: 빨간 막대(실제 몸값)가 주황 막대(예측 적정가)보다 훨씬 길면 거품. "
                    "영입 협상 시 이 수치를 근거로 가격 조정 요청 가능."
                )

                disp = ov[["player", "team", "season", "pos", "age_used", "market_value",
                           "predicted_value", "value_ratio"]].copy()
                disp["market_value"] = disp["market_value"].apply(lambda x: f"€{x/1e6:.1f}M" if x >= 1e6 else f"€{x/1e3:.0f}K")
                disp["predicted_value"] = disp["predicted_value"].apply(lambda x: f"€{x/1e6:.1f}M" if x >= 1e6 else f"€{x/1e3:.0f}K")
                disp["value_ratio"] = disp["value_ratio"].apply(lambda x: f"{x:.2f}x")
                disp.columns = ["선수", "팀", "시즌", "포지션", "나이", "실제 시장가치",
                                "예측 시장가치", "가치 비율"]
                st.dataframe(disp, use_container_width=True, hide_index=True)

        with sub3:
            # P6 절대 시장가치 예측 (XGBoost R²=0.87)
            p6 = _load_p6()
            if p6.empty:
                st.warning("p6_market_value_predictions.parquet 데이터 없음")
            else:
                st.markdown("**P6 절대 시장가치 예측** — XGBoost R²=0.87, 11,244건 선수-시즌 데이터")
                st.caption("S2(상대 저평가)와 달리 P6는 스탯 기반 **절대 적정 몸값**을 계산합니다.")

                # 포지션 필터
                pos_opts = ["전체"] + sorted(p6["pos_group"].dropna().unique().tolist())
                sel_pos = st.selectbox("포지션", pos_opts, key="p6_pos")
                p6f = p6 if sel_pos == "전체" else p6[p6["pos_group"] == sel_pos]

                # 저평가만 보기 토글
                only_under = st.checkbox("저평가 선수만 (예측가 > 실제가 1.5배)", value=True, key="p6_under")
                if only_under:
                    p6f = p6f[p6f["undervalued"] == True]

                st.markdown(f"**{len(p6f):,}명** 해당")

                # 상위 20명 차트
                top = p6f.nlargest(20, "undervalued_ratio")[
                    ["player", "team", "pos_group", "actual_market_value",
                     "predicted_market_value", "undervalued_ratio"]
                ].copy()
                top["실제 (M€)"] = (top["actual_market_value"] / 1e6).round(1)
                top["예측 (M€)"] = (top["predicted_market_value"] / 1e6).round(1)
                top["배율"] = top["undervalued_ratio"].apply(lambda x: f"{x:.1f}x")

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=top["player"].tolist(), x=top["실제 (M€)"].tolist(),
                    name="실제 시장가치", orientation="h", marker_color=EPL_MAGENTA,
                ))
                fig.add_trace(go.Bar(
                    y=top["player"].tolist(), x=top["예측 (M€)"].tolist(),
                    name="P6 예측 적정가", orientation="h", marker_color=EPL_GREEN,
                ))
                fig.update_layout(
                    barmode="group", xaxis_title="시장가치 (백만 EUR)",
                    yaxis=dict(autorange="reversed"), height=500,
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                )
                st.plotly_chart(fig, use_container_width=True, theme=None)

                disp = top[["player", "team", "pos_group", "실제 (M€)", "예측 (M€)", "배율"]].copy()
                disp.columns = ["선수", "팀", "포지션", "실제 (M€)", "예측 (M€)", "저평가 배율"]
                st.dataframe(disp, use_container_width=True, hide_index=True)
                st.caption("💡 **스카우터 포인트**: P6는 선수 스탯만으로 계산한 절대 적정가. S2와 함께 보면 두 모델이 모두 저평가 신호를 보내는 선수가 최우선 영입 후보.")

        with sub4:
            # ── WAR 순위 vs 시장가치 순위 역전 = 실력 대비 몸값이 낮은 진짜 바겐 ──
            st.markdown("### 🎯 WAR 순위 vs 시장가치 순위 역전 분석")
            st.markdown(
                "S2 모델이 잡지 못한 **FW/MID 저평가 후보**를 찾는 보조 지표입니다. "
                "포지션별로 WAR(실력) 순위 백분위에서 시장가치 순위 백분위를 뺀 값이 클수록 "
                "**실력에 비해 몸값이 저렴한 선수**입니다."
            )
            with st.expander("📖 분석 방법 설명", expanded=False):
                st.markdown("""
**undervalue_gap = WAR 순위(백분위) − 시장가치 순위(백분위)**

| 값 | 해석 |
|----|------|
| **+40% 이상** | 같은 포지션에서 WAR은 상위권인데 몸값은 하위권 → 강력한 바겐 신호 |
| **+20~40%** | 실력 대비 몸값이 낮음 → 영입 검토 대상 |
| **0 이하** | 몸값이 실력보다 높거나 같음 |

**왜 S2와 다른가?**
S2는 회귀 모델이 예측한 가격 vs 실제 가격을 비교합니다.
이 분석은 **리그 내 상대 순위**를 직접 비교하므로 S2가 훈련 데이터가 부족한 2024/25 시즌에도 작동합니다.
                """)

            scout_ratings = load_scout_ratings()
            decline_preds = load_decline_predictions()

            if scout_ratings.empty:
                st.warning("scout_ratings_v3.parquet 데이터 없음")
            else:
                # 포지션 필터 및 시즌 선택
                col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
                with col_f1:
                    avail_seasons = sorted(scout_ratings['season'].dropna().unique().tolist(), reverse=True)
                    sel_season = st.selectbox("시즌", avail_seasons, key="war_gap_season")
                with col_f2:
                    pos_opts_gap = ["전체", "FW", "MID", "DEF", "GK"]
                    sel_pos_gap = st.selectbox("포지션", pos_opts_gap, key="war_gap_pos")
                with col_f3:
                    min_gap = st.slider("최소 Gap (%)", min_value=0, max_value=60, value=20, step=5, key="war_gap_min")

                r = scout_ratings[scout_ratings['season'] == sel_season].copy()

                # pos_group 필터
                if sel_pos_gap != "전체":
                    r = r[r['pos_group'] == sel_pos_gap]

                if r.empty:
                    st.info(f"{sel_season} 시즌 데이터가 없습니다.")
                elif 'market_value' not in r.columns or 'war' not in r.columns:
                    st.warning("market_value 또는 war 컬럼이 없습니다.")
                else:
                    # 최소 출전시간 필터 + market_value > 0 (0이면 순위 계산 왜곡)
                    if 'total_min' in r.columns:
                        r = r[r['total_min'] >= 900]
                    r = r[r['market_value'].fillna(0) > 0].copy()

                    if r.empty:
                        st.info(f"{sel_season} 시즌에 시장가치 데이터가 있는 선수가 없습니다.")
                        return

                    # pos_group이 존재하는 경우 포지션 내 순위, 없으면 전체 순위
                    if 'pos_group' in r.columns and sel_pos_gap == "전체":
                        r['war_rank_pct'] = r.groupby('pos_group')['war'].rank(pct=True)
                        r['mv_rank_pct'] = r.groupby('pos_group')['market_value'].rank(pct=True)
                    else:
                        r['war_rank_pct'] = r['war'].rank(pct=True)
                        r['mv_rank_pct'] = r['market_value'].rank(pct=True)

                    r['undervalue_gap'] = r['war_rank_pct'] - r['mv_rank_pct']
                    candidates = r[r['undervalue_gap'] >= (min_gap / 100.0)].sort_values('undervalue_gap', ascending=False).copy()

                    # decline_risk 병합 (컬럼명: player_key, season_year, decline_prob_ensemble)
                    if not decline_preds.empty and 'season_year' in decline_preds.columns:
                        # season → season_year 변환 (예: "2024/25" → 2025)
                        # S6 데이터가 없으면 최신 시즌으로 폴백
                        try:
                            sel_year = int(sel_season.split('/')[0]) + 1
                            if sel_year not in decline_preds['season_year'].values:
                                sel_year = int(decline_preds['season_year'].max())
                        except Exception:
                            sel_year = None
                        if sel_year:
                            dec = (
                                decline_preds[decline_preds['season_year'] == sel_year]
                                [['player_key', 'decline_prob_ensemble']]
                                .sort_values('decline_prob_ensemble', ascending=False)
                                .drop_duplicates('player_key', keep='first')
                                .rename(columns={'player_key': 'player', 'decline_prob_ensemble': 'decline_prob'})
                            )
                            candidates = candidates.merge(dec, on='player', how='left')
                    if 'decline_prob' not in candidates.columns:
                        candidates['decline_prob'] = 0.0
                    candidates['decline_prob'] = candidates['decline_prob'].fillna(0.0)

                    st.markdown(f"**{len(candidates):,}명** 해당 (시즌: {sel_season}, 포지션: {sel_pos_gap}, 최소 Gap: {min_gap}%)")

                    if candidates.empty:
                        st.info("해당 조건에 맞는 선수가 없습니다. Gap 기준을 낮춰보세요.")
                    else:
                        # 차트: 상위 20명 scatter (WAR vs MV)
                        top_c = candidates.head(20).copy()
                        top_c['mv_m'] = top_c['market_value'] / 1e6
                        top_c['gap_pct'] = (top_c['undervalue_gap'] * 100).round(1)
                        top_c['decline_label'] = top_c['decline_prob'].apply(
                            lambda x: f"하락위험 {x*100:.0f}%"
                        )
                        top_c['marker_color'] = top_c['decline_prob'].apply(
                            lambda x: "#00ff87" if x < 0.3 else ("#ffd700" if x < 0.5 else "#ff6b6b")
                        )

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=top_c['mv_m'].tolist(),
                            y=top_c['war'].tolist(),
                            mode='markers+text',
                            text=top_c['player'].tolist(),
                            textposition='top center',
                            marker=dict(
                                size=top_c['gap_pct'].clip(8, 30).tolist(),
                                color=top_c['marker_color'].tolist(),
                                line=dict(width=1, color='#37003c'),
                            ),
                            customdata=top_c[['gap_pct', 'decline_label', 'pos_group']].values.tolist(),
                            hovertemplate=(
                                "<b>%{text}</b><br>"
                                "시장가치: €%{x:.1f}M<br>"
                                "WAR: %{y:.1f}<br>"
                                "저평가 Gap: %{customdata[0]:.1f}%<br>"
                                "%{customdata[1]}<br>"
                                "포지션: %{customdata[2]}<extra></extra>"
                            ),
                        ))
                        fig.update_layout(
                            xaxis_title="시장가치 (백만 EUR)",
                            yaxis_title="WAR (백분위)",
                            height=480,
                            plot_bgcolor="#1a1a2e", paper_bgcolor="#0d0d1a", font_color="#ffffff",
                            margin=dict(l=10, r=10, t=30, b=10),
                        )
                        fig.add_annotation(
                            text="← 저렴 / 비쌈 →",
                            xref="paper", yref="paper",
                            x=0.5, y=-0.08, showarrow=False,
                            font=dict(size=11, color="gray"),
                        )
                        st.plotly_chart(fig, use_container_width=True, theme=None)
                        st.caption(
                            "🟢 하락위험 30% 미만 · 🟡 30~50% · 🔴 50%+ | "
                            "버블 크기 = 저평가 Gap 크기 | "
                            "**좌상단**이 최고 바겐 (저렴+WAR 높음)"
                        )

                        # 테이블
                        disp_cols = ['player', 'team', 'pos_group', 'age', 'war',
                                     'mv_m', 'gap_pct', 'decline_prob']
                        disp_cols = [c for c in disp_cols if c in top_c.columns]
                        tbl = top_c[disp_cols].copy()

                        # 포맷
                        if 'war' in tbl.columns:
                            tbl['war'] = tbl['war'].round(1)
                        if 'mv_m' in tbl.columns:
                            tbl['mv_m'] = tbl['mv_m'].apply(lambda x: f"€{x:.1f}M")
                        if 'gap_pct' in tbl.columns:
                            tbl['gap_pct'] = tbl['gap_pct'].apply(lambda x: f"+{x:.1f}%")
                        if 'decline_prob' in tbl.columns:
                            tbl['decline_prob'] = tbl['decline_prob'].apply(
                                lambda x: f"{'🔴' if x >= 0.5 else '🟡' if x >= 0.3 else '🟢'} {x*100:.0f}%"
                            )

                        col_map = {
                            'player': '선수', 'team': '팀', 'pos_group': '포지션',
                            'age': '나이', 'war': 'WAR', 'mv_m': '시장가치',
                            'gap_pct': '저평가 Gap', 'decline_prob': '하락 위험',
                        }
                        tbl = tbl.rename(columns={k: v for k, v in col_map.items() if k in tbl.columns})
                        st.dataframe(tbl, use_container_width=True, hide_index=True)

                        st.caption(
                            "💡 **스카우터 포인트**: '저평가 Gap'이 크고 '하락 위험'이 🟢인 선수가 최우선 영입 후보입니다. "
                            "S2·P6가 같은 선수를 모두 저평가로 지목하면 신뢰도가 더 높아집니다."
                        )

    # ══════════════════════════════════════════
    # TAB 4: 하락 주의보 (S6) - 시즌별 선택
    # ══════════════════════════════════════════
    with tab4:
        st.subheader("S6. 하락 감지")
        with st.expander("📖 하락 감지 모델 해석 가이드 (스카우터 필독)", expanded=False):
            st.markdown("""
**S6 모델**: 선수의 최근 3시즌 성과 추세·나이·포지션별 피크 연령을 결합한 앙상블 하락 예측

**하락 확률 구간 해석:**
| 구간 | 의미 | 행동 |
|------|------|------|
| **70%+** | 고위험 — 성과 하락 거의 확실 | 계약 연장 불가, 방출 검토 |
| **50~70%** | 중위험 — 하락세 시작 | 단기 계약만, 옵션 없이 |
| **30~50%** | 주의 — 모니터링 필요 | 다음 시즌 관찰 후 판단 |
| **30% 미만** | 양호 — 현재 안정적 | 계약 연장 가능 |

**지표 해석:**
- **성과 추세(perf_slope)**: 음수(-) = 하락 중, 양수(+) = 상승 중
- **성과 점수(perf_score)**: 현 시즌 종합 성과 (높을수록 좋음)

**평균 회귀 경보**: 올 시즌 커리어 대비 이례적으로 좋은 시즌을 보낸 선수.
다음 시즌 성과가 평균으로 돌아올 가능성이 높음 → 고가 영입 전 반드시 확인.
            """)
        decline = load_decline_predictions()
        if len(decline) == 0:
            st.warning("decline_predictions_v3.parquet 데이터 없음")
        else:
            # ── 시즌 선택 (두 탭 공유) ──
            season_years = sorted(decline["season_year"].unique(), reverse=True)
            sel_dec_year = st.selectbox(
                "시즌 선택",
                season_years,
                format_func=lambda y: f"{y}/{str(y+1)[-2:]} 시즌",
                key="decline_season",
            )
            sel_data = (
                decline[decline["season_year"] == sel_dec_year]
                .sort_values("min", ascending=False)
                .drop_duplicates(subset=["player_key"], keep="first")
                .copy()
            )
            st.caption(f"현재 선택: **{sel_dec_year}/{str(sel_dec_year+1)[-2:]} 시즌** | 분석 대상 {len(sel_data):,}명")

            sub_a, sub_b = st.tabs(["커리어 하락 주의 (28세+)", "평균 회귀 경보"])

            with sub_a:
                st.markdown("**28세 이상 + 하락 확률 50%+** — 계약 연장 신중 검토 대상")
                watch = sel_data[
                    (sel_data["age"] >= 28) &
                    (sel_data["decline_prob_ensemble"] >= 0.5)
                ].sort_values("decline_prob_ensemble", ascending=False)

                if len(watch):
                    pos_color = {"FW": EPL_MAGENTA, "MID": EPL_CYAN, "DEF": EPL_GREEN, "GK": EPL_PURPLE}
                    fig = go.Figure()
                    for pos in watch["pos_group"].unique():
                        pdf = watch[watch["pos_group"] == pos]
                        fig.add_trace(go.Scatter(
                            x=pdf["age"].tolist(),
                            y=pdf["decline_prob_ensemble"].tolist(),
                            mode="markers",
                            name=pos,
                            marker=dict(size=10, color=pos_color.get(pos, "#888"), opacity=0.75),
                            text=pdf["player_key"].tolist(),
                            customdata=list(zip(
                                pdf["team"].tolist(),
                                pdf["perf_slope"].fillna(0).tolist(),
                                pdf["perf_score"].fillna(0).tolist(),
                            )),
                            hovertemplate=(
                                "<b>%{text}</b><br>"
                                "팀: %{customdata[0]}<br>"
                                "성과 추세: %{customdata[1]:+.3f}<br>"
                                "성과 점수: %{customdata[2]:.3f}<br>"
                                "나이: %{x}<br>"
                                "하락 확률: %{y:.1%}"
                                "<extra>%{fullData.name}</extra>"
                            ),
                        ))
                    fig.add_hline(y=0.7, line_dash="dash", line_color="red",
                                  annotation_text="고위험 70% — 계약 연장 불가 권고")
                    fig.add_hline(y=0.5, line_dash="dot", line_color="orange",
                                  annotation_text="주의선 50%")
                    fig.update_layout(
                        xaxis_title="나이", yaxis_title="하락 확률",
                        yaxis=dict(tickformat=".0%", range=[0.4, 1.05]),
                        height=450, plot_bgcolor="#1a1a2e", paper_bgcolor="#0d0d1a", font_color="#ffffff", margin=dict(l=10, r=10, t=10, b=10),
                        legend_title="포지션",
                    )
                    st.plotly_chart(fig, use_container_width=True, theme=None)

                    # 요약 통계
                    high_risk = watch[watch["decline_prob_ensemble"] >= 0.7]
                    mid_risk = watch[(watch["decline_prob_ensemble"] >= 0.5) & (watch["decline_prob_ensemble"] < 0.7)]
                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("분석 대상", f"{len(watch)}명")
                    rc2.metric("고위험 (70%+)", f"{len(high_risk)}명", delta=None)
                    rc3.metric("중위험 (50~70%)", f"{len(mid_risk)}명")

                    disp = watch[["player_key", "team", "pos_group", "age",
                                  "decline_prob_ensemble", "perf_slope", "perf_score"]].head(25).copy()
                    disp.columns = ["선수", "팀", "포지션", "나이", "하락 확률",
                                    "성과 추세", "성과 점수"]
                    disp["하락 확률"] = disp["하락 확률"].apply(lambda x: f"{x:.1%}")
                    disp["성과 추세"] = disp["성과 추세"].apply(lambda x: f"{x:+.3f}" if pd.notna(x) else "N/A")
                    disp["성과 점수"] = disp["성과 점수"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
                    st.dataframe(disp, use_container_width=True, hide_index=True)
                    st.caption(
                        "💡 **스카우터 포인트**: 성과 추세가 음수이면서 하락 확률 70% 이상이면 이미 하락 중. "
                        "이 선수들의 계약 연장은 단기(1년 이하) + 성과 연동 조건 권고."
                    )
                else:
                    st.info(f"{sel_dec_year}/{str(sel_dec_year+1)[-2:]} 시즌에 해당하는 하락 주의 선수가 없습니다.")

            with sub_b:
                st.markdown("**아웃라이어 시즌 후 평균 회귀 가능성** — 이 성적이 지속될까? 고가 영입 전 필독")
                st.caption(
                    "커리어 평균 대비 이번 시즌 성과가 통계적으로 이례적(Z-score 기준)인 선수. "
                    "다음 시즌 성과가 커리어 평균으로 회귀할 가능성이 높아 영입 가격 협상에 활용하세요."
                )
                regression = sel_data[sel_data["is_outlier_season"] == True].sort_values(
                    "perf_score", ascending=False
                )
                if len(regression):
                    # 올시즌 vs 커리어 평균 비교 차트
                    top_reg = regression.head(15).copy()
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        y=top_reg["player_key"].tolist(),
                        x=top_reg["perf_score"].tolist(),
                        name="올시즌 성과",
                        orientation="h",
                        marker_color=EPL_CYAN,
                    ))
                    fig.add_trace(go.Bar(
                        y=top_reg["player_key"].tolist(),
                        x=top_reg["career_perf_mean"].tolist(),
                        name="커리어 평균",
                        orientation="h",
                        marker_color=EPL_PURPLE,
                    ))
                    fig.update_layout(
                        barmode="group",
                        xaxis_title="성과 점수",
                        yaxis=dict(autorange="reversed"),
                        height=420,
                        plot_bgcolor="#1a1a2e", paper_bgcolor="#0d0d1a", font_color="#ffffff",
                        margin=dict(l=10, r=10, t=10, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
                    )
                    st.plotly_chart(fig, use_container_width=True, theme=None)

                    disp = regression[["player_key", "team", "pos_group", "age",
                                       "perf_score", "career_perf_mean",
                                       "decline_prob_ensemble"]].head(25).copy()
                    disp.columns = ["선수", "팀", "포지션", "나이",
                                    "올시즌 성과", "커리어 평균", "하락 확률"]
                    disp["올시즌 성과"] = disp["올시즌 성과"].apply(lambda x: f"{x:.3f}")
                    disp["커리어 평균"] = disp["커리어 평균"].apply(lambda x: f"{x:.3f}")
                    disp["하락 확률"] = disp["하락 확률"].apply(lambda x: f"{x:.1%}")
                    st.dataframe(disp, use_container_width=True, hide_index=True)
                    st.caption(
                        "💡 **스카우터 포인트**: 올시즌 성과가 커리어 평균을 크게 웃도는 선수는 "
                        "다음 시즌 성과가 평균으로 회귀할 가능성이 높습니다. "
                        "이 선수들에게 고가 영입 제안을 받았다면 커리어 평균 기준으로 협상하세요."
                    )
                else:
                    st.info(f"{sel_dec_year}/{str(sel_dec_year+1)[-2:]} 시즌에 해당하는 평균 회귀 대상 선수가 없습니다.")

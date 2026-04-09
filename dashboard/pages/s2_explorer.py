"""S2 저평가 선수 탐색기 - 시장 저평가 선수 발굴 전용 페이지.

s2_v4_all_predictions.parquet 기반으로 예산/포지션/나이 조건에 맞는
저평가 선수를 탐색하고 즉시 분석으로 연결합니다.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.data_loader import load_undervalued, load_scout_ratings
from dashboard.utils.image_utils import get_player_image_b64

EPL_PURPLE = "#37003c"
EPL_MAGENTA = "#e90052"
EPL_GREEN = "#00ff87"
EPL_CYAN = "#04f5ff"


@st.cache_data(ttl=3600)
def _load_s2_all() -> pd.DataFrame:
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent.parent / "data" / "scout" / "s2_v4_all_predictions.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def render():
    st.title("💎 S2 저평가 선수 탐색기")
    st.caption("모델 예측 가치 대비 시장 저평가 선수를 발굴합니다. value_ratio = 예측가 / 시장가. 1 초과이면 시장 저평가 (예측가 > 시장가).")

    s2_df = _load_s2_all()
    ratings = load_scout_ratings()

    if s2_df.empty:
        st.error("s2_v4_all_predictions.parquet 데이터가 없습니다.")
        return

    # WAR 병합
    if not ratings.empty and "player" in ratings.columns:
        latest_war = (
            ratings.sort_values("season", ascending=False)
            .groupby("player", as_index=False)
            .first()[["player", "war", "tier"]]
        )
        s2_df = s2_df.merge(latest_war, on="player", how="left")

    # 최신 시즌만 (중복 선수 있을 경우 최신 시즌 우선)
    if "season" in s2_df.columns:
        latest_season = s2_df["season"].max()
        s2_latest = s2_df[s2_df["season"] == latest_season].copy()
    else:
        s2_latest = s2_df.copy()

    # ── 필터 ────────────────────────────────────────────────────────
    st.markdown("### 🔍 탐색 조건")
    fc1, fc2, fc3, fc4 = st.columns(4)

    with fc1:
        budget_m = st.slider(
            "최대 예산 (€M)",
            min_value=1, max_value=200,
            value=50, step=1,
            key="s2_budget",
        )
        budget = budget_m * 1_000_000

    with fc2:
        pos_opts = ["전체", "FW", "MF", "DF", "GK"]
        sel_pos = st.selectbox("포지션", pos_opts, key="s2_pos")

    with fc3:
        age_range = st.slider(
            "나이 범위",
            min_value=16, max_value=38,
            value=(18, 28),
            key="s2_age",
        )

    with fc4:
        min_war = st.slider(
            "최소 WAR 백분위",
            min_value=0, max_value=100,
            value=30,
            key="s2_min_war",
        )

    ratio_threshold = st.slider(
        "최소 저평가 기준 (value_ratio 하한 — 높을수록 더 저평가)",
        min_value=1.00, max_value=5.00,
        value=1.10, step=0.05,
        key="s2_ratio",
        help="value_ratio = 모델예측가치 / 시장가치. 1.0 초과이면 저평가 (예측가 > 시장가).",
    )

    st.markdown("---")

    # ── 필터 적용 ────────────────────────────────────────────────────
    filtered = s2_latest.copy()

    if "market_value" in filtered.columns:
        filtered = filtered[filtered["market_value"] <= budget]

    if sel_pos != "전체" and "pos" in filtered.columns:
        filtered = filtered[filtered["pos"].str.contains(sel_pos, na=False)]

    if "age_used" in filtered.columns:
        filtered = filtered[
            (filtered["age_used"] >= age_range[0]) &
            (filtered["age_used"] <= age_range[1])
        ]

    if "war" in filtered.columns and min_war > 0:
        filtered = filtered[filtered["war"] >= min_war]

    if "value_ratio" in filtered.columns:
        filtered = filtered[filtered["value_ratio"] >= ratio_threshold]

    # 저평가 순 정렬 (value_ratio 높은 순 = 예측가/시장가 비율 높음 = 더 저평가)
    if "value_ratio" in filtered.columns:
        filtered = filtered.sort_values("value_ratio", ascending=False)

    # ── 요약 메트릭 ─────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("필터 결과", f"{len(filtered)}명")
    if "value_ratio" in filtered.columns and len(filtered) > 0:
        m2.metric("최고 저평가율", f"{(filtered['value_ratio'].max() - 1) * 100:.1f}%", help="예측가가 시장가를 초과하는 최대 비율")
        m3.metric("평균 value_ratio", f"{filtered['value_ratio'].mean():.3f}")
    if "market_value" in filtered.columns and len(filtered) > 0:
        avg_mv = filtered["market_value"].mean()
        m4.metric("평균 시장가치", f"€{avg_mv/1_000_000:.1f}M")

    if len(filtered) == 0:
        st.info("조건에 맞는 선수가 없습니다. 필터 조건을 완화해 보세요.")
        return

    # ── Top 탐색 결과 카드 ───────────────────────────────────────────
    st.markdown("### 🏆 저평가 탐색 결과 (Top 30)")

    top_n = filtered.head(30)
    for rank, (_, row) in enumerate(top_n.iterrows(), 1):
        player = row.get("player", "")
        team = row.get("team", "")
        pos = row.get("pos", "")
        age = row.get("age_used", None)
        mv = row.get("market_value", None)
        pred_val = row.get("predicted_value", None)
        vr = row.get("value_ratio", None)
        war = row.get("war", None)
        tier = row.get("tier", "")
        goals = row.get("goals_p90", None)
        assists = row.get("assists_p90", None)

        # 저평가 정도에 따른 색상 (value_ratio = 예측가/시장가, 높을수록 저평가)
        if vr is not None:
            if vr >= 2.0:
                card_color = EPL_MAGENTA  # 예측가 2배 이상 = 극저평가
                badge = "🔥 극저평가"
            elif vr >= 1.5:
                card_color = EPL_GREEN
                badge = "💎 저평가"
            else:
                card_color = EPL_CYAN
                badge = "🔍 소폭 저평가"
        else:
            card_color = "#888"
            badge = ""

        img_b64 = get_player_image_b64(player, size=(52, 52))
        img_html = (
            f'<img src="data:image/jpeg;base64,{img_b64}" '
            f'style="width:48px;height:48px;object-fit:cover;border-radius:50%;'
            f'border:2px solid {card_color};flex-shrink:0;">'
            if img_b64 else
            '<div style="width:48px;height:48px;border-radius:50%;background:#2a2a4a;'
            'display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0;">👤</div>'
        )

        mv_str = f"€{mv/1_000_000:.1f}M" if mv and not pd.isna(mv) else "-"
        pred_str = f"€{pred_val/1_000_000:.1f}M" if pred_val and not pd.isna(pred_val) else "-"
        vr_str = f"{vr:.3f}" if vr is not None else "-"
        discount_str = f"(+{(vr-1)*100:.1f}% 업사이드)" if vr is not None and vr > 1 else ""
        war_str = f"WAR {war:.0f}" if war and not pd.isna(war) else ""
        age_str = f"{int(age)}세" if age and not pd.isna(age) else ""
        stats_str = ""
        if goals is not None and not pd.isna(goals):
            stats_str += f"골 {goals:.2f}"
        if assists is not None and not pd.isna(assists):
            stats_str += f" · 어시 {assists:.2f}"

        col_card, col_btn = st.columns([10, 1])
        with col_card:
            st.markdown(
                f"""<div style='display:flex;align-items:center;gap:12px;
                background:#1a1a2e;border-radius:10px;padding:10px 14px;
                margin-bottom:6px;border-left:4px solid {card_color};'>
                <div style='color:{card_color};font-weight:700;font-size:1.1em;min-width:28px;'>#{rank}</div>
                {img_html}
                <div style='flex:1;min-width:0;'>
                  <div style='font-weight:700;color:#fff;font-size:0.95em;'>{badge} {player}</div>
                  <div style='color:#aaa;font-size:0.8em;'>{team} · {pos} · {age_str}{f" · {tier}" if tier else ""}</div>
                  <div style='color:#ccc;font-size:0.8em;margin-top:2px;'>
                    시장가 {mv_str} → 예측가 {pred_str} | ratio: <span style='color:{card_color};font-weight:700;'>{vr_str}</span> {discount_str}
                  </div>
                  <div style='color:#aaa;font-size:0.78em;'>
                    {war_str}{f" · {stats_str}/90분" if stats_str else ""}
                  </div>
                </div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col_btn:
            if st.button("🔍", key=f"s2_goto_{player}_{rank}", help=f"{player} 즉시 분석"):
                st.session_state["scout_report_player"] = player
                st.session_state["_nav_target"] = "🔍 선수 즉시 분석"
                st.rerun()

    # ── 테이블 뷰 ─────────────────────────────────────────────────────
    with st.expander("📋 전체 테이블로 보기"):
        tbl_cols = [c for c in ["player", "team", "pos", "age_used", "market_value", "predicted_value", "value_ratio", "war_norm", "goals_p90", "assists_p90", "tier"] if c in filtered.columns]
        tbl = filtered[tbl_cols].copy()

        tbl_rename = {
            "player": "선수", "team": "팀", "pos": "포지션", "age_used": "나이",
            "market_value": "시장가치(€M)", "predicted_value": "예측가치(€M)",
            "value_ratio": "value_ratio", "war_norm": "WAR(백분위)",
            "goals_p90": "골/90분", "assists_p90": "어시/90분", "tier": "등급",
        }
        tbl = tbl.rename(columns={k: v for k, v in tbl_rename.items() if k in tbl.columns})

        if "시장가치(€M)" in tbl.columns:
            tbl["시장가치(€M)"] = tbl["시장가치(€M)"].apply(lambda x: round(x / 1_000_000, 2) if pd.notna(x) else None)
        if "예측가치(€M)" in tbl.columns:
            tbl["예측가치(€M)"] = tbl["예측가치(€M)"].apply(lambda x: round(x / 1_000_000, 2) if pd.notna(x) else None)
        if "value_ratio" in tbl.columns:
            tbl["value_ratio"] = tbl["value_ratio"].apply(lambda x: round(x, 4) if pd.notna(x) else None)

        st.dataframe(tbl.reset_index(drop=True), use_container_width=True, hide_index=True)

    # ── 산점도: 시장가치 vs 예측가치 ────────────────────────────────
    st.markdown("### 📈 시장가치 vs 예측가치 분포")
    if "market_value" in filtered.columns and "predicted_value" in filtered.columns and len(filtered) > 0:
        plot_df = filtered.head(80).copy()
        plot_df["mv_m"] = plot_df["market_value"] / 1_000_000
        plot_df["pv_m"] = plot_df["predicted_value"] / 1_000_000

        fig_scatter = px.scatter(
            plot_df,
            x="mv_m", y="pv_m",
            text="player",
            color="value_ratio" if "value_ratio" in plot_df.columns else None,
            color_continuous_scale=[[0, EPL_GREEN], [0.5, "#FFD700"], [1, EPL_MAGENTA]],
            labels={"mv_m": "시장가치 (€M)", "pv_m": "모델 예측가치 (€M)", "value_ratio": "value_ratio"},
            hover_data={"player": True, "team": True, "value_ratio": ":.3f"},
        )
        # 동일선 (y=x)
        max_val = max(plot_df["mv_m"].max(), plot_df["pv_m"].max())
        fig_scatter.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines",
            line=dict(color="#888", dash="dash", width=1),
            name="시장가치=예측가치",
            showlegend=True,
        ))
        fig_scatter.update_traces(textposition="top center", selector=dict(mode="markers+text"))
        fig_scatter.update_layout(
            paper_bgcolor="#0d0d1a",
            plot_bgcolor="#1a1a2e",
            font_color="#fff",
            margin=dict(t=20, b=20, l=20, r=20),
            height=420,
        )
        st.plotly_chart(fig_scatter, use_container_width=True, theme=None)
        st.caption("💡 선 위쪽(예측가치 > 시장가치) = 저평가 선수. 선 아래쪽 = 시장 고평가 선수.")

    # ── 쇼트리스트 일괄 추가 ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⭐ 쇼트리스트 일괄 추가")
    top_names = filtered.head(20)["player"].tolist()
    batch_selected = st.multiselect(
        "추가할 선수 선택 (최대 10명)",
        options=top_names,
        default=[],
        max_selections=10,
        key="s2_batch_add",
    )
    if batch_selected:
        if st.button("⭐ 선택 선수 쇼트리스트에 추가", use_container_width=True, key="s2_batch_btn"):
            sl = st.session_state.setdefault("shortlist", {})
            from datetime import datetime
            added_count = 0
            for pname in batch_selected:
                if pname not in sl:
                    prow = filtered[filtered["player"] == pname]
                    p_team = prow.iloc[0].get("team", "") if len(prow) > 0 else ""
                    sl[pname] = {
                        "team": p_team,
                        "note": "S2 저평가 탐색기 발굴",
                        "priority": "🟡 모니터링",
                        "added": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    }
                    added_count += 1
            st.success(f"✅ {added_count}명 쇼트리스트에 추가되었습니다.")
            st.rerun()

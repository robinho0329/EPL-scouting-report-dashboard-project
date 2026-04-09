"""홈 대시보드 - 스카우트 브리핑 랜딩 페이지.

대시보드 진입 시 첫 화면. 오늘의 주요 인사이트를 한눈에 확인.
"""
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.data_loader import (
    load_scout_ratings, load_undervalued, load_growth_predictions_v4,
    load_decline_predictions,
)
from dashboard.utils.image_utils import get_player_image_b64, get_team_logo_b64

EPL_PURPLE = "#37003c"
EPL_MAGENTA = "#e90052"
EPL_GREEN = "#00ff87"
EPL_CYAN = "#04f5ff"

GROWTH_ICON = {"Improving": "🟢", "Stable": "🟡", "Declining": "🔴"}


def _mini_player_card(player: str, team: str, label: str, value: str, color: str, key: str):
    """작은 선수 카드 with 즉시분석 버튼."""
    img_b64 = get_player_image_b64(player, size=(48, 48))
    img_html = (
        f'<img src="data:image/jpeg;base64,{img_b64}" '
        f'style="width:44px;height:44px;object-fit:cover;border-radius:50%;'
        f'border:2px solid {color};flex-shrink:0;">'
        if img_b64 else
        '<div style="width:44px;height:44px;border-radius:50%;background:#2a2a4a;'
        'display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0;">👤</div>'
    )
    col_card, col_btn = st.columns([9, 1])
    with col_card:
        st.markdown(
            f"""<div style='display:flex;align-items:center;gap:10px;
            background:#1a1a2e;border-radius:8px;padding:8px 12px;
            margin-bottom:4px;border-left:3px solid {color};'>
            {img_html}
            <div style='flex:1;'>
              <div style='font-weight:700;color:#fff;font-size:0.88em;'>{player}</div>
              <div style='color:#aaa;font-size:0.75em;'>{team}</div>
              <div style='color:{color};font-size:0.78em;font-weight:600;'>{label}: {value}</div>
            </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_btn:
        if st.button("🔍", key=key, help=f"{player} 즉시 분석"):
            st.session_state["scout_report_player"] = player
            st.session_state["_nav_target"] = "🔍 선수 즉시 분석"
            st.rerun()


def render():
    st.title("⚽ EPL 스카우트 인텔리전스")
    st.markdown(f"**오늘의 스카우트 브리핑** — {date.today().strftime('%Y년 %m월 %d일')}")
    st.caption("모든 모델의 핵심 인사이트를 한눈에. 아래 카드에서 즉시 분석 🔍 버튼으로 선수 리포트로 이동합니다.")
    st.markdown("---")

    # 데이터 로딩
    ratings = load_scout_ratings()
    undervalued = load_undervalued()
    growth_v4 = load_growth_predictions_v4()
    decline = load_decline_predictions()

    if ratings.empty:
        st.error("데이터를 불러올 수 없습니다. 파이프라인을 먼저 실행하세요.")
        return

    latest_season = ratings["season"].max() if "season" in ratings.columns else None
    latest = ratings[ratings["season"] == latest_season].copy() if latest_season else ratings.copy()

    # 쇼트리스트
    shortlist = st.session_state.get("shortlist", {})

    # ── 요약 KPI 메트릭 ─────────────────────────────────────────────────
    km1, km2, km3, km4, km5 = st.columns(5)
    km1.metric("분석 선수", f"{len(latest):,}명", help="최신 시즌 데이터")
    km2.metric("저평가 기회", f"{len(undervalued):,}명", help="S2 저평가 판정 선수")
    if not growth_v4.empty and "pred_xgb" in growth_v4.columns:
        improving_count = (growth_v4["pred_xgb"] == "Improving").sum()
        km3.metric("성장 중인 선수", f"{improving_count:,}명", help="P7 Improving 예측")
    else:
        km3.metric("성장 중인 선수", "-")
    km4.metric("⭐ 쇼트리스트", f"{len(shortlist)}명", help="내 관심 선수 목록")
    if latest_season:
        km5.metric("시즌", latest_season)

    st.markdown("---")

    # ── 3열 레이아웃 ─────────────────────────────────────────────────────
    left_col, mid_col, right_col = st.columns(3)

    # ── [LEFT] WAR 리더 Top 5 ──────────────────────────────────────────
    with left_col:
        st.markdown("### 🏆 WAR 리더 Top 5")
        st.caption(f"{latest_season} 시즌 전체 포지션")
        if "war" in latest.columns:
            top5_war = latest.nlargest(5, "war")[["player", "team", "war", "tier", "pos_group"]]
            for rank, (_, row) in enumerate(top5_war.iterrows(), 1):
                color = EPL_MAGENTA if rank == 1 else (EPL_GREEN if rank <= 3 else EPL_CYAN)
                _mini_player_card(
                    player=row["player"],
                    team=f"{row.get('team','')} · {row.get('pos_group','')}",
                    label="WAR",
                    value=f"{row['war']:.1f}",
                    color=color,
                    key=f"home_war_{rank}",
                )
        else:
            st.info("WAR 데이터 없음")

    # ── [MID] 저평가 기회 Top 5 ──────────────────────────────────────────
    with mid_col:
        st.markdown("### 💎 저평가 기회 Top 5")
        st.caption("S2 모델 — 예측가/시장가 비율 높은 순 (저평가)")
        if not undervalued.empty and "player" in undervalued.columns:
            # value_ratio = predicted/market → 높을수록 저평가
            uv_sorted = undervalued.sort_values("value_ratio", ascending=False) if "value_ratio" in undervalued.columns else undervalued.head(5)
            uv_top5 = uv_sorted.head(5)
            # WAR 병합
            if "war" in latest.columns:
                uv_top5 = uv_top5.merge(latest[["player", "team", "war"]], on="player", how="left", suffixes=("", "_r"))
            for rank, (_, row) in enumerate(uv_top5.iterrows(), 1):
                vr = row.get("value_ratio", None)
                mv = row.get("market_value", None)
                pv = row.get("predicted_value", None)
                vr_str = f"{vr:.3f}" if vr is not None else "-"
                mv_str = f"€{mv/1e6:.1f}M→€{pv/1e6:.1f}M" if (mv and pv and not pd.isna(mv) and not pd.isna(pv)) else vr_str
                team = row.get("team", row.get("team_r", ""))
                _mini_player_card(
                    player=row["player"],
                    team=str(team) if pd.notna(team) else "",
                    label="ratio",
                    value=vr_str,
                    color=EPL_GREEN,
                    key=f"home_uv_{rank}",
                )
        else:
            st.info("저평가 데이터 없음")

    # ── [RIGHT] 성장 급등 선수 Top 5 ───────────────────────────────────
    with right_col:
        st.markdown("### 🚀 성장 급등 선수")
        st.caption("P7 Improving + 나이 ≤ 25세")
        if not growth_v4.empty and "player" in growth_v4.columns:
            pred_col = "pred_xgb" if "pred_xgb" in growth_v4.columns else "pred_ensemble"
            improving = growth_v4[growth_v4[pred_col] == "Improving"].copy()
            # WAR + 나이 병합
            if not latest.empty:
                improving = improving.merge(
                    latest[["player", "team", "war", "age", "pos_group"]],
                    on="player", how="left"
                )
                if "age" in improving.columns:
                    improving = improving[improving["age"].fillna(99) <= 25]
            if "war" in improving.columns:
                improving = improving.sort_values("war", ascending=False)
            top5_grow = improving.head(5)
            for rank, (_, row) in enumerate(top5_grow.iterrows(), 1):
                war = row.get("war", None)
                age = row.get("age", None)
                team = row.get("team", "")
                _mini_player_card(
                    player=row["player"],
                    team=f"{team} · {int(age)}세" if (age and not pd.isna(age)) else str(team),
                    label="WAR",
                    value=f"{war:.1f}" if (war and not pd.isna(war)) else "-",
                    color="#FFD700",
                    key=f"home_grow_{rank}",
                )
        else:
            st.info("성장 예측 데이터 없음")

    st.markdown("---")

    # ── 쇼트리스트 현황 ──────────────────────────────────────────────────
    st.markdown("### ⭐ 나의 쇼트리스트 현황")
    if shortlist:
        sl_rows = list(shortlist.items())
        priority_order = {"🔴 즉시": 0, "🟡 모니터링": 1, "🟢 장기": 2}
        sl_rows.sort(key=lambda x: priority_order.get(x[1].get("priority", ""), 9))

        sl_cols = st.columns(min(len(sl_rows), 5))
        for i, (player, info) in enumerate(sl_rows[:5]):
            priority = info.get("priority", "🟡 모니터링")
            p_color = {"🔴 즉시": EPL_MAGENTA, "🟡 모니터링": "#FFD700", "🟢 장기": EPL_GREEN}.get(priority, "#888")
            img_b64 = get_player_image_b64(player, size=(60, 60))

            # WAR 조회
            p_war_row = latest[latest["player"] == player]
            p_war = p_war_row.iloc[0].get("war", None) if not p_war_row.empty else None

            with sl_cols[i % 5]:
                img_html = (
                    f'<img src="data:image/jpeg;base64,{img_b64}" '
                    f'style="width:52px;height:52px;object-fit:cover;border-radius:50%;'
                    f'border:3px solid {p_color};">'
                    if img_b64 else
                    '<div style="width:52px;height:52px;border-radius:50%;background:#2a2a4a;'
                    'display:flex;align-items:center;justify-content:center;font-size:22px;">👤</div>'
                )
                st.markdown(
                    f"""<div style='text-align:center;background:#1a1a2e;border-radius:8px;
                    padding:8px 4px;border-top:3px solid {p_color};'>
                    <div style='display:flex;justify-content:center;margin-bottom:4px;'>{img_html}</div>
                    <div style='font-weight:700;color:#fff;font-size:0.8em;word-break:break-all;'>{player}</div>
                    <div style='color:{p_color};font-size:0.72em;'>{priority}</div>
                    <div style='color:#aaa;font-size:0.72em;'>WAR {p_war:.0f}</div>
                    </div>""" if (p_war and not pd.isna(p_war)) else
                    f"""<div style='text-align:center;background:#1a1a2e;border-radius:8px;
                    padding:8px 4px;border-top:3px solid {p_color};'>
                    <div style='display:flex;justify-content:center;margin-bottom:4px;'>{img_html}</div>
                    <div style='font-weight:700;color:#fff;font-size:0.8em;word-break:break-all;'>{player}</div>
                    <div style='color:{p_color};font-size:0.72em;'>{priority}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                if st.button("🔍", key=f"home_sl_{i}", help=f"{player} 즉시 분석", use_container_width=True):
                    st.session_state["scout_report_player"] = player
                    st.session_state["_nav_target"] = "🔍 선수 즉시 분석"
                    st.rerun()

        if len(shortlist) > 5:
            st.caption(f"... 외 {len(shortlist)-5}명. ⭐ 나의 쇼트리스트 페이지에서 전체 확인.")
        if st.button("⭐ 쇼트리스트 전체 보기", key="home_goto_sl"):
            st.session_state["_nav_target"] = "⭐ 나의 쇼트리스트"
            st.rerun()
    else:
        st.info("쇼트리스트가 비어 있습니다. 선수 즉시 분석 페이지에서 ⭐ 버튼을 눌러 추가하세요.")

    st.markdown("---")

    # ── 하락 주의 선수 (WAR 상위 내 하락 위험 상위) ─────────────────────────────
    st.markdown("### ⚠️ 하락 주의 선수 Top 5 (S6 하락 감지)")
    st.caption("WAR Top 50 내에서 하락 위험 확률이 높은 선수 — 계약 연장 또는 판매 검토 대상")
    if not decline.empty and "decline_prob_ensemble" in decline.columns:
        # 시즌별 중복 제거: player_key 기준 최신 시즌만 유지
        _key_col = "player_key" if "player_key" in decline.columns else "player"
        if "season_year" in decline.columns:
            dec_dedup = decline.sort_values("season_year", ascending=False).drop_duplicates(subset=[_key_col])
        else:
            dec_dedup = decline.drop_duplicates(subset=[_key_col])

        if not latest.empty:
            # latest도 player 기준 중복 제거 (동일 선수 다른 팀 케이스) — WAR 높은 기록 유지
            latest_dedup = latest.sort_values("war", ascending=False).drop_duplicates(subset=["player"])
            dec_merged = dec_dedup.merge(
                latest_dedup[["player", "team", "war", "pos_group"]].rename(columns={"war": "war_cur"}),
                left_on=_key_col,
                right_on="player",
                how="inner"
            )
            # merge 후에도 혹시 남은 중복을 완전히 제거 (하락확률 높은 기록 유지)
            dec_merged = dec_merged.sort_values("decline_prob_ensemble", ascending=False).drop_duplicates(subset=["player"])
        else:
            dec_merged = pd.DataFrame()

        if not dec_merged.empty and "war_cur" in dec_merged.columns:
            top_war_dec = dec_merged.nlargest(50, "war_cur")
            # 하락확률 높은 순 정렬 후 선수 이름 중복 제거
            top_all = top_war_dec.sort_values("decline_prob_ensemble", ascending=False)
            seen_players: set = set()
            unique_rows = []
            for _, row in top_all.iterrows():
                p = str(row.get(_key_col, row.get("player", "")))
                if p not in seen_players:
                    seen_players.add(p)
                    unique_rows.append(row)
                if len(unique_rows) >= 5:
                    break
            for rank, row in enumerate(unique_rows, 1):
                player = str(row.get(_key_col, row.get("player", "")))
                team = row.get("team", "")
                prob = row.get("decline_prob_ensemble", None)
                _mini_player_card(
                    player=player,
                    team=str(team) if pd.notna(team) else "",
                    label="하락확률",
                    value=f"{prob:.0%}" if (prob is not None and not pd.isna(prob)) else "-",
                    color=EPL_MAGENTA,
                    key=f"home_dec_{rank}",
                )
        else:
            st.info("WAR Top 50과 하락 예측 데이터 교차 분석 결과가 없습니다.")
    else:
        st.info("하락 예측 데이터 없음 (decline_predictions 파일 확인)")

    # ── 빠른 이동 버튼 ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🚀 빠른 이동")
    nav_cols = st.columns(4)
    nav_items = [
        ("🔍 선수 즉시 분석", "🔍 선수 즉시 분석"),
        ("💎 저평가 탐색기", "💎 S2 저평가 탐색기"),
        ("🏟️ 팀 프로파일", "🏟️ 팀 프로파일"),
        ("📊 선수 비교", "선수 비교"),
    ]
    for i, (label, page_key) in enumerate(nav_items):
        with nav_cols[i]:
            if st.button(label, use_container_width=True, key=f"home_nav_{i}"):
                st.session_state["nav_menu"] = page_key
                st.rerun()

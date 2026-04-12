"""선수 즉시 분석 리포트 - 이름 검색 한 번으로 전 모델 통합 스카우팅 보고서.

김태현 스카우트가 특정 선수에 대해 빠르게 의사결정할 수 있도록
S1~S6, P6, P7 모델 결과를 하나의 페이지에 종합.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config.settings import SHORTLIST_PATH
from dashboard.components.data_loader import (
    load_scout_ratings,
    load_decline_predictions,
    load_growth_predictions,
    load_growth_predictions_v4,
    load_transfer_predictions,
    load_similarity_matrix,
    load_clusters,
    load_player_profiles,
    load_s2_transfer_targets,
    load_undervalued,
    load_overvalued,
)

# ── 아이콘 / 색상 상수 ──────────────────────────────────────────────────
RISK_ICON = {"high": "🔴", "medium": "🟡", "low": "🟢"}
GROWTH_ICON = {"Improving": "🟢", "Stable": "🟡", "Declining": "🔴"}
EPL_PURPLE = "#37003c"
EPL_MAGENTA = "#e90052"
EPL_GREEN = "#00ff87"

logger = logging.getLogger(__name__)


def _save_shortlist(shortlist: dict) -> None:
    """쇼트리스트를 shortlist.json 에 저장."""
    try:
        path = Path(SHORTLIST_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(shortlist, f, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.error(f"쇼트리스트 파일 저장 실패: {e}")


# ── PDF 생성 ─────────────────────────────────────────────────────────────

def _generate_scout_pdf(data: dict) -> bytes:
    """선수 스카우트 리포트 PDF 생성 (fpdf2 사용).

    Args:
        data: 선수 정보 및 모델 결과 dict

    Returns:
        PDF 바이트 데이터
    """
    from fpdf import FPDF

    # 한국어 → 영어 변환 (Helvetica는 Latin-1만 지원)
    _kor_map = {
        "영입 강력 권고": "STRONGLY RECOMMENDED",
        "영입 긍정 검토": "POSITIVE CONSIDERATION",
        "추가 검토 필요": "FURTHER REVIEW NEEDED",
        "데이터 부족": "INSUFFICIENT DATA",
        "PIS 평가": "PIS Score",
        "S2 가치 평가": "S2 Value",
        "하락 안정성": "Decline Stability",
        "성장 전망": "Growth Outlook",
        "🔴 즉시": "[URGENT]",
        "🟡 모니터링": "[MONITOR]",
        "🟢 장기": "[LONG-TERM]",
        "하락확률": "Decline Prob",
        "가치비율": "Value Ratio",
        "내년 예측": "Next Yr Pred",
        "PIS ": "PIS ",
    }

    def _to_latin(text: str) -> str:
        """한국어 텍스트를 영어로 변환 (Latin-1 안전 처리)."""
        if not text:
            return text
        for kor, eng in _kor_map.items():
            text = text.replace(kor, eng)
        # 변환 후 남은 비 ASCII 문자 제거
        return text.encode("latin-1", errors="replace").decode("latin-1")

    class ScoutPDF(FPDF):
        """EPL 스카우트 리포트 PDF 레이아웃."""

        def header(self):
            # EPL 보라색 배경 헤더
            self.set_fill_color(55, 0, 60)
            self.rect(0, 0, 210, 22, "F")
            self.set_text_color(233, 0, 82)
            self.set_font("Helvetica", "B", 14)
            self.set_xy(10, 5)
            self.cell(0, 12, "EPL SCOUT REPORT", align="L")
            self.set_text_color(0, 255, 135)
            self.set_font("Helvetica", "", 9)
            self.set_xy(0, 5)
            self.cell(200, 12, "Powered by EPL Analytics", align="R")
            self.ln(18)

        def footer(self):
            self.set_y(-12)
            self.set_text_color(150, 150, 150)
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 8, f"EPL Scouting Dashboard  |  Page {self.page_no()}", align="C")

    pdf = ScoutPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── 생성일 / 시즌 ────────────────────────────────────────────────
    import datetime as _dt
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 6, f"Generated: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Season: {data.get('season', '')}", ln=True)
    pdf.ln(2)

    # ── 선수 기본 정보 박스 ──────────────────────────────────────────
    pdf.set_fill_color(26, 26, 46)
    pdf.set_draw_color(55, 0, 60)
    pdf.rect(10, pdf.get_y(), 190, 36, "FD")

    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(14, pdf.get_y() + 4)
    pdf.cell(0, 10, data.get("player", ""), ln=True)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(200, 200, 200)
    pdf.set_x(14)
    pdf.cell(0, 6, f"{data.get('team', '')}  |  {data.get('pos', '')}  |  Age {data.get('age', '')}", ln=True)

    pdf.set_x(14)
    pdf.set_text_color(0, 255, 135)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, f"Market Value: {data.get('market_value', 'N/A')}", ln=True)

    pdf.ln(10)

    # ── 헬퍼: 섹션 제목 ─────────────────────────────────────────────
    def section_title(title: str):
        pdf.set_fill_color(55, 0, 60)
        pdf.set_text_color(233, 0, 82)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_x(10)
        pdf.cell(190, 8, f"  {title}", fill=True, ln=True)
        pdf.ln(2)

    # ── 헬퍼: 2열 지표 행 ──────────────────────────────────────────
    def metric_row(label1: str, val1: str, label2: str = "", val2: str = ""):
        pdf.set_x(14)
        pdf.set_text_color(150, 150, 150)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(45, 7, label1)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(50, 7, val1)
        if label2:
            pdf.set_text_color(150, 150, 150)
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(45, 7, label2)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(50, 7, val2)
        pdf.ln()

    # ── S1 PIS 평가 ─────────────────────────────────────────────────
    section_title("S1  PIS Assessment (Player Impact Score)")
    metric_row("WAR (Percentile)", data.get("war", "N/A"), "Tier", data.get("tier", "N/A"))
    metric_row("Goals/90", data.get("goals_p90", "N/A"), "Assists/90", data.get("assists_p90", "N/A"))
    metric_row("Tackles/90", data.get("tackles_p90", "N/A"), "", "")
    pdf.ln(3)

    # ── S2 시장가치 평가 ─────────────────────────────────────────────
    section_title("S2  Market Value Assessment")
    s2_status_map = {
        "undervalued_strong": "STRONGLY UNDERVALUED - Immediate signing recommended",
        "undervalued_soft":   "UNDERVALUED CANDIDATE - Cross-check with other metrics",
        "overvalued":         "OVERVALUED - Use as negotiation leverage",
        "neutral":            "Neutral - Not in undervalued/overvalued list",
    }
    s2_label = s2_status_map.get(data.get("s2_status", "neutral"), "N/A")
    metric_row("Status", s2_label[:45] if len(s2_label) > 45 else s2_label, "", "")
    metric_row("Estimated Fair Value", data.get("s2_pred_mv", "N/A"), "Value Ratio", data.get("s2_ratio", "N/A"))
    pdf.ln(3)

    # ── S6 하락 위험 ─────────────────────────────────────────────────
    section_title("S6  Decline Risk Detection")
    decline_prob = data.get("decline_prob", "N/A")
    metric_row("Decline Probability", decline_prob, "Growth Classification (P7)", data.get("growth_class", "N/A"))
    # 위험 판정 텍스트
    pdf.set_x(14)
    try:
        prob_f = float(decline_prob.replace("%", "")) / 100
        if prob_f >= 0.7:
            risk_text = "HIGH RISK: Short-term contract + performance clauses recommended"
            pdf.set_text_color(233, 0, 82)
        elif prob_f >= 0.5:
            risk_text = "MEDIUM RISK: Include performance monitoring clauses"
            pdf.set_text_color(255, 215, 0)
        else:
            risk_text = "LOW RISK: Stable. Long-term contract feasible"
            pdf.set_text_color(0, 255, 135)
    except (ValueError, AttributeError):
        risk_text = "Risk level: N/A"
        pdf.set_text_color(200, 200, 200)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 7, risk_text, ln=True)
    pdf.ln(3)

    # ── 종합 판정 ────────────────────────────────────────────────────
    section_title("Overall Scout Verdict")
    pdf.set_x(14)
    verdict_raw = data.get("verdict", "N/A")
    verdict_en = _to_latin(verdict_raw)
    overall = data.get("overall_score", "N/A")
    if "강력" in verdict_raw or "STRONGLY" in verdict_en:
        pdf.set_text_color(0, 255, 135)
    elif "긍정" in verdict_raw or "POSITIVE" in verdict_en:
        pdf.set_text_color(255, 215, 0)
    else:
        pdf.set_text_color(233, 0, 82)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, f"{verdict_en}  ({overall})", ln=True)
    pdf.ln(2)

    # 항목별 점수
    score_items = data.get("score_items", [])
    for label, score, note in score_items:
        label_en = _to_latin(str(label))
        note_en = _to_latin(str(note))
        pdf.set_x(14)
        bar_w = int(score * 120)
        y_bar = pdf.get_y() + 1
        # 배경 바
        pdf.set_fill_color(50, 50, 70)
        pdf.rect(60, y_bar, 120, 4, "F")
        # 점수 바
        color = (0, 200, 100) if score >= 0.7 else ((255, 200, 0) if score >= 0.5 else (200, 0, 60))
        pdf.set_fill_color(*color)
        if bar_w > 0:
            pdf.rect(60, y_bar, bar_w, 4, "F")
        # 레이블
        pdf.set_text_color(180, 180, 180)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(44, 6, label_en)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(120, 6, "")
        pdf.set_text_color(200, 200, 200)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(0, 6, f"  {note_en}", ln=True)
    pdf.ln(4)

    # ── 쇼트리스트 메모 (있을 때만) ─────────────────────────────────
    note = _to_latin(data.get("shortlist_note", ""))
    priority = _to_latin(data.get("shortlist_priority", ""))
    if note or priority:
        section_title("Scout Notes")
        if priority:
            metric_row("Priority", priority, "", "")
        if note:
            pdf.set_x(14)
            pdf.set_text_color(220, 220, 220)
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(182, 6, f"Memo: {note}")
        pdf.ln(2)

    # ── 면책 문구 ────────────────────────────────────────────────────
    pdf.set_text_color(100, 100, 100)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_x(10)
    pdf.multi_cell(
        190, 5,
        "This report is generated by the EPL Analytics Scouting Dashboard for internal use only. "
        "Model predictions are based on historical data and should be used as supporting evidence, "
        "not as the sole basis for transfer decisions.",
    )

    return bytes(pdf.output())


# ── 헬퍼 ────────────────────────────────────────────────────────────────
def _fuzzy_search(query: str, candidates: list) -> list:
    """대소문자 무시 부분 문자열 검색."""
    q = query.strip().lower()
    if not q:
        return []
    return [c for c in candidates if q in c.lower()]


def _safe_val(series_or_val, fmt: str = ".2f") -> str:
    """None/NaN-safe 포맷 출력."""
    try:
        v = float(series_or_val)
        if pd.isna(v):
            return "N/A"
        return f"{v:{fmt}}"
    except (TypeError, ValueError):
        return "N/A"


def _get_player_row(df: pd.DataFrame, player: str, season: Optional[str] = None) -> Optional[pd.Series]:
    """선수명(+ 선택적 시즌)으로 DataFrame에서 단일 행 반환."""
    if df.empty or "player" not in df.columns:
        if "player_key" in df.columns:
            rows = df[df["player_key"] == player]
        else:
            return None
    else:
        rows = df[df["player"] == player]

    if season and "season" in df.columns:
        rows = rows[rows["season"] == season]
    if rows.empty:
        return None
    return rows.iloc[0]


# ── 메인 렌더 ────────────────────────────────────────────────────────────
def render():
    st.title("🔍 선수 즉시 분석 리포트")
    st.caption(
        "이름 직접 검색 또는 시즌 → 팀 → 선수 드롭다운으로 선수를 선택하면 PIS, 시장가치, 성장 궤적, 하락 위험, 이적 적응도까지 "
        "전 모델 결과를 한 페이지에서 확인합니다."
    )

    # ── 데이터 로딩 ─────────────────────────────────────────────────────
    ratings        = load_scout_ratings()
    decline        = load_decline_predictions()
    growth         = load_growth_predictions()
    growth_v4      = load_growth_predictions_v4()
    transfers      = load_transfer_predictions()
    sim_matrix     = load_similarity_matrix()
    clusters       = load_clusters()
    profiles       = load_player_profiles()
    s2_targets     = load_s2_transfer_targets()
    s2_undervalued = load_undervalued()
    s2_overvalued  = load_overvalued()

    if ratings.empty:
        st.warning("scout_ratings_v3.parquet 데이터가 없습니다.")
        return

    # ── 전체 선수 목록 (텍스트 검색용) ──────────────────────────────────
    all_players = sorted(ratings["player"].dropna().unique().tolist()) if "player" in ratings.columns else []

    # ── 타 페이지에서 넘어온 선수 자동 선택 ─────────────────────────────
    _nav_player = st.session_state.pop("scout_report_player", None)
    if _nav_player and _nav_player in all_players:
        st.session_state["report_search_query"] = _nav_player

    # ── 선택 UI: 텍스트 검색 OR 드롭다운 ────────────────────────────────
    search_query = st.text_input(
        "🔎 선수 이름 직접 검색 (선택사항)",
        placeholder="예: Muniz, Fernandes, Saka",
        key="report_search_query",
    )

    selected_player: Optional[str] = None
    selected_season: Optional[str] = None

    if search_query.strip():
        # ── 텍스트 검색 경로 ──────────────────────────────────────────
        matched = _fuzzy_search(search_query, all_players)
        if not matched:
            st.warning(f'"{search_query}"와 일치하는 선수가 없습니다. 다른 이름을 입력하거나 드롭다운을 사용하세요.')
            return
        search_player = st.selectbox("검색 결과에서 선수 선택", matched, key="report_search_result")
        # 해당 선수의 가장 최신 시즌을 자동 선택
        if "season" in ratings.columns:
            player_seasons = sorted(
                ratings[ratings["player"] == search_player]["season"].dropna().unique().tolist(),
                reverse=True,
            )
            if player_seasons:
                selected_season = player_seasons[0]
        selected_player = search_player
        st.caption(f"📌 검색 경로 사용 중 — 시즌: **{selected_season}** (해당 선수의 최신 시즌 자동 선택)")
    else:
        # ── 드롭다운 경로 ─────────────────────────────────────────────
        st.markdown('<p style="text-align:center; color:#888; margin:4px 0;">── 또는 ──</p>', unsafe_allow_html=True)
        sel_col1, sel_col2, sel_col3 = st.columns([1, 1, 2])

        # ① 시즌
        with sel_col1:
            seasons = sorted(ratings["season"].unique().tolist(), reverse=True) if "season" in ratings.columns else ["2024/25"]
            selected_season = st.selectbox("① 시즌 선택", seasons, key="report_season")

        season_ratings_dd = ratings[ratings["season"] == selected_season] if "season" in ratings.columns else ratings

        # ② 팀
        with sel_col2:
            teams = ["전체 팀"] + sorted(season_ratings_dd["team"].dropna().unique().tolist()) if "team" in season_ratings_dd.columns else ["전체 팀"]
            selected_team = st.selectbox("② 팀 필터", teams, key="report_team")

        if selected_team != "전체 팀":
            team_ratings_dd = season_ratings_dd[season_ratings_dd["team"] == selected_team]
        else:
            team_ratings_dd = season_ratings_dd

        # ③ 선수
        with sel_col3:
            player_list = sorted(team_ratings_dd["player"].dropna().unique().tolist()) if not team_ratings_dd.empty else []
            if not player_list:
                st.info("선수를 선택하면 분석이 시작됩니다.")
                return
            selected_player = st.selectbox("③ 선수 선택", player_list, key="report_player")

    # ── 선택된 시즌에서 선수 기본 행 ──────────────────────────────────
    if selected_player is None or selected_season is None:
        st.info("선수를 선택하면 분석이 시작됩니다.")
        return

    season_ratings = ratings[ratings["season"] == selected_season] if "season" in ratings.columns else ratings
    player_row = _get_player_row(season_ratings, selected_player)

    if player_row is None:
        st.warning(f"{selected_player}의 {selected_season} 데이터를 찾을 수 없습니다.")
        return

    # ──────────────────────────────────────────────────────────────────
    # 섹션 1: 선수 기본 정보 헤더
    # ──────────────────────────────────────────────────────────────────
    st.markdown("---")

    # ⭐ 쇼트리스트 버튼 + 📊 비교 페이지 연동
    shortlist = st.session_state.setdefault("shortlist", {})
    _is_in_sl = selected_player in shortlist
    sl_btn_col, cmp_btn_col, sl_info_col = st.columns([1, 1, 3])
    with sl_btn_col:
        if _is_in_sl:
            if st.button("⭐ 쇼트리스트 제거", key="sl_remove_btn", use_container_width=True):
                shortlist.pop(selected_player, None)
                _save_shortlist(shortlist)
                st.rerun()
        else:
            if st.button("☆ 쇼트리스트 추가", key="sl_add_btn", use_container_width=True):
                import datetime as _dt
                team = player_row.get("team", "")
                shortlist[selected_player] = {
                    "team": team,
                    "note": "",
                    "priority": "🟡 모니터링",
                    "added": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
                _save_shortlist(shortlist)
                st.success(f"✅ {selected_player} 쇼트리스트 추가 완료")
    with cmp_btn_col:
        if st.button("📊 비교 페이지", key="goto_compare_btn", use_container_width=True):
            st.session_state["compare_preset"] = selected_player
            st.session_state["_goto_compare"] = True
            st.rerun()
    with sl_info_col:
        if _is_in_sl:
            info = shortlist[selected_player]
            st.caption(f"⭐ 쇼트리스트 등록 | {info.get('priority','')} | {info.get('added','')}")
        elif st.session_state.get("compare_preset") == selected_player:
            st.caption(f"📊 비교 페이지로 이동하려면 사이드바 '선수 비교' 클릭")

    h_col1, h_col2, h_col3, h_col4, h_col5 = st.columns(5)
    with h_col1:
        st.metric("선수", selected_player)
    with h_col2:
        st.metric("팀", player_row.get("team", "N/A"))
    with h_col3:
        st.metric("포지션", player_row.get("pos_group", player_row.get("pos", "N/A")))
    with h_col4:
        age_val = player_row.get("age", player_row.get("age_tm", None))
        st.metric("나이", _safe_val(age_val, ".0f") if age_val else "N/A")
    with h_col5:
        mv = player_row.get("market_value", None)
        mv_str = f"€{mv/1_000_000:.1f}M" if mv and not pd.isna(mv) else "N/A"
        st.metric("시장가치", mv_str)

    # ──────────────────────────────────────────────────────────────────
    # 섹션 2: WAR + 핵심 지표 (S1)
    # ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 PIS 평가 (S1)")
    st.caption("PIS는 0~100 백분위(percentile) 스케일입니다. 리그 평균=50, 최상위(살라급)≈99")

    s1_col1, s1_col2, s1_col3, s1_col4, s1_col5, s1_col6 = st.columns(6)
    war_val = player_row.get("war", None)
    tier_val = player_row.get("tier", None)
    with s1_col1:
        st.metric("WAR (백분위)", _safe_val(war_val, ".1f"))
    with s1_col2:
        st.metric("등급", str(tier_val) if tier_val and not pd.isna(tier_val) else "N/A")
    with s1_col3:
        st.metric("골/90분", _safe_val(player_row.get("goals_p90")))
    with s1_col4:
        st.metric("어시스트/90분", _safe_val(player_row.get("assists_p90")))
    with s1_col5:
        st.metric("태클/90분", _safe_val(player_row.get("tackles_p90")))
    with s1_col6:
        st.metric("일관성", _safe_val(player_row.get("consistency")))

    # WAR 리그 백분위 계산
    if not ratings.empty and war_val and not pd.isna(war_val):
        pos_grp = player_row.get("pos_group", None)
        same_pos = season_ratings[season_ratings["pos_group"] == pos_grp]["war"].dropna() if pos_grp else season_ratings["war"].dropna()
        if len(same_pos) > 0:
            pct = (same_pos < war_val).mean() * 100
            st.caption(f"📍 동일 포지션 선수 중 상위 **{100 - pct:.0f}%** | WAR {war_val:.1f} (0~100 백분위 기준, 현재 시즌 내 상대 순위)")

    # WAR 시즌별 추이 차트
    if "season" in ratings.columns:
        war_history = (
            ratings[ratings["player"] == selected_player][["season", "war", "team"]]
            .dropna(subset=["war"])
            .sort_values("season")
        )
        if len(war_history) >= 2:
            fig_war = go.Figure()
            fig_war.add_trace(go.Scatter(
                x=war_history["season"],
                y=war_history["war"],
                mode="lines+markers+text",
                text=[f"{v:.0f}" for v in war_history["war"]],
                textposition="top center",
                line=dict(color=EPL_GREEN, width=2),
                marker=dict(size=8, color=EPL_GREEN),
                hovertemplate="%{x}<br>PIS: %{y:.1f}<extra></extra>",
                name="PIS",
            ))
            # 리그 평균선 (포지션 기준)
            pos_grp = player_row.get("pos_group", None)
            if pos_grp:
                avg_war_by_season = (
                    ratings[ratings["pos_group"] == pos_grp]
                    .groupby("season")["war"].mean()
                    .reset_index()
                    .sort_values("season")
                )
                if not avg_war_by_season.empty:
                    fig_war.add_trace(go.Scatter(
                        x=avg_war_by_season["season"],
                        y=avg_war_by_season["war"],
                        mode="lines",
                        line=dict(color="#888888", width=1, dash="dot"),
                        name=f"{pos_grp} 리그 평균",
                        hovertemplate="%{x}<br>평균 PIS: %{y:.1f}<extra></extra>",
                    ))
            fig_war.update_layout(
                title=dict(text="시즌별 PIS 추이", font=dict(size=13)),
                xaxis=dict(title="시즌", tickangle=-30),
                yaxis=dict(title="PIS (포지션 내 기여 백분위)", range=[0, 100]),
                height=240,
                margin=dict(l=10, r=10, t=36, b=40),
                legend=dict(orientation="h", y=-0.35),
                plot_bgcolor="#1a1a2e",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e0e0"),
            )
            st.plotly_chart(fig_war, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────
    # 섹션 2-A: 포지션 레이더 차트
    # ──────────────────────────────────────────────────────────────────
    pos_grp = player_row.get("pos_group", None)
    _radar_metrics_map = {
        "FW":  [("goals_p90", "골/90분"), ("assists_p90", "어시/90분"), ("shots_p90", "슈팅/90분"),
                ("consistency", "일관성"), ("war", "PIS"), ("minutes_share", "출전 비중")],
        "MID": [("assists_p90", "어시/90분"), ("tackles_p90", "태클/90분"), ("int_p90", "인터셉트/90분"),
                ("consistency", "일관성"), ("war", "PIS"), ("minutes_share", "출전 비중")],
        "DEF": [("tackles_p90", "태클/90분"), ("int_p90", "인터셉트/90분"), ("goals_p90", "골/90분"),
                ("consistency", "일관성"), ("war", "PIS"), ("minutes_share", "출전 비중")],
        "GK":  [("gk_save_pct", "선방률"), ("gk_cs_pct", "클린시트율"), ("gk_ga_p90", "실점/90분(역)"),
                ("consistency", "일관성"), ("war", "PIS"), ("minutes_share", "출전 비중")],
    }
    _radar_metrics = _radar_metrics_map.get(pos_grp, _radar_metrics_map.get("MID"))

    _radar_vals_player, _radar_vals_avg, _radar_labels = [], [], []
    for col, label in _radar_metrics:
        val = player_row.get(col, None)
        if col == "gk_ga_p90":
            # 실점/90분은 역수 스케일 (낮을수록 좋음)
            league_vals = season_ratings[season_ratings["pos_group"] == pos_grp][col].dropna() if pos_grp else season_ratings[col].dropna()
            p_norm = (1 - (val / (league_vals.max() + 1e-9))) * 100 if (val is not None and not pd.isna(val) and len(league_vals) > 0) else 50
            a_norm = 50.0  # 기준선
        else:
            league_vals = season_ratings[season_ratings["pos_group"] == pos_grp][col].dropna() if pos_grp else season_ratings[col].dropna()
            if len(league_vals) > 0 and val is not None and not pd.isna(val):
                p_norm = float((league_vals <= float(val)).mean() * 100) if col != "war" else float(val)
                a_norm = 50.0
            else:
                p_norm = 50.0
                a_norm = 50.0
        _radar_vals_player.append(round(p_norm, 1))
        _radar_vals_avg.append(a_norm)
        _radar_labels.append(label)

    if len(_radar_vals_player) >= 3:
        # 닫힌 다각형을 위해 첫 값 반복
        cats_closed = _radar_labels + [_radar_labels[0]]
        player_closed = _radar_vals_player + [_radar_vals_player[0]]
        avg_closed = _radar_vals_avg + [_radar_vals_avg[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=avg_closed, theta=cats_closed,
            fill="toself", fillcolor="rgba(136,136,136,0.1)",
            line=dict(color="#888", dash="dot", width=1),
            name="포지션 평균",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=player_closed, theta=cats_closed,
            fill="toself", fillcolor="rgba(0,255,135,0.15)",
            line=dict(color=EPL_GREEN, width=2),
            name=selected_player,
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#1a1a2e",
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9, color="#aaa"), gridcolor="#333"),
                angularaxis=dict(tickfont=dict(size=10, color="#fff"), gridcolor="#333"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#fff",
            legend=dict(orientation="h", y=-0.15),
            margin=dict(t=30, b=30, l=40, r=40),
            height=320,
            title=dict(text=f"{pos_grp} 포지션 레이더 (포지션 내 백분위)", font=dict(size=13)),
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption("💡 레이더 차트 값은 동일 포지션 선수 중 백분위(0~100). 포지션 평균선(점선) 대비 강점/약점을 확인하세요.")

    # ──────────────────────────────────────────────────────────────────
    # 섹션 2-B: 시장가치 평가 (S2)
    # ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💰 시장가치 평가 (S2)")

    # 저평가 여부 확인 (transfer_targets → undervalued → 없으면 overvalued 체크)
    _s2_status = None
    _s2_ratio = None
    _s2_pred_mv = None

    if not s2_targets.empty and "player" in s2_targets.columns:
        _t_row = s2_targets[s2_targets["player"] == selected_player]
        if not _t_row.empty:
            _t_row = _t_row.iloc[0]
            _s2_status = "undervalued_soft"
            _s2_ratio = _t_row.get("value_ratio", None)
            _s2_pred_mv = _t_row.get("predicted_value", None)

    if _s2_status is None and not s2_undervalued.empty and "player" in s2_undervalued.columns:
        _u_row = s2_undervalued[s2_undervalued["player"] == selected_player]
        if not _u_row.empty:
            _u_row = _u_row.iloc[0]
            _s2_status = "undervalued_strong"
            _s2_ratio = _u_row.get("value_ratio", None)
            _s2_pred_mv = _u_row.get("predicted_value", None)

    if _s2_status is None and not s2_overvalued.empty and "player" in s2_overvalued.columns:
        _o_row = s2_overvalued[s2_overvalued["player"] == selected_player]
        if not _o_row.empty:
            _o_row = _o_row.iloc[0]
            _s2_status = "overvalued"
            _s2_ratio = _o_row.get("value_ratio", None)
            _s2_pred_mv = _o_row.get("predicted_value", None)

    mv_cur = player_row.get("market_value", None)
    s2c1, s2c2, s2c3 = st.columns(3)
    with s2c1:
        mv_str = f"€{mv_cur/1_000_000:.1f}M" if mv_cur and not pd.isna(mv_cur) else "N/A"
        st.metric("현재 시장가치", mv_str)
    with s2c2:
        pred_str = f"€{_s2_pred_mv/1_000_000:.1f}M" if _s2_pred_mv and not pd.isna(_s2_pred_mv) else "N/A"
        st.metric("S2 예측 적정가", pred_str)
    with s2c3:
        ratio_str = f"{_s2_ratio:.2f}x" if _s2_ratio and not pd.isna(_s2_ratio) else "N/A"
        st.metric("가치 비율", ratio_str)

    if _s2_status == "undervalued_strong":
        st.success(f"🟢 **강력 저평가**: 예측 시장가치가 실제 가치의 {_s2_ratio:.1f}배 (즉시 영입 검토)")
    elif _s2_status == "undervalued_soft":
        st.info(f"💡 **저평가 후보**: S2 확장 기준 (value_ratio {_s2_ratio:.2f}x). 다른 지표와 교차 확인 권장.")
    elif _s2_status == "overvalued":
        st.warning(f"⚠️ **고평가 주의**: 예측 적정가가 현재 몸값의 {_s2_ratio:.1f}x — 협상 시 가격 인하 요구 근거로 활용")
    else:
        st.caption("이 선수는 S2 저평가/고평가 목록에 없습니다. '🎯 PIS 기반 저평가' 탭에서 순위 역전 분석을 확인하세요.")

    # ──────────────────────────────────────────────────────────────────
    # 섹션 3: 하락 위험 (S6)
    # ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚠️ 하락 위험 감지 (S6)")

    decline_row = None
    if not decline.empty:
        # player_key 또는 player 컬럼으로 검색
        key_col = "player_key" if "player_key" in decline.columns else "player"
        decline_filtered = decline[decline[key_col] == selected_player]
        if "season_year" in decline.columns:
            # 최신 시즌 먼저
            decline_filtered = decline_filtered.sort_values("season_year", ascending=False)
        if not decline_filtered.empty:
            decline_row = decline_filtered.iloc[0]

    if decline_row is not None:
        dc1, dc2, dc3, dc4 = st.columns(4)
        prob = decline_row.get("decline_prob_ensemble", None)
        with dc1:
            prob_str = f"{prob:.1%}" if prob and not pd.isna(prob) else "N/A"
            icon = "🔴" if prob and prob >= 0.6 else ("🟡" if prob and prob >= 0.4 else "🟢")
            st.metric(f"{icon} 하락 확률", prob_str)
        with dc2:
            yr = decline_row.get("season_year", None)
            st.metric("분석 시즌", f"{int(yr)}/{int(yr)+1}" if yr and not pd.isna(yr) else "N/A")
        with dc3:
            st.metric("성과 추세", _safe_val(decline_row.get("perf_slope")))
        with dc4:
            st.metric("부상 위험", _safe_val(decline_row.get("injury_proxy"), ".0f"))

        # 스카우트 판정
        if prob is not None and not pd.isna(prob):
            if prob >= 0.7:
                st.error("🚨 **고위험**: 계약 연장 신중 검토. 1년 단기 + 성과 연동 조건 권장.")
            elif prob >= 0.5:
                st.warning("⚠️ **중위험**: 2~3시즌 계약 시 성과 모니터링 조항 삽입 권장.")
            else:
                st.success("✅ **저위험**: 안정적 상태. 장기 계약 가능.")
    else:
        st.info("이 선수의 하락 감지 데이터가 없습니다.")

    # ──────────────────────────────────────────────────────────────────
    # 섹션 4: 성장 궤적 (S4 + P7)
    # ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 성장 궤적 (S4 / P7)")

    gc1, gc2 = st.columns(2)

    with gc1:
        st.markdown("**S4 성장 예측 (향후 3시즌)**")
        growth_row = None
        if not growth.empty:
            g_rows = growth[growth["player"] == selected_player]
            if not g_rows.empty:
                # 최신 시즌 기준
                if "season" in g_rows.columns:
                    g_rows = g_rows.sort_values("season", ascending=False)
                growth_row = g_rows.iloc[0]

        if growth_row is not None:
            gm1, gm2, gm3 = st.columns(3)
            with gm1:
                st.metric("내년 성과 예측", _safe_val(growth_row.get("pred_next1")))
            with gm2:
                st.metric("2년 후 예측", _safe_val(growth_row.get("pred_next2")))
            with gm3:
                st.metric("3년 후 예측", _safe_val(growth_row.get("pred_next3")))

            peak_age = growth_row.get("peak_age", None)
            cur_age = growth_row.get("current_age", None)
            if peak_age and not pd.isna(peak_age):
                age_diff = (float(cur_age) - float(peak_age)) if cur_age and not pd.isna(cur_age) else None
                if age_diff is not None:
                    if age_diff < -2:
                        st.caption(f"📍 전성기까지 **{abs(age_diff):.0f}년** 남음 (전성기: {int(peak_age)}세)")
                    elif age_diff <= 1:
                        st.caption(f"📍 **전성기 구간** 진행 중 (전성기: {int(peak_age)}세)")
                    else:
                        st.caption(f"📍 전성기({int(peak_age)}세) 이후 **{age_diff:.0f}년** 경과")
        else:
            st.info("S4 성장 데이터 없음")

    with gc2:
        st.markdown("**P7 성장 분류 (XGBoost)**")
        v4_row = None
        if not growth_v4.empty:
            v4_rows = growth_v4[growth_v4["player"] == selected_player]
            if not v4_rows.empty:
                v4_row = v4_rows.iloc[0]

        if v4_row is not None:
            pred = v4_row.get("pred_xgb", v4_row.get("pred_ensemble", "N/A"))
            icon = GROWTH_ICON.get(str(pred), "⚪")
            st.markdown(f"**{icon} {pred}**")
            gv1, gv2, gv3 = st.columns(3)
            with gv1:
                st.metric("개선 확률", _safe_val(v4_row.get("prob_improving"), ".1%"))
            with gv2:
                st.metric("유지 확률", _safe_val(v4_row.get("prob_stable"), ".1%"))
            with gv3:
                st.metric("하락 확률", _safe_val(v4_row.get("prob_declining"), ".1%"))
            inj = v4_row.get("injury_risk", None)
            if inj is not None and not (hasattr(inj, '__float__') and pd.isna(inj)):
                # 0/1 이진 또는 low/medium/high 문자열 모두 처리
                try:
                    inj_int = int(float(inj))
                    inj_label = "🔴 고위험" if inj_int >= 1 else "🟢 저위험"
                except (ValueError, TypeError):
                    inj_label = f"{RISK_ICON.get(str(inj), '⚪')} {inj}"
                st.caption(f"부상 위험: {inj_label}")
        else:
            st.info("P7 분류 데이터 없음 (1500분 미만 선수는 제외)")

    # ──────────────────────────────────────────────────────────────────
    # 섹션 5: 이적 리스크 히스토리 (S5)
    # ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔄 이적 적응 리스크 (S5)")

    if not transfers.empty:
        t_rows = transfers[transfers["player"] == selected_player] if "player" in transfers.columns else pd.DataFrame()
        if not t_rows.empty:
            t_show_cols = ["season_new", "team_old", "team_new", "age_at_transfer",
                           "pred_label", "prob_success", "prob_failure"]
            t_available = [c for c in t_show_cols if c in t_rows.columns]
            t_disp = t_rows[t_available].sort_values("season_new", ascending=False) if "season_new" in t_rows.columns else t_rows[t_available]
            if "pred_label" in t_disp.columns:
                t_disp = t_disp.copy()
                t_disp["pred_label"] = t_disp["pred_label"].map(
                    {"success": "✅ 성공", "failure": "❌ 실패", "partial": "🟡 부분"}
                ).fillna(t_disp["pred_label"])
            st.dataframe(t_disp.head(5), use_container_width=True, hide_index=True)

            # 현재 팀과 잠재 이적팀 시뮬레이션
            cur_team = player_row.get("team", None)
            if cur_team:
                st.caption(f"💡 현재 팀: **{cur_team}**. 다른 팀으로 이적 시 S5 이적 리스크를 이적 인텔리전스 탭에서 확인하세요.")
        else:
            st.info("이 선수의 이적 기록이 없습니다 (동일 팀 유지).")
    else:
        st.info("이적 예측 데이터 없음.")

    # ──────────────────────────────────────────────────────────────────
    # 섹션 6: 유사 선수 (S3)
    # ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 👥 유사 선수 (S3)")

    if not sim_matrix.empty:
        sim_rows = sim_matrix[
            (sim_matrix["player"] == selected_player) &
            (sim_matrix["season"] == selected_season)
        ] if "season" in sim_matrix.columns else sim_matrix[sim_matrix["player"] == selected_player]

        if sim_rows.empty and "season" in sim_matrix.columns:
            # 다른 시즌 폴백
            sim_rows = sim_matrix[sim_matrix["player"] == selected_player]

        if not sim_rows.empty:
            # rank 기준 정렬 후 상위 5명
            if "rank" in sim_rows.columns:
                sim_rows = sim_rows.sort_values("rank")
            sim_top = sim_rows.head(5)

            sim_show = ["neighbor", "nbr_season", "cosine_sim"] if "cosine_sim" in sim_rows.columns else ["neighbor"]
            sim_disp = sim_top[sim_show].copy()
            if "cosine_sim" in sim_disp.columns:
                sim_disp["cosine_sim"] = sim_disp["cosine_sim"].apply(lambda x: f"{x:.3f}")

            # 아키타입 + WAR 병합
            if not clusters.empty and "player" in clusters.columns:
                _arch_col = next((c for c in ["archetype_v4", "archetype", "archetype_kor"] if c in clusters.columns), None)
                if _arch_col:
                    _cluster_slim = clusters[["player", _arch_col]].drop_duplicates("player")
                    sim_disp = sim_disp.merge(_cluster_slim, left_on="neighbor", right_on="player", how="left").drop(columns=["player"], errors="ignore")
                    sim_disp = sim_disp.rename(columns={_arch_col: "아키타입"})
            if not ratings.empty and "player" in ratings.columns and "war" in ratings.columns:
                _war_slim = ratings.sort_values("season", ascending=False).drop_duplicates("player")[["player", "war"]] if "season" in ratings.columns else ratings[["player", "war"]].drop_duplicates("player")
                sim_disp = sim_disp.merge(_war_slim, left_on="neighbor", right_on="player", how="left").drop(columns=["player"], errors="ignore")
                sim_disp["war"] = sim_disp["war"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
                sim_disp = sim_disp.rename(columns={"war": "PIS"})

            rename_map = {"neighbor": "유사 선수", "nbr_season": "시즌", "cosine_sim": "유사도"}
            sim_disp = sim_disp.rename(columns={k: v for k, v in rename_map.items() if k in sim_disp.columns})
            st.dataframe(sim_disp, use_container_width=True, hide_index=True)
            st.caption("💡 유사 선수는 영입 대체재 탐색, 협상 레버리지, 임대 대안 검색에 활용하세요. 아키타입·WAR로 실력 비교 가능.")
        else:
            st.info("이 선수의 유사도 데이터가 없습니다.")
    else:
        st.info("유사도 매트릭스 데이터 없음.")

    # ──────────────────────────────────────────────────────────────────
    # 섹션 7: Big6 대전 성과 (scout_player_profiles)
    # ──────────────────────────────────────────────────────────────────
    if not profiles.empty:
        prof_rows = profiles[
            (profiles["player"] == selected_player) &
            (profiles["season"] == selected_season)
        ] if "season" in profiles.columns else profiles[profiles["player"] == selected_player]

        if not prof_rows.empty:
            prof_row = prof_rows.iloc[0]
            b6_val = prof_row.get("big6_contribution_p90", None)
            dep_val = prof_row.get("team_dependency_score", None)
            traj = prof_row.get("career_trajectory", None)

            if any(v is not None for v in [b6_val, dep_val, traj]):
                st.markdown("---")
                st.markdown("### 🏆 심층 프로파일")
                pc1, pc2, pc3 = st.columns(3)
                with pc1:
                    st.metric("빅6 상대 기여/90분", _safe_val(b6_val))
                with pc2:
                    st.metric("팀 의존도", _safe_val(dep_val))
                with pc3:
                    st.metric("커리어 궤적", str(traj) if traj and not pd.isna(traj) else "N/A")

    # ──────────────────────────────────────────────────────────────────
    # 섹션 8: 종합 스카우트 판정
    # ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 종합 스카우트 판정")

    # 점수 계산
    score_items = []
    war_num = player_row.get("war", None)
    if war_num and not pd.isna(war_num):
        war_score = min(war_num / 100.0, 1.0)
        score_items.append(("PIS 평가", war_score, f"WAR {war_num:.1f}"))

    if _s2_ratio and not pd.isna(_s2_ratio):
        # value_ratio 1.0 = 적정가, 1.5 = 33% 저평가 = good, 0.5 = overvalued = bad
        s2_score = min(max((_s2_ratio - 0.5) / 1.0, 0.0), 1.0)
        score_items.append(("S2 가치 평가", s2_score, f"가치비율 {_s2_ratio:.2f}x"))

    if decline_row is not None:
        prob = decline_row.get("decline_prob_ensemble", None)
        if prob and not pd.isna(prob):
            risk_score = 1.0 - prob
            score_items.append(("하락 안정성", risk_score, f"하락확률 {prob:.1%}"))

    if growth_row is not None:
        n1 = growth_row.get("pred_next1", None)
        if n1 and not pd.isna(n1):
            g_score = max(0, min((float(n1) + 1) / 2.0, 1.0))
            score_items.append(("성장 전망", g_score, f"내년 예측 {float(n1):.2f}"))

    if score_items:
        overall = sum(s for _, s, _ in score_items) / len(score_items)
        verdict = "영입 강력 권고" if overall >= 0.7 else ("영입 긍정 검토" if overall >= 0.5 else "추가 검토 필요")
        verdict_color = EPL_GREEN if overall >= 0.7 else ("#FFD700" if overall >= 0.5 else EPL_MAGENTA)

        st.markdown(
            f"""<div style='background:{EPL_PURPLE};padding:16px;border-radius:8px;
            border-left:6px solid {verdict_color};margin-bottom:8px;'>
            <span style='display:block;color:#ffffff;-webkit-text-fill-color:#ffffff;
            font-size:1.4em;font-weight:700;margin:0 0 6px 0;
            text-shadow:0 0 10px {verdict_color};'>{verdict}</span>
            <span style='display:block;color:#cccccc;-webkit-text-fill-color:#cccccc;font-size:0.9em;margin:0;'>
            종합 점수: {overall:.0%} | 평가 항목: {", ".join(label for label, _, _ in score_items)}
            </span></div>""",
            unsafe_allow_html=True,
        )

        # 항목별 막대 차트
        fig = go.Figure(go.Bar(
            x=[s * 100 for _, s, _ in score_items],
            y=[label for label, _, _ in score_items],
            orientation="h",
            marker_color=[EPL_GREEN if s >= 0.7 else ("#FFD700" if s >= 0.5 else EPL_MAGENTA) for _, s, _ in score_items],
            text=[note for _, _, note in score_items],
            textposition="outside",
        ))
        fig.update_layout(
            xaxis=dict(range=[0, 110], title="점수 (100=최고)"),
            height=200 + len(score_items) * 30,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("종합 판정을 위한 데이터가 충분하지 않습니다.")

    # ──────────────────────────────────────────────────────────────────
    # 섹션 9: PDF 내보내기
    # ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📄 PDF 리포트 내보내기")
    st.caption("선수 기본정보, S1 PIS, S2 가치 평가, S6 하락 위험, 종합 판정을 PDF로 저장합니다.")

    # PDF 생성에 필요한 데이터 구성
    _pdf_data = {
        "player": selected_player,
        "season": selected_season,
        "team": str(player_row.get("team", "N/A")),
        "pos": str(player_row.get("pos_group", player_row.get("pos", "N/A"))),
        "age": _safe_val(player_row.get("age", player_row.get("age_tm", None)), ".0f"),
        "market_value": mv_str,
        "war": _safe_val(war_val, ".1f"),
        "tier": str(tier_val) if tier_val and not pd.isna(tier_val) else "N/A",
        "goals_p90": _safe_val(player_row.get("goals_p90")),
        "assists_p90": _safe_val(player_row.get("assists_p90")),
        "tackles_p90": _safe_val(player_row.get("tackles_p90")),
        "s2_status": _s2_status or "neutral",
        "s2_ratio": f"{_s2_ratio:.2f}x" if _s2_ratio and not pd.isna(_s2_ratio) else "N/A",
        "s2_pred_mv": f"EUR{_s2_pred_mv/1_000_000:.1f}M" if _s2_pred_mv and not pd.isna(_s2_pred_mv) else "N/A",
        "decline_prob": _safe_val(decline_row.get("decline_prob_ensemble"), ".1%") if decline_row is not None else "N/A",
        "growth_class": str(v4_row.get("pred_xgb", v4_row.get("pred_ensemble", "N/A"))) if v4_row is not None else "N/A",
        "verdict": locals().get("verdict", "데이터 부족") if score_items else "데이터 부족",
        "overall_score": f"{locals().get('overall', 0):.0%}" if score_items else "N/A",
        "score_items": score_items if score_items else [],
        "shortlist_note": shortlist.get(selected_player, {}).get("note", ""),
        "shortlist_priority": shortlist.get(selected_player, {}).get("priority", ""),
    }

    _pdf_bytes = _generate_scout_pdf(_pdf_data)
    st.download_button(
        label="📄 PDF 다운로드",
        data=_pdf_bytes,
        file_name=f"scout_report_{selected_player.replace(' ', '_')}_{selected_season.replace('/', '-')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

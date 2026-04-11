"""EPL Scout Intelligence Dashboard

스카우트/이적 담당자 전용 분석 대시보드.
S1~S6 모델 통합 + 기존 통계 대시보드.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="EPL 스카우트 인텔리전스",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 쇼트리스트 세션 초기화 ────────────────────────────────────────────
if "shortlist" not in st.session_state:
    st.session_state["shortlist"] = {}  # {player: {"team": ..., "note": ..., "added": ...}}

# Load custom CSS
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# Streamlit MPA 자동 사이드바 네비게이터 숨김 (app.py 라디오 버튼으로 대체)
st.markdown(
    "<style>[data-testid='stSidebarNav']{display:none!important;}</style>",
    unsafe_allow_html=True,
)

# Navigation
st.sidebar.title("⚽ EPL 스카우트 허브")
st.sidebar.markdown("---")

st.sidebar.markdown("**스카우트 도구**")

MENU_OPTIONS = [
    "🏠 홈",
    "🔍 선수 즉시 분석",
    "⭐ 나의 쇼트리스트",
    "스카우트 개요",
    "선수 분석",
    "이적 인텔리전스",
    "🔄 이적 시나리오",
    "💎 S2 저평가 탐색기",
    "🏟️ 팀 프로파일",
    "선수 통계 순위",
    "시즌 개요",
    "선수 비교",
    "역대 기록",
    "모델 설명 (SHAP)",
]

# 페이지 이동 플래그 처리 (radio 위젯 렌더 전에 반드시 실행)
# _nav_target: 이동할 페이지명 — 각 페이지에서 직접 nav_menu 수정 불가(Streamlit 제약)
if st.session_state.get("_nav_target"):
    st.session_state["nav_menu"] = st.session_state.pop("_nav_target")
# 하위 호환: 비교 페이지 레거시 플래그
if st.session_state.get("_goto_compare"):
    st.session_state["_goto_compare"] = False
    st.session_state["nav_menu"] = "선수 비교"

page = st.sidebar.radio(
    "메뉴",
    MENU_OPTIONS,
    key="nav_menu",
    label_visibility="collapsed",
)

# 사이드바 쇼트리스트 카운터
shortlist = st.session_state.get("shortlist", {})
if shortlist:
    st.sidebar.markdown("---")
    st.sidebar.caption(f"⭐ 쇼트리스트 **{len(shortlist)}명**")
    for p in list(shortlist.keys())[:5]:
        st.sidebar.caption(f"  · {p}")

if page == "🏠 홈":
    from dashboard.pages.home import render
    render()
elif page == "🔍 선수 즉시 분석":
    from dashboard.pages.scout_report import render
    render()
elif page == "⭐ 나의 쇼트리스트":
    from dashboard.pages.shortlist import render
    render()
elif page == "스카우트 개요":
    from dashboard.pages.scout_overview import render
    render()
elif page == "선수 분석":
    from dashboard.pages.scout_players import render
    render()
elif page == "이적 인텔리전스":
    from dashboard.pages.scout_transfer import render
    render()
elif page == "🔄 이적 시나리오":
    # 이적 시나리오 시뮬레이터 탭(탭4)으로 직접 이동
    from dashboard.pages.scout_transfer import render
    render()
elif page == "💎 S2 저평가 탐색기":
    from dashboard.pages.s2_explorer import render
    render()
elif page == "🏟️ 팀 프로파일":
    from dashboard.pages.team_profiles import render
    render()
elif page == "선수 통계 순위":
    from dashboard.pages.player_rankings import render
    render()
elif page == "시즌 개요":
    from dashboard.pages.season_overview import render
    render()
elif page == "선수 비교":
    from dashboard.pages.comparison import render
    render()
elif page == "역대 기록":
    from dashboard.pages.records import render
    render()
elif page == "모델 설명 (SHAP)":
    from dashboard.pages.shap_explainer import render
    render()

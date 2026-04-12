"""나의 쇼트리스트 - 관심 선수 저장 및 비교 관리.

스카우트가 탐색 중 발견한 선수를 저장하고
우선순위와 메모를 관리하는 개인 워크스페이스.

브라우저를 닫아도 data/scout/shortlist.json 에 영구 저장됩니다.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from config.settings import SHORTLIST_PATH
from dashboard.components.data_loader import load_scout_ratings, load_undervalued, load_growth_predictions_v4
from dashboard.utils.image_utils import get_player_image_b64

logger = logging.getLogger(__name__)

EPL_PURPLE  = "#37003c"
EPL_MAGENTA = "#e90052"
EPL_GREEN   = "#00ff87"
GROWTH_ICON = {"Improving": "🟢", "Stable": "🟡", "Declining": "🔴"}
PRIORITY_COLORS = {"🔴 즉시": EPL_MAGENTA, "🟡 모니터링": "#FFD700", "🟢 장기": EPL_GREEN}


# ── 파일 기반 영구 저장 함수 ─────────────────────────────────────────────

def _load_shortlist() -> dict:
    """shortlist.json 에서 쇼트리스트 로드. 파일 없으면 빈 dict 반환."""
    try:
        path = Path(SHORTLIST_PATH)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"쇼트리스트 파일 로드 실패: {e}")
    return {}


def _save_shortlist(shortlist: dict) -> None:
    """쇼트리스트를 shortlist.json 에 저장."""
    try:
        path = Path(SHORTLIST_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(shortlist, f, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.error(f"쇼트리스트 파일 저장 실패: {e}")
        st.warning(f"쇼트리스트 저장 실패: {e}")


def _init_shortlist_from_file() -> None:
    """앱 시작 시 파일 → session_state 동기화 (최초 1회)."""
    if "_shortlist_loaded" not in st.session_state:
        st.session_state["shortlist"] = _load_shortlist()
        st.session_state["_shortlist_loaded"] = True


def _add_to_shortlist(player: str, team: str, note: str = "", priority: str = "🟡 모니터링"):
    sl = st.session_state.setdefault("shortlist", {})
    sl[player] = {
        "team": team,
        "note": note,
        "priority": priority,
        "added": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    _save_shortlist(sl)


def _remove_from_shortlist(player: str):
    sl = st.session_state.get("shortlist", {})
    sl.pop(player, None)
    _save_shortlist(sl)


def render():
    st.title("⭐ 나의 쇼트리스트")
    st.caption("관심 선수를 모아두고 우선순위·메모를 관리하세요. 선수 리포트 페이지에서 ⭐ 버튼으로 추가할 수 있습니다.")

    # 파일에서 쇼트리스트 로드 (최초 1회)
    _init_shortlist_from_file()

    shortlist: dict = st.session_state.get("shortlist", {})

    # ── 선수 직접 추가 UI ───────────────────────────────────────────────
    with st.expander("➕ 선수 직접 추가", expanded=not bool(shortlist)):
        ratings = load_scout_ratings()
        all_players = sorted(ratings["player"].dropna().unique().tolist()) if not ratings.empty and "player" in ratings.columns else []

        add_col1, add_col2, add_col3 = st.columns([3, 2, 2])
        with add_col1:
            add_player = st.selectbox("선수 선택", [""] + all_players, key="sl_add_player")
        with add_col2:
            add_priority = st.selectbox("우선순위", ["🔴 즉시", "🟡 모니터링", "🟢 장기"], key="sl_add_priority")
        with add_col3:
            add_note = st.text_input("메모 (선택)", placeholder="예: 계약 2026 만료", key="sl_add_note")

        if st.button("⭐ 쇼트리스트에 추가", use_container_width=True) and add_player:
            team = ""
            if not ratings.empty and "player" in ratings.columns and "team" in ratings.columns:
                rows = ratings[ratings["player"] == add_player]
                if not rows.empty:
                    team = rows.sort_values("season", ascending=False).iloc[0].get("team", "")
            _add_to_shortlist(add_player, team, add_note, add_priority)
            st.success(f"✅ {add_player} 추가 완료")
            st.rerun()

    st.markdown("---")

    if not shortlist:
        st.info("쇼트리스트가 비어 있습니다. 선수 즉시 분석 페이지에서 ⭐ 버튼을 눌러 추가하세요.")
        return

    # ── 현재 쇼트리스트 데이터에 WAR/성장 정보 병합 ──────────────────
    ratings = load_scout_ratings()
    growth_v4 = load_growth_predictions_v4()
    undervalued = load_undervalued()

    rows = []
    for player, info in shortlist.items():
        row = {"선수": player, "팀": info.get("team", ""), "우선순위": info.get("priority", ""), "메모": info.get("note", ""), "추가일": info.get("added", "")}

        # WAR 병합 (최신 시즌)
        if not ratings.empty and "player" in ratings.columns:
            r = ratings[ratings["player"] == player]
            if not r.empty:
                if "season" in r.columns:
                    r = r.sort_values("season", ascending=False)
                rr = r.iloc[0]
                row["PIS"] = rr.get("war", None)
                row["등급"] = rr.get("tier", None)
                mv = rr.get("market_value", None)
                row["시장가치"] = f"€{mv/1_000_000:.1f}M" if mv and not pd.isna(mv) else "-"
                row["나이"] = rr.get("age", rr.get("age_tm", None))
                if row["팀"] == "":
                    row["팀"] = rr.get("team", "")

        # P7 성장 분류
        if not growth_v4.empty and "player" in growth_v4.columns:
            g = growth_v4[growth_v4["player"] == player]
            if not g.empty:
                pred = g.iloc[0].get("pred_xgb", g.iloc[0].get("pred_ensemble", ""))
                row["성장예측"] = f"{GROWTH_ICON.get(pred, '⚪')} {pred}"

        # S2 저평가
        row["S2"] = ""
        if not undervalued.empty and "player" in undervalued.columns:
            if player in undervalued["player"].values:
                row["S2"] = "🟢 저평가"

        rows.append(row)

    sl_df = pd.DataFrame(rows)

    # ── 요약 메트릭 ─────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("전체", len(shortlist))
    m2.metric("🔴 즉시", sum(1 for v in shortlist.values() if v.get("priority") == "🔴 즉시"))
    m3.metric("🟡 모니터링", sum(1 for v in shortlist.values() if v.get("priority") == "🟡 모니터링"))
    m4.metric("🟢 S2 저평가", sl_df["S2"].str.contains("저평가").sum() if "S2" in sl_df.columns else 0)

    # ── 쇼트리스트 카드 목록 ─────────────────────────────────────────────
    st.markdown("### 쇼트리스트")

    # 우선순위 정렬
    priority_order = {"🔴 즉시": 0, "🟡 모니터링": 1, "🟢 장기": 2}
    if "우선순위" in sl_df.columns:
        sl_df["_sort"] = sl_df["우선순위"].map(priority_order).fillna(9)
        sl_df = sl_df.sort_values("_sort").drop(columns=["_sort"])

    for _, srow in sl_df.iterrows():
        player = srow["선수"]
        priority = srow.get("우선순위", "")
        team = srow.get("팀", "")
        war = srow.get("PIS", "-")
        tier = srow.get("등급", "")
        mv = srow.get("시장가치", "-")
        growth = srow.get("성장예측", "")
        s2 = srow.get("S2", "")
        note = srow.get("메모", "")

        priority_color = PRIORITY_COLORS.get(priority, "#888")
        img_b64 = get_player_image_b64(player, size=(56, 56))

        img_html = (
            f'<img src="data:image/jpeg;base64,{img_b64}" '
            f'style="width:52px;height:52px;object-fit:cover;border-radius:50%;'
            f'border:2px solid {priority_color};flex-shrink:0;">'
            if img_b64 else
            '<div style="width:52px;height:52px;border-radius:50%;background:#2a2a4a;'
            'display:flex;align-items:center;justify-content:center;font-size:22px;flex-shrink:0;">👤</div>'
        )
        war_str = f"WAR {war}" if war != "-" else ""
        badges = " ".join(filter(None, [s2, growth]))

        st.markdown(
            f"""<div style='display:flex;align-items:center;gap:14px;
            background:#1a1a2e;border-radius:10px;padding:10px 14px;
            margin-bottom:8px;border-left:4px solid {priority_color};'>
            {img_html}
            <div style='flex:1;min-width:0;'>
              <div style='font-weight:700;color:#fff;font-size:0.95em;'>{priority} {player}</div>
              <div style='color:#aaa;font-size:0.8em;'>{team} {f"· {tier}" if tier else ""}</div>
              <div style='color:#ccc;font-size:0.8em;margin-top:2px;'>
                {war_str}{f" · {mv}" if mv != "-" else ""}{f" · {badges}" if badges else ""}
              </div>
              {f'<div style="color:#888;font-size:0.75em;font-style:italic;">{note}</div>' if note else ""}
            </div>
            </div>""",
            unsafe_allow_html=True,
        )

    # 테이블 뷰 (접기)
    with st.expander("📋 테이블로 보기"):
        show_cols = [c for c in ["선수", "팀", "나이", "PIS", "등급", "시장가치", "성장예측", "S2", "우선순위", "메모"] if c in sl_df.columns]
        st.dataframe(sl_df[show_cols].reset_index(drop=True), use_container_width=True, hide_index=True)

    # ── 개별 선수 관리 ───────────────────────────────────────────────────
    st.markdown("### 선수 관리")
    for player, info in list(shortlist.items()):
        with st.expander(f"{info.get('priority', '')} {player} — {info.get('team', '')}"):
            ec1, ec2, ec3 = st.columns([2, 2, 1])
            with ec1:
                new_note = st.text_input("메모 수정", value=info.get("note", ""), key=f"sl_note_{player}")
            with ec2:
                new_priority = st.selectbox(
                    "우선순위 변경",
                    ["🔴 즉시", "🟡 모니터링", "🟢 장기"],
                    index=["🔴 즉시", "🟡 모니터링", "🟢 장기"].index(info.get("priority", "🟡 모니터링")),
                    key=f"sl_prio_{player}",
                )
            with ec3:
                st.write("")
                st.write("")
                if st.button("💾 저장", key=f"sl_save_{player}"):
                    shortlist[player]["note"] = new_note
                    shortlist[player]["priority"] = new_priority
                    _save_shortlist(shortlist)
                    st.rerun()

            if st.button(f"🗑️ {player} 제거", key=f"sl_del_{player}", type="secondary"):
                _remove_from_shortlist(player)
                st.rerun()

    # ── 비교 페이지 연동 ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 선수 비교")
    st.caption("쇼트리스트 중 2~4명을 선택해 비교 페이지로 이동합니다.")
    all_sl_players = list(shortlist.keys())
    compare_selected = st.multiselect(
        "비교할 선수 선택 (2~4명)",
        options=all_sl_players,
        default=all_sl_players[:2] if len(all_sl_players) >= 2 else all_sl_players,
        max_selections=4,
        key="sl_compare_select",
    )
    if len(compare_selected) >= 2:
        if st.button("📊 선택 선수 비교 페이지로 이동", use_container_width=True, key="sl_goto_compare"):
            # 첫 번째 선수를 preset으로, 나머지는 세션에 저장
            st.session_state["compare_preset"] = compare_selected[0]
            st.session_state["compare_preset_all"] = compare_selected
            st.session_state["_goto_compare"] = True
            st.rerun()
    else:
        st.info("2명 이상 선택하면 비교 버튼이 활성화됩니다.")

    # ── CSV 다운로드 ──────────────────────────────────────────────────────
    st.markdown("---")
    csv_data = sl_df[show_cols].to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📥 쇼트리스트 CSV 다운로드",
        data=csv_data.encode("utf-8-sig"),
        file_name=f"shortlist_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

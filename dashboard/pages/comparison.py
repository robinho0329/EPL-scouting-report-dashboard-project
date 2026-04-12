"""선수 비교 페이지 - 동일 포지션 선수 최대 4명을 스카우팅 관점에서 비교.

흐름: 시즌 → 팀 → 선수A/B 선택 → 핵심 지표 나란히 비교 → 레이더 차트
"""

import io
from typing import Optional

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# EPL 브랜드 컬러
EPL_PURPLE = "#37003c"
EPL_MAGENTA = "#e90052"
EPL_GREEN = "#00ff87"

from dashboard.components.data_loader import (
    load_player_season_stats,
    load_player_alltime_stats,
    load_scout_ratings,
    get_seasons,
)
from dashboard.utils.image_utils import get_player_image_b64, get_team_logo_b64

# -----------------------------------------------------------------------
# 스카우팅 핵심 지표 (중요도 순)
# scout_ratings_v3 컬럼명 기준으로 정렬
# -----------------------------------------------------------------------
# 1순위: 가치/생산성 지표 (영입 의사결정의 핵심)
# goals_p90 / assists_p90 는 scout_ratings_v3 병합으로 채워짐
SCOUT_PRIMARY = {
    "PIS": "war",
    "시장가치(M£)": "market_value_m",
    "나이": "age",
    "90분당 득점": "goals_p90",
    "90분당 어시스트": "assists_p90",
}

# 2순위: 공격 세부 지표
SCOUT_ATTACK = {
    "공격포인트(G+A)": "g_a",
    "득점": "gls",
    "어시스트": "ast",
    "슈팅/90": "shots_p90",
    "유효슈팅/90": "sot_p90",
    "골 기여율": "goal_contribution_rate",
}

# 3순위: 수비/기여 지표 (scout_ratings_v3 컬럼명)
SCOUT_DEFENSE = {
    "태클/90": "tackles_p90",
    "인터셉트/90": "int_p90",
    "압박 지수": "possession_proxy",
    "일관성": "consistency",
    "출전 비율": "minutes_share",
}

# 레이더 차트용: 1+2순위 병합 (정규화에 부적합한 나이/시장가치 제외)
RADAR_STATS = {
    "PIS": "war",
    "90분당 득점": "goals_p90",
    "90분당 어시스트": "assists_p90",
    "슈팅/90": "shots_p90",
    "공격포인트": "g_a",
    "태클/90": "tackles_p90",
    "인터셉트/90": "int_p90",
    "일관성": "consistency",
}

# 상세 테이블 표시 순서 (스카우팅 우선순위)
TABLE_STATS_ORDERED = {
    **SCOUT_PRIMARY,
    **SCOUT_ATTACK,
    **SCOUT_DEFENSE,
}


def _get_player_row(compare_df: pd.DataFrame, player: str) -> Optional[pd.Series]:
    """선수명으로 단일 행 반환. 없으면 None."""
    data = compare_df[compare_df["player"] == player]
    return data.iloc[0] if not data.empty else None


def _fmt(val, decimals: int = 2) -> str:
    """숫자 포맷 헬퍼."""
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return str(val) if pd.notna(val) else "-"


def _safe_val(val):
    """Excel 저장용 안전한 값 변환 (NaN → None, numpy 타입 → Python 기본 타입)."""
    import numpy as np
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def _build_comparison_excel(compare_df: pd.DataFrame, selected_players: list) -> bytes:
    """비교 결과를 3시트 Excel 바이트로 반환.

    시트 1: 기본정보 (이름/팀/포지션/나이/시장가치)
    시트 2: 주요 스탯 (PIS, goals_p90, assists_p90, xG 계열, 수비 지표)
    시트 3: 모델 점수 (가용한 s-score 컬럼)
    """
    # ── 시트 1: 기본정보 ───────────────────────────────────────────────────
    basic_cols_map = {
        "선수": "player",
        "팀": "team",
        "포지션": "pos_group",
        "나이": "age",
        "시장가치(M£)": "market_value_m",
        "시즌": "season",
    }
    sheet1_rows = []
    for player in selected_players:
        data = compare_df[compare_df["player"] == player]
        if data.empty:
            continue
        row = data.iloc[0]
        entry = {}
        for label, col in basic_cols_map.items():
            entry[label] = _safe_val(row.get(col, None))
        sheet1_rows.append(entry)
    sheet1_df = pd.DataFrame(sheet1_rows)

    # ── 시트 2: 주요 스탯 ──────────────────────────────────────────────────
    stat_cols_map = {
        "선수": "player",
        "PIS": "war",
        "90분당 득점": "goals_p90",
        "90분당 어시스트": "assists_p90",
        "슈팅/90": "shots_p90",
        "유효슈팅/90": "sot_p90",
        "공격포인트(G+A)": "g_a",
        "득점": "gls",
        "어시스트": "ast",
        "태클/90": "tackles_p90",
        "인터셉트/90": "int_p90",
        "압박 지수": "possession_proxy",
        "일관성": "consistency",
        "출전 비율": "minutes_share",
        "골 기여율": "goal_contribution_rate",
    }
    sheet2_rows = []
    for player in selected_players:
        data = compare_df[compare_df["player"] == player]
        if data.empty:
            continue
        row = data.iloc[0]
        entry = {}
        for label, col in stat_cols_map.items():
            entry[label] = _safe_val(row.get(col, None))
        sheet2_rows.append(entry)
    sheet2_df = pd.DataFrame(sheet2_rows)

    # ── 시트 3: 모델 점수 ──────────────────────────────────────────────────
    # 가용한 모델 점수 컬럼 탐색
    model_score_candidates = {
        "선수": "player",
        "WAR (S1)": "war",
        "가치비율 (S2)": "value_ratio",
        "클러스터 (S3)": "cluster_label",
        "성장 궤적 (S4)": "career_trajectory",
        "이적 적응 점수 (S5)": "transfer_adapt_score",
        "하락세 확률 (S6)": "decline_prob",
        "시장가치 효율 (S2파생)": "market_value_efficiency",
    }
    sheet3_rows = []
    for player in selected_players:
        data = compare_df[compare_df["player"] == player]
        if data.empty:
            continue
        row = data.iloc[0]
        entry = {}
        for label, col in model_score_candidates.items():
            entry[label] = _safe_val(row.get(col, None))
        sheet3_rows.append(entry)
    sheet3_df = pd.DataFrame(sheet3_rows)

    # ── Excel 직렬화 ────────────────────────────────────────────────────────
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        sheet1_df.to_excel(writer, sheet_name="기본정보", index=False)
        sheet2_df.to_excel(writer, sheet_name="주요스탯", index=False)
        sheet3_df.to_excel(writer, sheet_name="모델점수", index=False)

        # 컬럼 너비 자동 조정
        for sheet_name, df in [("기본정보", sheet1_df), ("주요스탯", sheet2_df), ("모델점수", sheet3_df)]:
            ws = writer.sheets[sheet_name]
            for col_cells in ws.columns:
                max_len = max(
                    (len(str(cell.value)) if cell.value is not None else 0)
                    for cell in col_cells
                )
                ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 4, 30)

    buffer.seek(0)
    return buffer.read()


def render():
    st.title("선수 비교")
    st.caption(
        "동일 포지션의 두 선수를 선택해서 영입 우선순위를 결정하세요. "
        "흐름: 시즌 선택 → 팀 필터 → 선수 선택 → 핵심 지표 비교"
    )

    season_stats = load_player_season_stats()
    alltime_stats = load_player_alltime_stats()
    scout_ratings = load_scout_ratings()  # WAR, 시장가치, 태클, 인터셉트 등

    if season_stats.empty and alltime_stats.empty:
        st.warning("선수 데이터가 없습니다. 크롤링 파이프라인을 먼저 실행하세요.")
        return

    # ── scout_ratings 컬럼 준비 (비교 지표 보강) ─────────────────────────
    if not scout_ratings.empty:
        # market_value_m 파생
        if "market_value" in scout_ratings.columns:
            scout_ratings = scout_ratings.copy()
            scout_ratings["market_value_m"] = scout_ratings["market_value"] / 1_000_000
        # season_stats에 병합할 컬럼 선택
        _merge_cols = ["player", "season", "war", "market_value_m",
                       "tackles_p90", "int_p90", "consistency", "minutes_share",
                       "possession_proxy", "goals_p90", "assists_p90"]
        _merge_cols = [c for c in _merge_cols if c in scout_ratings.columns]
        _ratings_slim = scout_ratings[_merge_cols].drop_duplicates(
            subset=["player", "season"]
        ) if "season" in scout_ratings.columns else pd.DataFrame()

        if not _ratings_slim.empty and not season_stats.empty:
            season_stats = season_stats.merge(
                _ratings_slim, on=["player", "season"], how="left"
            )
        # shots_p90, sot_p90 컬럼명 통일 (player_features 기준)
        _feat_cols = ["player", "season", "shots_p90", "sot_p90", "goal_contribution_rate"]
        # alltime_stats에는 PIS/시장가치 직접 병합 (시즌 무관, 최근 시즌 기준)
        if not _ratings_slim.empty and not alltime_stats.empty:
            _latest = (
                _ratings_slim.sort_values("season", ascending=False)
                .drop_duplicates("player")
                .drop(columns=["season"])
            )
            alltime_stats = alltime_stats.merge(_latest, on="player", how="left")

    # ── 비교 모드 ──────────────────────────────────────────────────────────
    mode = st.radio("비교 기준", ["시즌 통계", "통산 통계"], horizontal=True)

    if mode == "시즌 통계":
        df = season_stats
        col_s, col_t = st.columns(2)
        with col_s:
            seasons = get_seasons(df)
            selected_season = st.selectbox("① 시즌 선택", seasons)
        df = df[df["season"] == selected_season]

        with col_t:
            if "team" in df.columns:
                teams = sorted(df["team"].dropna().unique().tolist())
                team_options = ["전체 팀"] + teams
                selected_team = st.selectbox("② 팀 필터 (선택)", team_options)
                if selected_team != "전체 팀":
                    df = df[df["team"] == selected_team]
    else:
        df = alltime_stats if not alltime_stats.empty else season_stats

    if df.empty or "player" not in df.columns:
        st.info("선택한 조건에 해당하는 데이터가 없습니다.")
        return

    # ── 선수 선택 (최대 4명) ───────────────────────────────────────────────
    all_players = sorted(df["player"].unique().tolist())

    # 즉시 분석 / 쇼트리스트에서 넘어온 preset 선수 자동 선택
    _preset_all = st.session_state.pop("compare_preset_all", None)
    _preset = st.session_state.pop("compare_preset", None)
    if _preset_all:
        _default = [p for p in _preset_all if p in all_players]
        missing = [p for p in _preset_all if p not in all_players]
        if missing:
            st.info(f"일부 선수({', '.join(missing)})가 선택 조건(시즌/팀)에 없습니다. 시즌을 변경하거나 직접 검색하세요.")
    elif _preset:
        _default = [_preset] if _preset in all_players else []
        if _preset and _preset not in all_players:
            st.info(f"'{_preset}' 선수가 선택 조건(시즌/팀)에 없습니다. 시즌을 변경하거나 직접 검색하세요.")
    else:
        _default = []

    selected_players = st.multiselect(
        "③ 선수 선택 (2~4명, 동일 포지션 권장)",
        all_players,
        default=_default,
        max_selections=4,
    )

    if len(selected_players) < 2:
        st.info("2명 이상 선수를 선택하면 비교가 시작됩니다.")
        return

    compare_df = df[df["player"].isin(selected_players)]
    colors = ["#37003c", "#e90052", "#00ff87", "#04f5ff"]

    # ── 선수 사진 헤더 ──────────────────────────────────────────────────────
    st.markdown("---")
    photo_cols = st.columns(len(selected_players))
    for i, player in enumerate(selected_players):
        row = _get_player_row(compare_df, player)
        team = row.get("team", "") if row is not None else ""
        war = row.get("war", None) if row is not None else None
        pos = row.get("pos_group", "") if row is not None else ""

        with photo_cols[i]:
            img_b64 = get_player_image_b64(player, size=(120, 120))
            logo_b64 = get_team_logo_b64(team, size=(24, 24)) if team else None
            color = colors[i % len(colors)]

            # 사진
            if img_b64:
                st.markdown(
                    f"""<div style='text-align:center;'>
                    <img src='data:image/jpeg;base64,{img_b64}'
                    style='width:110px;height:110px;object-fit:cover;
                    border-radius:50%;border:3px solid {color};
                    display:block;margin:0 auto 8px auto;'>
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""<div style='text-align:center;'>
                    <div style='width:110px;height:110px;border-radius:50%;
                    background:#2a2a4a;border:3px solid {color};
                    display:flex;align-items:center;justify-content:center;
                    font-size:40px;margin:0 auto 8px auto;'>👤</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            # 이름 + 팀 로고
            logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="height:18px;vertical-align:middle;margin-right:4px;">' if logo_b64 else ""
            war_str = f"WAR {war:.1f}" if war is not None and pd.notna(war) else ""
            st.markdown(
                f"""<div style='text-align:center;'>
                <div style='font-weight:700;font-size:1em;color:#fff;'>{player}</div>
                <div style='font-size:0.8em;color:#aaa;margin-top:2px;'>{logo_html}{team}</div>
                <div style='font-size:0.85em;font-weight:600;color:{color};margin-top:2px;'>{war_str}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # ── 섹션 1: 핵심 지표 메트릭 카드 ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 핵심 지표 한눈에 보기")
    st.caption("영입 의사결정에 직결되는 1순위 지표 (PIS · 시장가치 · 나이 · 골/90 · 어시스트/90)")

    primary_available = {
        k: v for k, v in SCOUT_PRIMARY.items() if v in compare_df.columns
    }

    if primary_available:
        metric_cols = st.columns(len(selected_players))
        for col_idx, player in enumerate(selected_players):
            row = _get_player_row(compare_df, player)
            if row is None:
                continue
            with metric_cols[col_idx]:
                team_label = f"({row['team']})" if "team" in row.index and pd.notna(row.get("team")) else ""
                st.markdown(f"**{player}** {team_label}")
                for display_name, col_name in primary_available.items():
                    val = row.get(col_name, None)
                    decimals = 0 if display_name == "나이" else 2
                    st.metric(label=display_name, value=_fmt(val, decimals))

    # ── 섹션 2: 가로 바 차트 (핵심 5개 지표 나란히) ────────────────────────
    st.markdown("---")
    st.markdown("### 핵심 5개 지표 나란히 비교")
    st.caption("동일 기준으로 정규화된 값 (100 = 선택 선수 중 최고). 값이 클수록 우수.")

    # 레이더용 지표 중 데이터 있는 것만 최대 5개
    bar_candidates = {
        k: v for k, v in RADAR_STATS.items()
        if v in compare_df.columns and compare_df[v].notna().any()
    }
    bar_stats = dict(list(bar_candidates.items())[:5])

    if bar_stats:
        bar_fig = go.Figure()
        stat_labels = list(bar_stats.keys())
        stat_cols_bar = list(bar_stats.values())
        max_vals = compare_df[stat_cols_bar].max()

        for i, player in enumerate(selected_players):
            row = _get_player_row(compare_df, player)
            if row is None:
                continue
            norm_vals = []
            for col_name in stat_cols_bar:
                raw = row.get(col_name, 0) or 0
                mx = max_vals.get(col_name, 1)
                norm_vals.append(round(raw / mx * 100, 1) if mx > 0 else 0)

            bar_fig.add_trace(go.Bar(
                name=player,
                x=stat_labels,
                y=norm_vals,
                marker_color=colors[i % len(colors)],
                text=[f"{v:.0f}" for v in norm_vals],
                textposition="outside",
            ))

        bar_fig.update_layout(
            barmode="group",
            yaxis=dict(range=[0, 115], title="상대 점수 (100=최고)"),
            xaxis_title=None,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400,
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(bar_fig, use_container_width=True, theme=None)
    else:
        st.info("바 차트를 그릴 지표 데이터가 없습니다.")

    # ── 섹션 3: 레이더 차트 ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 레이더 비교 (다면 강점/약점)")
    st.caption("공격·수비·창의성을 포함한 다면 비교. 각 축은 선택 선수 기준 정규화.")

    available_radar = {
        k: v for k, v in RADAR_STATS.items()
        if v in compare_df.columns and compare_df[v].notna().any()
    }

    if len(available_radar) >= 3:
        radar_fig = go.Figure()
        categories = list(available_radar.keys())
        stat_cols_radar = list(available_radar.values())
        max_vals_radar = compare_df[stat_cols_radar].max()

        for i, player in enumerate(selected_players):
            row = _get_player_row(compare_df, player)
            if row is None:
                continue
            values = []
            for col_name in stat_cols_radar:
                val = row.get(col_name, 0) or 0
                mx = max_vals_radar.get(col_name, 1)
                values.append(round(val / mx * 100, 1) if mx > 0 else 0)

            values.append(values[0])
            cats = categories + [categories[0]]

            radar_fig.add_trace(go.Scatterpolar(
                r=values,
                theta=cats,
                fill="toself",
                name=player,
                line_color=colors[i % len(colors)],
                opacity=0.6,
            ))

        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            height=500,
        )
        st.plotly_chart(radar_fig, use_container_width=True, theme=None)
    else:
        st.info("레이더 차트를 그리기에 충분한 통계 항목이 없습니다.")

    # ── 섹션 4: 상세 통계 테이블 (스카우팅 우선순위 순) ───────────────────
    st.markdown("---")
    st.markdown("### 상세 통계 (스카우팅 우선순위 순)")
    st.caption("PIS · 시장가치 · 나이 → 공격 지표 → 수비/기여 지표 순 정렬")

    stat_display = []
    available_table = {
        k: v for k, v in TABLE_STATS_ORDERED.items() if v in compare_df.columns
    }

    for player in selected_players:
        row = _get_player_row(compare_df, player)
        if row is None:
            continue
        entry = {"선수": player}
        if "team" in row.index:
            entry["팀"] = row.get("team", "-")
        for display_name, col_name in available_table.items():
            entry[display_name] = row.get(col_name, "-")
        stat_display.append(entry)

    if stat_display:
        detail_df = pd.DataFrame(stat_display).set_index("선수").T
        st.dataframe(detail_df, use_container_width=True)

    # ── 섹션 5: 🏆 종합 추천 판정 ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏆 종합 추천 판정")

    if "war" in compare_df.columns and compare_df["war"].notna().any():
        best_row = compare_df.loc[compare_df["war"].idxmax()]
        best_player = best_row.get("player", "N/A")
        best_war = best_row.get("war", None)
        best_team = best_row.get("team", "")
        best_age = best_row.get("age", None)
        best_mv = best_row.get("market_value_m", best_row.get("market_value", None))

        # 2위와의 PIS 격차
        war_sorted = compare_df.dropna(subset=["war"]).sort_values("war", ascending=False)
        war_gap = (war_sorted.iloc[0]["war"] - war_sorted.iloc[1]["war"]) if len(war_sorted) >= 2 else 0
        gap_label = "압도적 우위" if war_gap >= 10 else ("명확한 우위" if war_gap >= 5 else "근소한 우위")

        # 판정 색상
        verdict_color = colors[0] if selected_players else EPL_MAGENTA

        mv_str = ""
        if best_mv is not None and pd.notna(best_mv):
            mv_val = float(best_mv)
            if mv_val >= 1:
                mv_str = f" | 시장가치 €{mv_val:.1f}M"
            elif mv_val > 0:
                mv_str = f" | 시장가치 €{mv_val*1000:.0f}K"
        age_str = f" | {int(best_age)}세" if best_age is not None and pd.notna(best_age) else ""

        st.markdown(
            f"""<div style='background:{EPL_PURPLE};padding:20px;border-radius:10px;
            border-left:6px solid {verdict_color};'>
            <span style='display:block;color:#ffffff;-webkit-text-fill-color:#ffffff;
            font-size:1.5em;font-weight:700;margin-bottom:8px;'>
            🏆 추천: {best_player}</span>
            <span style='display:block;color:#cccccc;-webkit-text-fill-color:#cccccc;font-size:0.95em;'>
            WAR {best_war:.1f} | {best_team}{age_str}{mv_str}</span>
            <span style='display:block;color:{EPL_GREEN};-webkit-text-fill-color:{EPL_GREEN};
            font-size:0.85em;margin-top:6px;'>
            ▶ 비교 선수 중 WAR 1위 ({gap_label}, 격차 {war_gap:.1f}점)</span>
            </div>""",
            unsafe_allow_html=True,
        )
        st.caption("💡 PIS 기준 자동 판정. 최종 결정은 포지션 필요도·예산·나이를 종합 판단하세요.")

        # 즉시 분석 & 쇼트리스트 버튼
        btn_c1, btn_c2, btn_c3 = st.columns([1, 1, 2])
        with btn_c1:
            if st.button(f"🔍 {best_player} 즉시 분석", key="compare_goto_report", use_container_width=True):
                st.session_state["scout_report_player"] = best_player
                st.session_state["_nav_target"] = "🔍 선수 즉시 분석"
                st.rerun()
        with btn_c2:
            if st.button(f"⭐ 쇼트리스트 추가", key="compare_add_shortlist", use_container_width=True):
                sl = st.session_state.setdefault("shortlist", {})
                sl[best_player] = {
                    "team": best_team,
                    "note": f"비교 분석 추천 (WAR {best_war:.1f})",
                    "priority": "🔴 즉시",
                    "added": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                }
                st.success(f"✅ {best_player} 쇼트리스트에 추가했습니다.")
    else:
        st.info("PIS 데이터가 없어 자동 추천을 생성할 수 없습니다. 상세 통계를 직접 비교하세요.")

    # ── 섹션 6: Excel 내보내기 ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Excel 내보내기")
    st.caption("비교 결과를 3개 시트(기본정보 · 주요스탯 · 모델점수)로 구성된 Excel 파일로 저장합니다.")

    excel_bytes = _build_comparison_excel(compare_df, selected_players)
    player_names_slug = "_vs_".join(p.split()[-1] for p in selected_players[:4])
    filename = f"EPL_비교_{player_names_slug}.xlsx"

    st.download_button(
        label="Excel 다운로드",
        data=excel_bytes,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=False,
    )

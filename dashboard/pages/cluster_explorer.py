"""선수 유형 탐색기 — P5 K=6 클러스터링 기반

P5 모델의 6가지 선수 유형별 탐색 도구.
cluster_assignments_k6.parquet + S1 PIS 데이터 병합.
"""
from __future__ import annotations

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

# 역할 메타데이터는 하드코딩하지 않고 클러스터링 산출물에서 만든다.
# 과거에는 포지션을 무시한 6분류(Elite Attackers 등)를 여기에 박아뒀는데,
# 모델이 역할 체계로 바뀌어도 화면이 따라가지 못했다.
_PALETTE = [EPL_MAGENTA, EPL_CYAN, EPL_GREEN, "#ff6b35", "#9b59b6",
            "#f39c12", "#4C9BE8", "#e74c3c", "#1abc9c", "#95a5a6"]

CLUSTER_INFO: dict[str, dict] = {}
CLUSTER_ORDER: list[str] = []


def _build_cluster_info(df: pd.DataFrame) -> None:
    """산출물의 archetype/archetype_desc로 메타데이터를 채운다."""
    global CLUSTER_INFO, CLUSTER_ORDER
    if df.empty or "cluster_name" not in df.columns:
        return
    names = sorted(df["cluster_name"].dropna().unique())
    descs = {}
    if "archetype_desc" in df.columns:
        descs = (df.dropna(subset=["cluster_name"])
                   .groupby("cluster_name")["archetype_desc"].first().to_dict())
    CLUSTER_INFO = {
        n: {"emoji": "", "color": _PALETTE[i % len(_PALETTE)],
            "desc": descs.get(n, "")}
        for i, n in enumerate(names)
    }
    CLUSTER_ORDER = names


@st.cache_data(ttl=3600)
def _load_cluster_data() -> pd.DataFrame:
    """P5 클러스터 할당 데이터 로드."""
    from pathlib import Path
    # S3 v4 역할 클러스터링 산출물. 구 p5(cluster_assignments_k6)는 포지션을
    # 무시하고 뭉뚱그린 6분류라 "공격형 풀백" 같은 오배정이 났다.
    path = (Path(__file__).resolve().parent.parent.parent
            / "data" / "scout" / "cluster_assignments_v4.parquet")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    # 이 페이지는 cluster_name을 기준으로 쓰므로 이름을 맞춰준다.
    if "archetype" in df.columns:
        df = df.rename(columns={"archetype": "cluster_name"})
    return df


def _merge_pis(cluster_df: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """클러스터 데이터에 PIS(war) 병합."""
    if ratings.empty or "war" not in ratings.columns:
        return cluster_df

    # player + season 기준 merge
    war_cols = ["player", "season", "war"]
    available = [c for c in war_cols if c in ratings.columns]
    if "player" not in available:
        return cluster_df

    war_df = ratings[available].drop_duplicates(subset=["player", "season"] if "season" in available else ["player"])
    merged = cluster_df.merge(war_df, on=[c for c in ["player", "season"] if c in cluster_df.columns and c in war_df.columns], how="left")
    return merged


def render():
    st.title("선수 유형 탐색기")
    st.caption("P5 클러스터링 모델 기반 6가지 선수 유형별 탐색. 원하는 유형의 선수를 즉시 찾아보세요.")

    # 데이터 로드
    cluster_df = _load_cluster_data()
    ratings = load_scout_ratings()

    if cluster_df.empty:
        st.error("클러스터 데이터(data/scout/cluster_assignments_v4.parquet)를 불러올 수 없습니다.")
        return

    # PIS 병합
    cluster_df = _merge_pis(cluster_df, ratings)

    # 역할 메타데이터를 데이터에서 생성 (이름·설명·색)
    _build_cluster_info(cluster_df)
    if not CLUSTER_ORDER:
        st.error("클러스터 유형을 읽지 못했습니다.")
        return

    # session_state 초기화 — 이전에 고른 유형이 새 체계에 없으면 첫 항목으로
    if st.session_state.get("cluster_selected") not in CLUSTER_ORDER:
        st.session_state["cluster_selected"] = CLUSTER_ORDER[0]

    # ── 섹션 1: 역할 카드 ──────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### {len(CLUSTER_ORDER)}가지 선수 유형")
    st.markdown("유형 카드를 클릭해 해당 클러스터 선수를 탐색하세요.")

    # 전체 데이터에서 클러스터별 집계 (최신 시즌)
    if "season" in cluster_df.columns:
        latest_season = cluster_df["season"].max()
        latest_df = cluster_df[cluster_df["season"] == latest_season]
    else:
        latest_df = cluster_df

    # 3열 레이아웃. 역할 수가 6개로 고정이 아니므로 행 수를 데이터에 맞춘다.
    n_rows = -(-len(CLUSTER_ORDER) // 3)   # 올림 나눗셈
    for row_idx in range(n_rows):
        row_clusters = CLUSTER_ORDER[row_idx * 3 : row_idx * 3 + 3]
        cols = st.columns(3)
        for col, cluster_name in zip(cols, row_clusters):
            info = CLUSTER_INFO[cluster_name]
            color = info["color"]
            emoji_label = info["emoji"]

            # 클러스터별 통계
            c_df = latest_df[latest_df["cluster_name"] == cluster_name]
            player_count = len(c_df)

            # 대표 선수 Top 3 (PIS 기준, 없으면 시장가치 기준)
            if "war" in c_df.columns:
                top3_df = c_df.nlargest(3, "war")
            elif "market_value" in c_df.columns:
                top3_df = c_df.nlargest(3, "market_value")
            else:
                top3_df = c_df.head(3)
            top3_names = ", ".join(top3_df["player"].tolist()) if len(top3_df) > 0 else "-"

            is_selected = st.session_state["cluster_selected"] == cluster_name
            border_style = f"3px solid {color}" if is_selected else f"1px solid {color}44"
            bg_color = "#1e1e3a" if is_selected else "#1a1a2e"

            with col:
                st.markdown(
                    f"""<div style='background:{bg_color};border-radius:10px;padding:14px;
                    border:{border_style};margin-bottom:10px;'>
                    <div style='color:{color};font-weight:700;font-size:1.05em;'>
                      [{emoji_label}] {cluster_name}
                    </div>
                    <div style='color:#aaa;font-size:0.8em;margin-top:5px;'>
                      선수 수: <span style='color:#fff;font-weight:600;'>{player_count}명</span>
                    </div>
                    <div style='color:#999;font-size:0.78em;margin-top:3px;'>
                      대표: {top3_names}
                    </div>
                    <div style='color:#bbb;font-size:0.78em;margin-top:5px;'>{info["desc"]}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                if st.button(
                    "선택" if not is_selected else "선택됨",
                    key=f"cluster_btn_{cluster_name}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary",
                ):
                    st.session_state["cluster_selected"] = cluster_name
                    st.rerun()

    # ── 섹션 2: 선택된 클러스터 상세 ────────────────────────────────
    st.markdown("---")
    selected = st.session_state["cluster_selected"]
    sel_info = CLUSTER_INFO[selected]
    sel_color = sel_info["color"]

    st.markdown(
        f"<h3 style='color:{sel_color};'>[{sel_info['emoji']}] {selected} — 상세 탐색</h3>",
        unsafe_allow_html=True,
    )
    st.caption(sel_info["desc"])

    # 시즌 선택
    if "season" in cluster_df.columns:
        seasons = sorted(cluster_df["season"].unique(), reverse=True)
        sel_season = st.selectbox("시즌 선택", options=seasons, index=0, key="cluster_season")
        season_cluster_df = cluster_df[
            (cluster_df["season"] == sel_season) &
            (cluster_df["cluster_name"] == selected)
        ].copy()
    else:
        sel_season = "전체"
        season_cluster_df = cluster_df[cluster_df["cluster_name"] == selected].copy()

    # 특성 요약 — 평균 나이, 평균 시장가치, 포지션 분포
    if len(season_cluster_df) > 0:
        col_meta1, col_meta2, col_meta3 = st.columns(3)

        age_col = "age_used" if "age_used" in season_cluster_df.columns else None
        mv_col = "market_value" if "market_value" in season_cluster_df.columns else None

        with col_meta1:
            if age_col:
                avg_age = season_cluster_df[age_col].mean()
                st.metric("평균 나이", f"{avg_age:.1f}세")
            else:
                st.metric("선수 수", f"{len(season_cluster_df)}명")

        with col_meta2:
            if mv_col:
                avg_mv = season_cluster_df[mv_col].dropna().mean()
                mv_disp = f"€{avg_mv/1_000_000:.1f}M" if not pd.isna(avg_mv) else "-"
                st.metric("평균 시장가치", mv_disp)
            elif "war" in season_cluster_df.columns:
                avg_war = season_cluster_df["war"].mean()
                st.metric("평균 PIS", f"{avg_war:.1f}")

        with col_meta3:
            if "war" in season_cluster_df.columns:
                avg_war = season_cluster_df["war"].mean()
                st.metric("평균 PIS", f"{avg_war:.1f}")
            else:
                st.metric("선수 수", f"{len(season_cluster_df)}명")

        # 포지션 분포 파이차트
        pos_col = "pos" if "pos" in season_cluster_df.columns else ("position" if "position" in season_cluster_df.columns else None)
        if pos_col:
            pos_counts = season_cluster_df[pos_col].value_counts().head(8)
            if len(pos_counts) > 0:
                fig_pie = px.pie(
                    names=pos_counts.index,
                    values=pos_counts.values,
                    color_discrete_sequence=[sel_color, EPL_GREEN, EPL_CYAN, "#ff6b35", "#9b59b6", "#f39c12", "#aaa", "#555"],
                    title="포지션 분포",
                )
                fig_pie.update_layout(
                    paper_bgcolor="#0d0d1a",
                    plot_bgcolor="#1a1a2e",
                    font_color="#fff",
                    margin=dict(t=40, b=10, l=10, r=10),
                    height=280,
                    legend=dict(font=dict(size=11)),
                )
                st.plotly_chart(fig_pie, use_container_width=True, theme=None)

    # 상세 필터
    st.markdown("**탐색 필터**")
    sf1, sf2 = st.columns(2)
    with sf1:
        age_range_detail = st.slider(
            "나이 범위",
            min_value=16, max_value=40,
            value=(16, 40),
            key="cluster_age_detail",
        )
    with sf2:
        min_pis = st.slider(
            "최소 PIS",
            min_value=0, max_value=100,
            value=0,
            key="cluster_min_pis",
        )

    # 필터 적용
    detail_df = season_cluster_df.copy()
    age_col_d = "age_used" if "age_used" in detail_df.columns else None
    if age_col_d and len(detail_df) > 0:
        detail_df = detail_df[
            (detail_df[age_col_d] >= age_range_detail[0]) &
            (detail_df[age_col_d] <= age_range_detail[1])
        ]
    if "war" in detail_df.columns and len(detail_df) > 0:
        detail_df = detail_df[detail_df["war"] >= min_pis]

    # PIS 내림차순 정렬
    if "war" in detail_df.columns and len(detail_df) > 0:
        detail_df = detail_df.sort_values("war", ascending=False)

    st.markdown(f"**{selected} 선수 목록 — {sel_season} ({len(detail_df)}명)**")

    if len(detail_df) == 0:
        st.info("조건에 맞는 선수가 없습니다.")
    else:
        for rank, (_, row) in enumerate(detail_df.head(30).iterrows(), 1):
            player = str(row.get("player", ""))
            team = str(row.get("team", ""))
            pos = str(row.get("pos", row.get("position", "")))
            age_val = row.get(age_col_d, None) if age_col_d else None
            war = row.get("war", None)
            mv = row.get("market_value", None)

            war_str = f"PIS {war:.1f}" if war is not None and not pd.isna(war) else "-"
            age_str = f"{int(age_val)}세" if age_val is not None and not pd.isna(age_val) else ""
            mv_str = f"€{mv/1_000_000:.1f}M" if mv is not None and not pd.isna(mv) else "-"

            col_card, col_btn = st.columns([10, 1])
            with col_card:
                st.markdown(
                    f"""<div style='display:flex;align-items:center;gap:12px;
                    background:#1a1a2e;border-radius:10px;padding:10px 14px;
                    margin-bottom:6px;border-left:4px solid {sel_color};'>
                    <div style='color:{sel_color};font-weight:700;font-size:1.05em;min-width:28px;'>#{rank}</div>
                    <div style='flex:1;min-width:0;'>
                      <div style='font-weight:700;color:#fff;font-size:0.95em;'>{player}</div>
                      <div style='color:#aaa;font-size:0.82em;'>{team} · {pos} · {age_str}</div>
                      <div style='color:#ccc;font-size:0.82em;margin-top:2px;'>
                        {war_str} &nbsp;|&nbsp; 시장가치: <span style='color:{sel_color};font-weight:600;'>{mv_str}</span>
                      </div>
                    </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with col_btn:
                if st.button("분석", key=f"cluster_goto_{selected}_{player}_{rank}", help=f"{player} 즉시 분석"):
                    st.session_state["scout_report_player"] = player
                    st.session_state["_nav_target"] = "선수 즉시 분석"
                    st.rerun()

    # ── 섹션 3: 유형 간 PIS 비교 ────────────────────────────────────
    st.markdown("---")
    st.markdown("### 유형 간 평균 PIS 비교")

    if "war" in cluster_df.columns and "cluster_name" in cluster_df.columns:
        # 전체 데이터 기준 클러스터별 평균 PIS (최신 시즌)
        compare_df_base = latest_df if "season" in cluster_df.columns else cluster_df
        avg_pis_series = (
            compare_df_base.groupby("cluster_name")["war"]
            .mean()
            .reindex(CLUSTER_ORDER)
            .dropna()
            .reset_index()
        )
        avg_pis_series.columns = ["cluster_name", "avg_war"]
        avg_pis_series["color"] = avg_pis_series["cluster_name"].map(
            lambda n: CLUSTER_INFO.get(n, {}).get("color", "#888")
        )

        if len(avg_pis_series) > 0:
            fig_bar = px.bar(
                avg_pis_series,
                x="cluster_name",
                y="avg_war",
                color="cluster_name",
                color_discrete_map={n: CLUSTER_INFO[n]["color"] for n in CLUSTER_ORDER},
                labels={"cluster_name": "선수 유형", "avg_war": "평균 PIS"},
                text="avg_war",
            )
            fig_bar.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_bar.update_layout(
                paper_bgcolor="#0d0d1a",
                plot_bgcolor="#1a1a2e",
                font_color="#fff",
                margin=dict(t=20, b=60, l=20, r=20),
                height=380,
                showlegend=False,
                xaxis=dict(tickangle=-20),
            )
            # 선택된 클러스터 강조
            fig_bar.add_vline(
                x=CLUSTER_ORDER.index(selected) if selected in CLUSTER_ORDER else 0,
                line_dash="dot",
                line_color=sel_color,
                opacity=0.5,
            )
            st.plotly_chart(fig_bar, use_container_width=True, theme=None)
            st.caption(f"최신 시즌({latest_season if 'season' in cluster_df.columns else '전체'}) 기준 클러스터별 평균 PIS 비교.")
    else:
        # war 컬럼 없는 경우 선수 수 비교
        count_df = (
            latest_df.groupby("cluster_name")
            .size()
            .reindex(CLUSTER_ORDER)
            .reset_index(name="count")
        )
        fig_bar = px.bar(
            count_df,
            x="cluster_name",
            y="count",
            color="cluster_name",
            color_discrete_map={n: CLUSTER_INFO[n]["color"] for n in CLUSTER_ORDER},
            labels={"cluster_name": "선수 유형", "count": "선수 수"},
            text="count",
        )
        fig_bar.update_traces(texttemplate="%{text}", textposition="outside")
        fig_bar.update_layout(
            paper_bgcolor="#0d0d1a",
            plot_bgcolor="#1a1a2e",
            font_color="#fff",
            margin=dict(t=20, b=60, l=20, r=20),
            height=360,
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True, theme=None)

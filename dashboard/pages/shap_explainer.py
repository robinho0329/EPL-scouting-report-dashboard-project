"""SHAP 모델 설명 페이지 - S1 WAR 모델 기여 피처 시각화.

"왜 이 선수의 PIS이 이렇게 나왔는가?" 를 감독/구단에 설명할 수 있도록
SHAP summary plot(전체 피처 중요도) + 선수별 waterfall plot(개별 기여도)을 제공한다.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── 경로 상수 ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
_SCOUT_DIR = _ROOT / "data" / "scout"
_MODEL_DIR = _ROOT / "models" / "s1_player_rating"
_PROCESSED_DIR = _ROOT / "data" / "processed"

# ── 피처 한국어 레이블 ────────────────────────────────────────────────────────
FEATURE_LABELS = {
    "age": "나이",
    "market_value": "시장가치(£)",
    "pos_group_enc": "포지션",
    "points": "팀 승점",
    "goal_diff": "팀 득실차",
    "nineties": "출전 90분 수",
    "match_count": "출전 경기 수",
    "season_year": "시즌 연도",
}

FEATURE_NAMES = list(FEATURE_LABELS.keys())
FEATURE_NAMES_KO = [FEATURE_LABELS[f] for f in FEATURE_NAMES]


# ── 데이터/모델 캐시 로더 ─────────────────────────────────────────────────────

@st.cache_resource(show_spinner="S1 PIS 모델 로드 중...")
def _load_model():
    """XGB 모델 + pos 인코더 로드."""
    xgb_path = _MODEL_DIR / "xgb_model.pkl"
    enc_path = _MODEL_DIR / "pos_encoder.pkl"
    if not xgb_path.exists():
        return None, None
    try:
        with open(xgb_path, "rb") as f:
            model = pickle.load(f)
        enc = None
        if enc_path.exists():
            with open(enc_path, "rb") as f:
                enc = pickle.load(f)
        return model, enc
    except (ModuleNotFoundError, ImportError) as e:
        st.warning(f"모델 로드 실패: {e}. xgboost 패키지가 필요합니다.")
        return None, None
    except Exception as e:
        st.warning(f"모델 로드 중 오류: {e}")
        return None, None


@st.cache_data(show_spinner="선수 데이터 로드 중...")
def _load_data() -> pd.DataFrame:
    """scout_ratings_v3 + team_summary 병합 후 XGB 피처 재구성."""
    sr_path = _SCOUT_DIR / "scout_ratings_v3.parquet"
    ts_path = _PROCESSED_DIR / "team_season_summary.parquet"

    if not sr_path.exists():
        return pd.DataFrame()

    sr = pd.read_parquet(sr_path)
    if ts_path.exists():
        ts = pd.read_parquet(ts_path).rename(columns={"Season": "season"})
        ts_slim = ts[["season", "team", "points", "goal_diff"]].drop_duplicates()
        sr = sr.merge(ts_slim, on=["season", "team"], how="left")

    # XGB 피처 재구성
    sr["season_year"] = pd.to_numeric(sr["season"].str[:4], errors="coerce").fillna(2020).astype(int)
    # nineties: WAR 역산 추정 (실제 90s는 scout_ratings에 없어서 war 비례로 추정)
    sr["nineties"] = (sr["war"] / 100.0 * 38.0).clip(0, 38)
    sr["match_count"] = sr.get("games_played", pd.Series(20, index=sr.index)).fillna(20)
    sr["points"] = sr["points"].fillna(50)
    sr["goal_diff"] = sr["goal_diff"].fillna(0)
    sr["market_value"] = sr["market_value"].fillna(0)
    sr["age"] = sr["age"].fillna(25)

    return sr


def _encode_pos(sr: pd.DataFrame, enc) -> pd.Series:
    """pos_group → 정수 인코딩."""
    if enc is None:
        mapping = {"DEF": 0, "FW": 1, "GK": 2, "MID": 3}
        return sr["pos_group"].map(mapping).fillna(0).astype(int)
    pos_map = {cls: i for i, cls in enumerate(enc)}
    return sr["pos_group"].map(pos_map).fillna(0).astype(int)


def _build_X(sr: pd.DataFrame, enc) -> pd.DataFrame:
    """모델 입력 DataFrame 생성."""
    df = sr.copy()
    df["pos_group_enc"] = _encode_pos(df, enc)
    X = df[FEATURE_NAMES].fillna(0)
    return X


@st.cache_data(show_spinner="SHAP 값 계산 중... (최초 1회 수행)")
def _compute_shap(_X_values: np.ndarray, _model_key: str) -> np.ndarray:
    """TreeExplainer로 SHAP 값 계산. 캐시 key는 모델 식별용."""
    import shap
    model, _ = _load_model()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame(_X_values, columns=FEATURE_NAMES))
    return shap_values


# ── 메인 렌더 함수 ────────────────────────────────────────────────────────────

def render():
    st.title("모델 설명 (SHAP)")
    st.caption(
        "S1 WAR 모델이 왜 이 점수를 줬는지 피처 기여도로 설명합니다. "
        "전체 중요도(Summary) + 선수별 기여도(Waterfall)를 제공합니다."
    )

    # ── 모델 로드 ────────────────────────────────────────────────────────────
    model, enc = _load_model()
    if model is None:
        st.error(
            "S1 WAR 모델 파일을 찾을 수 없습니다. "
            f"`{_MODEL_DIR / 'xgb_model.pkl'}` 가 존재하는지 확인하세요."
        )
        return

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    sr = _load_data()
    if sr.empty:
        st.error("선수 데이터를 불러올 수 없습니다. `data/scout/scout_ratings_v3.parquet` 를 확인하세요.")
        return

    X = _build_X(sr, enc)

    # ── 사이드바 필터 ─────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("**SHAP 필터**")
        seasons_all = sorted(sr["season"].dropna().unique().tolist(), reverse=True)
        sel_season = st.selectbox("시즌", seasons_all, key="shap_season")

    sr_f = sr[sr["season"] == sel_season].copy()
    X_f = _build_X(sr_f, enc)

    if sr_f.empty:
        st.info("선택한 시즌 데이터가 없습니다.")
        return

    # ── SHAP 계산 (선택 시즌 기준) ────────────────────────────────────────────
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_f)
        expected_value = float(explainer.expected_value)
    except Exception as e:
        st.error(f"SHAP 계산 오류: {e}")
        return

    # ── 섹션 1: 설명 텍스트 ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### PIS 점수는 어떻게 결정되는가?")
    st.info(
        "S1 WAR(Wins Above Replacement) 모델은 나이, 시장가치, 팀 전력(승점/득실차), "
        "출전 시간, 포지션 등 8개 피처로 선수의 기여도를 0~100 퍼센타일로 환산합니다. "
        "아래 차트는 각 피처가 PIS 점수를 얼마나 올리거나 낮추는지 보여줍니다."
    )

    # ── 섹션 2: 전체 피처 중요도 (Summary Bar) ──────────────────────────────
    st.markdown("---")
    st.markdown(f"### 전체 피처 중요도 — {sel_season} 시즌")
    st.caption("막대 길이 = 해당 피처의 평균 |SHAP| 값 (WAR에 미치는 평균 영향력)")

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    importance_df = pd.DataFrame({
        "피처": FEATURE_NAMES_KO,
        "피처_원본": FEATURE_NAMES,
        "평균_SHAP": mean_abs_shap,
    }).sort_values("평균_SHAP", ascending=True)

    bar_fig = go.Figure(go.Bar(
        x=importance_df["평균_SHAP"],
        y=importance_df["피처"],
        orientation="h",
        marker_color="#00ff87",
        text=[f"{v:.2f}" for v in importance_df["평균_SHAP"]],
        textposition="outside",
    ))
    bar_fig.update_layout(
        xaxis_title="평균 |SHAP| 값",
        yaxis_title=None,
        height=350,
        margin=dict(t=20, b=20, l=10, r=60),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0d0d1a",
        font_color="#ffffff",
        xaxis=dict(gridcolor="#333355"),
    )
    st.plotly_chart(bar_fig, use_container_width=True, theme=None)

    # ── 섹션 3: 선수별 WAR 기여도 (Waterfall) ───────────────────────────────
    st.markdown("---")
    st.markdown("### 선수별 WAR 기여도")
    st.caption(
        "선수를 선택하면 각 피처가 WAR 기준값(Expected Value)에서 "
        "얼마나 더하거나 빼는지 waterfall 차트로 보여줍니다."
    )

    players_in_season = sorted(sr_f["player"].unique().tolist())
    if not players_in_season:
        st.info("해당 시즌 선수 데이터가 없습니다.")
        return

    sel_player = st.selectbox(
        "선수 선택",
        players_in_season,
        key="shap_player",
    )

    player_idx_list = sr_f[sr_f["player"] == sel_player].index.tolist()
    if not player_idx_list:
        st.warning("선택한 선수 데이터를 찾을 수 없습니다.")
        return

    # sr_f 내 상대 인덱스
    sr_f_reset = sr_f.reset_index(drop=True)
    X_f_reset = X_f.reset_index(drop=True)
    player_rows = sr_f_reset[sr_f_reset["player"] == sel_player]
    p_idx = player_rows.index[0]

    player_shap = shap_vals[p_idx]  # shape: (8,)
    player_x = X_f_reset.iloc[p_idx]
    player_war = float(sr_f_reset.iloc[p_idx]["war"])

    # 팀/포지션 정보
    player_team = sr_f_reset.iloc[p_idx].get("team", "")
    player_pos = sr_f_reset.iloc[p_idx].get("pos_group", "")

    # 선수 요약 카드
    col_info, col_war = st.columns([3, 1])
    with col_info:
        st.markdown(
            f"**{sel_player}** | {player_team} | {player_pos} | {sel_season}"
        )
    with col_war:
        st.metric("PIS", f"{player_war:.1f}")

    # Waterfall 데이터 구성
    # 순서: expected_value → 피처별 SHAP 누적 → 최종 예측값
    sorted_idx = np.argsort(np.abs(player_shap))[::-1]  # 절댓값 내림차순
    sorted_feats_ko = [FEATURE_NAMES_KO[i] for i in sorted_idx]
    sorted_shap = [float(player_shap[i]) for i in sorted_idx]
    sorted_feat_vals = [float(player_x.iloc[i]) for i in sorted_idx]

    # x축 레이블: "피처명 = 값"
    x_labels = [
        f"{name}<br>({val:.1f})"
        for name, val in zip(sorted_feats_ko, sorted_feat_vals)
    ]

    wf_fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute"] + ["relative"] * len(sorted_shap) + ["total"],
        x=["기준값 (평균 WAR)"] + x_labels + ["최종 WAR 예측"],
        y=[expected_value] + sorted_shap + [0],  # total은 자동 계산
        connector=dict(line=dict(color="#555577")),
        decreasing=dict(marker_color="#e90052"),
        increasing=dict(marker_color="#00ff87"),
        totals=dict(marker_color="#04f5ff"),
        text=(
            [f"기준: {expected_value:.1f}"]
            + [f"{'+' if v >= 0 else ''}{v:.2f}" for v in sorted_shap]
            + [f"예측: {expected_value + sum(sorted_shap):.1f}"]
        ),
        textposition="outside",
    ))

    wf_fig.update_layout(
        title=dict(
            text=f"{sel_player} WAR 기여도 분해",
            font=dict(color="#ffffff"),
        ),
        yaxis_title="WAR 기여 (SHAP 값)",
        height=480,
        margin=dict(t=60, b=20),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0d0d1a",
        font_color="#ffffff",
        yaxis=dict(gridcolor="#333355"),
        xaxis=dict(tickfont=dict(size=10)),
        showlegend=False,
    )
    st.plotly_chart(wf_fig, use_container_width=True, theme=None)

    # ── 해석 텍스트 ──────────────────────────────────────────────────────────
    top_pos = [(sorted_feats_ko[i], sorted_shap[i]) for i in range(len(sorted_shap)) if sorted_shap[i] > 0]
    top_neg = [(sorted_feats_ko[i], sorted_shap[i]) for i in range(len(sorted_shap)) if sorted_shap[i] < 0]

    if top_pos or top_neg:
        st.markdown("**자동 해석:**")
        lines = []
        if top_pos:
            top3_pos = top_pos[:3]
            lines.append(
                "WAR을 높인 주요 요인: "
                + ", ".join(f"**{name}** (+{val:.1f})" for name, val in top3_pos)
            )
        if top_neg:
            top3_neg = top_neg[:2]
            lines.append(
                "WAR을 낮춘 요인: "
                + ", ".join(f"**{name}** ({val:.1f})" for name, val in top3_neg)
            )
        for line in lines:
            st.markdown(f"- {line}")

    # ── 섹션 4: 시즌 전체 SHAP 산점도 (피처별 값 vs SHAP) ────────────────────
    st.markdown("---")
    st.markdown("### 피처 값과 WAR 영향력 관계")
    st.caption("선택한 피처의 실제 값(X축)과 WAR에 미치는 영향(Y축)의 관계. 점이 위에 있을수록 WAR을 높임.")

    sel_feat_ko = st.selectbox(
        "피처 선택",
        FEATURE_NAMES_KO,
        index=0,
        key="shap_feat_scatter",
    )
    feat_idx = FEATURE_NAMES_KO.index(sel_feat_ko)
    feat_col = FEATURE_NAMES[feat_idx]

    scatter_x = X_f_reset.iloc[:, feat_idx].values
    scatter_y = shap_vals[:, feat_idx]
    scatter_war = sr_f_reset["war"].fillna(0).values
    scatter_names = sr_f_reset["player"].values

    sc_fig = go.Figure(go.Scatter(
        x=scatter_x,
        y=scatter_y,
        mode="markers",
        marker=dict(
            color=scatter_war,
            colorscale="Viridis",
            size=6,
            opacity=0.7,
            colorbar=dict(title="PIS (포지션 내 기여 백분위)", tickfont=dict(color="#fff")),
            showscale=True,
        ),
        text=[
            f"{n}<br>{sel_feat_ko}: {x:.1f}<br>SHAP: {y:+.2f}<br>WAR: {w:.1f}"
            for n, x, y, w in zip(scatter_names, scatter_x, scatter_y, scatter_war)
        ],
        hoverinfo="text",
    ))

    sc_fig.update_layout(
        xaxis_title=sel_feat_ko,
        yaxis_title=f"SHAP 값 ({sel_feat_ko} → WAR 영향)",
        height=400,
        margin=dict(t=20, b=20),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0d0d1a",
        font_color="#ffffff",
        xaxis=dict(gridcolor="#333355"),
        yaxis=dict(gridcolor="#333355", zeroline=True, zerolinecolor="#555577"),
    )
    st.plotly_chart(sc_fig, use_container_width=True, theme=None)

    # ── 섹션 5: 상위/하위 WAR 선수 테이블 ──────────────────────────────────
    st.markdown("---")
    st.markdown(f"### {sel_season} PIS 순위 (상위 20명)")

    display_cols = ["player", "team", "pos_group", "age", "war"]
    available = [c for c in display_cols if c in sr_f_reset.columns]
    rename_map = {
        "player": "선수", "team": "팀", "pos_group": "포지션",
        "age": "나이", "war": "PIS",
    }
    top20 = (
        sr_f_reset[available]
        .sort_values("war", ascending=False)
        .head(20)
        .rename(columns=rename_map)
        .reset_index(drop=True)
    )
    top20.index += 1
    st.dataframe(top20, use_container_width=True)

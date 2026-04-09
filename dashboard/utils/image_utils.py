"""이미지 로딩 유틸리티 - 선수 사진 & 팀 로고

대시보드에서 선수 사진과 팀 로고를 표시하기 위한 헬퍼 함수 모음.
이미지가 없으면 None 반환 → 호출부에서 fallback UI 처리.
"""

import io
import json
import base64
import logging
from pathlib import Path
from functools import lru_cache
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image

logger = logging.getLogger(__name__)

# ── 경로 설정 ───────────────────────────────────────────
ROOT           = Path(__file__).resolve().parent.parent.parent
IMAGE_MAP_PATH = ROOT / "data" / "images" / "image_map.parquet"
LOGO_MAP_PATH  = ROOT / "data" / "images" / "logo_map.json"
PLAYERS_DIR    = ROOT / "data" / "images" / "players"
LOGOS_DIR      = ROOT / "data" / "images" / "team_logos"

# 팀명 별칭 테이블 (대시보드 표시명 → 파일명 기준 팀명)
TEAM_ALIASES = {
    "Man City":        "Manchester City",
    "Man United":      "Manchester United",
    "Man Utd":         "Manchester United",
    "Wolves":          "Wolverhampton Wanderers",
    "Spurs":           "Tottenham Hotspur",
    "Nott'm Forest":   "Nottingham Forest",
    "West Brom":       "West Bromwich Albion",
    "Brighton":        "Brighton  Hove Albion",
    "Bournemouth":     "AFC Bournemouth",
    "Sheffield Utd":   "Sheffield United",
    "QPR":             "Queens Park Rangers",
    "Ipswich":         "Ipswich Town",
    "Leicester":       "Leicester City",
    "Newcastle":       "Newcastle United",
    "Nottingham Forest": "Nottingham Forest",
}


def normalize_team_name(name: str) -> str:
    """팀명 정규화: 대시보드 표시명 → 파일명 기준 팀명."""
    return TEAM_ALIASES.get(name, name)


@st.cache_data(show_spinner=False)
def _load_image_map() -> dict:
    """player → image_path 매핑 딕셔너리 로드."""
    if not IMAGE_MAP_PATH.exists():
        return {}
    df = pd.read_parquet(IMAGE_MAP_PATH)
    return dict(zip(df["player"], df["image_path"]))


@st.cache_data(show_spinner=False)
def _load_logo_map() -> dict:
    """팀명 → logo_path 매핑 딕셔너리 로드."""
    if not LOGO_MAP_PATH.exists():
        return {}
    return json.loads(LOGO_MAP_PATH.read_text(encoding="utf-8"))


def _img_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    """PIL Image → base64 문자열 변환."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def get_player_image(player_name: str, size: tuple = (80, 80)) -> Optional[Image.Image]:
    """선수 이름으로 PIL Image 반환. 없으면 None."""
    image_map = _load_image_map()
    img_path = image_map.get(player_name)
    if not img_path or not Path(img_path).exists():
        # 파일명 기반 직접 탐색 (이름이 약간 다를 경우)
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in player_name)
        candidate = PLAYERS_DIR / f"{safe}.jpg"
        if candidate.exists():
            img_path = str(candidate)
        else:
            return None
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(size, Image.LANCZOS)
        return img
    except Exception as e:
        logger.debug(f"이미지 로드 실패 ({player_name}): {e}")
        return None


def get_player_image_b64(player_name: str, size: tuple = (80, 80)) -> Optional[str]:
    """선수 이름으로 base64 인코딩된 JPEG 이미지 반환. 없으면 None."""
    img = get_player_image(player_name, size)
    if img is None:
        return None
    return _img_to_b64(img, fmt="JPEG")


def get_team_logo(team_name: str, size: tuple = (40, 40)) -> Optional[Image.Image]:
    """팀명으로 PIL Image 로고 반환. 없으면 None."""
    normalized = normalize_team_name(team_name)
    logo_map = _load_logo_map()

    # 1순위: logo_map에서 직접 찾기
    logo_path = logo_map.get(normalized) or logo_map.get(team_name)

    # 2순위: 파일명 기반 탐색 (공백→언더스코어)
    if not logo_path:
        candidate = LOGOS_DIR / f"{normalized.replace(' ', '_')}.png"
        if candidate.exists():
            logo_path = str(candidate)

    if not logo_path or not Path(logo_path).exists():
        return None
    try:
        img = Image.open(logo_path).convert("RGBA")
        img = img.resize(size, Image.LANCZOS)
        return img
    except Exception as e:
        logger.debug(f"로고 로드 실패 ({team_name}): {e}")
        return None


def get_team_logo_b64(team_name: str, size: tuple = (40, 40)) -> Optional[str]:
    """팀명으로 base64 인코딩된 PNG 로고 반환. 없으면 None."""
    img = get_team_logo(team_name, size)
    if img is None:
        return None
    return _img_to_b64(img, fmt="PNG")


def render_player_card(player_name: str, team: str = "", archetype: str = "",
                       similarity: float = None, extra_info: str = "") -> None:
    """Streamlit에 선수 카드 렌더링.

    레이아웃:
        [사진 80x80] | 선수명
                     | 팀명 · 아키타입
                     | 유사도 (있을 경우)
    """
    col_img, col_info = st.columns([1, 3])
    with col_img:
        img_b64 = get_player_image_b64(player_name, size=(80, 80))
        if img_b64:
            st.markdown(
                f'<img src="data:image/jpeg;base64,{img_b64}" '
                f'style="border-radius:8px; width:72px; height:72px; object-fit:cover;">',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="width:72px;height:72px;background:#2a2a4a;'
                'border-radius:8px;display:flex;align-items:center;'
                'justify-content:center;font-size:28px;">👤</div>',
                unsafe_allow_html=True,
            )
    with col_info:
        st.markdown(f"**{player_name}**")
        meta_parts = []
        if team:
            meta_parts.append(team)
        if archetype:
            meta_parts.append(archetype)
        if meta_parts:
            st.caption(" · ".join(meta_parts))
        if similarity is not None:
            st.caption(f"유사도: {similarity:.1%}")
        if extra_info:
            st.caption(extra_info)

"""이미지 로딩 유틸리티 - 선수 사진 & 팀 로고

대시보드에서 선수 사진과 팀 로고를 표시하기 위한 헬퍼 함수 모음.
이미지가 없으면 None 반환 → 호출부에서 fallback UI 처리.
"""

import io
import json
import base64
import logging
import unicodedata
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

# 팀명 → 로고 파일명 직접 매핑 (데이터 표시명 → team_logos/ 파일 stem)
# 파일명 규칙: 공백→언더스코어, 특수문자 제거
TEAM_LOGO_FILEMAP = {
    # 데이터 단축명
    "Arsenal":          "Arsenal_FC",
    "Aston Villa":      "Aston_Villa",
    "Birmingham":       "Birmingham_City",
    "Blackburn":        "Blackburn_Rovers",
    "Blackpool":        "Blackpool_FC",
    "Bolton":           "Bolton_Wanderers",
    "Bournemouth":      "AFC_Bournemouth",
    "Bradford":         "Bradford_City",
    "Brentford":        "Brentford_FC",
    "Brighton":         "Brighton__Hove_Albion",
    "Burnley":          "Burnley_FC",
    "Cardiff":          "Cardiff_City",
    "Charlton":         "Charlton_Athletic",
    "Chelsea":          "Chelsea_FC",
    "Coventry":         "Coventry_City",
    "Crystal Palace":   "Crystal_Palace",
    "Derby":            "Derby_County",
    "Everton":          "Everton_FC",
    "Fulham":           "Fulham_FC",
    "Ipswich":          "Ipswich_Town",
    "Leeds":            "Leeds_United",
    "Leicester":        "Leicester_City",
    "Liverpool":        "Liverpool_FC",
    "Luton":            "Luton_Town",
    "Man City":         "Manchester_City",
    "Man United":       "Manchester_United",
    "Man Utd":          "Manchester_United",
    "Middlesbrough":    "Middlesbrough_FC",
    "Newcastle":        "Newcastle_United",
    "Nott'm Forest":    "Nottingham_Forest",
    "Nottingham Forest":"Nottingham_Forest",
    "Portsmouth":       "Portsmouth_FC",
    "QPR":              "Queens_Park_Rangers",
    "Reading":          "Reading_FC",
    "Sheffield United": "Sheffield_United",
    "Sheffield Utd":    "Sheffield_United",
    "Southampton":      "Southampton_FC",
    "Spurs":            "Tottenham_Hotspur",
    "Stoke":            "Stoke_City",
    "Sunderland":       "Sunderland_AFC",
    "Tottenham":        "Tottenham_Hotspur",
    "Watford":          "Watford_FC",
    "West Brom":        "West_Bromwich_Albion",
    "West Ham":         "West_Ham_United",
    "Wigan":            "Wigan_Athletic",
    "Wolves":           "Wolverhampton_Wanderers",
    # 풀네임
    "Arsenal FC":               "Arsenal_FC",
    "AFC Bournemouth":          "AFC_Bournemouth",
    "Brighton & Hove Albion":   "Brighton__Hove_Albion",
    "Brighton  Hove Albion":    "Brighton__Hove_Albion",
    "Manchester City":          "Manchester_City",
    "Manchester United":        "Manchester_United",
    "Wolverhampton Wanderers":  "Wolverhampton_Wanderers",
    "West Bromwich Albion":     "West_Bromwich_Albion",
    "West Ham United":          "West_Ham_United",
    "Newcastle United":         "Newcastle_United",
    "Leicester City":           "Leicester_City",
    "Ipswich Town":             "Ipswich_Town",
    "Leeds United":             "Leeds_United",
    "Luton Town":               "Luton_Town",
}


def normalize_team_name(name: str) -> str:
    """팀명 정규화: 대시보드 표시명 → 파일명 기준 팀명."""
    # TEAM_LOGO_FILEMAP에 있으면 파일 stem 반환, 없으면 원본
    return TEAM_LOGO_FILEMAP.get(name, name)


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


def _normalize_player_name(name: str) -> str:
    """선수명 정규화: 유니코드 → ASCII 근사, 특수문자 → 언더스코어."""
    # NFD 분해 후 비ASCII 결합문자 제거 (é→e, ć→c 등)
    nfd = unicodedata.normalize("NFD", name)
    ascii_approx = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    # 알파벳, 숫자, 공백, 하이픈, 언더스코어 외 문자 → 언더스코어
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in ascii_approx)
    return safe.strip()


def get_player_image(player_name: str, size: tuple = (80, 80)) -> Optional[Image.Image]:
    """선수 이름으로 PIL Image 반환. 없으면 None.

    탐색 순서:
    1. image_map.parquet 직접 매핑 (정확한 선수명 일치)
    2. 특수문자 → 언더스코어 변환 후 파일 탐색
    3. 유니코드 정규화(NFD→ASCII 근사) 후 파일 탐색
    """
    image_map = _load_image_map()
    img_path = image_map.get(player_name)

    if not img_path or not Path(img_path).exists():
        # 2순위: 특수문자만 언더스코어로 변환
        safe = "".join(c if c.isalnum() or c in " -_." else "_" for c in player_name)
        candidate = PLAYERS_DIR / f"{safe}.jpg"
        if candidate.exists():
            img_path = str(candidate)
        else:
            # 3순위: 유니코드 정규화 후 탐색
            normalized = _normalize_player_name(player_name)
            candidate2 = PLAYERS_DIR / f"{normalized}.jpg"
            if candidate2.exists():
                img_path = str(candidate2)
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
    """팀명으로 PIL Image 로고 반환. 없으면 None.

    탐색 순서:
    1. TEAM_LOGO_FILEMAP → team_logos/{stem}.png 직접 탐색 (가장 신뢰도 높음)
    2. logo_map.json 경로 (절대 경로가 존재하는 경우)
    3. 팀명 공백→언더스코어 변환 후 파일 탐색
    """
    # 1순위: TEAM_LOGO_FILEMAP 직접 파일 탐색
    stem = TEAM_LOGO_FILEMAP.get(team_name)
    if stem:
        direct = LOGOS_DIR / f"{stem}.png"
        if direct.exists():
            try:
                img = Image.open(direct).convert("RGBA")
                img = img.resize(size, Image.LANCZOS)
                return img
            except Exception as e:
                logger.debug(f"로고 로드 실패 ({team_name}): {e}")

    # 2순위: logo_map.json 절대 경로
    logo_map = _load_logo_map()
    raw_path = logo_map.get(team_name)
    if raw_path and Path(raw_path).exists():
        try:
            img = Image.open(raw_path).convert("RGBA")
            img = img.resize(size, Image.LANCZOS)
            return img
        except Exception as e:
            logger.debug(f"로고 로드 실패 ({team_name}): {e}")

    # 3순위: 공백→언더스코어 파일 탐색
    candidate = LOGOS_DIR / f"{team_name.replace(' ', '_')}.png"
    if candidate.exists():
        try:
            img = Image.open(candidate).convert("RGBA")
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

"""Scout Transfer Tools - S3 유사선수 + S4 성장참조 + S5 이적리스크

이적/영입 의사결정을 위한 통합 분석 도구.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from dashboard.components.data_loader import (
    load_clusters, load_similarity_matrix, load_transfer_predictions,
    load_s4_reference, load_scout_ratings,
    load_growth_predictions, load_growth_predictions_v4, load_transfer_adapt_predictions,
    load_undervalued,
)
from dashboard.utils.image_utils import render_player_card

EPL_PURPLE = "#37003c"
EPL_MAGENTA = "#e90052"
EPL_GREEN = "#00ff87"
EPL_CYAN = "#04f5ff"

# 데이터프레임 컬럼 한국어 매핑
COL_RENAME = {
    "player": "선수",
    "team": "팀",
    "position": "포지션",
    "distance": "유사도(거리)",
    "goals_p90": "골/90분",
    "assists_p90": "어시스트/90분",
    "tackles_p90": "태클/90분",
    "market_value_raw": "시장가치",
    "team_old": "이전팀",
    "team_new": "새팀",
    "season_new": "이적시즌",
    "pos_group": "포지션",
    "age_at_transfer": "이적시 나이",
    "prob_failure": "실패확률",
    "prob_success": "성공확률",
    "elo_gap": "ELO 격차",
    "style_match_pct": "스타일매칭률",
    "pred_label": "예측결과",
    "market_value": "시장가치",
}

# ── 아키타입 병합 매핑 (v4 한국어 아키타입 기준, v3 영문 폴백용만 유지) ──
ARCHETYPE_MERGE = {
    # v3 → v4 변환 (구버전 데이터 폴백 지원)
    "High-Volume Attacking Full-Back": "🏃 공격형 풀백",
    "Standard Attacking Full-Back":    "🏃 공격형 풀백",
    "Low-Block Attacking Full-Back":   "🏃 공격형 풀백",
    "Modern Full-Back":                "🏃 공격형 풀백",
    "Attacking Full-Back":             "🏃 공격형 풀백",
    "Compact Ball-Winning Defender":   "💪 공중볼 수비수",
    "Ball-Winning Defender":           "💪 공중볼 수비수",
    "Aerial Ball-Winning Defender":    "🛡️ 전통 수비수",
    "Ball-Winning Midfielder":         "📈 박스 투 박스",
    "Defensive Midfielder":            "💪 수비형 MF",
    "Central Midfielder":              "🔑 플레이메이커",
    "Creative Playmaker":              "🎨 창의적 플레이메이커",
    "Attacking Wide Forward":          "🎯 측면 공격수",
    "High-Volume Goal Poacher":        "⚽ 박스 스트라이커",
    "Efficient Goal Poacher":          "🎯 효율형 득점왕",
    "Assist-Focused Creative Winger":  "🎨 창의적 공격수",
    "Dribbling Creative Winger":       "🎨 창의적 공격수",
    "Goalkeeper":                      "🧤 슈팅 스토퍼",
    # v4 이름은 그대로 통과
}

# ── pos_group별 허용 아키타입 (v4 기준, 전체 목록) ──────────────────────
POS_VALID_ARCHETYPES = {
    "GK":  {"🧤 슈팅 스토퍼", "🧱 클린시트 키퍼"},
    "DEF": {"🏃 공격형 풀백", "🛡️ 전통 수비수", "💪 공중볼 수비수", "🛡️ 볼배급 센터백"},
    "MID": {"📈 박스 투 박스", "🔑 플레이메이커", "🔑 딥라잉 플레이메이커",
            "🎨 창의적 플레이메이커", "💪 수비형 MF", "🎯 공격형 MF"},
    "FW":  {"⚽ 박스 스트라이커", "🎯 효율형 득점왕", "🎯 측면 공격수",
            "🎨 창의적 공격수", "🏃 압박 전방", "🔫 볼륨 슈터"},
}


def _fix_pos_archetype(df: pd.DataFrame) -> pd.DataFrame:
    """pos_group vs archetype 불일치 수정.
    v4는 포지션별 분리 클러스터링이므로 오배정이 거의 없음.
    v3 영문 이름 → v4 한국어 변환 (ARCHETYPE_MERGE)은 호출 이후 별도로 적용됨.
    """
    def _fix_row(row):
        pos  = row.get("pos_group", "")
        arch = row.get("archetype", "")
        valid = POS_VALID_ARCHETYPES.get(pos)
        # 이미 유효한 v4 아키타입이면 그대로 반환
        if valid is None or arch in valid:
            return arch
        # v3 영문 이름은 ARCHETYPE_MERGE로 처리되므로 여기서는 스탯 기반 폴백만 적용
        if pos == "GK":
            return "🧤 슈팅 스토퍼"
        if pos == "DEF":
            assists = row.get("assists_p90", 0) or 0
            key_passes = row.get("key_passes_p90", 0) or 0
            tackles = row.get("tackles_p90", 0) or 0
            if assists > 0.10 or key_passes > 2.0:
                return "🏃 공격형 풀백"
            elif tackles > 1.2:
                return "💪 공중볼 수비수"
            return "🛡️ 전통 수비수"
        if pos == "MID":
            key_passes = row.get("key_passes_p90", 0) or 0
            tackles = row.get("tackles_p90", 0) or 0
            if key_passes > 2.5:
                return "🎨 창의적 플레이메이커"
            elif tackles > 1.5:
                return "📈 박스 투 박스"
            return "🔑 플레이메이커"
        if pos == "FW":
            goals = row.get("goals_p90", 0) or 0
            key_passes = row.get("key_passes_p90", 0) or 0
            if key_passes > 2.0 and goals < 0.20:
                return "🎨 창의적 공격수"
            elif goals > 0.40:
                return "⚽ 박스 스트라이커"
            return "🎯 측면 공격수"
        return arch

    df = df.copy()
    df["archetype"] = df.apply(_fix_row, axis=1)
    return df

# ── 아키타입 특성 정보 (v4 한국어 기준, 실제 데이터 스탯 기반) ──────────
# 스탯: 2024/25 시즌 클러스터 평균값 (cluster_assignments_v4.parquet)
ARCHETYPE_INFO = {
    # ── FW 포지션 ──────────────────────────────────────────────────────────
    "⚽ 박스 스트라이커": {
        "icon": "⚽", "pos": "FW", "color": EPL_MAGENTA,
        "description": (
            "페널티박스 안에서 슈팅 볼륨과 결정력을 모두 갖춘 EPL 최상위 공격 타입. "
            "골·어시스트 모두 FW 중 최상위이며 팀 공격의 핵심 완성자 역할."
        ),
        "strengths": "득점 결정력, 슈팅 볼륨, 어시스트 병행",
        "weaknesses": "수비 가담 최하위, 수비 액션 0.82회 (FW 최저)",
        "scout_tip": (
            "팀 예산의 최우선 투자 포지션. 슈팅 전환율(실제골÷슈팅)로 진짜 결정력 확인 필수. "
            "고령 스트라이커는 WAR 하락세 체크 후 계약 기간 단기로 제한 권장."
        ),
        "key_stats": "골 0.51/90분, 슈팅 2.71회, 어시스트 0.23/90분, 평균나이 26.2세",
        "rep_players": ["Mohamed Salah", "Alexander Isak", "Ollie Watkins"],
        "small_sample": False,
    },
    "🎯 측면 공격수": {
        "icon": "🎯", "pos": "FW", "color": EPL_MAGENTA,
        "description": (
            "측면에서 득점과 창의성을 모두 구사하는 윙어. 슈팅 볼륨도 높고 키패스도 풍부해 "
            "박스 스트라이커보다 창의적이지만 득점력은 약간 낮음."
        ),
        "strengths": "득점+창의성 병행, 키패스 2.44회 (FW 중 최다), 광역 위협",
        "weaknesses": "박스 스트라이커 대비 골 결정력 열위",
        "scout_tip": (
            "측면 공격력과 중앙 득점을 함께 원하는 팀에 최적. "
            "어시스트 목표치를 높게 설정하기보다 직접 득점 기여로 평가하는 게 적합."
        ),
        "key_stats": "골 0.30/90분, 슈팅 2.39회, 키패스 2.44회, 평균나이 25.4세",
        "rep_players": ["Anthony Elanga", "Kai Havertz", "Kaoru Mitoma"],
        "small_sample": False,
    },
    "🎯 효율형 득점왕": {
        "icon": "🎯", "pos": "FW", "color": EPL_MAGENTA,
        "description": (
            "높은 슈팅 볼륨(2.76회)으로 꾸준히 득점하는 공격수. 측면 공격수보다 더 중앙 지향적이며 "
            "키패스도 유의미하게 창출. 실질적으로 고볼륨 스트라이커 유형."
        ),
        "strengths": "슈팅 볼륨 최상위(2.76회), 안정적 득점 생산",
        "weaknesses": "박스 스트라이커 대비 결정율 낮음 (슈팅 많지만 골은 적음)",
        "scout_tip": (
            "xG 대비 실제 골 성취율(Under-performance)이 있을 수 있음. "
            "슈팅 기회를 많이 만들어주는 팀에서 결정율이 올라가는 타입 — 팀 전술 맞춤 확인 필수."
        ),
        "key_stats": "골 0.40/90분, 슈팅 2.76회, 키패스 2.17회, 평균나이 25.0세",
        "rep_players": ["Rodrigo Muniz", "Diogo Jota", "Harvey Barnes"],
        "small_sample": False,
    },
    "🎨 창의적 공격수": {
        "icon": "🎨", "pos": "FW", "color": "#9B59B6",
        "description": (
            "득점보다 어시스트·키패스·드리블로 팀 공격을 설계하는 윙어/세컨드 스트라이커. "
            "직접 득점은 낮지만 팀 공격 흐름을 만드는 역할. 평균 나이 24세로 가장 젊음."
        ),
        "strengths": "키패스 2.36회, 창의성, 어린 나이 — 성장 여지 큼",
        "weaknesses": "직접 득점 0.15/90분 (FW 최저), 박스 결정력 기대 어려움",
        "scout_tip": (
            "박스 스트라이커와 조합 시 시너지 극대화. "
            "단독 득점 기여로 평가하면 과소평가. 어시스트+키패스 합산으로 가치 측정 권장."
        ),
        "key_stats": "골 0.15/90분, 키패스 2.36회, 슈팅 1.98회, 평균나이 24.1세",
        "rep_players": ["Georginio Rutter", "Jadon Sancho", "Simon Adingra"],
        "small_sample": False,
    },
    # ── MID 포지션 ─────────────────────────────────────────────────────────
    "🎨 창의적 플레이메이커": {
        "icon": "🎨", "pos": "MID", "color": EPL_CYAN,
        "description": (
            "MID 중 득점(0.27)·어시스트(0.26)·키패스(3.68) 모두 최상위인 공격형 미드필더. "
            "사실상 10번 역할로 팀의 창의적 공격을 주도하며 슈팅(2.71회)도 활발."
        ),
        "strengths": "키패스 3.68회 (MID 최고), 골·어시스트 동시 생산, 슈팅 위협",
        "weaknesses": "수비 기여 낮음 (def_actions 1.14회), 팀 수비 구조에 부담",
        "scout_tip": (
            "팀에 이 유형 부재 시 공격이 단조로워짐. "
            "보호해줄 수비형/박스투박스 MF가 필수 세트. 예산 집중 포지션."
        ),
        "key_stats": "골 0.27/90분, 어시스트 0.26/90분, 키패스 3.68회, 슈팅 2.71회",
        "rep_players": ["James Maddison", "Son Heung-min", "Justin Kluivert"],
        "small_sample": False,
    },
    "🔑 딥라잉 플레이메이커": {
        "icon": "🔑", "pos": "MID", "color": EPL_CYAN,
        "description": (
            "중앙 깊숙이서 공격 흐름을 설계하는 MF. 키패스(2.84회)가 높으면서 수비 기여(1.60회)도 "
            "창의적 플레이메이커보다 높아 양면을 겸비. 박스 침투보다 배급 중심."
        ),
        "strengths": "키패스 2.84회, 수비 기여 병행, 빌드업 핵심",
        "weaknesses": "창의적 PM 대비 직접 득점 열위 (0.13/90분)",
        "scout_tip": (
            "감독이 4-3-3 또는 4-2-3-1에서 중원 사령관 역할 원할 때 최적. "
            "패스 성공률·챈스 크리에이션 지표 중점 평가."
        ),
        "key_stats": "키패스 2.84회, 골 0.13/90분, 어시스트 0.18/90분, 수비액션 1.60회",
        "rep_players": ["Martin Ødegaard", "Morgan Gibbs-White", "Morgan Rogers"],
        "small_sample": False,
    },
    "🔑 플레이메이커": {
        "icon": "🔑", "pos": "MID", "color": EPL_CYAN,
        "description": (
            "공격과 수비를 균형 있게 소화하는 중앙 미드필더. 키패스(1.80회)와 태클(1.27회)이 "
            "모두 리그 평균 이상. 딥라잉보다 상위 포지션, 박스투박스보다 창의적."
        ),
        "strengths": "공수 균형, 키패스+태클 동시 기여, 멀티롤",
        "weaknesses": "창의적 PM·딥라잉PM 대비 특화 강점 부재",
        "scout_tip": (
            "WAR 대비 시장가치가 낮은 경우가 많은 숨은 보석 타입. "
            "로테이션 자원이나 시스템 안정화 목적 영입에 가성비 최고."
        ),
        "key_stats": "키패스 1.80회, 태클 1.27회, 골 0.08/90분, 어시스트 0.09/90분",
        "rep_players": ["Matt O'Riley", "Facundo Buonanotte", "Jean-Ricner Bellegarde"],
        "small_sample": False,
    },
    "📈 박스 투 박스": {
        "icon": "📈", "pos": "MID", "color": EPL_CYAN,
        "description": (
            "전체 MID 중 수비 액션(2.80회)·태클(1.59회) 최상위인 워크레이트형 미드필더. "
            "공격 기여는 적지만 볼 탈취와 압박으로 팀 수비 구조를 지탱."
        ),
        "strengths": "수비액션 2.80회 (MID 최고), 태클 1.59회, 압박 기여",
        "weaknesses": "창의성·득점 낮음 (골 0.06, 키패스 0.85회)",
        "scout_tip": (
            "창의적 플레이메이커 옆에 배치할 때 팀 균형 극대화. "
            "저렴하게 팀 수비력을 즉각 향상시킬 수 있는 유형. 체력 지표 필수 확인."
        ),
        "key_stats": "수비액션 2.80회, 태클 1.59회, 골 0.06/90분, 키패스 0.85회",
        "rep_players": ["Christian Nørgaard", "Mario Lemina", "Jack Hinshelwood"],
        "small_sample": False,
    },
    "💪 수비형 MF": {
        "icon": "💪", "pos": "MID", "color": EPL_GREEN,
        "description": (
            "순수 수비 커버에 집중하는 DM. 박스투박스보다 더 낮은 라인에 포지션하며 "
            "득점·창의성보다 차단·포지셔닝으로 팀 수비 구조의 기반 역할."
        ),
        "strengths": "포지셔닝, 팀 수비 구조 안정, 백포 커버",
        "weaknesses": "공격 기여 거의 없음, 키패스 낮음",
        "scout_tip": (
            "저렴하게 팀 수비 안정화 가능한 유형. "
            "단, 현재 EPL 최신 시즌 기준 소규모 클러스터 — 해당 유형 선수 수 제한적."
        ),
        "key_stats": "득점 0.00/90분, 키패스 1.40회, 태클 0.52회",
        "rep_players": ["Harry Winks", "Archie Gray"],
        "small_sample": True,  # ⚠️ 최신 시즌 2명
    },
    "🎯 공격형 MF": {
        "icon": "🎯", "pos": "MID", "color": EPL_CYAN,
        "description": (
            "포지션은 MF이지만 전방 지원 역할을 소화하는 공격 성향 미드필더. "
            "현재 EPL 데이터 기준 소규모 클러스터로 통계적 대표성이 낮음."
        ),
        "strengths": "전방 압박, 공격 연계",
        "weaknesses": "소규모 클러스터 — 통계 신뢰도 낮음",
        "scout_tip": (
            "⚠️ 이 아키타입은 최신 시즌 기준 배정 선수 수가 매우 적어 "
            "유사 선수 탐색 신뢰도가 낮습니다. 🎨 창의적 플레이메이커 또는 🔑 딥라잉 PM과 교차 확인 권장."
        ),
        "key_stats": "골 0.11/90분, 키패스 0.81회, 태클 0.98회",
        "rep_players": ["Abdoulaye Doucouré"],
        "small_sample": True,  # ⚠️ 최신 시즌 1명
    },
    # ── DEF 포지션 ─────────────────────────────────────────────────────────
    "🏃 공격형 풀백": {
        "icon": "🏃", "pos": "DEF", "color": EPL_GREEN,
        "description": (
            "오버래핑과 창의적 배급으로 측면 공격을 주도하는 현대형 풀백. "
            "어시스트(0.14/90분)·키패스(3.85회)가 수비수 중 압도적. "
            "트렌트 알렉산더-아놀드·테오 에르난데스 유형."
        ),
        "strengths": "키패스 3.85회 (수비수 최고), 어시스트 0.14/90분, 측면 공격 창출",
        "weaknesses": "오버래핑 시 뒤 공간 노출, 수비 복귀 속도",
        "scout_tip": (
            "윙어 예산 부족 시 이 유형 풀백으로 측면 공격 커버 가능. "
            "크로스 전술 구사 팀과 최고 시너지. 측면 수비 취약점 반드시 병행 확인."
        ),
        "key_stats": "키패스 3.85회, 어시스트 0.14/90분, 태클 1.47회, 평균나이 26.3세",
        "rep_players": ["Trent Alexander-Arnold", "Rayan Aït-Nouri", "Matheus Nunes"],
        "small_sample": False,
    },
    "💪 공중볼 수비수": {
        "icon": "💪", "pos": "DEF", "color": EPL_GREEN,
        "description": (
            "공중볼 경합과 적극적 태클로 상대 공격을 차단하는 현대적 CB/FB. "
            "수비 액션(2.02회)·태클(1.03회) 모두 높으며 세트피스 상황에서도 위협. "
            "공격 기여는 제한적이지만 수비 조직력의 핵심."
        ),
        "strengths": "태클 1.03회, 수비액션 2.02회, 공중볼 경합, 압박",
        "weaknesses": "키패스 0.49회 (빌드업 기여 낮음), 공격 참여 제한",
        "scout_tip": (
            "수비 강화 즉효 처방. 태클 성공률과 공중 경합 승률 함께 확인 권장. "
            "장기 계약 시 나이와 부상 이력 체크."
        ),
        "key_stats": "태클 1.03회, 수비액션 2.02회, 키패스 0.49회, 평균나이 25.3세",
        "rep_players": ["Nikola Milenković", "Trevoh Chalobah", "Gabriel"],
        "small_sample": False,
    },
    "🛡️ 전통 수비수": {
        "icon": "🛡️", "pos": "DEF", "color": EPL_GREEN,
        "description": (
            "수비 기여(2.03회)와 태클(1.08회)을 갖추면서 키패스(1.07회)도 공중볼 수비수보다 높아 "
            "빌드업 참여가 가능한 균형형 수비수. 공중볼보다 포지셔닝과 패스로 수비."
        ),
        "strengths": "수비액션 2.03회, 키패스 1.07회, 빌드업 기여 병행",
        "weaknesses": "공중볼 수비수 대비 압박·태클 강도 약함",
        "scout_tip": (
            "볼 보유 기반 팀에서 수비수도 빌드업 참여가 필요할 때 최적. "
            "패스 성공률 + 수비 성공률 동시 확인 권장."
        ),
        "key_stats": "태클 1.08회, 수비액션 2.03회, 키패스 1.07회, 평균나이 26.0세",
        "rep_players": ["Riccardo Calafiori", "Michael Keane", "Axel Disasi"],
        "small_sample": False,
    },
    "🛡️ 볼배급 센터백": {
        "icon": "🛡️", "pos": "DEF", "color": EPL_GREEN,
        "description": (
            "낮은 블록에서 안정적으로 포지션을 유지하며 간결한 패스로 수비를 정리하는 CB. "
            "수비 액션(1.13회)과 태클(0.53회)이 다른 수비 유형보다 낮아 "
            "'지키는 수비'보다 '안전한 수비' 지향. 빌드업 패스 지표로 평가해야 함."
        ),
        "strengths": "안정적 포지셔닝, 리스크 회피 수비, 팀 구조 유지",
        "weaknesses": "태클·수비액션 최저 (소극적 수비 스타일), 물리적 압박 부족",
        "scout_tip": (
            "패스 기반 팀의 후방 안전판. 단, '볼배급'이라는 이름과 달리 "
            "키패스 수치가 낮음 — 배급보다 안정성을 중시하는 유형으로 이해할 것. "
            "나이 많은 베테랑 CB 비율이 높음 (평균 28.4세)."
        ),
        "key_stats": "태클 0.53회, 수비액션 1.13회, 키패스 0.14회, 평균나이 28.4세",
        "rep_players": ["Pau Torres", "Conor Coady", "Cameron Burgess"],
        "small_sample": False,
    },
    # ── GK 포지션 ──────────────────────────────────────────────────────────
    "🧱 클린시트 키퍼": {
        "icon": "🧱", "pos": "GK", "color": EPL_PURPLE,
        "description": (
            "팀 수비 조직력과 함께 클린시트 비율이 높은 GK. 배급(어시스트 0.02/90분)도 "
            "슈팅 스토퍼 대비 약간 높아 볼 소유 기반 팀에 적합."
        ),
        "strengths": "클린시트 집중, 배급 기여, 팀 수비 안정",
        "weaknesses": "세이브 볼륨 자체는 슈팅 스토퍼보다 낮을 수 있음",
        "scout_tip": (
            "GK WAR는 팀 수비 시스템과 강하게 연동. "
            "클린시트 수는 팀 수비 전체의 결과이므로 GK 개인 기여 분리 어려움. "
            "xG 대비 실제 실점 지표(PSxG)로 개인 능력 따로 평가 권장."
        ),
        "key_stats": "클린시트율 高, 어시스트 0.02/90분, 평균나이 29.5세",
        "rep_players": ["Ederson", "Jordan Pickford", "Mark Flekken"],
        "small_sample": False,
    },
    "🧤 슈팅 스토퍼": {
        "icon": "🧤", "pos": "GK", "color": EPL_PURPLE,
        "description": (
            "세이브 능력에 특화된 GK. 슈팅이 많은 상대를 막아내는 능력이 강점이며 "
            "배급보다 반응속도와 포지셔닝으로 클린시트를 지킴."
        ),
        "strengths": "세이브율, 반응 속도, 위기 상황 방어",
        "weaknesses": "배급 기여 상대적으로 낮음",
        "scout_tip": (
            "수비 취약 팀에서 GK 능력으로 실점을 최소화해야 할 때 최적. "
            "단, 수비 전체가 취약하면 GK 혼자 막는 데 한계 — 수비 보강과 병행 필수."
        ),
        "key_stats": "세이브율 高, 배급 어시스트 낮음, 평균나이 27.5세",
        "rep_players": ["Emiliano Martínez", "Bernd Leno", "Bart Verbruggen"],
        "small_sample": False,
    },
}


def _rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    """데이터프레임 컬럼을 한국어로 변환"""
    return df.rename(columns={k: v for k, v in COL_RENAME.items() if k in df.columns})


def _render_archetype_card(arch_name: str, info: dict):
    """아키타입 카드 렌더링"""
    st.markdown(
        f"""
<div style="border-left: 4px solid {info['color']}; padding: 8px 12px; margin: 4px 0; background: #f8f8f8; border-radius: 0 6px 6px 0;">
<b>{info['icon']} {arch_name}</b> <span style="color:#666; font-size:0.85em;">({info['kor']})</span>
&nbsp; <code style="background:{info['color']}22; color:{info['color']}; border:none; padding:1px 6px; border-radius:3px;">{info['pos']}</code><br>
<span style="font-size:0.9em;">{info['description']}</span><br>
<span style="font-size:0.8em; color:#888;">📊 {info['key_stats']}</span>
</div>
""",
        unsafe_allow_html=True,
    )


def render():
    st.title("이적 인텔리전스")
    st.markdown(
        "**김태현 스카우트 관점** — 유사 선수 발굴(S3)·성장 레퍼런스(S4)·이적 리스크(S5)를 통합해 "
        "영입 제안 전 종합 의사결정을 지원합니다."
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "유사 선수 탐색 (S3)", "성장 레퍼런스 (S4)", "이적 리스크 (S5)", "🎮 이적 시뮬레이터"
    ])

    # ══════════════════════════════════════════
    # TAB 1: 유사 선수 탐색 (S3)
    # ══════════════════════════════════════════
    with tab1:
        st.subheader("S3. 유사 선수 탐색 엔진")
        with st.expander("📖 S3 아키타입 사용법 & 설명 (스카우터 필독)", expanded=False):
            st.markdown("""
**핵심 사용법:**
1. **UMAP 지도에서 가까운 선수 = 비슷한 플레이 스타일** → 이적한 선수 대체재 발굴에 활용
2. **같은 아키타입 내 유사도(거리) 낮은 선수** = 플레이 방식이 가장 유사
3. **'유사 선수 검색'** 에서 시즌 → 구단 → 선수 선택 → 같은 아키타입 내 대안 선수 목록 확인
4. **유사도 거리**가 낮을수록 플레이 스타일이 거의 동일 (0.000에 가까울수록 유사)

**스카우팅 활용 예시:**
- "이 선수 이적 갔는데 비슷한 선수 있어?" → 검색에서 해당 선수 → 유사 선수 목록 확인
- "측면 공격 보강하려는데 풀백으로 커버 가능한 선수?" → '공격형 풀백' 아키타입 탐색
            """)
            st.markdown("---")
            st.markdown("### 아키타입 완전 가이드 (v4 실제 데이터 기준)")
            st.caption("2024/25 시즌 cluster_assignments_v4 실제 스탯 평균 기반. 대표 선수는 해당 시즌 골+어시스트 상위 선수.")

            # v4 포지션별 분류 (실제 데이터 아키타입)
            pos_groups = {
                "⚔️ FW (공격 포지션)": [
                    "⚽ 박스 스트라이커", "🎯 측면 공격수",
                    "🎯 효율형 득점왕", "🎨 창의적 공격수",
                ],
                "🔧 MID (미드필드)": [
                    "🎨 창의적 플레이메이커", "🔑 딥라잉 플레이메이커",
                    "🔑 플레이메이커", "📈 박스 투 박스",
                    "💪 수비형 MF", "🎯 공격형 MF",
                ],
                "🛡️ DEF (수비 포지션)": [
                    "🏃 공격형 풀백", "💪 공중볼 수비수",
                    "🛡️ 전통 수비수", "🛡️ 볼배급 센터백",
                ],
                "🧤 GK (골키퍼)": [
                    "🧱 클린시트 키퍼", "🧤 슈팅 스토퍼",
                ],
            }

            for group_title, archs in pos_groups.items():
                st.markdown(f"**{group_title}**")
                for arch in archs:
                    if arch in ARCHETYPE_INFO:
                        info = ARCHETYPE_INFO[arch]
                        is_small = info.get("small_sample", False)
                        badge = " &nbsp;<span style='background:#e90052;color:#fff;font-size:0.75em;padding:1px 6px;border-radius:3px;'>⚠️ 소규모</span>" if is_small else ""
                        st.markdown(
                            f"<div style='background:#1e1e3a;border-left:3px solid "
                            f"{info.get('color','#e90052')};padding:8px 12px;"
                            f"margin:6px 0;border-radius:4px'>"
                            f"<b>{info['icon']} {arch}</b> — {info['pos']}{badge}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"**설명**: {info['description']}")
                            st.markdown(f"**강점**: {info['strengths']}")
                            st.markdown(f"**약점**: {info['weaknesses']}")
                        with c2:
                            st.markdown(f"**핵심 스탯**: `{info['key_stats']}`")
                            rep = info.get("rep_players", [])
                            if rep:
                                st.markdown(f"**대표 선수**: {', '.join(rep)}")
                            if is_small:
                                st.warning("⚠️ 소규모 클러스터: 최신 시즌 배정 선수가 매우 적어 유사 선수 탐색 신뢰도가 낮습니다.")
                            else:
                                st.info(f"🔍 **스카우터 팁**: {info['scout_tip']}")
                        st.markdown("---")

        clusters = load_clusters()
        if len(clusters) == 0:
            st.warning("cluster_assignments_v4.parquet 데이터 없음")
        else:
            col1, col2 = st.columns(2)
            seasons = sorted(clusters["season"].unique(), reverse=True)
            with col1:
                sel_season = st.selectbox("시즌", seasons, key="sim_season")
            with col2:
                pos_options = ["전체"] + sorted(clusters["pos_group"].dropna().unique().tolist())
                sel_pos = st.selectbox("포지션", pos_options, key="sim_pos")

            cdf = clusters[clusters["season"] == sel_season].copy()
            # 1단계: pos_group 오분류 수정 (DEF→Attacking Wide Forward 등)
            cdf = _fix_pos_archetype(cdf)
            # 2단계: 아키타입 병합 적용 (풀백 3종→공격형풀백, 수비수 2종→볼탈취수비수)
            cdf["archetype"] = cdf["archetype"].map(lambda x: ARCHETYPE_MERGE.get(x, x))
            if sel_pos != "전체":
                cdf = cdf[cdf["pos_group"] == sel_pos]

            # PCA/UMAP 시각화 (v4: pca_x/y, v3 폴백: umap_x/y)
            x_col = "pca_x" if "pca_x" in cdf.columns else ("umap_x" if "umap_x" in cdf.columns else None)
            y_col = "pca_y" if "pca_y" in cdf.columns else ("umap_y" if "umap_y" in cdf.columns else None)
            use_pca = x_col == "pca_x"

            if x_col and y_col:
                vis_df = cdf.dropna(subset=[x_col, y_col]).copy()
                if len(vis_df):
                    fig = go.Figure()
                    archetypes_vis = sorted(vis_df["archetype"].unique())
                    palette = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel1
                    arch_counts = vis_df["archetype"].value_counts()

                    for i, arch in enumerate(archetypes_vis):
                        adf = vis_df[vis_df["archetype"] == arch]
                        n = arch_counts.get(arch, len(adf))
                        is_small = ARCHETYPE_INFO.get(arch, {}).get("small_sample", False) or n < 5
                        # 범례에 선수 수 표시, 소규모 경고 마킹
                        legend_name = f"{arch} ({n}명)" + (" ⚠️" if is_small else "")
                        fig.add_trace(go.Scatter(
                            x=adf[x_col].tolist(), y=adf[y_col].tolist(),
                            mode="markers",
                            name=legend_name,
                            marker=dict(
                                size=8, opacity=0.8,
                                color=palette[i % len(palette)],
                                symbol="x" if is_small else "circle",  # 소규모는 X마커
                            ),
                            text=adf["player"].tolist(),
                            customdata=np.column_stack([
                                adf["team"].fillna("N/A").values,
                                adf["position"].fillna(adf["pos_group"]).values,
                                adf["goals_p90"].fillna(0).values,
                                adf["assists_p90"].fillna(0).values,
                                adf["tackles_p90"].fillna(0).values,
                                adf["key_passes_p90"].fillna(0).values,
                            ]).tolist(),
                            hovertemplate=(
                                "<b>%{text}</b><br>"
                                "팀: %{customdata[0]}<br>"
                                "세부포지션: %{customdata[1]}<br>"
                                "골/90분: %{customdata[2]:.2f}<br>"
                                "어시스트/90분: %{customdata[3]:.2f}<br>"
                                "태클/90분: %{customdata[4]:.2f}<br>"
                                "키패스/90분: %{customdata[5]:.2f}<br>"
                                "<extra>%{fullData.name}</extra>"
                            ),
                        ))

                    # PCA 축 의미 설명 (우측 하단 annotation)
                    if use_pca:
                        x_label = "← 수비 지향 ─────── PCA 1 (공수 성향 축) ─────── 공격 지향 →"
                        y_label = "PCA 2 (스타일 다양성 축)"
                    else:
                        x_label = "UMAP X (플레이 스타일 공간)"
                        y_label = "UMAP Y (플레이 스타일 공간)"

                    fig.update_layout(
                        height=580, plot_bgcolor="#1a1a2e",
                        paper_bgcolor="#0d0d1a",
                        font_color="#ffffff",
                        margin=dict(l=10, r=10, t=30, b=10),
                        legend=dict(
                            font=dict(size=9), title="아키타입 (선수 수) | ⚠️=소규모",
                            itemsizing="constant",
                            bgcolor="rgba(26,26,46,0.9)",
                        ),
                        xaxis=dict(title=x_label, gridcolor="#333"),
                        yaxis=dict(title=y_label, gridcolor="#333"),
                        title=dict(
                            text=f"{sel_season} 시즌 플레이 스타일 지도 ({len(vis_df)}명)",
                            font=dict(size=13),
                        ),
                    )
                    st.plotly_chart(fig, use_container_width=True, theme=None)

            st.caption(
                "💡 **스카우터 포인트**: 지도에서 가까운 점 = 비슷한 플레이 스타일. "
                "영입 타겟과 같은 클러스터 안의 선수들이 실질적인 대체재입니다. "
                "(v4: 포지션별 분리 클러스터링 — FW끼리, MID끼리, DEF끼리만 비교)"
            )

            st.markdown("---")

            # 선수 검색 — 시즌 → 구단 → 선수 순 필터
            st.markdown("### 유사 선수 검색")
            st.caption("시즌 → 구단 → 선수 순으로 선택하면 같은 아키타입 내에서 플레이 스타일이 가장 유사한 선수를 찾아드립니다.")

            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                search_season = st.selectbox("시즌 선택", seasons, key="search_season")
            search_base = clusters[clusters["season"] == search_season].copy()
            with sc2:
                team_list = ["전체 구단"] + sorted(search_base["team"].dropna().unique().tolist())
                search_team = st.selectbox("구단 선택", team_list, key="search_team")
            if search_team != "전체 구단":
                search_base = search_base[search_base["team"] == search_team]
            with sc3:
                player_list = sorted(search_base["player"].dropna().unique().tolist())
                selected_player = st.selectbox(
                    "선수 선택",
                    player_list if player_list else ["(선수 없음)"],
                    key="sim_player",
                )
            selected_player = selected_player if player_list else None

            if selected_player and selected_player != "(선수 없음)":
                # 선택 시즌 전체에서 선수 정보 조회 (구단 필터 적용 전 전체 기준, 병합 적용)
                season_all = clusters[clusters["season"] == search_season].copy()
                season_all = _fix_pos_archetype(season_all)
                season_all["archetype"] = season_all["archetype"].map(lambda x: ARCHETYPE_MERGE.get(x, x))
                player_matches = season_all[season_all["player"] == selected_player]
                if len(player_matches) == 0:
                    st.warning("해당 선수의 데이터를 찾을 수 없습니다.")
                else:
                    player_row = player_matches.iloc[0]
                    player_arch = player_row["archetype"]
                    arch_info = ARCHETYPE_INFO.get(player_arch, {})

                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        st.markdown(
                            f"**{selected_player}** | {player_row.get('team', 'N/A')} "
                            f"| {search_season} 시즌 "
                            f"| 아키타입: **{arch_info.get('icon', '')} {player_arch}** ({arch_info.get('kor', '')})"
                        )
                    with col_b:
                        if arch_info:
                            st.info(f"🔍 {arch_info.get('scout_tip', '')}")

                    # 같은 아키타입 선수 — 전 리그 기준(cdf 포지션 필터 무관)
                    same_arch = season_all[
                        (season_all["archetype"] == player_arch) &
                        (season_all["player"] != selected_player)
                    ].copy()

                    if "pca_x" in same_arch.columns:
                        px_val = player_row["pca_x"]
                        py_val = player_row["pca_y"]
                        same_arch["distance"] = np.sqrt(
                            (same_arch["pca_x"] - px_val)**2 +
                            (same_arch["pca_y"] - py_val)**2
                        )
                        same_arch = same_arch.sort_values("distance")

                    # 유사 선수 카드 렌더링 (상위 15명)
                    top_similar = same_arch.head(15).copy()
                    st.markdown("#### 유사 선수 목록")
                    for _, row in top_similar.iterrows():
                        p_name  = row.get("player", "")
                        p_team  = row.get("team", "")
                        p_dist  = row.get("distance", None)
                        mv_raw  = row.get("market_value_raw", None)
                        g90     = row.get("goals_p90", None)
                        a90     = row.get("assists_p90", None)

                        # 추가 정보 문자열 조립
                        extra_parts = []
                        if pd.notna(mv_raw) if mv_raw is not None else False:
                            mv_str = f"€{mv_raw/1e6:.1f}M" if mv_raw >= 1e6 else f"€{mv_raw/1e3:.0f}K"
                            extra_parts.append(f"시장가치 {mv_str}")
                        if g90 is not None and pd.notna(g90):
                            extra_parts.append(f"골 {g90:.2f}/90분")
                        if a90 is not None and pd.notna(a90):
                            extra_parts.append(f"어시 {a90:.2f}/90분")
                        extra_str = " | ".join(extra_parts)

                        # 유사도를 similarity 인자로 변환 (거리 → 유사도: 1/(1+dist))
                        sim_val = 1.0 / (1.0 + p_dist) if p_dist is not None and pd.notna(p_dist) else None

                        render_player_card(
                            player_name=p_name,
                            team=p_team,
                            archetype=player_arch,
                            similarity=sim_val,
                            extra_info=extra_str,
                        )
                        st.markdown('<hr style="margin:4px 0; border-color:#eee;">', unsafe_allow_html=True)
                    st.caption("💡 유사도가 높을수록 플레이 스타일이 유사합니다. 90% 이상이면 사실상 동일 유형.")

    # ══════════════════════════════════════════
    # TAB 2: 성장 레퍼런스 (S4)
    # ══════════════════════════════════════════
    with tab2:
        st.subheader("S4. 성장 곡선 & 피크 연령 레퍼런스")
        with st.expander("📖 S4 성장 레퍼런스 사용법 (스카우터 필독)", expanded=False):
            st.markdown("""
**핵심 지표:**
- **피크 연령**: 해당 포지션 선수들의 평균 최고 성과 연령
- **성과 곡선**: 같은 나이대 선수들 중 어느 위치에 있는지 파악 (중앙값 = 리그 평균, 상위 10% = 탑클래스)

**Buy/Sell 시그널 판단 기준:**
| 상태 | 조건 | 행동 |
|------|------|------|
| **Buy Signal** | 피크 연령 - 2세 이하 | 성장 잠재력 → 지금 사면 앞으로 오름 |
| **Hold** | 피크 연령 ± 1세 | 최전성기 → 가격 고점, 보유 유지 |
| **Sell Signal** | 피크 연령 + 2세 이상 | 하락 국면 시작 → 지금이 최고 매도 시점 |

**연령별 성과 곡선 읽는 법:**
- 초록 실선(중앙값) 기준 아래이면 동나이대 하위 절반
- 분홍 점선(탑클래스 기준선) 근처이면 동나이대 상위 10% — 즉각 영입 검토 대상
            """)
        ref = load_s4_reference()
        if not ref:
            st.warning("s4_reference_profiles.json 데이터 없음")
        else:
            # 피크 연령 카드
            if "peak_ages" in ref:
                st.markdown("### 포지션별 피크 연령")
                peaks = ref["peak_ages"]
                cols = st.columns(4)
                for i, (pos, info) in enumerate(peaks.items()):
                    with cols[i]:
                        peak = info.get("epl_data") or info.get("peak_age") or info.get("smoothed_peak_age", "?")
                        lit = info.get("literature", "")
                        pos_kor = info.get("pos_kor", pos)
                        delta_str = f"문헌 기준: {lit}세" if lit else ""
                        st.metric(f"{pos} ({pos_kor})", f"{peak}세", delta_str)
                st.caption(
                    "💡 **스카우터 포인트**: 피크 연령보다 2살 이상 어린 선수를 영입하면 성장 프리미엄을 기대할 수 있습니다. "
                    "피크 이후 선수를 영입할 때는 S6 하락 감지 탭과 반드시 크로스 체크하세요."
                )

            # 연령별 성과 퍼센타일 차트
            if "age_performance_percentiles" in ref:
                st.markdown("### 연령별 성과 곡선")
                sel_pos_growth = st.selectbox(
                    "포지션 선택",
                    list(ref["age_performance_percentiles"].keys()),
                    key="growth_pos"
                )
                pdata = ref["age_performance_percentiles"][sel_pos_growth]

                stat_options = {
                    "FW": {"gls_p90": "골/90분", "ast_p90": "어시스트/90분", "sh_p90": "슈팅/90분"},
                    "MF": {"ast_p90": "어시스트/90분", "gls_p90": "골/90분", "tklw_p90": "태클성공/90분"},
                    "DF": {"tklw_p90": "태클성공/90분", "int_p90": "인터셉트/90분"},
                    "GK": {"gls_p90": "실점", "int_p90": "배급"},
                }
                pos_stats = stat_options.get(sel_pos_growth, {"gls_p90": "골/90분"})
                sel_stat = st.selectbox(
                    "지표", list(pos_stats.keys()),
                    format_func=lambda x: pos_stats[x], key="growth_stat"
                )

                ages = [p["age"] for p in pdata]
                p25 = [p.get(f"{sel_stat}_p25", 0) for p in pdata]
                p50 = [p.get(f"{sel_stat}_p50", 0) for p in pdata]
                p75 = [p.get(f"{sel_stat}_p75", 0) for p in pdata]
                p90 = [p.get(f"{sel_stat}_p90", 0) for p in pdata]

                # 피크 연령 표시용
                peak_age_val = None
                if "peak_ages" in ref and sel_pos_growth in ref["peak_ages"]:
                    pa = ref["peak_ages"][sel_pos_growth]
                    peak_age_val = pa.get("epl_data") or pa.get("peak_age") or pa.get("smoothed_peak_age")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ages, y=p90, name="90번째 백분위 (상위 10%)", line=dict(color=EPL_MAGENTA, width=1, dash="dot")))
                fig.add_trace(go.Scatter(x=ages, y=p75, name="75번째 백분위 (상위 25%)", line=dict(color=EPL_CYAN, width=1.5)))
                fig.add_trace(go.Scatter(x=ages, y=p50, name="중앙값 (50번째)", line=dict(color=EPL_GREEN, width=3)))
                fig.add_trace(go.Scatter(x=ages, y=p25, name="25번째 백분위 (하위 25%)", line=dict(color="#aaa", width=1)))

                if peak_age_val:
                    fig.add_vline(
                        x=peak_age_val, line_dash="dash", line_color=EPL_PURPLE,
                        annotation_text=f"피크 {peak_age_val}세",
                        annotation_position="top right",
                    )
                fig.update_layout(
                    xaxis_title="나이",
                    yaxis_title=pos_stats[sel_stat],
                    height=420, plot_bgcolor="#1a1a2e", paper_bgcolor="#0d0d1a", font_color="#ffffff",
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend_title="백분위 구간",
                )
                st.plotly_chart(fig, use_container_width=True, theme=None)
                st.caption(
                    "💡 **스카우터 포인트**: 보라색 수직선이 피크 연령. "
                    "영입 타겟이 피크선 왼쪽(성장기) → Buy, 피크선 오른쪽(하락기) → 하락 리스크 주의. "
                    "현재 성과가 75번째 백분위 이상이면 해당 나이대 탑클래스입니다."
                )

            # 스쿼드 분석
            if "current_squad_profiles" in ref:
                st.markdown("### 스쿼드 분석 (매수/매도 시그널)")
                st.caption("팀 스쿼드의 연령 구조와 피크 연령 대비 현재 위치를 분석해 즉각적인 Buy/Sell 시그널을 제공합니다.")
                teams = sorted(ref["current_squad_profiles"].keys())
                sel_team = st.selectbox("팀 선택", teams, key="squad_team")
                squad = ref["current_squad_profiles"][sel_team]

                if isinstance(squad, dict):
                    scol1, scol2 = st.columns(2)
                    with scol1:
                        st.metric("스쿼드 선수 수", squad.get("n_players", "?"))
                    with scol2:
                        avg = squad.get("avg_age", "?")
                        st.metric("평균 연령", f"{avg:.1f}세" if isinstance(avg, (int, float)) else avg)

                    buy = squad.get("buy_signals", [])
                    if buy:
                        st.markdown("**🟢 매수 시그널** — 피크 전, 성장 잠재력 높은 선수 (지금 팔면 손해)")
                        buy_df = pd.DataFrame(buy)
                        if "market_value" in buy_df.columns:
                            buy_df["market_value"] = buy_df["market_value"].apply(
                                lambda x: f"€{x/1e6:.1f}M" if pd.notna(x) and x >= 1e6 else f"€{x/1e3:.0f}K" if pd.notna(x) else "N/A"
                            )
                        buy_df = _rename_cols(buy_df)
                        st.dataframe(buy_df, use_container_width=True, hide_index=True)

                    sell = squad.get("sell_signals", [])
                    if sell:
                        st.markdown("**🔴 매도 시그널** — 피크 이후, 지금이 최고 매도가 (영입 제안 받으면 검토)")
                        sell_df = pd.DataFrame(sell)
                        if "market_value" in sell_df.columns:
                            sell_df["market_value"] = sell_df["market_value"].apply(
                                lambda x: f"€{x/1e6:.1f}M" if pd.notna(x) and x >= 1e6 else f"€{x/1e3:.0f}K" if pd.notna(x) else "N/A"
                            )
                        sell_df = _rename_cols(sell_df)
                        st.dataframe(sell_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── 선수별 성장 예측 (P7 모델) ──────────────────────────────────────
        st.markdown("### 선수별 성장 예측 (P7 모델)")
        st.caption("P7 모델이 예측한 선수별 향후 3시즌 성과와 전성기 타이밍을 확인합니다.")

        growth_df = load_growth_predictions()
        if growth_df.empty:
            st.info("growth_predictions.parquet 데이터가 없습니다. P7 모델을 먼저 실행하세요.")
        else:
            g_col1, g_col2, g_col3 = st.columns(3)

            # 시즌 필터 (컬럼이 있을 때만)
            if "season" in growth_df.columns:
                with g_col1:
                    g_seasons = sorted(growth_df["season"].dropna().unique().tolist(), reverse=True)
                    sel_g_season = st.selectbox("시즌 선택", g_seasons, key="p7_season")
                g_filtered = growth_df[growth_df["season"] == sel_g_season].copy()
            else:
                sel_g_season = None
                g_filtered = growth_df.copy()
                with g_col1:
                    st.write("(시즌 정보 없음)")

            # 구단 필터 (컬럼이 있을 때만)
            if "team" in g_filtered.columns:
                with g_col2:
                    g_teams = ["전체 구단"] + sorted(g_filtered["team"].dropna().unique().tolist())
                    sel_g_team = st.selectbox("구단 선택", g_teams, key="p7_team")
                if sel_g_team != "전체 구단":
                    g_filtered = g_filtered[g_filtered["team"] == sel_g_team]
            else:
                with g_col2:
                    st.write("(구단 정보 없음)")

            # 선수 선택
            with g_col3:
                g_players = sorted(g_filtered["player"].dropna().unique().tolist())
                if g_players:
                    sel_g_player = st.selectbox("선수 선택", g_players, key="p7_player")
                else:
                    sel_g_player = None
                    st.info("조건에 해당하는 선수가 없습니다.")

            if sel_g_player:
                player_growth = g_filtered[g_filtered["player"] == sel_g_player]
                if len(player_growth) == 0:
                    st.warning(f"{sel_g_player}의 성장 예측 데이터가 없습니다.")
                else:
                    pr = player_growth.iloc[0]
                    current_age = pr.get("current_age", None)
                    peak_age = pr.get("peak_age", None)
                    decline_start_age = pr.get("decline_start_age", None)
                    pred1 = pr.get("pred_next1", None)
                    pred2 = pr.get("pred_next2", None)
                    pred3 = pr.get("pred_next3", None)

                    # 지표 카드
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("현재 나이", f"{int(current_age)}세" if pd.notna(current_age) else "N/A")
                    with m2:
                        st.metric("전성기 나이 (P7)", f"{int(peak_age)}세" if pd.notna(peak_age) else "N/A")
                    with m3:
                        st.metric("하락 시작 나이", f"{int(decline_start_age)}세" if pd.notna(decline_start_age) else "N/A")

                    # 스카우터 팁
                    if pd.notna(current_age) and pd.notna(peak_age):
                        seasons_to_peak = int(peak_age) - int(current_age)
                        if seasons_to_peak > 0:
                            st.success(f"전성기까지 약 {seasons_to_peak}시즌 남음 — 지금 영입하면 성장 프리미엄 기대 가능")
                        elif seasons_to_peak == 0:
                            st.warning("현재 전성기 시점 — 지금이 최고 가치. 고점 매도 또는 즉각 영입 검토")
                        else:
                            st.error(f"전성기 {abs(seasons_to_peak)}시즌 경과 — 이미 하락 시작. 계약 기간 및 이적료 재검토 권장")

                    # 향후 3시즌 예측 막대 차트
                    pred_vals = []
                    pred_labels = []
                    for label, val in [("내 시즌 +1", pred1), ("내 시즌 +2", pred2), ("내 시즌 +3", pred3)]:
                        if val is not None and pd.notna(val):
                            pred_labels.append(label)
                            pred_vals.append(float(val))

                    if pred_vals:
                        pred_fig = go.Figure(go.Bar(
                            x=pred_labels,
                            y=pred_vals,
                            marker_color=[EPL_GREEN, EPL_CYAN, EPL_MAGENTA],
                            text=[f"{v:.2f}" for v in pred_vals],
                            textposition="outside",
                        ))
                        pred_fig.update_layout(
                            title=f"{sel_g_player} — 향후 3시즌 예측 성과 (AC Z-score 기준)",
                            yaxis_title="예측 성과 지수",
                            height=350,
                            margin=dict(t=40, b=20),
                            plot_bgcolor="#1a1a2e", paper_bgcolor="#0d0d1a", font_color="#ffffff",
                        )
                        st.plotly_chart(pred_fig, use_container_width=True, theme=None)
                    else:
                        st.info("향후 예측값 데이터가 없습니다.")

        st.markdown("---")

        # ── 🔥 고신뢰도 영입 쇼트리스트: P7(Improving) ∩ S2(저평가) ∩ 나이≤26 ──
        st.markdown("### 🔥 고신뢰도 영입 쇼트리스트")
        st.caption("P7 성장 예측(Improving) + S2 저평가 + 나이 ≤ 26 삼박자 충족 선수 — 가장 확신도 높은 영입 후보.")

        _g_short = load_growth_predictions_v4()
        _s2_short = load_undervalued()
        if not _g_short.empty and not _s2_short.empty:
            # P7 Improving 선수
            _improving = set(_g_short[_g_short["pred_xgb"] == "Improving"]["player"].tolist())
            # S2 저평가 선수
            _underval  = set(_s2_short["player"].tolist()) if "player" in _s2_short.columns else set()
            # 교집합 + 나이 ≤ 26
            _short = _g_short[
                _g_short["player"].isin(_improving & _underval) &
                (_g_short["age"] <= 26)
            ].copy()
            # WAR 병합
            _ratings_sh = load_scout_ratings()
            if not _ratings_sh.empty and "war" in _ratings_sh.columns:
                _war_sh = _ratings_sh.sort_values("war", ascending=False).drop_duplicates("player")[["player","war"]]
                _short = _short.merge(_war_sh, on="player", how="left")
            # S2 가치비율 병합
            _mv_col = [c for c in _s2_short.columns if "ratio" in c or "value_ratio" in c]
            if _mv_col and "player" in _s2_short.columns:
                _short = _short.merge(_s2_short[["player", _mv_col[0]]].drop_duplicates("player"), on="player", how="left")

            if _short.empty:
                st.info("현재 조건을 모두 충족하는 선수가 없습니다. (나이 ≤ 26 조건 완화 필요할 수 있음)")
            else:
                _short = _short.sort_values("war", ascending=False) if "war" in _short.columns else _short
                _sh_cols = ["player","team","pos_simple","age","prob_improving","war"] + (_mv_col[:1] if _mv_col else [])
                _sh_cols = [c for c in _sh_cols if c in _short.columns]
                _sh_disp = _short[_sh_cols].copy()
                _sh_rename = {"player":"선수","team":"팀","pos_simple":"포지션","age":"나이",
                              "prob_improving":"개선확률","war":"WAR"}
                if _mv_col:
                    _sh_rename[_mv_col[0]] = "가치비율"
                _sh_disp = _sh_disp.rename(columns=_sh_rename)
                if "포지션" in _sh_disp.columns:
                    _sh_disp["포지션"] = _sh_disp["포지션"].replace({"MF":"MID","DF":"DEF"})
                if "개선확률" in _sh_disp.columns:
                    _sh_disp["개선확률"] = _sh_disp["개선확률"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
                if "WAR" in _sh_disp.columns:
                    _sh_disp["WAR"] = _sh_disp["WAR"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
                if "가치비율" in _sh_disp.columns:
                    _sh_disp["가치비율"] = _sh_disp["가치비율"].apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "-")

                st.success(f"✅ {len(_sh_disp)}명 발굴 — P7 성장 + S2 저평가 + 나이≤26 동시 충족")
                st.dataframe(_sh_disp, use_container_width=True, hide_index=True)
        else:
            st.info("P7 또는 S2 데이터가 없습니다.")

        st.markdown("---")

        # ── P7 성장 분류 예측 (2024/25) ──────────────────────────────────────
        st.markdown("### 🔬 P7 성장 분류 예측 (2024/25)")
        st.caption("P7 v4 XGBoost 분류 모델이 예측한 2024/25 시즌 선수별 성장 궤적 (Improving / Stable / Declining).")

        growth_v4_df = load_growth_predictions_v4()
        if growth_v4_df.empty:
            st.info("P7 성장 예측 데이터가 없습니다.")
        else:
            # ── WAR + 시장가치 평균 병합 (scout_ratings) ──
            gv4_base = growth_v4_df.copy()
            ratings_df = load_scout_ratings()

            # 탑클래스 판별: 최근 3시즌 평균 WAR ≥ 93 + 평균 MV ≥ 50M + 2시즌 이상
            ELITE_WAR_THRESHOLD = 93.0
            ELITE_MV_THRESHOLD  = 50_000_000
            elite_players: set = set()
            if not ratings_df.empty:
                recent_szns = sorted(ratings_df["season"].unique())[-3:]
                recent_r    = ratings_df[ratings_df["season"].isin(recent_szns)]
                agg_cols = {"war": "mean", "market_value": "mean", "season": "count"}
                agg_cols = {k: v for k, v in agg_cols.items() if k in recent_r.columns}
                elite_agg = (
                    recent_r.groupby("player")
                    .agg(**{
                        "avg_war": ("war", "mean"),
                        "avg_mv":  ("market_value", "mean"),
                        "n_szn":   ("season", "count"),
                    })
                    .reset_index()
                )
                elite_players = set(
                    elite_agg.loc[
                        (elite_agg["avg_war"] >= ELITE_WAR_THRESHOLD) &
                        (elite_agg["avg_mv"]  >= ELITE_MV_THRESHOLD)  &
                        (elite_agg["n_szn"]   >= 2),
                        "player"
                    ]
                )

                war_ref = (
                    ratings_df.sort_values("war", ascending=False)
                    .drop_duplicates("player", keep="first")
                    [["player", "war"]]
                )
                gv4_base = gv4_base.merge(war_ref, on="player", how="left")

            # ── 분류 레이블 계산 (필터 전) ──
            # 탑클래스는 XGBoost 예측 서브분류 포함: "👑 Improving" / "👑 Stable" / "👑 하락신호"
            ELITE_SUB = {"Improving": "👑 Improving", "Stable": "👑 Stable", "Declining": "👑 하락신호"}

            def _classify(row) -> str:
                xgb = row.get("pred_xgb", "")
                rf  = row.get("pred_rf",  "")
                if row.get("player") in elite_players:
                    return ELITE_SUB.get(xgb, "👑 탑클래스")
                # 두 모델이 서로 다른 예측 → 불확실
                if xgb and rf and xgb != rf:
                    return "🔘 불확실"
                return xgb  # Improving / Stable / Declining

            gv4_base["분류"] = gv4_base.apply(_classify, axis=1)

            # ── 필터 UI ──
            pos_col_v4 = "pos_simple" if "pos_simple" in gv4_base.columns else (
                "pos_group" if "pos_group" in gv4_base.columns else None
            )
            team_list_v4 = ["전체"] + sorted(gv4_base["team"].dropna().unique().tolist()) \
                if "team" in gv4_base.columns else ["전체"]

            fc1, fc2, fc3 = st.columns(3)
            sel_pos_v4  = fc1.selectbox("포지션", ["전체", "FW", "MF", "DF", "GK"], key="p7v4_pos")
            sel_team_v4 = fc2.selectbox("팀",     team_list_v4,                        key="p7v4_team")
            sel_cat_v4  = fc3.selectbox(
                "성장 분류",
                ["전체", "👑 Improving", "👑 Stable", "👑 하락신호",
                 "🟢 Improving", "🟡 Stable", "🔴 Declining", "🔘 불확실"],
                key="p7v4_cat",
            )

            # ── 필터 적용 ──
            gv4 = gv4_base.copy()
            if sel_pos_v4 != "전체" and pos_col_v4:
                gv4 = gv4[gv4[pos_col_v4] == sel_pos_v4]
            if sel_team_v4 != "전체" and "team" in gv4.columns:
                gv4 = gv4[gv4["team"] == sel_team_v4]
            cat_filter_map = {
                "👑 Improving":  "👑 Improving",
                "👑 Stable":     "👑 Stable",
                "👑 하락신호":   "👑 하락신호",
                "🟢 Improving":  "Improving",
                "🟡 Stable":     "Stable",
                "🔴 Declining":  "Declining",
                "🔘 불확실":    "🔘 불확실",
            }
            if sel_cat_v4 != "전체":
                gv4 = gv4[gv4["분류"] == cat_filter_map[sel_cat_v4]]

            # ── 요약 메트릭 카드 ──
            CAT_COLOR = {
                "Improving": EPL_GREEN, "Stable": "#FFD700", "Declining": EPL_MAGENTA,
                "👑 Improving": "#00C878", "👑 Stable": "#C0A000", "👑 하락신호": "#FF6B35",
                "🔘 불확실": "#888888",
            }
            cnt_total     = len(gv4)
            cnt_e_imp     = (gv4["분류"] == "👑 Improving").sum()
            cnt_e_stb     = (gv4["분류"] == "👑 Stable").sum()
            cnt_e_dec     = (gv4["분류"] == "👑 하락신호").sum()
            cnt_i         = (gv4["분류"] == "Improving").sum()
            cnt_s         = (gv4["분류"] == "Stable").sum()
            cnt_d         = (gv4["분류"] == "Declining").sum()
            cnt_unc       = (gv4["분류"] == "🔘 불확실").sum()

            mc1, mc2, mc3, mc4, mc5, mc6, mc7, mc8 = st.columns(8)
            mc1.metric("전체",          cnt_total)
            mc2.metric("👑 Improving",  cnt_e_imp, help="탑클래스 + 성장 중")
            mc3.metric("👑 Stable",     cnt_e_stb, help="탑클래스 + 안정화")
            mc4.metric("👑 하락신호",   cnt_e_dec, help="탑클래스지만 하락 예측 — 재계약 주의")
            mc5.metric("🟢 Improving",  cnt_i)
            mc6.metric("🟡 Stable",     cnt_s)
            mc7.metric("🔴 Declining",  cnt_d)
            mc8.metric("🔘 불확실",     cnt_unc, help="XGBoost↔RF 예측 불일치")

            # 탑클래스 선수 강조 배너
            if elite_players:
                elite_in_view = gv4_base[gv4_base["player"].isin(elite_players)][["player", "분류"]].drop_duplicates()
                if not elite_in_view.empty:
                    elite_badges = []
                    for _, erow in elite_in_view.iterrows():
                        sub_icon = {"👑 Improving": "🟢", "👑 Stable": "🟡", "👑 하락신호": "🔴"}.get(erow["분류"], "")
                        elite_badges.append(f"{sub_icon} {erow['player']} ({erow['분류']})")
                    st.markdown(
                        f"<div style='background:#1a1200;border-left:4px solid #C0A000;"
                        f"padding:8px 12px;border-radius:6px;margin:6px 0;'>"
                        f"<span style='color:#C0A000;font-weight:700;'>👑 탑클래스 ({len(elite_in_view)}명)</span>"
                        f"<span style='color:#e0e0e0;font-size:0.88em;'> — "
                        f"{' &nbsp;|&nbsp; '.join(elite_badges)}</span></div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("---")

            # ── 산점도: 나이 vs 개선 확률 ──
            if "prob_improving" in gv4.columns and "age" in gv4.columns and not gv4.empty:
                scatter_data = gv4.copy()
                scatter_data["_size"] = scatter_data["war"].fillna(20).clip(10, 100) \
                    if "war" in scatter_data.columns else 20

                def _hover_text(r):
                    war_str = f"{r['war']:.1f}" if pd.notna(r.get("war")) else "N/A"
                    return (
                        f"<b>{r['player']}</b>  {r['분류']}<br>"
                        f"팀: {r.get('team','N/A')} | 나이: {r.get('age','N/A')}<br>"
                        f"개선확률: {r.get('prob_improving',0):.1%} | "
                        f"하락확률: {r.get('prob_declining',0):.1%}<br>"
                        f"WAR: {war_str}"
                    )
                scatter_data["_hover"] = scatter_data.apply(_hover_text, axis=1)

                # 카테고리별 색상 + 아이콘
                PLOT_CATS = [
                    ("👑 Improving",  "#00C878",  "star"),
                    ("👑 Stable",     "#C0A000",  "star"),
                    ("👑 하락신호",   "#FF6B35",  "star"),
                    ("Improving",     EPL_GREEN,  "circle"),
                    ("Stable",        "#FFD700",  "circle"),
                    ("Declining",     EPL_MAGENTA,"circle"),
                    ("🔘 불확실",     "#888888",  "diamond"),
                ]
                # 탑클래스 + WAR 상위 10명만 이름 표시 (라벨 겹침 방지)
                war_top10 = set(
                    scatter_data.nlargest(10, "war")["player"].tolist()
                    if "war" in scatter_data.columns else []
                )
                elite_set = set(elite_players)

                sc_fig = go.Figure()
                for cat_label, color, symbol in PLOT_CATS:
                    sub = scatter_data[scatter_data["분류"] == cat_label].copy()
                    if sub.empty:
                        continue
                    # 라벨: 탑클래스 or WAR 상위 10명만, 나머지 빈 문자열
                    sub["_label"] = sub["player"].apply(
                        lambda p: p if (p in elite_set or p in war_top10) else ""
                    )
                    sc_fig.add_trace(go.Scatter(
                        x=sub["age"].tolist(),
                        y=sub["prob_improving"].tolist(),
                        mode="markers+text",
                        name=cat_label,
                        marker=dict(
                            color=color, symbol=symbol,
                            size=sub["_size"].tolist(),
                            opacity=0.88,
                            line=dict(width=1, color="rgba(255,255,255,0.3)"),
                        ),
                        text=sub["_label"].tolist(),
                        textposition="top center",
                        textfont=dict(size=9, color=color),
                        hovertext=sub["_hover"].tolist(),
                        hovertemplate="%{hovertext}<extra></extra>",
                    ))

                sc_fig.update_layout(
                    title=dict(text="나이 vs 개선 확률  (버블 크기 = WAR | ★ = 탑클래스 | 이름 = WAR 상위 10)", font=dict(size=13)),
                    xaxis=dict(title="나이", gridcolor="#333"),
                    yaxis=dict(
                        title="개선 확률", tickformat=".0%", gridcolor="#333",
                        range=[0, 1.05],  # 0~100% 고정 (음수 제거)
                    ),
                    height=440,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                    plot_bgcolor="#0e1117",
                    paper_bgcolor="#0e1117",
                    font=dict(color="#e0e0e0"),
                    margin=dict(t=60, b=40, l=40, r=20),
                )
                sc_fig.add_hline(
                    y=0.5, line_dash="dot", line_color="#555",
                    annotation_text="50% 기준선", annotation_font_color="#888",
                    annotation_position="bottom right",
                )
                sc_fig.add_hrect(y0=0.4, y1=0.6, fillcolor="#888888", opacity=0.07,
                                  annotation_text="불확실 구간", annotation_font_color="#888",
                                  annotation_position="top left")
                st.plotly_chart(sc_fig, use_container_width=True, theme=None)

            # ── 단일 테이블 ──
            if gv4.empty:
                st.info("해당 조건에 맞는 선수가 없습니다.")
            else:
                tbl = gv4.copy()

                # 분류 아이콘 최종 매핑
                disp_map = {
                    "👑 Improving":  "👑🟢 Improving",
                    "👑 Stable":     "👑🟡 Stable",
                    "👑 하락신호":   "👑🔴 하락신호",
                    "Improving":     "🟢 Improving",
                    "Stable":        "🟡 Stable",
                    "Declining":     "🔴 Declining",
                    "🔘 불확실":    "🔘 불확실",
                }
                tbl["분류"] = tbl["분류"].map(lambda x: disp_map.get(x, x))

                if "injury_risk" in tbl.columns:
                    tbl["injury_risk"] = tbl["injury_risk"].apply(
                        lambda x: "🔴" if x == 1 else "🟢"
                    )

                # WAR 기준 내림차순 정렬
                sort_col = "war" if "war" in tbl.columns else ("prob_improving" if "prob_improving" in tbl.columns else None)
                if sort_col:
                    tbl = tbl.sort_values(sort_col, ascending=False)

                _tbl_cols = ["player", "team", pos_col_v4, "age", "분류",
                             "prob_improving", "prob_stable", "prob_declining",
                             "war", "injury_risk"]
                _tbl_cols = [c for c in _tbl_cols if c and c in tbl.columns]
                tbl = tbl[_tbl_cols].copy()

                _rename = {
                    "player": "선수", "team": "팀", "age": "나이",
                    "pos_simple": "포지션", "pos_group": "포지션",
                    "분류": "성장 분류",
                    "prob_improving": "개선확률", "prob_stable": "유지확률",
                    "prob_declining": "하락확률",
                    "war": "WAR", "injury_risk": "부상",
                }
                tbl = tbl.rename(columns={k: v for k, v in _rename.items() if k in tbl.columns})
                # 포지션 표시명 통일 (MF→MID, DF→DEF)
                if "포지션" in tbl.columns:
                    tbl["포지션"] = tbl["포지션"].replace({"MF": "MID", "DF": "DEF"})

                for col in ["개선확률", "유지확률", "하락확률"]:
                    if col in tbl.columns:
                        tbl[col] = tbl[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
                if "WAR" in tbl.columns:
                    tbl["WAR"] = tbl["WAR"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")

                st.dataframe(tbl, use_container_width=True, hide_index=True)
                st.caption(f"📋 총 **{len(tbl)}명** | 성장 분류 기준: XGBoost + RF 앙상블 예측")

                # ── 분류별 스카우트 액션 가이드 ──
                with st.expander("📋 분류별 스카우트 액션 가이드", expanded=False):
                    st.markdown("""
| 분류 | 스카우트 액션 | 타이밍 |
|------|-------------|--------|
| 👑🟢 **탑클래스 Improving** | 계약 연장 최우선 협의 / 타 구단 접촉 즉시 차단 | 즉시 |
| 👑🟡 **탑클래스 Stable** | 현 계약 유지 / 2~3년 후 판매 검토 | 모니터링 |
| 👑🔴 **탑클래스 하락신호** | 재계약 조건 재검토 / 보유 vs 판매 협의 시작 | 이번 시즌 내 |
| 🟢 **Improving** | 영입 타겟 리스트 등록 / S2 저평가 교차 확인 | 우선 검토 |
| 🟡 **Stable** | 보유 선수라면 유지 / 영입 대상이면 낮은 우선순위 | 다음 시즌 재평가 |
| 🔴 **Declining** | 보유 선수라면 이번 이적 시장 판매 / 영입 대상 제외 | 이번 여름 |
| 🔘 **불확실** | XGBoost↔RF 모델 의견 충돌 → 실제 경기 영상 추가 검토 | 신중 검토 |
                    """)
                    st.caption("💡 Improving + 나이 ≤ 24 + S2 저평가 선수는 상단 '🔥 고신뢰도 영입 쇼트리스트'에서 확인하세요.")

    # ══════════════════════════════════════════
    # TAB 3: 이적 리스크 (S5)
    # ══════════════════════════════════════════
    with tab3:
        st.subheader("S5. 이적 적응 리스크 예측")
        with st.expander("📖 S5 이적 리스크 모델 사용법 (스카우터 필독)", expanded=False):
            st.markdown("""
**S5 모델**: 2000~2025시즌 1,723건의 EPL 이적 데이터 분석 → 이적 후 첫 시즌 적응 성공/실패 예측

**핵심 변수:**
- **ELO 격차**: 이전 팀 vs 새 팀의 수준 차이 (클수록 적응 어려움)
- **스타일 매칭률**: 이전 팀과 새 팀의 전술 유사도 (높을수록 적응 쉬움)
- **이적 시 나이**: 28세 이상 대형 이적은 실패 확률 급상승
- **스텝업/스텝다운**: 상위 팀으로 올라갈수록(스텝업) 실패 확률 증가

**적응 결과 분류:**
| 결과 | 의미 | 기준 |
|------|------|------|
| **success** | 이적 후 성공 적응 | 주전 + 기대 이상 성과 |
| **partial** | 부분 적응 | 벤치 또는 기대치 근접 |
| **failure** | 적응 실패 | 주전 실패 또는 재이적 |

**영입 협상 활용법:**
- 실패 확률 50% 이상 → 이적료 20~30% 할인 요청 근거로 활용
- 스타일 매칭률 낮은 이적은 적응 기간(1~2시즌) 반드시 고려
- 스텝업 이적은 1년 임대 먼저 제안 → 위험 분산
            """)
        transfers = load_transfer_predictions()

        # ── 📋 2024/25 이적 선수 적응 예측 ──────────────────────────────────
        st.markdown("### 📋 2024/25 이적 선수 적응 예측")
        st.caption("2024/25 시즌 이적 선수의 첫 시즌 적응 성공/실패 예측 결과입니다.")

        if len(transfers) == 0:
            st.info("transfer_predictions_v3.parquet 데이터가 없습니다.")
        else:
            recent_transfers = transfers[transfers["season_new"] == "2024/25"].copy() if "season_new" in transfers.columns else pd.DataFrame()
            if recent_transfers.empty:
                st.info("2024/25 시즌 이적 데이터가 없습니다.")
            else:
                # prob_failure 기준 컬러 코딩
                def _transfer_risk_icon(prob_failure):
                    if pd.isna(prob_failure):
                        return "⚪"
                    if prob_failure >= 0.6:
                        return "🔴"
                    if prob_failure >= 0.4:
                        return "🟡"
                    return "🟢"

                recent_display = recent_transfers.copy()
                if "prob_failure" in recent_display.columns:
                    recent_display["risk_icon"] = recent_display["prob_failure"].apply(_transfer_risk_icon)

                # 표시 컬럼 선택 (있는 컬럼만)
                rt_show_cols = ["risk_icon", "player", "team_old", "team_new", "age_at_transfer",
                                "pred_label", "prob_success", "adapt_risk"]
                rt_available = [c for c in rt_show_cols if c in recent_display.columns]
                rt_disp = recent_display[rt_available].copy()

                # pred_label 한국어
                if "pred_label" in rt_disp.columns:
                    rt_disp["pred_label"] = rt_disp["pred_label"].map(
                        {"success": "성공", "partial": "부분 적응", "failure": "실패"}
                    ).fillna(rt_disp["pred_label"])

                if "prob_success" in rt_disp.columns:
                    rt_disp["prob_success"] = rt_disp["prob_success"].apply(
                        lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
                    )

                if "adapt_risk" in rt_disp.columns:
                    risk_icon_map_rt = {"high": "🔴 고위험", "medium": "🟡 중위험", "low": "🟢 저위험"}
                    rt_disp["adapt_risk"] = rt_disp["adapt_risk"].map(lambda x: risk_icon_map_rt.get(x, x))

                rt_rename = {
                    "risk_icon": "위험",
                    "player": "선수",
                    "team_old": "이전팀",
                    "team_new": "새팀",
                    "age_at_transfer": "이적시 나이",
                    "pred_label": "예측결과",
                    "prob_success": "성공확률",
                    "adapt_risk": "적응 리스크",
                }
                rt_disp = rt_disp.rename(columns={k: v for k, v in rt_rename.items() if k in rt_disp.columns})

                # prob_failure 기준 정렬 (위험 높은 순)
                if "prob_failure" in recent_display.columns:
                    sort_idx = recent_display["prob_failure"].sort_values(ascending=False).index
                    rt_disp = rt_disp.loc[sort_idx]

                st.dataframe(rt_disp, use_container_width=True, hide_index=True)
                st.caption(
                    "💡 🔴 실패 확률 60% 이상 | 🟡 40~60% | 🟢 40% 미만. "
                    "고위험 선수는 계약 전 임대 먼저 제안하거나 조건부 계약 구조를 검토하세요."
                )

        st.markdown("---")

        if len(transfers) == 0:
            st.warning("transfer_predictions_v3.parquet 데이터 없음")
        else:
            st.markdown(f"이적 후 적응 성공/실패 예측 — **{len(transfers):,}건** 이적 데이터 분석")

            tcol1, tcol2 = st.columns(2)
            with tcol1:
                t_seasons = sorted(transfers["season_new"].unique(), reverse=True)
                sel_t_season = st.selectbox("이적 시즌", ["전체"] + t_seasons, key="transfer_season")
            with tcol2:
                sel_result = st.selectbox(
                    "예측 결과",
                    ["전체", "failure", "partial", "success"],
                    format_func=lambda x: {"전체": "전체", "failure": "실패", "partial": "부분 적응", "success": "성공"}.get(x, x),
                    key="transfer_result"
                )

            tdf = transfers.copy()
            if sel_t_season != "전체":
                tdf = tdf[tdf["season_new"] == sel_t_season]
            if sel_result != "전체":
                tdf = tdf[tdf["pred_label"] == sel_result]

            # 실패 위험 높은 이적
            st.markdown("### 적응 실패 고위험 이적")
            high_risk = tdf.sort_values("prob_failure", ascending=False).head(20)
            show_cols = ["player", "team_old", "team_new", "season_new", "pos_group",
                         "age_at_transfer", "prob_failure", "prob_success",
                         "elo_gap", "style_match_pct", "pred_label"]
            available = [c for c in show_cols if c in high_risk.columns]
            disp = high_risk[available].copy()
            if "prob_failure" in disp.columns:
                disp["prob_failure"] = disp["prob_failure"].apply(lambda x: f"{x:.1%}")
            if "prob_success" in disp.columns:
                disp["prob_success"] = disp["prob_success"].apply(lambda x: f"{x:.1%}")
            if "style_match_pct" in disp.columns:
                disp["style_match_pct"] = disp["style_match_pct"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "N/A")
            if "pred_label" in disp.columns:
                disp["pred_label"] = disp["pred_label"].map({"success": "성공", "partial": "부분 적응", "failure": "실패"}).fillna(disp["pred_label"])
            disp = _rename_cols(disp)
            st.dataframe(disp, use_container_width=True, hide_index=True)
            st.caption(
                "💡 **스카우터 포인트**: ELO 격차가 크고 스타일 매칭률이 낮은 이적이 가장 위험. "
                "실패 확률 60% 이상 선수는 임대 먼저, 영구 이적은 2번째 시즌 이후 검토 권고."
            )

            # 스텝업/스텝다운 분석
            st.markdown("### 이동 유형별 적응률")
            st.caption("상위 팀(스텝업)으로 갈수록 적응 성공률이 낮아집니다. 이적 유형을 파악해 리스크를 예측하세요.")
            step_up = tdf[tdf["is_step_up"] == True] if "is_step_up" in tdf.columns else pd.DataFrame()
            step_down = tdf[tdf["is_step_down"] == True] if "is_step_down" in tdf.columns else pd.DataFrame()
            neutral_mask = pd.Series([True] * len(tdf), index=tdf.index)
            if "is_step_up" in tdf.columns:
                neutral_mask &= (tdf["is_step_up"] == False)
            if "is_step_down" in tdf.columns:
                neutral_mask &= (tdf["is_step_down"] == False)
            neutral = tdf[neutral_mask]

            step_rows = []
            # pred_label 컬럼 사용 (label은 학습용 원본 레이블, pred_label은 모델 예측 결과)
            _label_col = "pred_label" if "pred_label" in tdf.columns else "label"
            for name, sub in [("⬆️ 스텝업 (상위 팀으로)", step_up),
                               ("➡️ 수평이동", neutral),
                               ("⬇️ 스텝다운 (하위 팀으로)", step_down)]:
                if len(sub) and _label_col in sub.columns:
                    succ = (sub[_label_col] == "success").mean() * 100
                    fail = (sub[_label_col] == "failure").mean() * 100
                    part = (sub[_label_col] == "partial").mean() * 100
                    step_rows.append({
                        "이동 유형": name,
                        "건수": len(sub),
                        "성공률": f"{succ:.0f}%",
                        "부분 적응": f"{part:.0f}%",
                        "실패율": f"{fail:.0f}%",
                    })
            if step_rows:
                st.dataframe(pd.DataFrame(step_rows), use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── P8 이적 적응 리스크 분석 ─────────────────────────────────────────
        st.markdown("### P8 이적 적응 리스크 분석 (선수별 예측)")
        st.caption("P8 모델이 예측한 이적 후 WAR 변화량과 적응 리스크 등급을 선수별로 확인합니다.")

        adapt_df = load_transfer_adapt_predictions()
        if adapt_df.empty:
            st.info("transfer_adapt_predictions.parquet 데이터가 없습니다. P8 모델을 먼저 실행하세요.")
        else:
            # adapt_risk 필터
            risk_options = ["전체"] + sorted(adapt_df["adapt_risk"].dropna().unique().tolist()) if "adapt_risk" in adapt_df.columns else ["전체"]
            sel_risk = st.selectbox(
                "리스크 등급 필터",
                risk_options,
                format_func=lambda x: {"전체": "전체", "high": "🔴 고위험", "medium": "🟡 중위험", "low": "🟢 저위험"}.get(x, x),
                key="p8_risk_filter"
            )

            p8_filtered = adapt_df.copy()
            if sel_risk != "전체" and "adapt_risk" in p8_filtered.columns:
                p8_filtered = p8_filtered[p8_filtered["adapt_risk"] == sel_risk]

            # 표시 컬럼 선택 (있는 컬럼만)
            p8_show_cols = ["player", "from_team", "to_team", "style_distance", "predicted_war_change", "adapt_risk"]
            p8_available = [c for c in p8_show_cols if c in p8_filtered.columns]
            p8_disp = p8_filtered[p8_available].copy()

            # adapt_risk 이모지 표시
            if "adapt_risk" in p8_disp.columns:
                risk_icon_map = {"high": "🔴 고위험", "medium": "🟡 중위험", "low": "🟢 저위험"}
                p8_disp["adapt_risk"] = p8_disp["adapt_risk"].map(lambda x: risk_icon_map.get(x, x))

            # 컬럼 한국어 변환
            p8_rename = {
                "player": "선수",
                "from_team": "이전팀",
                "to_team": "이적팀",
                "style_distance": "전술 거리",
                "predicted_war_change": "예측 WAR 변화",
                "adapt_risk": "적응 리스크",
            }
            p8_disp = p8_disp.rename(columns={k: v for k, v in p8_rename.items() if k in p8_disp.columns})

            # 수치 포맷
            if "전술 거리" in p8_disp.columns:
                p8_disp["전술 거리"] = p8_disp["전술 거리"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
            if "예측 WAR 변화" in p8_disp.columns:
                p8_disp["예측 WAR 변화"] = p8_disp["예측 WAR 변화"].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "N/A")

            st.dataframe(p8_disp, use_container_width=True, hide_index=True)
            st.caption(
                "💡 **스카우터 포인트**: 전술 거리가 클수록 적응이 어렵습니다. "
                "예측 WAR 변화가 음수이면 이적 후 성과 하락이 예상됩니다. "
                "고위험(🔴) 선수는 임대 먼저 제안하거나 이적료 할인 협상 근거로 활용하세요."
            )

    # ══════════════════════════════════════════
    # TAB 4: 이적 시뮬레이터
    # ══════════════════════════════════════════
    with tab4:
        st.subheader("🎮 이적 시뮬레이터")
        st.caption("선수와 영입 희망 팀을 선택하면 유사 사례 기반으로 적응 성공 가능성을 시뮬레이션합니다.")

        adapt_sim = load_transfer_adapt_predictions()
        ratings_sim = load_scout_ratings()

        if ratings_sim.empty:
            st.info("선수 데이터를 불러올 수 없습니다.")
        else:
            # 최신 시즌 선수 목록
            latest_season = ratings_sim["season"].max() if "season" in ratings_sim.columns else None
            if latest_season:
                latest_ratings = ratings_sim[ratings_sim["season"] == latest_season].copy()
            else:
                latest_ratings = ratings_sim.copy()

            all_sim_players = sorted(latest_ratings["player"].dropna().unique().tolist())
            all_target_teams = sorted(adapt_sim["to_team"].dropna().unique().tolist()) if not adapt_sim.empty else []

            sim_c1, sim_c2 = st.columns(2)
            with sim_c1:
                sim_player = st.selectbox("📌 선수 선택", [""] + all_sim_players, key="tab4_sim_player")
            with sim_c2:
                sim_target_team = st.selectbox("🏟️ 영입 희망 팀", [""] + all_target_teams, key="tab4_sim_target")

            if sim_player and sim_target_team:
                # 선수 현재 정보
                p_row = latest_ratings[latest_ratings["player"] == sim_player]
                if p_row.empty:
                    p_row = ratings_sim[ratings_sim["player"] == sim_player].sort_values("season", ascending=False)

                if not p_row.empty:
                    pr = p_row.iloc[0]
                    p_war = pr.get("war", None)
                    p_team = pr.get("team", "")
                    p_age = pr.get("age", pr.get("age_tm", None))
                    p_pos = pr.get("pos_group", pr.get("pos", ""))
                    p_tier = pr.get("tier", "")
                    p_mv = pr.get("market_value", None)

                    # ── 선수 현황 카드 ──────────────────────────────────────
                    st.markdown("### 📋 선수 현황")
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    sc1.metric("현 팀", p_team or "-")
                    sc2.metric("나이", f"{int(p_age)}" if p_age and not pd.isna(p_age) else "-")
                    sc3.metric("WAR", f"{p_war:.1f}" if p_war and not pd.isna(p_war) else "-")
                    sc4.metric("등급", p_tier or "-")
                    if p_mv and not pd.isna(p_mv):
                        st.caption(f"💰 시장가치: €{p_mv/1_000_000:.1f}M  |  포지션: {p_pos}")

                    st.markdown(f"**{sim_player}** → **{sim_target_team}** 이적 시뮬레이션")
                    st.markdown("---")

                if not adapt_sim.empty:
                    # ── 유사 이적 사례 검색 ──────────────────────────────────
                    # 1) 해당 팀으로의 이적 사례
                    to_team_cases = adapt_sim[adapt_sim["to_team"] == sim_target_team].copy()

                    # 2) 선수 나이/포지션 필터
                    p_age_val = p_age if (p_age and not pd.isna(p_age)) else 25
                    age_similar = to_team_cases[
                        (to_team_cases["age"] >= p_age_val - 3) &
                        (to_team_cases["age"] <= p_age_val + 3)
                    ] if "age" in to_team_cases.columns else to_team_cases

                    # 3) 같은 팀에서의 이적 사례 (현팀 → 동일팀 경로)
                    same_from_cases = to_team_cases[to_team_cases["from_team"] == p_team] if p_team else pd.DataFrame()

                    # ── 시뮬레이션 결과 요약 ────────────────────────────────
                    st.markdown("### 🔮 시뮬레이션 결과")

                    if len(to_team_cases) == 0:
                        st.warning(f"'{sim_target_team}' 팀으로의 이적 사례가 데이터에 없습니다.")
                    else:
                        # 위험 분포
                        risk_map = {"low": "🟢 저위험", "medium": "🟡 중위험", "high": "🔴 고위험"}
                        risk_counts = to_team_cases["adapt_risk"].value_counts() if "adapt_risk" in to_team_cases.columns else pd.Series()

                        # 평균 WAR 변화
                        avg_war_change = to_team_cases["predicted_war_change"].mean() if "predicted_war_change" in to_team_cases.columns else None
                        actual_avg = to_team_cases["actual_war_change"].mean() if "actual_war_change" in to_team_cases.columns else None
                        success_rate = (to_team_cases["adapted_actual"] == 1).mean() * 100 if "adapted_actual" in to_team_cases.columns else None

                        # 나이 유사 그룹 지표
                        age_avg_war = age_similar["predicted_war_change"].mean() if "predicted_war_change" in age_similar.columns and len(age_similar) > 0 else None
                        age_risk = age_similar["adapt_risk"].mode().iloc[0] if "adapt_risk" in age_similar.columns and len(age_similar) > 0 else None

                        # 종합 위험 평가
                        if age_risk:
                            risk_display = risk_map.get(age_risk, age_risk)
                        elif avg_war_change is not None:
                            if avg_war_change < -2:
                                risk_display = "🔴 고위험"
                            elif avg_war_change < 0:
                                risk_display = "🟡 중위험"
                            else:
                                risk_display = "🟢 저위험"
                        else:
                            risk_display = "⚪ 데이터 부족"

                        # 메트릭 표시
                        rm1, rm2, rm3, rm4 = st.columns(4)
                        rm1.metric("종합 리스크", risk_display)
                        rm2.metric(
                            "예상 WAR 변화",
                            f"{age_avg_war:+.2f}" if age_avg_war is not None else (f"{avg_war_change:+.2f}" if avg_war_change is not None else "-"),
                            help="나이 유사 사례 기준 예상 WAR 변화량"
                        )
                        rm3.metric(
                            f"{sim_target_team} 적응 성공률",
                            f"{success_rate:.0f}%" if success_rate is not None else "-",
                            help="해당 팀으로의 이적 사례 중 적응 성공 비율"
                        )
                        rm4.metric("유사 사례 수", f"{len(age_similar)}건" if len(age_similar) > 0 else f"{len(to_team_cases)}건")

                        # 스카우트 조언
                        if age_risk == "high" or (avg_war_change is not None and avg_war_change < -2):
                            advice_color = EPL_MAGENTA
                            advice = "⚠️ **고위험 이적**: 임대 먼저 제안하거나 성과 연동 계약 구조를 검토하세요. 이적료 20~30% 할인 협상 근거로 활용 가능합니다."
                        elif age_risk == "medium" or (avg_war_change is not None and -2 <= avg_war_change < 0):
                            advice_color = "#FFD700"
                            advice = "🔍 **중간 리스크**: 첫 시즌 벤치 역할부터 적응 기간을 고려한 다년 계약을 권장합니다."
                        else:
                            advice_color = EPL_GREEN
                            advice = "✅ **저위험 이적**: 유사 사례 기준 적응 성공 가능성이 높습니다. 즉시 주전 투입을 검토하세요."

                        st.markdown(
                            f'<div style="background:#1a1a2e;border-left:4px solid {advice_color};'
                            f'padding:12px 16px;border-radius:8px;margin:8px 0;">{advice}</div>',
                            unsafe_allow_html=True,
                        )

                        # ── 같은 경로 이적 사례 ──────────────────────────────
                        if len(same_from_cases) > 0:
                            st.markdown(f"#### 🔄 {p_team} → {sim_target_team} 이적 사례 ({len(same_from_cases)}건)")
                            sfc_cols = [c for c in ["player", "season_new", "age", "predicted_war_change", "actual_war_change", "adapt_risk"] if c in same_from_cases.columns]
                            sfc_disp = same_from_cases[sfc_cols].copy()
                            if "adapt_risk" in sfc_disp.columns:
                                sfc_disp["adapt_risk"] = sfc_disp["adapt_risk"].map(risk_map).fillna(sfc_disp["adapt_risk"])
                            if "predicted_war_change" in sfc_disp.columns:
                                sfc_disp["predicted_war_change"] = sfc_disp["predicted_war_change"].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "-")
                            if "actual_war_change" in sfc_disp.columns:
                                sfc_disp["actual_war_change"] = sfc_disp["actual_war_change"].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "-")
                            sfc_rename = {"player": "선수", "season_new": "시즌", "age": "나이", "predicted_war_change": "예측 WAR 변화", "actual_war_change": "실제 WAR 변화", "adapt_risk": "적응 리스크"}
                            sfc_disp = sfc_disp.rename(columns={k: v for k, v in sfc_rename.items() if k in sfc_disp.columns})
                            st.dataframe(sfc_disp, use_container_width=True, hide_index=True)

                        # ── 나이 유사 이적 사례 ──────────────────────────────
                        st.markdown(f"#### 👥 {sim_target_team}으로의 유사 연령대 이적 사례 (나이 ±3세, {len(age_similar)}건)")
                        if len(age_similar) > 0:
                            ac_cols = [c for c in ["player", "from_team", "season_new", "age", "style_distance", "predicted_war_change", "actual_war_change", "adapt_risk"] if c in age_similar.columns]
                            ac_disp = age_similar[ac_cols].sort_values("season_new", ascending=False).copy()
                            if "adapt_risk" in ac_disp.columns:
                                ac_disp["adapt_risk"] = ac_disp["adapt_risk"].map(risk_map).fillna(ac_disp["adapt_risk"])
                            if "predicted_war_change" in ac_disp.columns:
                                ac_disp["predicted_war_change"] = ac_disp["predicted_war_change"].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "-")
                            if "actual_war_change" in ac_disp.columns:
                                ac_disp["actual_war_change"] = ac_disp["actual_war_change"].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "-")
                            if "style_distance" in ac_disp.columns:
                                ac_disp["style_distance"] = ac_disp["style_distance"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
                            ac_rename = {"player": "선수", "from_team": "이전팀", "season_new": "시즌", "age": "나이", "style_distance": "전술 거리", "predicted_war_change": "예측 WAR 변화", "actual_war_change": "실제 WAR 변화", "adapt_risk": "적응 리스크"}
                            ac_disp = ac_disp.rename(columns={k: v for k, v in ac_rename.items() if k in ac_disp.columns})
                            st.dataframe(ac_disp, use_container_width=True, hide_index=True)
                        else:
                            st.info("나이 유사 사례가 없어 전체 해당 팀 이적 사례를 표시합니다.")
                            tc_cols = [c for c in ["player", "from_team", "season_new", "age", "predicted_war_change", "adapt_risk"] if c in to_team_cases.columns]
                            tc_disp = to_team_cases[tc_cols].sort_values("season_new", ascending=False).head(10).copy()
                            if "adapt_risk" in tc_disp.columns:
                                tc_disp["adapt_risk"] = tc_disp["adapt_risk"].map(risk_map).fillna(tc_disp["adapt_risk"])
                            if "predicted_war_change" in tc_disp.columns:
                                tc_disp["predicted_war_change"] = tc_disp["predicted_war_change"].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "-")
                            tc_rename = {"player": "선수", "from_team": "이전팀", "season_new": "시즌", "age": "나이", "predicted_war_change": "예측 WAR 변화", "adapt_risk": "적응 리스크"}
                            tc_disp = tc_disp.rename(columns={k: v for k, v in tc_rename.items() if k in tc_disp.columns})
                            st.dataframe(tc_disp, use_container_width=True, hide_index=True)

                        # ── 리스크 분포 파이차트 ─────────────────────────────
                        if len(risk_counts) > 0:
                            st.markdown(f"#### 📊 {sim_target_team} 이적 적응 리스크 분포")
                            risk_labels = [risk_map.get(r, r) for r in risk_counts.index]
                            risk_colors = {"🟢 저위험": EPL_GREEN, "🟡 중위험": "#FFD700", "🔴 고위험": EPL_MAGENTA}
                            fig_risk = px.pie(
                                values=risk_counts.values,
                                names=risk_labels,
                                color=risk_labels,
                                color_discrete_map=risk_colors,
                                hole=0.4,
                            )
                            fig_risk.update_layout(
                                paper_bgcolor="#0d0d1a",
                                plot_bgcolor="#0d0d1a",
                                font_color="#fff",
                                margin=dict(t=10, b=10, l=10, r=10),
                                legend=dict(orientation="h", y=-0.1),
                                height=280,
                            )
                            st.plotly_chart(fig_risk, use_container_width=True, theme=None)

                        st.caption(
                            "💡 **시뮬레이터 해석**: 전술 거리가 클수록 적응이 어렵습니다. "
                            "예측 WAR 변화가 음수면 이적 후 성과 하락 가능성이 있습니다. "
                            "최소 3~5건 이상의 유사 사례가 있을 때 시뮬레이션 신뢰도가 높아집니다."
                        )
            else:
                st.info("선수와 영입 희망 팀을 모두 선택하면 시뮬레이션 결과가 표시됩니다.")

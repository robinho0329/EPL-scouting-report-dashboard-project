"""generate_meeting_note.py — 아침 스카우트팀 미팅 노트를 Claude API로 생성.

배경:
  기존에는 클라우드 트리거(Claude 세션)가 매일 09:00 KST에 미팅 노트를 만들어
  리포에 push했다. 그 트리거가 2026-05-27 이후 중단되면서 노트가 끊겼고,
  github_actions_dev.py가 매일 "미팅 노트 없음 — 종료"로 빠져나가
  데일리 루프가 3개월 가까이 빈손으로 돌았다. 이 스크립트가 그 자리를 대신한다.

흐름:
  1. models/*/results_summary.json + 직전 미팅 노트 수집
  2. Claude API로 미팅 노트 생성 (김태현 스카우트 / Marcus Webb 페르소나)
  3. github_actions_dev.parse_action_items로 파싱 검증 → 실패 시 1회 재시도
  4. reports/daily_meeting/YYYY-MM-DD_meeting.md 저장 (커밋은 워크플로가 담당)

액션아이템은 반드시 github_actions_dev.MODEL_MAP의 키워드를 포함해야 한다.
그래야 후속 단계가 대응 train 스크립트를 찾아 실제로 학습을 돌린다.
"""

import json
import os
import sys
from datetime import date
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.github_actions_dev import MODEL_MAP, parse_action_items  # noqa: E402

MEETING_DIR = ROOT / "reports" / "daily_meeting"
MODEL_ID = "claude-opus-5"
MAX_JSON_CHARS = 4000  # 모델당 프롬프트 반영 상한
DASHBOARD_URL = "https://epl-scouting-report-dashboard-project-ffyb8msh6uafxyyg8txsm8.streamlit.app"


def dashboard_page_count() -> int:
    """dashboard/app.py의 MENU_OPTIONS 항목 수. 읽기 실패 시 0."""
    app = ROOT / "dashboard" / "app.py"
    try:
        text = app.read_text(encoding="utf-8")
        start = text.index("MENU_OPTIONS = [")
        block = text[start : text.index("]", start)]
    except (OSError, ValueError):
        return 0
    return sum(1 for line in block.splitlines() if '"' in line)


def collect_model_results() -> list[str]:
    """models/*/results_summary.json 수집 → 프롬프트용 청크 목록."""
    chunks = []
    for path in sorted(ROOT.glob("models/*/results_summary.json")):
        if "archived" in path.parent.name:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ⚠️  읽기 실패 {path.parent.name}: {e}")
            continue
        # s3(50KB)·p5(26KB) 등은 선수별 배열이 대부분이고 미팅 노트에 필요한
        # 지표는 상단에 모여 있다. 매일 도는 작업이라 모델당 상한을 둔다.
        body = json.dumps(data, ensure_ascii=False, indent=2)
        if len(body) > MAX_JSON_CHARS:
            body = body[:MAX_JSON_CHARS] + "\n... (이하 생략 — 상세 배열)"
        chunks.append(f"### {path.parent.name}\n{body}")
    return chunks


def latest_previous_note() -> tuple[str, str]:
    """가장 최근 미팅 노트 (파일명, 본문 앞부분) 반환. 없으면 빈 문자열."""
    notes = sorted(MEETING_DIR.glob("*_meeting.md"), reverse=True)
    if not notes:
        return "", ""
    return notes[0].name, notes[0].read_text(encoding="utf-8")[:6000]


def build_prompt(today: str, results: list[str], prev_name: str, prev_body: str) -> str:
    """미팅 노트 생성 프롬프트."""
    results_text = "\n\n".join(results)
    keywords = "\n".join(
        f"- {keys[0]} ({dirname}): 키워드 {', '.join(keys)}" for keys, dirname, _ in MODEL_MAP
    )
    prev_section = (
        f"## 직전 미팅 노트 ({prev_name})\n\n{prev_body}"
        if prev_name
        else "## 직전 미팅 노트\n\n없음 (첫 미팅)."
    )

    return f"""당신은 EPL 스카우트 인텔리전스 프로젝트의 데일리 미팅 노트를 작성합니다.
오늘 날짜는 {today}입니다. 모든 내용은 한국어로 작성하세요.

## 팀 구성
- **김태현 스카우트**: EPL 중위권 구단 8년차 이적/스카우트 담당. 예산 30~50M 파운드.
  검증 기준은 항상 "이 분석이 실제 영입 회의에서 쓸 수 있나?"입니다.
- **Marcus Webb (Analytics Agent)**: 데이터 사이언티스트. 모델 성능과 피처 엔지니어링 담당.

## 현재 모델 성능 (results_summary.json 원본)

{results_text}

## 대시보드
Streamlit Cloud {dashboard_page_count()}개 페이지 운영 중: {DASHBOARD_URL}

{prev_section}

---

# 작성 지시

아래 5개 섹션을 순서대로 가진 마크다운 문서를 출력하세요.
문서 외의 설명, 인사말, 코드펜스는 절대 붙이지 마세요.

## 0. 액션아이템

**이 섹션이 가장 중요합니다.** 후속 자동화가 이 표를 파싱해 모델 재학습을 실행합니다.

형식을 정확히 지키세요 — 각 행은 반드시 아래 모양이어야 합니다:

| 순위 | 담당 | 내용 | 목표 | 기한 |
|------|------|------|------|------|
| 🥇 1 | Marcus Webb | **제목**: 상세 설명 | 목표 지표 | {today} |
| 🥈 2 | Marcus Webb | **제목**: 상세 설명 | 목표 지표 | {today} |
| 🥉 3 | Marcus Webb | **제목**: 상세 설명 | 목표 지표 | {today} |

절대 규칙:
1. 정확히 3행. 순위 칸은 `🥇 1`, `🥈 2`, `🥉 3`.
2. 내용 칸은 반드시 `**제목**: 상세` 형태 — 제목을 `**`로 감싸고 바로 뒤에 콜론.
3. 셀 안에서 `|` 문자를 쓰지 마세요 (표가 깨집니다).
4. **3개 중 최소 2개는 아래 모델 키워드 중 하나를 제목이나 상세에 반드시 포함**하세요.
   키워드가 없으면 자동화가 대응 스크립트를 못 찾아 아무 학습도 실행되지 않습니다.

{keywords}

5. 가장 성능이 약한 모델(목표 미달, R²/AUC/정확도가 낮은 쪽)을 우선 겨냥하세요.
6. 목표 지표는 현재 수치와 목표 수치를 함께 적으세요 (예: `P7 R² 0.5752 → 0.62`).

## 1. 모델 성능 현황

전체 모델을 표로 정리하세요. 열: 모델 / 지표 / 현재값 / 목표 / 상태.
상태는 🟢(목표 달성) 🟡(근접) 🔴(미달)로 표기하세요.

## 2. 팀 토론

김태현 스카우트와 Marcus Webb의 대화 형식. 다음을 다루세요:
- 가장 약한 모델 식별과 원인 진단
- 직전 미팅 액션아이템의 달성/미달 점검 (직전 노트가 있는 경우)
- 대시보드 실무 활용도 — 영입 회의에서 쓸 수 있는 수준인가

## 3. 액션아이템 상세

섹션 0의 3개 항목을 각각 풀어 씁니다. 담당, 구체적 파일 경로, 목표 지표, 기한 포함.

## 4. 김태현 스카우트 종합 평가

`X.X / 5.0` 형태의 점수와 근거, 그리고 5.0에 도달하기 위한 로드맵.
"""


def generate(client: anthropic.Anthropic, prompt: str) -> str:
    """Claude API 호출 → 미팅 노트 본문 반환."""
    with client.messages.stream(
        model=MODEL_ID,
        max_tokens=32000,
        output_config={"effort": "high"},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        message = stream.get_final_message()

    if message.stop_reason == "refusal":
        raise RuntimeError(f"모델이 요청을 거부했습니다: {message.stop_details}")

    return "".join(b.text for b in message.content if b.type == "text").strip()


def main() -> int:
    today = date.today().strftime("%Y-%m-%d")
    out_path = MEETING_DIR / f"{today}_meeting.md"

    if out_path.exists():
        print(f"✅ 오늘 미팅 노트가 이미 있습니다: {out_path.name}")
        return 0

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY 미설정 — 리포 시크릿을 확인하세요.", file=sys.stderr)
        return 1

    results = collect_model_results()
    if not results:
        print("❌ results_summary.json을 하나도 읽지 못했습니다.", file=sys.stderr)
        return 1

    prev_name, prev_body = latest_previous_note()
    print(f"📋 모델 {len(results)}개 / 직전 노트: {prev_name or '없음'}")

    client = anthropic.Anthropic()
    prompt = build_prompt(today, results, prev_name, prev_body)

    # 파싱 검증 — 액션아이템이 안 잡히면 후속 학습이 통째로 스킵되므로 1회 재시도
    for attempt in (1, 2):
        print(f"🤖 미팅 노트 생성 중... (시도 {attempt}/2)")
        note = generate(client, prompt)

        out_path.write_text(note, encoding="utf-8")
        items = parse_action_items(out_path)
        if items:
            print(f"✅ 액션아이템 {len(items)}개 파싱 확인 → {out_path.name}")
            for i, item in enumerate(items, 1):
                print(f"   {i}. [{item['assignee']}] {item['content'][:60]}")
            return 0

        print("  ⚠️  액션아이템 파싱 실패 — 표 형식이 어긋났습니다.")
        out_path.unlink(missing_ok=True)
        prompt += (
            "\n\n---\n\n직전 시도의 액션아이템 표가 파싱되지 않았습니다. "
            "섹션 0의 표 형식 규칙(순위 칸 `🥇 1`, 내용 칸 `**제목**: 상세`, "
            "셀 안 `|` 금지)을 그대로 지켜 다시 작성하세요."
        )

    print("❌ 2회 시도 모두 액션아이템 파싱 실패 — 노트를 저장하지 않았습니다.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

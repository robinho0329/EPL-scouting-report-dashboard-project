"""github_actions_dev.py — GitHub Actions에서 실행되는 Claude API 기반 개발 자동화.

흐름:
  1. 오늘 미팅 노트 읽기 (reports/daily_meeting/)
  2. 액션아이템 파싱
  3. Claude Opus API로 각 항목 구현 코드 생성
  4. 생성된 파일 저장 (git add는 workflow에서 수행)

제약:
  - GitHub Actions 환경이라 데이터 파일(.parquet, .joblib)이 없음
  - 모델 학습 스크립트만 생성 — 실제 학습은 로컬 PC에서 실행
  - 대시보드 코드는 즉시 배포 가능하므로 직접 수정
"""

import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent
MEETING_DIR = ROOT / "reports" / "daily_meeting"


# ─── 클라이언트 초기화 ────────────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def find_meeting_note() -> Path | None:
    """오늘 또는 어제 미팅 노트 반환."""
    for delta in [0, 1]:
        d = date.today() - timedelta(days=delta)
        path = MEETING_DIR / f"{d.strftime('%Y-%m-%d')}_meeting.md"
        if path.exists():
            return path
    return None


def parse_action_items(note_path: Path) -> list[dict]:
    """미팅 노트에서 액션아이템 파싱."""
    text = note_path.read_text(encoding="utf-8")

    section_match = re.search(
        r"##\s+(?:\d+\.\s+)?(?:오늘의\s+)?액션아이템[^\n]*\n(.*?)(?=\n##\s|\Z)",
        text, re.DOTALL,
    )
    if not section_match:
        print("⚠️  액션아이템 섹션 없음")
        return []

    section = section_match.group(1)
    items = []
    for row in re.finditer(
        r"\|\s*(?:🥇|🥈|🥉|\d)\s*\d?\s*\|\s*([^|]+)\|\s*\*\*([^*]+)\*\*:?\s*([^|]+)\|",
        section,
    ):
        assignee = row.group(1).strip()
        title    = row.group(2).strip()
        detail   = row.group(3).strip()
        content  = f"{title}: {detail}"

        # 타입 분류
        lower = content.lower()
        if any(k in lower for k in ["모델", "r²", "mae", "피처", "p7", "p8", "s1", "s2", "s3"]):
            item_type = "model"
        elif any(k in lower for k in ["대시보드", "페이지", "ui", "streamlit"]):
            item_type = "dashboard"
        else:
            item_type = "model"   # 기본값: 모델 스크립트 생성

        items.append({"assignee": assignee, "content": content, "type": item_type})

    return items


def get_context(item: dict) -> str:
    """액션아이템 관련 기존 코드 수집 (프롬프트 컨텍스트용)."""
    parts = []
    lower = item["content"].lower()

    def read_file(path: Path, max_chars: int = 2500) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
        except FileNotFoundError:
            return ""

    # 모델별 학습 스크립트 수집
    model_map = {
        ("p8", "이적 적응"): ROOT / "models" / "p8_transfer_adapt",
        ("s2", "시장가치"):  ROOT / "models" / "s2_market_value",
        ("p7", "성장 곡선"): ROOT / "models" / "p7_growth_curve",
        ("s1", "레이팅"):    ROOT / "models" / "s1_player_rating",
        ("p6", "시장가치"):  ROOT / "models" / "p6_market_value",
    }
    for keys, model_dir in model_map.items():
        if any(k in lower for k in keys):
            for py_file in sorted(model_dir.glob("*.py"))[-2:]:
                code = read_file(py_file)
                if code:
                    parts.append(f"# {py_file.relative_to(ROOT)}\n{code}")
            # results_summary.json도 포함
            summary = model_dir / "results_summary.json"
            if summary.exists():
                parts.append(f"# {summary.relative_to(ROOT)}\n{read_file(summary, 800)}")

    # 대시보드 컨텍스트
    if item["type"] == "dashboard":
        parts.append(f"# dashboard/app.py\n{read_file(ROOT / 'dashboard' / 'app.py', 1500)}")
        for page in sorted((ROOT / "dashboard" / "pages").glob("*.py"))[-3:]:
            parts.append(f"# {page.relative_to(ROOT)}\n{read_file(page, 1000)}")

    return "\n\n---\n\n".join(parts)


def call_claude(item: dict, context: str) -> str:
    """Claude API 호출 — 구현 코드 생성."""
    if item["type"] == "model":
        instructions = """개선된 Python 학습 스크립트를 작성하세요.

규칙:
- GitHub Actions 환경이라 실제 데이터(.parquet)가 없으므로 스크립트만 생성 (실행 X)
- 로컬 PC에서 나중에 실행될 코드입니다
- 파일 경로: models/<모델명>/train_v_next.py 형식으로 명시
- random_state=42 일관 사용
- 한국어 주석
- 코드 블록 형식: ```python:경로/파일명.py"""
    else:
        instructions = """Streamlit 대시보드 코드를 작성하세요.

규칙:
- EPL_PURPLE = "#37003c", EPL_MAGENTA = "#e90052" 등 기존 색상 유지
- @st.cache_data 데코레이터 사용
- render() 함수 최상위로 노출
- 기존 dashboard/pages/ 파일 스타일 준수
- 파일 경로 명시: ```python:dashboard/pages/파일명.py
- 한국어 주석"""

    prompt = f"""EPL 스카우트 인텔리전스 프로젝트 자동 개발.

## 오늘의 액션아이템
{item['content']}

## 담당자
{item['assignee']}

## 기존 관련 코드
{context[:5000] if context else "(관련 파일 없음)"}

## 작업 지시
{instructions}

파일 전체를 완성된 코드로 작성하세요. 설명은 코드 내 주석으로만."""

    resp = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=8096,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def write_generated_files(output: str) -> list[str]:
    """Claude 출력에서 ```python:경로 블록 추출 후 파일 저장."""
    written = []
    for m in re.finditer(r"```python:([^\n]+)\n(.*?)```", output, re.DOTALL):
        rel_path = m.group(1).strip()
        code = m.group(2).strip()
        target = ROOT / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code + "\n", encoding="utf-8")
        written.append(rel_path)
        print(f"  ✅ {rel_path}")
    return written


def main() -> None:
    print("=" * 60)
    print("🤖 GitHub Actions Daily Dev Loop")
    print(f"   실행일: {date.today()}")
    print("=" * 60)

    # 1. 미팅 노트 로드
    note = find_meeting_note()
    if not note:
        print("❌ 미팅 노트 없음 — 종료")
        sys.exit(0)
    print(f"📋 미팅 노트: {note.name}")

    # 2. 액션아이템 파싱
    items = parse_action_items(note)
    if not items:
        print("❌ 액션아이템 없음 — 종료")
        sys.exit(0)
    print(f"🎯 액션아이템 {len(items)}개\n")

    # 3. 항목별 구현
    all_written: list[str] = []
    for idx, item in enumerate(items, 1):
        print(f"[{idx}/{len(items)}] {item['content'][:70]}")
        print(f"       타입: {item['type']} | 담당: {item['assignee']}")
        context = get_context(item)
        output  = call_claude(item, context)
        written = write_generated_files(output)
        all_written.extend(written)
        print()

    # 4. 결과 요약
    print("=" * 60)
    print(f"✅ 완료: {len(all_written)}개 파일 생성/수정")
    for f in all_written:
        print(f"   - {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

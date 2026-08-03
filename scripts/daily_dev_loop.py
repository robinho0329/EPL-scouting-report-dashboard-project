"""daily_dev_loop.py — 매일 아침 미팅 노트를 읽어 액션아이템을 자동 구현하는 로컬 오케스트레이터.

실행 흐름:
  1. git pull (최신 미팅 노트 가져오기)
  2. 오늘 미팅 노트에서 액션아이템 파싱
  3. Claude Code CLI로 epl-model / epl-dashboard 에이전트 실행
  4. git push → GitHub → Streamlit Cloud 자동 배포
  5. Streamlit Cloud 앱 에러 체크 → 에러 시 자동 수정 & 재푸시
"""

import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MEETING_DIR = ROOT / "reports" / "daily_meeting"
LOG_DIR = ROOT / "reports" / "dev_loop_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─── 로깅 ─────────────────────────────────────────────────────────────────────
today_str = date.today().strftime("%Y-%m-%d")
log_file = LOG_DIR / f"{today_str}_dev_loop.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("daily_dev_loop")

# ─── Claude CLI 경로 ──────────────────────────────────────────────────────────
# 설치 위치는 사람마다 다르다. PATH 에서 찾고, 없으면 CLAUDE_CLI 환경변수를 쓴다.
CLAUDE_CLI = (
    os.environ.get("CLAUDE_CLI")
    or shutil.which("claude")
    or shutil.which("claude.cmd")
    or "claude"
)

# ─── Streamlit Cloud 앱 URL ───────────────────────────────────────────────────
# ※ 실제 배포 URL로 교체 필요
STREAMLIT_URL = "https://epl-scouting-report-dashboard-project-ffyb8msh6uafxyyg8txsm8.streamlit.app"
STREAMLIT_WAIT_SEC = 90   # push 후 재배포 대기 시간(초)


def run(cmd: list[str], cwd: Path = ROOT, timeout: int = 60) -> str:
    """subprocess 실행 후 stdout 반환. 실패 시 예외 발생."""
    result = subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        raise RuntimeError(f"명령 실패 ({result.returncode}): {result.stderr[:500]}")
    return result.stdout.strip()


def git_pull() -> None:
    """원격 최신 변경사항 가져오기."""
    logger.info("git pull origin master ...")
    run(["git", "pull", "origin", "master"])
    logger.info("git pull 완료")


def find_meeting_note(target_date: date | None = None) -> Path | None:
    """오늘(또는 지정일) 미팅 노트 파일 경로 반환. 없으면 None."""
    if target_date is None:
        target_date = date.today()
    path = MEETING_DIR / f"{target_date.strftime('%Y-%m-%d')}_meeting.md"
    if path.exists():
        return path
    # 어제 노트도 허용 (실행 타이밍 여유)
    yesterday = target_date - timedelta(days=1)
    path_y = MEETING_DIR / f"{yesterday.strftime('%Y-%m-%d')}_meeting.md"
    if path_y.exists():
        logger.warning(f"오늘 노트 없음 — 어제 노트 사용: {path_y.name}")
        return path_y
    return None


def parse_action_items(note_path: Path) -> list[dict]:
    """미팅 노트에서 액션아이템 파싱.

    Returns:
        [{"rank": 1, "assignee": "Marcus", "content": "...", "type": "model"|"dashboard"|"general"}, ...]
    """
    text = note_path.read_text(encoding="utf-8")

    # 액션아이템 섹션 추출 (## 3. 액션아이템 ... 또는 ## 오늘의 액션아이템 ... 등)
    section_match = re.search(
        r"##\s+(?:\d+\.\s+)?(?:오늘의\s+)?액션아이템[^\n]*\n(.*?)(?=\n##\s|\Z)",
        text, re.DOTALL
    )
    if not section_match:
        logger.warning("액션아이템 섹션을 찾을 수 없습니다.")
        return []

    section = section_match.group(1)

    # 표 형식 파싱: | 🥇 1 | 담당자 | **제목**: 내용 | ... |
    items = []
    for row in re.finditer(
        r"\|\s*(?:🥇|🥈|🥉|\d)\s*\d?\s*\|\s*([^|]+)\|\s*\*\*([^*]+)\*\*:?\s*([^|]+)\|",
        section
    ):
        assignee = row.group(1).strip()
        title = row.group(2).strip()
        detail = row.group(3).strip()
        content = f"{title}: {detail}"

        # 타입 분류
        item_type = "general"
        content_lower = content.lower()
        if any(k in content_lower for k in ["모델", "r²", "mae", "학습", "피처", "p1","p2","p3","p4","p5","p6","p7","p8","s1","s2","s3","s4","s5","s6"]):
            item_type = "model"
        elif any(k in content_lower for k in ["대시보드", "페이지", "ui", "dashboard", "streamlit"]):
            item_type = "dashboard"

        items.append({"assignee": assignee, "content": content, "type": item_type})

    # 표 형식 미검출 시 목록 형식 파싱 (1. [담당: X] ...)
    if not items:
        for m in re.finditer(r"\d+\.\s+\[담당:\s*([^\]]+)\]\s*(.+)", section):
            assignee = m.group(1).strip()
            content = m.group(2).strip()
            item_type = "model" if any(k in content for k in ["모델","R²","MAE","피처","P7","P8","S2"]) else "dashboard"
            items.append({"assignee": assignee, "content": content, "type": item_type})

    logger.info(f"파싱된 액션아이템 {len(items)}개: {[i['content'][:40] for i in items]}")
    return items


def build_agent_prompt(items: list[dict], agent_type: str, meeting_path: Path) -> str:
    """에이전트 타입에 맞는 Claude CLI 프롬프트 생성."""
    filtered = [i for i in items if i["type"] == agent_type]
    if not filtered:
        return ""

    items_text = "\n".join(
        f"{idx+1}. [{i['assignee']}] {i['content']}"
        for idx, i in enumerate(filtered)
    )

    meeting_content = meeting_path.read_text(encoding="utf-8")[:3000]  # 앞부분만

    if agent_type == "model":
        return f"""EPL 스카우트 프로젝트 자동 개발 루틴 — 오늘 미팅 액션아이템 구현.

오늘 미팅 노트 요약:
{meeting_content}

## 구현할 모델 액션아이템
{items_text}

## 지침
- `{ROOT}` 기준으로 작업
- 파이썬은 지금 PATH 에 있는 `python` 을 쓴다 (로컬은 .venv, Actions 는 setup-python)
- 기존 train 스크립트(train_v2.py, train_v3.py 등)를 참고해 개선 버전 작성
- 모델 학습 후 results_summary.json 업데이트
- random_state=42 일관 사용
- 완료 후 변경된 파이썬 파일과 results_summary.json을 git add까지만 수행 (커밋은 오케스트레이터가 함)"""

    else:  # dashboard
        return f"""EPL 스카우트 프로젝트 자동 개발 루틴 — 오늘 미팅 액션아이템 구현.

오늘 미팅 노트 요약:
{meeting_content}

## 구현할 대시보드 액션아이템
{items_text}

## 지침
- `{ROOT}/dashboard/` 기준으로 작업
- `@st.cache_data` 데코레이터 사용, 기존 페이지 스타일 유지
- 신규 페이지는 `pages/` 디렉토리에, 메뉴 추가는 `app.py`에
- AST 문법 검사 후 git add까지만 수행 (커밋은 오케스트레이터가 함)"""


def run_claude_agent(prompt: str, agent_type: str) -> bool:
    """Claude Code CLI로 에이전트 실행. 성공 여부 반환."""
    logger.info(f"Claude {agent_type} 에이전트 실행 시작...")
    try:
        result = subprocess.run(
            [CLAUDE_CLI, "--dangerously-skip-permissions", "-p", prompt],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=1800,  # 30분
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode == 0:
            logger.info(f"{agent_type} 에이전트 완료")
            logger.info(f"출력 (마지막 500자):\n{result.stdout[-500:]}")
            return True
        else:
            logger.error(f"{agent_type} 에이전트 실패 (코드 {result.returncode})")
            logger.error(f"stderr: {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"{agent_type} 에이전트 타임아웃 (30분 초과)")
        return False


def check_streamlit_and_fix(streamlit_url: str) -> bool:
    """push 후 Streamlit Cloud 앱 에러 확인.

    1단계: playwright 헤드리스로 에러 감지 (GitHub Actions와 동일)
    2단계: 에러 있으면 Claude CLI로 자동 수정 (로컬 전용)
    """
    import time
    logger.info(f"Streamlit Cloud 재배포 대기 중 ({STREAMLIT_WAIT_SEC}초)...")
    time.sleep(STREAMLIT_WAIT_SEC)

    # ── 1단계: playwright 헤드리스로 에러 감지 ──────────────────
    errors_found = []
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            logger.info(f"🌐 {streamlit_url} 접속 중...")
            try:
                page.goto(streamlit_url, timeout=60_000)
                page.wait_for_selector("[data-testid='stApp']", timeout=60_000)
            except PWTimeout:
                logger.error("페이지 로딩 타임아웃")
                browser.close()
                return False

            error_els = page.query_selector_all(
                "[data-testid='stException'], .element-container.stAlert"
            )
            for el in error_els:
                errors_found.append(el.inner_text()[:500])

            content = page.content()
            if "Traceback (most recent call last)" in content:
                idx = content.find("Traceback")
                errors_found.append(content[idx:idx + 500])

            browser.close()

    except ImportError:
        logger.warning("playwright 미설치 — 에러 감지 스킵 (pip install playwright 후 playwright install chromium)")
        return True

    if not errors_found:
        logger.info("✅ Streamlit 앱 정상")
        return True

    # ── 2단계: 에러 있으면 Claude CLI로 자동 수정 (로컬 전용) ───
    logger.warning(f"Streamlit 에러 {len(errors_found)}건 감지 → Claude CLI 자동 수정 시도")
    err_text = "\n\n".join(errors_found)

    prompt = f"""EPL 스카우트 대시보드 Streamlit 에러 자동 수정.

## 감지된 에러
{err_text}

## 작업
1. 에러 원인 분석
2. `{ROOT}/dashboard/` 내 해당 파일 수정
   - FileNotFoundError → Path(__file__) 기반 상대경로로 교체
   - ModuleNotFoundError → requirements.txt 추가
   - KeyError / AttributeError → 컬럼명 또는 데이터 로딩 수정
3. 수정 후 `git add <파일>` (커밋은 오케스트레이터가 처리)

## 주의
- 정상 작동 중인 페이지 건드리지 말 것
- git add까지만 수행, 커밋·푸시 금지"""

    result = subprocess.run(
        [CLAUDE_CLI, "--dangerously-skip-permissions", "-p", prompt],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=1800,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode == 0:
        logger.info("Claude CLI 자동 수정 완료")
        return True
    else:
        logger.error(f"자동 수정 실패: {result.stderr[:300]}")
        return False


def git_commit_push(items: list[dict], msg_prefix: str = "") -> bool:
    """변경사항 커밋 & 푸시."""
    try:
        status = run(["git", "status", "--short"])
        if not status:
            logger.info("변경사항 없음 — 커밋 스킵")
            return True

        run(["git", "add", "-A"])

        titles = " / ".join(i["content"][:30] for i in items[:2])
        prefix = f"{msg_prefix} " if msg_prefix else ""
        commit_msg = f"auto({today_str}): {prefix}{titles}"
        run(["git", "commit", "-m", commit_msg])
        run(["git", "push", "origin", "master"], timeout=120)

        logger.info(f"git push 완료: {commit_msg}")
        return True
    except RuntimeError as e:
        logger.error(f"git 작업 실패: {e}")
        return False


def main() -> None:
    logger.info(f"=== daily_dev_loop 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

    # 1. git pull
    try:
        git_pull()
    except RuntimeError as e:
        logger.error(f"git pull 실패: {e}")
        sys.exit(1)

    # 2. 미팅 노트 로드
    note_path = find_meeting_note()
    if not note_path:
        logger.error(f"미팅 노트를 찾을 수 없습니다: {MEETING_DIR}")
        sys.exit(1)
    logger.info(f"미팅 노트 로드: {note_path.name}")

    # 3. 액션아이템 파싱
    items = parse_action_items(note_path)
    if not items:
        logger.warning("액션아이템이 없습니다. 종료.")
        sys.exit(0)

    # 4. 기존 train 스크립트 직접 실행 (GitHub Actions와 동일 방식)
    logger.info("기존 train 스크립트 실행 (github_actions_dev.py 위임)...")
    try:
        venv_python = str(ROOT / ".venv" / "Scripts" / "python")
        train_result = subprocess.run(
            [venv_python, str(ROOT / "scripts" / "github_actions_dev.py")],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=1800,
            encoding="utf-8",
            errors="replace",
        )
        logger.info(train_result.stdout[-1000:])
        train_ok = train_result.returncode == 0
        if not train_ok:
            logger.error(f"학습 실패:\n{train_result.stderr[-500:]}")
    except subprocess.TimeoutExpired:
        logger.error("학습 타임아웃 (30분)")
        train_ok = False

    # 5. git commit & push (1차: 학습 결과)
    push_ok = git_commit_push(items)

    # 6. Streamlit Cloud 에러 체크 → 에러 시 자동 수정 + 재푸시
    streamlit_ok = False
    if push_ok:
        streamlit_ok = check_streamlit_and_fix(STREAMLIT_URL)
        # 에이전트가 수정한 파일이 있으면 2차 커밋
        git_commit_push(items, msg_prefix="[streamlit-fix]")
    else:
        logger.warning("push 실패로 Streamlit 체크 스킵")

    # 결과 요약
    logger.info("=== 실행 결과 요약 ===")
    logger.info(f"  학습 실행    : {'✅ 성공' if train_ok else '❌ 실패'}")
    logger.info(f"  git push    : {'✅ 성공' if push_ok else '❌ 실패'}")
    logger.info(f"  streamlit   : {'✅ 정상' if streamlit_ok else '⚠️  확인 필요'}")
    logger.info(f"=== daily_dev_loop 종료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")


if __name__ == "__main__":
    main()

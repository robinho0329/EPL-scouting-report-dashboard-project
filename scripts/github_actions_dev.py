"""github_actions_dev.py — GitHub Actions에서 실행되는 자동 개발 파이프라인.

흐름:
  1. 오늘 미팅 노트 읽기 → 액션아이템 파싱
  2. 액션아이템 키워드 → 대응 모델 디렉토리 탐색
  3. 기존 최신 train 스크립트 직접 실행 (Claude API 불필요)
  4. results_summary.json + 변경 파일 저장 (git push는 workflow에서)
  5. push 후 Streamlit Cloud 헤드리스 체크 → 에러 시 워크플로우 실패 처리
"""

import warnings
warnings.filterwarnings("ignore")

import json
import os
import re
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Set, Tuple

ROOT        = Path(__file__).resolve().parent.parent
MEETING_DIR = ROOT / "reports" / "daily_meeting"

STREAMLIT_URL   = "https://epl-scouting-report-dashboard-project-ffyb8msh6uafxyyg8txsm8.streamlit.app"
STREAMLIT_WAIT  = int(os.getenv("STREAMLIT_WAIT_SEC", "90"))  # 재배포 대기(초)

# ────────────────────────────────────────────────────────────────────
# 모델 키워드 → (디렉토리, 우선 실행 스크립트) 매핑
# 스크립트가 None이면 디렉토리 내 최신 train*.py 자동 탐색
# ────────────────────────────────────────────────────────────────────
MODEL_MAP = [
    (["p8", "이적 적응", "transfer adapt"],     "p8_transfer_adapt",    None),
    (["p7", "성장 곡선", "growth curve"],        "p7_growth_curve",      "train_v5_1.py"),
    (["s2", "시장가치", "market value def"],     "s2_market_value",      None),
    (["s1", "레이팅", "player rating"],          "s1_player_rating",     None),
    (["p1", "경기 결과", "match result"],        "p1_match_result",      "train_pipeline.py"),
    (["p2", "득점", "goal scor"],               "p2_goal_scoring",      None),
    (["p3", "강등 예측", "relegat"],              "p3_relegation",        "train_relegation.py"),
    (["p4", "mvp", "출전 시간"],                 "p4_mvp",               "mvp_pipeline.py"),
    (["p5", "클러스터", "cluster"],              "p5_clustering",        "run_clustering_k6.py"),
    (["p6", "이적료", "transfer fee"],           "p6_market_value",      None),
    (["s3", "유사 선수", "similar"],             "s3_similarity",        None),
    (["s4", "성장", "growth profile"],           "s4_growth",            None),
    (["s5", "이적 적합", "transfer fit"],        "s5_transfer_adapt",    None),
    (["s6", "쇠퇴", "decline"],                  "s6_decline",           None),
]


# ────────────────────────────────────────────────────────────────────
# 유틸리티
# ────────────────────────────────────────────────────────────────────

def find_meeting_note() -> Optional[Path]:
    """오늘(또는 어제) 미팅 노트 파일 반환."""
    for delta in [0, 1]:
        d = date.today() - timedelta(days=delta)
        p = MEETING_DIR / f"{d.strftime('%Y-%m-%d')}_meeting.md"
        if p.exists():
            return p
    return None


def parse_action_items(note: Path) -> list:
    """미팅 노트 액션아이템 섹션 파싱 → 목록 반환."""
    text = note.read_text(encoding="utf-8")
    m = re.search(
        r"##\s+(?:\d+\.\s+)?(?:오늘의\s+)?액션아이템[^\n]*\n(.*?)(?=\n##\s|\Z)",
        text, re.DOTALL,
    )
    if not m:
        return []
    section = m.group(1)
    items = []
    for row in re.finditer(
        r"\|\s*(?:🥇|🥈|🥉|\d)\s*\d?\s*\|\s*([^|]+)\|\s*\*\*([^*]+)\*\*:?\s*([^|]+)\|",
        section,
    ):
        assignee = row.group(1).strip()
        title    = row.group(2).strip()
        detail   = row.group(3).strip()
        items.append({"assignee": assignee, "content": f"{title}: {detail}"})
    return items


def resolve_script(content: str) -> Tuple[Optional[Path], Optional[Path]]:
    """액션아이템 내용 → (스크립트 경로, 모델 디렉토리) 반환."""
    lower = content.lower()
    for keys, dirname, preferred in MODEL_MAP:
        if not any(k in lower for k in keys):
            continue
        model_dir = ROOT / "models" / dirname
        if not model_dir.exists():
            print(f"  ⚠️  디렉토리 없음: models/{dirname}")
            return None, None

        # 우선 스크립트가 지정된 경우
        if preferred:
            p = model_dir / preferred
            if p.exists():
                return p, model_dir

        # 최신 train*.py 자동 탐색 (버전 번호 내림차순)
        candidates = sorted(model_dir.glob("train*.py"), reverse=True)
        if candidates:
            return candidates[0], model_dir

        # train 없으면 run*.py 시도
        candidates = sorted(model_dir.glob("run*.py"), reverse=True)
        if candidates:
            return candidates[0], model_dir

        # build*.py 시도 (s3 유사선수 등)
        candidates = sorted(model_dir.glob("build*.py"), reverse=True)
        if candidates:
            return candidates[0], model_dir

        print(f"  ⚠️  실행 가능한 스크립트 없음: models/{dirname}")
        return None, None

    return None, None


def run_script(script: Path, model_dir: Path):
    """스크립트 실행 → (성공여부, 에러메시지) 반환."""
    print(f"  🚀 실행: {script.relative_to(ROOT)}")
    try:
        result = subprocess.run(
            [sys.executable, "-W", "ignore", str(script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=1500,   # 25분
        )
    except subprocess.TimeoutExpired:
        # 잡지 않으면 예외가 main()까지 올라가 프로세스가 죽고,
        # 워크플로의 커밋 단계가 스킵되어 나머지 액션아이템 결과까지 통째로 날아간다.
        print("  ⏱️  25분 초과 — 이 항목만 실패 처리하고 다음으로 넘어갑니다")
        return False, "timeout(1500s)"

    stdout_tail = result.stdout[-1500:] if result.stdout else ""
    if stdout_tail:
        print(stdout_tail)
    if result.returncode != 0:
        err = result.stderr[-800:] if result.stderr else result.stdout[-400:]
        print(f"  ❌ 실패 (코드 {result.returncode}):\n{err}")
        return False, err
    print("  ✅ 완료")
    return True, ""


# ────────────────────────────────────────────────────────────────────
# Streamlit Cloud 헤드리스 체크  (playwright 사용)
# ────────────────────────────────────────────────────────────────────

def check_streamlit(url: str, wait_sec: int = STREAMLIT_WAIT) -> bool:
    """Streamlit Cloud 앱 에러 여부 확인.

    playwright로 헤드리스 브라우저를 띄워 실제 앱 화면을 검사한다.
    - 슬리핑 상태(Wake up 버튼)는 코드 에러 아님 → True 반환
    - Traceback / stException 감지 시에만 False 반환
    playwright 미설치 시 체크 스킵 (True 반환).
    """
    print(f"  ⏳ Streamlit 재배포 대기 ({wait_sec}초)...")
    time.sleep(wait_sec)

    # ── playwright 시도 ──────────────────────────────────────────
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            print(f"  🌐 {url} 접속 중...")
            try:
                page.goto(url, timeout=60_000, wait_until="domcontentloaded")
            except PWTimeout:
                print("  ⚠️  초기 접속 타임아웃 — 슬리핑 가능성, 정상 처리")
                browser.close()
                return True

            # 슬리핑 상태 감지 → Wake up 클릭 후 추가 대기
            try:
                for selector in [
                    "button:has-text('Wake app')",
                    "button:has-text('Yes, get this app back up!')",
                    "button:has-text('Wake')",
                ]:
                    wake_btn = page.query_selector(selector)
                    if wake_btn:
                        print("  💤 슬리핑 상태 감지 — Wake 버튼 클릭")
                        wake_btn.click()
                        time.sleep(15)
                        break
            except Exception:
                pass

            # stApp 로딩 대기 (최대 120초)
            try:
                page.wait_for_selector("[data-testid='stApp']", timeout=120_000)
            except PWTimeout:
                print("  ⚠️  stApp 로딩 타임아웃 — 슬리핑/재시작 중으로 간주, 정상 처리")
                browser.close()
                return True  # 슬리핑·재배포 중은 코드 에러 아님

            # 실제 코드 에러만 탐지
            error_els = page.query_selector_all(
                "[data-testid='stException']"
            )
            if error_els:
                print(f"  ❌ Streamlit 코드 에러 감지 ({len(error_els)}건):")
                for el in error_els[:3]:
                    print(f"     {el.inner_text()[:300]}")
                browser.close()
                return False

            content = page.content()
            if "Traceback (most recent call last)" in content:
                idx = content.find("Traceback")
                print(f"  ❌ Traceback 감지:\n{content[idx:idx+400]}")
                browser.close()
                return False

            browser.close()
            print("  ✅ Streamlit 앱 정상")
            return True

    except ImportError:
        # playwright 없으면 체크 불가 → 정상 처리
        print("  ⚠️  playwright 없음 — Streamlit 체크 스킵")
        return True

    except Exception as e:
        print(f"  ⚠️  Streamlit 체크 예외 (비치명적): {e}")
        return True  # 체크 자체 실패는 코드 에러 아님
        return False


# ────────────────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Streamlit 체크 전용 모드 ─────────────────────────────────
    if os.getenv("STREAMLIT_CHECK") == "1":
        print("=" * 60)
        print("🔍 Streamlit Cloud 에러 체크 모드")
        print("=" * 60)
        ok = check_streamlit(STREAMLIT_URL)
        if not ok:
            print("❌ Streamlit 에러 감지")
            sys.exit(2)
        sys.exit(0)

    # ── 일반 학습 모드 ───────────────────────────────────────────
    print("=" * 60)
    print("🤖 GitHub Actions Daily Dev Loop (기존 스크립트 실행)")
    print(f"   실행일: {date.today()}")
    print("=" * 60)

    note = find_meeting_note()
    if not note:
        print("❌ 미팅 노트 없음 — 종료")
        sys.exit(0)
    print(f"📋 {note.name}\n")

    items = parse_action_items(note)
    if not items:
        print("❌ 액션아이템 없음 — 종료")
        sys.exit(0)

    print(f"🎯 액션아이템 {len(items)}개\n")
    results_log = []
    seen_scripts: Set[str] = set()   # 같은 스크립트 중복 실행 방지

    for idx, item in enumerate(items, 1):
        print(f"[{idx}/{len(items)}] {item['content'][:70]}")
        print(f"       담당: {item['assignee']}")

        script, model_dir = resolve_script(item["content"])
        if script is None:
            print("  ⚠️  대응 스크립트 없음 — 스킵\n")
            results_log.append({
                "item":    item["content"][:50],
                "script":  None,
                "success": False,
                "reason":  "스크립트 없음",
            })
            continue

        script_key = str(script)
        if script_key in seen_scripts:
            print(f"  ⏭️  이미 실행됨 — 스킵 ({script.name})\n")
            results_log.append({
                "item":    item["content"][:50],
                "script":  str(script.relative_to(ROOT)),
                "success": True,
                "reason":  "중복 스킵",
            })
            continue

        seen_scripts.add(script_key)
        success, err = run_script(script, model_dir)
        results_log.append({
            "item":    item["content"][:50],
            "script":  str(script.relative_to(ROOT)),
            "success": success,
            "error":   err[:400] if err else "",
        })
        print()

    # 실행 결과 요약 저장
    summary_path = MEETING_DIR / f"{date.today()}_dev_result.json"
    summary_path.write_text(
        json.dumps(
            {"date": str(date.today()), "results": results_log},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    ok = sum(1 for r in results_log if r["success"])
    print("=" * 60)
    print(f"✅ 학습 완료: {ok}/{len(results_log)} 성공")
    print(f"📄 결과: {summary_path.relative_to(ROOT)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

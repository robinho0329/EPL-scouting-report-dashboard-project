"""github_actions_dev.py — GitHub Actions에서 실행되는 자동 개발 파이프라인.

흐름:
  1. 오늘 미팅 노트 읽기 → 액션아이템 파싱
  2. Claude API로 개선된 학습 스크립트 / 대시보드 코드 생성
  3. 모델 항목: 생성된 스크립트를 실제로 실행 → 학습 완료
  4. 대시보드 항목: 코드 파일 저장 (Streamlit Cloud 자동 배포)
  5. results_summary.json + 변경 파일 저장 (git push는 workflow에서)
"""

import os
import re
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import anthropic

ROOT        = Path(__file__).resolve().parent.parent
MEETING_DIR = ROOT / "reports" / "daily_meeting"
DATA_FEAT   = ROOT / "data" / "features" / "player_features.parquet"
DATA_MATCH  = ROOT / "data" / "features" / "match_features.parquet"
DATA_SCOUT  = ROOT / "data" / "scout" / "scout_ratings_v3.parquet"

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# ────────────────────────────────────────────────────────────────────
# 유틸리티
# ────────────────────────────────────────────────────────────────────

def find_meeting_note() -> Path | None:
    for delta in [0, 1]:
        d = date.today() - timedelta(days=delta)
        p = MEETING_DIR / f"{d.strftime('%Y-%m-%d')}_meeting.md"
        if p.exists():
            return p
    return None


def parse_action_items(note: Path) -> list[dict]:
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
        content  = f"{title}: {detail}"
        lower    = content.lower()
        if any(k in lower for k in ["모델","r²","mae","피처","p7","p8","s1","s2","s3","학습"]):
            kind = "model"
        elif any(k in lower for k in ["대시보드","페이지","ui","streamlit"]):
            kind = "dashboard"
        else:
            kind = "model"
        items.append({"assignee": assignee, "content": content, "type": kind})
    return items


def read_file(path: Path, max_chars: int = 3000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except FileNotFoundError:
        return ""


def get_context(item: dict) -> str:
    """액션아이템 관련 기존 코드 수집."""
    parts = []
    lower = item["content"].lower()

    model_dirs = {
        ("p8", "이적 적응"): "p8_transfer_adapt",
        ("s2", "시장가치"):  "s2_market_value",
        ("p7", "성장 곡선"): "p7_growth_curve",
        ("s1", "레이팅"):    "s1_player_rating",
        ("p6",):             "p6_market_value",
    }
    for keys, dirname in model_dirs.items():
        if any(k in lower for k in keys):
            mdir = ROOT / "models" / dirname
            for f in sorted(mdir.glob("train*.py"))[-2:]:
                parts.append(f"# {f.relative_to(ROOT)}\n{read_file(f)}")
            summary = mdir / "results_summary.json"
            if summary.exists():
                parts.append(f"# {summary.relative_to(ROOT)}\n{read_file(summary, 600)}")

    if item["type"] == "dashboard":
        parts.append(f"# dashboard/app.py\n{read_file(ROOT/'dashboard'/'app.py', 1500)}")
        for p in sorted((ROOT/"dashboard"/"pages").glob("*.py"))[-3:]:
            parts.append(f"# {p.relative_to(ROOT)}\n{read_file(p, 800)}")

    return "\n\n---\n\n".join(parts)


# ────────────────────────────────────────────────────────────────────
# Claude API 호출
# ────────────────────────────────────────────────────────────────────

def generate_train_script(item: dict, context: str) -> tuple[str, Path] | None:
    """모델 학습 스크립트 생성 → (코드, 저장경로) 반환."""
    lower = item["content"].lower()

    # 대상 모델 디렉토리 매핑
    target_dir = None
    for keys, dirname in [
        (["p8","이적 적응"], "p8_transfer_adapt"),
        (["s2","시장가치"], "s2_market_value"),
        (["p7","성장 곡선"], "p7_growth_curve"),
        (["s1","레이팅"], "s1_player_rating"),
    ]:
        if any(k in lower for k in keys):
            target_dir = ROOT / "models" / dirname
            break

    if target_dir is None:
        target_dir = ROOT / "models" / "misc"

    target_dir.mkdir(parents=True, exist_ok=True)
    script_path = target_dir / "train_auto.py"

    prompt = f"""GitHub Actions 서버(ubuntu-latest)에서 실행되는 EPL 모델 학습 스크립트를 작성하세요.

## 액션아이템
{item['content']}

## 기존 관련 코드
{context[:4500] if context else '(없음)'}

## 환경 정보
- Python 3.12, ubuntu-latest
- 설치된 패키지: pandas, pyarrow, scikit-learn, xgboost, lightgbm, optuna, joblib, numpy
- 데이터 경로 (존재 보장):
  * {DATA_FEAT}
  * {DATA_MATCH}
  * {DATA_SCOUT}
- 모델 저장 경로: {target_dir}/

## 요구사항
1. 위 데이터 파일을 읽어 실제 학습 수행 (GitHub Actions에서 바로 실행됨)
2. Optuna는 최대 20 trials (시간 제한으로 인해)
3. results_summary.json 저장 (R², MAE, 학습 시간 포함)
4. 모델 바이너리(.joblib)는 gitignore 대상이므로 저장 불필요
5. random_state=42
6. 한국어 주석
7. 완료 시 "✅ 학습 완료 R²=X.XX" 출력

완성된 Python 파일 전체를 코드 블록 없이 바로 작성하세요 (```python 없이):"""

    resp = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=6000,
        messages=[{"role": "user", "content": prompt}],
    )
    code = resp.content[0].text.strip()
    # 혹시 코드블록이 포함되면 제거
    code = re.sub(r"^```(?:python)?\n?", "", code)
    code = re.sub(r"\n?```$", "", code)
    return code, script_path


def generate_dashboard_code(item: dict, context: str) -> list[tuple[str, Path]]:
    """대시보드 코드 생성 → [(코드, 경로), ...] 반환."""
    prompt = f"""EPL 스카우트 대시보드 Streamlit 페이지를 작성하세요.

## 액션아이템
{item['content']}

## 기존 코드
{context[:4500] if context else '(없음)'}

## 규칙
- EPL_PURPLE="#37003c", EPL_MAGENTA="#e90052", EPL_GREEN="#00ff87", EPL_CYAN="#04f5ff"
- @st.cache_data(ttl=3600) 사용
- render() 함수로 최상위 노출
- 한국어 주석
- 파일 경로 주석으로 첫 줄에 명시: # FILE: dashboard/pages/파일명.py

완성된 Python 파일 전체를 코드 블록 없이 바로 작성하세요 (```python 없이):"""

    resp = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=6000,
        messages=[{"role": "user", "content": prompt}],
    )
    code = resp.content[0].text.strip()
    code = re.sub(r"^```(?:python)?\n?", "", code)
    code = re.sub(r"\n?```$", "", code)

    # 첫 줄에서 파일 경로 추출
    first_line = code.split("\n")[0]
    path_match = re.search(r"FILE:\s*([\w/\-\.]+\.py)", first_line)
    if path_match:
        rel_path = path_match.group(1)
    else:
        rel_path = "dashboard/pages/auto_generated.py"

    return [(code, ROOT / rel_path)]


# ────────────────────────────────────────────────────────────────────
# 실행
# ────────────────────────────────────────────────────────────────────

def run_training(script_path: Path) -> bool:
    """생성된 학습 스크립트 실행."""
    print(f"  🚀 학습 실행: {script_path.name}")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=1500,    # 25분
    )
    print(result.stdout[-1000:] if result.stdout else "")
    if result.returncode != 0:
        print(f"  ❌ 학습 실패:\n{result.stderr[-500:]}")
        return False
    print("  ✅ 학습 완료")
    return True


def main() -> None:
    print("=" * 60)
    print("🤖 GitHub Actions Daily Dev Loop (with 모델 학습)")
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

    for idx, item in enumerate(items, 1):
        print(f"[{idx}/{len(items)}] {item['content'][:70]}")
        print(f"       타입: {item['type']} | 담당: {item['assignee']}")
        context = get_context(item)

        if item["type"] == "model":
            code, script_path = generate_train_script(item, context)
            script_path.write_text(code, encoding="utf-8")
            print(f"  📝 스크립트 생성: {script_path.relative_to(ROOT)}")
            success = run_training(script_path)
            results_log.append({
                "item": item["content"][:50],
                "type": "model",
                "script": str(script_path.relative_to(ROOT)),
                "success": success,
            })
        else:
            files = generate_dashboard_code(item, context)
            for code, path in files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(code, encoding="utf-8")
                print(f"  📝 대시보드 파일: {path.relative_to(ROOT)}")
            results_log.append({
                "item": item["content"][:50],
                "type": "dashboard",
                "files": [str(p.relative_to(ROOT)) for _, p in files],
                "success": True,
            })
        print()

    # 실행 결과 요약 저장
    import json
    summary_path = ROOT / "reports" / "daily_meeting" / f"{date.today()}_dev_result.json"
    summary_path.write_text(
        json.dumps({"date": str(date.today()), "results": results_log}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 60)
    ok = sum(1 for r in results_log if r["success"])
    print(f"✅ 완료: {ok}/{len(results_log)} 성공")
    print("=" * 60)


if __name__ == "__main__":
    main()

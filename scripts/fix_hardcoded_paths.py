# fix_hardcoded_paths.py - 모델 train 스크립트 하드코딩 경로 일괄 수정
# 실행: python scripts/fix_hardcoded_paths.py
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TARGETS = [
    "models/p1_match_result/train.py",
    "models/p1_match_result/train_pipeline.py",
    "models/p2_goal_scoring/train.py",
    "models/p3_relegation/train.py",
    "models/p3_relegation/train_relegation.py",
    "models/p4_mvp/mvp_pipeline.py",
    "models/p4_mvp_scoring/train.py",
    "models/p6_market_value/train.py",
    "models/s1_player_rating/train.py",
    "models/s1_player_rating/train_v2.py",
    "models/s1_player_rating/train_v3.py",
    "models/s2_market_value/train.py",
    "models/s2_market_value/train_v2.py",
    "models/s2_market_value/train_v3.py",
    "models/s2_market_value/train_v4.py",
    "models/s4_growth/age_performance_profiles.py",
    "models/s4_growth/train.py",
    "models/s4_growth/train_v2.py",
    "models/s4_growth/train_v3.py",
    "models/s4_growth/train_v4.py",
    "models/s5_transfer_adapt/train.py",
    "models/s5_transfer_adapt/train_v2.py",
    "models/s5_transfer_adapt/train_v3.py",
    "models/s6_decline/fix_s6_results.py",
    "models/s6_decline/train.py",
    "models/s6_decline/train_v2.py",
    "models/s6_decline/train_v3.py",
    "models/s6_decline/train_v4.py",
    "models/p7_growth_curve/train_v3.py",
    "models/p7_growth_curve/train_v4.py",
]

RELATIVE_ROOT = "Path(__file__).resolve().parent.parent.parent"
PAT_IMPORT = re.compile(r"from pathlib import.*\bPath\b|import pathlib")

# 패턴 1: VAR = Path("C:/Users/xcv54/...") 또는 Path(r"C:\Users\xcv54\...")
PAT1 = re.compile(
    r'(\w+)\s*=\s*Path\s*\(\s*r?"[^"]*xcv54[^"]*"\s*\)',
    re.IGNORECASE,
)
# 패턴 2: VAR = r"C:/Users/xcv54/..." (Path 없이 문자열만)
PAT2 = re.compile(
    r'(\w+)\s*=\s*r?"[^"]*xcv54[^"]*"',
    re.IGNORECASE,
)


def fix_file(rel_path):
    fpath = ROOT / rel_path
    if not fpath.exists():
        return False, "파일 없음"

    src = fpath.read_text(encoding="utf-8", errors="ignore")
    original = src
    changed_vars = set()

    # 패턴 1 교체: VAR = Path("...xcv54...") → VAR = Path(__file__).resolve()...
    def repl1(m):
        changed_vars.add(m.group(1))
        return f"{m.group(1)} = {RELATIVE_ROOT}"
    src = PAT1.sub(repl1, src)

    # 패턴 2 교체: VAR = "...xcv54..." → VAR = Path(__file__).resolve()...
    # (이미 패턴1에서 처리된 줄은 xcv54가 없으므로 중복 처리 안 됨)
    def repl2(m):
        var = m.group(1)
        # 변수명이 DATA/MODEL/SCOUT/FIG 관련이면 스킵 (서브 경로들)
        if any(k in var.upper() for k in ("DATA", "MODEL", "SCOUT", "FIG", "OUT", "LOG")):
            return m.group(0)  # 변경 안 함
        changed_vars.add(var)
        return f"{var} = {RELATIVE_ROOT}"
    src = PAT2.sub(repl2, src)

    # os.path.join(BASE_DIR, ...) 패턴이 남아있으면 Path 연산자로 교체
    for var in changed_vars:
        # os.path.join(VAR, "a", "b") → VAR / "a" / "b"
        def make_repl(v):
            def repl_join(m):
                args = m.group(1).split(",")
                # 첫 인자가 변수명이면 제거하고 나머지를 / 연결
                if args and args[0].strip() == v:
                    parts = [a.strip().strip('"').strip("'") for a in args[1:]]
                    return v + " / " + " / ".join(f'"{p}"' for p in parts)
                return m.group(0)
            return repl_join
        src = re.sub(
            rf'os\.path\.join\(({re.escape(var)}(?:[^)]*?))\)',
            make_repl(var),
            src,
        )

    # pathlib import 없으면 추가
    if src != original and not PAT_IMPORT.search(src):
        src = "from pathlib import Path\n" + src

    if src == original:
        return False, "변경 없음 (패턴 미탐지)"

    # 문법 검사
    try:
        ast.parse(src)
    except SyntaxError as e:
        # 롤백
        return False, f"SyntaxError: {e}"

    fpath.write_text(src, encoding="utf-8")
    return True, f"수정완료 ({', '.join(sorted(changed_vars))})"


def main():
    print("=" * 65)
    print("  하드코딩 경로 일괄 수정")
    print("=" * 65)

    changed, skipped, errors = [], [], []
    for rel in TARGETS:
        ok, msg = fix_file(rel)
        icon = "✅" if ok else ("⚠️ " if "없음" in msg or "미탐지" in msg else "❌")
        print(f"  {icon} {rel:<52} {msg}")
        if ok:
            changed.append(rel)
        elif "SyntaxError" in msg:
            errors.append((rel, msg))
        else:
            skipped.append(rel)

    print("=" * 65)
    print(f"  수정: {len(changed)}개 | 스킵: {len(skipped)}개 | 오류: {len(errors)}개")
    print("=" * 65)

    if errors:
        print("\n수동 확인 필요:")
        for rel, msg in errors:
            print(f"  - {rel}: {msg}")
        sys.exit(1)

    return changed


if __name__ == "__main__":
    main()

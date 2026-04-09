"""FBref 데이터 품질 검수 스크립트"""
import pandas as pd
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "fbref")

# ── 1. Squad Stats 로드 ──
squad_files = glob.glob(os.path.join(BASE, "**", "squad_stats.csv"), recursive=True)
print(f"Squad stats 파일 수: {len(squad_files)}")

all_squads = []
for f in squad_files:
    try:
        df = pd.read_csv(f, encoding="utf-8")
        all_squads.append(df)
    except Exception as e:
        print(f"  읽기 실패: {f} -> {e}")

if not all_squads:
    print("squad_stats 데이터 없음!")
    sys.exit(1)

sq = pd.concat(all_squads, ignore_index=True)
print(f"총 행수: {len(sq)}")
print(f"컬럼수: {len(sq.columns)}")
print(f"컬럼: {sq.columns.tolist()}")
print()

# ── 2. 선수 이름 이슈 ──
print("=" * 60)
print("1. 선수 이름 이슈")
print("=" * 60)

# 인코딩 깨짐 (mojibake 패턴)
mojibake_pat = re.compile(r"Ã©|Ã¡|Ã³|Ã¼|Ã±|Ã§|Ã¶|Ã¸|Ä|Å|Ã‰|Ã–|Ãº|Ã­|Ã¢|Ã£|Ã®|Ã¯|Ã°|Ã¤|Ã¦|ÃŸ|Ã")
broken = sq[sq["player"].apply(lambda x: bool(mojibake_pat.search(str(x))))]
print(f"인코딩 깨진 이름: {len(broken)}개 ({len(broken)/len(sq)*100:.1f}%)")
if len(broken) > 0:
    uniq = broken["player"].unique()
    print(f"  고유 깨진 이름: {len(uniq)}개")
    print("  샘플:")
    for n in uniq[:20]:
        print(f"    {repr(n)}")
print()

# 특수문자 이름 (정상 유니코드 악센트)
accent_pat = re.compile(r"[àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ]", re.IGNORECASE)
accent_names = sq[sq["player"].apply(lambda x: bool(accent_pat.search(str(x))))]
print(f"악센트 포함 (정상 유니코드): {len(accent_names)}개")
if len(accent_names) > 0:
    for n in accent_names["player"].unique()[:10]:
        print(f"    {n}")
print()

# 빈/NaN 이름
empty = sq[sq["player"].isna() | (sq["player"].astype(str).str.strip() == "")]
print(f"빈/NaN 이름: {len(empty)}개")
print()

# ── 3. 포지션 이슈 ──
print("=" * 60)
print("2. 포지션 이슈")
print("=" * 60)

pos_col = None
for c in ["pos", "position", "Pos"]:
    if c in sq.columns:
        pos_col = c
        break

if pos_col:
    print(f"포지션 컬럼: '{pos_col}'")
    null_cnt = sq[pos_col].isna().sum()
    empty_cnt = (sq[pos_col].astype(str).str.strip() == "").sum()
    print(f"결측(NaN): {null_cnt}개 ({null_cnt/len(sq)*100:.1f}%)")
    print(f"빈문자열: {empty_cnt}개")
    print(f"고유값 수: {sq[pos_col].nunique()}")
    print("값 분포:")
    for v, c in sq[pos_col].value_counts(dropna=False).head(20).items():
        print(f"    {repr(v):30s}: {c}개")
else:
    print("포지션 컬럼 없음! 가능 컬럼들:")
    for c in sq.columns:
        if "pos" in c.lower():
            print(f"    {c}")
print()

# ── 4. 선발/출장 이슈 ──
print("=" * 60)
print("3. 선발/출장/시간 이슈")
print("=" * 60)

check_cols = ["mp", "starts", "min", "games", "playing_time_mp",
              "playing_time_starts", "playing_time_min", "90s"]
for col in check_cols:
    if col in sq.columns:
        null_cnt = sq[col].isna().sum()
        # 타입 확인
        non_numeric = 0
        if sq[col].dtype == object:
            non_numeric = sq[col].apply(lambda x: not str(x).replace(".", "").replace("-", "").isdigit() if pd.notna(x) else False).sum()
        print(f"  {col:25s}: dtype={str(sq[col].dtype):10s} | 결측={null_cnt:5d} | 비숫자={non_numeric:5d} | 샘플={sq[col].dropna().head(3).tolist()}")
print()

# ── 5. 매치로그 검사 ──
print("=" * 60)
print("4. 매치로그 이슈")
print("=" * 60)

ml_files = glob.glob(os.path.join(BASE, "**", "matchlogs", "*.csv"), recursive=True)
print(f"매치로그 파일 수: {len(ml_files)}")

if ml_files:
    sample_files = ml_files[:5] + ml_files[-5:]
    all_ml = []
    for f in ml_files[:200]:  # 200개만 샘플
        try:
            df = pd.read_csv(f, encoding="utf-8")
            all_ml.append(df)
        except:
            pass

    if all_ml:
        ml = pd.concat(all_ml, ignore_index=True)
        print(f"샘플 매치로그 행수: {len(ml)}")
        print(f"컬럼: {ml.columns.tolist()}")
        print()

        # 선수명 깨짐
        if "player" in ml.columns:
            ml_broken = ml[ml["player"].apply(lambda x: bool(mojibake_pat.search(str(x))))]
            print(f"매치로그 이름 깨짐: {ml_broken['player'].nunique()}명")
            if len(ml_broken) > 0:
                for n in ml_broken["player"].unique()[:10]:
                    print(f"    {repr(n)}")
        print()

        # 주요 컬럼 결측
        for col in ["date", "comp", "result", "gls", "ast", "min", "started"]:
            if col in ml.columns:
                null_cnt = ml[col].isna().sum()
                print(f"  {col:15s}: 결측={null_cnt:5d}/{len(ml)} ({null_cnt/len(ml)*100:.1f}%)")
print()

# ── 6. 전체 요약 ──
print("=" * 60)
print("5. 전체 요약")
print("=" * 60)
print(f"Squad stats: {len(squad_files)}파일, {len(sq)}행, {sq['player'].nunique()}명")
print(f"매치로그: {len(ml_files)}파일")

# 시즌 커버리지
if "season" in sq.columns:
    seasons = sorted(sq["season"].unique())
    print(f"시즌: {seasons[0]} ~ {seasons[-1]} ({len(seasons)}시즌)")
    for s in seasons:
        cnt = len(sq[sq["season"] == s])
        teams = sq[sq["season"] == s]["team"].nunique() if "team" in sq.columns else "?"
        print(f"    {s}: {cnt}명, {teams}팀")

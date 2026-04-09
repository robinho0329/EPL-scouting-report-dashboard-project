"""이번 크롤에서 잘못 저장된 이미지 제거 스크립트.

동일한 portrait URL을 공유하는 선수들 = 엉뚱한 이미지가 저장된 것.
해당 DB 레코드 + 파일 삭제 → 다음 크롤 시 재시도.
"""
import sqlite3
import hashlib
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent
db = ROOT / "data" / "image_crawl_checkpoint.db"
conn = sqlite3.connect(db)

# 이번 크롤(2026-04-08 10:xx)에서 추가된 레코드 조회
rows = conn.execute(
    "SELECT key, local_path, url FROM downloaded WHERE downloaded_at >= '2026-04-08 10:00:00'"
).fetchall()
print(f"이번 크롤 레코드: {len(rows)}개")

# URL 중복 카운트 → 동일 URL = 잘못된 이미지
url_counter = Counter(r[2] for r in rows)
wrong_urls = {url for url, cnt in url_counter.items() if cnt > 1}
print(f"중복 URL (잘못된 이미지): {len(wrong_urls)}개")
for u in sorted(wrong_urls):
    cnt = url_counter[u]
    print(f"  {cnt}명이 공유: {u.split('/')[-1]}")

# 삭제 대상 선별
to_delete = [(key, path) for key, path, url in rows if url in wrong_urls]
print(f"\n삭제 대상: {len(to_delete)}개")

confirm = input("\n삭제하시겠습니까? (y/n): ").strip().lower()
if confirm != "y":
    print("취소됨.")
    conn.close()
    exit()

deleted_files, deleted_db = 0, 0
for key, path in to_delete:
    # 파일 삭제
    p = Path(path)
    if p.exists():
        p.unlink()
        deleted_files += 1
    # DB 레코드 삭제
    conn.execute("DELETE FROM downloaded WHERE key=?", (key,))
    deleted_db += 1

conn.commit()
conn.close()
print(f"\n완료 — 파일 {deleted_files}개, DB 레코드 {deleted_db}개 삭제")
print("이제 크롤러를 다시 실행하면 삭제된 선수들만 재시도합니다.")

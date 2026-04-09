import sqlite3
import hashlib
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent
db = ROOT / "data" / "image_crawl_checkpoint.db"
conn = sqlite3.connect(db)

# 전체 현황
total = conn.execute("SELECT COUNT(*) FROM downloaded").fetchone()[0]
player_dir = ROOT / "data" / "images" / "players"
imgs = list(player_dir.glob("*.jpg")) + list(player_dir.glob("*.png"))
print(f"총 downloaded 레코드: {total}")
print(f"실제 이미지 파일 수: {len(imgs)}")

# Son Heung-min 확인
name = "Son Heung-min"
key = f"player_{hashlib.md5(name.encode()).hexdigest()[:8]}_{name}"
r = conn.execute("SELECT local_path, url FROM downloaded WHERE key=?", (key,)).fetchone()
son_file = player_dir / "Son Heung-min.jpg"
print(f"\n[Son Heung-min]")
print(f"  DB 등록: {r}")
print(f"  파일 존재: {son_file.exists()}")
if son_file.exists():
    print(f"  파일 크기: {son_file.stat().st_size:,} bytes")

# 이번 크롤 결과 (최근 추가분)
recent = conn.execute(
    "SELECT key, url FROM downloaded WHERE downloaded_at >= '2026-04-08 10:00:00'"
).fetchall()
print(f"\n이번 크롤 신규 등록: {len(recent)}개")

# 중복 URL 검사
url_counter = Counter(r[1] for r in recent)
dup_urls = {url: cnt for url, cnt in url_counter.items() if cnt > 1}
print(f"중복 URL (잘못된 이미지): {len(dup_urls)}개")
if dup_urls:
    for url, cnt in sorted(dup_urls.items(), key=lambda x: -x[1]):
        print(f"  {cnt}명 공유: ...{url[-40:]}")

# 로그 마지막 15줄
log = ROOT / "logs" / "image_crawler.log"
lines = log.read_text(encoding="utf-8").splitlines()
print(f"\n로그 마지막 15줄:")
for l in lines[-15:]:
    print(l)

conn.close()

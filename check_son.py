import sqlite3
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 1. checkpoint DB 확인
db_path = ROOT / "data" / "image_crawl_checkpoint.db"
conn = sqlite3.connect(db_path)
total = conn.execute("SELECT COUNT(*) FROM downloaded").fetchone()[0]
print(f"총 downloaded 레코드: {total}")

# Son 관련 검색
rows = conn.execute("SELECT key, local_path FROM downloaded WHERE key LIKE '%Son%'").fetchall()
print(f"'Son' 포함 레코드: {len(rows)}")
for r in rows:
    print(" ", r)

# Son Heung-min 정확한 키 계산
import hashlib
player_name = "Son Heung-min"
key = f"player_{hashlib.md5(player_name.encode()).hexdigest()[:8]}_{player_name}"
print(f"\nSon Heung-min 예상 키: {key}")
found = conn.execute("SELECT * FROM downloaded WHERE key=?", (key,)).fetchone()
print(f"DB에 있음: {found}")
conn.close()

# 2. image_map.parquet 확인
img_map_path = ROOT / "data" / "images" / "image_map.parquet"
if img_map_path.exists():
    df = pd.read_parquet(img_map_path)
    print(f"\nimage_map 컬럼: {df.columns.tolist()}")
    print(f"image_map 전체 행수: {len(df)}")
    son_rows = df[df.apply(lambda r: 'Son' in str(r.values), axis=1)]
    print(f"'Son' 포함 행수: {len(son_rows)}")
    print(son_rows)
else:
    print("\nimage_map.parquet 없음")

# 3. 실제 파일 확인
img_dir = ROOT / "data" / "images" / "players"
files = list(img_dir.glob("*Son*")) + list(img_dir.glob("*son*")) + list(img_dir.glob("*Heung*"))
print(f"\n선수 이미지 폴더에서 Son/Heung 파일: {files}")
print(f"\n총 선수 이미지 파일 수: {len(list(img_dir.glob('*.jpg'))) + len(list(img_dir.glob('*.png')))}")

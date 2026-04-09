"""image_map.parquet 재생성 스크립트.

data/images/players/ 폴더의 실제 파일 기준으로 player→image_path 매핑을 재생성.
기존 3380개 → 4459개로 업데이트.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLAYERS_DIR = ROOT / "data" / "images" / "players"
IMG_MAP_PATH = ROOT / "data" / "images" / "image_map.parquet"

files = list(PLAYERS_DIR.glob("*.jpg")) + list(PLAYERS_DIR.glob("*.png"))
rows = [{"player": f.stem, "image_path": str(f)} for f in files]
df = pd.DataFrame(rows)
df.to_parquet(IMG_MAP_PATH, index=False)
print(f"image_map.parquet 재생성 완료: {len(df)}개")

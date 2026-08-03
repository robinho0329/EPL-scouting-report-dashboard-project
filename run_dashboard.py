"""Launch script for the EPL dashboard."""
from pathlib import Path
import subprocess
import sys

# 프로젝트 루트 — 절대경로를 박으면 다른 PC에서 클론했을 때 이 줄에서 멈춘다
_PROJECT_ROOT = Path(__file__).resolve().parent

subprocess.run([
    sys.executable, "-m", "streamlit", "run",
    str(_PROJECT_ROOT / "dashboard" / "app.py"),
    "--server.headless", "true",
    "--server.port", "8520",
])
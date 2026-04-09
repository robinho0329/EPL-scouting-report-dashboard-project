"""Launch script for the EPL dashboard."""
import subprocess
import sys

subprocess.run([
    sys.executable, "-m", "streamlit", "run",
    r"C:\Users\xcv54\workspace\EPL project\dashboard\app.py",
    "--server.headless", "true",
    "--server.port", "8520",
])

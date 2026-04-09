@echo off
cd /d "C:\Users\xcv54\workspace\EPL project"
".venv\Scripts\python.exe" -m streamlit run dashboard/app.py --server.port 8520 --server.headless true

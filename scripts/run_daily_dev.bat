@echo off
:: EPL 스카우트 일일 개발 루프 실행 배치파일
:: Windows 작업 스케줄러에서 매일 09:15 KST 실행

cd /d "C:\Users\xcv54\workspace\EPL project"
call .venv\Scripts\activate.bat
python scripts\daily_dev_loop.py >> reports\dev_loop_logs\scheduler.log 2>&1

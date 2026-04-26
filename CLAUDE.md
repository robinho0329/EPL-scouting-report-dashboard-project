# EPL 스카우트 프로젝트 — Claude 지침

## 🚨 세션 시작 시 필수 작업

**이 프로젝트 세션이 시작될 때마다 아래를 자동으로 실행하고 결과를 채팅에 출력할 것.**

### 오늘의 미팅 & 개발 결과 자동 요약

아래 두 파일을 GitHub API로 읽어서 즉시 요약 출력:

1. **미팅 노트**: `reports/daily_meeting/YYYY-MM-DD_meeting.md` (오늘 날짜)
2. **개발 결과**: `reports/daily_meeting/YYYY-MM-DD_dev_result.json` (오늘 날짜)

GitHub API URL:
```
https://api.github.com/repos/robinho0329/EPL-scouting-report-dashboard-project/contents/reports/daily_meeting/YYYY-MM-DD_meeting.md
https://api.github.com/repos/robinho0329/EPL-scouting-report-dashboard-project/contents/reports/daily_meeting/YYYY-MM-DD_dev_result.json
```

출력 형식:
```
---
## 📅 [날짜] EPL 스카우트팀 데일리 리포트

### 🤝 오늘의 미팅 요약
- 김태현 스카우트 평가: X.X / 5.0
- 주요 논의 포인트 (3줄 이내 핵심만)

### 🎯 액션아이템 Top 3
1. [담당자] 내용 (목표 지표)
2. ...
3. ...

### 🤖 자동 학습 결과 (GitHub Actions)
| 모델 | 스크립트 | 결과 |
|------|---------|------|
| ...  | ...     | ✅/❌ |

### 📊 Streamlit 앱
https://epl-scouting-report-dashboard-project-ffyb8msh6uafxyyg8txsm8.streamlit.app
---
```

파일이 없으면: "오늘 미팅 노트 또는 개발 결과가 없습니다 (GitHub Actions 실행 전이거나 미팅 없는 날)." 출력 후 바로 대화 시작.

base64 인코딩 파일은 디코딩해서 읽을 것.

---

## 프로젝트 개요

- **아키텍처**: 크롤링 → 파이프라인 → 피처 엔지니어링 → ML 14종 → Streamlit 대시보드
- **모델**: p1~p8 (예측), s1~s6 (스카우트)
- **파이프라인 순서**: aggregate → preprocess → scout_features (순차 필수)
- **레이트 리밋**: FBref 6초, Transfermarkt 5초

## 자동화 루틴

| 구분 | 방식 | 시간 |
|------|------|------|
| 미팅 시뮬레이션 | CCR RemoteTrigger | 매일 09:00 KST |
| 모델 학습 + Streamlit 체크 | GitHub Actions | 매일 09:15 KST (cron: 20:50 UTC, 지연 3h24m 역산) |
| 로컬 실행 (PC 켤 때) | Windows Task Scheduler | 매일 09:15 KST |

- **학습**: 기존 train 스크립트 직접 실행 (Claude API 불필요)
- **Streamlit 체크**: playwright 헤드리스 브라우저 (두 환경 동일)
- **에러 자동수정**: 로컬에서만 Claude CLI로 수정

## Streamlit Cloud

URL: `https://epl-scouting-report-dashboard-project-ffyb8msh6uafxyyg8txsm8.streamlit.app`
GitHub: `robinho0329/EPL-scouting-report-dashboard-project`

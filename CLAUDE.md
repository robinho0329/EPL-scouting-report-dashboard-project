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
| **데일리 PPT 생성** | 미팅 루틴 5단계 (아래 참고) | 매일 09:00 KST (미팅 노트 생성 직후) |

- **학습**: 기존 train 스크립트 직접 실행 (Claude API 불필요)
- **Streamlit 체크**: playwright 헤드리스 브라우저 (두 환경 동일)
- **에러 자동수정**: 로컬에서만 Claude CLI로 수정

## PPT Team Agent

### 아키텍처
```
미팅 노트 + results_summary.json
    ↓
ppt-organizer 에이전트   → reports/portfolio_ppt/slide-outline.md
    ↓
ppt-designer 에이전트    → reports/portfolio_ppt/slides/slide-NN.html
    ↓
node reports/portfolio_ppt/build_ppt.js
    ↓
reports/portfolio_ppt/output/EPL_Scout_Briefing_YYYY-MM-DD.pptx
```

### 에이전트 파일
| 파일 | 역할 |
|------|------|
| `.claude/agents/ppt-organizer.md` | 데이터 읽기 → slide-outline.md 생성 |
| `.claude/agents/ppt-designer.md` | outline → slide-NN.html 생성 |
| `.claude/skills/design-skill/SKILL.md` | HTML 디자인 시스템 |
| `.claude/skills/pptx-skill/scripts/html2pptx.cjs` | Playwright 렌더링 → PPTX 변환 |
| `reports/portfolio_ppt/build_ppt.js` | 빌드 진입점 |

### HTML 슬라이드 작성 규칙 (ppt-designer 필수 준수)
- **텍스트는 반드시 `<p>`, `<h1>`~`<h6>`, `<ul>`, `<ol>` 태그로 감쌀 것** — `<div>` 안에 직접 텍스트 금지
- 슬라이드 크기: `width: 720pt; height: 405pt` (body에 고정)
- 외부 폰트 CDN은 허용하나, ERR_CERT 환경에서도 fallback 폰트로 렌더링 가능하게 설정
- JS 없이 정적 HTML만 사용

### 데일리 PPT 생성 수동 호출
"PPT 만들어줘" 또는 "데일리 브리핑 PPT" → ppt-organizer → ppt-designer → build_ppt.js 순으로 자동 실행

## Streamlit Cloud

URL: `https://epl-scouting-report-dashboard-project-ffyb8msh6uafxyyg8txsm8.streamlit.app`
GitHub: `robinho0329/EPL-scouting-report-dashboard-project`

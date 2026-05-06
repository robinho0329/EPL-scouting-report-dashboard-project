# EPL 스카우트 프로젝트 — Claude 지침

## 🚨 세션 시작 시 필수 작업 (매일 09:00 KST 자동 실행)

**이 프로젝트 세션이 시작될 때마다 아래 5단계를 순서대로 실행할 것.**  
각 단계 완료 후 결과를 채팅에 출력하고 다음 단계로 진행.

---

### 1단계: 현재 상태 파악

다음을 병렬로 읽어 현황 파악:
- `models/*/results_summary.json` — 12개 모델 핵심 지표 (R², AUC, Acc, MAE)
- `dashboard/app.py` MENU_OPTIONS — 현재 페이지 수
- `reports/daily_meeting/` — 직전 미팅 노트 (어제 날짜 meeting.md)
- `reports/daily_meeting/YYYY-MM-DD_dev_result.json` (어제 날짜) — 어젯밤 Actions 결과

---

### 2단계: 팀 미팅 시뮬레이션

**김태현 스카우트**와 **Marcus Webb (Analytics Agent)** 대화 형식으로 작성:
1. 모델 현황 리뷰 — 가장 약한 모델 식별 및 원인 진단
2. 어제 액션아이템 후속 점검 (달성/미달/블로킹 항목)
3. 대시보드 실무 활용도 평가 (영입 회의에서 쓸 수 있나?)
4. 오늘 개발 우선순위 Top 3 합의 — 구체적 목표 지표 포함

---

### 3단계: 미팅 노트 파일 생성

`reports/daily_meeting/YYYY-MM-DD_meeting.md` 작성:
- 섹션 0: 액션아이템 Top 3 (GitHub Actions가 읽는 섹션 — 반드시 포함)
- 섹션 1: 모델 성능 현황 표 (12개 모델, 신호등 상태)
- 섹션 2: 팀 토론 내용 (2단계 내용)
- 섹션 3: 액션아이템 상세 (담당/목표 지표/기한)
- 섹션 4: 김태현 스카우트 종합 평가 (X.X/5.0 + 5.0 로드맵)

---

### 4단계: Git 커밋 및 푸시

```bash
git config user.email epl-scout-bot@anthropic.com
git config user.name EPL Scout Bot
git remote set-url origin https://${GITHUB_TOKEN}@github.com/robinho0329/EPL-scouting-report-dashboard-project
git add reports/daily_meeting/
git commit -m "daily: YYYY-MM-DD 아침 스카우트팀 미팅 결과"
git push origin HEAD:master
```

---

### 5단계: 데일리 PPT 자동 생성

미팅 노트 push 직후 PPT Team Agent를 순서대로 실행:

**5-1. ppt-organizer 에이전트 호출**
- 오늘 meeting.md + 12개 모델 results_summary.json 읽기
- 데일리 브리핑 6슬라이드 구조 설계
- `reports/portfolio_ppt/slide-outline.md` 생성

**5-2. ppt-designer 에이전트 호출**
- slide-outline.md 읽어 EPL 다크테마 HTML 슬라이드 생성
- `reports/portfolio_ppt/slides/slide-01.html` ~ `slide-06.html` 생성
- 슬라이드 구성:
  - 01: 표지 (날짜, 제목)
  - 02: 모델 성능 현황표 (신호등 테이블)
  - 03: 어제 달성/미달 체크리스트
  - 04: 오늘 액션아이템 Top 3 (카드 레이아웃)
  - 05: 김태현 종합 평가 (X.X/5.0 게이지 + 5.0 로드맵)
  - 06: Streamlit 대시보드 현황 (페이지 수 + URL)

**5-3. PPTX 빌드**
```bash
node reports/portfolio_ppt/build_ppt.js
```
→ `reports/portfolio_ppt/output/EPL_Scout_Briefing_YYYY-MM-DD.pptx` 생성

**5-4. 결과 push**
```bash
git add reports/portfolio_ppt/output/
git commit -m "ppt: YYYY-MM-DD 데일리 브리핑 PPT 생성"
git push origin HEAD:master
```

**5단계 완료 후 채팅 출력 형식:**
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

### 🖼️ 데일리 PPT
reports/portfolio_ppt/output/EPL_Scout_Briefing_YYYY-MM-DD.pptx (N슬라이드)
---
```

**파일이 없는 경우:** "오늘 미팅 노트 또는 개발 결과가 없습니다." 출력 후 1단계부터 실행.  
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

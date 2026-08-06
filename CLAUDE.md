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
git add reports/daily_meeting/
git commit -m "daily: YYYY-MM-DD 아침 스카우트팀 미팅 결과"
git push origin HEAD:master
```

> 원격 URL에 토큰을 박지 말 것. 자동 실행 경로는 GitHub Actions이며
> 푸시는 워크플로의 `GITHUB_TOKEN`으로 처리된다. 로컬에서는 기존 자격증명을 쓴다.

---

### 5단계: 데일리 PPT 자동 생성

미팅 노트 push 직후 PPT Team Agent를 순서대로 실행:

**5-0. 웹 리서치 에이전트 호출 (신규 — 매일 실행)**
- 오늘 미팅 주요 주제(모델 성능 현황, 스카우트 인텔리전스, 스포츠 애널리틱스 브리핑 등)를 키워드로 추출
- general-purpose 에이전트가 WebSearch로 유사한 PPT/프레젠테이션 레퍼런스를 구글링
  - 검색 예시: "sports analytics daily briefing presentation template", "football scouting report PPT", "data science sprint review slide deck"
  - 최소 3개 레퍼런스 수집 목표
- 각 레퍼런스에서 다음을 파악:
  - 슬라이드 총 페이지 수
  - 페이지별 콘텐츠 구성 패턴 (표지 → 요약 → 상세 → 액션아이템 등)
  - 시각화 유형 (테이블/게이지/카드/타임라인 등)
- 수집 결과를 `reports/portfolio_ppt/ppt_references.md`에 저장

**5-1. ppt-organizer 에이전트 호출**
- 오늘 meeting.md + 12개 모델 results_summary.json 읽기
- `reports/portfolio_ppt/ppt_references.md` 읽기 — 레퍼런스 구조를 반드시 참고
- 레퍼런스 기반으로 최적 슬라이드 수(통상 6~10장) 및 페이지별 구성 결정
- `reports/portfolio_ppt/slide-outline.md` 생성 (레퍼런스 출처 명시 포함)

**5-2. ppt-designer 에이전트 호출**
- slide-outline.md 읽어 EPL 다크테마 HTML 슬라이드 생성
- 슬라이드 수는 outline에서 결정된 수에 따름 (고정 6장 아님)
- 기본 포함 항목 (레퍼런스에 따라 순서/분량 조정 가능):
  - 표지 (날짜, 제목, 미팅 회차)
  - 모델 성능 현황표 (신호등 테이블)
  - 어제 달성/미달 체크리스트
  - 오늘 액션아이템 Top 3 (카드 레이아웃)
  - 김태현 종합 평가 (X.X/5.0 게이지 + 5.0 로드맵)
  - Streamlit 대시보드 현황 (페이지 수 + URL)

**5-3. PPTX 빌드**
```bash
node reports/portfolio_ppt/build_ppt.js
```
→ `reports/portfolio_ppt/output/EPL_Scout_Briefing_YYYY-MM-DD.pptx` 생성

**5-4. 결과 push**
```bash
git add -f reports/portfolio_ppt/output/ reports/portfolio_ppt/slides/ reports/portfolio_ppt/ppt_references.md
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
| 미팅 노트 생성 | GitHub Actions (`scripts/generate_meeting_note.py`) | 매일 09:15 KST — 학습 단계 직전 |
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
general-purpose 에이전트 (WebSearch)
  → 유사 PPT 레퍼런스 3개 이상 수집
  → reports/portfolio_ppt/ppt_references.md 저장
    ↓
ppt-organizer 에이전트
  → ppt_references.md 참고하여 최적 슬라이드 수·구조 결정
  → reports/portfolio_ppt/slide-outline.md 생성
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
| `general-purpose` | WebSearch로 유사 PPT 레퍼런스 수집 |
| `.claude/agents/ppt-organizer.md` | 레퍼런스 참고 + 데이터 읽기 → slide-outline.md 생성 |
| `.claude/agents/ppt-designer.md` | outline → slide-NN.html 생성 |
| `.claude/skills/design-skill/SKILL.md` | HTML 디자인 시스템 |
| `.claude/skills/pptx-skill/scripts/html2pptx.cjs` | Playwright 렌더링 → PPTX 변환 |
| `reports/portfolio_ppt/build_ppt.js` | 빌드 진입점 |

### 레퍼런스 수집 기준
- 검색 키워드: 오늘 미팅 주제에서 추출 (예: "sports analytics briefing", "scouting report deck", "ML model performance review presentation")
- 레퍼런스당 수집 항목: 슬라이드 총 수, 페이지별 섹션명, 주요 시각화 유형
- 저장 경로: `reports/portfolio_ppt/ppt_references.md` (날짜별로 덮어쓰기)

### HTML 슬라이드 작성 규칙 (ppt-designer 필수 준수)
- **텍스트는 반드시 `<p>`, `<h1>`~`<h6>`, `<ul>`, `<ol>` 태그로 감쌀 것** — `<div>` 안에 직접 텍스트 금지
- 슬라이드 크기: `width: 720pt; height: 405pt` (body에 고정)
- 외부 폰트 CDN은 허용하나, ERR_CERT 환경에서도 fallback 폰트로 렌더링 가능하게 설정
- JS 없이 정적 HTML만 사용

### 데일리 PPT 생성 수동 호출
"PPT 만들어줘" 또는 "데일리 브리핑 PPT" → general-purpose(웹 리서치) → ppt-organizer → ppt-designer → build_ppt.js 순으로 자동 실행

## Streamlit Cloud

URL: `https://epl-scouting-report-dashboard-project-ffyb8msh6uafxyyg8txsm8.streamlit.app`
GitHub: `robinho0329/EPL-scouting-report-dashboard-project`

# ⚽ EPL Scout Intelligence Dashboard

> EPL(잉글리시 프리미어리그) 스카우팅 & 이적 의사결정 지원 시스템  
> FBref + Transfermarkt 데이터 기반 14개 ML 모델 + Streamlit 대시보드

---

## 📌 프로젝트 개요

"**김태현 스카우트**" 페르소나를 기반으로 설계된 실무형 스카우팅 분석 시스템입니다.  
중위권 구단 예산(€30~50M/시즌) 기준으로 WAR·시장가치·이적 리스크를 통합해  
선수 영입 제안 전 종합 의사결정을 지원합니다.

---

## 🏗️ 시스템 아키텍처

```
[데이터 수집]          [파이프라인]           [ML 모델]            [대시보드]
FBref 크롤러    →   aggregate.py      →   P1~P8 예측 모델   →   Streamlit
Transfermarkt  →   preprocess.py     →   S1~S6 스카우트    →   12개 페이지
(Selenium)         scout_features.py      모델
```

---

## 📅 프로젝트 진행 과정

### Phase 1 — 데이터 수집 인프라 구축 (2025년 3월)

- **FBref 크롤러** (`crawlers/fbref_agent.py`)
  - 2015/16 ~ 2024/25 시즌 10년치 선수 통계 수집
  - 골/어시스트/태클/인터셉트/xG 등 50+ 피처
  - Cloudflare 우회: `undetected-chromedriver` 적용
  - 레이트 리밋: 6초 딜레이 + 체크포인트 재개 기능

- **Transfermarkt 크롤러** (`crawlers/transfermarkt_agent.py`)
  - 선수 시장가치(€) + 이적 이력 수집
  - 5초 레이트 리밋, 이름 정규화 매핑 (`config/team_name_mapping.json`)

- **이미지 크롤러** (`crawlers/image_crawler.py`)
  - 선수 프로필 사진 수집 + base64 캐시 (`data/images/`)

### Phase 2 — 데이터 파이프라인 (2025년 3월)

- `pipeline/aggregate.py` — 시즌별 원본 CSV 병합
- `pipeline/preprocess.py` — 결측치 처리, 포지션 통합, 정규화
- `pipeline/scout_features.py` — WAR(Wins Above Replacement) 지표 생성
  - 포지션별 가중 합산 → 0~100 백분위 스케일
  - EPL 전체 선수 대비 상대 기여도 산출

### Phase 3 — ML 모델 개발 (2025년 3월 ~ 4월)

#### 예측 모델 (P 시리즈)

| 모델 | 목적 | 알고리즘 |
|------|------|---------|
| **P1** 경기 결과 예측 | 홈/원정 승패 예측 | XGBoost + MLP + LSTM 앙상블 |
| **P2** 득점 예측 | 선수/팀 득점 기대값 | XGBoost + Poisson 회귀 |
| **P3** 강등 예측 | 시즌 말 강등권 확률 | XGBoost + RandomForest |
| **P4** MVP 예측 | 시즌 최우수 선수 순위 | LTR(Learning to Rank) + XGBoost |
| **P5** 플레이스타일 클러스터링 | 포지션별 유형 분류 | K-Means(k=6) + UMAP |
| **P6** 시장가치 예측 | 적정 이적가 추정 | XGBoost + MLP |
| **P7** 성장 곡선 예측 | 향후 성장/쇠퇴 확률 | XGBoost (포지션별 분리) |
| **P8** 이적 적응 예측 | 새 리그 적응 가능성 | XGBoost |

#### 스카우트 모델 (S 시리즈)

| 모델 | 목적 | 산출물 |
|------|------|--------|
| **S1** WAR 선수 평가 | 시즌 전체 선수 기여도 순위 | WAR 백분위 점수 (0~100) |
| **S2** 시장 저평가 발굴 | value_ratio = 예측가/시장가 | 저평가 Top 탐색기 |
| **S3** 유사 선수 탐색 | UMAP 기반 플레이스타일 유사도 | 대체 선수 목록 |
| **S4** 성장 레퍼런스 | 나이별 성장 궤적 비교 | 성장 가능성 판단 |
| **S5** 이적 리스크 평가 | 리그 이동 시 적응 실패 확률 | 리스크 점수 |
| **S6** 하락 조기 경보 | 선수 폼 하락 감지 | 하락 주의 선수 목록 |

### Phase 4 — Streamlit 대시보드 구축 (2025년 4월)

12개 페이지로 구성된 멀티페이지 대시보드:

| 페이지 | 주요 기능 |
|--------|---------|
| 🏠 홈 | WAR Top5, 저평가 Top5, 성장 급등 선수 요약 |
| 🔍 선수 즉시 분석 | 선수 검색 → WAR/시장가치/레이더차트/S1~S6 전체 분석 |
| ⭐ 나의 쇼트리스트 | 관심 선수 저장·관리·우선순위 설정 |
| 스카우트 개요 | 맞춤 영입 추천 + 하락 주의 + 성장 급등 필터 |
| 선수 분석 | WAR 순위 / 숨은 보석 / 시장가치 / 하락 주의보 탭 |
| 이적 인텔리전스 | S3 유사선수 / S4 성장레퍼런스 / S5 이적리스크 / 시뮬레이터 |
| 💎 S2 저평가 탐색기 | 예산/포지션/나이/WAR 필터 → value_ratio 기반 발굴 |
| 🏟️ 팀 프로파일 | 팀별 선수단 WAR 분포 / 버블차트 / 포지션 공백 분석 |
| 선수 통계 순위 | 골·어시·태클·xG 등 통계 순위표 |
| 시즌 개요 | 시즌별 EPL 전체 흐름 분석 |
| 선수 비교 | 2~3명 선수 레이더차트 직접 비교 |
| 역대 기록 | 역대 시즌 WAR 최고 선수 기록 |

### Phase 5 — 버그 수정 및 UI 개선 (2025년 4월)

- **다크 테마 전환**: `style.css` 배경 `#f5f5f5` → `#0d0d1a`, `.streamlit/config.toml` 생성
- **네비게이션 오류 수정**: `st.session_state["nav_menu"]` 위젯 충돌 → `_nav_target` 플래그 패턴 적용 (8개 페이지)
- **S2 저평가 로직 수정**: `value_ratio = predicted/market` 방향 반전 (기존 로직이 고평가 선수를 저평가로 표시하던 버그)
- **버블차트 NaN 오류**: `px.scatter(size=NaN)` → `go.Scatter` Python 리스트 방식으로 전환
- **Plotly 흰 배경 수정**: `theme=None` + `plot_bgcolor="#1a1a2e"` 전체 적용 (23개 차트)

---

## 🗂️ 프로젝트 구조

```
EPL project/
├── crawlers/               # 데이터 수집
│   ├── fbref_agent.py      # FBref 크롤러
│   ├── transfermarkt_agent.py  # Transfermarkt 크롤러
│   └── image_crawler.py    # 선수 이미지 수집
├── pipeline/               # 데이터 처리
│   ├── aggregate.py        # 원본 병합
│   ├── preprocess.py       # 전처리
│   └── scout_features.py   # WAR 피처 생성
├── models/                 # ML 모델
│   ├── p1_match_result/    # 경기 결과 예측
│   ├── p2_goal_scoring/    # 득점 예측
│   ├── p3_relegation/      # 강등 예측
│   ├── p4_mvp/             # MVP 예측
│   ├── p5_clustering/      # 플레이스타일 클러스터링
│   ├── p6_market_value/    # 시장가치 예측
│   ├── p7_growth_curve/    # 성장 곡선
│   ├── p8_transfer_adapt/  # 이적 적응
│   ├── s1_player_rating/   # WAR 평가 (S1)
│   ├── s2_market_value/    # 저평가 발굴 (S2)
│   ├── s3_similarity/      # 유사 선수 (S3)
│   ├── s4_growth/          # 성장 레퍼런스 (S4)
│   ├── s5_transfer_adapt/  # 이적 리스크 (S5)
│   └── s6_decline/         # 하락 조기 경보 (S6)
├── dashboard/              # Streamlit 앱
│   ├── app.py              # 메인 (네비게이션)
│   ├── assets/style.css    # 다크 테마 CSS
│   ├── components/         # 공통 컴포넌트
│   └── pages/              # 12개 페이지
├── config/                 # 설정
│   ├── settings.py
│   └── team_name_mapping.json
├── .streamlit/config.toml  # Streamlit 테마 설정
└── requirements.txt
```

---

## 🚀 로컬 실행

```bash
# 1. 가상환경 생성 및 활성화
python -m venv .venv
.venv\Scripts\activate  # Windows

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 대시보드 실행
streamlit run dashboard/app.py
```

> **주의**: `data/` 폴더의 parquet 파일이 필요합니다. 크롤러를 먼저 실행하거나 데이터를 별도로 준비해야 합니다.

---

## 🛠️ 기술 스택

| 영역 | 사용 기술 |
|------|---------|
| 데이터 수집 | Selenium, undetected-chromedriver, BeautifulSoup4 |
| 데이터 처리 | pandas, numpy, pyarrow (Parquet) |
| ML 모델 | XGBoost, scikit-learn, PyTorch (MLP/LSTM) |
| 클러스터링/시각화 | UMAP, K-Means, Plotly |
| 대시보드 | Streamlit |
| 언어 | Python 3.12 |

---

## 📊 주요 지표 설명

- **WAR (Wins Above Replacement)**: 선수의 팀 기여도를 0~100 백분위로 환산. 50이 EPL 평균, 99가 최상위권
- **value_ratio**: `예측 시장가치 / 현재 시장가치`. 1.0 초과 = 시장 저평가 (예측가 > 시장가)
- **S2 저평가 기준**: value_ratio ≥ 1.1 이면 탐색 대상, ≥ 2.0 이면 극저평가

---

## 👤 개발자

- **개발**: robinho0329
- **페르소나**: 김태현 스카우트 (중위권 구단 예산 기준 실무 검증)

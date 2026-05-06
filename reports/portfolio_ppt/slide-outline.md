# EPL 스카우트 인텔리전스 — 2026-05-06 데일리 브리핑
**생성일**: 2026-05-06 | **유형**: 데일리 브리핑 | **슬라이드**: 6장

---

## 프레젠테이션 개요

| 항목 | 내용 |
|------|------|
| 목적 | 14번째 정기 미팅 결과 요약 및 오늘 액션아이템 공유 |
| 청중 | 김태현 스카우트, 스카우트팀 |
| 핵심 메시지 | P1 torch 오류 3연속 — 오늘 반드시 해결. P2 대시보드 통합 오늘 기한. P7 v5.1 내일까지. 현재 4.80/5.0, 이번 주 안 5.0 달성 가능 |
| 데이터 소스 | 2026-05-06_meeting.md + 12개 모델 results_summary.json + dashboard/app.py |

---

## 슬라이드 구성

---

### 슬라이드 01: 표지
- **유형**: 표지
- **핵심 메시지**: P1 오류 3연속 — 오늘이 분기점
- **내용**:
  - 제목: EPL 스카우트 인텔리전스
  - 부제: 2026년 05월 06일 (수) — 데일리 브리핑
  - 미팅 회차: 14번째 정기 미팅
  - 팀: 김태현 스카우트 x Marcus Webb (Analytics Agent)
  - 긴급 배너: "P1 torch 오류 3연속 — 오늘 09:15 반드시 해결"
- **시각 요소**: EPL 로고 영역(좌상단) / 날짜 대형 타이포그래피(중앙) / 긴급 배너(하단 강조 바)
- **색상 테마**: EPL_DARK 배경(`#0d1117`) / 제목 흰색 / 긴급 배너 EPL_ACCENT(`#e94560`)

---

### 슬라이드 02: 모델 성능 현황표 — 12개 모델 실제 지표 신호등
- **유형**: 현황
- **핵심 메시지**: 10개 모델 정상, P1·P7 미완성 — 핵심 모델 품질은 탄탄
- **내용**:
  모든 수치는 각 모델의 results_summary.json에서 직접 읽은 실제 값.

  | 모델 | 모델명 | 버전 | 핵심 지표 | 상태 | 전일 대비 |
  |------|--------|------|-----------|------|-----------|
  | P1 | 경기 결과 예측 | v2 | XGBoost Acc 57.26% / F1 45.21% | 빨간불 | torch 오류 3연속 실패 (긴급) |
  | P2 | 시즌 득점 예측 | v2 | XGBoost R² 0.8877 / MAE 0.96골 | 초록불 2단계 | 대시보드 미통합 (오늘 기한) |
  | P3 | 강등권 예측 | v2 | XGBoost val AUC 99.02% / test AUC 100% | 초록불 2단계 | 변동 없음 |
  | P4 | MVP 스코어링 | - | XGBoost Spearman 0.9987 / NDCG@10 0.9993 | 초록불 2단계 | 변동 없음 |
  | P5 | 선수 클러스터링 | v2 | K-means K=3 Silhouette 0.2851 | 초록불 | 변동 없음 |
  | P6 | 시장가치 예측 | - | XGBoost R² 0.8734 / MAE £3.28M | 초록불 | 변동 없음 |
  | P7 | 성장 곡선 예측 | v5 | LightGBM R² 0.586 / val-test 갭 0.055 | 노란불 | v5.1 오늘 착수 (R² 목표 0.62 미달) |
  | P8 | 이적 적응 예측 | v2 | XGBoost AUC 0.6612 / F1 88.23% | 초록불 | 변동 없음 |
  | S1 | 선수 종합 레이팅 | v1.2 | XGBoost R² 0.9093 / MAE 4.66 | 초록불 2단계 | 변동 없음 |
  | S2 | 포지션별 시장가치 | v4 | XGBoost overall R² 0.8812 / MAE £3.23M | 초록불 | 변동 없음 |
  | S3 | 유사 선수 탐색 | v4 | Silhouette 0.3224 (K=20, 포지션별 분리) | 초록불 | 변동 없음 |
  | S6 | 선수 하락세 감지 | - | XGBoost AUC 0.8324 / F1 61.74% | 초록불 | 변동 없음 |

  - 긴급 블로킹: P1 (torch ImportError 3연속), P7 (R² 0.034 부족)
  - 오늘 기한: P2 대시보드 통합 (player_intelligence.py st.columns 블록 추가)
- **시각 요소**: 12행 컬러 테이블 / 상태 열에 신호등 원형 아이콘 (초록 = 정상, 초록2단계 = 탁월, 노란 = 부분달성, 빨간 = 긴급) / P1 행 빨간 하이라이트
- **색상 테마**: EPL_DARK 배경 / 테이블 헤더 `#0f3460` / 신호등: 초록 `#00c851`, 노란 `#ffbb33`, 빨간 `#e94560`

---

### 슬라이드 03: 어제(2026-05-05) 달성·미달 체크리스트 + 전일 대비
- **유형**: 현황
- **핵심 메시지**: 어제 3개 액션아이템 전부 미달 — 스크립트 매핑 오류와 torch 오류가 이틀을 날렸다
- **내용**:

  **어제(5/5) 결과 — 전일 대비**

  | # | 항목 | 결과 | 비고 |
  |---|------|------|------|
  | 1 | P1 torch 의존성 제거 + 재학습 | 미달 | torch 실제 LSTM/MLP 사용 확인 → 단순 삭제 불가. 오류 전략 수정 필요 |
  | 2 | P2 대시보드 통합 | 미달 | dev_result 스크립트 매핑 오류 — P7 학습 스크립트가 P2 항목으로 잘못 실행됨. 코드 한 줄도 미작성 |
  | 3 | P7 v5.1 착수 | 미달 | "중복 스킵" 처리로 실제 코드 미작성 |

  **전일(5/4→5/5) 대비 누적 변화**

  | 항목 | 5/4 상태 | 5/5 상태 | 변화 |
  |------|----------|----------|------|
  | P1 학습 성공 여부 | torch 오류 2연속 | torch 오류 3연속 | 악화 |
  | P2 대시보드 통합 | 미완 | 미완 (스크립트 오류로 하루 낭비) | 동일 |
  | P7 v5 갭 | 갭 0.055 달성 (5/5 신규) | 유지 | 유지 |
  | 스카우트 평가 | 4.80 | 4.80 (전일 동일) | 동일 |

  핵심: 어제 신규 달성 없음. dev_result 스크립트 매핑 버그가 1일 손실 발생.
- **시각 요소**: 두 테이블 상하 배치 / 미달 행에 빨간 X 아이콘 / 변화 열에 방향 화살표 아이콘 / 스크립트 매핑 오류 원인 텍스트박스(강조)
- **색상 테마**: EPL_DARK 배경 / 미달 강조 EPL_ACCENT / 달성(없음) EPL_GOLD / 원인 분석 박스 `#16213e` 테두리

---

### 슬라이드 04: 오늘 액션아이템 Top 3 — 담당·목표지표·기한
- **유형**: 액션아이템
- **핵심 메시지**: P1 해결이 이번 주 5.0의 관문 — 오늘 09:15 Actions가 승부처
- **내용**:

  **액션아이템 #1 — 최우선 (긴급)**
  - 담당: Marcus Webb / GitHub Actions
  - 내용: P1 torch 조건부 임포트 수정 + 재학습
    - `train_pipeline.py` 상단: `try: import torch; TORCH_AVAILABLE = True` / `except ImportError: torch = None; TORCH_AVAILABLE = False`
    - `TORCH_AVAILABLE = False` 시 LSTM·MLP 학습 블록 전체 스킵 (XGBoost·LightGBM만 실행)
    - `requirements.txt` torch 항목 주석 처리 (CI 환경 용량 문제)
    - 재학습 후 results_summary.json 정상 업데이트 확인
  - 목표 지표: P1 학습 성공 / XGBoost test Acc 57.26% 이상 유지
  - 기한: 2026-05-06 09:15 (오늘 Actions)
  - 근거: XGBoost Acc 57.26% vs LSTM 44.37% — torch 없이도 최고 성능 유지 가능

  **액션아이템 #2 — 오늘 기한**
  - 담당: Marcus Webb
  - 내용: P2 대시보드 통합 (오늘 최종 기한)
    - `dashboard/pages/player_intelligence.py` 성장 예측 섹션
    - `st.columns(2)` 추가: 좌=P7 성장 곡선(현행), 우=P2 득점 예측 블록
    - 우측 블록: "내년 예상 득점: X.X골 (±1.9골 신뢰구간, FW 상위 Z%)"
    - `models/p2_goal_scoring/` XGBoost 모델 로드 + 포지션별 백분위 계산
  - 목표 지표: P2 UI 통합 완료 / player_intelligence P2 블록 추가
  - 기한: 2026-05-06 종일

  **액션아이템 #3 — 내일 Actions**
  - 담당: Marcus Webb
  - 내용: P7 v5.1 학습 스크립트 작성 + Actions 예약
    - `models/p7_growth_curve/train_v5_1.py` 신규 작성
    - 전략 3가지: (1) `gc_trend_3yr` 피처 복원, (2) FW/MID 분리 모델(FW ~1,200명, MID ~1,800명), (3) Optuna 100 trials
    - 메타 모델: Ridge → LinearRegression(비음수 제약) 교체 (xgb 음수 가중치 -0.157 해소)
    - results_summary.json `version: v5.1` 업데이트
  - 목표 지표: R² 0.62 이상 / val-test 갭 0.07 이하
  - 기한: 2026-05-07 (내일 09:15 Actions 실행)
- **시각 요소**: 3단 카드 레이아웃 (세로 배치) / 각 카드 우측에 우선순위 배지 (#1 빨간, #2 주황, #3 파란) / 기한 타이머 아이콘 / 목표지표 강조 수치 EPL_GOLD 색상
- **색상 테마**: 카드 배경 `#16213e` / 테두리: #1 EPL_ACCENT, #2 `#ffbb33`, #3 `#0f3460` / EPL_DARK 배경

---

### 슬라이드 05: 김태현 스카우트 종합 평가 — 4.80/5.0 + 5.0 달성 로드맵
- **유형**: 분석
- **핵심 메시지**: 4.80/5.0 실질적 수준 — P1·P2 오늘, P7 내일로 이번 주 5.0 현실적
- **내용**:

  **현재 평가: 4.80 / 5.0** (전일 동일 — 어제 신규 달성 없음)

  **누적 달성 항목 (탄탄한 기반)**

  | 모델 | 핵심 성과 | 실무 활용 |
  |------|-----------|-----------|
  | P2 v2 | R² 0.8877, MAE 0.96골 | 영입 회의 즉시 활용 가능 |
  | P7 v5 갭 0.055 | val-test 갭 목표 달성 | 과적합 해소 완료 |
  | P8 v2 | AUC 0.6612 + 이적 리스크 UI | 영입 회의 활용 가능 |
  | S1 R² 0.9093 | MAE 4.66, 선수 레이팅 안정 | 스카우트 핵심 |
  | S2 R² 0.8812 | MAE £3.23M, 포지션별 분리 | 스카우트 핵심 |
  | P3 AUC 99.02%(val)/100%(test) | 강등 예측 최고 신뢰도 | 즉시 활용 |
  | P4 Spearman 0.9987 | NDCG@10 0.9993 | MVP 순위 안정 |
  | S3 Silhouette 0.3224 | K=20 포지션별 분리 | 유사선수 탐색 |
  | S6 AUC 0.8324 | F1 61.74% | 하락세 감지 실용적 |
  | 대시보드 16페이지 | 전방위 커버리지 | Streamlit 배포 완료 |

  **5.0 달성 로드맵 (수요일 기준)**

  ```
  현재: 4.80
    ├─ P1 torch 조건부 임포트 성공 → +0.05 → 4.85  [오늘 09:15]
    ├─ P2 대시보드 통합 완료       → +0.05 → 4.90  [오늘 종일]
    └─ P7 v5.1 R² 0.62 달성       → +0.10 → 5.00  [내일 09:15]
  ```

  | 날짜 | 목표 | 기대 점수 |
  |------|------|-----------|
  | 5/6 (오늘, 수) | P1 성공 + P2 대시보드 통합 | 4.85~4.90 |
  | 5/7 (목) | P7 v5.1 R² 0.62 달성 | 4.95~5.0 |
  | 5/8 (금) | 전체 통합 테스트 + Streamlit 배포 확인 | 5.0 확정 |

  김태현 발언: "P1 하나가 발목을 잡고 있습니다. XGBoost만 돌려도 57%가 나오는데 torch 때문에 학습 자체가 안 되고 있으니, 오늘 조건부 임포트로 반드시 해결해주십시오."
- **시각 요소**: 좌측 — 원형 게이지 차트 (4.80/5.0, 96% 채움, EPL_GOLD 색상) / 우측 상단 — 달성 항목 체크리스트 표 / 우측 하단 — 로드맵 스텝 다이어그램 (3단계 화살표)
- **색상 테마**: 게이지 EPL_GOLD(`#f5a623`) / 달성항목 초록 체크 / 로드맵 스텝 배경 `#16213e` / EPL_DARK 배경

---

### 슬라이드 06: Streamlit 대시보드 현황 — 16페이지 + URL
- **유형**: 현황
- **핵심 메시지**: 16페이지 완전 가동 — P2 통합 시 블록 확장 (별도 페이지 미추가)
- **내용**:

  **현재 대시보드: 16페이지 (Streamlit Cloud 배포 완료)**

  | # | 페이지명 | 분류 |
  |---|---------|------|
  | 1 | 홈 | 허브 |
  | 2 | 선수 즉시 분석 | 스카우트 도구 |
  | 3 | 나의 쇼트리스트 | 스카우트 도구 |
  | 4 | 선수 종합 인텔리전스 | 스카우트 도구 |
  | 5 | 스카우트 개요 | 스카우트 도구 |
  | 6 | 선수 분석 | 스카우트 도구 |
  | 7 | 이적 인텔리전스 | 스카우트 도구 |
  | 8 | 강등권 탐색기 | 스카우트 도구 |
  | 9 | S2 저평가 탐색기 | 스카우트 도구 |
  | 10 | 팀 프로파일 | 통계 |
  | 11 | 선수 통계 순위 | 통계 |
  | 12 | 시즌 개요 | 통계 |
  | 13 | 선수 비교 | 통계 |
  | 14 | 역대 기록 | 통계 |
  | 15 | 모델 설명 (SHAP) | 분석 |
  | 16 | 선수 유형 탐색기 | 분석 |

  **오늘 변경 예정**: 페이지 추가 없음 / player_intelligence(#4) 내 P2 블록 추가

  **접속 URL**:
  https://epl-scouting-report-dashboard-project-ffyb8msh6uafxyyg8txsm8.streamlit.app

  GitHub: robinho0329/EPL-scouting-report-dashboard-project
- **시각 요소**: 좌측 — 16행 페이지 목록 표(분류별 색상 구분) / 우측 — URL QR코드 플레이스홀더 + URL 텍스트 / 하단 — "P2 블록 추가 예정" 안내 배너
- **색상 테마**: EPL_DARK 배경 / 표 헤더 `#0f3460` / URL 박스 EPL_GREEN 테두리 / QR 영역 흰색 배경 / 분류 배지: 스카우트 도구 EPL_ACCENT, 통계 `#0f3460`, 분석 EPL_GOLD

---

## 디자인 시스템 요약

| 항목 | 값 |
|------|-----|
| 슬라이드 크기 | 720pt × 405pt (16:9) |
| 폰트 | Pretendard (CDN) / fallback: Apple SD Gothic Neo, sans-serif |
| 배경색 | `#0d1117` (다크모드 기본) |
| 네이비 다크 | `#1a1a2e` |
| 포인트 블루 | `#0f3460` |
| 강조 레드 | `#e94560` (EPL_ACCENT) |
| 골드 | `#f5a623` (EPL_GOLD, 성과 강조) |
| 본문 흰색 | `#e6edf3` |
| 서브 텍스트 | `#8b949e` |

## 수치 출처 명세

| 수치 | 출처 파일 |
|------|-----------|
| P1 XGBoost Acc 57.26% / F1 45.21% | models/p1_match_result/results_summary.json |
| P2 XGBoost R² 0.8877 / MAE 0.9566골 | models/p2_goal_scoring/results_summary.json |
| P3 val AUC 99.02% / test AUC 100% | models/p3_relegation/results_summary.json (results_full_season.XGBoost) |
| P4 Spearman 0.9987 / NDCG@10 0.9993 | models/p4_mvp/results_summary.json |
| P5 Silhouette 0.2851 (K=3) | models/p5_clustering/results_summary.json |
| P6 R² 0.8734 / MAE £3.28M | models/p6_market_value/results_summary.json |
| P7 LightGBM R² 0.586 / 갭 0.055 | models/p7_growth_curve/results_summary.json |
| P8 AUC 0.6612 / F1 88.23% | models/p8_transfer_adapt/results_summary.json |
| S1 R² 0.9093 / MAE 4.66 | models/s1_player_rating/results_summary.json |
| S2 overall R² 0.8812 / MAE £3.23M | models/s2_market_value/results_summary.json |
| S3 Silhouette 0.3224 (K=20) | models/s3_similarity/results_summary.json |
| S6 AUC 0.8324 / F1 61.74% | models/s6_decline/results_summary.json |
| 대시보드 16페이지 | dashboard/app.py MENU_OPTIONS (16개 항목 직접 계산) |
| 스카우트 평가 4.80/5.0 | 2026-05-06_meeting.md 섹션 4 |
| 5.0 로드맵 수치 (+0.05, +0.05, +0.10) | 2026-05-06_meeting.md 섹션 4 |

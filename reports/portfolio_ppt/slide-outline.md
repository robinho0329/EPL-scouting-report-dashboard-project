# EPL 스카우트 인텔리전스 — 2026-05-15 데일리 브리핑
**생성일**: 2026-05-15 | **유형**: 데일리 브리핑 | **슬라이드**: 6장

---

## 프레젠테이션 개요

| 항목 | 내용 |
|------|------|
| 목적 | 금요일 이번 주 마지막 정기 미팅 결과 공유 및 영입 회의 준비 최종 점검 |
| 청중 | 김태현 스카우트, Marcus Webb (Analytics Agent) |
| 핵심 메시지 | 9개 모델 즉시 활용 가능 — P1·P7 오늘 수동 학습으로 최종 결정, demo_script_draft 오늘 저녁 마감 |
| 특이사항 | 24번째 정기 미팅 / 금요일 이번 주 마지막 세션 / GitHub Actions 4일 연속 미실행 |

---

## 슬라이드 구성

---

### 슬라이드 01: 표지 — 24번째 정기 미팅, 이번 주 마지막 금요일 브리핑
- **유형**: 표지
- **핵심 메시지**: 이번 주 마지막 미팅 — 오늘 수동 학습과 시연 스크립트로 주간 목표 완결
- **내용**:
  - 제목: EPL 스카우트 인텔리전스
  - 부제: 데일리 브리핑 — 2026년 05월 15일 (금)
  - 미팅 회차: 24번째 정기 미팅 | 09:00 KST
  - 참석자: 김태현 스카우트 x Marcus Webb (Analytics Agent)
  - 현재 평가 점수 배지: 4.75 / 5.0 (전일 4.80 → 4.75 ▼0.05)
  - 하단 슬로건 배너: "이번 주 마지막 기회 — 오늘 수동 학습 + 시연 스크립트 완성"
  - WEEK FINAL 워터마크 (배경 희미하게)
- **시각 요소**: EPL 로고 영역 (좌상단) / 날짜 대형 타이포그래피 (중앙) / 4.75/5.0 점수 원형 배지 (우하단, 전일 대비 ▼ 빨간 화살표) / 슬로건 박스 (하단 강조, EPL_ACCENT 테두리) / "WEEK FINAL" 배경 워터마크
- **색상 테마**: EPL_DARK(`#0d1117`) 배경 / 제목 흰색(`#e6edf3`) / 부제 EPL_ACCENT(`#e94560`) / 배지 EPL_GOLD(`#f5a623`) / 하락 화살표 EPL_ACCENT(`#e94560`) / 슬로건 박스 배경 `#0f3460` 테두리 `#e94560`

---

### 슬라이드 02: 12개 모델 성능 현황표 — 9개 즉시 활용 / P1·P7 오늘 수동 학습 대기
- **유형**: 현황
- **핵심 메시지**: 12개 모델 중 9개 즉시 영입 회의 활용 가능 — P1·P7은 오늘 수동 학습 결과로 최종 결정
- **내용**:

  | 모델 ID | 모델명 | 버전 | 핵심 지표 (실측값) | 목표 | 신호등 |
  |---------|--------|------|------------------|------|--------|
  | P1 | 경기 결과 예측 (H/D/A) | - | XGBoost test Acc **54.79%** / F1 **40.65%** | Acc ≥57% | 빨강 (코드 수정 완료, Actions 미실행) |
  | P2 | 시즌 득점 예측 | v2 | XGBoost test R² **0.8877** / MAE **0.96골** | R² ≥0.65 | 초록 |
  | P3 | 강등권 예측 | - | LogReg/MLP test AUC **100%** / Acc **100%** | AUC ≥0.90 | 초록초록 |
  | P4 | MVP 스코어링 | - | XGBoost Spearman **0.9987** / NDCG@10 **0.9993** | Spearman ≥0.95 | 초록초록 |
  | P5 | 선수 클러스터링 | v2 | K-Means K=3 Silhouette **0.2851** | 0.25~0.35 적정 | 노랑 |
  | P6 | 시장가치 예측 | - | XGBoost test R² **0.8734** / MAE **£3.28M** | R² ≥0.80 | 초록 |
  | P7 | 성장 곡선 예측 | v5 (→v5.1 대기) | LGBM test R² **0.586** / val-test 갭 **0.055** | R² ≥0.62 | 빨강 (코드 수정 완료, Actions 미실행) |
  | P8 | 이적 적응 예측 | v2 | XGBoost AUC **0.6612** / F1 **88.23%** / Acc **80.65%** | AUC ≥0.65 | 초록 |
  | S1 | 선수 종합 레이팅 | v1.2 | XGBoost test R² **0.9093** / MAE **4.66** | R² ≥0.88 | 초록초록 |
  | S2 | 포지션별 시장가치 | v4 | XGBoost overall R² **0.8812** / MAE **£3.23M** | R² ≥0.80 | 초록 |
  | S3 | 유사선수 탐색 | v4 | FW **0.2085** / MID **0.2299** / DEF **0.3637** / GK **0.4873** | 포지션별 ≥0.20 | 노랑 |
  | S6 | 선수 하락세 감지 | - | XGBoost AUC **0.8324** / F1 **61.74%** / Acc **75%** | AUC ≥0.80 | 노랑 |

  - 요약 카운트: 초록초록 3개(P3·P4·S1) / 초록 4개(P2·P6·P8·S2) / 노랑 3개(P5·S3·S6) / 빨강 2개(P1·P7)
  - 즉시 활용 가능 9개: S1·P4·P6·S2·P8·P3·P2·S3·S6
  - 비고: P1·P7 코드 수정 완료 — GitHub Actions 4일 연속 미실행(05-12 이후)으로 재학습 결과 없음. 오늘 수동 실행 예정

- **시각 요소**: 12행 신호등 표 (색상 도트 + 수치 병기) / P1·P7 행 EPL_ACCENT(`#e94560`) 배경 틴트 강조 + "수동학습 대기" 배지 / P3·P4·S1 행 EPL_GOLD(`#f5a623`) 10% opacity 강조 / 하단 집계 카운트 배지 4개
- **색상 테마**: EPL_DARK(`#0d1117`) 배경 / 초록초록 `#00d26a` / 초록 `#2ecc71` / 노랑 `#f5a623` / 빨강 `#e94560` / 테이블 헤더 `#0f3460` / 셀 배경 `#16213e`

---

### 슬라이드 03: 어제(05-14) 달성·미달 — 코드 수정 2건 달성, 실행 연결은 실패
- **유형**: 현황
- **핵심 메시지**: 코드 수정 2건 달성 — Actions가 4일째 멈춰 학습 결과 없고 demo_script는 3일 연속 미생성
- **내용**:

  섹션 A — 달성 항목 (초록 체크 2개)
  - `scripts/github_actions_dev.py` 36번 줄 P7 경로 수정 완료 (`"train_v5_1.py"` 확인)
  - `models/p1_match_result/train_pipeline.py` 피처 추가 완료 (`elo_trend_3`, `RobustScaler` 확인)

  섹션 B — 미달 항목 (빨간 X 4개)

  | 항목 | 결과 | 연속 미달 | 세부 내용 |
  |------|------|-----------|-----------|
  | demo_script_draft.md 생성 | ❌ 미달 | 3일 연속 | 오늘 저녁 마감이 이번 주 마지막 기회 |
  | dev_result.json 생성 | ❌ 미달 | 4일 연속 | GitHub Actions 05-12 이후 미실행 (원인 불명) |
  | P7 v5.1 R² ≥ 0.62 달성 | ❌ 미확인 | 코드 수정 완료 | 학습 실행 필요 — 오늘 수동 실행으로 확인 예정 |
  | P1 Acc ≥ 57% 달성 | ❌ 미확인 | 코드 수정 완료 | 학습 실행 필요 — 오늘 수동 실행으로 확인 예정 |

  요약: 달성 2개 / 미달·미확인 4개 — "코드는 준비됐다. 학습이 연결되지 않으면 결과가 없는 것과 같다"

- **시각 요소**: 2단 레이아웃 (좌: 달성 초록 카드 2개 / 우: 미달 빨강 카드 4개) / 각 카드 아이콘 + 항목명 + 상태 배지 / 하단 요약 인용 박스 (EPL_ACCENT 테두리)
- **색상 테마**: 달성 카드 배경 rgba(0,210,106,0.10) 테두리 `#00d26a` / 미달 카드 배경 rgba(233,69,96,0.10) 테두리 `#e94560` / 배경 EPL_DARK / 인용 박스 배경 `#16213e`

---

### 슬라이드 04: 오늘 액션아이템 Top 3 — 이번 주 마지막 실행 기회
- **유형**: 액션아이템
- **핵심 메시지**: 수동 학습 실행 + 시연 스크립트 완성 + 발표 자료 점검 — 오늘 못 하면 주간 목표 미달 확정
- **내용**:

  카드 1 (1순위 — 금메달 / 오전 즉시)
  - 담당: Marcus Webb
  - 과제: GitHub Actions 원인 파악 및 수동 학습 실행
  - 실행 명령:
    - `python models/p7_growth_curve/train_v5_1.py`
    - `python models/p1_match_result/train_pipeline.py`
  - 병행: Actions cron 설정 점검 (4일 연속 미실행 원인 해소)
  - 목표 지표: P7 test R² ≥ 0.62 / P1 test Acc ≥ 57%
  - 기한: 2026-05-15 오전 즉시

  카드 2 (2순위 — 은메달 / 오전~오후 / 저녁 마감)
  - 담당: Marcus Webb
  - 과제: 포트폴리오 시연 스크립트 초안 작성
  - 저장 경로: `reports/portfolio_ppt/demo_script_draft.md`
  - 내용: S1→S2→P8→P6→S6 5분 시연 흐름 5장 / 노팅엄 포레스트·브렌트포드 30-50M 파운드 예산 시나리오
  - 목표 지표: 5장 초안 완성
  - 기한: 2026-05-15 저녁 마감 (미완료 시 이번 주 포트폴리오 즉흥 진행)

  카드 3 (3순위 — 동메달 / 오후)
  - 담당: Marcus Webb
  - 과제: 이번 주 발표 자료 최종 점검
  - 내용: 9개 모델 Streamlit 페이지 정상 작동 확인 / 대시보드 URL 공유 준비 / P1·P7 오늘 결과에 따라 포트폴리오 포함 여부 최종 결정
  - 목표 지표: 9개 모델 페이지 정상 작동 확인
  - 기한: 2026-05-15 오후
  - 비고: P1·P7 목표 미달 시 이번 주 포트폴리오는 9개 모델 기준으로 진행

- **시각 요소**: 3열 카드 레이아웃 / 카드 상단 메달 아이콘(금·은·동) / 각 카드 내 실행 명령어 코드 블록 (카드 1) / 목표 지표 강조 배지 / 기한 라벨 배지 (오전=빨강, 오후=노랑) / 하단 결정 조건 박스 (P1·P7 미달 시 9개 모델 기준)
- **색상 테마**: EPL_DARK(`#0d1117`) 배경 / 카드 배경 `#16213e` / 카드 1 테두리 EPL_GOLD / 카드 2 테두리 EPL_ACCENT / 카드 3 테두리 `#0f3460` / 기한 배지 오전 `#e94560` / 오후 `#f5a623`

---

### 슬라이드 05: 김태현 스카우트 종합 평가 — 4.75/5.0 (전일 4.80에서 하락) + 5.0 로드맵
- **유형**: 분석
- **핵심 메시지**: 4.75/5.0 — demo_script 3일 미생성·Actions 4일 미실행으로 하락, 오늘 실행 완료 시 4.92까지 회복 가능
- **내용**:

  현재 점수: 4.75 / 5.0
  전일 대비: 4.80 → 4.75 (▼0.05 — demo_script_draft.md 3일 연속 미생성 / Actions 4일 연속 미실행)

  5.0 달성 로드맵:

  | 단계 | 조건 | 점수 변화 | 목표 시점 |
  |------|------|-----------|-----------|
  | 현재 | 9개 모델 즉시 활용, 대시보드 16페이지 | **4.75** | 2026-05-15 시작 |
  | 오늘 수동 학습 | P7 v5.1 R² ≥ 0.62 달성 확인 | +0.07 → **4.82** | 2026-05-15 오전 |
  | 오늘 수동 학습 | P1 XGBoost Acc ≥ 57% 달성 확인 | +0.05 → **4.87** | 2026-05-15 오전 |
  | 오늘 오전~오후 | 포트폴리오 시연 스크립트 완성 | +0.05 → **4.92** | 2026-05-15 저녁 |
  | 이번 주 내 | Actions 자동화 복구 | +0.04 → **4.96** | 이번 주 |
  | 이번 주 내 | 실제 영입 회의 시연 | +0.04 → **5.00** | 이번 주 |

  누적 달성 강점 하이라이트 (12개):

  | 항목 | 핵심 지표 | 평가 |
  |------|-----------|------|
  | P3 강등권 예측 | AUC 100% / Acc 100% | 최우수 |
  | P4 MVP 스코어링 | Spearman 0.9987 / NDCG@10 0.9993 | 최우수 |
  | S1 선수 종합 레이팅 | R² 0.9093 / MAE 4.66 (총 12,283명) | 최우수 |
  | P2 시즌 득점 예측 | R² 0.8877 / MAE 0.96골 | 우수 |
  | P6 시장가치 예측 | R² 0.8734 / MAE £3.28M | 우수 |
  | S2 포지션별 시장가치 | R² 0.8812 / MAE £3.23M | 우수 |
  | P8 이적 적응 예측 | AUC 0.6612 / F1 88.23% | 우수 |
  | S3 유사선수 탐색 | GK Sil 0.4873 / DEF 0.3637 | 양호 |
  | S6 하락세 감지 | AUC 0.8324 / F1 61.74% | 양호 |
  | 대시보드 16페이지 | 즉시 활용 9개 | 달성 |
  | github_actions_dev.py P7 경로 수정 | train_v5_1.py 확인 | 달성 (어제) |
  | train_pipeline.py 피처 추가 | elo_trend_3, RobustScaler 확인 | 달성 (어제) |

  김태현 최종 코멘트 (인용):
  "시스템은 준비됐습니다. 실행만 남았습니다. 오늘 금요일, 이번 주 마지막 기회입니다."

- **시각 요소**: 좌측 대형 반원형 게이지 (4.75/5.0 = 95%, EPL_GOLD, 전일 대비 ▼ 빨간 화살표) / 중앙 6단계 수평 로드맵 화살표 (4.75 → +0.07 → 4.82 → +0.05 → 4.87 → +0.05 → 4.92 → +0.04 → 4.96 → +0.04 → 5.0) / 우측 누적 달성 체크리스트 (초록 체크 12개) / 하단 인용구 박스
- **색상 테마**: EPL_DARK(`#0d1117`) 배경 / 게이지 EPL_GOLD(`#f5a623`) / 하락 화살표 EPL_ACCENT(`#e94560`) / 로드맵 노드 `#0f3460` → `#00d26a` 그라데이션 / 달성 항목 `#00d26a` / 인용구 박스 배경 `#16213e` 테두리 `#f5a623`

---

### 슬라이드 06: Streamlit 대시보드 현황 — 16페이지, 즉시 활용 9개, 영입 회의 시연 준비
- **유형**: 결론
- **핵심 메시지**: 16페이지 대시보드 완성 — S1→S2→P8→P6→S6 시연 루트로 선수 한 명당 5분 데이터 근거 제시 가능
- **내용**:

  현황 숫자 배지 (3개):
  - 총 페이지: 16개
  - 즉시 활용 가능: 9개 (S1·P4·P6·S2·P8·P3·P2·S3·S6)
  - 학습 결과 대기: 2개 (P1·P7 — 오늘 수동 학습 결과 확인 후 결정)

  영입 회의 시연 루트 — 5단계 플로우 (한 후보 당 약 5분):

  | 순서 | 페이지 | 핵심 질문 | 지표 |
  |------|--------|-----------|------|
  | 1 | S1 선수 종합 레이팅 | 이 선수가 EPL 수준에서 얼마나 뛰어난가? | R²=0.9093 / MAE=4.66 |
  | 2 | S2 포지션별 시장가치 | 시장에서 저평가된 타깃이 있는가? | R²=0.8812 / MAE=£3.23M |
  | 3 | P8 이적 적응 예측 | 새 구단에서 바로 적응할 수 있는가? | AUC=0.6612 / F1=88.23% |
  | 4 | P6 시장가치 예측 | 적정 이적료는? | R²=0.8734 / MAE=£3.28M |
  | 5 | S6 하락세 감지 | 향후 2-3년 부상·하락 리스크는? | AUC=0.8324 / F1=61.74% |

  대상 클럽 가정: 노팅엄 포레스트 / 브렌트포드 (예산 30-50M 파운드)

  전체 메뉴 16개 (소형 나열, 2단):
  홈 / 선수 즉시 분석 / 나의 쇼트리스트 / 선수 종합 인텔리전스 / 스카우트 개요 /
  선수 분석 / 이적 인텔리전스 / 강등권 탐색기 / S2 저평가 탐색기 / 팀 프로파일 /
  선수 통계 순위 / 시즌 개요 / 선수 비교 / 역대 기록 / 모델 설명(SHAP) / 선수 유형 탐색기

  대시보드 URL:
  https://epl-scouting-report-dashboard-project-ffyb8msh6uafxyyg8txsm8.streamlit.app

- **시각 요소**: 상단 숫자 배지 3개 (총 16페이지 골드 / 즉시 활용 9개 초록 / 대기 2개 노랑) / 중앙 5단계 화살표 플로우 다이어그램 (각 단계 페이지명·핵심 질문·지표 표기) / 전체 메뉴 소형 리스트 (좌측 2단 세로 목록) / URL 강조 박스 (우측 하단) / QR 자리 표시자
- **색상 테마**: EPL_DARK(`#0d1117`) 배경 / 배지 숫자 EPL_GOLD(`#f5a623`) / 즉시 활용 배지 `#00d26a` / 대기 배지 `#f5a623` / 플로우 화살표 `#f5a623` / 플로우 카드 배경 `#16213e` 테두리 `#0f3460` / URL 박스 배경 `#0f3460` 텍스트 `#e94560`

---

## 디자인 시스템 요약

| 항목 | 값 |
|------|-----|
| 슬라이드 크기 | 720pt x 405pt (16:9) |
| 배경색 | `#0d1117` (다크모드 기본) |
| EPL_DARK | `#1a1a2e` |
| EPL_GREEN | `#16213e` (패널) / `#0f3460` (강조) |
| EPL_ACCENT | `#e94560` (강조 레드 / 미달·경고) |
| EPL_GOLD | `#f5a623` (성과 강조) |
| EPL_SUCCESS | `#00d26a` (달성 초록) |
| 카드 배경 | `#16213e` |
| 본문 흰색 | `#e6edf3` |
| 서브 텍스트 | `#8b949e` |
| 폰트 | Pretendard (CDN 로드) / fallback: Apple SD Gothic Neo, system-ui, sans-serif |
| 텍스트 태그 규칙 | p, h1~h6, ul, ol 필수 사용 — div 직접 텍스트 금지 |
| JS 사용 | 금지 (정적 HTML only) |

---

## 수치 출처 명세 (results_summary.json 직접 독해 확인)

| 수치 | 출처 파일 | JSON 키 경로 |
|------|-----------|-------------|
| P1 XGBoost test Acc 54.79% / F1 40.65% | models/p1_match_result/results_summary.json | models.xgboost.test.accuracy / f1_macro |
| P2 XGBoost test R² 0.8877 / MAE 0.9566골 | models/p2_goal_scoring/results_summary.json | metrics.xgboost.test.r2 / mae |
| P3 LogReg/MLP test AUC 1.0 / Acc 1.0 | models/p3_relegation/results_summary.json | results_full_season.LogisticRegression.test.auc_roc |
| P4 XGBoost Spearman 0.9987 / NDCG@10 0.9993 | models/p4_mvp/results_summary.json | models.xgboost.test.Spearman / NDCG@10 |
| P5 K-Means K=3 Silhouette 0.2851 | models/p5_clustering/results_summary.json | kmeans.metrics.silhouette |
| P6 XGBoost test_R2 0.8734 / test_MAE 3,281,636 (£3.28M) | models/p6_market_value/results_summary.json | model_metrics.XGBoost.test_R2 / test_MAE |
| P7 LGBM test R² 0.586 / val-test 갭 0.055 | models/p7_growth_curve/results_summary.json | metrics.r2 / val_test_gap |
| P8 XGBoost AUC 0.6612 / F1 0.8823 | models/p8_transfer_adapt/results_summary.json | metrics.auc / f1 |
| S1 XGBoost test R² 0.9093 / MAE 4.6617 | models/s1_player_rating/results_summary.json | model_metrics.XGBoost.test.r2 / mae |
| S2 overall test_R2 0.8812 / test_MAE 3,231,674 (£3.23M) | models/s2_market_value/results_summary.json | overall_metrics.test_R2 / test_MAE |
| S3 FW 0.2085 / MID 0.2299 / DEF 0.3637 / GK 0.4873 | models/s3_similarity/results_summary.json | metadata.silhouette_by_position |
| S6 XGBoost AUC 0.8324 / F1 0.6174 | models/s6_decline/results_summary.json | model_performance.XGBoost.auc_roc / f1_decline |
| 스카우트 평가 4.75/5.0 (전일 4.80 → ▼0.05) | reports/daily_meeting/2026-05-15_meeting.md 섹션 4 | - |
| 5.0 로드맵 (+0.07, +0.05, +0.05, +0.04, +0.04) | reports/daily_meeting/2026-05-15_meeting.md 섹션 4 | - |
| 어제(05-14) 달성·미달 (달성 2 / 미달·미확인 4) | reports/daily_meeting/2026-05-15_meeting.md 섹션 4 | - |
| 대시보드 16페이지 / 즉시 활용 9개 | dashboard/app.py MENU_OPTIONS (42~59번 줄, 16개 항목) | - |
| S1 총 12,283명 레이팅 | models/s1_player_rating/results_summary.json | total_players_rated |
| P8 적응률 74.98% | models/p8_transfer_adapt/results_summary.json | adaptation_rate |

---

*생성: EPL Scout PPT Organizer | 2026-05-15 09:00 KST*
*데이터 소스: 2026-05-15_meeting.md + 12개 모델 results_summary.json 전체 직접 독해 + dashboard/app.py MENU_OPTIONS*
*다음 단계: ppt-designer 에이전트에게 HTML 슬라이드 생성 위임 (slide-01.html ~ slide-06.html)*

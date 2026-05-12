# EPL 스카우트 인텔리전스 — 데일리 브리핑 2026-05-12
**생성일**: 2026-05-12 | **유형**: 데일리 브리핑 | **슬라이드**: 6장

---

## 프레젠테이션 개요

| 항목 | 내용 |
|------|------|
| 목적 | 2026-05-12 아침 팀 미팅 결과 공유 및 오늘 액션아이템 정렬 |
| 청중 | 김태현 스카우트, Marcus Webb, 분석팀 내부 |
| 핵심 메시지 | P2 JSON 직접 복구로 즉시 활용 페이지 9개 달성. P1(5일 연속 Acc 54.79%)·P7(5일 연속 R² 0.586) 코드 직접 작성 방식으로 오늘 반드시 돌파 |
| 데이터 소스 | 2026-05-12_meeting.md + 12개 models/*/results_summary.json 직접 독해 |

---

## 슬라이드 구성

---

### 슬라이드 01: 표지 — 제20회 미팅, P2 복구로 4.85/5.0 · 오늘 코드 직접 작성
- **유형**: 표지
- **핵심 메시지**: P2 JSON 오늘 아침 직접 복구 완료 — 즉시 활용 9개, 포트폴리오 시연 준비 개시 수준
- **내용**:
  - 제목: EPL 스카우트 인텔리전스
  - 부제: 2026년 05월 12일 (화) — 데일리 브리핑
  - 미팅 회차: 20번째 정기 미팅 | 화요일 정기 미팅
  - 참석자: 김태현 스카우트 x Marcus Webb (Analytics Agent)
  - 현재 평가 점수 배지: 4.85 / 5.0 (전일 4.82에서 +0.03 — P2 JSON 복구, 즉시 활용 9개 달성)
  - 오늘 슬로건: "스크립트 실행이 아닌 코드 직접 작성으로 — P7·P1 오늘 돌파"
- **시각 요소**: EPL 로고 영역 (좌상단) / 날짜 대형 타이포그래피 (중앙) / 4.85/5.0 점수 원형 배지 (우하단) / +0.03 전일 대비 증분 배지 (초록 화살표)
- **색상 테마**: EPL_DARK(`#0d1117`) 배경 / 제목 흰색(`#e6edf3`) / 부제 EPL_ACCENT(`#e94560`) / 배지 EPL_GOLD(`#f5a623`) / 증분 배지 `#00e676`

---

### 슬라이드 02: 12개 모델 성능 현황표 — 오늘 아침 P2 복구, P1·P7 5일 연속 미달
- **유형**: 현황
- **핵심 메시지**: 9개 모델 목표 달성, P1·P7 코드 수정 없이 5일 연속 미달 — 오늘 코드 직접 작성으로 돌파
- **내용**:

  | 모델 ID | 모델명 | 버전 | 핵심 지표 (실측값) | 목표 | 신호등 |
  |---------|--------|------|------------------|------|--------|
  | P1 | 경기 결과 예측 (H/D/A) | - | XGBoost test Acc **54.79%** / F1 **40.65%** | Acc >= 57% | 빨강 |
  | P2 | 시즌 득점 예측 | v2 | XGBoost test R² **0.8877** / MAE **0.96골** | R² >= 0.65 | 초록 |
  | P3 | 강등권 예측 | - | LogReg/XGB test AUC **100%** / Acc **100%** | AUC >= 0.90 | 초록초록 |
  | P4 | MVP 스코어링 | - | XGBoost test Spearman **0.9987** / NDCG@10 **0.9993** | Spearman >= 0.95 | 초록초록 |
  | P5 | 선수 클러스터링 | v2 | K-means K=3 Silhouette **0.2851** | 0.25~0.35 적정 | 노랑 |
  | P6 | 시장가치 예측 | - | XGBoost test R² **0.8734** / MAE **3.28M** | R² >= 0.80 | 초록 |
  | P7 | 성장 곡선 예측 | v5 | LGBM test R² **0.586** / val-test 갭 **0.055** | R² >= 0.62 | 빨강 |
  | P8 | 이적 적응 예측 | v2 | XGBoost AUC **0.6612** / F1 **88.23%** | AUC >= 0.65 | 초록 |
  | S1 | 선수 종합 레이팅 | v1.2 | XGBoost test R² **0.9093** / MAE **4.66** | R² >= 0.88 | 초록초록 |
  | S2 | 포지션별 시장가치 | v4 | XGBoost overall R² **0.8812** / MAE **3.23M** | R² >= 0.80 | 초록 |
  | S3 | 유사선수 탐색 | v4 | FW **0.2085** / MID **0.2299** / DEF **0.3637** / GK **0.4873** | 포지션별 >= 0.20 | 노랑 |
  | S6 | 선수 하락세 감지 | - | XGBoost AUC **0.8324** / F1 **61.74%** / Acc **75%** | AUC >= 0.80 | 노랑 |

  - 요약 카운트: 초록초록 3개(P3·P4·S1) / 초록 4개(P2·P6·P8·S2) / 노랑 3개(P5·S3·S6) / 빨강 2개(P1·P7)
  - 오늘 아침 조치: P2 JSON truncation 직접 복구 완료 — test R²=0.8877, MAE=0.96골 정상 확인
  - P7 비고: v5 step_progression 참고 — v2(0.54)→v3(0.6089)→v5(0.586). LGBM val R²=0.6395로 알고리즘 문제 없음, 포지션 분리 코드 미삽입이 원인
  - P1 비고: 49개 피처 사용 중, ELO trend·RobustScaler 미적용 상태

- **시각 요소**: 12행 신호등 표 (색상 도트 + 수치 병기) / P1·P7 행 EPL_ACCENT(`#e94560`) 배경 틴트 강조 / 하단 집계 카운트 배지 4개
- **색상 테마**: EPL_DARK(`#0d1117`) 배경 / 초록초록 `#00e676` / 초록 `#2ecc71` / 노랑 `#ffeb3b` / 빨강 `#e94560` / 테이블 헤더 `#0f3460` / 셀 배경 `#16213e`

---

### 슬라이드 03: 어제(05-11) 달성·미달 + 오늘 아침 직접 수정 결과
- **유형**: 현황
- **핵심 메시지**: 어제 3개 항목 전부 미달 — 오늘 아침 P2 직접 복구로 영입 회의 활용 페이지 9번째 확보
- **내용**:

  섹션 A — 어제(05-11) 결과 체크리스트

  | 항목 | 결과 | 세부 내용 |
  |------|------|-----------|
  | P7 v5.1 포지션 분리 모델 구현 | 미달 | success=true이나 코드 미반영, R²=0.586 동일 (5일 연속) |
  | P1 피처 추가 및 재학습 | 미달 | success=true이나 코드 미반영, Acc=54.79% 동일 (5일 연속) |
  | P2 JSON encoder 오류 수정 | 미달 | success=false — JSON encoder 오류로 파일 미생성 |

  섹션 B — 오늘 아침 직접 처리 (미팅 중 완료)

  | 항목 | 결과 | 세부 내용 |
  |------|------|-----------|
  | P2 JSON truncation 직접 복구 | 완료 | json.load() 파싱 성공, test R²=0.8877·MAE=0.96골 정상 확인. 대시보드 P2 페이지 파싱 오류 해결 |

  섹션 C — 근본 원인 및 오늘 접근 방식 변경

  - 5일 연속 실패 원인 1: GitHub Actions 워크플로가 기존 스크립트를 재실행하는 구조 — 코드 수정 없이 실행하면 결과 동일
  - 5일 연속 실패 원인 2: dev_result.json의 success:true 판단 기준이 "스크립트 종료 코드 0" — 목표 지표 달성 여부와 무관
  - 오늘 전략: train_v5_1.py·train_pipeline.py 파일에 코드 블록 직접 작성 (스크립트 실행이 아닌 코드 작성)

- **시각 요소**: 3분할 카드 레이아웃 (어제 미달 빨강 테두리 카드 / 오늘 달성 초록 테두리 카드 / 원인 파란 카드) / 체크 아이콘·X 아이콘 마커 / 즉시 활용 페이지 카운터 8→9 증가 표시
- **색상 테마**: EPL_DARK(`#0d1117`) 배경 / 미달 카드 배경 `#2a0a0a` 테두리 `#e94560` / 달성 카드 배경 `#0a2a0a` 테두리 `#00e676` / 원인 카드 배경 `#0f3460` / 텍스트 `#e6edf3`

---

### 슬라이드 04: 오늘 액션아이템 Top 3 — 코드 직접 작성
- **유형**: 액션아이템
- **핵심 메시지**: 오늘은 스크립트 실행이 아닌 코드 직접 작성 — P7·P1 파일에 로직을 명시적으로 구현
- **내용**:

  카드 1 (우선순위 1위 — 종일)
  - 담당: Marcus Webb
  - 과제: P7 v5.1 포지션 분리 모델 직접 구현
  - 파일: models/p7_growth_curve/train_v5_1.py
  - 핵심 코드: (1) pos_group으로 FW/MID 데이터 분리, (2) 각 포지션별 독립 XGBoost 학습, (3) gc_trend_3yr = market_value.shift(3) 피처 추가, (4) 포지션별 예측 결합 후 stacking
  - 완료 기준: results_summary.json version: v5.1 업데이트, R² >= 0.62 확인 후 커밋
  - 현재: test R²=0.586 (목표 대비 -0.034)
  - 목표: test R² >= 0.62 / val-test 갭 <= 0.07

  카드 2 (우선순위 2위 — 오전)
  - 담당: Marcus Webb
  - 과제: P1 피처 추가 및 재학습 (코드 직접 수정)
  - 파일: models/p1_match_result/train_pipeline.py
  - 핵심 코드: (1) elo_trend_3 = home_elo_pre.diff(3) 추가, (2) form_elo_composite = elo_diff * form_diff_5 추가, (3) StandardScaler → RobustScaler 교체, (4) XGBoost params n_estimators=600·max_depth=5·learning_rate=0.03 적용
  - 완료 기준: 수정 완료 후 학습 실행 및 커밋
  - 현재: XGBoost test Acc=54.79% (목표 대비 -2.21%p)
  - 목표: XGBoost test Acc >= 57%

  카드 3 (우선순위 3위 — 오후)
  - 담당: Marcus Webb
  - 과제: dev_result.json 성공 판단 로직 강화
  - 대상: GitHub Actions 워크플로 + dev_result.json 생성 로직
  - 핵심 변경: results_summary.json의 목표 지표(P7 R²>=0.60·P1 Acc>=0.57) 임계값 체크 추가. 달성 시만 success:true. success:false + 실제 지표값 기록 형식으로 개선
  - 목표: 수치 임계값 달성 여부가 반영된 dev_result.json 생성

- **시각 요소**: 3열 카드 레이아웃 / 카드 상단 순위 메달 배지 (금·은·동) / 목표 vs 현재 프로그레스 바 (P7·P1) / 기한 라벨 (종일·오전·오후) / 코드 스니펫 강조 박스 (카드 내 회색 배경 블록)
- **색상 테마**: EPL_DARK(`#0d1117`) 배경 / 카드 1 테두리 EPL_GOLD(`#f5a623`) / 카드 2 테두리 `#c0c0c0` / 카드 3 테두리 `#cd7f32` / 카드 배경 `#16213e` / 코드 블록 배경 `#0d1117` / 목표 지표 강조 EPL_ACCENT(`#e94560`)

---

### 슬라이드 05: 김태현 스카우트 종합 평가 — 4.85/5.0 + 5.0 로드맵
- **유형**: 분석
- **핵심 메시지**: P2 복구로 4.82→4.85(+0.03). P7·P1 달성 후 4.97, 포트폴리오 시연으로 5.0 도달
- **내용**:

  현재 점수: 4.85 / 5.0
  (전일 4.82 대비 +0.03 — P2 JSON truncation 오늘 아침 직접 복구, 영입 회의 즉시 활용 페이지 9개 달성)

  5.0 달성 로드맵:

  | 단계 | 조건 | 점수 변화 | 목표 시점 |
  |------|------|-----------|-----------|
  | 현재 | P2 복구 +0.03 반영 | **4.85** | 2026-05-12 시작 |
  | Step 1 | P7 v5.1 R² >= 0.62 실질 달성 (오늘) | +0.07 → **4.92** | 오늘 종일 |
  | Step 2 | P1 XGBoost Acc >= 57% 회복 (오늘~내일) | +0.05 → **4.97** | 오늘 오전~내일 |
  | Step 3 | 포트폴리오 시연 스크립트 완성 (이번 주) | +0.03 → **5.0** | 이번 주 내 |

  누적 달성 항목 (영입 회의 즉시 활용 가능 9개 페이지):

  | 항목 | 핵심 지표 | 평가 |
  |------|-----------|------|
  | P3 강등권 예측 | AUC 100% / Acc 100% | 최우수 |
  | P4 MVP 스코어링 | Spearman 0.9987 / NDCG@10 0.9993 | 최우수 |
  | S1 선수 종합 레이팅 | R² 0.9093 / MAE 4.66 | 최우수 |
  | P2 시즌 득점 예측 | R² 0.8877 / MAE 0.96골 | 우수 (오늘 JSON 복구) |
  | P6 시장가치 예측 | R² 0.8734 / MAE 3.28M | 우수 |
  | S2 포지션별 시장가치 | R² 0.8812 / MAE 3.23M | 우수 |
  | P8 이적 적응 예측 | AUC 0.6612 / F1 88.23% | 우수 |
  | S3 유사선수 탐색 | GK Sil 0.4873 / DEF 0.3637 | 양호 |
  | S6 하락세 감지 | AUC 0.8324 / F1 61.74% | 양호 |
  | 대시보드 16페이지 | 9개 즉시 활용 | 구축 완료 |

  포트폴리오 시연 스크립트 방향 (이번 주):
  - 목표 구단: 브렌트포드·노팅엄 포레스트 수준 중위권 구단 분석팀
  - 예산 가정: 30~50M 파운드
  - 시연 스토리라인: S1(선수 역량 확인) → S2(저평가 탐색) → P8(이적 적응 예측)

  김태현 최종 코멘트 (인용):
  "오늘 아침 P2 JSON이 직접 복구됐습니다. 영입 회의 활용 가능 페이지가 9개가 됐고, 포트폴리오 시연 준비를 시작할 수 있는 수준입니다. P7과 P1은 5일 연속 미달입니다. 오늘은 코드 작성으로 접근해야 합니다."

- **시각 요소**: 좌측 대형 아크 게이지 (4.85/5.0, EPL_GOLD) / 중앙 4단계 수평 로드맵 (4.85→+0.07→4.92→+0.05→4.97→+0.03→5.0) / 우측 누적 달성 체크리스트 표 / 하단 인용구 박스
- **색상 테마**: EPL_DARK(`#0d1117`) 배경 / 게이지 EPL_GOLD(`#f5a623`) / 로드맵 노드 `#0f3460`→`#e94560`→EPL_GOLD 그라데이션 / 달성 항목 `#00e676` / 인용구 박스 배경 `#16213e` 테두리 `#e94560`

---

### 슬라이드 06: Streamlit 대시보드 현황 — 16페이지·9개 즉시 활용
- **유형**: 결론
- **핵심 메시지**: 16페이지 대시보드 구축 완료 — 영입 회의 즉시 활용 9개 페이지, 포트폴리오 시연 준비 수준
- **내용**:

  현황 수치 (대형 타이포):
  - 총 페이지: 16개 (MENU_OPTIONS 기준)
  - 즉시 활용 가능: 9개 (오늘 P2 복구 후 8→9개)
  - 조건부·검토 중: 2개 (P7 조건부 / P1 비활성화 검토)
  - 기타: 5개

  영입 회의 즉시 활용 9개 페이지:

  | 페이지 | 연계 모델 | 핵심 지표 | 활용 여부 |
  |--------|-----------|-----------|-----------|
  | S1 선수 종합 레이팅 | S1 v1.2 | R²=0.9093, MAE=4.66 | 즉시 활용 |
  | P4 MVP 스코어링 | P4 | Spearman=0.9987, NDCG@10=0.9993 | 즉시 활용 |
  | P6 시장가치 예측 | P6 | R²=0.8734, MAE=3.28M | 즉시 활용 |
  | S2 저평가 탐색기 | S2 v4 | R²=0.8812, MAE=3.23M | 즉시 활용 |
  | P8 이적 적응 예측 | P8 v2 | AUC=0.6612, F1=88.23% | 즉시 활용 |
  | P3 강등권 탐색기 | P3 | AUC=100%, Acc=100% | 즉시 활용 |
  | P2 득점 예측 | P2 v2 | R²=0.8877, MAE=0.96골 | 즉시 활용 (오늘 JSON 복구) |
  | S3 유사선수 탐색 | S3 v4 | GK Sil=0.4873 / DEF=0.3637 | 실용적 |
  | S6 하락세 감지 | S6 | AUC=0.8324, F1=61.74% | 참고 활용 |
  | P1 경기 결과 예측 | P1 | Acc=54.79% (목표 57% 미달) | 비활성화 검토 |
  | P7 성장 곡선 | P7 v5 | R²=0.586 (목표 0.62 미달) | 조건부 |

  저평가 탐색 하이라이트 (S2 v4 기준):
  - Oliver Arblaster (Sheffield United, MF, age 20): 실제 45만 / 예측 248만 (5.5배 저평가)
  - Jakub Stolarczyk (Leicester, GK, age 24): 실제 100만 / 예측 425만 (4.3배 저평가)

  대시보드 URL:
  https://epl-scouting-report-dashboard-project-ffyb8msh6uafxyyg8txsm8.streamlit.app

- **시각 요소**: 상단 수치 배지 3개 (총 16페이지 / 즉시 활용 9개 / 검토 중 2개) / 중앙 9개 즉시 활용 페이지 카드 그리드 (3열 3행) / 각 카드에 모델 ID·핵심 지표·상태 배지 / 저평가 하이라이트 카드 (우측 하단) / URL 배지
- **색상 테마**: EPL_DARK(`#0d1117`) 배경 / 즉시 활용 카드 배경 `#16213e` 테두리 `#00e676` / 조건부 카드 배경 `#2a2a0a` / 비활성화 카드 배경 `#2a0a0a` / 수치 배지 숫자 EPL_GOLD(`#f5a623`) / URL 텍스트 EPL_ACCENT(`#e94560`)

---

## 디자인 시스템 요약

| 항목 | 값 |
|------|-----|
| 슬라이드 크기 | 720pt x 405pt (16:9) |
| 배경색 | `#0d1117` (다크모드 기본) |
| EPL_DARK | `#1a1a2e` |
| EPL_ACCENT | `#e94560` (강조 레드) |
| EPL_GOLD | `#f5a623` (성과 강조) |
| EPL_SUCCESS | `#00e676` (달성 초록) |
| 카드 배경 | `#16213e` |
| 포인트 블루 | `#0f3460` |
| 본문 흰색 | `#e6edf3` |
| 서브 텍스트 | `#8b949e` |
| 폰트 | Pretendard (CDN 로드) / fallback: system-ui, sans-serif |
| 텍스트 태그 규칙 | p, h1~h6, ul, ol 필수 — div 직접 텍스트 금지 |
| JS 사용 | 금지 (정적 HTML만) |

---

## 수치 출처 명세

| 수치 | 출처 파일 |
|------|-----------|
| P1 XGBoost test Acc 0.5479 / F1 0.4065 | models/p1_match_result/results_summary.json (models.xgboost.test) |
| P2 XGBoost test R² 0.8877 / MAE 0.9566 | models/p2_goal_scoring/results_summary.json (metrics.xgboost.test) |
| P3 LogReg/XGB test AUC 1.0 / Acc 1.0 | models/p3_relegation/results_summary.json (results_full_season.XGBoost.test) |
| P4 XGBoost test Spearman 0.9987 / NDCG@10 0.9993 | models/p4_mvp/results_summary.json (models.xgboost.test) |
| P5 K=3 Silhouette 0.2851 | models/p5_clustering/results_summary.json (kmeans.metrics.silhouette) |
| P6 XGBoost test R² 0.8734 / MAE 3,281,636 | models/p6_market_value/results_summary.json (model_metrics.XGBoost) |
| P7 LGBM test R² 0.586 / val-test 갭 0.055 | models/p7_growth_curve/results_summary.json (metrics.r2, val_test_gap) |
| P8 XGBoost AUC 0.6612 / F1 0.8823 | models/p8_transfer_adapt/results_summary.json (metrics.auc, metrics.f1) |
| S1 XGBoost test R² 0.9093 / MAE 4.6617 | models/s1_player_rating/results_summary.json (model_metrics.XGBoost.test) |
| S2 overall test R² 0.8812 / MAE 3,231,674 | models/s2_market_value/results_summary.json (overall_metrics) |
| S3 Silhouette FW 0.2085 / MID 0.2299 / DEF 0.3637 / GK 0.4873 | models/s3_similarity/results_summary.json (metadata.silhouette_by_position) |
| S6 XGBoost AUC 0.8324 / F1 0.6174 / Acc 0.75 | models/s6_decline/results_summary.json (model_performance.XGBoost) |
| 스카우트 평가 4.85/5.0 (전일 4.82에서 +0.03) | reports/daily_meeting/2026-05-12_meeting.md 섹션 4 |
| 5.0 로드맵 (+0.07, +0.05, +0.03) | reports/daily_meeting/2026-05-12_meeting.md 섹션 4 |
| 어제(05-11) 달성·미달 결과 | reports/daily_meeting/2026-05-12_meeting.md 섹션 2-1 |
| 대시보드 16페이지 / 즉시 활용 9개 | reports/daily_meeting/2026-05-12_meeting.md 섹션 1·2-4 |
| S2 저평가 1위 Oliver Arblaster | models/s2_market_value/results_summary.json (top20_undervalued[0]) |

---

*생성: EPL Scout PPT Organizer | 2026-05-12 09:00 KST*
*데이터 소스: 2026-05-12_meeting.md + 12개 models/*/results_summary.json 직접 독해*

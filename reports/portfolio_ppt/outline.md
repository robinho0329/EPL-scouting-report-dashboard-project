# EPL Scout Intelligence Dashboard — Portfolio PPT Outline

> 생성일: 2026-05-05 | 슬라이드: 14장 | 비율: 16:9 Widescreen

## 디자인 가이드

| 항목 | 내용 |
|------|------|
| 배경 | 흰색 (#FFFFFF), Cover만 Navy (#1B2A4A) |
| 주 색상 | Navy #1B2A4A / Blue #4472C4 / Light Blue #BDD7EE / Gray #595959 |
| 폰트 | Calibri (본문), Calibri Light (Cover 타이틀) |
| 헤더 | 좌측 브랜드명 / 중앙 섹션명 / 우측 "2026.04 \| Portfolio \| Confidential" |
| KEY INSIGHT | 하단 Light Blue (#BDD7EE) 박스 |
| 페이지번호 | 우측 하단 "N / 14" |

---

## 슬라이드 구성

### Slide 1 — Cover
- **배경**: Navy (#1B2A4A) + 좌측 Blue accent bar
- **제목**: "EPL Scout Intelligence Dashboard" (대형)
- **부제**: FBref + Transfermarkt 기반 EPL 스카우팅 의사결정 지원 시스템
- **정보**: 프로젝트 기간·데이터 출처·ML 모델·대시보드
- **기술 스택**: Python · XGBoost · LightGBM · Streamlit · Selenium · pandas · Optuna · SHAP

### Slide 2 — Executive Summary
- 5단계 파이프라인 흐름도 (크롤링→파이프라인→피처→ML→대시보드)
- KPI 5종: ML 14종 / 15페이지 / 9시즌 / 60+ 피처 / 3.7/5 평가
- 모델 구성 요약 표 (P1·P3·P5·P8·S2·S3)

### Slide 3 — Problem Definition (01. 문제 정의)
- 4가지 스카우팅 질문 2×2 레이아웃
- Q1 저평가 탐색 → S2, Q2 대체 선수 → S3, Q3 이적 리스크 → P8, Q4 강등 경보 → P3

### Slide 4 — Data Collection (02. 데이터 수집)
- FBref vs Transfermarkt 소스 박스 (레이트 리밋·수집 범위)
- data/raw/ 출력 구조 5종
- 체크포인트 재시작 기능 설명

### Slide 5 — Data Coverage (02. 데이터 커버리지)
- KPI: 3,420+ 경기 / 14,000+ rows / 60+ 피처 / 9시즌
- Train(2016-2021) / Val(2021-2023) / Test(2023-2025) 분리 시각화
- 전처리 5단계

### Slide 6 — Feature Engineering (03. 피처 엔지니어링)
- 6카테고리 카드 (공격·수비·경험/이적·팀·시장가치·선수 프로파일)
- 각 카테고리별 주요 피처 목록

### Slide 7 — S2 Market Value Model (04. 모델: S2)
- XGBoost 성능: R²=0.876, MAE≈3.4M€, MAPE 29.8%
- 저평가 Top3: Arblaster 6.61× / Morsy 4.33× / Stolarczyk 3.83×
- 과대평가 Top3: Greaves 0.21× / Sangaré 0.22× / Ugarte 0.23×
- v4 스마트 필터: 38세+ 제외 / 유스 잠재력 보정

### Slide 8 — S3 Player Clustering (04. 모델: S3)
- Before(전체 K=6, Sil 0.115) → After(포지션별 분리)
- 포지션별: FW K=7(0.2085) / MID K=7(0.2299) / DEF K=4(0.3637) / GK K=2(0.4873)
- 스카우팅 활용 3시나리오

### Slide 9 — P8 Transfer Adaptation (04. 모델: P8)
- 회귀 실패(R² 0.127) → 이진 분류 전환 스토리
- AUC 0.735 / F1 0.669 / Recall 0.722 / Acc 0.683
- 리스크 분류: High(514명) / Medium(649명) / Low(253명)

### Slide 10 — P1 Match Result (04. 모델: P1)
- Baseline F1 0.404 → v2 F1 0.479 (+18.7%)
- 무승부 Recall 0% → 22% 개선
- 피처 49 → 88개, ELO diff 1위 피처

### Slide 11 — P3 Relegation (04. 모델: P3)
- Full-season: XGBoost Test Acc 97.5%, F1 92.3%
- Mid-season(19R): XGBoost Test Acc 100%, F1 100%
- 4개 모델 성능 비교 테이블

### Slide 12 — Streamlit Dashboard (05. Streamlit)
- Streamlit Cloud URL 표시
- 15페이지 그리드 (4×3, 핵심 페이지 강조)

### Slide 13 — System Architecture (06. 아키텍처)
- 5단계 레이어 다이어그램 (포지션별 컬러)
- data/ 폴더 흐름도
- GitHub Actions 자동화 설정 (cron 상세)

### Slide 14 — Portfolio Summary (07. 요약)
- 배운 점 3가지 (번호 버블)
- 한계와 개선 방향 6항목
- 스카우트 실무 평가: 3.7/5.0 → 목표 4.0/5.0

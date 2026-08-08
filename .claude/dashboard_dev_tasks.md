# EPL 대시보드 디테일 개발 태스크

> epl-dashboard 에이전트가 이 파일을 읽고 순서대로 작업함
> 완료된 항목은 [x]로 표시

## 작업 규칙
- 각 항목 구현 후 반드시 **김태현 스카우트 검증** 실행 (3.5/5 이상 목표)
- 점수 미달 시 피드백 반영 후 재구현
- 대시보드 코드 수정 시 자동으로 포트 8520 재시작됨

---

## 우선순위 1 — 즉시 효과 큰 것

- [x] **p4_mvp_scoring 모델 학습**
  - 파일: `models/p4_mvp_scoring/`
  - 참고: `models/p4_mvp/` 코드 구조 그대로
  - 검증: 김태현 스카우트 — "MVP 후보 예측이 실전 스카우팅에 쓸 수 있나?"

- [x] **선수 이미지 적용**
  - 파일: `dashboard/utils/image_utils.py`
  - 현황: `data/images/` 에 이미지 있음 (image_crawl_checkpoint.db)
  - 작업: `get_player_image_b64()` 에서 실제 파일 로드 로직 완성
  - 검증: 홈, 스카우트 리포트 페이지에서 이미지 표시 확인

- [x] **팀 로고 적용**
  - 파일: `dashboard/utils/image_utils.py`
  - 작업: 20개 EPL 팀 로고 매핑 (team_profiles.py에 하드코딩된 팀명 활용)
  - 검증: 팀 프로파일 페이지에서 로고 표시 확인

---

## 우선순위 2 — 스카우트 실무 가치 향상

- [x] **쇼트리스트 영구 저장**
  - 파일: `dashboard/pages/shortlist.py`
  - 현황: 세션 기반 (새로고침 시 초기화)
  - 작업: `data/scout/shortlist.json` 에 파일로 저장/로드
  - 검증: 김태현 스카우트 — "실제로 관심 선수 목록 유지되나?"

- [x] **스카우트 리포트 PDF 내보내기**
  - 파일: `dashboard/pages/scout_report.py`
  - 작업: `st.download_button` + fpdf2로 PDF 생성
  - 포함: 선수 기본정보, S1~S6 점수, P6/P7 예측, 김태현 코멘트
  - 검증: 김태현 스카우트 — "이걸 감독한테 그대로 가져갈 수 있나?"

- [x] **비교 결과 Excel 내보내기**
  - 파일: `dashboard/pages/comparison.py`
  - 작업: `st.download_button` + pandas to_excel
  - 검증: 김태현 스카우트 — "비교표를 보고서에 붙여 쓸 수 있나?"

---

## 우선순위 3 — 고급 기능

- [x] **SHAP 시각화 페이지 추가**
  - 파일: `dashboard/pages/shap_explainer.py` (신규)
  - 작업: S1 WAR 모델 SHAP summary plot, force plot
  - 사이드바 메뉴에 추가
  - 검증: 김태현 스카우트 — "왜 이 선수 WAR이 높은지 설명할 수 있나?"

- [x] **이적 시나리오 시뮬레이터**
  - 파일: `dashboard/pages/scout_transfer.py` (탭4 전면 개편)
  - 작업: "A팀 → B팀 이적 시 적응 점수" 슬라이더 기반 What-if
    - 나이/출전 시간 비율/리그 순위 차이/전술 스타일 거리 슬라이더
    - 적응 점수 게이지 차트 (0~100, 실시간)
    - 팀 환경 레이더 차트 비교 (현재팀 vs 목적지팀)
    - 김태현 스카우트 종합 의견 (점수 구간별 4단계)
    - 리스크 지표: 유사 이적 사례 성공률 + 파이차트
    - 사이드바 "🔄 이적 시나리오" 메뉴 추가 (app.py)
  - 검증: 김태현 스카우트 — **4.0/5** (슬라이더로 시나리오 조정 후 즉시 감독 보고 가능 수준)

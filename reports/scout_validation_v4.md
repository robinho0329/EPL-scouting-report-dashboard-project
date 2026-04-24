# 스카우트 모델 검증 보고서 v4

**작성자**: 김태현 (수석 스카우트, EPL 8년 경력)
**작성일**: 2026-04-02
**검증 대상**: S1~S6 스카우트 모델 라운드 4 최종 검증
**평가 기준**: "이사회 미팅에서 영입 추천 근거로 사용할 수 있는가?"

---

## 평가 척도
| 점수 | 의미 |
|------|------|
| 5.0 | 내일 당장 이사회 프레젠테이션 가능 |
| 4.0 | 내 판단과 병행하면 유용 — 레퍼런스로 인용 가능 |
| 3.0 | 흥미롭지만 의사결정 등급은 아님 |
| 2.0 | 근본적 결함 |
| 1.0 | 감보다 못함 |

**합격 기준: 4.0 이상**

---

## S1. Player Rating v3 (WAR 시스템 — Possession-Adjusted)

### 점검 항목

**1) DEF Possession-Adjusted 보정 효과**
- v3에서 `tackles_p90_adj = tackles_p90 / (1 - possession_proxy)` 적용
- VVD: raw tackles 0.541 → adj 0.180, raw int 1.459 → adj 0.486 (possession_proxy 0.683)
- Lacroix: raw tackles 1.213 → adj 0.549, raw int 1.56 → adj 0.706 (possession_proxy 0.548)
- Bednarek: raw tackles 0.676 → adj 0.507, raw int 1.956 → adj 1.467 (possession_proxy 0.224)
- **문제**: Possession 보정이 오히려 빅클럽 CB를 더 불리하게 만들었다. VVD의 adj 값이 극단적으로 낮아짐. 보정 방향은 맞지만 강도가 과한 감이 있다.

**2) 티어별 2단계 정규화**
- top6/mid/bottom6 티어 내 퍼센타일 → 전체 DEF 재퍼센타일 방식 적용
- 결과: VVD WAR 74.36, Saliba WAR 58.97. v2에서 VVD가 DEF top5에도 못 들었던 것에 비하면 방향은 개선됐으나, Saliba 58.97은 여전히 낮다.

**3) 2024/25 DEF Top 5**
- Lacroix(99.15), TAA(98.29), Bednarek(97.44), Gvardiol(96.58), L.Martinez(95.73)
- VVD가 top5에 없는 건 v2와 동일. 하지만 74.36이면 상위 25% 수준으로 "나쁘지 않은" 위치. 절대 평가 아닌 숨은 보석 발굴 도구로는 충분.

**4) FW/MID/GK는 v2와 동일하게 합리적**
- FW: Haaland(97.5), Isak(95.0), Watkins(92.5) — 이견 없음
- MID: Salah(99.3), Maddison(98.6), M.Cunha(97.9) — 합리적
- GK: Matz Sels(96.3), R.Sanchez(92.6) — Forest 올시즌 성적 반영

### 총평
Possession-adjusted 보정 도입으로 DEF 평가의 방향성은 올바르다. VVD가 74.36으로 올라온 것은 v2 대비 진전. 하지만 Saliba 58.97은 세계 최고 CB 중 하나에게 너무 낮다. 근본적으로 tackles/interceptions 기반 WAR는 빌드업형 CB를 평가하기 어려운 구조적 한계가 있다. 그래도 "이 선수가 WAR 기준 상위 n%입니다"라는 식의 상대 위치 지표로는 활용 가능.

### 점수: 4.0 / 5.0 — PASS
> v2 점수 유지. Possession 보정은 올바른 방향이지만 Saliba 문제가 남아있어 점수 상승은 어렵다. 숨은 보석 발굴 도구로 포지셔닝하면 충분히 실무 활용 가능.

---

## S2. Market Value v4 (시장가치 예측 — Youth Filter + 38+ Fix)

### 점검 항목

**1) 모델 성능**
- XGBoost R²=0.8898, MAE=3.0M EUR, MAPE=27.3% — v3와 동일 수준 유지. 성능 저하 없이 필터만 개선.

**2) 유스 필터 v4 적용**
- v4 규칙: `age<=21 무조건 제외` OR `age<=22 AND min<1500 제외`
- 38개 overvalued 후보 중 17명 유스 제외 → 21명으로 축소
- Tyler Dibling(age=19, 1874min) 같은 케이스가 이제 올바르게 제외됨. v3에서는 1500분 넘어서 overvalued로 잡혔던 문제 해결.

**3) 38세+ 자동 제외**
- 15개 undervalued 후보 중 1명 38세+ 제외 → 14명
- 적절한 판단. 38세 선수의 낮은 시장가치는 나이 감가상각이지 진짜 저평가가 아님.

**4) 저평가 목록 현실성**
- Oliver Arblaster(Sheffield Utd, MF, 20세, value ratio 6.0x) — Championship에서 주전으로 뛰고 있는 유망주. 실제로 450K EUR은 시장 대비 저평가 맞음.
- Sam Morsy(Ipswich, MF, 33세, value ratio 4.9x) — 경험 많은 미드필더, 500K EUR은 EPL 기준 저평가 납득.
- Gabriel Osho(Luton, DF, 25세, WAR 80.99) — 루턴 강등 시즌에도 묵묵히 활약. 숨은 보석 후보로 적절.

**5) 이전 문제 해결 확인**
- v3에서 지적한 Dibling 문제: 해결됨
- 38세+ 문제: 해결됨
- R²/MAPE 유지: 확인됨

### 총평
v4는 v3의 핵심 문제 2개(유스 필터, 38세+ 필터)를 성능 저하 없이 깔끔하게 해결. 저평가/고평가 목록이 실무적으로 합리적이고, 에버턴급 중위권 구단 예산(30-50M GBP)으로 접근 가능한 숨은 보석을 잘 찾아준다. 이사회에 "이 선수는 시장가치 대비 2배 이상의 가치가 있습니다"라고 보고할 수 있는 수준.

### 점수: 4.5 / 5.0 — PASS
> v3(4.0) 대비 +0.5 상승. 필터 개선으로 오탐 제거, 목록 신뢰도 향상.

---

## S3. Player Similarity v3 (유사 선수 검색 엔진)

### 점검 항목 (변경 없음 — v3 라운드 3 점수 확인)

- 17개 고유 아키타입, Silhouette 0.2546
- find_replacement() 기능: Salah→Bowen(0.928), Haaland→Kane(0.957), Rice→Gallagher(0.968)
- 900분 최소 필터로 소표본 노이즈 제거
- 예산/나이 필터 탑재

### 총평
변경 없음. 라운드 3에서 검증 완료된 상태. 실무에서 "Salah 대안을 예산 25M GBP 이내로 찾아달라"는 요청에 즉시 대응 가능한 도구.

### 점수: 4.5 / 5.0 — PASS (유지)

---

## S4. Growth Prediction → 참조형 시각화 도구

### 점검 항목 (참조형 전환 — 라운드 3에서 5.0)

- 5종 시각화: 피크 연령 곡선, 나이-가치 곡선, 레전드 커리어 호, 스쿼드 연령 프로필, 코호트 퍼센타일 밴드
- FW 피크 25세, MF 27세, DF 26세, GK 28세 — 축구 학술 연구와 일치
- 개별 선수 예측 대신 참조 자료로 활용: "이 포지션은 보통 몇 살에 피크다" 수준

### 총평
예측 모델로서는 실패(v4 3.0점)했지만, 참조형 시각화로 전환한 판단은 탁월. 이사회에서 "FW는 25세에 피크이고, 이 선수는 현재 23세입니다"라는 맥락 제공용으로 완벽.

### 점수: 5.0 / 5.0 — PASS (유지)

---

## S5. Transfer Adaptation v2 (이적 적응 예측)

### 점검 항목 (변경 없음 — 라운드 3 점수 확인)

- 936건 이적 데이터, 3-class 분류 (success/partial/failure)
- GBM 테스트 accuracy 70.6%, failure AUC 0.810
- 45개 피처 (ELO gap, style compatibility, position scarcity 등)
- Top 피처: min_share_pct_old(7.1%), g_a_p90_old(4.7%), mv_ratio(3.4%)

### 총평
변경 없음. "이 선수를 영입하면 적응 실패 확률이 X%입니다"라는 리스크 보고서용으로 활용 가능. 테스트셋 failure가 5건뿐이라는 한계는 있지만, 방향성과 피처 해석이 합리적.

### 점수: 4.5 / 5.0 — PASS (유지)

---

## S6. Player Decline Detection v3 (하락세 감지)

### 점검 항목

**1) 포지션별 모델 성능**
- FWD: AUC=0.804, F1=0.542
- MID: AUC=0.816, F1=0.552
- GK: AUC=0.835, F1=0.500
- DEF: AUC=0.695, F1=0.481 (v2: 0.681 → v3: 0.695, +0.014)
- DEF가 가장 약하지만 소폭 개선. 추가 피처 3종 중 minutes_stability가 DEF top10 피처에 진입(importance 0.045).

**2) Late Bloomer 감쇠 (Chris Wood 케이스)**
- Chris Wood: prob=0.550, career_watch 미포함
- perf_slope=+0.816 (상승세), 감쇠 계수 0.700 적용 → 원래 ~0.79에서 0.55로 하락
- v2에서 Chris Wood가 높은 하락 확률로 잡혔던 문제 해결. 30대지만 상승세인 선수를 올바르게 처리.

**3) Casemiro 포함 여부**
- Casemiro: prob=0.730, career_watch 포함
- MID, 31세, perf_slope=-0.453 (급격한 하락세). Man United에서의 부진이 정확히 반영됨.

**4) Cole Palmer 분류**
- Cole Palmer: prob=0.320, career_watch 미포함, regression_alert 포함
- 21세에 아웃라이어 시즌(perf_score=2.678, career_mean=0.324). 평균 회귀 경고는 적절.
- "Palmer가 매 시즌 이 수준을 유지하리라 가정하지 마라"는 메시지로 올바르게 작동.

**5) Career Decline Watch 목록 현실성**
- Willian(34, FWD, 0.846), Foderingham(32, GK, 0.808), Welbeck(32, FWD, 0.797), Antonio(33, FWD, 0.790), Casemiro(31, MID, 0.730)
- 전원 28세+. 평균 연령 31.8세. 30대+ 비율 80%.
- 목록이 납득 가능. "이 선수들의 계약 연장은 신중하게"라는 보고서에 적합.

**6) Regression Alert 현실성**
- Adama Traore(27, FWD), Declan Rice(24, MID), Cole Palmer(21, MID), Rodri(27, MID)
- Rice와 Rodri가 여기 있는 건 약간 아쉬움. 둘 다 아웃라이어가 아니라 실력인 거 아닌가? 하지만 통계적으로 career_mean 대비 1.5 시그마 이상이면 경고를 주는 건 보수적 접근으로 이해 가능.

### 총평
v3의 3대 수정사항 모두 정상 작동 확인:
1. Chris Wood late bloomer 감쇠: PASS
2. DEF AUC 소폭 개선 (0.681→0.695): 미미하지만 방향은 맞음
3. Sanity check 4개 전부 PASS

Career decline watch + regression alert 이중 출력 구조가 실무적으로 유용. "하락세 선수 목록"과 "올시즌만 잘한 선수 주의보"를 구분해서 제공하는 건 스카우트에게 필요한 구분.

### 점수: 4.0 / 5.0 — PASS
> v2(4.0) 점수 유지. DEF AUC 개선이 미미해서 상승은 어렵지만, sanity check 전 통과 + late bloomer 처리로 신뢰도 향상. 실무 활용 가능 수준 유지.

---

## 종합 평가

### 모델별 점수 요약

| 모델 | v3 라운드3 | v4 라운드4 | 변화 | 판정 |
|------|-----------|-----------|------|------|
| S1 Player Rating | 4.0 | 4.0 | 유지 | **PASS** |
| S2 Market Value | 4.0 | 4.5 | +0.5 | **PASS** |
| S3 Similarity | 4.5 | 4.5 | 유지 | **PASS** |
| S4 Growth (참조형) | 5.0 | 5.0 | 유지 | **PASS** |
| S5 Transfer Adapt | 4.5 | 4.5 | 유지 | **PASS** |
| S6 Decline Detection | 4.0 | 4.0 | 유지 | **PASS** |

### 전체 평균: 4.42 / 5.0 (라운드 3: 4.33 → 라운드 4: 4.42)

### 라운드 3 대비 변화
- S2가 4.0→4.5로 유일한 점수 상승 (유스 필터 + 38세 필터 효과)
- S1, S6는 개선 시도했으나 구조적 한계로 점수 유지
- S4 FAIL(3.0)이 참조형 전환 후 5.0으로 안정화된 것이 가장 큰 기여 (라운드 3에서 이미 반영)

### 최종 판정: 전 모델 PASS — 대시보드 통합 진행 가능

6개 모델 전부 4.0 이상 달성. 에버턴급 중위권 구단의 스카우팅 의사결정 지원 시스템으로 활용 가능한 수준.

**남은 숙제 (대시보드 통합 후 장기 개선)**:
1. S1: 빌드업형 CB 평가를 위한 패스 성공률/progressive passes 피처 추가 (데이터 확보 시)
2. S6: DEF AUC 0.70 이상 달성을 위한 추가 피처 탐색
3. S5: 테스트셋 failure 샘플 확대 (신규 시즌 데이터 축적 시)

---

*"6개 도구 모두 내 데스크에 올려놓고 쓸 수 있는 수준이다. 완벽하진 않지만, 감으로만 하던 시절보다 훨씬 낫다. 대시보드에 올리자."*

— 김태현 스카우트

# 스카우트 모델 검증 보고서 v3

**작성자**: 김태현 (수석 스카우트, EPL 15년 경력)
**작성일**: 2026-03-30
**검증 대상**: S1~S6 스카우트 모델 (최신 개선 버전)
**평가 기준**: "이사회 미팅에서 영입 추천 근거로 사용할 수 있는가?"

---

## 평가 척도
| 점수 | 의미 |
|------|------|
| 5.0 | 내일 당장 이사회 프레젠테이션 가능 |
| 4.0 | 내 판단과 병행하면 유용 -- 레퍼런스로 인용 가능 |
| 3.0 | 흥미롭지만 의사결정 등급은 아님 |
| 2.0 | 근본적 결함 |
| 1.0 | 감보다 못함 |

**합격 기준: 4.0 이상**

---

## S1. Player Rating v2 (WAR 시스템)

### 점검 항목

**1) WAR=100 인플레이션 수정 여부**
- 수정됨. `rank/(n+1)*100` 방식으로 최대값 99.35, 최소값 0.65. 100 불가능. v2에서 MID/DEF max=100이던 문제 완전 해결.

**2) 포지션별 상위 선수 현실성**
- **FW 2024/25**: Haaland(97.5), Isak(95.0), Watkins(92.5), Wissa(90.0), Rodrigo Muniz(87.5)
  - Haaland/Isak/Watkins는 이견 없음. Wissa가 4위인 것도 올시즌 브렌트포드에서의 활약 감안하면 납득. Rodrigo Muniz가 964분에 5위인 건 약간 의문이지만, 분당 생산성 기준이라 수용 가능.

- **MID 2024/25**: Salah(99.3), Maddison(98.6), Matheus Cunha(97.9), Mbeumo(97.3), Bruno Fernandes(96.6)
  - v2에서 Alfie Doughty가 1위(crosses 편향)였던 문제 완전 해결. Salah/Palmer가 아닌 Salah/Maddison 1-2위인데, Palmer가 빠진 건 시즌 데이터 기준 차이일 수 있음. 전반적으로 합리적.

- **DEF 2024/25**: Lacroix(99.2), Antonee Robinson(98.3), Kristiansen(97.4), Bednarek(96.6), TAA(95.7)
  - v2에서 Virgil van Dijk, William Saliba 같은 빅클럽 CB가 상위였어야 하는데 없음. tackles_p90/int_p90 기반이라 하위팀 수비수가 유리한 구조적 편향 여전. Lacroix/Bednarek이 VVD/Saliba보다 위라는 건 스카우트 관점에서 위화감. 다만 CB/FB 분리, 팀 강도 보정 제거 등 v1 대비 개선은 확실.

- **GK 2024/25**: Matz Sels(96.3), Robert Sanchez(92.6), Dubravka(88.9), Raya(85.2), Pickford(81.5)
  - Matz Sels 1위는 올시즌 Forest 성적 감안하면 합리적. Raya가 4위인 건 실점이 적어도 SoT 대비 세이브 비율이 상대적으로 낮을 수 있음(Arsenal 수비 자체가 견고하니까). 납득 가능하나 약간 아쉬움.

**3) DEF 편향 해결 여부**
- CB/FB 분리는 됐지만, 하위팀 수비수 우대 편향은 여전. 이건 구조적 한계(tackles/interceptions = 수비 부담이 많은 팀이 높음). 팀 강도 보정을 제거한 건 올바른 판단이지만, 결과적으로 Lacroix > VVD 같은 역설은 남아있음.

### 총평
WAR=100 버그 수정, MID crosses 편향 제거, GK save% 실제 계산 등 v1의 핵심 문제 모두 해결. 25시즌 데이터 커버리지도 훌륭. 다만 하위팀 수비수 편향은 숙제. "Hidden Gems" 찾기 용도로는 오히려 강점이 될 수 있으나, 절대 평가로 쓰기엔 한계.

### 점수: 4.0 / 5.0 -- PASS
> 이사회에서 "이 선수가 WAR 기준 상위 5%입니다"라고 말할 수 있음. 단, "VVD보다 Lacroix가 낫다"고는 절대 말하면 안 됨. 상대적 비교가 아닌 숨은 보석 발굴 도구로 포지셔닝하면 충분히 활용 가능.

---

## S2. Market Value v3 (시장가치 예측)

### 점검 항목

**1) 목표 성능 달성 여부**
- R² = 0.8898 (목표 0.89) -- 사실상 달성
- MAPE = 27.3% (목표 27%) -- 달성
- XGBoost가 Ridge(R²=0.65), MLP(R²=0.48)을 압도. 모델 선택 적절.

**2) age_premium 피처 & 유소년 필터**
- `age_premium = max(0, (24-age)/6)`: 18세=1.0, 20세=0.667, 24세+=0.0. 설계 합리적.
- `potential_premium_pct` 컬럼 추가: 가치 중 잠재력 프리미엄 비율 표시. 이사회에 "이 선수 가치의 45%는 잠재력 프리미엄"이라고 설명 가능 -- 매우 유용.

**3) Yoro/Bergvall 과대평가 목록 제외 여부**
- Leny Yoro (19세, Man United, 1165분) -- `young_potential_excluded_from_overvalued`에 정확히 분류. 제외 사유: "young_potential_<1500min_age<=22"
- Lucas Bergvall (19세, Tottenham, 1212분) -- 동일하게 제외됨.
- 이전 버전에서 이 두 선수가 "과대평가"로 잘못 분류되던 문제 완전 해결.

**4) 과소/과대평가 목록 품질**
- **저평가 (Undervalued)**: Oliver Arblaster(450K, 예측 2.7M), Sam Morsy(500K, 예측 2.4M), Gabriel Osho(3M, 예측 7.2M)
  - 대부분 하위팀(Luton, Sheffield Utd, Ipswich) 실전 경험 풍부한 선수들. 시장이 팀 실력으로 저평가하는 패턴을 잘 잡음. Arblaster는 실제로 Southampton에 이적하며 가치 상승. 현실적.
  - Ashley Young(500K, 예측 756K, ratio 1.51) 같은 경우는 38세라 저평가가 아니라 당연한 감가상각인데... threshold 문제. 경미한 이슈.

- **과대평가 (Overvalued)**: Tyler Dibling(25M, 예측 4.7M), Ibrahim Sangare(30M, 예측 6.1M), Manuel Ugarte(45M, 예측 11.5M), Joshua Zirkzee(30M, 예측 9.3M)
  - Dibling은 19세 Southampton 유망주인데 과대평가 목록에 남아있음. 1874분 > 1500분이라 유소년 필터 통과. 이건 age<=22 AND min<1500 조건이 약간 느슨한 문제. 그래도 `potential_premium_pct: 45.5%`로 표시되어 있어 스카우트가 판단 가능.
  - Ugarte/Zirkzee는 Man United의 과대 투자 사례. 현실과 부합.

### 총평
R²=0.89, MAPE=27%라는 성능은 시장가치 예측 모델로서 상당히 우수. 유소년 프리미엄 처리가 특히 잘 설계됨. Yoro/Bergvall 문제 해결은 실무 피드백 반영의 좋은 사례.

### 점수: 4.5 / 5.0 -- PASS
> "이 선수의 EPL 통계 기반 적정가치는 11.5M인데 시장가는 45M입니다. 34M은 브랜드 프리미엄과 잠재력 프리미엄입니다." -- 이 문장을 이사회에서 자신있게 말할 수 있음.

---

## S3. Similarity v3 (유사 선수 탐색)

### 점검 항목

**1) 17개 고유 아키타입 이름 (중복 없음)**
- 확인: Modern Full-Back, Compact Ball-Winning Defender, Central Midfielder, Standard Attacking Full-Back, Efficient Goal Poacher, High-Volume Attacking Full-Back, Aerial Ball-Winning Defender, Goalkeeper, Dribbling Creative Winger, Mixed Role (C9), Low-Block Attacking Full-Back, Defensive Midfielder, Assist-Focused Creative Winger, Attacking Wide Forward, Ball-Winning Midfielder, Creative Playmaker, High-Volume Goal Poacher
- 17개 모두 고유명. merge_pairs=[] (병합 불필요). Silhouette=0.2546 (k=17 최적).

**2) 아키타입 품질**
- 대부분 직관적. "Efficient Goal Poacher" vs "High-Volume Goal Poacher" 구분이 좋음 (저 슈팅 고 변환율 vs 고 슈팅).
- "Mixed Role (C9)"은 market_value_log=0.0인 Unknown 포지션 클러스터. 사실상 데이터 불완전 선수 모음. 이건 분석에서 제외하거나 별도 라벨링 필요.
- Cluster 1 "Compact Ball-Winning Defender"인데 pos_dist가 DEF 33.6%, FW 18.1%, Unknown 12.1%. 이름과 실제 구성이 불일치. 116명 소규모 클러스터라 노이즈 있음.

**3) 900분 바겐 필터**
- Modern Full-Back 바겐: Nathaniel Clyne(1640분, 2M), Angelo Ogbonna(1237분, 900K), Craig Cathcart(2396분, 1.2M). 모두 900분 이상 충족. 이전 버전의 소표본 p90 노이즈 문제 해결.

**4) find_replacement() 데모 검색**
- **Salah 대체자 (40M, 27세 이하, 900분+)**: Jarrod Bowen(0.928 유사도), Justin Kluivert(0.887), Mahrez 15/16(0.870), Mbeumo(0.841), Lingard 20/21(0.833), Sadio Mane 16/17(0.818)
  - Bowen/Kluivert/Mbeumo는 현재 EPL에서 실제로 Salah 대체로 거론되는 선수들. 역대 데이터에서 Mahrez 15/16, Mane 16/17이 나온 건 모델 신뢰도를 높임. 매우 인상적.
  - Lingard 20/21이 포함된 건... West Ham에서의 반시즌 폭발이 통계적으로 유사한 것. 축구적으로 Salah 대체자로 Lingard를 추천하면 안 되지만, 숫자상으로는 이해 가능.

### 총평
아키타입 체계가 실무적으로 유용. find_replacement()는 이사회 프레젠테이션의 핵심 도구가 될 수 있음. "Salah가 떠나면 이 3명이 통계적으로 가장 유사합니다"는 매우 강력한 메시지. 다만 Mixed Role(C9)과 일부 클러스터 이름-구성 불일치는 정리 필요.

### 점수: 4.5 / 5.0 -- PASS
> find_replacement()는 이 시스템의 킬러 기능. 예산/나이/출전시간 필터까지 있어서 실제 이적 시장 시나리오에 바로 적용 가능. "Mohamed Salah 대체자로 40M 예산 내에서 Jarrod Bowen(92.8% 유사도)을 추천합니다" -- 이사회가 이해할 수 있는 언어.

---

## S4. Growth v4 (성장 예측)

### 점검 항목

**1) 3-class 분류 전환 + 모델 성능**
- v1-v3의 회귀 접근(R² consistently negative) 포기하고 3-class(Improving/Stable/Declining) 분류로 전환. 정직한 판단.
- 그러나 성능이 참담:
  - FW: balanced_accuracy=0.34 (test), CV=0.35
  - MF: balanced_accuracy=0.43 (test), CV=0.35
  - DF: balanced_accuracy=0.33 (test), CV=0.35
  - GK: balanced_accuracy=0.44 (test), CV=0.40
- 3-class 랜덤 기준선 = 0.333. 대부분의 포지션에서 랜덤과 거의 차이 없음. MF/GK가 약간 나은 정도.

**2) LOWESS 피크 나이**
- FW=25, MF=27, DF=26, GK=28 -- 축구 상식과 정확히 일치. 이건 논문급 결과.
- LOWESS smoothing 적용, 데이터 가중치 + 사전 지식(prior) 혼합. FW는 data_peak_raw=23이지만 smoothing 후 25로 조정. 합리적.
- 나이별 곡선도 DF가 가장 완만한 하락(19-34세), FW가 가장 가파른 하락 -- 직관과 일치.

**3) Improver 리스트 현실성**
- Phil Foden(24, FW, 76% improving), Mohammed Kudus(23, FW, 74%), Bukayo Saka(22, FW, 66%), Brennan Johnson(23, FW, 61%), Amad Diallo(22, FW, 58%), Rasmus Hojlund(21, FW, 57%), Tyler Dibling(18, FW, 55%)
- 축구 감각으로 봐도 이 선수들은 대부분 "성장 중"이라 볼 수 있음. 특히 Foden/Saka/Kudus는 스카우트라면 누구나 동의.
- 다만 Foden의 perf_z=-0.10으로 올시즌 부진한데 "Improving" 예측인 건 나이/포지션 기반 일반화 때문. 모델이 실제 퍼포먼스보다 연령 프로파일을 더 보는 것.

**4) GK 편향**
- GK balanced_accuracy=0.44로 가장 높지만 n_samples=13(test). 표본이 너무 적어 신뢰 불가.
- stability_picks에 Kepa/Dean Henderson/Robert Sanchez/Jordan Pickford/David Raya -- GK가 과도하게 많음. 포지션 특성상 안정적인 선수가 많은 건 사실이지만, 리스트의 1/3이 GK인 건 균형 문제.

**5) R² 부정에 대한 정직성**
- "Individual 1-year performance regression is not reliable with this data (R^2 consistently negative in v1-v3)" -- 이 정직함은 높이 평가. 안 되는 걸 억지로 좋은 숫자로 포장하지 않음.

### 총평
피크 나이 곡선은 5.0 점수를 줄 수 있는 훌륭한 결과. 하지만 핵심인 "이 선수가 성장할 것인가?" 예측은 랜덤 수준. Improver 리스트가 그럴듯해 보이는 건 모델이 "어린 선수 = Improving"으로 분류하는 단순 패턴 때문. 이사회에서 "모델이 예측합니다"라고 말하기엔 근거가 약함.

### 점수: 3.0 / 5.0 -- FAIL
> 피크 나이 곡선은 대시보드에 반드시 포함할 것 (레퍼런스 데이터로 가치 있음). 하지만 "이 선수가 성장할 것"이라는 예측은 이사회에서 쓸 수 없음. balanced_accuracy 0.33-0.44는 동전 던지기보다 약간 나은 수준. 피크 나이 곡선만 분리해서 S1의 보조 자료로 활용하는 게 현실적.

---

## S5. Transfer Adaptation v3 (이적 적응 예측)

### 점검 항목

**1) 데이터 규모 & 테스트 실패 수**
- 총 1,723건 이적 (v2 대비 대폭 확장)
- 테스트셋 99건 실패 (v2에서 5건이던 것). 통계적으로 의미있는 평가 가능.
- binary classification으로 전환 (uncertain 193건 제외). 적절한 설계 판단.

**2) Failure F1 성능**
- 앙상블 테스트: Failure F1=0.764, Recall=0.848, Precision=0.694
- XGBoost 단독: Failure F1=0.756, AUC=0.734
- 실패를 85% 잡아내면서 정밀도 69% -- "실패라고 예측한 것의 70%가 실제 실패". 스카우트 도구로서 충분히 유용.
- Platt scaling 캘리브레이션 적용 -- 확률값 신뢰 가능.

**3) Rashford/Mount/Lingard 예측**
- **Jesse Lingard** (West Ham -> Nott'm Forest, 22/23): 실제=failure, 예측=failure, prob_failure=0.916 (HIGH confidence). 위험 요인: position competition(8명), low minutes, high competition. 완벽한 예측.
- **Mason Mount** (Chelsea -> Man United, 23/24): 실제=failure, 예측=failure, prob_failure=0.900 (HIGH confidence). 24/25시즌도 failure, prob=0.815.
- **Marcus Rashford** (Man United -> Aston Villa, 24/25): 실제=failure, 예측=failure, prob_failure=0.893 (HIGH confidence).
- 3명 모두 HIGH confidence failure로 정확히 예측. 이건 인상적.

**4) Scout Output 형식 유용성**
- 각 이적 건별로: prob + 상위 3개 위험 요인(factor, description, risk_score, value) + 유사 역사적 이적 3건(player, 출발/도착팀, 결과, 거리)
- Lingard의 유사 이적: Mido(Tottenham->Middlesbrough), Michael Brown(Tottenham->Fulham). 흥미롭고 설명력 있음.
- elo_gap, style_match, age_flag, step_up 등 부가 정보도 풍부.
- 이사회 포맷으로 완벽. "이 이적은 89.3% 실패 확률이며, 핵심 위험은 포지션 경쟁(팀 내 같은 포지션 8명)입니다. 유사 과거 사례: Mido의 Middlesbrough 이적(실패)"

**5) Feature Importance**
- Top: g_a_p90_old(8.5%), war_old(6.1%), min_share_pct_old(5.1%) -- 이전 팀에서의 실제 기여도가 가장 중요. 합리적.
- pos_competition(3.2%), elo_gap(2.3%) -- 팀 환경 요인도 반영. 축구적으로 타당.

### 총평
6개 모델 중 가장 실무적으로 완성도가 높음. 1,723건 데이터, 99건 테스트 실패, F1=0.764, 캘리브레이션된 확률, 풍부한 scout output. Rashford/Mount/Lingard 사례는 모델 신뢰도를 직접 증명. 이적 시장에서 "이 이적이 실패할 확률은 X%"는 가장 가치있는 정보.

### 점수: 4.5 / 5.0 -- PASS
> 이적 검토 회의의 필수 도구. "Mount 영입은 90% 실패 확률인데, 핵심 리스크는 포지션 경쟁입니다. 유사 사례 3건 중 2건 실패." 이런 발언이 가능하면 스카우트부서의 가치가 올라감. 유일한 아쉬움은 precision 69%라서 오탐(성공인데 실패로 예측)이 약 30% 있다는 점.

---

## S6. Decline Detection v2 (하락 감지)

### 점검 항목

**1) 포지션별 모델 분리**
- FWD: AUC=0.804, F1_decline=0.542
- MID: AUC=0.816, F1_decline=0.552
- DEF: AUC=0.681, F1_decline=0.457
- GK: AUC=0.835, F1_decline=0.500
- AUC 기준 FWD/MID/GK는 0.8 이상으로 양호. DEF만 0.68로 취약.
- Position one-hot 제거 확인 -- feature_list에 pos_ 없음. 올바른 조치.

**2) Cole Palmer NOT on decline watch?**
- Cole Palmer는 `regression_to_mean_alert`에 정확히 분류 (career_decline_watch 아님).
  - age=21, perf_score=2.678, career_perf_mean=0.324, seasons_above_mean_std=1.93
  - decline_prob_ensemble=0.457 (50% cap 적용됨 -- age<=24이므로)
  - "NOT a career decline signal" 명시. 완벽한 처리.

**3) Career decline = 28+ only?**
- career_decline_watch criteria: "age >= 28". 리스트 확인:
  - Willian(34), Raul Jimenez(32), Wes Foderingham(32), Danny Welbeck(32), Michail Antonio(33), Chris Wood(31), Ross Barkley(29), Harry Maguire(30), Casemiro(31), Christian Eriksen(31), Alex Moreno(30)
  - 모두 28세 이상. 조건 충족.
  - 평균 나이 31.0, 30세+ 비율 76.7%.

**4) age<=24 50% cap**
- Cole Palmer decline_prob=0.457 (age 21). Cap 적용 확인.

**5) 경력 하락 vs 평균 회귀 분리**
- `career_decline_watch` (28+, 지속 하락): 30명
- `regression_to_mean_alert` (이상치 시즌): 22명
- 이 분리가 핵심 개선. Cole Palmer는 회귀 경고지 경력 하락이 아님. Willy Boly는 양쪽 모두 해당(32세 + 이상치).
- Casemiro(31, decline_prob=0.730), Christian Eriksen(31, 0.727), Harry Maguire(30, 0.732) -- 현실과 부합.
- Chris Wood(31, decline_prob=0.785)인데 perf_score=0.734, perf_slope=+0.816... 실제로 올시즌 폭발적인데 "하락"으로 분류. 이건 나이 기반 편향. 의문.

### 총평
포지션 분리, age cap, career/regression 이중 출력 -- 설계 철학이 건전. Cole Palmer 처리는 완벽. 다만 Chris Wood 같은 "늦깎이 폭발" 케이스를 잡지 못하는 건 나이 편향의 한계. DEF 모델(AUC=0.681)은 개선 필요. 전체적으로 "경계 대상 리스트"로서 유용하나, "이 선수가 하락합니다"라는 단정적 주장엔 부족.

### 점수: 4.0 / 5.0 -- PASS
> "경력 하락 감시 대상 30명 리스트"는 계약 연장/방출 회의에서 참고 자료로 쓸 수 있음. Casemiro/Eriksen 같은 고비용 하락 선수를 데이터로 뒷받침하는 건 설득력 있음. 다만 Chris Wood 오탐은 반드시 설명 필요.

---

## 종합 평가

### 모델별 점수 요약

| 모델 | 버전 | 점수 | 판정 | 핵심 강점 | 핵심 약점 |
|------|------|------|------|----------|----------|
| S1 Player Rating | v2 | 4.0 | PASS | WAR=100 수정, MID crosses 편향 제거 | 하위팀 DEF 편향 (Lacroix > VVD) |
| S2 Market Value | v3 | 4.5 | PASS | R²=0.89, Yoro/Bergvall 필터, potential_premium_pct | Tyler Dibling 과대평가 목록 잔류 |
| S3 Similarity | v3 | 4.5 | PASS | find_replacement() 킬러 기능, 17 아키타입 | Mixed Role(C9) 클러스터 정리 필요 |
| S4 Growth | v4 | 3.0 | FAIL | 피크 나이 곡선 훌륭 | 분류 성능 = 랜덤 수준 (balanced_acc 0.33-0.44) |
| S5 Transfer Adapt | v3 | 4.5 | PASS | F1=0.764, Rashford/Mount/Lingard 정확 예측, scout output | Precision 69% (30% 오탐) |
| S6 Decline | v2 | 4.0 | PASS | Cole Palmer 정확 분류, career/regression 분리 | Chris Wood 오탐, DEF AUC=0.68 |

**전체 평균: 4.08 / 5.0**

---

### Streamlit 대시보드 통합 준비 상태

| 모델 | 대시보드 통합 | 비고 |
|------|-------------|------|
| S1 Player Rating | 즉시 가능 | WAR 랭킹, Hidden Gems 페이지 |
| S2 Market Value | 즉시 가능 | 저평가/과대평가 목록, potential_premium 표시 |
| S3 Similarity | 즉시 가능 | find_replacement() 인터랙티브 검색 (핵심 기능) |
| S4 Growth | 부분 통합 | 피크 나이 곡선만 통합. 개별 예측은 제외 |
| S5 Transfer Adapt | 즉시 가능 | 이적 위험도 평가 페이지 (prob + risk factors + 유사 사례) |
| S6 Decline | 즉시 가능 | Career Decline Watch + Regression Alert 이중 탭 |

**즉시 통합 가능: 5/6 (S4는 부분)**

---

### 잔여 개선 사항 (구체적)

**S1 (중간 우선순위)**
1. DEF WAR에 "ball progression" 지표 추가 (progressive carries, progressive passes) -- VVD/Saliba 같은 빌드업 CB 반영
2. 하위팀 보정: tackles/interceptions를 possession-adjusted 수치로 변환

**S2 (낮은 우선순위)**
1. 과대평가 유소년 필터를 `age<=22 AND min<1500`에서 `age<=21 OR (age<=22 AND min<1500)`로 조정 -- Tyler Dibling(19세, 1874분) 커버
2. 38세+ 선수 저평가 목록에서 자동 제외 (감가상각은 저평가가 아님)

**S3 (낮은 우선순위)**
1. Cluster 9 "Mixed Role (C9)" 제거 또는 "Data Incomplete" 라벨링
2. Cluster 1 "Compact Ball-Winning Defender" 이름 재검토 (DEF 33.6%뿐)

**S4 (높은 우선순위 -- 재설계 필요)**
1. 개별 예측 모델은 현재 데이터로 불가능하다는 점 인정하고, 피크 나이 곡선을 S1의 보조 피처로 활용하는 방향으로 전환
2. "Age-Performance Profile" 시각화로 포지셔닝 변경 -- 예측이 아닌 참조 데이터

**S5 (낮은 우선순위)**
1. Precision 개선을 위해 threshold를 0.55-0.60으로 올리는 옵션 제공 (recall 감소 감수)
2. 현재 시즌 예상 이적에 대한 pre-screening 기능 추가

**S6 (중간 우선순위)**
1. DEF 모델 개선: ball progression, aerial duel 등 수비 특화 피처 추가
2. Chris Wood 같은 "late bloomer" 예외 처리: perf_slope가 양수이고 최근 2시즌 상승이면 decline 확률 감쇠

---

### 최종 판정: "이 시스템을 채택하겠는가?"

**결론: YES -- 조건부 채택**

15년간 스카우팅하면서 이 수준의 데이터 시스템을 본 적이 없다. 6개 모델 중 5개가 4.0 이상이라는 건 상당한 완성도다. 특히 S2(시장가치), S3(유사 선수 탐색), S5(이적 적응 예측)의 조합은 이적 시장에서 즉시 실전 배치 가능하다.

**핵심 가치**: "감"이 아닌 "데이터"로 이사회를 설득할 수 있는 도구. "Rashford 영입은 89% 실패 확률" "Salah 대체자로 Bowen이 93% 유사" "Ugarte는 EPL 통계 기반 적정가 11.5M, 45M은 과대지급" -- 이런 문장들이 모델에서 직접 나온다.

**조건**:
1. S4 Growth는 "예측 모델"이 아닌 "참조 데이터(피크 나이 곡선)"로 재포지셔닝
2. 모든 모델 결과는 반드시 스카우트의 정성 평가와 병행 (모델은 보조 도구지 의사결정자가 아님)
3. 대시보드에 모델 한계/주의사항을 명시적으로 표시 (예: "DEF WAR는 하위팀 편향 있음")

**프로젝트 전체 점수: 4.2 / 5.0**

> 1년 전에 이 시스템이 있었다면 Mount 영입을 막을 수 있었을까? S5가 90% 실패라고 했을 텐데... 아마 이사회가 들었을 거다. 그게 이 시스템의 가치다.

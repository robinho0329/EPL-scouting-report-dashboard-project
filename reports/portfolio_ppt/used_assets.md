# Used Assets — EPL Scout Portfolio PPT

> 생성일: 2026-05-05

## 데이터 소스 (읽은 파일)

| 파일 | 용도 | 주요 수치 |
|------|------|----------|
| `models/s2_market_value/results_summary_v4.json` | S2 모델 성능·저평가/과대평가 목록 | R²=0.876, MAE=3.4M€ |
| `models/p3_relegation/results_summary.json` | P3 모델 성능 | Acc 97.5%, F1 92.3% |
| `models/p8_transfer_adapt/results_summary.json` | P8 모델 성능·리스크 분류 | AUC 0.7348, F1 0.6691 |
| `models/p1_match_result/results_summary.json` | P1 모델 성능·피처 중요도 | F1 0.4791, 88 features |

## 수치 검증 (파일에서 직접 확인)

### S2 시장가치 예측 (v4)
- XGBoost Test R² = **0.876** (0.8763)
- XGBoost Test MAE = **3,356,941€** (≈3.4M€)
- XGBoost Test MAPE = **29.8%**
- 학습: 8,800건 / Val: 1,210건 / Test: 1,234건
- 저평가 Top3: Oliver Arblaster 6.609× / Sam Morsy 4.327× / Jakub Stolarczyk 3.831×
- 과대평가 Top3: Jacob Greaves 0.210× / Ibrahim Sangaré 0.220× / Manuel Ugarte 0.234×
- 유스 필터(v4): age≤21 OR (age≤22 AND min<1,500) → 18명 제외
- 노인 필터(v4): age≥38 → 1명 제외 (Ashley Young)

### P3 강등 예측
- Full-season XGBoost Test: Acc=97.5%, F1=92.3%, AUC=100%
- Full-season XGBoost Val: Acc=97.5%, F1=90.9%
- Mid-season XGBoost Test: Acc=100%, F1=100%
- Mid-season XGBoost Val: Acc=90.0%, F1=60.0%

### P8 이적 적응 예측
- 앙상블(XGB+LR+RF) AUC=**0.7348**, F1=**0.6691**, Recall=**0.7222**, Acc=0.6831
- 타겟: G+A/90 이전 시즌 80% 이상 유지 (90s_new ≥ 5.5 필터)
- 리스크: High(≤0.40)=514명 / Medium=649명 / Low(≥0.70)=253명
- 상위 피처: pos_code(0.0727) > g_a_per90_rel(0.0588) > src_La Liga(0.0532)

### P1 경기 결과 예측
- v2 XGBoost Test: F1=**0.4791**, Acc=0.5233
- Baseline: F1=0.4037, Acc=0.5384
- 개선: ΔF1=+0.0754 (+18.7%)
- 무승부 Test confusion: [63, **38**, 69] → Recall 22.4%
- 피처: 49 → 88개 | 상위 피처: elo_diff(0.109) > elo_ratio(0.071)
- 학습 7,890 / Val 760 / Test 730 경기

## 이미지 (미사용 — 텍스트 기반 슬라이드로 대체)

아래 파일들은 존재하나 이번 PPT에 삽입하지 않음 (텍스트 기반 수치 표현으로 대체):
- `models/s2_market_value/figures/predicted_vs_actual_v4.png`
- `models/s2_market_value/figures/undervalued_war_scatter_v4.png`
- `models/s2_market_value/figures/feature_importance_v4.png`
- (기타 22개 PNG 파일)

## 출력 파일

| 파일 | 크기 |
|------|------|
| `epl_scout_dashboard_portfolio.pptx` | 570KB |
| `outline.md` | - |
| `used_assets.md` | - |
| `create_pptx.js` | (생성 스크립트, 재실행 가능) |

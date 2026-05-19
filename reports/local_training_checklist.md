# 로컬 학습 실행 체크리스트
**버전:** v1.0  
**작성일:** 2026-05-19  
**목적:** P7 v5.1 + P1 재학습 실행 순서, 데이터 경로, 커밋 방법 명시

---

## 전제 조건 확인

```bash
# 데이터 경로 확인 (로컬 PC에서만 존재)
ls data/features/scout_features.parquet   # P7용
ls data/features/match_features.parquet   # P1용
ls data/features/player_stats.parquet     # 공통

# 환경 확인
python --version   # 3.9+ 권장
pip list | grep -E "lightgbm|xgboost|optuna|sklearn"
```

---

## Step 1 — P7 v5.1 성장 곡선 학습

**예상 소요 시간:** 20~40분 (Optuna 35 trials × LGBM)

```bash
cd /path/to/EPL-scouting-report-dashboard-project

# 학습 실행
python models/p7_growth_curve/train_v5_1.py

# 결과 확인 (목표: test R² ≥ 0.62, MAE ≤ 1.05)
python -c "
import json
with open('models/p7_growth_curve/results_summary.json') as f:
    d = json.load(f)
print('R²:', d.get('metrics', {}).get('r2'))
print('MAE:', d.get('metrics', {}).get('mae'))
print('목표 달성:', d.get('goal_met'))
"
```

**성공 기준:**
- `metrics.r2 >= 0.62` — 목표치
- `metrics.mae <= 1.05`
- `goal_met: true`

**실패 시 대응:**
- R² 0.60 미만 → lag4 피처 수 재검토 (현재 10개)
- val-test 갭 0.08 이상 → regularization 강화 (min_child_samples 증가)

---

## Step 2 — P1 경기 결과 재학습

**예상 소요 시간:** 15~25분 (XGBoost + LightGBM 비교)

```bash
# P7 학습 완료 확인 후 실행
python models/p1_match_result/train_pipeline.py

# 결과 확인 (목표: XGBoost test Acc ≥ 57%)
python -c "
import json
with open('models/p1_match_result/results_summary.json') as f:
    d = json.load(f)
xgb = d.get('models', {}).get('xgboost', {}).get('test', {})
print('XGBoost Acc:', xgb.get('accuracy'))
print('XGBoost F1:', xgb.get('f1_macro'))
"
```

**성공 기준:**
- `models.xgboost.test.accuracy >= 0.57`
- `models.xgboost.test.f1_macro >= 0.42`

**실패 시 대응:**
- Acc 55% 미만 → elo_trend_3 피처 계산 재확인
- RobustScaler 적용 여부 체크

---

## Step 3 — 결과 커밋 및 푸시

```bash
# 결과 확인 후 커밋
git add models/p7_growth_curve/results_summary.json
git add models/p1_match_result/results_summary.json

# 학습 스크립트도 변경됐으면 포함
git add models/p7_growth_curve/train_v5_1.py
git add models/p1_match_result/train_pipeline.py

git commit -m "model: P7 v5.1 R²=X.XX + P1 Acc=XX.XX% 로컬 학습 결과"
git push origin master
```

**커밋 메시지 형식:**
```
model: P7 v5.1 R²=0.62+ + P1 Acc=57%+ 로컬 학습 완료
```

---

## 실행 순서 요약

```
[오늘 저녁]
1. 전제 조건 확인 (5분)
2. P7 train_v5_1.py 실행 → 결과 확인 (20~40분)
3. P1 train_pipeline.py 실행 → 결과 확인 (15~25분)
4. git commit + push (5분)

[내일 아침 09:00 KST 미팅 전]
5. push 완료 상태 확인
6. results_summary.json 수치 미팅 노트 반영
```

---

## 참고 — 모델 파일 경로

| 모델 | 학습 스크립트 | 결과 파일 | 데이터 |
|------|------------|---------|--------|
| P7 v5.1 | `models/p7_growth_curve/train_v5_1.py` | `models/p7_growth_curve/results_summary.json` | `data/features/scout_features.parquet` |
| P1 | `models/p1_match_result/train_pipeline.py` | `models/p1_match_result/results_summary.json` | `data/features/match_features.parquet` |

---

*작성: Marcus Webb (Analytics Agent) | 2026-05-19*

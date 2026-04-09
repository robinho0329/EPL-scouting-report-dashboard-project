"""
S4 성장 예측 → 레퍼런스 시각화 도구 (age_performance_profiles.py)
═══════════════════════════════════════════════════════════════════
S4 모델 정확도가 낮아 예측 도구에서 참조 시각화 도구로 전환.
스카우트 "김태현"이 실무에서 활용하는 연령-성능 기준선 제공.

생성하는 시각화 (5종):
  1. 포지션별 피크 연령 곡선 (LOWESS + 문헌 기준선 비교)
  2. 연령-이적료 곡선 (포지션별)
  3. 전설적 선수 커리어 아크 갤러리 (6명)
  4. 현재 팀 연령 프로파일 (2024/25)
  5. 포지션별 연령 코호트 백분위수 벤치마크

저장:
  models/s4_growth/figures/ref_*.png
  data/scout/s4_reference_profiles.json
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings('ignore')

# ─────────────────────────── 경로 설정 ───────────────────────────
BASE_DIR  = Path("C:/Users/xcv54/workspace/EPL project")
DATA_DIR  = BASE_DIR / "data" / "processed"
SCOUT_DIR = BASE_DIR / "data" / "scout"
FIG_DIR   = BASE_DIR / "models" / "s4_growth" / "figures"

SCOUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── 폰트 / 색상 설정 ───────────────────────────
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

# EPL 공식 색상 팔레트
EPL_PURPLE  = '#3D195B'
EPL_CYAN    = '#04F5FF'
EPL_PINK    = '#E90052'
EPL_GREEN   = '#00FF87'

# 포지션별 색상
POS_COLORS = {
    'FW': EPL_PINK,
    'MF': EPL_CYAN,
    'DF': EPL_GREEN,
    'GK': '#FFD700',
}

# 배경 스타일
BG_COLOR  = '#1a1a2e'
GRID_COLOR = '#2d2d4e'
TEXT_COLOR = '#e8e8e8'

POSITIONS = ['FW', 'MF', 'DF', 'GK']
POS_KOR   = {'FW': '공격수', 'MF': '미드필더', 'DF': '수비수', 'GK': '골키퍼'}

# 문헌 기반 피크 연령 (스포츠 과학 메타분석)
PEAK_AGE_PRIORS = {'FW': 27, 'MF': 27, 'DF': 28, 'GK': 30}

# 이 분석의 EPL 데이터 기반 피크 연령 (train_v4.py에서 검증된 값)
PEAK_AGE_DATA = {'FW': 25, 'MF': 27, 'DF': 26, 'GK': 28}

# ═══════════════════════════════════════════════════════════════
# 유틸리티 함수
# ═══════════════════════════════════════════════════════════════

def season_to_year(s):
    try:
        return int(str(s).split('/')[0])
    except:
        return np.nan


def simplify_position(pos):
    if pd.isna(pos):
        return 'MF'
    pos = str(pos).upper()
    if 'GK' in pos:
        return 'GK'
    if 'FW' in pos:
        return 'FW'
    if 'DF' in pos:
        return 'DF'
    return 'MF'


def lowess_smooth(x, y, frac=0.35):
    """수동 LOWESS: tricube 커널 가중 국소 선형 회귀."""
    n = len(x)
    if n < 4:
        return y.copy()
    yhat = np.zeros(n)
    r = np.abs(x - x[:, None])  # (n, n) 거리 행렬
    bandwidth = max(1, int(np.ceil(frac * n)) - 1)
    h = np.sort(r, axis=1)[:, bandwidth]
    for i in range(n):
        wi = np.clip(1 - (r[i] / (h[i] + 1e-10)) ** 3, 0, None) ** 3
        W  = np.diag(wi)
        X_ = np.column_stack([np.ones(n), x])
        try:
            beta = np.linalg.lstsq(X_.T @ W @ X_, X_.T @ W @ y, rcond=None)[0]
            yhat[i] = beta[0] + beta[1] * x[i]
        except np.linalg.LinAlgError:
            yhat[i] = y[i]
    return yhat


def set_dark_style(ax, title='', xlabel='', ylabel=''):
    """다크 테마 적용."""
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.spines[:].set_color(GRID_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.grid(True, color=GRID_COLOR, alpha=0.5, linestyle='--', linewidth=0.7)
    if title:
        ax.set_title(title, color=TEXT_COLOR, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def compute_composite_score(row):
    """포지션별 가중 복합 성능 점수 계산."""
    pos = row['pos_simple']
    if pos == 'FW':
        return (row.get('gls_p90', 0) * 3.0 + row.get('ast_p90', 0) * 2.0 +
                row.get('sh_p90', 0) * 0.3  + row.get('fld_p90', 0) * 0.5)
    elif pos == 'MF':
        return (row.get('gls_p90', 0) * 2.0 + row.get('ast_p90', 0) * 2.5 +
                row.get('tklw_p90', 0) * 0.8 + row.get('int_p90', 0) * 0.8 +
                row.get('crs_p90', 0) * 0.5  + row.get('fld_p90', 0) * 0.4)
    elif pos == 'DF':
        return (row.get('tklw_p90', 0) * 3.0 + row.get('int_p90', 0) * 3.0 +
                row.get('gls_p90', 0) * 1.0  + row.get('ast_p90', 0) * 0.8 +
                row.get('fld_p90', 0) * 0.3)
    else:  # GK
        return row.get('90s_safe', 1) * 0.1


# ═══════════════════════════════════════════════════════════════
# 1. 데이터 로딩 및 피처 준비
# ═══════════════════════════════════════════════════════════════
print("=" * 65)
print("[1] 데이터 로딩 중...")
print("=" * 65)

season_df = pd.read_parquet(DATA_DIR / "player_season_stats.parquet")
match_df  = pd.read_parquet(DATA_DIR / "player_match_logs.parquet")

print(f"  시즌 통계: {season_df.shape}")
print(f"  경기 로그: {match_df.shape}")

season_df['season_year'] = season_df['season'].apply(season_to_year)
season_df['pos_simple']  = season_df['pos'].apply(simplify_position)

# 경기 로그 집계
match_detail = match_df[match_df['detail_stats_available'] == True].copy()
match_detail['season_year'] = match_detail['season'].apply(season_to_year)

match_agg = (
    match_detail.groupby(['player', 'season_year'])
    .agg(
        tklw_total=('tklw', lambda x: x.fillna(0).sum()),
        int_total=('int',   lambda x: x.fillna(0).sum()),
        sh_total=('sh',    lambda x: x.fillna(0).sum()),
        sot_total=('sot',  lambda x: x.fillna(0).sum()),
        crs_total=('crs',  lambda x: x.fillna(0).sum()),
        fld_total=('fld',  lambda x: x.fillna(0).sum()),
    )
    .reset_index()
)

# 최소 출전 시간 필터 (450분 이상 — 더 많은 데이터 포함)
df_filtered = season_df[season_df['min'].fillna(0) >= 450].copy()
df_filtered = df_filtered.merge(match_agg, on=['player', 'season_year'], how='left')
df_filtered['90s_safe'] = df_filtered['90s'].fillna(1).clip(lower=0.1)

for col_src, col_dst in [('gls', 'gls_p90'), ('ast', 'ast_p90')]:
    df_filtered[col_dst] = df_filtered[col_src].fillna(0) / df_filtered['90s_safe']

for col_src, col_dst in [('tklw_total', 'tklw_p90'), ('int_total', 'int_p90'),
                          ('sh_total', 'sh_p90'), ('crs_total', 'crs_p90'),
                          ('fld_total', 'fld_p90')]:
    df_filtered[col_dst] = df_filtered[col_src].fillna(0) / df_filtered['90s_safe']

df_filtered['raw_score'] = df_filtered.apply(compute_composite_score, axis=1)

# 포지션×시즌 내 Z-점수 표준화
def zscore_within_group(df, col, group_cols):
    result = pd.Series(np.nan, index=df.index)
    for _, idx in df.groupby(group_cols).groups.items():
        vals = df.loc[idx, col]
        if len(vals) >= 3:
            mu, sigma = vals.mean(), vals.std()
            result.loc[idx] = (vals - mu) / sigma if sigma > 0 else 0.0
        else:
            result.loc[idx] = 0.0
    return result

df_filtered['perf_z'] = zscore_within_group(
    df_filtered, 'raw_score', ['pos_simple', 'season_year']
)

# 연령 유효값만 (16-40세)
df_valid = df_filtered[(df_filtered['age'] >= 16) & (df_filtered['age'] <= 40)].copy()
print(f"  유효 데이터: {df_valid.shape[0]} 행")

# 2024/25 시즌 데이터
df_2425 = df_valid[df_valid['season'] == '2024/25'].copy()
print(f"  2024/25 시즌: {df_2425.shape[0]} 선수")


# ═══════════════════════════════════════════════════════════════
# 2. 연령별 통계 계산 (모든 시각화에 공통 사용)
# ═══════════════════════════════════════════════════════════════
print("\n[2] 연령별 기준선 통계 계산 중...")

age_stats_by_pos = {}
peak_age_results = {}
percentile_data  = {}

for pos in POSITIONS:
    pos_data = df_valid[df_valid['pos_simple'] == pos].copy()

    # 연령별 통계
    age_stats = (
        pos_data.groupby('age')['perf_z']
        .agg(['mean', 'std', 'count',
              lambda x: np.percentile(x, 10),
              lambda x: np.percentile(x, 25),
              lambda x: np.percentile(x, 50),
              lambda x: np.percentile(x, 75),
              lambda x: np.percentile(x, 90)])
        .reset_index()
    )
    age_stats.columns = ['age', 'mean', 'std', 'count', 'p10', 'p25', 'p50', 'p75', 'p90']
    age_stats = age_stats.sort_values('age')
    age_valid = age_stats[age_stats['count'] >= 20].copy()

    age_stats_by_pos[pos] = age_valid

    # LOWESS 피크 연령 추출
    if len(age_valid) >= 4:
        ages   = age_valid['age'].values.astype(float)
        means  = age_valid['mean'].values
        smooth = lowess_smooth(ages, means, frac=0.4)
        peak_idx = np.argmax(smooth)
        peak_age = int(ages[peak_idx])
    else:
        peak_age = PEAK_AGE_DATA[pos]
        ages   = age_valid['age'].values.astype(float)
        means  = age_valid['mean'].values
        smooth = means.copy()

    peak_age_results[pos] = {
        'peak_age': peak_age,
        'peak_age_literature': PEAK_AGE_PRIORS[pos],
        'data_peak_age': PEAK_AGE_DATA[pos],
        'ages': ages.tolist(),
        'smooth': smooth.tolist(),
        'means': means.tolist(),
        'p25': age_valid['p25'].tolist(),
        'p75': age_valid['p75'].tolist(),
    }

    # 백분위수 데이터 (per-90 스탯)
    pct_records = []
    for age_val in range(16, 38):
        age_data = pos_data[(pos_data['age'] >= age_val - 0.5) &
                            (pos_data['age'] < age_val + 0.5)]
        if len(age_data) >= 10:
            rec = {'age': age_val}
            for col in ['gls_p90', 'ast_p90', 'tklw_p90', 'int_p90', 'sh_p90']:
                vals = age_data[col].dropna()
                if len(vals) >= 5:
                    for p in [25, 50, 75, 90]:
                        rec[f'{col}_p{p}'] = float(np.percentile(vals, p))
            pct_records.append(rec)
    percentile_data[pos] = pct_records

    print(f"  {pos} ({POS_KOR[pos]}): 유효 연령 {len(age_valid)}개, "
          f"EPL 피크={peak_age}세, 문헌 피크={PEAK_AGE_PRIORS[pos]}세")


# ═══════════════════════════════════════════════════════════════
# 시각화 1: 포지션별 피크 연령 곡선
# ═══════════════════════════════════════════════════════════════
print("\n[VIZ 1] 포지션별 피크 연령 곡선 생성 중...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor(BG_COLOR)
fig.suptitle('포지션별 피크 연령 곡선\nEPL 데이터 기반 (2000/01 ~ 2024/25)',
             color=TEXT_COLOR, fontsize=16, fontweight='bold', y=1.01)

for idx, pos in enumerate(POSITIONS):
    ax  = axes[idx // 2][idx % 2]
    ax.set_facecolor(BG_COLOR)
    res = peak_age_results[pos]
    av  = age_stats_by_pos[pos]

    ages   = np.array(res['ages'])
    smooth = np.array(res['smooth'])
    means  = np.array(res['means'])
    p25    = np.array(res['p25'])
    p75    = np.array(res['p75'])
    color  = POS_COLORS[pos]

    if len(ages) < 2:
        ax.text(0.5, 0.5, '데이터 부족', ha='center', va='center',
                transform=ax.transAxes, color=TEXT_COLOR)
        set_dark_style(ax, f'{POS_KOR[pos]} ({pos})')
        continue

    # 90% 성능 구간 음영 (p25~p75)
    ax.fill_between(ages, p25, p75, alpha=0.2, color=color, label='25~75 백분위수')

    # 실제 평균 점 (반투명)
    ax.scatter(ages, means, color=color, alpha=0.4, s=20, zorder=3)

    # LOWESS 곡선
    ax.plot(ages, smooth, color=color, linewidth=2.5, label='LOWESS 평활 곡선', zorder=4)

    # EPL 데이터 피크 표시
    peak_age   = res['peak_age']
    peak_z     = smooth[np.argmin(np.abs(ages - peak_age))] if len(ages) > 0 else 0
    ax.axvline(x=peak_age, color=color, linestyle='-', alpha=0.8, linewidth=1.5)
    ax.annotate(f'EPL 피크\n{peak_age}세',
                xy=(peak_age, peak_z),
                xytext=(peak_age + 1.2, peak_z + 0.15),
                color=color, fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    # 문헌 기준선 피크
    lit_age = res['peak_age_literature']
    ax.axvline(x=lit_age, color='white', linestyle='--', alpha=0.5, linewidth=1.2,
               label=f'문헌 기준 ({lit_age}세)')
    ax.text(lit_age + 0.3, ax.get_ylim()[0] + 0.05 if ax.get_ylim()[0] != 0 else -0.3,
            f'문헌\n{lit_age}세', color='white', fontsize=8, alpha=0.7)

    # 연령 구간 레이블 (성장/전성기/하락)
    ymin, ymax = smooth.min() - 0.2, smooth.max() + 0.3
    ax.axvspan(16, 22, alpha=0.05, color=EPL_GREEN, label='성장기')
    ax.axvspan(22, peak_age + 2, alpha=0.05, color=EPL_CYAN, label='전성기')
    ax.axvspan(peak_age + 2, 38, alpha=0.05, color=EPL_PINK, label='하락기')

    set_dark_style(ax, f'{POS_KOR[pos]} ({pos})', '연령 (세)', '성능 Z-점수')
    ax.set_xlim(16, 38)
    ax.legend(loc='upper left', fontsize=8, facecolor='#2d2d4e',
              labelcolor=TEXT_COLOR, framealpha=0.8)

    # 샘플 수 표시
    sample_text = f'n = {int(av["count"].sum()):,} 선수-시즌'
    ax.text(0.98, 0.05, sample_text, transform=ax.transAxes,
            ha='right', va='bottom', color=TEXT_COLOR, fontsize=8, alpha=0.7)

plt.tight_layout(pad=3.0)
out1 = FIG_DIR / 'ref_01_peak_age_curves.png'
plt.savefig(out1, bbox_inches='tight', facecolor=BG_COLOR, dpi=150)
plt.close()
print(f"  저장: {out1}")


# ═══════════════════════════════════════════════════════════════
# 시각화 2: 연령-이적료 곡선
# ═══════════════════════════════════════════════════════════════
print("\n[VIZ 2] 연령-이적료 곡선 생성 중...")

# 이적료 데이터 준비 (1500만 유로 이상 선수만 의미 있는 곡선)
df_val = df_valid[(df_valid['market_value'].notna()) &
                  (df_valid['market_value'] > 0) &
                  (df_valid['age'] >= 16) &
                  (df_valid['age'] <= 36)].copy()
df_val['market_value_m'] = df_val['market_value'] / 1_000_000  # 백만 유로 단위

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor(BG_COLOR)
fig.suptitle('포지션별 연령-이적료 곡선\n(중앙값 ± IQR)',
             color=TEXT_COLOR, fontsize=16, fontweight='bold', y=1.01)

val_peak_ages = {}

for idx, pos in enumerate(POSITIONS):
    ax = axes[idx // 2][idx % 2]
    ax.set_facecolor(BG_COLOR)
    pos_val = df_val[df_val['pos_simple'] == pos].copy()
    color   = POS_COLORS[pos]

    age_val_stats = (
        pos_val.groupby('age')['market_value_m']
        .agg(['median', 'mean', 'count',
              lambda x: np.percentile(x, 25),
              lambda x: np.percentile(x, 75)])
        .reset_index()
    )
    age_val_stats.columns = ['age', 'median', 'mean', 'count', 'p25', 'p75']
    age_val_stats = age_val_stats[age_val_stats['count'] >= 10].sort_values('age')

    if len(age_val_stats) < 3:
        ax.text(0.5, 0.5, '데이터 부족', ha='center', va='center',
                transform=ax.transAxes, color=TEXT_COLOR)
        set_dark_style(ax, f'{POS_KOR[pos]} ({pos})')
        val_peak_ages[pos] = None
        continue

    ages_v  = age_val_stats['age'].values.astype(float)
    medians = age_val_stats['median'].values
    p25_v   = age_val_stats['p25'].values
    p75_v   = age_val_stats['p75'].values

    # LOWESS 적용
    if len(ages_v) >= 4:
        smooth_v = lowess_smooth(ages_v, medians, frac=0.4)
    else:
        smooth_v = medians.copy()

    # 이적료 피크 연령
    val_peak_idx = np.argmax(smooth_v)
    val_peak_age = int(ages_v[val_peak_idx])
    val_peak_ages[pos] = val_peak_age

    # IQR 음영
    ax.fill_between(ages_v, p25_v, p75_v, alpha=0.25, color=color, label='25~75 백분위수')

    # 중앙값 산포
    ax.scatter(ages_v, medians, color=color, alpha=0.4, s=25, zorder=3)

    # LOWESS 곡선
    ax.plot(ages_v, smooth_v, color=color, linewidth=2.5, label='LOWESS 평활', zorder=4)

    # 이적료 피크 표시
    peak_val_y = smooth_v[val_peak_idx]
    ax.axvline(x=val_peak_age, color=color, linestyle='-', alpha=0.8, linewidth=1.5)
    ax.annotate(f'가치 피크\n{val_peak_age}세\n({peak_val_y:.0f}M€)',
                xy=(val_peak_age, peak_val_y),
                xytext=(val_peak_age + 1.5, peak_val_y * 0.85),
                color=color, fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    # 성능 피크 비교선
    perf_peak = PEAK_AGE_DATA[pos]
    ax.axvline(x=perf_peak, color='white', linestyle=':', alpha=0.6, linewidth=1.5,
               label=f'성능 피크 ({perf_peak}세)')

    # 피크 차이 주석
    diff = val_peak_age - perf_peak
    if diff != 0:
        mid_x = (val_peak_age + perf_peak) / 2
        y_pos = smooth_v.max() * 0.1
        ax.annotate('',
                    xy=(val_peak_age, y_pos),
                    xytext=(perf_peak, y_pos),
                    arrowprops=dict(arrowstyle='<->', color='yellow', lw=1.5))
        ax.text(mid_x, y_pos + 0.5, f'Δ{abs(diff)}년', ha='center',
                color='yellow', fontsize=9)

    set_dark_style(ax, f'{POS_KOR[pos]} ({pos})', '연령 (세)', '이적료 (백만 유로)')
    ax.set_xlim(16, 36)
    ax.legend(loc='upper right', fontsize=8, facecolor='#2d2d4e',
              labelcolor=TEXT_COLOR, framealpha=0.8)

    sample_txt = f'n = {int(age_val_stats["count"].sum()):,}'
    ax.text(0.98, 0.05, sample_txt, transform=ax.transAxes,
            ha='right', va='bottom', color=TEXT_COLOR, fontsize=8, alpha=0.7)

plt.tight_layout(pad=3.0)
out2 = FIG_DIR / 'ref_02_age_value_curves.png'
plt.savefig(out2, bbox_inches='tight', facecolor=BG_COLOR, dpi=150)
plt.close()
print(f"  저장: {out2}")


# ═══════════════════════════════════════════════════════════════
# 시각화 3: 전설적 선수 커리어 아크 갤러리
# ═══════════════════════════════════════════════════════════════
print("\n[VIZ 3] 커리어 아크 갤러리 생성 중...")

LEGENDS = {
    'Wayne Rooney':   {'pos': 'FW', 'color': '#e74c3c', 'label': '루니 (FW)'},
    'Frank Lampard':  {'pos': 'MF', 'color': '#3498db', 'label': '램파드 (MF)'},
    'John Terry':     {'pos': 'DF', 'color': '#2ecc71', 'label': '테리 (DF)'},
    'Thierry Henry':  {'pos': 'FW', 'color': '#e67e22', 'label': '앙리 (FW)'},
    'Alan Shearer':   {'pos': 'FW', 'color': '#1abc9c', 'label': '시어러 (FW)'},
    'Steven Gerrard': {'pos': 'MF', 'color': '#9b59b6', 'label': '제라드 (MF)'},
}

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.patch.set_facecolor(BG_COLOR)
fig.suptitle('EPL 레전드 커리어 아크\n(EPL 데이터 + 포지션 평균 오버레이)',
             color=TEXT_COLOR, fontsize=16, fontweight='bold', y=1.01)

for idx, (player_name, pinfo) in enumerate(LEGENDS.items()):
    ax    = axes[idx // 3][idx % 3]
    ax.set_facecolor(BG_COLOR)
    pos   = pinfo['pos']
    color = pinfo['color']
    label = pinfo['label']

    # 해당 선수 데이터
    player_data = df_valid[df_valid['player'] == player_name].sort_values('age').copy()

    if player_data.empty:
        ax.text(0.5, 0.5, '데이터 없음', ha='center', va='center',
                transform=ax.transAxes, color=TEXT_COLOR, fontsize=12)
        set_dark_style(ax, label)
        continue

    # 포지션 평균 곡선 (배경)
    pos_avg = age_stats_by_pos[pos]
    if len(pos_avg) >= 2:
        pos_ages   = pos_avg['age'].values.astype(float)
        pos_means  = pos_avg['mean'].values
        if len(pos_ages) >= 4:
            pos_smooth = lowess_smooth(pos_ages, pos_means, frac=0.4)
        else:
            pos_smooth = pos_means
        ax.plot(pos_ages, pos_smooth, color='white', linewidth=1.5,
                alpha=0.35, linestyle='--', label=f'{POS_KOR[pos]} 평균')
        ax.fill_between(pos_ages,
                        pos_avg['p25'].values,
                        pos_avg['p75'].values,
                        alpha=0.08, color='white')

    # 선수 커리어 곡선
    player_ages = player_data['age'].values.astype(float)
    player_perf = player_data['perf_z'].values

    # 가우시안 평활 (선수 데이터 충분할 때)
    if len(player_ages) >= 4:
        sort_idx     = np.argsort(player_ages)
        player_ages  = player_ages[sort_idx]
        player_perf  = player_perf[sort_idx]
        player_smooth = gaussian_filter1d(player_perf, sigma=0.8)
        ax.plot(player_ages, player_smooth, color=color, linewidth=2.5,
                zorder=4, label=label)
        ax.scatter(player_ages, player_perf, color=color, s=40, zorder=5, alpha=0.7)
    else:
        ax.plot(player_ages, player_perf, color=color, linewidth=2.5,
                marker='o', markersize=6, zorder=4, label=label)

    # 커리어 피크 표시
    if len(player_ages) > 0:
        peak_idx_p  = np.argmax(player_perf)
        peak_age_p  = player_ages[peak_idx_p]
        peak_perf_p = player_perf[peak_idx_p]
        ax.annotate(f'피크\n{peak_age_p:.0f}세',
                    xy=(peak_age_p, peak_perf_p),
                    xytext=(peak_age_p + 1, peak_perf_p + 0.3),
                    color=color, fontsize=8.5, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.0))

        # 하락 시작점 (피크 이후 연속 하락)
        if peak_idx_p < len(player_perf) - 2:
            decline_candidates = np.where(
                (np.arange(len(player_perf)) > peak_idx_p) &
                (player_perf < peak_perf_p - 0.5)
            )[0]
            if len(decline_candidates) > 0:
                decline_idx = decline_candidates[0]
                ax.axvline(x=player_ages[decline_idx], color=color,
                           linestyle=':', alpha=0.5, linewidth=1.2)
                ax.text(player_ages[decline_idx] + 0.2,
                        ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -1.5,
                        '하락\n시작', color=color, fontsize=7.5, alpha=0.8)

    # 포지션 피크 수직선
    data_peak = PEAK_AGE_DATA[pos]
    ax.axvline(x=data_peak, color=POS_COLORS[pos], linestyle='--',
               alpha=0.4, linewidth=1.0, label=f'포지션 피크 ({data_peak}세)')

    set_dark_style(ax, label, '연령 (세)', '성능 Z-점수')
    ax.legend(loc='upper left', fontsize=7.5, facecolor='#2d2d4e',
              labelcolor=TEXT_COLOR, framealpha=0.8)

    # 커리어 요약 텍스트
    n_seasons = len(player_data)
    age_range = f'{player_ages.min():.0f}-{player_ages.max():.0f}세'
    ax.text(0.98, 0.05, f'EPL {n_seasons}시즌 | {age_range}',
            transform=ax.transAxes, ha='right', va='bottom',
            color=TEXT_COLOR, fontsize=8, alpha=0.7)

plt.tight_layout(pad=3.0)
out3 = FIG_DIR / 'ref_03_legend_career_arcs.png'
plt.savefig(out3, bbox_inches='tight', facecolor=BG_COLOR, dpi=150)
plt.close()
print(f"  저장: {out3}")


# ═══════════════════════════════════════════════════════════════
# 시각화 4: 현재 팀 연령 프로파일 (2024/25)
# ═══════════════════════════════════════════════════════════════
print("\n[VIZ 4] 팀 연령 프로파일 생성 중...")

teams_2425 = sorted(df_2425['team'].unique())
n_teams    = len(teams_2425)

# 각 팀별 신호 계산 (매수/매도 신호)
def get_team_signals(team_name, df_season, peak_ages_dict):
    """팀 내 선수별 매수/매도 신호 판단."""
    team_data = df_season[df_season['team'] == team_name].copy()
    signals   = []
    for _, row in team_data.iterrows():
        pos        = row['pos_simple']
        age        = row['age']
        peak_age   = peak_ages_dict.get(pos, 26)
        mv         = row.get('market_value', 0) or 0
        perf       = row.get('perf_z', 0) or 0
        age_diff   = age - peak_age

        # 매수 신호: 24세 이하, 성능 평균 이상
        buy  = age <= 24 and perf >= 0
        # 매도 신호: 피크 +3년 이상, 이적료 100만 유로 이상
        sell = age_diff >= 3 and mv >= 1_000_000

        signals.append({
            'player': row['player'],
            'pos': pos,
            'age': age,
            'age_diff': age_diff,
            'market_value': mv,
            'perf_z': perf,
            'buy_signal': buy,
            'sell_signal': sell,
        })
    return pd.DataFrame(signals)

# 20개 팀을 4×5 그리드로 배치
ncols = 4
nrows = 5
fig, axes = plt.subplots(nrows, ncols, figsize=(20, 20))
fig.patch.set_facecolor(BG_COLOR)
fig.suptitle('2024/25 시즌 팀별 연령 프로파일\n(녹색=매수 신호 / 빨간색=매도 신호)',
             color=TEXT_COLOR, fontsize=16, fontweight='bold', y=1.01)

squad_profiles_json = {}

for t_idx, team_name in enumerate(teams_2425):
    ax     = axes[t_idx // ncols][t_idx % ncols]
    ax.set_facecolor(BG_COLOR)

    sigs   = get_team_signals(team_name, df_2425, PEAK_AGE_DATA)
    if sigs.empty:
        ax.text(0.5, 0.5, '데이터 없음', ha='center', va='center',
                transform=ax.transAxes, color=TEXT_COLOR)
        set_dark_style(ax, team_name)
        continue

    # 매수/매도 신호 선수 분류
    buy_players  = sigs[sigs['buy_signal']]
    sell_players = sigs[sigs['sell_signal']]
    neutral      = sigs[~sigs['buy_signal'] & ~sigs['sell_signal']]

    # 산점도: x=age, y=perf_z, 색상=신호
    ax.scatter(neutral['age'], neutral['perf_z'],
               color=EPL_CYAN, alpha=0.6, s=40, label='중립', zorder=3)
    ax.scatter(buy_players['age'], buy_players['perf_z'],
               color=EPL_GREEN, alpha=0.9, s=70, marker='^',
               label=f'매수({len(buy_players)})', zorder=4)
    ax.scatter(sell_players['age'], sell_players['perf_z'],
               color=EPL_PINK, alpha=0.9, s=70, marker='v',
               label=f'매도({len(sell_players)})', zorder=4)

    # 피크 연령 참조선 (포지션별)
    for pos, peak in PEAK_AGE_DATA.items():
        if pos in sigs['pos'].values:
            ax.axvline(x=peak, color=POS_COLORS[pos], linestyle='--',
                       alpha=0.3, linewidth=0.8)

    # y=0 기준선
    ax.axhline(y=0, color='white', linestyle='-', alpha=0.3, linewidth=0.7)

    set_dark_style(ax, team_name, '연령', 'Z-점수')
    ax.set_xlim(15, 38)
    ax.legend(loc='upper right', fontsize=6.5, facecolor='#1a1a2e',
              labelcolor=TEXT_COLOR, framealpha=0.8)

    # JSON 저장용 데이터
    squad_profiles_json[team_name] = {
        'n_players': len(sigs),
        'avg_age': float(sigs['age'].mean()),
        'buy_signals': buy_players[['player', 'pos', 'age', 'perf_z', 'market_value']].to_dict('records'),
        'sell_signals': sell_players[['player', 'pos', 'age', 'perf_z', 'market_value']].to_dict('records'),
    }

# 빈 서브플롯 숨기기
for t_idx in range(len(teams_2425), nrows * ncols):
    axes[t_idx // ncols][t_idx % ncols].set_visible(False)

plt.tight_layout(pad=2.5)
out4 = FIG_DIR / 'ref_04_squad_age_profiles.png'
plt.savefig(out4, bbox_inches='tight', facecolor=BG_COLOR, dpi=150)
plt.close()
print(f"  저장: {out4}")


# ═══════════════════════════════════════════════════════════════
# 시각화 5: 포지션별 연령 코호트 백분위수 벤치마크
# ═══════════════════════════════════════════════════════════════
print("\n[VIZ 5] 연령 코호트 백분위수 벤치마크 생성 중...")

# 각 포지션의 주요 스탯 per-90 백분위수 밴드
STAT_CONFIG = {
    'FW': [
        ('gls_p90', '골/90분', EPL_PINK),
        ('ast_p90', '어시스트/90분', EPL_CYAN),
        ('sh_p90',  '슈팅/90분', EPL_GREEN),
    ],
    'MF': [
        ('gls_p90',  '골/90분', EPL_PINK),
        ('ast_p90',  '어시스트/90분', EPL_CYAN),
        ('tklw_p90', '성공 태클/90분', EPL_GREEN),
    ],
    'DF': [
        ('tklw_p90', '성공 태클/90분', EPL_GREEN),
        ('int_p90',  '인터셉트/90분', EPL_CYAN),
        ('gls_p90',  '골/90분', EPL_PINK),
    ],
    'GK': [
        ('gls_p90',  '실점/90분', EPL_PINK),
        ('ast_p90',  '어시스트/90분', EPL_CYAN),
        ('tklw_p90', '태클/90분', EPL_GREEN),
    ],
}

fig, big_axes = plt.subplots(4, 3, figsize=(18, 20))
fig.patch.set_facecolor(BG_COLOR)
fig.suptitle('포지션별 연령 코호트 백분위수 벤치마크\n(선수 X의 해당 연령 기준 순위 파악)',
             color=TEXT_COLOR, fontsize=16, fontweight='bold', y=1.01)

for p_idx, pos in enumerate(POSITIONS):
    stats_cfg = STAT_CONFIG[pos]
    pos_data  = df_valid[(df_valid['pos_simple'] == pos) &
                         (df_valid['age'] >= 16) &
                         (df_valid['age'] <= 36)].copy()
    color_pos = POS_COLORS[pos]

    for s_idx, (stat_col, stat_label, stat_color) in enumerate(stats_cfg):
        ax = big_axes[p_idx][s_idx]
        ax.set_facecolor(BG_COLOR)

        if stat_col not in pos_data.columns:
            set_dark_style(ax, f'{POS_KOR[pos]}: {stat_label}')
            continue

        age_pct = []
        ages_range = range(17, 37)
        for age_val in ages_range:
            age_cohort = pos_data[
                (pos_data['age'] >= age_val - 0.5) &
                (pos_data['age'] < age_val + 0.5)
            ][stat_col].dropna()

            if len(age_cohort) >= 8:
                age_pct.append({
                    'age': age_val,
                    'p25': float(np.percentile(age_cohort, 25)),
                    'p50': float(np.percentile(age_cohort, 50)),
                    'p75': float(np.percentile(age_cohort, 75)),
                    'p90': float(np.percentile(age_cohort, 90)),
                    'n': len(age_cohort),
                })

        if not age_pct:
            set_dark_style(ax, f'{POS_KOR[pos]}: {stat_label}')
            continue

        pct_df = pd.DataFrame(age_pct)
        ages_a = pct_df['age'].values

        # 백분위수 밴드 그리기
        ax.fill_between(ages_a, pct_df['p25'], pct_df['p75'],
                        alpha=0.25, color=stat_color, label='25~75 백분위수')
        ax.fill_between(ages_a, pct_df['p75'], pct_df['p90'],
                        alpha=0.12, color=stat_color, label='75~90 백분위수')
        ax.fill_between(ages_a, 0, pct_df['p25'],
                        alpha=0.08, color='white', label='0~25 백분위수')

        # 중앙값 선
        ax.plot(ages_a, pct_df['p50'], color=stat_color, linewidth=2.0,
                label='중앙값 (50th)', zorder=4)
        # 90백분위수 선
        ax.plot(ages_a, pct_df['p90'], color=stat_color, linewidth=1.2,
                linestyle='--', alpha=0.7, label='90 백분위수', zorder=4)

        # 피크 연령 수직선
        peak_age_p = PEAK_AGE_DATA[pos]
        ax.axvline(x=peak_age_p, color='white', linestyle=':', alpha=0.5,
                   linewidth=1.2, label=f'피크 ({peak_age_p}세)')

        # 백분위수 레이블
        if len(ages_a) > 0:
            last_idx = -1
            for pct_name, pct_key, pct_y_off in [('90th', 'p90', 0.01),
                                                   ('50th', 'p50', 0.01)]:
                y_val = pct_df[pct_key].values[last_idx]
                ax.text(ages_a[last_idx] + 0.3, y_val, pct_name,
                        color=stat_color, fontsize=7.5, va='center', alpha=0.9)

        title_str = f'{POS_KOR[pos]}: {stat_label}'
        set_dark_style(ax, title_str, '연령 (세)', f'{stat_label}')
        ax.set_xlim(17, 37)
        ax.legend(loc='upper right', fontsize=6.5, facecolor='#1a1a2e',
                  labelcolor=TEXT_COLOR, framealpha=0.8)

plt.tight_layout(pad=2.5)
out5 = FIG_DIR / 'ref_05_cohort_percentile_bands.png'
plt.savefig(out5, bbox_inches='tight', facecolor=BG_COLOR, dpi=150)
plt.close()
print(f"  저장: {out5}")


# ═══════════════════════════════════════════════════════════════
# JSON 요약 저장
# ═══════════════════════════════════════════════════════════════
print("\n[JSON] 요약 데이터 저장 중...")

# 이적료 피크 연령
val_peak_clean = {}
for pos in POSITIONS:
    vp = val_peak_ages.get(pos)
    val_peak_clean[pos] = int(vp) if vp is not None else PEAK_AGE_DATA[pos]

# 백분위수 데이터 (JSON 직렬화 가능한 형태)
pct_json = {}
for pos in POSITIONS:
    pct_json[pos] = [
        {k: (float(v) if isinstance(v, (np.floating, float)) else
             int(v) if isinstance(v, (np.integer, int)) else v)
         for k, v in rec.items()}
        for rec in percentile_data[pos]
    ]

# 피크 연령 비교 테이블
peak_comparison = {}
for pos in POSITIONS:
    peak_comparison[pos] = {
        'epl_data_peak': PEAK_AGE_DATA[pos],
        'epl_calculated_peak': peak_age_results[pos]['peak_age'],
        'literature_peak': PEAK_AGE_PRIORS[pos],
        'value_peak': val_peak_clean[pos],
    }

# 2024/25 팀 프로파일 (직렬화)
def clean_record(rec):
    cleaned = {}
    for k, v in rec.items():
        if isinstance(v, (np.floating, float)):
            cleaned[k] = float(v) if not np.isnan(v) else None
        elif isinstance(v, (np.integer, int)):
            cleaned[k] = int(v)
        else:
            cleaned[k] = v
    return cleaned

squad_profiles_clean = {}
for team, profile in squad_profiles_json.items():
    squad_profiles_clean[team] = {
        'n_players': profile['n_players'],
        'avg_age': float(profile['avg_age']),
        'buy_signals': [clean_record(r) for r in profile['buy_signals']],
        'sell_signals': [clean_record(r) for r in profile['sell_signals']],
    }

# 최종 JSON
output_json = {
    'meta': {
        'description': 'S4 성장 레퍼런스 시각화 — 예측 모델이 아닌 참조 도구',
        'generated_at': '2026-03-31',
        'data_range': '2000/01 ~ 2024/25',
        'note': '예측 정확도 낮음 (balanced_acc 0.33-0.44). 피크 연령 곡선은 참조용으로만 사용.',
    },
    'peak_ages': {
        pos: {
            'epl_data': PEAK_AGE_DATA[pos],
            'epl_calculated': peak_age_results[pos]['peak_age'],
            'literature': PEAK_AGE_PRIORS[pos],
            'value_peak': val_peak_clean[pos],
            'pos_kor': POS_KOR[pos],
        }
        for pos in POSITIONS
    },
    'peak_comparison': peak_comparison,
    'age_performance_percentiles': pct_json,
    'current_squad_profiles': squad_profiles_clean,
    'figures': {
        'ref_01_peak_age_curves': str(out1),
        'ref_02_age_value_curves': str(out2),
        'ref_03_legend_career_arcs': str(out3),
        'ref_04_squad_age_profiles': str(out4),
        'ref_05_cohort_percentile_bands': str(out5),
    },
}

json_path = SCOUT_DIR / "s4_reference_profiles.json"
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(output_json, f, ensure_ascii=False, indent=2)
print(f"  저장: {json_path}")

# ─────────────────────────── 완료 요약 ───────────────────────────
print("\n" + "=" * 65)
print("  S4 레퍼런스 시각화 완료!")
print("=" * 65)
print("\n포지션별 피크 연령 비교:")
print(f"  {'포지션':<8} {'EPL 계산':<10} {'EPL 기존':<10} {'문헌 기준':<10} {'이적료 피크':<10}")
print("-" * 50)
for pos in POSITIONS:
    calc_peak = peak_age_results[pos]['peak_age']
    data_peak = PEAK_AGE_DATA[pos]
    lit_peak  = PEAK_AGE_PRIORS[pos]
    val_peak  = val_peak_clean[pos]
    print(f"  {POS_KOR[pos]:<8} {calc_peak:<10} {data_peak:<10} {lit_peak:<10} {val_peak:<10}")

print(f"\n생성된 시각화 ({FIG_DIR}):")
for fname in ['ref_01_peak_age_curves.png', 'ref_02_age_value_curves.png',
              'ref_03_legend_career_arcs.png', 'ref_04_squad_age_profiles.png',
              'ref_05_cohort_percentile_bands.png']:
    fpath = FIG_DIR / fname
    size  = fpath.stat().st_size // 1024 if fpath.exists() else 0
    print(f"  {fname} ({size}KB)")

print(f"\nJSON 레퍼런스: {json_path}")
print("\n김태현 스카우트 메모:")
print("  - 예측 모델(S4) 정확도 부족 → 참조 시각화 도구로 전환 완료")
print("  - 피크 연령 곡선은 스카우트 의사결정 기준선으로 활용")
print("  - 팀별 매수/매도 신호는 ref_04에서 확인")
print("  - 코호트 백분위수는 ref_05에서 선수 위치 파악 가능")
print("=" * 65)

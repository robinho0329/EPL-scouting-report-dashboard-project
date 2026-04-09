"""
EPL EDA Visualization Script
Generates 30 PNG plots covering match, team, player, matchlog, value, correlation, and era analyses.
"""

import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = r'C:/Users/xcv54/workspace/EPL project'
DATA = os.path.join(BASE, 'data')
PROC = os.path.join(DATA, 'processed')
FIGS = os.path.join(BASE, 'reports', 'figures')
os.makedirs(FIGS, exist_ok=True)

# EPL colour palette
EPL = ['#3D195B', '#04F5FF', '#E90052', '#00FF87', '#38003C', '#2D2D2D']

# ── Load data ─────────────────────────────────────────────────────────────────
print("데이터 로딩 중...")

# epl_final.csv  (check both locations)
csv_path = os.path.join(BASE, 'epl_final.csv')
if not os.path.exists(csv_path):
    csv_path = os.path.join(DATA, 'epl_final.csv')
df_csv = pd.read_csv(csv_path, encoding='utf-8-sig')

df_match  = pd.read_parquet(os.path.join(PROC, 'match_results.parquet'))
df_team   = pd.read_parquet(os.path.join(PROC, 'team_season_summary.parquet'))
df_pss    = pd.read_parquet(os.path.join(PROC, 'player_season_stats.parquet'))
df_pml    = pd.read_parquet(os.path.join(PROC, 'player_match_logs.parquet'))

print(f"  epl_final.csv  : {df_csv.shape}")
print(f"  match_results  : {df_match.shape}")
print(f"  team_season    : {df_team.shape}")
print(f"  player_season  : {df_pss.shape}")
print(f"  player_match   : {df_pml.shape}")

saved = []

def save(fig, name):
    path = os.path.join(FIGS, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved.append(name)
    print(f"  저장: {name}")

# ═══════════════════════════════════════════════════════════════════════════════
# MATCH RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

# 1. 시즌별 평균 득점 추이
try:
    m = df_match.copy()
    m['total_goals'] = m['FullTimeHomeGoals'] + m['FullTimeAwayGoals']
    season_goals = m.groupby('Season')['total_goals'].mean().reset_index()
    season_goals.columns = ['Season', 'avg_goals']

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(season_goals['Season'], season_goals['avg_goals'],
            color=EPL[2], marker='o', linewidth=2.5, markersize=6)
    ax.fill_between(range(len(season_goals)), season_goals['avg_goals'],
                    alpha=0.15, color=EPL[2])
    ax.set_xticks(range(len(season_goals)))
    ax.set_xticklabels(season_goals['Season'], rotation=45, ha='right', fontsize=9)
    ax.set_title('시즌별 경기당 평균 득점 추이', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('시즌', fontsize=12)
    ax.set_ylabel('평균 득점', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_facecolor('#F8F8F8')
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    save(fig, 'match_01_goals_per_season.png')
except Exception as e:
    print(f"  [SKIP] match_01: {e}")

# 2. 홈승/무/원정승 비율 (stacked bar per season)
try:
    m = df_match.copy()
    res = m.groupby(['Season', 'FullTimeResult']).size().unstack(fill_value=0)
    for col in ['H', 'D', 'A']:
        if col not in res.columns:
            res[col] = 0
    res = res[['H', 'D', 'A']]
    totals = res.sum(axis=1)
    res_pct = res.div(totals, axis=0) * 100

    fig, ax = plt.subplots(figsize=(14, 6))
    bottoms = np.zeros(len(res_pct))
    labels = {'H': '홈 승', 'D': '무승부', 'A': '원정 승'}
    colors = [EPL[0], EPL[1], EPL[2]]
    for col, color, label in zip(['H', 'D', 'A'], colors, ['홈 승', '무승부', '원정 승']):
        ax.bar(range(len(res_pct)), res_pct[col], bottom=bottoms,
               color=color, label=label, alpha=0.9)
        bottoms += res_pct[col].values

    ax.set_xticks(range(len(res_pct)))
    ax.set_xticklabels(res_pct.index, rotation=45, ha='right', fontsize=9)
    ax.set_title('시즌별 경기 결과 분포 (홈승 / 무 / 원정승)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('시즌', fontsize=12)
    ax.set_ylabel('비율 (%)', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save(fig, 'match_02_result_distribution.png')
except Exception as e:
    print(f"  [SKIP] match_02: {e}")

# 3. 시즌별 홈 승률 추이 (annotate COVID 2020/21)
try:
    m = df_match.copy()
    home_wr = m.groupby('Season').apply(
        lambda x: (x['FullTimeResult'] == 'H').sum() / len(x) * 100
    ).reset_index()
    home_wr.columns = ['Season', 'home_win_rate']

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(home_wr['Season'], home_wr['home_win_rate'],
            color=EPL[0], marker='o', linewidth=2.5, markersize=6, label='홈 승률')
    ax.axhline(home_wr['home_win_rate'].mean(), color=EPL[2], linestyle='--',
               linewidth=1.5, alpha=0.7, label=f"평균 {home_wr['home_win_rate'].mean():.1f}%")

    # Annotate COVID season
    covid_seasons = ['2019/20', '2020/21']
    for cs in covid_seasons:
        if cs in home_wr['Season'].values:
            idx = home_wr[home_wr['Season'] == cs].index[0]
            row_pos = home_wr.index.get_loc(idx)
            yval = home_wr.loc[idx, 'home_win_rate']
            ax.annotate(f'COVID\n{cs}', xy=(row_pos, yval),
                        xytext=(row_pos + 0.5, yval + 3),
                        fontsize=8, color=EPL[2],
                        arrowprops=dict(arrowstyle='->', color=EPL[2], lw=1.2))

    seasons = home_wr['Season'].tolist()
    ax.set_xticks(range(len(seasons)))
    ax.set_xticklabels(seasons, rotation=45, ha='right', fontsize=9)
    ax.set_title('시즌별 홈 승률 추이', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('시즌', fontsize=12)
    ax.set_ylabel('홈 승률 (%)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_facecolor('#F8F8F8')
    plt.tight_layout()
    save(fig, 'match_03_home_advantage.png')
except Exception as e:
    print(f"  [SKIP] match_03: {e}")

# 4. 최다 스코어라인 Top 15
try:
    m = df_match.copy()
    m['scoreline'] = m['FullTimeHomeGoals'].astype(str) + '-' + m['FullTimeAwayGoals'].astype(str)
    top_sl = m['scoreline'].value_counts().head(15).sort_values()

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(top_sl.index, top_sl.values, color=EPL[0], alpha=0.85)
    for bar, val in zip(bars, top_sl.values):
        ax.text(val + 5, bar.get_y() + bar.get_height() / 2,
                str(val), va='center', fontsize=9)
    ax.set_title('최다 스코어라인 Top 15', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('빈도', fontsize=12)
    ax.set_ylabel('스코어라인', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save(fig, 'match_04_top_scorelines.png')
except Exception as e:
    print(f"  [SKIP] match_04: {e}")

# 5. 시즌별 카드 추이
try:
    m = df_match.copy()
    m['yellow'] = m['HomeYellowCards'] + m['AwayYellowCards']
    m['red'] = m['HomeRedCards'] + m['AwayRedCards']
    card_trend = m.groupby('Season')[['yellow', 'red']].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(card_trend['Season'], card_trend['yellow'],
            color='#FFD700', marker='o', linewidth=2.5, markersize=5, label='경기당 황색 카드')
    ax2 = ax.twinx()
    ax2.plot(card_trend['Season'], card_trend['red'],
             color=EPL[2], marker='s', linewidth=2.5, markersize=5, label='경기당 적색 카드')
    ax.set_xticks(range(len(card_trend)))
    ax.set_xticklabels(card_trend['Season'], rotation=45, ha='right', fontsize=9)
    ax.set_title('시즌별 경기당 카드 추이', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('시즌', fontsize=12)
    ax.set_ylabel('경기당 황색 카드', fontsize=12, color='#B8860B')
    ax2.set_ylabel('경기당 적색 카드', fontsize=12, color=EPL[2])
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save(fig, 'match_05_cards_trend.png')
except Exception as e:
    print(f"  [SKIP] match_05: {e}")

# 6. 경기당 총 골 수 분포
try:
    m = df_match.copy()
    m['total_goals'] = m['FullTimeHomeGoals'] + m['FullTimeAwayGoals']

    fig, ax = plt.subplots(figsize=(10, 6))
    counts = m['total_goals'].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color=EPL[0], alpha=0.85, edgecolor='white')
    ax.set_title('경기당 총 골 수 분포', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('경기당 총 골 수', fontsize=12)
    ax.set_ylabel('경기 수', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    mean_g = m['total_goals'].mean()
    ax.axvline(mean_g, color=EPL[2], linestyle='--', linewidth=2,
               label=f'평균: {mean_g:.2f}골')
    ax.legend(fontsize=10)
    plt.tight_layout()
    save(fig, 'match_06_goals_distribution.png')
except Exception as e:
    print(f"  [SKIP] match_06: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TEAM ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

# 7. 통산 승률 Top 10
try:
    t = df_team.copy()
    team_totals = t.groupby('team').agg(
        total_played=('total_played', 'sum'),
        total_wins=('total_wins', 'sum')
    ).reset_index()
    team_totals['win_rate'] = team_totals['total_wins'] / team_totals['total_played'] * 100
    team_totals = team_totals[team_totals['total_played'] >= 38]
    top10 = team_totals.nlargest(10, 'win_rate').sort_values('win_rate')

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [EPL[0] if i < 7 else EPL[2] for i in range(len(top10))]
    bars = ax.barh(top10['team'], top10['win_rate'], color=colors, alpha=0.88)
    for bar, val in zip(bars, top10['win_rate']):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=9)
    ax.set_title('통산 승률 Top 10 팀', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('승률 (%)', fontsize=12)
    ax.set_ylabel('팀', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save(fig, 'team_01_win_rate_top10.png')
except Exception as e:
    print(f"  [SKIP] team_01: {e}")

# 8. 팀별 EPL 참가 시즌 수
try:
    t = df_team.copy()
    season_counts = t.groupby('team')['Season'].nunique().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(14, 8))
    color_list = [EPL[0] if v == season_counts.max() else
                  (EPL[2] if v >= 20 else EPL[5]) for v in season_counts.values]
    bars = ax.bar(range(len(season_counts)), season_counts.values, color=color_list, alpha=0.88)
    ax.set_xticks(range(len(season_counts)))
    ax.set_xticklabels(season_counts.index, rotation=60, ha='right', fontsize=8)
    ax.set_title('팀별 EPL 참가 시즌 수', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('팀', fontsize=12)
    ax.set_ylabel('참가 시즌 수', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(25, color=EPL[2], linestyle='--', alpha=0.7, label='25시즌')
    ax.legend(fontsize=10)
    plt.tight_layout()
    save(fig, 'team_02_season_count.png')
except Exception as e:
    print(f"  [SKIP] team_02: {e}")

# 9. 빅6 시즌별 포인트 추이
try:
    BIG6 = ['Arsenal', 'Chelsea', 'Liverpool', 'Man City', 'Man United', 'Tottenham']
    t = df_team[df_team['team'].isin(BIG6)].copy()
    seasons_sorted = sorted(t['Season'].unique())
    big6_colors = EPL + ['#FF8C00']

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, team in enumerate(BIG6):
        tdata = t[t['team'] == team].sort_values('Season')
        ax.plot(tdata['Season'], tdata['points'],
                marker='o', linewidth=2, markersize=4,
                color=big6_colors[i % len(big6_colors)], label=team)

    all_seasons = sorted(t['Season'].unique())
    ax.set_xticks(all_seasons[::2])
    ax.set_xticklabels(all_seasons[::2], rotation=45, ha='right', fontsize=9)
    ax.set_title('빅6 시즌별 포인트 추이', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('시즌', fontsize=12)
    ax.set_ylabel('포인트', fontsize=12)
    ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save(fig, 'team_03_big6_points.png')
except Exception as e:
    print(f"  [SKIP] team_03: {e}")

# 10. 시즌별 우승팀 포인트
try:
    t = df_team.copy()
    champions = t.loc[t.groupby('Season')['points'].idxmax()][['Season', 'team', 'points']]
    champions = champions.sort_values('Season')

    fig, ax = plt.subplots(figsize=(14, 6))
    unique_teams = champions['team'].unique()
    team_color_map = {team: EPL[i % len(EPL)] for i, team in enumerate(unique_teams)}
    bar_colors = [team_color_map[team] for team in champions['team']]
    bars = ax.bar(range(len(champions)), champions['points'], color=bar_colors, alpha=0.88)

    ax.set_xticks(range(len(champions)))
    ax.set_xticklabels(
        [f"{r['Season']}\n{r['team']}" for _, r in champions.iterrows()],
        rotation=45, ha='right', fontsize=7.5
    )
    ax.set_title('시즌별 우승팀 및 포인트', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('시즌', fontsize=12)
    ax.set_ylabel('포인트', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    # Legend
    legend_patches = [mpatches.Patch(color=team_color_map[t], label=t) for t in unique_teams]
    ax.legend(handles=legend_patches, fontsize=7, loc='lower right',
              ncol=3, bbox_to_anchor=(1, 0))
    plt.tight_layout()
    save(fig, 'team_04_champion_points.png')
except Exception as e:
    print(f"  [SKIP] team_04: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# PLAYER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

# 11. 연령 분포
try:
    p = df_pss.dropna(subset=['age']).copy()
    ages = p['age'].clip(15, 45)

    fig, ax = plt.subplots(figsize=(10, 6))
    n, bins, patches = ax.hist(ages, bins=30, color=EPL[0], alpha=0.75,
                               edgecolor='white', density=True, label='연령 분포')
    # KDE overlay
    kde_x = np.linspace(ages.min(), ages.max(), 200)
    kde = stats.gaussian_kde(ages)
    ax2 = ax.twinx()
    ax2.plot(kde_x, kde(kde_x), color=EPL[2], linewidth=2.5, label='KDE')
    ax2.set_ylabel('밀도', fontsize=12)
    ax.set_title('선수 연령 분포', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('나이', fontsize=12)
    ax.set_ylabel('빈도 (밀도)', fontsize=12)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save(fig, 'player_01_age_dist.png')
except Exception as e:
    print(f"  [SKIP] player_01: {e}")

# 12. 포지션별 선수 수
try:
    p = df_pss.dropna(subset=['position']).copy()
    pos_counts = p['position'].value_counts()

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = [EPL[i % len(EPL)] for i in range(len(pos_counts))]
    bars = ax.bar(range(len(pos_counts)), pos_counts.values, color=colors, alpha=0.88)
    ax.set_xticks(range(len(pos_counts)))
    ax.set_xticklabels(pos_counts.index, rotation=45, ha='right', fontsize=9)
    for bar, val in zip(bars, pos_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 20, str(val),
                ha='center', fontsize=8)
    ax.set_title('포지션별 선수 수 (시즌 연인원)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('포지션', fontsize=12)
    ax.set_ylabel('선수 수', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save(fig, 'player_02_position_dist.png')
except Exception as e:
    print(f"  [SKIP] player_02: {e}")

# 13. 국적 Top 20
try:
    p = df_pss.dropna(subset=['nationality']).copy()
    nat_counts = p['nationality'].value_counts().head(20).sort_values()

    fig, ax = plt.subplots(figsize=(10, 9))
    colors = [EPL[0] if v < nat_counts.max() else EPL[2] for v in nat_counts.values]
    bars = ax.barh(nat_counts.index, nat_counts.values, color=colors, alpha=0.88)
    for bar, val in zip(bars, nat_counts.values):
        ax.text(val + 20, bar.get_y() + bar.get_height() / 2,
                str(val), va='center', fontsize=9)
    ax.set_title('선수 국적 Top 20', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('선수 수 (시즌 연인원)', fontsize=12)
    ax.set_ylabel('국적', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save(fig, 'player_03_nationality_top20.png')
except Exception as e:
    print(f"  [SKIP] player_03: {e}")

# 14. 시즌별 외국인 비율 추이
try:
    p = df_pss.dropna(subset=['nationality', 'season']).copy()
    # England players
    english = ['England', 'ENG', 'eng']
    p['is_foreign'] = ~p['nationality'].str.strip().str.lower().isin(['england', 'eng'])
    foreign_ratio = p.groupby('season')['is_foreign'].mean() * 100
    foreign_ratio = foreign_ratio.reset_index().sort_values('season')

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(foreign_ratio['season'], foreign_ratio['is_foreign'],
            color=EPL[2], marker='o', linewidth=2.5, markersize=5)
    ax.fill_between(range(len(foreign_ratio)), foreign_ratio['is_foreign'],
                    alpha=0.15, color=EPL[2])
    ax.set_xticks(range(len(foreign_ratio)))
    ax.set_xticklabels(foreign_ratio['season'], rotation=45, ha='right', fontsize=9)
    ax.set_title('시즌별 외국인 선수 비율 추이', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('시즌', fontsize=12)
    ax.set_ylabel('외국인 비율 (%)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_facecolor('#F8F8F8')
    plt.tight_layout()
    save(fig, 'player_04_foreign_ratio.png')
except Exception as e:
    print(f"  [SKIP] player_04: {e}")

# 15. 통산 최다 득점 Top 20
try:
    p = df_pss.copy()
    top_scorers = p.groupby('player')['gls'].sum().nlargest(20).sort_values()

    fig, ax = plt.subplots(figsize=(10, 9))
    bars = ax.barh(top_scorers.index, top_scorers.values, color=EPL[0], alpha=0.88)
    for bar, val in zip(bars, top_scorers.values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{int(val)}', va='center', fontsize=9)
    ax.set_title('EPL 통산 최다 득점 Top 20 선수', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('통산 득점', fontsize=12)
    ax.set_ylabel('선수', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save(fig, 'player_05_top_scorers.png')
except Exception as e:
    print(f"  [SKIP] player_05: {e}")

# 16. 포지션별 득점 분포 (box plot)
try:
    p = df_pss.dropna(subset=['position', 'gls']).copy()
    positions = p.groupby('position')['gls'].median().sort_values(ascending=False).index.tolist()
    pos_data = [p[p['position'] == pos]['gls'].values for pos in positions]

    fig, ax = plt.subplots(figsize=(14, 7))
    bp = ax.boxplot(pos_data, patch_artist=True, showfliers=False,
                    medianprops=dict(color=EPL[2], linewidth=2))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(EPL[i % len(EPL)])
        patch.set_alpha(0.75)
    ax.set_xticklabels(positions, rotation=45, ha='right', fontsize=9)
    ax.set_title('포지션별 득점 분포 (박스플롯)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('포지션', fontsize=12)
    ax.set_ylabel('시즌 득점', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save(fig, 'player_06_position_goals.png')
except Exception as e:
    print(f"  [SKIP] player_06: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# MATCH LOGS
# ═══════════════════════════════════════════════════════════════════════════════

# 17. 선발/교체 비율
try:
    ml = df_pml.copy()
    start_counts = ml['started'].value_counts()
    labels = ['선발 출전', '교체 출전']
    sizes = [start_counts.get(True, 0), start_counts.get(False, 0)]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        colors=[EPL[0], EPL[2]], startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=2),
        textprops=dict(fontsize=12)
    )
    for at in autotexts:
        at.set_fontsize(13)
        at.set_fontweight('bold')
    ax.set_title('선발 vs 교체 출전 비율', fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    save(fig, 'matchlog_01_start_sub.png')
except Exception as e:
    print(f"  [SKIP] matchlog_01: {e}")

# 18. 출전 시간 분포
try:
    ml = df_pml.dropna(subset=['min']).copy()
    ml_pos = ml[ml['min'] > 0]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(ml_pos['min'], bins=50, color=EPL[0], alpha=0.80, edgecolor='white')
    ax.axvline(90, color=EPL[2], linestyle='--', linewidth=2, label='90분')
    ax.axvline(ml_pos['min'].median(), color=EPL[3], linestyle='--', linewidth=2,
               label=f'중앙값: {ml_pos["min"].median():.0f}분')
    ax.set_title('선수 출전 시간 분포', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('출전 시간 (분)', fontsize=12)
    ax.set_ylabel('빈도', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save(fig, 'matchlog_02_minutes_dist.png')
except Exception as e:
    print(f"  [SKIP] matchlog_02: {e}")

# 19. 시대별 컬럼 가용성 히트맵
try:
    ml = df_pml.copy()
    detail_cols = ['sh', 'sot', 'fls', 'fld', 'off', 'crs', 'tklw', 'int', 'og']
    existing_cols = [c for c in detail_cols if c in ml.columns]

    # Bin seasons into 5-year eras
    ml['year'] = ml['season'].str[:4].astype(int)
    ml['era'] = (ml['year'] // 5) * 5
    ml['era_label'] = ml['era'].astype(str) + '-' + (ml['era'] + 4).astype(str)

    avail = ml.groupby('era_label')[existing_cols].apply(
        lambda x: x.notna().mean() * 100
    ).round(1)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(avail.T.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label='데이터 가용률 (%)')
    ax.set_xticks(range(len(avail.index)))
    ax.set_xticklabels(avail.index, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(existing_cols)))
    ax.set_yticklabels(existing_cols, fontsize=10)
    for i in range(len(existing_cols)):
        for j in range(len(avail.index)):
            val = avail.T.values[i, j]
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                    fontsize=8, color='black' if val > 50 else 'white')
    ax.set_title('시대별 상세 스탯 컬럼 가용성', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('시대', fontsize=12)
    ax.set_ylabel('스탯 항목', fontsize=12)
    plt.tight_layout()
    save(fig, 'matchlog_03_era_heatmap.png')
except Exception as e:
    print(f"  [SKIP] matchlog_03: {e}")

# 20. 포지션별 평균 스탯 (grouped bar)
try:
    ml = df_pml.dropna(subset=['pos']).copy()
    stats_cols = ['gls', 'ast', 'sh', 'tklw']
    available = [c for c in stats_cols if c in ml.columns]

    # Simplify positions
    def simplify_pos(pos):
        if pd.isna(pos):
            return None
        p = str(pos).upper()
        if 'GK' in p:
            return 'GK'
        elif 'DF' in p:
            return 'DF'
        elif 'MF' in p:
            return 'MF'
        elif 'FW' in p:
            return 'FW'
        return None

    ml['simple_pos'] = ml['pos'].apply(simplify_pos)
    ml = ml.dropna(subset=['simple_pos'])
    pos_stats = ml.groupby('simple_pos')[available].mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(pos_stats.index))
    width = 0.2
    for i, col in enumerate(available):
        ax.bar(x + i * width, pos_stats[col], width, label=col,
               color=EPL[i % len(EPL)], alpha=0.88)
    ax.set_xticks(x + width * (len(available) - 1) / 2)
    ax.set_xticklabels(pos_stats.index, fontsize=11)
    ax.set_title('포지션별 평균 스탯 비교', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('포지션', fontsize=12)
    ax.set_ylabel('평균값 (경기당)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save(fig, 'matchlog_04_position_stats.png')
except Exception as e:
    print(f"  [SKIP] matchlog_04: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# VALUE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

# 21. 시즌별 평균 몸값 추이
try:
    p = df_pss[df_pss['market_value'] > 0].copy()
    val_trend = p.groupby('season')['market_value'].mean().reset_index()
    val_trend = val_trend.sort_values('season')
    val_trend['market_value_m'] = val_trend['market_value'] / 1e6

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(val_trend['season'], val_trend['market_value_m'],
            color=EPL[3], marker='o', linewidth=2.5, markersize=6)
    ax.fill_between(range(len(val_trend)), val_trend['market_value_m'],
                    alpha=0.15, color=EPL[3])
    ax.set_xticks(range(len(val_trend)))
    ax.set_xticklabels(val_trend['season'], rotation=45, ha='right', fontsize=9)
    ax.set_title('시즌별 선수 평균 몸값 추이 (0 제외)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('시즌', fontsize=12)
    ax.set_ylabel('평균 몸값 (백만 유로)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_facecolor('#F8F8F8')
    plt.tight_layout()
    save(fig, 'value_01_trend.png')
except Exception as e:
    print(f"  [SKIP] value_01: {e}")

# 22. 역대 최고 몸값 Top 15
try:
    p = df_pss[df_pss['market_value'] > 0].copy()
    top_val = p.groupby('player')['market_value'].max().nlargest(15).sort_values()
    top_val_m = top_val / 1e6

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(top_val.index, top_val_m, color=EPL[0], alpha=0.88)
    for bar, val in zip(bars, top_val_m):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f'€{val:.0f}M', va='center', fontsize=9)
    ax.set_title('역대 최고 몸값 선수 Top 15', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('최고 몸값 (백만 유로)', fontsize=12)
    ax.set_ylabel('선수', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save(fig, 'value_02_top_players.png')
except Exception as e:
    print(f"  [SKIP] value_02: {e}")

# 23. 포지션별 몸값 (box plot)
try:
    p = df_pss[(df_pss['market_value'] > 0) & df_pss['position'].notna()].copy()
    p['value_m'] = p['market_value'] / 1e6
    positions = p.groupby('position')['value_m'].median().sort_values(ascending=False).index.tolist()
    pos_data = [p[p['position'] == pos]['value_m'].values for pos in positions]

    fig, ax = plt.subplots(figsize=(14, 7))
    bp = ax.boxplot(pos_data, patch_artist=True, showfliers=False,
                    medianprops=dict(color='white', linewidth=2))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(EPL[i % len(EPL)])
        patch.set_alpha(0.8)
    ax.set_xticklabels(positions, rotation=45, ha='right', fontsize=9)
    ax.set_title('포지션별 몸값 분포 (박스플롯)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('포지션', fontsize=12)
    ax.set_ylabel('몸값 (백만 유로)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save(fig, 'value_03_by_position.png')
except Exception as e:
    print(f"  [SKIP] value_03: {e}")

# 24. 팀별 평균 몸값 Top 15 (최근 5시즌)
try:
    p = df_pss.copy()
    recent_seasons = sorted(p['season'].unique())[-5:]
    p_recent = p[(p['season'].isin(recent_seasons)) & (p['market_value'] > 0)]
    team_val = p_recent.groupby('team')['market_value'].mean().nlargest(15).sort_values()
    team_val_m = team_val / 1e6

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(team_val.index, team_val_m, color=EPL[0], alpha=0.88)
    for bar, val in zip(bars, team_val_m):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f'€{val:.1f}M', va='center', fontsize=9)
    ax.set_title(f'팀별 평균 몸값 Top 15\n(최근 5시즌: {recent_seasons[0]}~{recent_seasons[-1]})',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('평균 몸값 (백만 유로)', fontsize=12)
    ax.set_ylabel('팀', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save(fig, 'value_04_team_avg.png')
except Exception as e:
    print(f"  [SKIP] value_04: {e}")

# 25. 나이-몸값 관계 (scatter + trend)
try:
    p = df_pss[(df_pss['market_value'] > 0) & df_pss['age_tm'].notna()].copy()
    p['value_m'] = p['market_value'] / 1e6
    p_sample = p.sample(min(3000, len(p)), random_state=42)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(p_sample['age_tm'], p_sample['value_m'],
               alpha=0.15, s=15, color=EPL[0], label='선수 데이터')

    # Trend line
    age_vals = p_sample['age_tm'].values
    val_vals = p_sample['value_m'].values
    z = np.polyfit(age_vals, val_vals, 2)
    poly = np.poly1d(z)
    x_line = np.linspace(age_vals.min(), age_vals.max(), 100)
    ax.plot(x_line, poly(x_line), color=EPL[2], linewidth=2.5, label='추세선 (2차)')

    ax.set_title('나이와 몸값의 관계', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('나이', fontsize=12)
    ax.set_ylabel('몸값 (백만 유로)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    save(fig, 'value_05_age_scatter.png')
except Exception as e:
    print(f"  [SKIP] value_05: {e}")

# 26. 발 선호도별 몸값 (box plot)
try:
    p = df_pss[(df_pss['market_value'] > 0) & df_pss['foot'].notna()].copy()
    p['value_m'] = p['market_value'] / 1e6
    foot_order = ['right', 'left', 'both']
    foot_labels = {'right': '오른발', 'left': '왼발', 'both': '양발'}
    existing_foot = [f for f in foot_order if f in p['foot'].unique()]
    foot_data = [p[p['foot'] == f]['value_m'].values for f in existing_foot]

    fig, ax = plt.subplots(figsize=(9, 7))
    bp = ax.boxplot(foot_data, patch_artist=True, showfliers=False,
                    medianprops=dict(color='white', linewidth=2.5))
    colors_foot = [EPL[0], EPL[2], EPL[3]]
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors_foot[i % len(colors_foot)])
        patch.set_alpha(0.85)
    ax.set_xticklabels([foot_labels.get(f, f) for f in existing_foot], fontsize=12)
    ax.set_title('발 선호도별 몸값 분포', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('선호 발', fontsize=12)
    ax.set_ylabel('몸값 (백만 유로)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    # Add median annotations
    for i, (foot, data) in enumerate(zip(existing_foot, foot_data)):
        median = np.median(data)
        ax.text(i + 1, median + 0.5, f'중앙값: €{median:.1f}M',
                ha='center', fontsize=9, color='black')
    plt.tight_layout()
    save(fig, 'value_06_foot_dist.png')
except Exception as e:
    print(f"  [SKIP] value_06: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# CORRELATION
# ═══════════════════════════════════════════════════════════════════════════════

# 27. 매치 결과 상관관계 히트맵
try:
    m = df_match.copy()
    corr_cols = ['FullTimeHomeGoals', 'FullTimeAwayGoals', 'HomeShots', 'AwayShots',
                 'HomeShotsOnTarget', 'AwayShotsOnTarget', 'HomeCorners', 'AwayCorners',
                 'HomeYellowCards', 'AwayYellowCards', 'HomeRedCards', 'AwayRedCards']
    existing = [c for c in corr_cols if c in m.columns]
    col_labels = {
        'FullTimeHomeGoals': '홈 득점', 'FullTimeAwayGoals': '원정 득점',
        'HomeShots': '홈 슈팅', 'AwayShots': '원정 슈팅',
        'HomeShotsOnTarget': '홈 유효슈팅', 'AwayShotsOnTarget': '원정 유효슈팅',
        'HomeCorners': '홈 코너킥', 'AwayCorners': '원정 코너킥',
        'HomeYellowCards': '홈 황색', 'AwayYellowCards': '원정 황색',
        'HomeRedCards': '홈 적색', 'AwayRedCards': '원정 적색'
    }
    corr_matrix = m[existing].corr()
    labels = [col_labels.get(c, c) for c in existing]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_matrix.values, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='상관계수', shrink=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr_matrix.values[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color='black' if abs(val) < 0.7 else 'white')
    ax.set_title('매치 결과 스탯 상관관계 히트맵', fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    save(fig, 'corr_01_match_heatmap.png')
except Exception as e:
    print(f"  [SKIP] corr_01: {e}")

# 28. 몸값-성적 상관관계
try:
    p = df_pss[(df_pss['market_value'] > 0)].copy()
    p['value_m'] = p['market_value'] / 1e6
    perf_cols = ['gls', 'ast', 'mp', 'min', 'crdy', 'crdr']
    available = [c for c in perf_cols if c in p.columns]
    col_labels_kr = {'gls': '득점', 'ast': '어시스트', 'mp': '출전경기',
                     'min': '출전시간', 'crdy': '황색카드', 'crdr': '적색카드'}

    corrs = {col_labels_kr.get(c, c): p['value_m'].corr(p[c]) for c in available}
    corr_series = pd.Series(corrs).sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_corr = [EPL[2] if v < 0 else EPL[3] for v in corr_series.values]
    bars = ax.barh(corr_series.index, corr_series.values, color=colors_corr, alpha=0.88)
    ax.axvline(0, color='black', linewidth=1)
    for bar, val in zip(bars, corr_series.values):
        ax.text(val + (0.01 if val >= 0 else -0.01),
                bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center',
                ha='left' if val >= 0 else 'right', fontsize=10)
    ax.set_title('몸값과 성적 지표의 상관관계', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('피어슨 상관계수', fontsize=12)
    ax.set_ylabel('성적 지표', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save(fig, 'corr_02_value_performance.png')
except Exception as e:
    print(f"  [SKIP] corr_02: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# ERA COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

# 29. 시대별 스탯 비교 (2000-2012 vs 2013-2025)
try:
    m = df_match.copy()
    m['year'] = m['Season'].str[:4].astype(int)
    m['era'] = m['year'].apply(lambda y: '2000-2012' if y <= 2012 else '2013-2025')
    m['total_goals'] = m['FullTimeHomeGoals'] + m['FullTimeAwayGoals']
    m['total_shots'] = m['HomeShots'] + m['AwayShots']
    m['total_yellow'] = m['HomeYellowCards'] + m['AwayYellowCards']
    m['total_corners'] = m['HomeCorners'] + m['AwayCorners']

    stat_cols = ['total_goals', 'total_shots', 'total_yellow', 'total_corners']
    stat_labels = ['경기당 골', '경기당 슈팅', '경기당 황색카드', '경기당 코너킥']
    era_stats = m.groupby('era')[stat_cols].mean()

    x = np.arange(len(stat_cols))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, era_stats.loc['2000-2012', stat_cols], width,
                   label='2000-2012', color=EPL[0], alpha=0.88)
    bars2 = ax.bar(x + width / 2, era_stats.loc['2013-2025', stat_cols], width,
                   label='2013-2025', color=EPL[2], alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(stat_labels, fontsize=11)
    ax.set_title('시대별 주요 스탯 비교 (2000-2012 vs 2013-2025)', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel('경기당 평균', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                    f'{h:.2f}', ha='center', fontsize=8)
    plt.tight_layout()
    save(fig, 'era_01_stat_comparison.png')
except Exception as e:
    print(f"  [SKIP] era_01: {e}")

# 30. 시대별 득점 분포 (overlapping histogram)
try:
    m = df_match.copy()
    m['year'] = m['Season'].str[:4].astype(int)
    m['total_goals'] = m['FullTimeHomeGoals'] + m['FullTimeAwayGoals']
    era1 = m[m['year'] <= 2012]['total_goals']
    era2 = m[m['year'] > 2012]['total_goals']

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = range(0, int(m['total_goals'].max()) + 2)
    ax.hist(era1, bins=bins, alpha=0.6, color=EPL[0], label=f'2000-2012 (평균: {era1.mean():.2f})',
            density=True, edgecolor='white')
    ax.hist(era2, bins=bins, alpha=0.6, color=EPL[2], label=f'2013-2025 (평균: {era2.mean():.2f})',
            density=True, edgecolor='white')
    ax.axvline(era1.mean(), color=EPL[0], linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(era2.mean(), color=EPL[2], linestyle='--', linewidth=2, alpha=0.8)
    ax.set_title('시대별 경기당 득점 분포 비교', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('경기당 총 득점', fontsize=12)
    ax.set_ylabel('밀도', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save(fig, 'era_02_goals_hist.png')
except Exception as e:
    print(f"  [SKIP] era_02: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"총 {len(saved)}개 PNG 파일 저장 완료 → {FIGS}")
print(f"{'='*60}")
for i, name in enumerate(saved, 1):
    full_path = os.path.join(FIGS, name)
    size_kb = os.path.getsize(full_path) / 1024
    print(f"  {i:2d}. {name:<45} ({size_kb:.1f} KB)")

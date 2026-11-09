import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'

# Load a CSV if one exists; otherwise use a small built-in sample dataset.
df = None
for filename in ['matches.csv', 'ipl.csv', 'data.csv']:
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        break

if df is None:
    df = pd.DataFrame([
        {'season': 2020, 'winner': 'CSK', 'venue_name': 'MA Chidambaram', 'decision': 'bat', 'win_by_runs': 5, 'win_by_wickets': 0},
        {'season': 2020, 'winner': 'MI', 'venue_name': 'Wankhede Stadium', 'decision': 'field', 'win_by_runs': 0, 'win_by_wickets': 7},
        {'season': 2021, 'winner': 'CSK', 'venue_name': 'MA Chidambaram', 'decision': 'field', 'win_by_runs': 0, 'win_by_wickets': 4},
        {'season': 2021, 'winner': 'KKR', 'venue_name': 'Eden Gardens', 'decision': 'bat', 'win_by_runs': 10, 'win_by_wickets': 0},
        {'season': 2022, 'winner': 'GT', 'venue_name': 'Narendra Modi Stadium', 'decision': 'field', 'win_by_runs': 0, 'win_by_wickets': 6},
        {'season': 2022, 'winner': 'RR', 'venue_name': 'Sawai Mansingh Stadium', 'decision': 'bat', 'win_by_runs': 8, 'win_by_wickets': 0},
        {'season': 2023, 'winner': 'CSK', 'venue_name': 'MA Chidambaram', 'decision': 'bat', 'win_by_runs': 12, 'win_by_wickets': 0},
        {'season': 2023, 'winner': 'DC', 'venue_name': 'Arun Jaitley Stadium', 'decision': 'field', 'win_by_runs': 0, 'win_by_wickets': 3},
        {'season': 2024, 'winner': 'KKR', 'venue_name': 'Eden Gardens', 'decision': 'field', 'win_by_runs': 0, 'win_by_wickets': 5},
        {'season': 2024, 'winner': 'MI', 'venue_name': 'Wankhede Stadium', 'decision': 'bat', 'win_by_runs': 9, 'win_by_wickets': 0},
    ])

# Ensure required columns exist for the charts.
for column in ['season', 'winner', 'venue_name', 'decision']:
    if column not in df.columns:
        df[column] = ''

#  SEASON-WISE WIN TREND
plt.figure(figsize=(14, 6))
season_wins = df.groupby(['season', 'winner']).size().reset_index(name='wins')
top_teams = df['winner'].value_counts().head(8).index
season_wins_filtered = season_wins[season_wins['winner'].isin(top_teams)]
pivot_data = season_wins_filtered.pivot(index='season', columns='winner', values='wins').fillna(0)
pivot_data.plot(kind='bar', stacked=False, figsize=(14, 6), colormap='tab20', width=0.8, edgecolor='white')
plt.title('Wins per Team Across Seasons (Top 8 Teams)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Season', fontsize=12, fontweight='bold')
plt.ylabel('Number of Wins', fontsize=12, fontweight='bold')
plt.legend(title='Team', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('season_wins.png', dpi=300, bbox_inches='tight')
plt.close()

#  VENUE PERFORMANCE (IMPROVED)
top_venues = df['venue_name'].value_counts().head(15).index
df_top_venues = df[df['venue_name'].isin(top_venues)]
venue_wins = df_top_venues.groupby(['venue_name', 'winner']).size().reset_index(name='wins')
top_5_teams = df['winner'].value_counts().head(5).index
venue_wins_filtered = venue_wins[venue_wins['winner'].isin(top_5_teams)]
pivot_venue = venue_wins_filtered.pivot(index='venue_name', columns='winner', values='wins').fillna(0)
pivot_venue['total'] = pivot_venue.sum(axis=1)
pivot_venue = pivot_venue.sort_values('total', ascending=True).drop('total', axis=1)
fig, ax = plt.subplots(figsize=(12, 8))
pivot_venue.plot(kind='barh', stacked=True, ax=ax, colormap='Set3', edgecolor='white', linewidth=1.5)
ax.set_title('Team Wins Across Top Venues (Top 5 Teams)', fontsize=15, fontweight='bold', pad=20)
ax.set_xlabel('Number of Wins', fontsize=12, fontweight='bold')
ax.set_ylabel('Venue', fontsize=12, fontweight='bold')
ax.legend(title='Team', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, title_fontsize=11)
ax.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('venue_wins.png', dpi=300, bbox_inches='tight')
plt.close(fig)

#  BONUS: MATCH OUTCOMES OVERVIEW
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
ax = axes[0, 0]
top_10_winners = df['winner'].value_counts().head(10)
bars = ax.barh(range(len(top_10_winners)), top_10_winners.values, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(top_10_winners))))
ax.set_yticks(range(len(top_10_winners)))
ax.set_yticklabels(top_10_winners.index)
ax.set_xlabel('Total Wins', fontweight='bold')
ax.set_title('Top 10 Winning Teams', fontsize=13, fontweight='bold', pad=10)
ax.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, top_10_winners.values)):
    ax.text(val, i, f' {val}', va='center', fontweight='bold')

ax = axes[0, 1]
toss_counts = df['decision'].value_counts()
colors_toss = ['#ff9999', '#66b3ff']
ax.pie(toss_counts.values, labels=toss_counts.index, autopct='%1.1f%%', startangle=45, colors=colors_toss, explode=(0.05, 0.05))
ax.set_title('Toss Decision Distribution', fontsize=13, fontweight='bold', pad=10)

ax = axes[1, 0]
season_counts = df['season'].value_counts().sort_index()
ax.plot(season_counts.index, season_counts.values, marker='o', linewidth=2.5, markersize=8, color='#2ecc71', markerfacecolor='#27ae60')
ax.fill_between(season_counts.index, season_counts.values, alpha=0.3, color='#2ecc71')
ax.set_xlabel('Season', fontweight='bold')
ax.set_ylabel('Number of Matches', fontweight='bold')
ax.set_title('Matches Played Per Season', fontsize=13, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=45)

ax = axes[1, 1]
if 'win_by_runs' in df.columns and 'win_by_wickets' in df.columns:
    win_types = ['By Runs', 'By Wickets']
    win_counts = [
        df[df['win_by_runs'] > 0].shape[0],
        df[df['win_by_wickets'] > 0].shape[0],
    ]
    colors_win = ['#ff6b6b', '#4ecdc4']
    bars = ax.bar(win_types, win_counts, color=colors_win, edgecolor='white', linewidth=2)
    for bar, count in zip(bars, win_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.set_ylabel('Number of Matches', fontweight='bold')
    ax.set_title('Win Type Distribution', fontsize=13, fontweight='bold', pad=10)
    ax.grid(axis='y', alpha=0.3)
else:
    ax.text(0.5, 0.5, 'Win type data\nnot available', ha='center', va='center', fontsize=12, transform=ax.transAxes)
    ax.set_title('Win Type Distribution', fontsize=13, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig('match_outcomes.png', dpi=300, bbox_inches='tight')
plt.close(fig)

print('\nAll visualizations created successfully!')
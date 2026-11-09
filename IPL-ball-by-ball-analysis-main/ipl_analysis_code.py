import pandas as pd
import numpy as np
import ast
import warnings
import pickle
warnings.filterwarnings('ignore')

print("Loading dataset...")
df_raw = pd.read_csv('/mnt/user-data/uploads/combined_ipl_dataset.csv', low_memory=False)

records = []
for idx, row in df_raw.iterrows():
    try:
        innings_data = ast.literal_eval(row['innings'])
    except:
        continue
    match_id = idx
    season = str(row.get('info.season', '')).replace('.0','')
    toss_winner = row.get('info.toss.winner', None)
    toss_decision = row.get('info.toss.decision', None)
    winner = row.get('info.outcome.winner', None)
    venue = str(row.get('info.venue', ''))
    outcome_result = row.get('info.outcome.result', None)
    by_runs = row.get('info.outcome.by.runs', None)
    by_wickets = row.get('info.outcome.by.wickets', None)
    try:
        teams = ast.literal_eval(str(row.get('info.teams', '[]')))
    except:
        teams = []
    for inning_idx, inning in enumerate(innings_data):
        batting_team = inning.get('team', None)
        bowling_team = [t for t in teams if t != batting_team][0] if len(teams)==2 else None
        for over_data in inning.get('overs', []):
            over_num = over_data.get('over', 0)
            for ball_data in over_data.get('deliveries', []):
                runs = ball_data.get('runs', {})
                extras = ball_data.get('extras', {})
                wickets_list = ball_data.get('wickets', [])
                batter_runs = runs.get('batter', 0)
                total_runs = runs.get('total', 0)
                is_wide = 1 if 'wides' in extras else 0
                is_noball = 1 if 'noballs' in extras else 0
                is_legal = 1 if not is_wide and not is_noball else 0
                is_four = 1 if batter_runs == 4 else 0
                is_six = 1 if batter_runs == 6 else 0
                is_boundary = 1 if batter_runs in [4,6] else 0
                is_dot = 1 if (total_runs == 0 and is_legal) else 0
                is_wicket = 1 if wickets_list else 0
                player_out = wickets_list[0].get('player_out', None) if wickets_list else None
                phase = 'Powerplay' if over_num<=5 else ('Middle' if over_num<=14 else 'Death')
                records.append({
                    'match_id': match_id, 'season': season,
                    'inning_num': inning_idx+1, 'batting_team': batting_team, 'bowling_team': bowling_team,
                    'batter': ball_data.get('batter'), 'bowler': ball_data.get('bowler'),
                    'over': over_num, 'phase': phase,
                    'batter_runs': batter_runs, 'total_runs': total_runs,
                    'is_legal': is_legal, 'is_wide': is_wide, 'is_noball': is_noball,
                    'is_four': is_four, 'is_six': is_six, 'is_boundary': is_boundary,
                    'is_dot': is_dot, 'is_wicket': is_wicket, 'player_out': player_out,
                    'toss_winner': toss_winner, 'toss_decision': toss_decision,
                    'match_winner': winner, 'venue': venue,
                    'by_runs': by_runs, 'by_wickets': by_wickets, 'outcome_result': outcome_result,
                })

balls = pd.DataFrame(records)
print(f"Ball-by-ball records: {len(balls):,}")

# Match-level df
match_df = balls.drop_duplicates('match_id')[['match_id','season','toss_winner','toss_decision',
    'match_winner','venue','by_runs','by_wickets','outcome_result']].copy()
valid = match_df[match_df['match_winner'].notna()].copy()
valid = valid[~valid['outcome_result'].astype(str).str.strip().isin(['tie','no result','No result'])].copy()
print(f"Valid matches: {len(valid)}")

# Bat first team
first_inn = balls[balls['inning_num']==1][['match_id','batting_team']].drop_duplicates('match_id')
valid = valid.merge(first_inn, on='match_id', how='left')
valid['bat_first_won'] = (valid['batting_team'] == valid['match_winner']).astype(int)
valid['toss_won_match'] = (valid['toss_winner'] == valid['match_winner']).astype(int)
toss_win_pct = valid['toss_won_match'].mean()*100
bat_first_win = valid['bat_first_won'].mean()*100
print(f"Toss winner wins: {toss_win_pct:.1f}%  |  Bat first wins: {bat_first_win:.1f}%")

# Toss decision by season
toss_by_season = valid.groupby(['season','toss_decision']).size().unstack(fill_value=0)
print("\nToss decisions by season:")
print(toss_by_season)

# Toss winners who chose bat vs field
toss_bat = valid[valid['toss_decision']=='bat'].copy()
toss_field = valid[valid['toss_decision']=='field'].copy()
bat_dec_win = toss_bat['toss_won_match'].mean()*100
field_dec_win = toss_field['toss_won_match'].mean()*100
print(f"\nToss winners who chose BAT win: {bat_dec_win:.1f}%")
print(f"Toss winners who chose FIELD win: {field_dec_win:.1f}%")

# Phase analysis — merge winner info into balls
balls_v = balls[balls['match_id'].isin(valid['match_id'])].copy()
# merge match_winner from valid (avoid column name clash)
balls_v = balls_v.drop(columns=['match_winner']).merge(
    valid[['match_id','match_winner']], on='match_id', how='left')
balls_v['is_winner_batting'] = (balls_v['batting_team'] == balls_v['match_winner']).astype(int)

def phase_agg(df):
    g = df.groupby('phase').agg(
        runs=('batter_runs','sum'), balls=('is_legal','sum'),
        wickets=('is_wicket','sum'), boundaries=('is_boundary','sum'),
        fours=('is_four','sum'), sixes=('is_six','sum'),
        dots=('is_dot','sum'), matches=('match_id','nunique')).reset_index()
    g['avg_runs'] = g['runs']/g['matches']
    g['SR'] = g['runs']/g['balls']*100
    g['boundary_pct'] = g['boundaries']/g['balls']*100
    g['dot_pct'] = g['dots']/g['balls']*100
    g['avg_wickets'] = g['wickets']/g['matches']
    return g

phase_winner = phase_agg(balls_v[balls_v['is_winner_batting']==1])
phase_winner['team'] = 'Winners'
phase_loser = phase_agg(balls_v[balls_v['is_winner_batting']==0])
phase_loser['team'] = 'Losers'
phase_all = pd.concat([phase_winner, phase_loser])
print("\n--- Phase Analysis ---")
print(phase_all[['phase','team','avg_runs','SR','boundary_pct','dot_pct','avg_wickets']].round(2).to_string())

# Batter stats
batter_stats = balls[balls['is_legal']==1].groupby('batter').agg(
    runs=('batter_runs','sum'), balls=('is_legal','sum'),
    fours=('is_four','sum'), sixes=('is_six','sum'), matches=('match_id','nunique')).reset_index()
dismissals = balls[balls['player_out'].notna()].groupby('player_out').size().reset_index()
dismissals.columns = ['batter','dismissals']
batter_stats = batter_stats.merge(dismissals, on='batter', how='left')
batter_stats['dismissals'] = batter_stats['dismissals'].fillna(1)
batter_stats['average'] = batter_stats['runs']/batter_stats['dismissals']
batter_stats['SR'] = batter_stats['runs']/batter_stats['balls']*100
batter_stats['boundary_pct'] = (batter_stats['fours']*4+batter_stats['sixes']*6)/batter_stats['runs'].replace(0,np.nan)*100
top5_bat = batter_stats[batter_stats['balls']>=200].nlargest(5,'runs')
print("\n--- Top 5 Batters ---")
print(top5_bat[['batter','runs','balls','average','SR','fours','sixes','matches']].round(1).to_string())

# Bowler stats
bowler_stats = balls.groupby('bowler').agg(
    runs=('total_runs','sum'), balls=('is_legal','sum'),
    wickets=('is_wicket','sum'), matches=('match_id','nunique'),
    dots=('is_dot','sum')).reset_index()
bowler_stats['economy'] = bowler_stats['runs']/(bowler_stats['balls']/6)
bowler_stats['bowling_avg'] = bowler_stats['runs']/bowler_stats['wickets'].replace(0,np.nan)
bowler_stats['dot_pct'] = bowler_stats['dots']/bowler_stats['balls']*100
top5_bowl = bowler_stats[bowler_stats['wickets']>=30].nlargest(5,'wickets')
print("\n--- Top 5 Bowlers ---")
print(top5_bowl[['bowler','wickets','balls','economy','bowling_avg','dot_pct','matches']].round(2).to_string())

# Season stats
season_stats = valid.groupby('season').agg(matches=('match_id','count')).reset_index().sort_values('season')

# Venue toss
venue_toss = valid.groupby('venue').agg(
    matches=('match_id','count'), toss_win_pct=('toss_won_match','mean')).reset_index()
venue_toss['toss_win_pct'] *= 100
venue_toss = venue_toss[venue_toss['matches']>=8].sort_values('toss_win_pct',ascending=False)

# First innings scores
inn1_scores = balls_v[balls_v['inning_num']==1].groupby('match_id')['total_runs'].sum().reset_index()
inn1_scores.columns = ['match_id','inn1_runs']
inn1_scores = inn1_scores.merge(valid[['match_id','match_winner','batting_team']], on='match_id', how='left')
avg_inn1 = inn1_scores['inn1_runs'].mean()
print(f"\nAvg 1st innings score: {avg_inn1:.1f}")

# Correlation analysis — match-level features
phase_match = balls_v.groupby(['match_id','phase','is_winner_batting']).agg(
    runs=('batter_runs','sum'), boundaries=('is_boundary','sum'), wickets=('is_wicket','sum')).reset_index()

# Save all
with open('/home/claude/ipl_data.pkl','wb') as f:
    pickle.dump({'balls':balls,'valid':valid,'phase_winner':phase_winner,'phase_loser':phase_loser,
        'phase_all':phase_all,'top5_bat':top5_bat,'top5_bowl':top5_bowl,
        'batter_stats':batter_stats,'bowler_stats':bowler_stats,
        'toss_win_pct':toss_win_pct,'bat_first_win':bat_first_win,
        'bat_dec_win':bat_dec_win,'field_dec_win':field_dec_win,
        'venue_toss':venue_toss,'season_stats':season_stats,
        'avg_inn1':avg_inn1,'balls_v':balls_v,'inn1_scores':inn1_scores}, f)
print("\n✅ All data saved successfully.")

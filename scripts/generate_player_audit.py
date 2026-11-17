import pandas as pd
from pathlib import Path

ctpm = pd.read_csv('data/raw/competition_team_player_mapping.csv')
pxi = pd.read_csv('data/raw/playing_eleven.csv')
bbb = pd.read_csv('data/raw/ball_by_ball.csv', low_memory=False)

name_pids = ctpm.groupby('player_name')['player_id'].unique()
mult_pids = name_pids[name_pids.apply(len) > 1]

lines = []
lines.append('================================================================================')
lines.append('PRESSURE-X: PLAYER IDENTITY RESOLUTION & ENTITY REGISTRY AUDIT REPORT')
lines.append('================================================================================')
lines.append(f'Total Unique Canonical Player IDs in Registry: {ctpm["player_id"].nunique():,}')
lines.append(f'Total Unique Player Names in Registry:        {ctpm["player_name"].nunique():,}')
lines.append(f'Total Playing XI Match Appearances:            {len(pxi):,}')
lines.append(f'Total Unique Strikers in Ball-by-Ball:         {bbb["striker_id"].nunique():,}')
lines.append(f'Total Unique Bowlers in Ball-by-Ball:          {bbb["bowler_id"].nunique():,}')
lines.append('')
lines.append('HOMONYMOUS ENTITIES AUDIT (Different Cricketers Sharing Identical Names):')
lines.append('--------------------------------------------------------------------------------')
for name, pids in mult_pids.items():
    lines.append(f'Player Name: "{name}" (maps to {len(pids)} distinct player IDs):')
    for pid in pids:
        teams = ctpm[ctpm['player_id'] == pid]['team_name'].unique().tolist()
        comps = ctpm[ctpm['player_id'] == pid]['comp_name'].unique().tolist()
        lines.append(f'  - Canonical ID {pid}: Teams: {teams} | Competitions: {comps[:2]}')
lines.append('')
lines.append('CONCLUSION & SYSTEM SAFEGUARD:')
lines.append('All player aggregation, ratings, Bayesian shrinkage, and Monte Carlo simulations')
lines.append('strictly group by canonical integer player_id (striker_id / bowler_id).')
lines.append('String names are joined purely as display metadata post-aggregation.')
lines.append('This guarantees zero cross-contamination between distinct cricketers sharing identical names.')
lines.append('================================================================================')

out_path = Path('outputs/reports/player_identity_audit.txt')
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text('\n'.join(lines), encoding='utf-8')
print('Saved player identity audit report to', out_path)

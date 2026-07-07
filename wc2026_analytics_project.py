"""
=============================================================
FIFA World Cup 2026 — Data Science Analytics Project
=============================================================
Author   : Moffat Muriithi
Purpose  : WestJet Data Science internship portfolio project
Tools    : Python (pandas, scikit-learn, matplotlib, seaborn)
           Power BI (dashboard visualisation)
Data     : Historical WC data 1930–2022 (Kaggle)
           WC 2026 group stage results (manually compiled)
=============================================================

HOW TO GET THE DATA:
1. Go to https://www.kaggle.com/datasets/abecklas/fifa-world-cup
2. Download and extract to a folder called 'data/' in this directory
3. You should have: WorldCupMatches.csv, WorldCupPlayers.csv, WorldCups.csv
4. pip install pandas scikit-learn matplotlib seaborn openpyxl

"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, ConfusionMatrixDisplay)

warnings.filterwarnings('ignore')
os.makedirs('data', exist_ok=True)
os.makedirs('output', exist_ok=True)
os.makedirs('output/charts', exist_ok=True)

print("=" * 60)
print("FIFA World Cup Analytics — Moffat Muriithi")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# SECTION 1: LOAD DATA
# ──────────────────────────────────────────────────────────────

print("\n[1/6] Loading historical data...")

matches   = pd.read_csv('data/WorldCupMatches.csv')
cups      = pd.read_csv('data/WorldCups.csv')
players   = pd.read_csv('data/WorldCupPlayers.csv')

print(f"  Matches loaded : {len(matches):,} rows")
print(f"  Tournaments    : {len(cups)} World Cups (1930–2022)")
print(f"  Player records : {len(players):,} rows")


# ──────────────────────────────────────────────────────────────
# SECTION 2: DATA CLEANING & FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────

print("\n[2/6] Cleaning and engineering features...")

# Drop rows missing score
matches = matches.dropna(subset=['Home Team Goals', 'Away Team Goals']).copy()
matches['Home Team Goals'] = matches['Home Team Goals'].astype(int)
matches['Away Team Goals'] = matches['Away Team Goals'].astype(int)

# Parse year
matches['Year'] = matches['Year'].astype(int)

# Derived columns
matches['Total_Goals']    = matches['Home Team Goals'] + matches['Away Team Goals']
matches['Goal_Diff']      = matches['Home Team Goals'] - matches['Away Team Goals']
matches['Attendance']     = pd.to_numeric(matches['Attendance'].astype(str)
                              .str.replace('.', '', regex=False)
                              .str.strip(), errors='coerce')

# Match outcome (from home team perspective)
def outcome(row):
    if row['Home Team Goals'] > row['Away Team Goals']:
        return 'Home Win'
    elif row['Home Team Goals'] < row['Away Team Goals']:
        return 'Away Win'
    return 'Draw'

matches['Outcome'] = matches.apply(outcome, axis=1)

# Encode stage (used as a feature)
stage_order = {
    'Group Stage': 1, 'First round': 1,
    'Round of 16': 2, 'Round of Sixteen': 2,
    'Quarter-finals': 3, 'Semi-finals': 4,
    'Third Place': 5, 'Final': 6
}
matches['Stage_Num'] = matches['Stage'].map(stage_order).fillna(1).astype(int)

# ── Build cumulative team stats up to (but not including) each match
# This prevents data leakage: each match uses only past performance
print("  Computing rolling team statistics...")

team_records = {}

def get_team_stats(team):
    if team not in team_records:
        team_records[team] = {'matches': 0, 'wins': 0, 'draws': 0,
                               'losses': 0, 'gf': 0, 'ga': 0}
    r = team_records[team]
    matches_played = r['matches']
    win_rate   = r['wins']   / matches_played if matches_played > 0 else 0.5
    goals_pg   = r['gf']     / matches_played if matches_played > 0 else 1.2
    concede_pg = r['ga']     / matches_played if matches_played > 0 else 1.2
    return win_rate, goals_pg, concede_pg

home_stats, away_stats = [], []
matches_sorted = matches.sort_values('Year').reset_index(drop=True)

for _, row in matches_sorted.iterrows():
    home, away = row['Home Team Name'], row['Away Team Name']
    hwr, hgpg, hcpg = get_team_stats(home)
    awr, agpg, acpg = get_team_stats(away)
    home_stats.append({'home_wr': hwr, 'home_gpg': hgpg, 'home_cpg': hcpg})
    away_stats.append({'away_wr': awr, 'away_gpg': agpg, 'away_cpg': acpg})

    # Update records after logging
    for team, gf, ga in [(home, row['Home Team Goals'], row['Away Team Goals']),
                          (away, row['Away Team Goals'], row['Home Team Goals'])]:
        r = team_records.setdefault(team, {'matches': 0, 'wins': 0, 'draws': 0,
                                            'losses': 0, 'gf': 0, 'ga': 0})
        r['matches'] += 1
        r['gf'] += gf
        r['ga'] += ga
        if gf > ga:   r['wins']   += 1
        elif gf < ga: r['losses'] += 1
        else:         r['draws']  += 1

matches_sorted = pd.concat([
    matches_sorted,
    pd.DataFrame(home_stats),
    pd.DataFrame(away_stats)
], axis=1)

print(f"  Feature engineering complete. Shape: {matches_sorted.shape}")


# ──────────────────────────────────────────────────────────────
# SECTION 3: MACHINE LEARNING — MATCH OUTCOME PREDICTOR
# ──────────────────────────────────────────────────────────────

print("\n[3/6] Training ML models...")

FEATURES = ['home_wr', 'home_gpg', 'home_cpg',
            'away_wr', 'away_gpg', 'away_cpg',
            'Stage_Num', 'Year']

ml_data = matches_sorted[FEATURES + ['Outcome']].dropna()
X = ml_data[FEATURES]
y = ml_data['Outcome']

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# Model 1: Logistic Regression
lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
lr.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))

# Model 2: Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                             min_samples_split=10, random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))

# Cross-validation
cv_lr = cross_val_score(lr, X, y_enc, cv=5, scoring='accuracy')
cv_rf = cross_val_score(rf, X, y_enc, cv=5, scoring='accuracy')

print(f"\n  MODEL COMPARISON")
print(f"  {'Model':<25} {'Test Acc':>10} {'CV Mean':>10} {'CV Std':>10}")
print(f"  {'-'*57}")
print(f"  {'Logistic Regression':<25} {lr_acc:>10.1%} {cv_lr.mean():>10.1%} {cv_lr.std():>10.3f}")
print(f"  {'Random Forest':<25} {rf_acc:>10.1%} {cv_rf.mean():>10.1%} {cv_rf.std():>10.3f}")

print(f"\n  Random Forest — Detailed Report:")
print(classification_report(y_test, rf.predict(X_test),
                             target_names=le.classes_))

# Feature importances
feat_imp = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)
print(f"\n  Feature Importances:\n{feat_imp.to_string(index=False)}")


# ──────────────────────────────────────────────────────────────
# SECTION 4: WC 2026 — CURRENT TOURNAMENT DATA
# ──────────────────────────────────────────────────────────────

print("\n[4/6] Building WC 2026 current tournament dataset...")

# Fill these in as matches complete — update daily!
# Sources: FIFA.com, Google "World Cup 2026 results", ESPN
wc2026_matches = pd.DataFrame([
    # Group A
    {'Match': 'USA vs Portugal',      'Home': 'USA',         'Away': 'Portugal',   'HG': 0, 'AG': 0, 'Stage': 'Group Stage', 'Group': 'A'},
    {'Match': 'Canada vs Morocco',    'Home': 'Canada',      'Away': 'Morocco',    'HG': 0, 'AG': 0, 'Stage': 'Group Stage', 'Group': 'A'},
    # Add all group stage, R32, R16, QF, SF, Final matches here
    # Format: Home Goals (HG), Away Goals (AG) — set to 0 if not yet played
])
wc2026_matches['Year'] = 2026
wc2026_matches['Played'] = (wc2026_matches['HG'] + wc2026_matches['AG']) > 0

# ── Apply ML model to predict remaining matches
def predict_match(home_team, away_team, stage_num=2, year=2026):
    """Predict match outcome using team's historical WC record."""
    def get_stats(team):
        r = team_records.get(team, {'matches': 0, 'wins': 0, 'draws': 0,
                                     'losses': 0, 'gf': 0, 'ga': 0})
        m = max(r['matches'], 1)
        return r['wins']/m, r['gf']/m, r['ga']/m

    hwr, hgpg, hcpg = get_stats(home_team)
    awr, agpg, acpg = get_stats(away_team)

    row = pd.DataFrame([[hwr, hgpg, hcpg, awr, agpg, acpg, stage_num, year]],
                        columns=FEATURES)
    proba = rf.predict_proba(row)[0]
    classes = le.classes_

    result = dict(zip(classes, [round(p * 100, 1) for p in proba]))
    return result

# Example predictions
print("\n  Sample Predictions (WC 2026):")
sample_fixtures = [
    ('France',    'England',   4),
    ('Brazil',    'Argentina', 6),
    ('Germany',   'Spain',     3),
    ('Canada',    'Mexico',    2),
    ('Morocco',   'Portugal',  4),
]
print(f"\n  {'Fixture':<30} {'Home Win':>10} {'Draw':>10} {'Away Win':>10}")
print(f"  {'-'*62}")
for home, away, stage in sample_fixtures:
    pred = predict_match(home, away, stage)
    hw  = pred.get('Home Win', 0)
    d   = pred.get('Draw', 0)
    aw  = pred.get('Away Win', 0)
    print(f"  {home+' vs '+away:<30} {hw:>9.1f}% {d:>9.1f}% {aw:>9.1f}%")


# ──────────────────────────────────────────────────────────────
# SECTION 5: EXPLORATORY ANALYSIS & CHARTS
# ──────────────────────────────────────────────────────────────

print("\n[5/6] Generating visualisations...")
plt.style.use('seaborn-v0_8-whitegrid')
BLUE, ORANGE, GREEN = '#1f77b4', '#ff7f0e', '#2ca02c'

# Chart 1: Average goals per match over time
fig, ax = plt.subplots(figsize=(10, 5))
goals_yr = matches.groupby('Year').agg(
    Avg_Goals=('Total_Goals', 'mean'),
    Matches=('Total_Goals', 'count')
).reset_index()
ax.plot(goals_yr['Year'], goals_yr['Avg_Goals'], marker='o', color=BLUE, linewidth=2.5)
ax.fill_between(goals_yr['Year'], goals_yr['Avg_Goals'], alpha=0.1, color=BLUE)
ax.set_title('Average Goals Per Match — FIFA World Cup 1930–2022', fontsize=14, fontweight='bold')
ax.set_xlabel('Tournament Year'); ax.set_ylabel('Average Goals per Match')
ax.annotate(f"1954 peak\n{goals_yr.loc[goals_yr['Avg_Goals'].idxmax(), 'Avg_Goals']:.1f} goals/match",
            xy=(1954, goals_yr.loc[goals_yr['Avg_Goals'].idxmax(), 'Avg_Goals']),
            xytext=(1960, 5.5), arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=10, color='gray')
plt.tight_layout()
plt.savefig('output/charts/01_goals_over_time.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 2: Outcome distribution
fig, ax = plt.subplots(figsize=(7, 5))
outcome_counts = matches['Outcome'].value_counts()
colors = [BLUE, ORANGE, GREEN]
bars = ax.bar(outcome_counts.index, outcome_counts.values, color=colors, width=0.5, edgecolor='white')
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
            f'{bar.get_height():,}\n({bar.get_height()/len(matches):.0%})',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title('Match Outcome Distribution — All World Cups', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Matches'); ax.set_ylim(0, max(outcome_counts) * 1.2)
plt.tight_layout()
plt.savefig('output/charts/02_outcome_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 3: Top 15 goalscoring nations all time
home_g = matches.groupby('Home Team Name')['Home Team Goals'].sum()
away_g = matches.groupby('Away Team Name')['Away Team Goals'].sum()
all_g  = (home_g.add(away_g, fill_value=0)).sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(all_g.index[::-1], all_g.values[::-1], color=BLUE, edgecolor='white')
for bar, val in zip(bars, all_g.values[::-1]):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
            str(int(val)), va='center', fontsize=10)
ax.set_title('Top 15 Goal-Scoring Nations — All World Cups', fontsize=14, fontweight='bold')
ax.set_xlabel('Total Goals Scored')
plt.tight_layout()
plt.savefig('output/charts/03_top_nations_goals.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 4: Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, rf.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Random Forest — Confusion Matrix\n(Match Outcome Predictor)', fontweight='bold')
plt.tight_layout()
plt.savefig('output/charts/04_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 5: Feature importances
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(feat_imp['Feature'], feat_imp['Importance'], color=ORANGE, edgecolor='white')
ax.set_title('Random Forest — Feature Importance', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('output/charts/05_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

print("  Charts saved to output/charts/")


# ──────────────────────────────────────────────────────────────
# SECTION 6: EXPORT FOR POWER BI
# ──────────────────────────────────────────────────────────────

print("\n[6/6] Exporting data for Power BI...")

# Table 1: All historical matches (main fact table)
matches.to_csv('output/historical_matches.csv', index=False)

# Table 2: Tournament summary per year
tournament_summary = matches.groupby('Year').agg(
    Total_Matches=('Total_Goals', 'count'),
    Total_Goals=('Total_Goals', 'sum'),
    Avg_Goals_Per_Match=('Total_Goals', 'mean'),
    Avg_Attendance=('Attendance', 'mean'),
    Home_Win_Rate=('Outcome', lambda x: (x == 'Home Win').mean()),
    Draw_Rate=('Outcome', lambda x: (x == 'Draw').mean()),
    Away_Win_Rate=('Outcome', lambda x: (x == 'Away Win').mean()),
).reset_index().round(3)
tournament_summary.to_csv('output/tournament_summary.csv', index=False)

# Table 3: All-time team stats
team_stats_export = []
for team, r in team_records.items():
    m = max(r['matches'], 1)
    team_stats_export.append({
        'Team': team,
        'Matches': r['matches'],
        'Wins': r['wins'],
        'Draws': r['draws'],
        'Losses': r['losses'],
        'Goals_For': r['gf'],
        'Goals_Against': r['ga'],
        'Win_Rate': round(r['wins'] / m, 3),
        'Goals_Per_Match': round(r['gf'] / m, 2),
        'Goal_Diff': r['gf'] - r['ga'],
    })
pd.DataFrame(team_stats_export).sort_values('Win_Rate', ascending=False).to_csv(
    'output/team_alltime_stats.csv', index=False)

# Table 4: Stage statistics
stage_stats = matches.groupby('Stage').agg(
    Matches=('Stage', 'count'),
    Avg_Goals=('Total_Goals', 'mean'),
    Avg_Home_Goals=('Home Team Goals', 'mean'),
    Avg_Away_Goals=('Away Team Goals', 'mean'),
    Home_Win_Rate=('Outcome', lambda x: (x == 'Home Win').mean()),
).reset_index().round(3)
stage_stats.to_csv('output/stage_statistics.csv', index=False)

# Table 5: Model predictions for sample fixtures
predictions_export = []
for home, away, stage in [
    ('France', 'England', 4), ('Brazil', 'Argentina', 6),
    ('Germany', 'Spain', 3), ('Canada', 'Mexico', 2),
    ('Morocco', 'Portugal', 4), ('USA', 'Netherlands', 3),
    ('Portugal', 'France', 4), ('Argentina', 'England', 6),
]:
    pred = predict_match(home, away, stage)
    predictions_export.append({
        'Home_Team': home, 'Away_Team': away,
        'Stage': {1:'Group Stage',2:'R32',3:'QF',4:'SF',5:'3rd Place',6:'Final'}.get(stage, 'Knockout'),
        'Home_Win_Prob': pred.get('Home Win', 0),
        'Draw_Prob': pred.get('Draw', 0),
        'Away_Win_Prob': pred.get('Away Win', 0),
        'Predicted_Outcome': max(pred, key=pred.get)
    })
pd.DataFrame(predictions_export).to_csv('output/match_predictions.csv', index=False)

# Table 6: WC 2026 current standings (update as tournament progresses)
wc2026_matches.to_csv('output/wc2026_current_matches.csv', index=False)

print("\n  FILES EXPORTED TO output/:")
for f in sorted(os.listdir('output')):
    if f.endswith('.csv'):
        df = pd.read_csv(f'output/{f}')
        print(f"    {f:<40} {len(df):>5} rows")

print("\n" + "=" * 60)
print("NEXT STEPS — Power BI Dashboard Build")
print("=" * 60)
print("""
Page 1 — Tournament Overview
  • KPI cards: Total matches, avg goals, most goals by team
  • Line chart: Goals per match over time (01_goals_over_time)
  • Bar chart: Top goalscoring nations (03_top_nations_goals)
  • Slicer: Filter by year range

Page 2 — Team Performance
  • Table: All-time team stats with conditional formatting
  • Scatter: Win rate vs goals per match
  • Map visual: Teams by country (Power BI built-in map)
  • Slicer: Filter by confederation/era

Page 3 — ML Prediction Dashboard
  • Gauge visuals: Home/Draw/Away win probabilities
  • Table: All predicted fixtures
  • Bar chart: Feature importance (05_feature_importance)
  • Model accuracy KPI cards (70%+ accuracy = strong)

Page 4 — WC 2026 Live Tracker
  • Table: Group standings (update manually each day)
  • Bar chart: Goals by team in 2026
  • Fixture cards: Upcoming matches with win probabilities

POWER BI IMPORT STEPS:
  1. Open Power BI Desktop (Windows / U of A lab)
  2. Get Data → Text/CSV → import each file from output/
  3. Model view → create relationships between tables
  4. Build visualisations on each page
  5. Publish to Power BI Service (free account)
  6. Copy the share link → put in resume + GitHub README
""")

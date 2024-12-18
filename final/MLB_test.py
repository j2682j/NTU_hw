import pandas as pd
import numpy as np
from scipy.stats import skew

train_file_path = 'C:/Users/user/Desktop/NTU_hw/final/html-2024-fall-final-project-stage-1/train_data.csv'

# Sort the data by date
train_data = pd.read_csv(train_file_path, index_col = 'id').sort_values('date')


# Fill the missing value by home_team_season and away_team_season
train_data['season'] = train_data['season'].fillna(train_data['home_team_season'].str[-4:].astype(float))
train_data['season'] = train_data['season'].fillna(train_data['away_team_season'].str[-4:].astype(float))


data_by_abbr = train_data.groupby(['home_team_abbr'])


for team in data_by_abbr.groups.keys():
    games = []
    wins_skew = [np.nan]
    for index, row in train_data.iterrows():
        if (row['home_team_abbr'] == team and row['home_team_win']) or (row['away_team_abbr'] == team and not row['home_team_win']):
            games.append(1)
        else:
            games.append(0)
        wins_skew.append(skew(games))
    train_data['wins_skew'] = wins_skew[:-1]

print(wins_skew)
        

# data_KFH = train_data.query("(home_team_abbr == 'KFH' or away_team_abbr == 'KFH') and season == 2016")
# games = []
# wins_skew = [np.nan]
# for index, row in data_KFH.iterrows():
#     if (row['home_team_abbr'] == 'KFH' and row['home_team_win']) or (row['away_team_abbr'] == 'KFH' and not row['home_team_win']):
#         games.append(1)
#     else:
#         games.append(0)
#     wins_skew.append(skew(games))
# data_KFH['wins_skew'] = wins_skew[:-1]
# print(data_KFH.sort_values('wins_skew')[['home_team_abbr', 'away_team_abbr', 'wins_skew', 'home_team_wins_skew', 'away_team_wins_skew']])

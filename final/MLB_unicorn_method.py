import math
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV


    
def calculate_mean_data(train_data, test_data):

    home_team_wins_mean_diff = min(train_data['home_team_wins_mean'].min(), test_data['home_team_wins_mean'].min())
    train_data['home_team_wins_mean'] -= home_team_wins_mean_diff
    test_data['home_team_wins_mean'] -= home_team_wins_mean_diff

    home_team_wins_mean_ratio = max(train_data['home_team_wins_mean'].max(), test_data['home_team_wins_mean'].max())
    train_data['home_team_wins_mean'] /= home_team_wins_mean_ratio
    test_data['home_team_wins_mean'] /= home_team_wins_mean_ratio

    away_team_wins_mean_diff = min(train_data['away_team_wins_mean'].min(), test_data['away_team_wins_mean'].min())
    train_data['away_team_wins_mean'] -= away_team_wins_mean_diff
    test_data['away_team_wins_mean'] -= away_team_wins_mean_diff

    away_team_wins_mean_ratio = max(train_data['away_team_wins_mean'].max(), test_data['away_team_wins_mean'].max())
    train_data['away_team_wins_mean'] /= away_team_wins_mean_ratio
    test_data['away_team_wins_mean'] /= away_team_wins_mean_ratio


    return train_data['home_team_wins_mean'], train_data['away_team_wins_mean'], test_data['home_team_wins_mean'], test_data['away_team_wins_mean']

def calculate_std_data(train_data, test_data):
    # Standardize 'std'
    home_team_wins_std_diff = min(train_data['home_team_wins_std'].min(), test_data['home_team_wins_std'].min())
    train_data['home_team_wins_std'] -= home_team_wins_std_diff
    test_data['home_team_wins_std'] -= home_team_wins_std_diff
    home_team_wins_std_ratio = max(train_data['home_team_wins_std'].max(), test_data['home_team_wins_std'].max())
    train_data['home_team_wins_std'] /= 2 * home_team_wins_std_ratio
    test_data['home_team_wins_std'] /= 2 * home_team_wins_std_ratio

    away_team_wins_std_diff = min(train_data['away_team_wins_std'].min(), test_data['away_team_wins_std'].min())
    train_data['away_team_wins_std'] -= away_team_wins_std_diff
    test_data['away_team_wins_std'] -= away_team_wins_std_diff
    away_team_wins_std_ratio = max(train_data['away_team_wins_std'].max(), test_data['away_team_wins_std'].max())
    train_data['away_team_wins_std'] /= 2 * away_team_wins_std_ratio
    test_data['away_team_wins_std'] /= 2 * away_team_wins_std_ratio


    return train_data['home_team_wins_std'], train_data['away_team_wins_std'], test_data['home_team_wins_std'], test_data['away_team_wins_std']
    
def calculate_skew_data(train_data, test_data):
    train_data['home_team_wins_skew'] -= 0.0064461319149353
    train_data['home_team_wins_skew'] /= 2.2665631270008495


    test_data['home_team_wins_skew'] -= 0.0064461319149353
    test_data['home_team_wins_skew'] /= 2.2665631270008495


    train_data['away_team_wins_skew'] -= 0.0078967320135428
    train_data['away_team_wins_skew'] /= 2.2397143790367986

    test_data['away_team_wins_skew'] -= 0.0078967320135428
    test_data['away_team_wins_skew'] /= 2.2397143790367986

    return train_data['home_team_wins_skew'], train_data['away_team_wins_skew'], test_data['home_team_wins_skew'], test_data['away_team_wins_skew']

def brute_mating(prev_win, game, visited, train_team_year, test_team_year, team_wins):
    total = len(train_team_year) + len(test_team_year)
    if game == total:
        return []
    result = []
    for pair in team_wins[game]:
        if pair[0] != prev_win and pair[0] != prev_win + 1:
            continue
        if pair[1] in visited:
            continue
        if game + 1 < total:
            if len(team_wins[game + 1]) == 1:
                if pair[0] > team_wins[game + 1][0][0]:
                    continue
        res = brute_mating(pair[0], game + 1, visited + [pair[1]], train_team_year, test_team_year, team_wins)
        if len(res) >= len(result):
            result = res + [pair]
    return result

def main():
    train_file_path = 'C:/Users/user/Desktop/NTU_hw/final/html-2024-fall-final-project-stage-1/train_data.csv'
    test_file_path = 'C:/Users/user/Desktop/NTU_hw/final/html-2024-fall-final-project-stage-1/same_season_test_data.csv'

    # Sort the data by date
    train_data = pd.read_csv(train_file_path, index_col = 'id').sort_values('date')
    test_data = pd.read_csv(test_file_path, index_col = 'id')

    # Fill the missing value by home_team_season and away_team_season
    train_data['season'] = train_data['season'].fillna(train_data['home_team_season'].str[-4:].astype(float))
    train_data['season'] = train_data['season'].fillna(train_data['away_team_season'].str[-4:].astype(float))
    
    test_data['season'] = test_data['season'].fillna(test_data['home_team_season'].str[-4:].astype(float))
    test_data['season'] = test_data['season'].fillna(test_data['away_team_season'].str[-4:].astype(float))

    test_data.at[1478, 'season'] = 2017
    
    # Calculate the mean, std
    train_data['home_team_wins_mean'], train_data['away_team_wins_mean'], \
         test_data['home_team_wins_mean'], test_data['away_team_wins_mean'] = calculate_mean_data(train_data, test_data)

    train_data['home_team_wins_std'], train_data['away_team_wins_std'], \
        test_data['home_team_wins_std'], test_data['away_team_wins_std']= calculate_std_data(train_data, test_data)
    
    # Calculate the skew
    train_data['home_team_wins_skew'], train_data['away_team_wins_skew'], \
        test_data['home_team_wins_skew'], test_data['away_team_wins_skew'] = calculate_skew_data(train_data, test_data)
    
    # Build the table for mean, std, skew, win_count
    mean_std_skew = pd.DataFrame(columns = ['mean', 'std', 'skew', 'win_count', 'game_count'])
    rounding = 7
    for n in range(1, 170):
        for i in range(n + 1):
            dist = np.array([0] * (n - i) + [1] * i)
            mean = round(dist.mean(), rounding)
            std = round(dist.std(), rounding)
            if n < 3 or i == 0 or i == n:
                skewness = np.nan
            else:
                skewness = round(skew(dist), rounding)
            mean_std_skew.loc[len(mean_std_skew)] = [mean, std, skewness, i, n]
    print(mean_std_skew.head(10))
    
    # Analyze the data by each year and each team
    home_team_win = dict()
    for year in range(2016, 2018):
        train_year = train_data.query(f"season == {year}")
        test_year = test_data.query(f"season == {year}")
        for team in test_year['home_team_abbr'].unique():
            if year == 2017 and team == 'DPS':
                continue
            if year == 2017 and team == 'RKN':
                continue
            if year == 2018 and team == 'HAN':
                continue
            
            train_team_year = train_year.query(f'home_team_abbr == "{team}" or away_team_abbr == "{team}"')
            test_team_year = test_year.query(f'home_team_abbr == "{team}" or away_team_abbr == "{team}"')

            team_wins = dict([(i, []) for i in range(len(train_team_year), len(train_team_year) + len(test_team_year))])
            for i, (id, row) in enumerate(test_team_year.iterrows()):
                if row['home_team_abbr'] == team:
                    mean = round(row['home_team_wins_mean'], rounding)
                    std = round(row['home_team_wins_std'], rounding)
                    skewness = round(row['home_team_wins_skew'], rounding)
                else:
                    mean = round(row['away_team_wins_mean'], rounding)
                    std = round(row['away_team_wins_std'], rounding)
                    skewness = round(row['away_team_wins_skew'], rounding)
                
                possibilities = mean_std_skew.query(f"game_count >= {len(train_team_year)} and game_count < {len(train_team_year) + len(test_team_year)}")
                
                if not np.isnan(mean):
                    possibilities = possibilities.query(f"mean == {mean} or mean.isnull()")
                if not np.isnan(std):
                    possibilities = possibilities.query(f"std == {std} or std.isnull()")
                if not np.isnan(skewness):
                    possibilities = possibilities.query(f"skew == {skewness} or skew.isnull()")

                for _, possibility in possibilities.iterrows():
                    team_wins[possibility['game_count']].append((possibility['win_count'], id))
                    
            print(team_wins)
            print(f"=== {team} {year} ===")
            win_count = len(train_team_year.query(f"(home_team_abbr == '{team}' and home_team_win == True) or (away_team_abbr == '{team}' and home_team_win == False)"))
            if (train_team_year.iloc[-1]['home_team_abbr'] == team) ^ train_team_year.iloc[-1]['home_team_win']:
                win_count -= 1
            print(win_count)


            schedule = brute_mating(win_count, len(train_team_year), [], train_team_year, test_team_year, team_wins)
            if len(schedule) == 0:
                schedule = brute_mating(win_count + 1, len(train_team_year), [], train_team_year, test_team_year, team_wins)
                if len(schedule) != 0:
                    win_count += 1
            if len(schedule) == 0:
                schedule = brute_mating(win_count - 1, len(train_team_year), [], train_team_year, test_team_year, team_wins)
                if len(schedule) != 0:
                    win_count -= 1
            print(len(test_team_year), len(schedule))
            
            prev_win = win_count
            prev_id = None
            for win, id in reversed(schedule):
                if prev_id is None:
                    prev_win = win
                    prev_id = id
                    continue
                if (win > prev_win) ^ (test_team_year.loc[prev_id]['home_team_abbr'] == team):
                    res = False
                else:
                    res = True
                if prev_id in home_team_win:
                    if home_team_win[prev_id] != res:
                        if len(test_team_year) == len(schedule):
                            home_team_win[prev_id] = res        
                else:
                    home_team_win[prev_id] = res
                prev_win = win
                prev_id = id
  
if __name__ == '__main__':
    main()


   
    
  

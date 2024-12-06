import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from liblinear.liblinearutil import *

# Preprocess data
def process_func(file_path):
    # Read data
    data = pd.read_csv(file_path)

    
    # Select features
    key_words_to_drop = ['abbr', 'date', 'id', 'rest', 'night']
    column_to_drop = data.columns[data.columns.str.contains('|'.join(key_words_to_drop), regex=True)]
    data = data.drop(column_to_drop, axis=1)
    data = data.drop(columns=['season', 'home_team_season'])

    # Boolean columns converted to 1/0
    if 'home_team_win' in data.columns:
        data['home_team_win'] = data['home_team_win'].astype(int)
        
    # Handle missing values of numeric columns with mean (median can also be used)
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

    # Handle missing values of categorical columns with 'Unknown'
    categorical_columns = data.select_dtypes(include=['object']).columns
    data[categorical_columns] = data[categorical_columns].fillna('Unknown')

    # Drop the columns without 'mean'、'std'、'average'、'win'
    key_words_to_select = ['mean', 'std', 'average', 'win']
    column_to_select = data.columns[data.columns.str.contains('|'.join(key_words_to_select), regex=True)]
    data = data[column_to_select]

    # Store the processed data
    processed_file_path = 'C:/Users/user/Desktop/NTU_myHW/final/processed_data.csv'
    data.to_csv(processed_file_path, index=False)
    print("Data process completed")
    
    
    return data

# Train the model
def train_func(y_train, X_train, lambdas):
    for lambda_val in lambdas:
        C_val = 1 / lambda_val
        parameter_str = f'-s 6 -c {C_val} -q'
        model = train(y_train, X_train, parameter_str)
    return model

# Cross validation
def cross_validation(y_train, X_train, lambdas):
    for lambda_val in lambdas:
        C_val = 1 / lambda_val
        parameter_str = f'-s 6 -c {C_val} -v 5 -q'
        Eval = train(y_train, X_train, parameter_str)
    return Eval

def main():
    train_file_path = 'C:/Users/user/Desktop/NTU_hw/final/html-2024-fall-final-project-stage-1/train_data.csv'
    #test_file_path = 'C:/Users/user/Desktop/NTU_hw/final/html-2024-fall-final-project-stage-1/same_season_test_data.csv'

    processed_train_data = process_func(train_file_path)

    # Split data into features and target
    y_train = np.array(processed_train_data['home_team_win'])
    X_train = np.array(processed_train_data.drop(columns=['home_team_win']))

    print(X_train)
    print(len(X_train), len(y_train))
    
    print()
    # Set the initial parameters
    lambdas = [10**(-2), 10**(-1), 1, 10, 100, 1000]
    
    
    

    #model = train_func(y_train, X_train, lambdas)
    Eval = cross_validation(y_train, X_train, lambdas)

    #print(Eval)
      


if __name__ == '__main__':
    main()
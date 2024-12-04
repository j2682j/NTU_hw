from multiprocessing import process
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from liblinear.liblinearutil import *
import test

# Preprocess data
def process_func(file_path):
    # Read data
    data = pd.read_csv(file_path)

    # Select features
    key_words_to_drop = ['abbr', 'away', 'date', 'id', 'rest', 'night']
    column_to_drop = data.columns[data.columns.str.contains('|'.join(key_words_to_drop), regex=True)]
    data = data.drop(column_to_drop, axis=1)
    data = data.drop(columns=['home_pitcher', 'season'])

    # Boolean columns converted to 1/0
    data['home_team_win'] = data['home_team_win'].fillna(0).astype(int)

    # Handle missing values of numeric columns with mean (median can also be used)
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

    # Handle missing values of categorical columns with 'Unknown'
    categorical_columns = data.select_dtypes(include=['object']).columns
    data[categorical_columns] = data[categorical_columns].fillna('Unknown')

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

# Evaluate the model
def evaluate_func(y_test, X_test, model):
    _, Eout, _ = predict(y_test, X_test, model, '-q')
    return Eout

def main():
    train_file_path = 'C:/Users/user/Desktop/NTU_hw/final/html-2024-fall-final-project-stage-1/train_data.csv'
    test_file_path = 'C:/Users/user/Desktop/NTU_hw/final/html-2024-fall-final-project-stage-1/same_season_test_data.csv'

    processed_train_data = process_func(train_file_path)
    processed_test_data = process_func(test_file_path)

    # Split data into features and target
    X_train = processed_train_data['home_team_win']
    y_train = np.array(processed_train_data.drop(columns=['home_team_win']))

    X_test = processed_test_data['home_team_win']
    y_test = np.array(processed_test_data.drop(columns=['home_team_win']))

    # Set the initial parameters
    lambdas = [10**(-2), 10**(-1), 1, 10, 100, 1000]
    
    for i in range(10):

        model_trained = train_func(y_train, X_train, lambdas)

        Eout = evaluate_func(y_test, X_test, model_trained)

        print(Eout)


if __name__ == '__main__':
    main()
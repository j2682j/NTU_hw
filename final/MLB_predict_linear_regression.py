from math import e
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

# Calculate the W_lin
def linear_regression(X, y):
    X_dagger = np.linalg.pinv(X)  
    return np.dot(X_dagger, y)    

# Calculate the error_out
def Error_measure(X, y, w):
    total_square_error = 0
    for i in range(len(X)):
        item = X[i]
        predictions = np.dot(item, w)
        square_error = (predictions - y[i]) ** 2
        total_square_error += square_error
    return total_square_error/len(X)


def main():
    train_file_path = 'C:/Users/user/Desktop/NTU_hw/final/html-2024-fall-final-project-stage-1/train_data.csv'
    #test_file_path = 'C:/Users/user/Desktop/NTU_hw/final/html-2024-fall-final-project-stage-1/same_season_test_data.csv'

    processed_train_data = process_func(train_file_path)

    # Split data into features and target
    y = np.array(processed_train_data['home_team_win'])
    X = np.array(processed_train_data.drop(columns=['home_team_win']))

    # Split data into training and testing
    N_values = range(25, 2001, 25)
    total_examples = len(X)
    iterarion_perN = 16
    Eout_average = []
    for N in N_values:
        Eout_N_Sum = 0
        for k in range(100):
            random_indices = np.random.choice(total_examples, N, replace=False)
            X_train, y_train = X[random_indices], y[random_indices]

            test_indices = [i for i in range(total_examples) if i not in random_indices]
            X_test, y_test = X[test_indices], y[test_indices]

            # Linear regression
            w_lin = linear_regression(X_train, y_train)

            # Error measure
            Eout = 100 - Error_measure(X_test, y_test, w_lin)
            Eout_N_Sum += Eout

            
    Eout_average.append(Eout_N_Sum / iterarion_perN)

    plt.plot(N_values, Eout_average, label='Avg.Eout(N)')
    plt.xlabel('N')
    plt.ylabel('Error')
    plt.title('Avg.Eout(N) over 16 experiments')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
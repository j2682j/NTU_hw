from libsvm.svmutil import *
import numpy as np
import ctypes
import matplotlib.pyplot as plt
import pandas as pd

# Read the data
def load_and_prepare_data(file_path):
    """
    Load the train_data.csv

    Args:
        file_path (str): Path to the train_data.csv.

    Returns:
        tuple: (features, labels) for the data.
    """
    data = pd.read_csv(file_path)

    # Select features
    key_words_to_drop = ['abbr', 'date', 'id', 'rest', 'night']
    column_to_drop = data.columns[data.columns.str.contains('|'.join(key_words_to_drop), regex=True)]
    data = data.drop(column_to_drop, axis=1)
    data = data.drop(columns=['season', 'home_team_season', 'away_team_season','home_pitcger', 'away_pitcher'])

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
    '''
    key_words_to_select = ['mean', 'std', 'average', 'win']
    column_to_select = data.columns[data.columns.str.contains('|'.join(key_words_to_select), regex=True)]
    data = data[column_to_select]
    '''
    # Store the processed data
    processed_file_path = 'C:/Users/user/Desktop/NTU_myHW/final/processed_data.csv'
    data.to_csv(processed_file_path, index=False)
    print("Data process completed")

    y_train = np.array(data['home_team_win'])
    X = np.array(data.drop(columns=['home_team_win']))
    X_train = [{i+1 : X[i] for i in range(len(X))}]
    print(X_train)
    
    return X_train,y_train

def convert_sv_to_matrix(sv, feature_dim):
    """
    Convert the list of dictionaries (support vectors) to a NumPy matrix.

    Args:
        sv (list of dict): Support vectors in dictionary format.
        feature_dim (int): Total number of features.

    Returns:
        np.ndarray: Support vectors as a NumPy matrix.
    """
    sv_matrix = np.zeros((len(sv), feature_dim))
    for i, vec in enumerate(sv):
        for key, value in vec.items():
            sv_matrix[i, key - 1] = value  
    return sv_matrix
    
def calculate_gaussian_kernel(sv_matrix, gamma):
    """
    Calculate the Gaussian kernel for the support vectors.

    Args:
        sv (list): Support vectors.
        gamma (float): Gamma value.

    Returns:
        np.ndarray: Kernel matrix.
    """
    kernel_matrix = np.zeros((len(sv_matrix), len(sv_matrix)))
    for i in range(len(sv_matrix)):
        for j in range(len(sv_matrix)):
            kernel_matrix[i, j] = np.exp(-gamma * np.linalg.norm(sv_matrix[i] - sv_matrix[j]) ** 2)
    return kernel_matrix


# Find the number of support vectors for each combination of C and Q
def train_SVM(X, y, C_values, gamma_values):
    """
    Train SVM models for each combination of C and Q, and count support vectors.

    Args:
        X (list): Feature vectors.
        y (list): Labels.
        C_values (list): List of C values to test.
        gamma_values (list): List of gamma values to test.

    Returns:
        dict: Results with (C, gamma) as keys and margin as values.
    """
    results = {
        (C, gamma): None for C in C_values for gamma in gamma_values
    }
    
    feature_dim = max(max(x.keys()) for x in X)

    for C in C_values:
        for gamma in gamma_values:
            
            param = f"-s 0 -t 2 -c {C} -g {gamma} -q"

            model = svm_train(y, X, param)
            
            # Retrieve support vector coefficients (sv_coef) and support vectors (SV)
            sv_coef = np.array(model.get_sv_coef())  # Dual coefficients
            sv_indices = model.get_sv_indices()
            sv_labels = np.array([y[i - 1] for i in sv_indices])
            sv = model.get_SV()
            sv_matrix = convert_sv_to_matrix(sv, feature_dim)
            kernel_matrix = calculate_gaussian_kernel(sv_matrix, gamma)

            # Calculate the w & margin
            w = np.sum(sv_coef * sv_labels * kernel_matrix, axis=0)

            margin = 1 / np.linalg.norm(w)

            print(f"margin : {margin}")

            results[(C, gamma)] = round(margin)

    return results



def main():
    # Step 1: Load and prepare data
    data_set_path = "C:/Users/user/Desktop/NTU_hw/final/html-2024-fall-final-project-stage-1/train_data.csv" 
    X_train, y_train = load_and_prepare_data(data_set_path)

    # Step 2: Define parameter values
    C_values = [0.1, 1, 10]
    gamma_values = [0.1, 1, 10]

    # Step 3: Count Support Vectors
    #results_list = train_SVM(X_train, y_train, C_values, gamma_values)
   


if __name__ == "__main__":
    main()


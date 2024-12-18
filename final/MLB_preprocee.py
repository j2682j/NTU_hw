from operator import inv
from re import X
from matplotlib import category
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def load_and_prepare_data(file_path):
    """
    Load the train_data.csv

    Args:
        file_path (str): Path to the train_data.csv.

    Returns:
        data: replae the outlier with mean.
    """
    data = pd.read_csv(file_path)

    data = data.drop(columns = ['season', 'id', 'date'])
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

    # Try to normalize the data
    # Replace the outlier with mean
    for column in numeric_columns:
        mean = data[column].mean()
        std = data[column].std()
        threshold = 3
        for value in data[column]:
            if (abs(value - mean) > threshold * std) or (abs(value - mean) < -threshold * std):
                data[column] = data[column].replace(value, mean)
    
    #Apply PCA
    X_features = data[numeric_columns]

    for i in range(1, len(X_features.columns)+1):
        pca = PCA(n_components = i)
        new_X = pca.fit_transform(X_features[0:i])
        print(f'{i}-components : {pca.explained_variance_ratio_}')
    

    
    category_columns = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=category_columns)
    

    # Store the processed data
    processed_file_path = 'C:/Users/user/Desktop/NTU_myHW/final/processed_data.csv'
    data.to_csv(processed_file_path, index=False)
    print("Data process completed")

    return data

#def train_model(X_train, y_train):

def main():
    file_path = 'C:/Users/user/Desktop/NTU_hw/final/html-2024-fall-final-project-stage-1/train_data.csv'
    data = load_and_prepare_data(file_path)

if __name__ == '__main__':
    main()








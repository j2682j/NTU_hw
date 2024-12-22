from matplotlib.pylab import f
import numpy as np
import matplotlib.pyplot as plt
from liblinear.liblinearutil import *
import test
import pandas as pd
from sklearn.model_selection import KFold
  
def retrieve_column(train_file_path, test_file_path):
    
    # Load the train data
    unprocessed_train_data = pd.read_csv(train_file_path, index_col = 'id')
    unprocessed_test_data = pd.read_csv(test_file_path, index_col = 'id')
    
    # Get all the columns containing the word 'mean' in the train data
    mean_columns = [col for col in unprocessed_train_data.columns if 'mean' in col]
    _10RA_colums = []
    X_10RA_mean_train = np.zeros((len(unprocessed_train_data), len(mean_columns)))
    X_all_mean_train = np.zeros((len(unprocessed_train_data), len(mean_columns)))
    for col in mean_columns:
        if col.find('10RA') != -1:
            _10RA_colums.append(col)
            unprocessed_10RA_mean_train_data = unprocessed_train_data[_10RA_colums]
            processed_10RA_mean_train_data = unprocessed_10RA_mean_train_data.fillna(unprocessed_10RA_mean_train_data.mean())
            X_10RA_mean_train = np.array(processed_10RA_mean_train_data)           
        else:
            unprocessed_mean_train_data = unprocessed_train_data[mean_columns]
            processed_mean_train_data = unprocessed_mean_train_data.fillna(unprocessed_mean_train_data.mean())
            X_all_mean_train = np.array(processed_mean_train_data)
    
    # Get all the columns containing the word 'mean' in the test data
    mean_test_columns = [col for col in unprocessed_test_data.columns if 'mean' in col]
    _10RA_test_colums = []
    X_10RA_mean_test = np.zeros((len(unprocessed_test_data), len(mean_test_columns)))
    X_all_mean_test = np.zeros((len(unprocessed_test_data), len(mean_test_columns)))
    for col in mean_test_columns:
        if col.find('10RA') != -1:
            _10RA_test_colums.append(col)
            unprocessed_10RA_mean_test_data = unprocessed_test_data[_10RA_test_colums]
            processed_10RA_mean_test_data = unprocessed_10RA_mean_test_data.fillna(unprocessed_10RA_mean_test_data.mean())
            X_10RA_mean_test = np.array(processed_10RA_mean_test_data)           

        else:
            unprocessed_mean_test_data = unprocessed_test_data[mean_test_columns]
            processed_mean_test_data = unprocessed_mean_test_data.fillna(unprocessed_mean_test_data.mean())
            X_all_mean_test = np.array(processed_mean_test_data)

    # Get all the columns containing the word 'std'
    std_columns = [col for col in unprocessed_train_data.columns if 'std' in col]
    _10RA_colums = []
    X_10RA_std_train = np.zeros((len(unprocessed_train_data), len(std_columns)))
    X_all_std_train = np.zeros((len(unprocessed_train_data), len(std_columns)))
    for col in std_columns:
        if col.find('10RA') != -1:
            _10RA_colums.append(col)
            unprocessed_10RA_std_train_data = unprocessed_train_data[_10RA_colums]
            processed_10RA_std_train_data = unprocessed_10RA_std_train_data.fillna(unprocessed_10RA_std_train_data.mean())
            X_10RA_std_train = np.array(processed_10RA_std_train_data)
        else:
            unprocessed_std_train_data = unprocessed_train_data[std_columns]
            processed_std_train_data = unprocessed_std_train_data.fillna(unprocessed_std_train_data.mean())
            X_all_std_train = np.array(processed_std_train_data)

    
    # Get all the columns containing the word 'std' in the test data
    std_test_columns = [col for col in unprocessed_test_data.columns if 'std' in col]
    _10RA_test_colums = []
    X_10RA_std_test = np.zeros((len(unprocessed_test_data), len(std_test_columns)))
    X_all_std_test = np.zeros((len(unprocessed_test_data), len(std_test_columns)))
    for col in std_test_columns:
        if col.find('10RA') != -1:
            _10RA_test_colums.append(col)
            unprocessed_10RA_std_test_data = unprocessed_test_data[_10RA_test_colums]
            processed_10RA_std_test_data = unprocessed_10RA_std_test_data.fillna(unprocessed_10RA_std_test_data.mean())
            X_10RA_std_test = np.array(processed_10RA_std_test_data)
        else:
            unprocessed_std_test_data = unprocessed_test_data[std_test_columns]
            processed_std_test_data = unprocessed_std_test_data.fillna(unprocessed_std_test_data.mean())
            X_all_std_test = np.array(processed_std_test_data)
       

    # Get all the columns containing the word 'skew' in the train data
    skew_columns = [col for col in unprocessed_train_data.columns if 'skew' in col]
    _10RA_colums = []
    X_10RA_skew_train = np.zeros((len(unprocessed_train_data), len(skew_columns)))
    X_all_skew_train = np.zeros((len(unprocessed_train_data), len(skew_columns)))
    for col in skew_columns:
        if col.find('10RA') != -1:
            _10RA_colums.append(col)
            unprocessed_10RA_skew_train_data = unprocessed_train_data[_10RA_colums]
            processed_10RA_skew_train_data = unprocessed_10RA_skew_train_data.fillna(unprocessed_10RA_skew_train_data.mean())
            X_10RA_skew_train = np.array(processed_10RA_skew_train_data)
        else:
            unprocessed_skew_train_data = unprocessed_train_data[skew_columns]
            processed_skew_train_data = unprocessed_skew_train_data.fillna(unprocessed_skew_train_data.mean())
            X_all_skew_train = np.array(processed_skew_train_data)
        
    
    # Get all the columns containing the word 'skew' in the test data
    skew_test_columns = [col for col in unprocessed_test_data.columns if 'skew' in col]
    _10RA_test_colums = []
    X_10RA_skew_test = np.zeros((len(unprocessed_test_data), len(skew_test_columns)))
    X_all_skew_test = np.zeros((len(unprocessed_test_data), len(skew_test_columns)))
    for col in skew_test_columns:
        if col.find('10RA') != -1:
            _10RA_test_colums.append(col)
            unprocessed_10RA_skew_test_data = unprocessed_test_data[_10RA_test_colums]
            processed_10RA_skew_test_data = unprocessed_10RA_skew_test_data.fillna(unprocessed_10RA_skew_test_data.mean())
            X_10RA_skew_test = np.array(processed_10RA_skew_test_data)
        else:
            unprocessed_skew_test_data = unprocessed_test_data[skew_test_columns]
            processed_skew_test_data = unprocessed_skew_test_data.fillna(unprocessed_skew_test_data.mean())
            X_all_skew_test = np.array(processed_skew_test_data)     


    # Combine the 10RA mean std skew data 
    X_10RA_train = np.concatenate((X_10RA_mean_train, X_10RA_std_train, X_10RA_skew_train), axis = 1)
    X_10RA_test = np.concatenate((X_10RA_mean_test, X_10RA_std_test, X_10RA_skew_test), axis = 1)

    # Combine all the mean std skew data
    X_all_train = np.concatenate((X_all_mean_train, X_all_std_train, X_all_skew_train), axis = 1)
    X_all_test = np.concatenate((X_all_mean_test, X_all_std_test, X_all_skew_test), axis = 1)

    y_train = unprocessed_train_data['home_team_win'].astype(int)

    return X_10RA_train, X_10RA_test, X_all_train, X_all_test, y_train

# Decision stump algorithm for multi-dimensional data
def decision_stump_multidim(X_train, y_train, sample_weights):
    """
    input : 
        X_train : training data, shape = (N, d)
        y_train : training labels, shape = (N,)
        sample_weights : sample weights, shape = (N,)
    output :
        best_stump : the best decision stump found, a dictionary
        best_predictions : the predictions of the best decision stump    
    """
    d = X_train.shape[1]
    min_error = 1
    best_predictions = None
    best_threshold = None
    best_s = None
    for i in range(d):
        # Sort the data by the i-th feature
        feature_values = X_train[:, i]
        sorted_indices = np.argsort(feature_values)
        sorted_X_train = X_train[sorted_indices]

        # Calculate the thresholds(mean of two adjacent points) for the i-th feature
        thresholds = (sorted_X_train[:-1, i] + sorted_X_train[1:, i]) / 2

        # Try s = {1, -1}
        for s in [1, -1]:
            for threshold in thresholds:
                predictions = s * np.sign(feature_values - threshold)
                E_u_in = np.average(predictions != y_train, weights=sample_weights)                 
                if E_u_in < min_error:
                    min_error = E_u_in
                    best_s = s
                    best_threshold = threshold
                    best_predictions = predictions
    return best_predictions, best_threshold, best_s
    
# AdaBoost algorithm with multi-dimensional decision stumps
def adaboost_stump(X_train, y_train, sample_weights, T = 1):
    """
    input :
        X : training data, shape = (N, d)
        y : training labels, shape = (N,)
        sample_weights : sample weights, shape = (N,)
        T : number of iterations
    output :
        epsilon_list : list of epsilon_t
        ein_list : list of E_in(t)
    """
    N_train = len(y_train)
    U_list = []
    

    # Initialize the cumulative g_t(x)
    G_train = np.zeros(N_train)

    # Initialize the cumulative U_t
    U_t = np.zeros(N_train)

    for t in range(T):
        # Step 1: Train multi-dimensional decision stump with weighted data
        predictions, best_threshold, best_s = decision_stump_multidim(X_train, y_train, sample_weights)
        
        # Step 2: Update the weights
        epsilon_t = np.sum(sample_weights * (predictions != y_train)) / np.sum(sample_weights)
        
        diamond_t = np.sqrt((1 - epsilon_t) / epsilon_t)
        sample_weights = np.where(predictions != y_train, sample_weights * diamond_t, sample_weights / diamond_t)

        # Calculate U_t
        U_t += np.sum(sample_weights)
        
        U_list.append(U_t)
        

        # Step 3: Calculate the alpha
        alpha_t = np.log(diamond_t)   
    
        # Step 4: Update the cumulative g_t(x)
        G_train += alpha_t * predictions

        # Output the prediction of the final hypothesis
        print(f'Iteration {t + 1} best_threshold : {best_threshold}')
    
    return alpha_t, best_threshold, best_s


def main():
    train_path = 'C:/Users/user/Desktop/NTU_hw/final/html-2024-fall-final-project-stage-1/train_data.csv'
    test_path = 'C:/Users/user/Desktop/NTU_hw/final/html-2024-fall-final-project-stage-1/same_season_test_data.csv'

    _, _, X_train, X_test, y_train = retrieve_column(train_path, test_path)
    
    G = np.zeros(len(y_train))

    '''
    sample_weights = np.ones(len(X_train)) / len(X_train)


    alpha_t, best_threshold, best_s = adaboost_stump(X_train, y_train, sample_weights, T = 1)
    
    for i in range(X_test.shape[1]):
            G += alpha_t * best_s * np.sign(X_test[:, i] - best_threshold)
    '''
    Eval_list = {}
    best_Eval = 1
    best_threshold = 0
    best_s = 0
    best_alpha_t = 0
    for t in range(50, 300, 50):
        kfold = KFold(n_splits = 5, shuffle = True)
        for (train_index, valid_index) in kfold.split(X_train, y_train):
            X_train_train, X_train_valid = X_train[train_index], X_train[valid_index]
            y_train_train, y_train_valid = y_train[train_index], y_train[valid_index]
            sample_weights = np.ones(len(X_train_train)) / len(X_train_train)
            alpha_t, best_threshold, best_s = adaboost_stump(X_train_train, y_train_train, sample_weights, t = 1)
            for i in range(X_train_valid.shape[1]):
                G += alpha_t * best_s * np.sign(X_train_valid[:, i] - best_threshold)
                Eval = np.mean(np.sign(G) != y_train_valid)
                Eval_list[t] = Eval
                if Eval < best_Eval:
                    best_Eval = Eval
                    best_threshold = best_threshold
                    best_s = best_s
                    best_alpha_t = alpha_t
    print(f"best_Eval : {best_Eval}, best_threshold : {best_threshold}, best_s : {best_s}, best_alpha_t : {best_alpha_t}")

    for i in range(X_test.shape[1]):
            G += best_alpha_t * best_s * np.sign(X_test[:, i] - best_threshold)
    
    with open("C:/Users/user/Desktop/NTU_hw/final/result_all.csv", 'w') as f:
        f.write("id,home_team_win\n")
        for i in range(len(G)):
            f.write(f"{i},{(G[i])}\n")


if __name__ == '__main__':
    main()
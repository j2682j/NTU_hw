import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
import pandas as pd
from sklearn.model_selection import KFold
#Read data
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

def find_best_stump(X, y, weights):
    n_samples, n_dimension = X.shape
    best_stump = {"dimension": None, "threshold": None, "direction": 1, "error": float("inf")}
    
    for dimension in range(n_dimension):
        thresholds = np.unique(X[:, dimension])
        for threshold in thresholds:
            for direction in [1, -1]:
                prediction = np.ones(n_samples)
                if direction == 1:
                    prediction[X[:, dimension] < threshold] = -1
                else:
                    prediction[X[:, dimension] >= threshold] = -1
                
                weighted_error = np.sum(weights[y != prediction])/np.sum(weights)
                
                if weighted_error < best_stump["error"]:
                    best_stump["dimension"] = dimension
                    best_stump["threshold"] = threshold
                    best_stump["direction"] = direction
                    best_stump["error"] = weighted_error
    
    return best_stump

def stump_predict(X, stump):
    dimension, threshold, direction = stump["dimension"], stump["threshold"], stump["direction"]
    prediction = np.ones(X.shape[0])
    if direction == 1:
        prediction[X[:, dimension] < threshold] = -1
    else:
        prediction[X[:, dimension] >= threshold] = -1
    return prediction

def adaboost(X, y, T = 5):
    n_samples, n_dimension = X.shape
    weights = np.ones(n_samples) / n_samples
    stumps = []
    alphas = []
    errors = []
    sum_of_weights = []
    
    for t in range(T):
        sum_of_weights.append(np.sum(weights))

        # Find best stump
        stump = find_best_stump(X, y, weights)
        prediction = stump_predict(X, stump)

        # Calculate alpha (stump weight)
        epsilon = stump["error"]
        alpha = 0.5 * np.log((1 - epsilon) / (epsilon + 1e-10))
        
        # Update weights
        weights *= np.exp(-alpha * y * prediction)
        
        # Save stump and alpha
        stumps.append(stump)
        alphas.append(alpha)
        errors.append(epsilon)
        print(t)
    
    return stumps, alphas, errors, sum_of_weights

def calculate_error(X, y, stumps, alphas):
    n_samples = len(y)
    final_prediction = np.zeros(n_samples)
    for stump, alpha in zip(stumps, alphas):
        final_prediction += alpha * stump_predict(X, stump)
    final_prediction = np.sign(final_prediction)
    error = np.mean(final_prediction != y)
    return error

#train
def main():
    train_path = 'C:/Users/user/Desktop/NTU_hw/final/html2024-fall-final-project-stage-2/train_data.csv'
    test_path = 'C:/Users/user/Desktop/NTU_hw/final/html2024-fall-final-project-stage-2/2024_test_data.csv'

    _, _, X_train, X_test, y_train = retrieve_column(train_path, test_path)
    
    '''
    # Scale the training data, Scalibility = 0.75
    random_indices = np.random.choice(len(X_train), int(0.75 * len(X_train)), replace=False)
    X_train_scale = X_train[random_indices]
    y_train_scale = y_train[random_indices]
    '''

    best_Eval = 1
    best_stumps = None
    best_alphas = None
    Eval_t = []
   
    kfold = KFold(n_splits = 5, shuffle = True)
    for (train_index, valid_index) in kfold.split(X_train, y_train):
        X_train_train, X_valid = X_train[train_index], X_train[valid_index]
        y_train_train, y_valid = y_train[train_index], y_train[valid_index]
        stumps, alphas, _, _ = adaboost(X_train_train, y_train_train, T = 5)
        Eval = calculate_error(X_valid, y_valid, stumps, alphas)
        Eval_t.append(Eval)
        print(f"T = {T}, Eval = {Eval}")
        if Eval < best_Eval:
            best_stumps = stumps
            best_alphas = alphas
            best_Eval = Eval
    print(f"Best Eval = {best_Eval}")
    n_samples = len(y_train)

    final_prediction = np.zeros(n_samples)
    for stump, alpha in zip(best_stumps, best_alphas):
        final_prediction += alpha * stump_predict(X_test, stump)
    final_prediction = np.sign(final_prediction)
   
    with open("C:/Users/user/Desktop/NTU_hw/final/result_all.csv", 'w') as f:
        f.write("id,home_team_win\n")
        for i in range(len(final_prediction)):
            f.write(f"{i},{(final_prediction[i])}\n")
    
    Eval_avg = np.mean(Eval_t)

    plt.figure(figsize=(10, 6))
    plt.plot(Eval_t, label="Eval")
    plt.plot(Eval_avg, label="Eval_avg")
    plt.xlabel("Iterations (t)")
    plt.ylabel("Error")
    plt.title("Eval vs Iterations")
    plt.legend()
    plt.grid(True)
    plt.show()
        
    



if __name__ == '__main__':
    main()




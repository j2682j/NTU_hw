from matplotlib.pylab import f
import numpy as np
import matplotlib.pyplot as plt
from liblinear.liblinearutil import *

  
def load_libsvm_data(file_path):
    y, X = svm_read_problem(file_path)  
    features = [] 
    for item in X:
        for _, value in item.items():
            features.append(value)         
    X = np.array(features).reshape(len(y), -1)
    y = np.array(y)
    return y, X


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
def adaboost_stump(X_train, y_train, X_test, y_test,sample_weights, T = 500):
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
    N_train, N_test = len(y_train), len(y_test)
    Eout_list = []
    Ein_list = []

    # Initialize the cumulative g_t(x)
    G_train = np.zeros(N_train)
    G_test = np.zeros(N_test)

    for t in range(T):
        # Step 1: Train multi-dimensional decision stump with weighted data
        predictions, best_threshold, best_s = decision_stump_multidim(X_train, y_train, sample_weights)
        
        # Step 2: Update the weights
        epsilon_t = np.sum(sample_weights * (predictions != y_train)) / np.sum(sample_weights)
        
        diamond_t = np.sqrt((1 - epsilon_t) / epsilon_t)
        sample_weights = np.where(predictions != y_train, sample_weights * diamond_t, sample_weights / diamond_t)
       
        # Step 3: Calculate the alpha
        alpha_t = np.log(diamond_t)   
    
        # Step 4: Update the cumulative g_t(x)
        G_train += alpha_t * predictions
        for i in range(X_test.shape[1]):
            G_test += alpha_t * best_s * np.sign(X_test[:, i] - best_threshold)

        # Calculate E_in(t) and E_out(t)
        Ein_t = np.mean(np.sign(G_train) != y_train)
        Eout_t = np.mean(np.sign(G_test) != y_test)

        Ein_list.append(Ein_t)
        Eout_list.append(Eout_t)

        print(f'Iteration{t+1} : E_in(t) = {Ein_t:.4f}, E_out(t) = {Eout_t:.4f}')
    return Ein_list, Eout_list



def plot(Ein_list, Eout_list):

    t_values = np.arange(1, len(Ein_list) + 1)  
    
    plt.plot(t_values, Ein_list, label=r'$E_{\text{in}}(G_t)$ (Training Error)', color='blue')
    plt.plot(t_values, Eout_list, label=r'$E_{\text{out}}(G_t)$ (Testing Error)', color='green')
    
    plt.title(r'Comparison of $E_{\text{in}}(G_t)$ and $E_{\text{out}}(G_t)$ over Iterations')
    plt.xlabel('Iteration t')
    plt.ylabel('Error')
    plt.legend()
    plt.grid()
    plt.show()


def main():
    train_path = 'C:/Users/user/Desktop/NTU_myHW/HW7/madelon.t'
    test_path = 'C:/Users/user/Desktop/NTU_myHW/HW7/madelon.t'

    y_train, X_train = load_libsvm_data(train_path)
    y_test, X_test = load_libsvm_data(test_path)

    sample_weights = np.ones(len(X_train)) / len(X_train)


    Ein_list, Eout_list = adaboost_stump(X_train, y_train, X_test, y_test, sample_weights, T = 500)

    plot(Ein_list, Eout_list)


if __name__ == '__main__':
    main()
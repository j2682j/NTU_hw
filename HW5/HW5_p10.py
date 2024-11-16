from liblinear.liblinearutil import *
import numpy as np
import ctypes
import matplotlib.pyplot as plt


# Load the data file
def load_data(filename, target_classes = [2, 6]):
    y, x = svm_read_problem(filename)
    y = np.array(y)
    x = np.array(x)

    # Filter out the data with target classes
    indices = np.where((y == target_classes[0]) | (y == target_classes[1]))[0]
    y = y[indices]
    x = x[indices]

    # Change the label to -1 and 1
    y = np.where(y == target_classes[0], -1, 1)

    return y, x


def main():
    # Read the training & testing data
    y_train, x_train = load_data('C:/Users/user/Desktop/NTU_myHW/HW5/mnist.scale.train/mnist.scale')
    y_test, x_test = load_data('C:/Users/user/Desktop/NTU_myHW/HW5/mnist.scale.test/mnist.scale.t')

    # Set the initial parameters
    lambdas = [10**(-2), 10**(-1), 1, 10, 100, 1000]  # the list of lambda
    lambda_star = 0
    best_Ein = 0
    best_Eout = 0
    g = []
    Eout_list = []
    weights = []

    for i in range(1126):# run 1126 times
        model_best = None                                               
        for lambda_val in lambdas:                                          # each lambda
            C_val = 1 / lambda_val
            parameter_str = f'-s 6 -c {C_val} -q'
    
            model = train(y_train, x_train, parameter_str)        
            _, Ein, _ = predict(y_train, x_train, model, '-q')
            
            if Ein[0] > best_Ein:
                best_Ein = Ein[0]                                           # update the best Ein  
                model_best = model                                          # update the best model                                                             
            elif Ein[0] == best_Ein and lambda_val > lambda_star:
                model_best = model
        
        _, Eout, _ = predict(y_test, x_test, model_best, '-q')              # calculate Eout with g
        best_Eout = (100 - Eout[0])                                         # update the best Eout
        Eout_list.append(best_Eout)
        weights = np.ctypeslib.as_array(model_best.w, shape=(model_best.nr_feature,)) # get the weights
        g.append(weights)


    # Count the number of non-zero in each g
    g_nonzero = []
    for vector in g:
        cnt = 0
        for element in vector:
            if element != 0:
                cnt += 1
        g_nonzero.append(cnt)


    # Display the results of Eout(g) and # of Non-zero in each g
    plt.figure(figsize=(8, 6))
    plt.hist(Eout_list, bins = None, color='skyblue', edgecolor='black')
    plt.title("Histogram of Eout(g)")
    plt.xlabel("Eout(g)")
    plt.ylabel("Frequency")
        
    plt.figure(figsize=(8, 6))
    plt.hist(g_nonzero, bins= None, color='skyblue', edgecolor='black')
    plt.title("Histogram of Non-zero in g")
    plt.xlabel("# of Non-zero in each g")
    plt.ylabel("Frequency")

    plt.show() 
if __name__ == '__main__':
    main()
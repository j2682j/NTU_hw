from liblinear.liblinearutil import *
import numpy as np
import ctypes
import matplotlib.pyplot as plt

# Load the data file
def load_data(filename):
    y, x = svm_read_problem(filename)
    y = np.array(y)
    x = np.array(x)
    return y, x

def main():
    # Set the initial parameters
    lambdas = [10**(-2), 10**(-1), 1, 10, 100, 1000]  # the list of lambda
    lambda_star = 0
    best_Eval = 0
    Eout_g_list = []

    for i in range(1126): # run 1126 times 

        # Read the training & testing data with random seed for indice
        y, x = load_data('C:/Users/user/Desktop/NTU_myHW/HW5/mnist.scale.train/mnist.scale')
        np.random.seed()
        random_indices = np.random.choice(len(y), 8000, replace=False)
        y_train, x_train = y[random_indices], x[random_indices]

        test_indices = np.setdiff1d(range(len(y)), random_indices)
        y_test, x_test = y[test_indices], x[test_indices]
        
        
        for lambda_val in lambdas: # each lambda
            C_val = 1 / lambda_val
            parameter_str = f'-s 6 -c {C_val} -q'
    
            model = train(y_train, x_train, parameter_str)        
            _, Eval, _ = predict(y_test, x_test, model, '-q')
            
            if Eval[0] > best_Eval:
                best_Eval = Eval[0]                      # update the best Ein   
                lambda_star = lambda_val                 # update the best lambda           
            elif Eval[0] == best_Eval and lambda_val > lambda_star:
                lambda_star = lambda_val
    
        model = train(y, x, f'-s 6 -c {1/lambda_star} -q')
        _, Eout_g, _ = predict(y, x, model, '-q')
        Eout_g_list.append(100 - Eout_g[0])
        print(Eout_g_list)


    # Display the results of Eout(g)
    plt.figure(figsize=(8, 6))
    plt.hist(Eout_g_list, bins = None, color='skyblue', edgecolor='black')
    plt.title("Histogram of Eout(g)")
    plt.xlabel("Eout(g)")
    plt.ylabel("Frequency")
    plt.show()     
if __name__ == '__main__':
    main()    


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


# Set the initial parameters
lambdas = [10**(-2), 10**(-1), 1, 10, 100, 1000]  # the list of lambda
lambda_star = 0
best_Ecv = 0
Eout_g_list = []

for i in range(3): # run 1126 times 

    # Read the training & testing data with random seed for indice
    y_train, x_train = load_data('C:/Users/user/Desktop/NTU_myHW/HW5/mnist.scale.train/mnist.scale')
    y_test, x_test = load_data('C:/Users/user/Desktop/NTU_myHW/HW5/mnist.scale.test/mnist.scale.t')  

    for lambda_val in lambdas:                              # each lambda
        C_val = 1 / lambda_val
        parameter_str = f'-s 6 -c {C_val} -v 3 -q'
   
        Ecv = train(y_train, x_train, parameter_str)        # 3-fold cross validation              
        
        if Ecv > best_Ecv:
            best_Ecv = Ecv                                  # update the best Ecv   
            lambda_star = lambda_val                        # update the best lambda           
        elif Ecv == best_Ecv and lambda_val > lambda_star:
            lambda_star = lambda_val
 
    model = train(y_train, x_train, f'-s 6 -c {1/lambda_star} -q')
    _, Eout_g, _ = predict(y_test, x_test, model, '-q')
    Eout_g_list.append(100 - Eout_g[0])
    print(Eout_g_list)


# Display the results of Eout(g)
plt.figure(figsize=(8, 6))
plt.hist(Eout_g_list, bins = None, color='skyblue', edgecolor='black')
plt.title("Histogram of Eout(g)")
plt.xlabel("Eout(g)")
plt.ylabel("Frequency")
plt.show()     

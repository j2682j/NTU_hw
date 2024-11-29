from libsvm.svmutil import *
import numpy as np
import ctypes
import matplotlib.pyplot as plt

# Read the data
def load_and_prepare_data(file_path, target_classes=[3, 7]):
    """
    Load the mnist.scale dataset and filter it for labels 3 and 7.

    Args:
        file_path (str): Path to the mnist.scale file.

    Returns:
        tuple: (features, labels) for the filtered data.
    """
    y, x = svm_read_problem(file_path)
    y = np.array(y)
    x = np.array(x)

    # Filter out the data with target classes
    indices = np.where((y == target_classes[0]) | (y == target_classes[1]))[0]
    y = y[indices]
    x = x[indices]

    # Change the label to -1 and 1
    y = np.where(y == target_classes[0], -1, 1)

    return y, x


# Find the number of support vectors for each combination of C and Q
def train_SVM(X, y, C_values, Q_values):
    """
    Train SVM models for each combination of C and Q, and count support vectors.

    Args:
        X (list): Feature vectors.
        y (list): Labels.
        C_values (list): List of C values to test.
        Q_values (list): List of Q values to test.

    Returns:
        dict: Results with (C, Q) as keys and number of support vectors as values.
    """
    results = {
        (C, Q): None for C in C_values for Q in Q_values
    }

    for C in C_values:
        for Q in Q_values:
            
            param = f"-s 0 -t 1 -g 1 -r 1 -c {C} -d {Q} -q"
            
            prob = svm_problem(y, X)

            model = svm_train(prob, param)

            # Get the number of support vectors
            support_vector_count = np.ctypeslib.as_ctypes(model.l)
            results[(C, Q)] = support_vector_count
            print(f"C={C}, Q={Q}, Support Vectors={support_vector_count}")
    return results


def main():
    # Step 1: Load and prepare data
    data_set_path = "C:/Users/user/Desktop/NTU_myHW/HW5/mnist.scale.train/mnist.scale" 
    X, y = load_and_prepare_data(data_set_path)

    # Step 2: Define parameter values
    C_values = [0.1, 1, 10]
    Q_values = [2, 3, 4]

    # Step 3: Count Support Vectors, Find min # of support vectors 
    results = train_SVM(X, y, C_values, Q_values)

    # Step 4: Find the minimum number of support vectors
    min_num_support_vectors = min(results.values())
    min = [key for key, value in min_num_support_vectors.items() if value == min_num_support_vectors]
    print(f"Minimum number of support vectors: {min_num_support_vectors} for C={min[0][0]} and Q={min[0][1]}")

    # Step 5: Plot the number of support vectors for each combination of C and Q
    


if __name__ == "__main__":
    main()

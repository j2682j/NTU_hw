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

# Plot the number of support vectors for each combination of C and Q
def plot_support_vectors(results):
    """
    Plots the number of support vectors for each (C, Q) combination.

    Args:
        results (dict): Results from train_SVM, where keys are (C, Q) tuples and values are the number of support vectors.
    """
    # Extract data for plotting
    C_values = [key[0] for key in results.keys()]
    Q_values = [key[1] for key in results.keys()]
    support_vector_counts = list(results.values())

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(C_values, Q_values, c=support_vector_counts, cmap='viridis', s=100)
    plt.colorbar(scatter, label='Number of Support Vectors')

    # Label axes and title
    plt.xlabel('C Values')
    plt.ylabel('Q Values')
    plt.title('Support Vector Counts for Different (C, Q) Combinations')

    # Annotate each point with its support vector count
    for i, count in enumerate(support_vector_counts):
        plt.text(C_values[i], Q_values[i], str(count), fontsize=9, ha='center', va='center', color='black')

    # Show plot
    plt.grid(True)
    plt.show()


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
    plot_support_vectors(results)

if __name__ == "__main__":
    main()

from libsvm.svmutil import *
import numpy as np
import ctypes
import matplotlib.pyplot as plt

# Read the data
def load_and_prepare_data(file_path):
    """
    Load the mnist.scale dataset and filter it for labels 3 and 7.

    Args:
        file_path (str): Path to the mnist.scale file.

    Returns:
        tuple: (features, labels) for the filtered data.
    """
    X, y = [], []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            label = int(parts[0])
            if label == 3 or label == 7:
                y.append(1 if label == 3 else -1)  # Map 3 -> 1, 7 -> -1
                features = {int(k): float(v) for k, v in (item.split(':') for item in parts[1:])}
                X.append(features)
    return X, y
    


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
            
            param = f"-s 0 -g {C} -q"

            model = svm_train(y, X, param)

            # Calculate 1/∥w∥, which is the margin, at the optimal solution with the kernel trick.
            # Retrieve ||w|| using the dual coefficients and kernel trick
            sv_coef = model.get_sv_coef().flatten()  # Dual coefficients
            kernel_matrix = np.dot(sv_coef.T, sv_coef)  # Simplified for RBF kernel

            norm_w = np.sqrt(kernel_matrix)  # ||w|| for RBF kernel
            margin = 1 / norm_w if norm_w > 0 else float('inf')  # Avoid division by zero
            results[(C, Q)] = margin

    return results

# Plot the number of support vectors for each combination of C and Q
def plot_margin(results_list):
    """
    Plots the number of support vectors for each (C, Q) combination.

    Args:
        results (dict): Results from train_SVM, where keys are (C, Q) tuples and values are the number of support vectors.
    """
    # Extract data for plotting
    C_values = [key[0] for key in results_list.keys()]
    Q_values = [key[1] for key in results_list.keys()]
    margin_list = list(results_list.values())

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(C_values, Q_values, c=margin_list, cmap='viridis', s=100)
    plt.colorbar(scatter, label='Value of margin')

    # Label axes and title
    plt.xlabel('C Values')
    plt.ylabel('Q Values')
    plt.title('Margin for Different (C, Q) Combinations')

    # Annotate each point with its support vector count
    for i, count in enumerate(margin_list):
        plt.text(C_values[i], Q_values[i], str(count), fontsize=9, ha='center', va='center', color='black')

    # Show plot
    plt.grid(True)
    plt.show()


def main():
    # Step 1: Load and prepare data
    data_set_path = "C:/Users/user/Desktop/NTU_myHW/HW5/mnist.scale.train/mnist.scale" 
    X_train, y_train = load_and_prepare_data(data_set_path)

    # Step 2: Define parameter values
    C_values = [0.1, 1, 10]
    Q_values = [2, 3, 4]

    # Step 3: Count Support Vectors
    results_list = train_SVM(X_train, y_train, C_values, Q_values)
    
    # Step 4: Find the minimum number of support vectors
    min_num_support_vectors = min(results_list.values())
    ans = {key: value for key, value in results_list.items() if value == min_num_support_vectors}
    

    # Step 5: Plot the number of support vectors for each combination of C and Q
    plot_margin(results_list)

if __name__ == "__main__":
    main()


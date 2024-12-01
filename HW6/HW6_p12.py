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

def convert_sv_to_matrix(sv, feature_dim):
    """
    Convert the list of dictionaries (support vectors) to a NumPy matrix.

    Args:
        sv (list of dict): Support vectors in dictionary format.
        feature_dim (int): Total number of features.

    Returns:
        np.ndarray: Support vectors as a NumPy matrix.
    """
    sv_matrix = np.zeros((len(sv), feature_dim))
    for i, vec in enumerate(sv):
        for key, value in vec.items():
            sv_matrix[i, key - 1] = value  
    return sv_matrix
    
def calculate_gaussian_kernel(sv_matrix, gamma):
    """
    Calculate the Gaussian kernel for the support vectors.

    Args:
        sv (list): Support vectors.
        gamma (float): Gamma value.

    Returns:
        np.ndarray: Kernel matrix.
    """
    kernel_matrix = np.zeros((len(sv_matrix), len(sv_matrix)))
    for i in range(len(sv_matrix)):
        for j in range(len(sv_matrix)):
            kernel_matrix[i, j] = np.exp(-gamma * np.linalg.norm(sv_matrix[i] - sv_matrix[j]) ** 2)
    return kernel_matrix


# Find the number of support vectors for each combination of C and Q
def train_SVM(X, y, C_values, gamma_values):
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
        (C, gamma): None for C in C_values for gamma in gamma_values
    }
    
    feature_dim = max(max(x.keys()) for x in X)

    for C in C_values:
        for gamma in gamma_values:
            
            param = f"-s 0 -t 2 -c {C} -g {gamma} -q"

            model = svm_train(y, X, param)
            
            # Retrieve support vector coefficients (sv_coef) and support vectors (SV)
            sv_coef = np.array(model.get_sv_coef())  # Dual coefficients
            sv_indices = model.get_sv_indices()
            sv_labels = np.array([y[i - 1] for i in sv_indices])
            sv = model.get_SV()
            sv_matrix = convert_sv_to_matrix(sv, feature_dim)
            kernel_matrix = calculate_gaussian_kernel(sv_matrix, gamma)

            # Calculate the w & margin
            w = 0
            for i in sv_coef:
                for j in sv_labels:
                    w += i * j * kernel_matrix

            margin = 1 / np.linalg.norm(w)

            print(f"margin : {margin}")

            results[(C, gamma)] = round(margin)

    return results


# Plot margin for each combination of C and gamma
def plot_margin(results_list):
    """
    Plots margin for each (C, gamma) combination.

    Args:
        results (dict): Results from train_SVM, where keys are (C, gamma) tuples and values are margin values.
    """
    # Extract data for plotting
    C_values = [key[0] for key in results_list.keys()]
    gamma_values = [key[1] for key in results_list.keys()]
    margin_list = list(results_list.values())

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(C_values, gamma_values, c=margin_list, cmap='viridis', s=100)
    plt.colorbar(scatter, label='Value of margin')

    # Label axes and title
    plt.xlabel('C Values')
    plt.ylabel('gamma Values')
    plt.title('Margin for Different (C, gamma) Combinations')

    # Annotate each point with its support vector count
    for i, value in enumerate(margin_list):
        plt.text(C_values[i], gamma_values[i], f"{value:.2e}", fontsize=9, ha='center', va='center', color='black')

    # Show plot
    plt.grid(True)
    plt.show()


def main():
    # Step 1: Load and prepare data
    data_set_path = "C:/Users/user/Desktop/NTU_myHW/HW5/mnist.scale.train/mnist.scale" 
    X_train, y_train = load_and_prepare_data(data_set_path)

    # Step 2: Define parameter values
    C_values = [0.1, 1, 10]
    gamma_values = [2, 3, 4]

    # Step 3: Count Support Vectors
    results_list = train_SVM(X_train, y_train, C_values, gamma_values)
    
    # Step 4: Plot margin for each combination of C and Q
    plot_margin(results_list)

if __name__ == "__main__":
    main()


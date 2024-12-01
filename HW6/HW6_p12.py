import random
from collections import Counter
from libsvm.svmutil import *
import numpy as np
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

def split_data(X, y, val_size=200):
    """
    Split the dataset into training and validation sets.

    Args:
        X (list): Feature vectors.
        y (list): Labels.
        val_size (int): Number of samples for the validation set.

    Returns:
        tuple: (X_train, y_train, X_val, y_val)
    """
    indices = list(range(len(X)))
    random.shuffle(indices)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    X_val = [X[i] for i in val_indices]
    y_val = [y[i] for i in val_indices]
    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]

    return X_train, y_train, X_val, y_val

def evaluate_model(model, X_val, y_val):
    """
    Evaluate the model on the validation set using 0/1 error.

    Args:
        model: Trained SVM model.
        X_val (list): Validation feature vectors.
        y_val (list): Validation labels.

    Returns:
        float: Validation error.
    """
    _, p_acc, _ = svm_predict(y_val, X_val, model, '-q')
    error = 100 - p_acc[0]  # 0/1 error as percentage
    return error

def validation_procedure(X, y, gamma_values, num_repeats=128, val_size=200):
    """
    Perform the validation procedure to choose the best gamma.

    Args:
        X (list): Feature vectors.
        y (list): Labels.
        gamma_values (list): List of gamma values to test.
        num_repeats (int): Number of repetitions for the procedure.
        val_size (int): Number of samples for the validation set.

    Returns:
        Counter: Frequency of each gamma being selected.
    """
    gamma_counts = Counter()

    for _ in range(num_repeats):
        X_train, y_train, X_val, y_val = split_data(X, y, val_size)
        best_gamma = None
        lowest_error = float('inf')

        for gamma in gamma_values:
            param = f"-s 0 -t 2 -c 1 -g {gamma} -q"
            model = svm_train(y_train, X_train, param)
            Eval = evaluate_model(model, X_val, y_val)

            if Eval < lowest_error or (Eval == lowest_error and (best_gamma is None or gamma < best_gamma)):
                lowest_error = Eval
                best_gamma = gamma

        gamma_counts[best_gamma] += 1

    return gamma_counts

def plot_gamma_selection(gamma_counts):
    """
    Plot the frequency of each gamma being selected.

    Args:
        gamma_counts (Counter): Frequency of each gamma being selected.
    """
    gamma_values = list(gamma_counts.keys())
    frequencies = list(gamma_counts.values())

    plt.figure(figsize=(8, 5))
    plt.bar(gamma_values, frequencies, color='skyblue', edgecolor='black')
    plt.xlabel('Gamma Values')
    plt.ylabel('Selection Frequency')
    plt.title('Gamma Selection Frequency')
    plt.xticks(gamma_values)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def main():
    # Load and prepare data
    data_set_path = "C:/Users/user/Desktop/NTU_myHW/HW5/mnist.scale.train/mnist.scale_small.txt"
    X, y = load_and_prepare_data(data_set_path)

    # Define gamma values
    gamma_values = [0.01, 0.1, 1, 10, 100]

    # Perform validation procedure
    gamma_counts = validation_procedure(X, y, gamma_values, num_repeats=128, val_size=200)

    # Plot the results
    plot_gamma_selection(gamma_counts)

if __name__ == "__main__":
    main()

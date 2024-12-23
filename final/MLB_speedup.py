import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import pandas as pd

# Read data
def retrieve_column(train_file_path, test_file_path):
    unprocessed_train_data = pd.read_csv(train_file_path, index_col='id')
    unprocessed_test_data = pd.read_csv(test_file_path, index_col='id')

    # Process train and test data for keywords ['mean', 'std', 'skew']
    def process_columns(data, keywords):
        columns = {kw: [col for col in data.columns if kw in col] for kw in keywords}
        processed_data = {kw: data[cols].fillna(data[cols].mean()) for kw, cols in columns.items() if cols}
        return processed_data

    train_processed = process_columns(unprocessed_train_data, ['mean', 'std', 'skew'])
    test_processed = process_columns(unprocessed_test_data, ['mean', 'std', 'skew'])

    # Combine all data
    def combine_data(processed_data):
        all_data_list = [np.array(processed_data[kw]) for kw in processed_data]

        # Ensure at least one array exists
        if all_data_list:
            all_data = np.concatenate(all_data_list, axis=1)
        else:
            all_data = np.empty((processed_data[next(iter(processed_data))].shape[0], 0))

        return all_data

    X_all_train = combine_data(train_processed)
    X_all_test = combine_data(test_processed)
    y_train = unprocessed_train_data['home_team_win'].astype(int)

    return X_all_train, X_all_test, y_train

# Find best decision stump
def find_best_stump(X, y, weights):
    n_samples, n_dimensions = X.shape
    best_stump = {"dimension": None, "threshold": None, "direction": 1, "error": float("inf")}

    for dimension in range(n_dimensions):
        thresholds = np.unique(X[:, dimension])
        for threshold in thresholds:
            for direction in [1, -1]:
                # 預測結果
                prediction = np.ones(n_samples)
                if direction == 1:
                    prediction[X[:, dimension] < threshold] = -1
                else:
                    prediction[X[:, dimension] >= threshold] = -1
                
                # 計算加權誤差
                weighted_error = np.sum(weights[y != prediction]) / np.sum(weights)

                # 更新最佳樹樁
                if weighted_error < best_stump["error"]:
                    best_stump.update({
                        "dimension": dimension,
                        "threshold": threshold,
                        "direction": direction,
                        "error": weighted_error
                    })

    return best_stump


# Predict using a single stump
def stump_predict(X, stump):
    dimension, threshold, direction = stump["dimension"], stump["threshold"], stump["direction"]
    prediction = np.ones(X.shape[0])
    if direction == 1:
        prediction[X[:, dimension] < threshold] = -1
    else:
        prediction[X[:, dimension] >= threshold] = -1
    return prediction

# AdaBoost algorithm
def adaboost(X, y, T=5):
    n_samples, n_dimensions = X.shape
    weights = np.ones(n_samples) / n_samples
    stumps = []
    alphas = []
    errors = []

    for t in range(T):
        # Find best stump
        stump = find_best_stump(X, y, weights)
        prediction = stump_predict(X, stump)

        # Calculate alpha (stump weight)
        epsilon = stump["error"]
        alpha = 0.5 * np.log((1 - epsilon) / (epsilon + 1e-10))
        alphas.append(alpha)
        stumps.append(stump)

        # Update weights
        weights *= np.exp(-alpha * y * prediction)
        weights /= np.sum(weights)  # Normalize

        errors.append(epsilon)
        print(f"Iteration {t + 1}/{T}, Error: {epsilon}")

    return stumps, alphas, errors

# Calculate final error
def calculate_error(X, y, stumps, alphas):
    n_samples = len(y)
    final_prediction = np.zeros(n_samples)
    for stump, alpha in zip(stumps, alphas):
        final_prediction += alpha * stump_predict(X, stump)
    final_prediction = np.sign(final_prediction)
    error = np.mean(final_prediction != y)
    return error

# Train and evaluate model
def main():
    train_path = 'C:/Users/user/Desktop/NTU_hw/final/html2024-fall-final-project-stage-2/train_data.csv'
    test_path = 'C:/Users/user/Desktop/NTU_hw/final/html2024-fall-final-project-stage-2/2024_test_data.csv'

    X_train, X_test, y_train = retrieve_column(train_path, test_path)

    best_eval = 1
    best_stumps = None
    best_alphas = None
    eval_t = []

    kfold = KFold(n_splits=5, shuffle=True)
    for train_index, valid_index in kfold.split(X_train, y_train):
        X_train_train, X_valid = X_train[train_index], X_train[valid_index]
        y_train_train, y_valid = y_train[train_index], y_train[valid_index]

        stumps, alphas, _ = adaboost(X_train_train, y_train_train, T=5)
        eval_ = calculate_error(X_valid, y_valid, stumps, alphas)
        eval_t.append(eval_)

        if eval_ < best_eval:
            best_stumps = stumps
            best_alphas = alphas
            best_eval = eval_

    print(f"Best Eval = {best_eval}")

    final_prediction = np.zeros(len(X_test))
    for stump, alpha in zip(best_stumps, best_alphas):
        final_prediction += alpha * stump_predict(X_test, stump)
    final_prediction = np.sign(final_prediction)

    with open("C:/Users/user/Desktop/NTU_hw/final/result_all.csv", 'w') as f:
        f.write("id,home_team_win\n")
        for i, pred in enumerate(final_prediction):
            f.write(f"{i},{int(pred)}\n")

    eval_avg = np.mean(eval_t)

    plt.figure(figsize=(10, 6))
    plt.plot(eval_t, label="Eval")
    plt.axhline(eval_avg, color='red', linestyle='--', label="Eval_avg")
    plt.xlabel("Fold")
    plt.ylabel("Error")
    plt.title("Eval vs Folds")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()

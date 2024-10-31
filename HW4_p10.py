from typing import Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt


def read_libsvm_file(filepath: str, dim: int, num_examples: int = 8192)\
        -> Tuple[np.ndarray[Any, np.dtype[np.float64]],
                 np.ndarray[Any, np.dtype[np.int64]]]:
    """
    讀取 LIBSVM 格式的資料檔案，並返回特徵矩陣和標籤。
    """
    X: List[np.ndarray[Any, np.dtype[np.float64]]] = []
    y: List[int] = []
    with open(filepath, 'r') as file:
        for i, line in enumerate(file):
            if i >= num_examples:
                break
            tokens = line.strip().split()
            label = int(tokens[0])
            features = np.zeros(dim + 1)
            features[0] = 1  # 增加 x0 = 1
            for item in tokens[1:]:
                index, value = item.split(":")
                features[int(index)] = float(value)
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)


def linear_regression(X: np.ndarray[Any, np.dtype[np.float64]],
                      y: np.ndarray[Any, np.dtype[np.int64]]):
    """
    使用最小二乘法進行線性回歸，計算最優權重。
    """
    X_dagger = np.linalg.pinv(X)  # 計算X的suedo-inverse
    return np.dot(X_dagger, y)    # w = X† * y


def sgd(X_train: np.ndarray[Any, np.dtype[np.float64]],
        y_train: np.ndarray[Any, np.dtype[np.int64]],
        learning_rate: float = 0.01,
        max_iterations: int = 100000,
        record_interval: int = 200)\
        -> Tuple[np.ndarray[Any, np.dtype[np.float64]],
                 List[float],
                 List[float]]:
    '''
    Stochastic Gradient Descent隨機梯度下降
    '''
    global X, y
    num_features = X.shape[1]
    w = np.zeros(num_features)
    Ein_records: List[float] = []
    Eout_records: List[float] = []
    cnt = 0
    for _ in range(max_iterations):
        cnt += 1
        i = np.random.randint(0, len(y_train))
        xi, yi = X_train[i], y_train[i]

        gradient = (np.dot(xi, w) - yi) * xi
        w -= learning_rate * gradient

        if cnt == record_interval:
            Ein = Error_measure(X_train, y, w)
            Ein_records.append(Ein)

            test_indices = np.setdiff1d(range(total_examples), random_indices)
            X_test, y_test = X[test_indices], y[test_indices]
            Eout = Error_measure(X_test, y_test, w)
            Eout_records.append(Eout)
            cnt = 0

    return w, Ein_records, Eout_records


def Error_measure(X: np.ndarray[Any, np.dtype[np.float64]],
                  y: np.ndarray[Any, np.dtype[np.int64]],
                  w: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    """
    計算Error_measure。
    """
    total_square_error = 0
    for i in range(len(X)):
        item = X[i]
        predictions = np.dot(item, w)
        square_error = (predictions - y[i]) ** 2
        total_square_error += square_error
    return total_square_error / len(X)


# 參數設定
N = 64          # 訓練樣本數量
total_examples = 8192
feature_dim = 12  # 特徵維度，根據檔案內容設定

# 讀取資料集
X, y = read_libsvm_file(
    'C:/Users/user/Desktop/台大機器學習/HW4/cpusmall_scale.txt', feature_dim, total_examples)

all_Ein_wlin, all_Eout_wlin = [], []
Ein_sgd_all, Eout_sgd_all = None, None

for k in range(1126):
    # 隨機選擇N個樣本作為訓練集
    np.random.seed(k)
    random_indices = np.random.choice(total_examples, N, replace=False)
    X_train, y_train = X[random_indices], y[random_indices]

    test_indices = np.setdiff1d(range(total_examples), random_indices)
    X_test, y_test = X[test_indices], y[test_indices]

    w_lin = linear_regression(X_train, y_train)
    Ein_wlin = Error_measure(X_train, y_train, w_lin)
    Eout_wlin = Error_measure(X_test, y_test, w_lin)

    w_sgd, Ein_records, Eout_records = sgd(X_train, y_train)

    if k == 0:
        Ein_sgd_all: np.ndarray[Any,
                                np.dtype[np.float64]] = np.array(Ein_records)
        Eout_sgd_all: np.ndarray[Any,
                                 np.dtype[np.float64]] = np.array(Eout_records)
    else:
        Ein_sgd_all += np.array(Ein_records)
        Eout_sgd_all += np.array(Eout_records)

Ein_sgd_avg = Ein_sgd_all / 1126
Eout_sgd_avg = Eout_sgd_all / 1126

avg_Ein_wlin = np.mean(all_Ein_wlin) # todo
avg_Eout_wlin = np.mean(all_Eout_wlin) # todo

# 繪製 Ein 和 Eout 的變化圖
iterations = np.arange(200, 100001, 200)
plt.plot(iterations, Ein_sgd_avg, label="Average Ein(w_t)")
plt.plot(iterations, Eout_sgd_avg, label="Average Eout(w_t)")
plt.axhline(y=avg_Ein_wlin, color='r', linestyle='--',
            label="Average Ein(w_lin)")
plt.axhline(y=avg_Eout_wlin, color='b', linestyle='--',
            label="Average Eout(w_lin)")
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.legend()
plt.title("SGD: Ein and Eout over iterations")
plt.grid(True)
plt.show()

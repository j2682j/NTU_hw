from typing import Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sys import stdout


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

def polynomial_transform(X, Q=3):
    """
    對 X 進行同次階數為 Q 的多項式變換
    """
    X_poly = [X]
    for q in range(2, Q + 1):
        X_poly.append(X[:, 1:] ** q)  # 不包括 x0
    return np.hstack(X_poly)

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


def main():
    # 參數設定
    N = 64          # 訓練樣本數量
    total_examples = 8192
    feature_dim = 12  # 特徵維度，根據檔案內容設定

    # 讀取資料集
    X, y = read_libsvm_file(
        'C:/Users/user/Desktop/台大機器學習/HW4/cpusmall_scale.txt', feature_dim, total_examples)

    # 進行多項式轉換並加上 x0
    X_poly = polynomial_transform(X, Q=3)
    X_poly = np.hstack((np.ones((X_poly.shape[0], 1)), X_poly))  # 增加 x0 = 1
    
    Ein_gain = [] #紀錄Ein

    for k in range(1126):
        # 隨機選擇N個樣本作為訓練集
        np.random.seed(k)
        random_indices = np.random.choice(total_examples, N, replace=False)
        X_train, y_train = X[random_indices], y[random_indices]
        X_train_poly = X_poly[random_indices]

        # 計算 w_lin 和 w_poly
        w_lin = linear_regression(X_train, y_train)
        w_poly = linear_regression(X_train_poly, y_train)

        # 計算 E_in(w_lin) 和 E_in(w_poly)
        Ein_wlin = Error_measure(X_train, y_train, w_lin)
        Ein_wpoly = Error_measure(X_train_poly, y_train, w_poly)

        # 計算 E_in 增益
        Ein_gain.append(Ein_wlin - Ein_wpoly)

    # 計算平均增益
    average_Ein_gain = np.mean(Ein_gain)
    print("Average Ein Gain:", average_Ein_gain)
    
    # 繪製 Ein 增益的直方圖
    plt.hist(Ein_gain, bins=30, edgecolor='black')
    plt.xlabel("Ein(w_lin) - Ein(w_poly)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Ein Gain for Polynomial Transform (Q=3)")
    plt.show()

    


if __name__ == "__main__":
    main()
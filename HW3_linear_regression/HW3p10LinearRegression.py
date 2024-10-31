import numpy as np
import matplotlib.pyplot as plt

def read_libsvm_file(filepath, dim, num_examples=8192):
    """
    讀取 LIBSVM 格式的資料檔案，並返回特徵矩陣和標籤。
    """
    X = []
    y = []
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

def linear_regression(X, y):
    """
    使用最小二乘法進行線性回歸，計算最優權重。
    """
    X_dagger = np.linalg.pinv(X)  # 計算X的suedo-inverse
    return np.dot(X_dagger, y)    # w = X† * y

def Error_measure(X, y, w):
    """
    計算Error_measure。
    """
    total_square_error = 0
    for i in range(len(X)):
        item = X[i]
        predictions = np.dot(item, w)
        square_error = (predictions - y[i]) ** 2
        total_square_error += square_error
    return total_square_error/len(X)
        

# 參數設定
N = 32          # 訓練樣本數量
total_examples = 8192
feature_dim = 12  # 特徵維度，根據檔案內容設定

# 讀取資料集
X, y = read_libsvm_file('C:/Users/user/Desktop/台大機器學習/新增文字文件(2).txt', feature_dim, total_examples)

Ein_list = []
Eout_list = []
for k in range(1126):
    # 隨機選擇N個樣本作為訓練集
    random_indices = np.random.choice(total_examples, N, replace=False)
    X_train, y_train = X[random_indices], y[random_indices]

    # 剩餘樣本給E_out
    test_indices = [i for i in range(total_examples) if i not in random_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    # 線性回歸訓練
    w_lin = linear_regression(X_train, y_train)
    
    #Ein error measure
    error_in = Error_measure(X_train, y_train, w_lin)
    

    #Eout error measure
    error_out = Error_measure(X_test, y_test, w_lin)   
    

    #保存結果
    Ein_list.append(error_in)
    Eout_list.append(error_out)

# 繪製 (Ein, Eout) 的散佈圖
plt.scatter(Ein_list, Eout_list, alpha=0.6)
plt.xlabel('Ein(w_lin)')
plt.ylabel('Eout(w_lin)')
plt.title('Scatter plot of Ein(w_lin) vs. Eout(w_lin)')
plt.grid(True)
plt.show()










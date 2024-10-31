import numpy as np
from sklearn.linear_model import LinearRegression
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
    使用 sklearn 線性回歸進行訓練。
    """
    model = LinearRegression(fit_intercept=False)  # 已經添加 x0 = 1
    model.fit(X, y)
    return model.coef_

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
N_values = range(25, 2001, 25) # 訓練樣本數量
iterarion_perN = 16
total_examples = 8192
feature_dim = 12  # 特徵維度，根據檔案內容設定

# 讀取資料集
X, y = read_libsvm_file('C:/Users/user/Desktop/台大機器學習/新增文字文件(2).txt', feature_dim, total_examples)

Ein_average = []
Eout_average = []

for N in N_values:
    Ein_N_Sum = 0
    Eout_N_Sum = 0
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

        #16次實驗結果
        Ein_N_Sum += error_in
        Eout_N_Sum += error_out 
    

    #保存結果
    Ein_average.append(Ein_N_Sum / iterarion_perN)
    Eout_average.append(Eout_N_Sum / iterarion_perN)

# 繪製 Avg.(Ein, Eout)與N 的learning curve
plt.plot(N_values, Ein_average, label='Avg.Ein(N)')
plt.plot(N_values, Eout_average, label='Avg.Eout(N)')
plt.xlabel('N')
plt.ylabel('Error')
plt.title('Avg.Ein(N) and Avg.Eout(N) over 16 experiments')
plt.legend()
plt.grid(True)
plt.show()









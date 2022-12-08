import numpy as np
import matplotlib.pyplot as plt

def get_parallel_line_data(k, b, n = 50, sigma = 0.1, n_outlier = 50):
    b.sort()
    global m # m lines
    m = len(b)
    x = np.random.uniform(0, 1, m * n + n_outlier)
    y = np.zeros(m * n + n_outlier)

    for i in range(m):
        y[i*n : i*n+n] = k * x[i*n : i*n+n] + np.random.normal(b[i], sigma, n)
    y[-n_outlier:] = np.random.uniform(b[0], k + b[-1], n_outlier)
    return np.array([x, y]).T

def OLS(data):
    X = np.c_[np.ones(len(data)), data[:, 0]]
    Y = data[:, 1]
    theta = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(Y)
    k = theta[1]
    dt = Y - k * X[:, 1]
    dt.sort()
    n = len(data) // m
    b = [dt[i*n : i*n + n].mean() for i in range(m)]
    return b, k

def dis(x, y, A, B, C):
    return abs(A * x + B * y + C) / np.sqrt(A ** 2 + B ** 2)

def RANSAC(data, thr, K):
    X, Y = data[:, 0], data[:, 1]
    num = len(data)
    k = 0
    cnt_max = 0
    for rnd in range(K):
        cnt = 0
        p1 = np.random.randint(num)
        p2 = np.random.randint(num)
        while p1 == p2: p2 = np.random.randint(num)
        k_now = (Y[p1] - Y[p2]) / (X[p1] - X[p2])
        b_now = Y[p1] - k_now * X[p1]
        for i in range(num):
            if dis(X[i], Y[i], k_now, -1, b_now) < thr: cnt += 1
        if cnt > cnt_max:
            cnt_max = cnt
            k = k_now
    dt = Y - k * X
    dt.sort()
    n = num // m
    for i in range(m):
        arr = np.array(dt[i*n : i*n + n])
        cnt_max = 0
        for j in range(n):
            cnt = ((arr[j] - thr < arr) & (arr < arr[j] + thr)).sum()
            if cnt > cnt_max:
                cnt_max = cnt
                b[i] = arr[j]
    return b, k

if __name__ == "__main__":
    k = 5
    b = [0, 3, 7]
    data = get_parallel_line_data(k, b)
    b_ols, k_ols = OLS(data)
    b_ransac, k_ransac = RANSAC(data, 0.01, 10000)

    plt.plot(data[:, 0], data[:, 1], 'o', label='sample')
    X0 = np.array([0, 1])
    for i in range(m):
        if i == 0:
            plt.plot(X0, k * X0 + b[i], 'r', label='real')
            plt.plot(X0, k_ols * X0 + b_ols[i], 'b', label='OLS')
            plt.plot(X0, k_ransac * X0 + b_ransac[i], 'g', label='RANSAC')
        else:
            plt.plot(X0, k * X0 + b[i], 'r')
            plt.plot(X0, k_ols * X0 + b_ols[i], 'b')
            plt.plot(X0, k_ransac * X0 + b_ransac[i], 'g')
    plt.legend(loc=1)
    plt.title('Parallel lines fitting with outliers')
    plt.show()
import numpy as np
import matplotlib.pyplot as plt

def get_line_data(k, b, n = 50, sigma = 0.1, m = 10):
    """
        Draw n samples from line: y = kx + b with Gaussian noise ~ N(0, sigma),
        where x in [0, 1),
        then add m outliers

        return [[x_0, y_0], [x_1, y_1], ..., [x_{n-1}, y_{n-1}]]
    """
    x = np.random.uniform(0, 1, n + m)
    y_inlier = k * x[:n] + np.random.normal(b, sigma, n)
    y_outlier = np.random.uniform(b, k + b, m)
    y = np.concatenate((y_inlier, y_outlier))
    return np.array([x, y]).T

def OLS(data):
    X = np.array([[1, data[i][0]] for i in range(len(data))])
    Y = data[:, 1]
    return np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(Y)

def dis(x, y, A, B, C):
    return abs(A * x + B * y + C) / np.sqrt(A ** 2 + B ** 2)

def RANSAC(data, thr, N):
    X, Y = data[:, 0], data[:, 1]
    m = len(data)
    k = 0
    b = 0
    cnt_max = 0
    for rnd in range(N):
        cnt = 0
        p1 = np.random.randint(m)
        p2 = np.random.randint(m)
        while p1 == p2: p2 = np.random.randint(m)
        k_now = (Y[p1] - Y[p2]) / (X[p1] - X[p2])
        b_now = Y[p1] - k_now * X[p1]
        for i in range(m):
            if dis(X[i], Y[i], k_now, -1, b_now) < thr: cnt += 1
        if cnt > cnt_max:
            k = k_now
            b = b_now
            cnt_max = cnt
    return b, k

if __name__ == "__main__":
    k = 5
    b = 7
    data = get_line_data(k, b)
    b_ols, k_ols = OLS(data)
    b_ransac, k_ransac = RANSAC(data, 0.01, 10000)

    plt.plot(data[:, 0], data[:, 1], 'o', label='sample')
    X0 = np.array([0, 1])
    plt.plot(X0, k * X0 + b, 'r', label='real')
    plt.plot(X0, k_ols * X0 + b_ols, 'b', label='OLS')
    plt.plot(X0, k_ransac * X0 + b_ransac, 'g', label='RANSAC')
    plt.legend(loc=1)
    plt.title('line fitting with outliers')
    plt.show()
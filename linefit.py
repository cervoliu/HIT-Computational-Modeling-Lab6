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

def least_square(data):
    X = np.array([[1, data[i][0]] for i in range(len(data))])
    Y = data[:, 1]
    return np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(Y)

def dis(x, y, A, B, C):
    return abs(A * x + B * y + C) / np.sqrt(A ** 2 + B ** 2)

def RANSAC(data, thr, N):
    X, Y = data[:, 0], data[:, 1]
    m = len(data)
    k_ret = 0
    b_ret = 0
    tot_max = 0
    for ep in range(N):
        tot = 0
        p1 = np.random.randint(m)
        p2 = np.random.randint(m)
        while p1 == p2: p2 = np.random.randint(m)
        k_now = (Y[p1] - Y[p2]) / (X[p1] - X[p2])
        b_now = Y[p1] - k_now * X[p1]
        for i in range(m):
            if dis(X[i], Y[i], k_now, -1, b_now) < thr:
                tot += 1
        if tot > tot_max:
            k_ret = k_now
            b_ret = b_now
            tot_max = tot
    return b_ret, k_ret

if __name__ == "__main__":
    k = 5
    b = 7
    data = get_line_data(k, b)
    b_ls, k_ls = least_square(data)
    b_r, k_r = RANSAC(data, 0.01, 10000)

    plt.plot(data[:, 0], data[:, 1], 'o', label='sample')
    X0 = np.array([0, 1])
    plt.plot(X0, k * X0 + b, 'r', label='real')
    plt.plot(X0, k_ls * X0 + b_ls, 'b', label='least square')
    plt.plot(X0, k_r * X0 + b_r, 'g', label='RANSAC')
    plt.legend(loc=1)
    plt.title('line fitting with outliers')
    plt.show()
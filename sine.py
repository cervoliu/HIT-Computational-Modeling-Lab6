import numpy as np
import matplotlib.pyplot as plt

def get_sin_data(n = 50, mu = 0, sigma = 0.05, m = 5):
    """
        Draw n samples from y = sin(2 * pi * x) with Gaussian noise ~ N(mu, sigma),
        where x in [0, 2),
        then add m outliers

        return [[x_0, y_0], [x_1, y_1], ..., [x_{n-1}, y_{n-1}]]
    """
    x = np.random.uniform(0, 2, n + m)
    y_inlier = np.sin(2 * np.pi * x[:n]) + np.random.normal(mu, sigma, n)
    y_outlier = np.random.uniform(-2, 2, m)
    y = np.concatenate((y_inlier, y_outlier))
    data = np.array([x, y]).T
    return data

def OLS(data, lam = 0., m = 9):
    """
        Use least square method with L2 regularization coefficient
        lambda = lam (default = 0, i.e. without regularization) to fit data,
        polynomial degree = m

        return [w_0, w_1, ..., w_m] (coefficient of the polynomial)
    """
    X = np.array([data[:, 0] ** i for i in range(m + 1)]).T
    Y = data[:, 1]
    return np.linalg.inv(np.dot(X.T, X) + lam * np.eye(m + 1)).dot(X.T).dot(Y)

def draw(data, w, title):
    train_x = data[:, 0]
    train_y = data[:, 1]
    real_x = np.linspace(0, 2, 100)
    real_y = np.sin(2 * np.pi * real_x)
    mat = np.array([real_x ** i for i in range(len(w))]).T
    fit_y = np.dot(mat, w)
    plt.plot(train_x, train_y, 'o', label='sample')
    plt.plot(real_x, fit_y, 'b', label='result')
    plt.plot(real_x, real_y, 'r', label='real')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=1)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    data = get_sin_data()
    w0 = OLS(data)
    draw(data, w0, 'OLS')
    w1 = OLS(data, 0.0005)
    draw(data, w1, 'OLS with L2 regularization')

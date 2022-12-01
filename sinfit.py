import numpy as np
import matplotlib.pyplot as plt

def get_sin_data(n = 100, mu = 0, sigma = 0.1, dirt = 0, m = 10):
    """
        Draw n samples from y = sin(2 * pi * x) with Gaussian noise ~ N(mu, sigma),
        where x in [0, 2),
        then add m outliers, absolute offset in [0.3, 1)

        return [[x_0, y_0], [x_1, y_1], ..., [x_{n-1}, y_{n-1}]]
    """
    x = np.random.uniform(0, 2, n + m)
    y = np.sin(2 * np.pi * x)
    noise = np.random.normal(mu, sigma, n)
    offset = np.random.choice([-1, 1], m) * np.random.uniform(0.3, 1, m)
    z = y + np.concatenate((noise, offset))
    data = np.array([x, z]).T
    return data

def fit(data, lam = 0, m = 14):
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

    plot1 = plt.plot(train_x, train_y, 's', label='sample')
    plot2 = plt.plot(real_x, fit_y, 'b', label='result')
    plot3 = plt.plot(real_x, real_y, 'r', label='real')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=1)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    data = get_sin_data()
    w0 = fit(data)
    draw(data, w0, 'least square method')
    w1 = fit(data, 0.005)
    draw(data, w1, 'least square method with L2 regularization')

import numpy as np
import matplotlib.pyplot as plt

circle_a = np.linspace(0, 2 * np.pi, 100)

def plot_circle(xc, yc, r, color, label):
    x, y = xc + r * np.cos(circle_a), yc + r * np.sin(circle_a)
    plt.plot(x, y, color, label=label)

def generate_circle_data(n = 100, m = 10, xc = 2, yc = 2, r = 1, cov = [[0.01, 0], [0, 0.01]]):
    """
        generate n points on a circle(xc, yc, r) with 2D Gaussian noise(cov),
        then add m outliers
    """
    angle = np.random.uniform(0, 2 * np.pi, n)
    points = r * np.array([np.cos(angle), np.sin(angle)]).T
    inliers = points + np.random.multivariate_normal([xc, yc], cov, n)
    outliers = np.array([np.random.uniform(xc - 2*r, xc + 2*r, m), np.random.uniform(yc - 2*r, yc + 2*r, m)]).T
    data = np.concatenate((inliers, outliers))

    plt.plot(data[:, 0], data[:, 1], 'o', label='sample')
    plot_circle(xc, yc, r, 'b', 'real')
    return data

def OLS(data):
    n = len(data)
    X = np.c_[data[:, 0], data[:, 1], np.ones(n)]
    Y = -data[:, 0] ** 2 - data[:, 1] ** 2
    w = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(Y)
    xc, yc = w[0] / -2, w[1] / -2
    r = xc ** 2 + yc ** 2 - w[2]
    return xc, yc, r

def get_circle(p1, p2, p3):
    """
        to get the circle via 3 points, using complex number method.
    """
    x, y, z = p1[0] + p1[1] * 1j, p2[0] + p2[1] * 1j, p3[0] + p3[1] * 1j
    w = (z - x) / (y - x)
    c = (x - y) * (w - abs(w) ** 2) / 2j / w.imag - x
    return -c.real, -c.imag, abs(c + x)

def RANSAC(data, thr, N = 10000):
    n = len(data)
    cnt_max = 0
    xc_res, yc_res, r_res = 0, 0, 0
    for rnd in range(N):
        i1 = np.random.randint(n)
        i2 = np.random.randint(n)
        while i1 == i2: i2 = np.random.randint(n)
        i3 = np.random.randint(n)
        while i1 == i3 or i2 == i3: i3 = np.random.randint(n)

        xc, yc, r = get_circle(data[i1], data[i2], data[i3])
        cnt = 0
        for j in range(n):
            if abs(((data[j][0] - xc) ** 2 + (data[j][1] - yc) ** 2) ** 0.5 - r) < thr: cnt += 1
        if cnt > cnt_max:
            cnt_max = cnt
            xc_res, yc_res, r_res = xc, yc, r
    return xc_res, yc_res, r_res

if __name__ == "__main__":
    data = generate_circle_data()

    xc_ols, yc_ols, r_ols = OLS(data)
    plot_circle(xc_ols, yc_ols, r_ols, 'r', 'OLS')

    xc_ransac, yc_ransac, r_ransac = RANSAC(data, 0.01)
    plot_circle(xc_ransac, yc_ransac, r_ransac, 'g', 'RANSAC')

    plt.axis('equal')
    plt.legend(loc=1)
    plt.show()
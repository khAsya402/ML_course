import numpy as np
class tsne:
    def __init__(self, learning_rate, sigma, T, n_dim):
        self.learning_rate = learning_rate
        self.sigma = sigma  # S
        self.T = T
        self.n_dim = n_dim  # desired low dimention

    def perplexity(self, X, S):  # X is raw data
        n = X.shape[0]
        perp = np.zeros((n, n))
        P = np.zeros((n, n))
        distances = np.zeros((n, n))
        sum_dist = 0
        for i in range(n):
            for j in range(n):
                distances[i][j] = np.linalg.norm(X[i] - X[j]) ** 2
        # anyway for i = j distance is 0, so we can let them be in sum
        sum_dist = np.sum(np.exp(-distances / (2 * S ** 2)))
        for i in range(n):
            for j in range(n):
                P[i][j] = (np.exp(-distances[i][j] / (2 * S ** 2))) / sum_dist
                perp[i][j] = P[i][j] / n
        return perp

    def gradient_descent(self, X, y, p):

        n = X.shape[0]
        sum_dist = 0
        deriv = np.zeros(n)
        distances = np.zeros((n, n))
        q = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i][j] = np.linalg.norm(y[i] - y[j]) ** 2
        sum_dist = np.sum(1 / (1 + distances))
        for i in range(n):
            for j in range(n):
                q[i][j] = (1 / (1 + distances[i][j])) / sum_dist
                # derivative of loss with respect to y-i
                deriv[i] += (p[i][j] - q[i][j]) * (distances[i][j]) * (1 / (1 + distances[i][j]))

        return deriv

    def fit_predict(self, X):
        n = X.shape[0]
        # initial low dimentional data
        Y = np.random.normal(0, 10 ** -4, size=(n, self.n_dim))
        p = self.perplexity(X, self.sigma)
        D = self.gradient_descent(X, Y, p)
        for t in range(self.T):
            if t < 250:
                momentum = 0.5
            else:
                momentum = 0.8
            for i in range(Y.shape[0]):
                Y[i] = Y[i - 1] + (4 * self.learning_rate * D[i]) + (momentum * (Y[i - 1] - Y[i - 2]))
        return Y
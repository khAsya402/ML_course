import numpy as np
class Spectral_Cluestering:
    def __init__(self, k, sigma):
        self.k = k
        self.sigma = sigma
        self.y = None

    def make_graph(self, X, S):  # X is raw data
        affinity_matrix = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                distances = np.linalg.norm(X[i] - X[j]) ** 2
                affinity_matrix[i][j] = np.exp(-distances / (S ** 2))
        # return np.fill_diagonal(affinity_matrix, 0)
        return affinity_matrix

    def laplasian_matrix(self, X, W, k):
        D = np.zeros((X.shape[0], X.shape[0]))
        I = np.ones((X.shape[0], X.shape[0]))
        d_s = W.sum(axis=1)
        d = []
        for i in range(d_s.size):
            d.append(float(d_s[i]))

        np.fill_diagonal(D, d)

        # if W has elements that are equal to zero, the corresponding elements in
        # D will also be zero, causing division by zero and resulting in NaN values in the L_sym matrix

        D_inv_sqrt = np.linalg.inv(np.sqrt(D + np.finfo(float).eps))  # add small constant to diagonal
        L_sym = D_inv_sqrt.dot(D - W).dot(D_inv_sqrt)

        # compute the first k eingrnvectors of L

        eigenvalues, eigenvectors = np.linalg.eig(L_sym)
        eigenvectors = eigenvectors[:, :k]
        U = np.vstack([eigenvectors[:, i] for i in range(k)])
        return U

    def fit(self, X):
        w = self.make_graph(X, self.sigma)
        U = self.laplasian_matrix(X, w, self.k)

        self.y = np.zeros((U.shape[0], U.shape[1]))
        for i in range(U.shape[0]):
            self.y = U[i, :]

    def predict(self, X):

        data = self.y
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.k)
        data = data.reshape(-1, 1)
        kmeans.fit(data)
        labels = kmeans.labels_

        return labels
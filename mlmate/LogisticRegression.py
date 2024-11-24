import numpy as np

class LogisticRegression:
    '''Binary Classifier using Stochastic Gradient on BCE-loss without regularization'''
    def __init__(self, weights_num):
        self.W = np.random.rand(weights_num, 1)
        self.b = np.random.rand(1, 1)
        self.history = []

    @staticmethod
    def _BCE(y_pred, y_true):
        return -1 * np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def _sigmoid(x):
        return 1/(1 + np.exp(-x))

    def _compute_grad_w(self, X, y):
        N = X.shape[0]
        return 1/N * (X.T @ (self.predict(X) - y))

    def _compute_grad_b(self, X, y):
        N = X.shape[0]
        return 1/N * (np.sum(self.predict(X) - y))

    # perform Batch gradient descent
    def train(self, X, y, lr = 0.05, tol = 1e-4, limit = 1000):

        # Превращаем input в вектора нужной формы

        X = np.array(X, ndmin = 2)
        y = np.array(y, ndmin = 2).reshape(-1,1)

        for i in range(limit):

            delta_w = lr * self._compute_grad_w(X, y)
            delta_b = lr * self._compute_grad_b(X, y)

            # stopping criteria
            if np.linalg.norm(delta_w) <= tol:
                break

            self.W = self.W - delta_w
            self.b = self.b - delta_b

            self.history.append(self._BCE(self.predict(X), y))

        return self.history

    def predict(self, X):
        return self._sigmoid(X @ self.W + self.b)

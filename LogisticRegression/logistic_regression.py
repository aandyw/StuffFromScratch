import numpy as np


class LogisticRegression:

    def __init__(self, iterations=100, learning_rate=1e-3, threshold=0.5):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.W = None  # weights
        self.b = None  # bias
        self.losses = []

    def sigmoid(self, z):
        # sigmoid activation function
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y, y_pred, epsilon=1e-9):
        # binary cross-entropy loss (BCE)
        return -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))

    def fit(self, X, y):
        N, features = X.shape
        self.W = np.random.randn(features)
        self.b = 0

        for iter in range(self.iterations):
            z = np.matmul(X, self.W) + self.b
            y_pred = self.sigmoid(z)
            loss = self.compute_loss(y, y_pred)

            ### compute gradients ###
            residuals = y_pred - y
            grad_W = (1 / N) * np.matmul(X.T, residuals)
            grad_b = (1 / N) * np.sum(residuals)

            ### parameter updates ###
            self.W -= self.learning_rate * grad_W
            self.b -= self.learning_rate * grad_b
            self.losses.append(loss)

    def predict(self, X):
        z = np.matmul(X, self.W) + self.b
        y_pred = self.sigmoid(z)
        y_pred = np.where(y_pred >= self.threshold, 1, 0)
        return y_pred

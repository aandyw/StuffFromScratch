import numpy as np


class LogisticRegression:

    def __init__(self, iterations=100, learning_rate=1e-3, threshold=0.5):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.W = None  # weights
        self.b = None  # bias
        self.losses = []

    def sigmoid(self, x):
        # sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def compute_loss(self, y, y_pred):
        # binary cross-entropy loss (BCE)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def compute_gradients(self, X, y, y_pred):
        N = X.shape[0]
        diff = y_pred - y
        grad_W = (1 / N) * np.matmul(diff, X)
        grad_b = (1 / N) * diff
        return grad_W, grad_b

    def update_model_weights(self, grad_W, grad_b):
        self.W -= self.learning_rate * grad_W
        self.b -= self.learning_rate * grad_b

    def fit(self, X, y):
        n, features = X.shape
        self.W = np.random.randn(features)
        self.b = 0

        for iter in range(self.iterations):
            z = np.matmul(X, self.W) + self.b
            y_pred = self.sigmoid(z)
            loss = self.compute_loss(y, y_pred)
            grad_W, grad_b = self.compute_gradients(X, y, y_pred)
            self.update_model_weights(grad_W, grad_b)
            self.losses.append(loss)

    def predict(self, X):
        z = np.matmul(X, self.W) + self.b
        y_pred = self.sigmoid(z)
        y_pred = y_pred[y_pred >= self.threshold]
        return y_pred

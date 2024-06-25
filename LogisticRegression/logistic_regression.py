import numpy as np
import scipy


class LogisticRegression:

    def __init__(self, epochs=1000, learning_rate=1e-2, threshold=0.5):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.W = None  # weights
        self.b = None  # bias
        self.losses = []

    def __sigmoid(self, z):
        # sigmoid activation function

        # return 1 / (1 + np.exp(-z))
        return scipy.special.expit(z)  # handles np.exp(.) overflow

    def __compute_loss(self, y, y_pred, epsilon=1e-9):
        # binary cross-entropy loss (BCE)

        # epsilon added to prevent log(0)
        return -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))

    def fit(self, X, y):
        N, features = X.shape
        self.W = np.random.randn(features)
        self.b = 0

        for epoch in range(self.epochs):
            z = np.matmul(X, self.W) + self.b
            y_pred = self.__sigmoid(z)
            loss = self.__compute_loss(y, y_pred)

            ### compute gradients ###
            residuals = y_pred - y
            grad_W = (1 / N) * np.matmul(X.T, residuals)
            grad_b = (1 / N) * np.sum(residuals)

            ### parameter updates ###
            self.W -= self.learning_rate * grad_W
            self.b -= self.learning_rate * grad_b
            self.losses.append(loss)

            if (epoch + 1) % 100 == 0:
                print(f"[Epoch {epoch + 1}/{self.epochs}] Loss: {round(loss, 5)}")

    def predict(self, X):
        z = np.matmul(X, self.W) + self.b
        y_pred = self.__sigmoid(z)
        y_pred = np.where(y_pred >= self.threshold, 1, 0)
        return y_pred

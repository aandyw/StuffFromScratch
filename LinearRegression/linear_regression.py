import numpy as np
import scipy


class LinearRegression:
    """Linear Regression with Least Squared Error"""

    def __init__(self, epochs=1000, learning_rate=1e-2):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.W = None  # weights
        self.b = None  # bias
        self.losses = []

    def __compute_loss(self, y, y_pred):
        # Mean Squared Error (MSE) is our cost function

        least_squares = (y_pred - y) ** 2
        return np.mean(least_squares)

    def fit(self, X, y):
        N, features = X.shape
        self.W = np.random.randn(features)
        self.b = 0

        for epoch in range(self.epochs):
            y_pred = self.predict(X)
            loss = self.__compute_loss(y, y_pred)  # MSE loss

            ### compute gradients ###
            residuals = y_pred - y
            grad_W = (2 / N) * np.matmul(X.T, residuals)
            grad_b = (2 / N) * np.sum(residuals)

            ### parameter updates ###
            self.W -= self.learning_rate * grad_W
            self.b -= self.learning_rate * grad_b
            self.losses.append(loss)

            if (epoch + 1) % 1000 == 0:
                print(f"[Epoch {epoch + 1}/{self.epochs}] Loss: {round(loss, 5)}")

    def predict(self, X):
        return np.matmul(X, self.W) + self.b

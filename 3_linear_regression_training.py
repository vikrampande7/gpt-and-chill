import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        for _ in range(num_iterations):
            # Get the Model Prediction
            y_pred = self.get_model_prediction(X, initial_weights)

            # you will need to call get_derivative() for each weight
            derivatives = np.zeros_like(initial_weights)
            for i in range(len(initial_weights)):
                derivatives[i] = self.get_derivative(y_pred, Y, len(X), X, i)

            # Update Weights
            initial_weights = initial_weights - self.learning_rate * derivatives

            # you will need to call get_derivative() for each weight
            # d_w1 = self.get_derivative(y_pred, Y, len(X), X, 0)
            # d_w2 = self.get_derivative(y_pred, Y, len(X), X, 1)
            # d_w3 = self.get_derivative(y_pred, Y, len(X), X, 2)

            # # Update weights
            # initial_weights[0] = initial_weights[0] - self.learning_rate * d_w1
            # initial_weights[1] = initial_weights[1] - self.learning_rate * d_w2
            # initial_weights[2] = initial_weights[2] - self.learning_rate * d_w3

        # return np.round(your_answer, 5)
        return np.round(initial_weights, 5)

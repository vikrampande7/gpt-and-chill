import numpy as np
from numpy.typing import NDArray


# Helpful functions:
# https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
# https://numpy.org/doc/stable/reference/generated/numpy.mean.html
# https://numpy.org/doc/stable/reference/generated/numpy.square.html

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is an Nx3 NumPy array
        #X = NDArray(shape=(len(X),3), dtype=float, order='F')
        # weights is a 3x1 NumPy array
        #weights = NDArray(shape=(len(weights),1), dtype=float, order='F')
        # HINT: np.matmul() will be useful
        y_pred = np.matmul(X, weights)
        # return np.round(your_answer, 5)
        return np.round(y_pred, 5)


    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # model_prediction is an Nx1 NumPy array
        # ground_truth is an Nx1 NumPy array
        # HINT: np.mean(), np.square() will be useful
        loss = np.mean(np.square(model_prediction - ground_truth))
        # return round(your_answer, 5)
        return np.round(loss, 5)

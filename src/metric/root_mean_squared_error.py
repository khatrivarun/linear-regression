import numpy as np

"""
Calculates the RMSE using the true Y values and predicted Y values.
"""


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray):
    n = y_pred.shape[0]
    return np.sqrt((1 / n) * np.sum(np.square(y_true) - np.square(y_pred)))

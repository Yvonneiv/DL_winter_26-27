import numpy as np

def mse(y_true, y_pred):
    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Calculate the difference between predictions and actual values
    errors = y_pred - y_true

    # Square the errors and calculate their mean
    return np.mean(errors ** 2)
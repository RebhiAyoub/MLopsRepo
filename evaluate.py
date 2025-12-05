"""
evaluate.py

Module for evaluating regression models with standard metrics:
RMSE, R2 score, and MAE.
"""

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np


def evaluate(model, x_test, y_test):
    """
    Evaluates a regression model on test data.

    Parameters:
    model : trained regression model with a predict() method
    x_test (pd.DataFrame or np.ndarray): Test features
    y_test (pd.Series or np.ndarray): True target values for the test set

    Returns:
    dict: Dictionary containing RMSE, R2, and MAE metrics
    """
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    metrics = {"RMSE": rmse, "R2": r2, "MAE": mae}

    return metrics

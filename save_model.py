"""
save_model.py

Module for saving trained machine learning models to a file using joblib.
"""

import joblib


def save_model(model, filename="model.joblib"):
    """
    Saves a trained machine learning model to a file.

    Parameters:
    model : object
        The trained model to save.
    filename : str, optional
        The path where the model will be saved (default is "model.joblib").

    Returns:
    None
    """
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

"""
load_model.py

Module for loading a trained machine learning model from a file using joblib.
"""

import joblib


def load_model(filename="model.joblib"):
    """
    Loads a trained machine learning model from a file.

    Parameters:
    filename (str): Path to the saved model file. Defaults to "model.joblib".

    Returns:
    object: The loaded model.
    """
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

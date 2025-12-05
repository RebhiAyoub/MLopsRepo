"""
loading_data.py

Module to load the Uber dataset from a CSV file.
"""

import pandas as pd


def load_data():
    """
    Loads the Uber dataset from a CSV file.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    return pd.read_csv("datauber.csv")

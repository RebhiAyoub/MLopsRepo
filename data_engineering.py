"""
data_engineering.py

Contains functions for data preparation and feature engineering.
"""

import pandas as pd


def prepare(df_old):
    """
    Prepares the DataFrame by cleaning and extracting time-based features.

    Parameters:
    df_old (pd.DataFrame): Input DataFrame with a 'pickup_datetime' column.

    Returns:
    pd.DataFrame: A new DataFrame with 'hour', 'day_of_week', and 'day_name' columns added,
                  and invalid datetime rows removed.
    """
    df = df_old.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df = df.dropna(subset=["pickup_datetime"])
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek

    day_names = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }

    df["day_name"] = df["day_of_week"].map(day_names)

    return df

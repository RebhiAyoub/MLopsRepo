"""
data_preparation.py

Module for preparing and preprocessing taxi ride dataset.
Includes handling missing values, splitting features and target,
train/test splitting, and scaling numeric features.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


def data_preparation(df):
    """
    Prepares the dataset for modeling.

    Steps:
    1. Drop missing values and duplicates.
    2. Split features and target.
    3. Split into train/test sets.
    4. Scale numeric features.

    Parameters:
    df (pd.DataFrame): Input DataFrame with taxi ride data.

    Returns:
    tuple: (features_train_scaled, features_test_scaled, y_train, y_test)
    """
    # Drop missing values and duplicates
    df_clean = df.dropna().drop_duplicates()

    # Separate target and features
    y = df_clean["fare_amount"]
    features = df_clean[
        [
            "hour",
            "day_of_week",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
        ]
    ]

    # Train/Test split
    features_train, features_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=42
    )

    # Preprocessing: scaling numeric features
    numeric_features = [
        "hour",
        "day_of_week",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]

    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_features)], remainder="drop"
    )

    # Fit & transform
    features_train_scaled = preprocessor.fit_transform(features_train)
    features_test_scaled = preprocessor.transform(features_test)

    return features_train_scaled, features_test_scaled, y_train, y_test

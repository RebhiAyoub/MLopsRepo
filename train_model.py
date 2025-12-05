"""
train_model.py

Module for training a Random Forest regression model.
"""

from sklearn.ensemble import RandomForestRegressor


def train_model(features_train, y_train):
    """
    Trains a Random Forest regression model on the given training data.

    Parameters:
    features_train (pd.DataFrame or np.ndarray): Training features.
    y_train (pd.Series or np.ndarray): Target values for training.

    Returns:
    RandomForestRegressor: The trained Random Forest model.
    """
    model = RandomForestRegressor(
        n_estimators=100, random_state=42, max_depth=None, n_jobs=-1
    )

    # Fit the model
    model.fit(features_train, y_train)

    return model

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from datetime import datetime
import os

from load_model import load_model
from model_pipeline import prepare  # your feature engineering


app = FastAPI(title="Uber Fare Prediction API")

# CORS
origins = ["*", "http://localhost", "http://127.0.0.1"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# -------------------------------
# Input schemas
# -------------------------------
class RideInput(BaseModel):
    pickup_datetime: datetime
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    passenger_count: int


class RetrainRequest(BaseModel):
    n_estimators: int = 100
    max_depth: int = None
    min_samples_split: int = 2
    test_size: float = 0.2


# -------------------------------
# Load model + preprocessor
# -------------------------------
model = load_model("random_forest_uber_model.joblib")

prepared = joblib.load("prepared_data.joblib")

from sklearn.preprocessing import StandardScaler

# Case 1 → scaler exists → load it
if isinstance(prepared, dict) and "scaler" in prepared:
    preprocessor = prepared["scaler"]

# Case 2 → scaler missing → create unused StandardScaler
else:
    print("⚠️ WARNING: 'scaler' missing in prepared_data.joblib. Using a new StandardScaler().")
    preprocessor = StandardScaler()


# -------------------------------
# Input transformation
# -------------------------------
def transform_input(data: RideInput) -> np.ndarray:
    import pandas as pd

    df = pd.DataFrame([{
        "pickup_datetime": data.pickup_datetime,
        "pickup_longitude": data.pickup_longitude,
        "pickup_latitude": data.pickup_latitude,
        "dropoff_longitude": data.dropoff_longitude,
        "dropoff_latitude": data.dropoff_latitude,
        "passenger_count": data.passenger_count,
    }])

    df_prep = prepare(df)

    X = df_prep[[
        "hour", "day_of_week",
        "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude",
        "passenger_count"
    ]]

    try:
        X_scaled = preprocessor.transform(X)
    except:
        X_scaled = X.values

    return X_scaled


# -------------------------------
# ROUTES
# -------------------------------

# Home page → root opens index.html
@app.get("/", response_class=HTMLResponse)
def serve_index():
    file_path = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(file_path):
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    return FileResponse(file_path)


# /admin page → opens admin.html
@app.get("/admin", response_class=HTMLResponse)
def serve_admin():
    file_path = os.path.join(BASE_DIR, "admin.html")
    if not os.path.exists(file_path):
        return HTMLResponse("<h1>admin.html not found</h1>", status_code=404)
    return FileResponse(file_path)


# Predict endpoint
@app.post("/predict")
def predict_fare(ride: RideInput):
    X = transform_input(ride)
    pred = model.predict(X)[0]
    return {"predicted_fare": float(pred)}


# Retrain endpoint
@app.post("/retrain")
def retrain_model(req: RetrainRequest):
    from loading_data import load_data
    from evaluate import  evaluate
    from save_model import save_model
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer

    df = load_data()
    if df is None:
        return {"detail": "Data file not found"}

    df_prep = prepare(df)
    df_clean = df_prep.dropna().drop_duplicates()

    y = df_clean["fare_amount"]
    X = df_clean[[
        "hour", "day_of_week",
        "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude",
        "passenger_count"
    ]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=req.test_size, random_state=42
    )

    numeric_features = [
        "hour", "day_of_week",
        "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude",
        "passenger_count"
    ]

    preproc = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_features)],
        remainder="drop"
    )

    X_train_prep = preproc.fit_transform(X_train)
    X_test_prep = preproc.transform(X_test)

    rf = RandomForestRegressor(
        n_estimators=req.n_estimators,
        max_depth=req.max_depth,
        min_samples_split=req.min_samples_split,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train_prep, y_train)

    metrics = evaluate(rf, X_test_prep, y_test)

    save_model(rf, "random_forest_uber_model.joblib")
    joblib.dump({
        "features_train": X_train_prep,
        "features_test": X_test_prep,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": preproc
    }, "prepared_data.joblib")

    global model, preprocessor
    model = rf
    preprocessor = preproc

    return {
        "message": "Model retrained successfully",
        "n_estimators": req.n_estimators,
        "max_depth": req.max_depth,
        "min_samples_split": req.min_samples_split,
        "test_size": req.test_size,
        "RMSE": float(metrics["RMSE"]),
        "R2": float(metrics["R2"])
    }

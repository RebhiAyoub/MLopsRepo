"""
main.py

Uber Fare Prediction Pipeline script. Handles data loading, preparation,
training, model saving, and evaluation through command-line arguments.
"""

import argparse
import joblib

# First-party imports
from data_engineering import prepare
from data_preparation import data_preparation
from train_model import train_model
from evaluate import evaluate
from save_model import save_model
from load_model import load_model
from loading_data import load_data


def main():
    """
    Main entry point for the Uber Fare Prediction Pipeline.

    Supports command-line arguments to run specific steps:
    --prepare : Only prepare data
    --train   : Only train model
    --evaluate: Only evaluate model
    --all     : Run complete pipeline (default)
    """
    parser = argparse.ArgumentParser(description="Uber Fare Prediction Pipeline")
    parser.add_argument("--prepare", action="store_true", help="Only prepare data")
    parser.add_argument("--train", action="store_true", help="Only train model")
    parser.add_argument("--evaluate", action="store_true", help="Only evaluate model")
    parser.add_argument(
        "--all", action="store_true", help="Run complete pipeline (default)"
    )
    args = parser.parse_args()

    # If no arguments provided, run everything
    if not any([args.prepare, args.train, args.evaluate, args.all]):
        args.all = True

    print("Starting Uber Fare Prediction Pipeline...")

    # -----------------------------
    # STEP 1: Prepare Data
    # -----------------------------
    if args.prepare or args.all:
        print("\nSTEP 1: Preparing Data...")
        df = load_data()
        df_prep = prepare(df)
        features_train, features_test, y_train, y_test = data_preparation(df_prep)

        # Save prepared data for later
        joblib.dump(
            {
                "features_train": features_train,
                "features_test": features_test,
                "y_train": y_train,
                "y_test": y_test,
            },
            "prepared_data.joblib",
        )
        print("Data preparation completed and saved!")

    # -----------------------------
    # STEP 2: Train Model
    # -----------------------------
    if args.train or args.all:
        print("\nSTEP 2: Training Model...")
        try:
            data = joblib.load("prepared_data.joblib")
            features_train = data["features_train"]
            features_test = data["features_test"]
            y_train = data["y_train"]
            y_test = data["y_test"]
        except FileNotFoundError:
            print("No prepared data found. Please run --prepare first.")
            return

        model = train_model(features_train, y_train)
        save_model(model, "random_forest_uber_model.joblib")
        print("Model training completed and saved!")

    # -----------------------------
    # STEP 3: Evaluate Model
    # -----------------------------
    if args.evaluate or args.all:
        print("\nSTEP 3: Evaluating Model...")
        try:
            model = load_model("random_forest_uber_model.joblib")
            data = joblib.load("prepared_data.joblib")
            features_test = data["features_test"]
            y_test = data["y_test"]
        except FileNotFoundError:
            print("Model or prepared data not found. Please run --train first.")
            return

        results = evaluate(model, features_test, y_test)
        print("Model Evaluation Results:")
        print(f"RMSE: {results['RMSE']:.4f}")
        print(f"MAE: {results['MAE']:.4f}")
        print(f"R2: {results['R2']:.4f}")
        print("Model evaluation completed!")

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()

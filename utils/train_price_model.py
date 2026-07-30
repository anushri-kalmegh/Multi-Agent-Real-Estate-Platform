import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

def train_model():
    print("Loading properties dataset...")
    workspace_dir = "/home/sunbeam/Desktop/ML_PROJECT/propwise-ai"
    data_path = os.path.join(workspace_dir, "data/properties_clean.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Clean properties dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    # Preprocessing: drop rows with missing bhk, locality, or price_per_sqft
    df = df.dropna(subset=["bhk", "locality", "price_per_sqft", "area_sqft"])

    # Features and target
    X = df[["city", "locality", "bhk", "area_sqft"]]
    y = df["price_per_sqft"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training set size: {len(X_train)} | Test set size: {len(X_test)}")

    # Define preprocessing pipeline
    categorical_features = ["city", "locality"]
    numerical_features = ["bhk", "area_sqft"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", TargetEncoder(random_state=42), categorical_features),
            ("num", StandardScaler(), numerical_features)
        ]
    )

    # Define full model pipeline
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1))
        ]
    )

    print("Training RandomForest model (predicting price_per_sqft)...")
    model_pipeline.fit(X_train, y_train)

    # Evaluate model
    y_pred = model_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Model Evaluation:")
    print(f"  R2 Score: {r2:.4f}")
    print(f"  MAE: {mae:.2f} INR/sqft")

    # Save the model
    models_dir = os.path.join(workspace_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "price_predictor.pkl")

    with open(model_save_path, "wb") as f:
        pickle.dump(model_pipeline, f)

    print(f"Model successfully saved to {model_save_path}")

if __name__ == "__main__":
    train_model()

"""
Model training module for the Credit Scoring Prediction System.

Trains and compares multiple classifiers, then saves the best one.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

from preprocess import Preprocessor, split_data
from feature_engineering import engineer_features
from evaluate import ModelEvaluator, json_serialize

# Try to import XGBoost if available
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
SAVED_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "credit_data.csv")


def build_models():
    """Return a dict of model instances."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, random_state=42
        ),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=6,
            random_state=42, eval_metric="mlogloss", use_label_encoder=False
        )
    return models


def train():
    """Train all models, select the best, and save artifacts."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SAVED_DIR, exist_ok=True)

    # Load or generate dataset
    if not os.path.exists(DATASET_PATH):
        print("Dataset not found. Generating...")
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dataset"))
        from generate_data import generate_dataset
        generate_dataset(output_path=DATASET_PATH)

    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded dataset with {len(df)} records")

    # Feature engineering
    df = engineer_features(df)

    # Preprocess
    preprocessor = Preprocessor()
    df_processed = preprocessor.fit_transform(df)

    # Encode target
    target_le = LabelEncoder()
    df_processed["Credit_Risk"] = target_le.fit_transform(df_processed["Credit_Risk"])

    # Split
    X_train, X_test, y_train, y_test = split_data(
        df_processed, target_col="Credit_Risk"
    )

    # Train and evaluate models
    models = build_models()
    results = {}
    best_model = None
    best_score = -1
    best_name = None

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        evaluator = ModelEvaluator(model, X_test, y_test)
        summary = evaluator.get_summary()
        results[name] = summary
        print(f"  {name}: Accuracy={summary['accuracy']}%, F1={summary['f1']}%")
        if summary["accuracy"] > best_score:
            best_score = summary["accuracy"]
            best_model = model
            best_name = name

    print(f"\n🏆 Best model: {best_name} with accuracy {best_score}%")

    # Save artifacts
    model_path = os.path.join(MODEL_DIR, "credit_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    joblib.dump(best_model, model_path)
    joblib.dump(preprocessor.scaler, scaler_path)

    # Save metadata
    metadata = {
        "best_model": best_name,
        "best_accuracy": best_score,
        "results": results,
        "feature_columns": X_train.columns.tolist(),
        "target_classes": target_le.classes_.tolist(),
        "n_samples": len(df),
        "n_features": X_train.shape[1],
    }
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(json_serialize(metadata), f, indent=2)

    # Save a copy of best model in saved_models
    joblib.dump(best_model, os.path.join(SAVED_DIR, "best_model.pkl"))

    # Also save the full preprocessor for predictions
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.pkl"))
    joblib.dump(target_le, os.path.join(MODEL_DIR, "target_encoder.pkl"))

    print("\n✅ Model training complete. Artifacts saved.")
    return metadata


if __name__ == "__main__":
    train()

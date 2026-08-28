"""
Prediction module for the Credit Scoring Prediction System.

Loads the saved model and preprocessor to make credit risk predictions.
"""

import os
import json
import joblib
import pandas as pd

from feature_engineering import engineer_features

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")


class CreditRiskPredictor:
    """Loads the trained model and makes credit risk predictions."""

    def __init__(self, model_dir=None):
        self.model_dir = model_dir or MODEL_DIR
        self.model = None
        self.preprocessor = None
        self.target_encoder = None
        self.metadata = {}
        self._load()

    def _load(self):
        """Load all model artifacts."""
        model_path = os.path.join(self.model_dir, "credit_model.pkl")
        preprocess_path = os.path.join(self.model_dir, "preprocessor.pkl")
        encoder_path = os.path.join(self.model_dir, "target_encoder.pkl")
        meta_path = os.path.join(self.model_dir, "model_metadata.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Model not found. Please run 'python model/train_model.py' first."
            )

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocess_path)
        self.target_encoder = joblib.load(encoder_path)

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.metadata = json.load(f)

    def predict(self, form_data):
        """
        Make a prediction from raw form data.

        Parameters
        ----------
        form_data : dict
            Raw applicant features.

        Returns
        -------
        dict
            Risk level, probabilities, confidence, and recommendations.
        """
        df = pd.DataFrame([form_data])

        # Convert types: categorical columns stay as strings, numeric as numbers
        categorical_fields = [
            "Gender", "Occupation", "Education", "Marital_Status",
            "Residence_Type", "Loan_Purpose"
        ]
        for col in df.columns:
            if col in categorical_fields:
                df[col] = df[col].astype(str)
            else:
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    df[col] = 0

# Feature engineering
        df = engineer_features(df)

        # Keep all training features in the EXACT order used during fit.
        # This order is stored in the model metadata (feature_columns).
        numeric_cols = self.preprocessor.numeric_columns or []
        categorical_cols = self.preprocessor.categorical_columns or []

        # Prefer the exact training column order from metadata to avoid
        # mismatches when numeric/categorical columns are interleaved.
        keep_cols = self.metadata.get("feature_columns") or (numeric_cols + categorical_cols)

        for col in keep_cols:
            if col not in df.columns:
                if col in numeric_cols:
                    df[col] = 0
                else:
                    df[col] = "Unknown"

        df = df[[c for c in keep_cols if c in df.columns]]

        # Transform (applies label encoders + scaling)
        df_processed = self.preprocessor.transform(df)

        # Predict
        pred = self.model.predict(df_processed)[0]
        proba = self.model.predict_proba(df_processed)[0]

        risk_label = self.target_encoder.inverse_transform([pred])[0]

        # Map classes to probabilities
        classes = self.target_encoder.classes_
        prob_dict = {str(c): round(float(p) * 100, 2) for c, p in zip(classes, proba)}

        confidence = round(float(max(proba)) * 100, 2)

        # Determine risk level and recommendations
        if "Low Risk" in risk_label:
            risk_level = "Low"
            color = "green"
            reason = "Strong financial profile with healthy income-to-debt ratios."
            recommendations = [
                "Continue maintaining on-time payments.",
                "Consider increasing savings and investments.",
                "You are eligible for competitive interest rates.",
            ]
        elif "Medium Risk" in risk_label:
            risk_level = "Medium"
            color = "yellow"
            reason = "Moderate financial indicators with some room for improvement."
            recommendations = [
                "Reduce credit card utilization below 30%.",
                "Pay down high-interest debt first.",
                "Build an emergency fund of 3-6 months of expenses.",
            ]
        else:
            risk_level = "High"
            color = "red"
            reason = "Elevated financial risk indicators detected."
            recommendations = [
                "Work with a financial advisor to improve credit health.",
                "Avoid taking on new debt until indicators improve.",
                "Address missed payments and reduce outstanding balances.",
                "Consider a debt consolidation plan.",
            ]

        return {
            "risk_level": risk_level,
            "risk_label": risk_label,
            "probability": prob_dict,
            "confidence": confidence,
            "reason": reason,
            "recommendations": recommendations,
            "color": color,
        }

    def get_feature_importance(self):
        """Return feature importance from the model if available."""
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            features = self.metadata.get("feature_columns", [])
            pairs = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
            labels = [p[0] for p in pairs][:15]
            values = [round(float(p[1]) * 100, 2) for p in pairs][:15]
            return {"labels": labels, "values": values}
        return None


if __name__ == "__main__":
    predictor = CreditRiskPredictor()
    print("Predictor loaded successfully.")
    print("Feature importance:", predictor.get_feature_importance())

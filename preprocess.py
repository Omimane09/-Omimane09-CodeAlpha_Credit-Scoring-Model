"""
Preprocessing module for the Credit Scoring Prediction System.

Handles missing values, outliers, encoding, scaling, and train/test split.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class Preprocessor:
    """Handles all data preprocessing steps."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.ohe_columns = None
        self.numeric_columns = None
        self.categorical_columns = None

    def _handle_missing(self, df):
        """Fill or drop missing values."""
        df = df.copy()
        for col in df.columns:
            if df[col].dtype in ["float64", "int64"]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")
        return df

    def _handle_outliers(self, df):
        """Cap outliers using IQR method for numeric columns."""
        df = df.copy()
        for col in self.numeric_columns:
            if col not in df.columns:
                continue
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            df[col] = df[col].clip(lower, upper)
        return df

    def fit_transform(self, df, target_col="Credit_Risk"):
        """Fit preprocessing and transform training data."""
        df = df.copy()
        df = self._handle_missing(df)

        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
        if target_col in self.categorical_columns:
            self.categorical_columns.remove(target_col)

        # Handle outliers after identifying numeric columns
        df = self._handle_outliers(df)

        # Label encode low-cardinality categoricals
        for col in self.categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        # Scale numeric features
        numeric_features = [c for c in self.numeric_columns if c != target_col]
        if numeric_features:
            df[numeric_features] = self.scaler.fit_transform(df[numeric_features])

        return df

    def transform(self, df):
        """Transform new data using fitted preprocessing."""
        df = df.copy()
        df = self._handle_missing(df)

        # Apply label encoders
        for col, le in self.label_encoders.items():
            if col in df.columns:
                df[col] = df[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )

        # Scale numeric features
        numeric_features = [c for c in self.numeric_columns if c != "Credit_Risk"]
        numeric_features = [c for c in numeric_features if c in df.columns]
        if numeric_features:
            df[numeric_features] = self.scaler.transform(df[numeric_features])

        return df


def split_data(df, target_col="Credit_Risk", test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

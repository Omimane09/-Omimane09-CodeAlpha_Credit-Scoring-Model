"""
Feature engineering module for the Credit Scoring Prediction System.

Creates derived financial features from raw applicant data to improve
model predictive power.
"""

import numpy as np
import pandas as pd


def engineer_features(df):
    """
    Create engineered features from the raw dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset with financial features.

    Returns
    -------
    pd.DataFrame
        Dataset with additional engineered features.
    """
    df = df.copy()

    # Debt-to-Income Ratio (DTI) - percentage of income going to debt
    df["Debt_to_Income_Ratio"] = (
        (df["Debt"] + df["Credit_Card_Debt"]) / df["Annual_Income"].clip(lower=1) * 100
    ).round(2)

    # Savings Ratio - percentage of income saved
    df["Savings_Ratio"] = (
        df["Savings"] / df["Annual_Income"].clip(lower=1) * 100
    ).round(2)

    # Credit Utilization % - already present, keep consistent
    df["Credit_Utilization_Pct"] = df["Credit_Utilization"].round(2)

    # Payment Score - combines payment history and missed payments
    df["Payment_Score"] = (
        df["Payment_History"] * 0.5 - df["Missed_Payments"] * 15
    ).clip(0, 500).round(2)

    # Income Stability - how regular the income is (proxy using employment)
    df["Income_Stability"] = np.random.RandomState(1).uniform(0, 100, len(df)).round(2)

    # Employment Stability - years employed normalized
    df["Employment_Stability"] = (
        df["Employment_Years"].clip(upper=20) / 20 * 100
    ).round(2)

    # Financial Health Index - composite 0-100 (higher is healthier)
    health = (
        np.minimum(100, df["Savings_Ratio"] * 0.4) +
        (100 - np.minimum(100, df["Debt_to_Income_Ratio"]) * 0.3) +
        (100 - np.minimum(100, df["Credit_Utilization"]) * 0.2) +
        np.minimum(100, df["Employment_Stability"] * 0.1)
    )
    df["Financial_Health_Index"] = health.clip(0, 100).round(2)

    # Debt Burden Score - EMI + debt relative to income
    monthly_rate = df["Interest_Rate"] / 100 / 12
    emi = (df["Loan_Amount"] * monthly_rate * (1 + monthly_rate) ** df["Loan_Term_Months"]) / \
          ((1 + monthly_rate) ** df["Loan_Term_Months"] - 1)
    emi = np.where(monthly_rate > 0, emi, df["Loan_Amount"] / df["Loan_Term_Months"].clip(lower=1))
    df["EMI"] = emi.round(2)
    df["Debt_Burden_Score"] = (
        (emi + df["Debt"] / 12) / df["Monthly_Income"].clip(lower=1) * 100
    ).clip(0, 200).round(2)

    # Asset-to-Debt ratio
    df["Asset_to_Debt_Ratio"] = (
        (df["Assets"] + df["Savings"] + df["Investments"]) /
        (df["Debt"] + df["Credit_Card_Debt"] + 1)
    ).round(2)

    # Loan-to-Income ratio
    df["Loan_to_Income_Ratio"] = (
        df["Loan_Amount"] / df["Annual_Income"].clip(lower=1)
    ).round(2)

    # Net worth
    df["Net_Worth"] = (
        df["Assets"] + df["Savings"] + df["Investments"] + df["Bank_Balance"] -
        df["Debt"] - df["Credit_Card_Debt"]
    ).round(2)

    return df


def get_feature_columns(df):
    """
    Return the final feature columns used for modeling
    (drops identifiers and the target).
    """
    drop_cols = {"Credit_Risk"}
    return [c for c in df.columns if c not in drop_cols]

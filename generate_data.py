"""
Dataset generation script for the Credit Scoring Prediction System.

Generates a realistic synthetic dataset of 10,000+ loan applicants with
various financial features and a target credit risk label.
"""

import numpy as np
import pandas as pd
import os

# Fix random seed for reproducibility
np.random.seed(42)


def generate_dataset(n_samples=10000, output_path="dataset/credit_data.csv"):
    """Generate a synthetic credit scoring dataset."""
    occupations = [
        "Engineer", "Doctor", "Teacher", "Manager", "Analyst",
        "Designer", "Accountant", "Lawyer", "Nurse", "Salesperson",
        "Technician", "Entrepreneur", "Consultant", "Driver", "Chef"
    ]
    education_levels = ["High School", "Diploma", "Bachelor", "Master", "PhD"]
    marital_statuses = ["Single", "Married", "Divorced", "Widowed"]
    residence_types = ["Owned", "Rented", "Mortgage", "Living with Family"]
    loan_purposes = [
        "Home", "Car", "Education", "Personal", "Business",
        "Debt Consolidation", "Medical", "Travel"
    ]
    genders = ["Male", "Female"]

    data = {
        "Age": np.random.randint(18, 70, n_samples),
        "Gender": np.random.choice(genders, n_samples, p=[0.52, 0.48]),
        "Annual_Income": np.random.normal(60000, 30000, n_samples).clip(15000, 400000),
        "Monthly_Income": np.random.normal(5000, 2500, n_samples).clip(1200, 35000),
        "Occupation": np.random.choice(occupations, n_samples),
        "Employment_Years": np.random.exponential(8, n_samples).clip(0, 45),
        "Education": np.random.choice(education_levels, n_samples, p=[0.2, 0.25, 0.35, 0.15, 0.05]),
        "Marital_Status": np.random.choice(marital_statuses, n_samples, p=[0.35, 0.5, 0.1, 0.05]),
        "Loan_Amount": np.random.normal(50000, 40000, n_samples).clip(1000, 500000),
        "Loan_Term_Months": np.random.choice([12, 24, 36, 48, 60, 72, 84, 120, 180, 240, 360], n_samples),
        "Interest_Rate": np.random.normal(11, 4, n_samples).clip(3, 25),
        "Debt": np.random.exponential(20000, n_samples).clip(0, 250000),
        "Credit_Card_Debt": np.random.exponential(8000, n_samples).clip(0, 90000),
        "Savings": np.random.exponential(15000, n_samples).clip(0, 300000),
        "Investments": np.random.exponential(10000, n_samples).clip(0, 500000),
        "Number_of_Credit_Cards": np.random.randint(0, 8, n_samples),
        "Missed_Payments": np.random.poisson(2, n_samples),
        "Payment_History": np.random.randint(300, 900, n_samples),
        "Credit_Utilization": np.random.uniform(0, 100, n_samples),
        "Existing_Loans": np.random.randint(0, 6, n_samples),
        "Bank_Balance": np.random.exponential(12000, n_samples).clip(0, 400000),
        "Assets": np.random.exponential(80000, n_samples).clip(0, 1000000),
        "Dependents": np.random.poisson(1, n_samples),
        "Residence_Type": np.random.choice(residence_types, n_samples),
        "Previous_Defaults": np.random.poisson(0.5, n_samples),
        "Credit_History_Length_Years": np.random.exponential(10, n_samples).clip(0, 40),
    }

    df = pd.DataFrame(data)

    # EMI calculation based on loan amount, rate, term
    monthly_rate = df["Interest_Rate"] / 100 / 12
    emi = (df["Loan_Amount"] * monthly_rate * (1 + monthly_rate) ** df["Loan_Term_Months"]) / \
          ((1 + monthly_rate) ** df["Loan_Term_Months"] - 1)
    df["EMI"] = np.where(monthly_rate > 0, emi, df["Loan_Amount"] / df["Loan_Term_Months"])

    df["Loan_Purpose"] = np.random.choice(loan_purposes, n_samples)

    # Feature engineering - engineered risk factors
    debt_to_income = (df["Debt"] + df["Credit_Card_Debt"]) / df["Annual_Income"].clip(lower=1) * 100
    savings_ratio = df["Savings"] / df["Annual_Income"].clip(lower=1) * 100
    payment_score = df["Payment_History"] * 0.5 - df["Missed_Payments"] * 15
    income_instability = np.random.uniform(0, 30, n_samples)
    employment_stability = df["Employment_Years"].clip(upper=20) / 20 * 100
    emi_to_income = df["EMI"] / df["Monthly_Income"].clip(lower=1) * 100

    # Compute risk score (higher = riskier)
    risk_score = (
        0.15 * np.maximum(0, 100 - payment_score) +
        0.20 * np.minimum(100, debt_to_income) +
        0.12 * (100 - np.minimum(100, savings_ratio)) +
        0.10 * np.minimum(100, df["Credit_Utilization"]) +
        0.10 * np.minimum(100, emi_to_income) +
        0.08 * np.minimum(100, income_instability) +
        0.08 * (100 - employment_stability) +
        0.07 * df["Previous_Defaults"] * 20 +
        0.05 * df["Missed_Payments"] * 10 +
        0.05 * np.minimum(100, df["Existing_Loans"] * 15)
    )

    # Add noise to make it realistic
    risk_score = np.clip(risk_score + np.random.normal(0, 8, n_samples), 0, 100)

    # Assign risk labels
    conditions = [
        risk_score < 33,
        (risk_score >= 33) & (risk_score < 66)
    ]
    choices = ["Low Risk", "Medium Risk"]
    df["Credit_Risk"] = np.select(conditions, choices, default="High Risk")

    # Round numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(2)

    # Introduce a small number of missing values (realistic)
    mask = np.random.rand(n_samples, 5) < 0.01
    for i, col in enumerate(["Savings", "Investments", "Bank_Balance", "Assets", "EMI"]):
        df.loc[mask[:, i], col] = np.nan

    # Ensure directories exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated with {n_samples} records at {output_path}")
    print(f"Risk distribution:\n{df['Credit_Risk'].value_counts()}")
    return df


if __name__ == "__main__":
    generate_dataset()

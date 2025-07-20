# src/data_cleaning.py

import pandas as pd
import numpy as np

def clean_ecommerce_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the e-commerce transaction data.

    Args:
        df (pd.DataFrame): The raw e-commerce transaction DataFrame (Fraud_Data.csv).

    Returns:
        pd.DataFrame: The cleaned e-commerce transaction DataFrame.
    """
    print("--- Starting Cleaning for E-commerce Data ---")

    # Convert timestamp columns to datetime, coerce errors to NaT
    # Corrected column names from 'event_timestamp' to 'signup_time' and 'purchase_time'
    if 'signup_time' in df.columns:
        df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')
    else:
        print("Warning: 'signup_time' column not found in e-commerce data.")

    if 'purchase_time' in df.columns:
        df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')
    else:
        print("Warning: 'purchase_time' column not found in e-commerce data.")

    # Drop rows where critical timestamp conversions failed
    # Assuming both signup_time and purchase_time are critical
    initial_rows = len(df)
    df.dropna(subset=['signup_time', 'purchase_time'], inplace=True)
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows due to missing/invalid timestamps.")

    # Convert categorical features to 'category' dtype for efficiency and proper handling by LightGBM/preprocessing
    # Based on Fraud_Data.csv: 'source', 'browser', 'sex' are categorical
    categorical_cols = ['source', 'browser', 'sex']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            print(f"Converted '{col}' to category type.")
        else:
            print(f"Warning: Categorical column '{col}' not found in e-commerce data.")

    # Convert 'class' (target variable) to integer type if it's not already
    if 'class' in df.columns:
        df['class'] = df['class'].astype(int)
        print("Converted 'class' to integer type.")
    else:
        print("Warning: Target 'class' column not found in e-commerce data.")

    # Handle missing values (example: for numerical columns if any, e.g., 'purchase_value', 'age')
    # For simplicity, let's fill numerical NaNs with median/mean or 0 if appropriate for the domain.
    # Check for NaNs in numerical columns (e.g., 'purchase_value', 'age')
    # For 'purchase_value' and 'age', it's less likely to be NaN, but good to check.
    numerical_cols_to_check = ['purchase_value', 'age']
    for col in numerical_cols_to_check:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Filled NaN in '{col}' with median value ({median_val}).")
        elif col not in df.columns:
            print(f"Warning: Numerical column '{col}' not found in e-commerce data.")

    print("--- E-commerce Data Cleaning Complete ---")
    return df

def clean_bank_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the bank transaction data (creditcard.csv).

    Args:
        df (pd.DataFrame): The raw bank transaction DataFrame (creditcard.csv).

    Returns:
        pd.DataFrame: The cleaned bank transaction DataFrame.
    """
    print("--- Starting Cleaning for Bank Data ---")

    # The creditcard.csv dataset is generally very clean, with no explicit NaNs
    # in V1-V28, Time, Amount, or Class. However, it's good practice to ensure.

    # Check for any missing values across all columns
    if df.isnull().sum().sum() > 0:
        print("Warning: Missing values detected in bank data. Consider imputation strategies if significant.")
        # For this dataset, usually no NaNs. If found, a simple dropna or imputation might be needed.
        # Example: df.dropna(inplace=True)
        # For now, we'll assume it's clean as per typical creditcard.csv datasets.

    # Ensure 'Time' and 'Amount' are numerical
    if 'Time' in df.columns:
        df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    if 'Amount' in df.columns:
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

    # Convert 'Class' (target variable) to integer type if it's not already
    if 'Class' in df.columns:
        df['Class'] = df['Class'].astype(int)
        print("Converted 'Class' to integer type.")
    else:
        print("Warning: Target 'Class' column not found in bank data.")

    print("--- Bank Data Cleaning Complete ---")
    return df

# You might have other cleaning functions here for other datasets if applicable
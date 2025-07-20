# src/data_cleaning.py

import pandas as pd
import numpy as np

def clean_ecommerce_data(df):
    """
    Performs basic cleaning on the e-commerce transaction dataset.
    - Handles missing values (simple imputation or dropping).
    - Ensures correct data types.

    Args:
        df (pd.DataFrame): The input e-commerce DataFrame.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    print("--- Starting Cleaning for E-commerce Data ---")

    # Convert 'event_timestamp' to datetime, coerce errors to NaT
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'], errors='coerce')

    # Drop rows where 'event_timestamp' is NaT (if conversion failed)
    df.dropna(subset=['event_timestamp'], inplace=True)

    # Handle missing 'brand' - fill with 'unknown'
    df['brand'].fillna('unknown', inplace=True)

    # Handle missing 'price' - fill with median or mean, or drop rows
    # For simplicity, let's fill with median
    if 'price' in df.columns and df['price'].isnull().any():
        median_price = df['price'].median()
        df['price'].fillna(median_price, inplace=True)
        print(f"Filled missing 'price' values with median: {median_price}")

    # Ensure numerical columns are numeric, coerce errors
    for col in ['price']: # Add other numerical columns if any
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Drop rows with NaN if coercion failed for critical numerical columns
            df.dropna(subset=[col], inplace=True)


    # Drop duplicate rows based on all columns or a subset of key columns
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    if df.shape[0] < initial_rows:
        print(f"Dropped {initial_rows - df.shape[0]} duplicate rows.")

    print("--- E-commerce Data Cleaning Complete ---")
    return df

def clean_bank_data(df):
    """
    Performs basic cleaning on the bank transaction (credit card fraud) dataset.
    - Handles missing values (if any, though creditcard.csv is typically clean).
    - Ensures correct data types.

    Args:
        df (pd.DataFrame): The input bank transaction DataFrame.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    print("--- Starting Cleaning for Bank Data ---")

    # The credit card fraud dataset (creditcard.csv) is usually very clean with no NaNs.
    # We'll include a check here for robustness, but it's unlikely to find any.
    if df.isnull().any().any():
        print("Warning: Missing values found in bank data. Handling by dropping rows.")
        initial_rows = df.shape[0]
        df.dropna(inplace=True)
        if df.shape[0] < initial_rows:
            print(f"Dropped {initial_rows - df.shape[0]} rows with missing values.")

    # Ensure 'Time' and 'Amount' are float64, 'Class' is int64
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df['Class'] = pd.to_numeric(df['Class'], errors='coerce').astype(int) # Class should be 0 or 1

    # Drop rows where conversion failed for critical columns (should not happen with this dataset)
    df.dropna(subset=['Time', 'Amount', 'Class'], inplace=True)


    # Drop duplicate rows based on all columns
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    if df.shape[0] < initial_rows:
        print(f"Dropped {initial_rows - df.shape[0]} duplicate rows.")


    print("--- Bank Data Cleaning Complete ---")
    return df

if __name__ == '__main__':
    print("Testing data_cleaning.py")

    # Test E-commerce cleaning
    ecommerce_test_data = {
        'event_timestamp': ['2023-01-01 10:00:00', '2023-01-01 10:30:00', 'invalid_date', '2023-01-02 15:00:00'],
        'event_type': ['view', 'purchase', 'view', 'purchase'],
        'product_id': [1, 2, 1, 3],
        'category_id': [10, 10, 20, 20],
        'brand': ['brandA', np.nan, 'brandA', 'brandC'],
        'price': [10.0, np.nan, 15.0, 25.0],
        'user_id': [101, 101, 102, 102]
    }
    df_ecommerce_test = pd.DataFrame(ecommerce_test_data)
    print("\nOriginal E-commerce test data:")
    print(df_ecommerce_test)
    df_ecommerce_cleaned = clean_ecommerce_data(df_ecommerce_test.copy())
    print("\nCleaned E-commerce test data head:")
    print(df_ecommerce_cleaned.head())
    print("Missing values after cleaning (E-commerce):")
    print(df_ecommerce_cleaned.isnull().sum())

    # Test Bank cleaning
    bank_test_data = {
        'Time': [0.0, 1.0, 2.0, np.nan],
        'V1': [0.1, 0.2, 0.3, 0.4],
        'Amount': [10.0, 20.0, np.nan, 40.0],
        'Class': [0, 0, 1, 0]
    }
    df_bank_test = pd.DataFrame(bank_test_data)
    print("\nOriginal Bank test data:")
    print(df_bank_test)
    df_bank_cleaned = clean_bank_data(df_bank_test.copy())
    print("\nCleaned Bank test data head:")
    print(df_bank_cleaned.head())
    print("Missing values after cleaning (Bank):")
    print(df_bank_cleaned.isnull().sum())
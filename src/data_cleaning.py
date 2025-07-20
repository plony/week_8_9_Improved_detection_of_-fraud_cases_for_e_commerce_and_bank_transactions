# src/data_cleaning.py

import pandas as pd
import numpy as np

def clean_ecommerce_data(df):
    """
    Performs initial data cleaning for the e-commerce fraud dataset.
    - Removes duplicate rows.
    - Corrects data types for time columns.
    - Converts specified categorical columns to 'category' dtype.
    """
    print("Starting e-commerce data cleaning...")

    # Remove duplicates
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Removed {initial_rows - len(df)} duplicate rows.")

    # Correct data types for time
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    print("Converted 'signup_time' and 'purchase_time' to datetime objects.")

    # Convert specified categorical columns to 'category' dtype
    categorical_cols = ['source', 'browser', 'sex']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    print(f"Converted {', '.join(categorical_cols)} to 'category' type.")

    # Ensure target variable is int8
    if 'class' in df.columns:
        df['class'] = df['class'].astype('int8')

    print("E-commerce data cleaning complete.")
    return df

def clean_bank_data(df):
    """
    Performs initial data cleaning for the bank transaction fraud dataset.
    - Removes duplicate rows.
    (Other cleaning steps like type conversion are less critical for this dataset
    as features are mostly numerical and pre-processed).
    """
    print("Starting bank data cleaning...")

    # Remove duplicates
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Removed {initial_rows - len(df)} duplicate rows.")

    # For creditcard.csv, V1-V28 are already numerical. Time and Amount are also.
    # No specific type conversions usually needed.
    # Check for missing values (typically none in this dataset)
    if df.isnull().sum().sum() > 0:
        print("Warning: Missing values found in bank data. This dataset typically has none.")
        print(df.isnull().sum()[df.isnull().sum() > 0])
        # Add imputation logic here if necessary for your specific dataset
        # For typical creditcard.csv, this block won't execute.

    print("Bank data cleaning complete.")
    return df

if __name__ == '__main__':
    # Example usage (for testing this module directly)
    # This block will only run if data_cleaning.py is executed directly
    # python src/data_cleaning.py

    print("--- Testing data_cleaning.py ---")

    # Create dummy dataframes for testing
    ecommerce_data = {
        'user_id': [1, 1, 2, 3, 4, 4],
        'signup_time': ['2023-01-01 10:00:00', '2023-01-01 10:00:00', '2023-01-02 11:00:00', '2023-01-03 12:00:00', '2023-01-04 13:00:00', '2023-01-04 13:00:00'],
        'purchase_time': ['2023-01-01 10:30:00', '2023-01-01 10:30:00', '2023-01-02 11:30:00', '2023-01-03 12:30:00', '2023-01-04 13:30:00', '2023-01-04 13:30:00'],
        'purchase_value': [50, 50, 100, 20, 75, 75],
        'device_id': ['A', 'A', 'B', 'C', 'D', 'D'],
        'source': ['SEO', 'SEO', 'Ads', 'Direct', 'SEO', 'SEO'],
        'browser': ['Chrome', 'Chrome', 'Safari', 'Firefox', 'Edge', 'Edge'],
        'sex': ['M', 'M', 'F', 'M', 'F', 'F'],
        'age': [30, 30, 25, 40, 35, 35],
        'ip_address': ['192.168.1.1', '192.168.1.1', '192.168.1.2', '192.168.1.3', '192.168.1.4', '192.168.1.4'],
        'class': [0, 0, 1, 0, 0, 0]
    }
    df_eco_test = pd.DataFrame(ecommerce_data)
    print("\nOriginal E-commerce Test Data:")
    print(df_eco_test)
    print("\nInfo before cleaning:")
    df_eco_test.info()

    df_eco_cleaned = clean_ecommerce_data(df_eco_test.copy())
    print("\nCleaned E-commerce Test Data:")
    print(df_eco_cleaned)
    print("\nInfo after cleaning:")
    df_eco_cleaned.info()

    bank_data = {
        'Time': [0, 1, 2, 3, 4, 4],
        'V1': [1, 2, 3, 4, 5, 5],
        'Amount': [10, 20, 30, 40, 50, 50],
        'Class': [0, 0, 1, 0, 0, 0]
    }
    df_bank_test = pd.DataFrame(bank_data)
    print("\nOriginal Bank Test Data:")
    print(df_bank_test)
    print("\nInfo before cleaning:")
    df_bank_test.info()

    df_bank_cleaned = clean_bank_data(df_bank_test.copy())
    print("\nCleaned Bank Test Data:")
    print(df_bank_cleaned)
    print("\nInfo after cleaning:")
    df_bank_cleaned.info()
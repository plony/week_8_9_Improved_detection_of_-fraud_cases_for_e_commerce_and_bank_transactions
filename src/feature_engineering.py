# src/feature_engineering.py

import pandas as pd
import numpy as np

def create_ecommerce_features(df):
    """
    Performs feature engineering for the e-commerce transaction dataset.

    Args:
        df (pd.DataFrame): The input e-commerce DataFrame.
            Expected to have 'event_timestamp' as datetime.

    Returns:
        pd.DataFrame: The DataFrame with new engineered features.
    """
    print("--- Starting Feature Engineering for E-commerce Data ---")

    # Time-based Features (ensure 'event_timestamp' is datetime)
    if not pd.api.types.is_datetime64_any_dtype(df['event_timestamp']):
        df['event_timestamp'] = pd.to_datetime(df['event_timestamp'], errors='coerce')
        df.dropna(subset=['event_timestamp'], inplace=True) # Drop rows where conversion failed

    df['hour_of_day'] = df['event_timestamp'].dt.hour.astype('category')
    df['day_of_week'] = df['event_timestamp'].dt.dayofweek.astype('category') # Monday=0, Sunday=6
    df['month'] = df['event_timestamp'].dt.month.astype('category')
    df['day_of_month'] = df['event_timestamp'].dt.day.astype('category')

    # Convert original `user_id`, `product_id`, etc., to category for potential one-hot encoding
    df['user_id'] = df['user_id'].astype('category')
    df['product_id'] = df['product_id'].astype('category')
    df['category_id'] = df['category_id'].astype('category')
    df['brand'] = df['brand'].astype('category')
    df['event_type'] = df['event_type'].astype('category')


    # Transaction-based Features
    # Sort by user and time for sequential features
    df = df.sort_values(by=['user_id', 'event_timestamp'])

    # Time since last transaction for each user (in seconds)
    df['time_since_last_transaction'] = df.groupby('user_id')['event_timestamp'].diff().dt.total_seconds().fillna(0)

    # Count of transactions per user
    df['user_transaction_count'] = df.groupby('user_id').cumcount() + 1

    # Average price per user (cumulative average up to current transaction)
    df['user_avg_price'] = df.groupby('user_id')['price'].expanding().mean().reset_index(level=0, drop=True)
    df['user_avg_price'].fillna(df['price'], inplace=True) # For the first transaction

    # Indicate if it's the first transaction for a user in the dataset
    df['is_first_transaction'] = (df['user_transaction_count'] == 1).astype(int)


    print("--- E-commerce Feature Engineering Complete ---")
    return df

def create_bank_features(df):
    """
    Performs feature engineering for the bank transaction (credit card fraud) dataset.

    Args:
        df (pd.DataFrame): The input bank transaction DataFrame.
        This DF is expected to have 'Time' and 'Amount' features along with V1-V28 PCA features.

    Returns:
        pd.DataFrame: The DataFrame with new engineered features.
    """
    print("--- Starting Feature Engineering for Bank Data ---")

    # The 'Time' column in this dataset is seconds elapsed since the first transaction.
    # We will derive cyclical time features from it.

    # Calculate hour of day (0-23)
    # Convert seconds to hours, take integer part (floor), then modulo 24
    df['hour_of_day'] = (df['Time'] / 3600).astype(int) % 24
    df['hour_of_day'] = df['hour_of_day'].astype('category') # Convert to category type


    # Calculate day of week (0-6, assuming dataset starts on day 0)
    # Convert seconds to days, take integer part (floor), then modulo 7
    df['day_of_week'] = (df['Time'] / (3600 * 24)).astype(int) % 7
    df['day_of_week'] = df['day_of_week'].astype('category') # Convert to category type


    # Interaction features: Amount related to time
    # This can be tricky due to the nature of the 'Time' column in this dataset.
    # 'Amount_per_hour' or 'Amount_per_day' might be less intuitive since 'Time' is continuous seconds.
    # Instead, consider ratios of Amount to other features (e.g., mean, max) or transformations.
    # For now, let's keep it simple with just hour_of_day and day_of_week to avoid high cardinality issues.
    # You could add things like:
    # df['Amount_log'] = np.log1p(df['Amount']) # Log transform for skewed Amount
    # df['Amount_per_V_feature'] = df['Amount'] / (df['V_feature'].abs() + 1e-6) # Example: amount / V1.abs()

    print("--- Bank Feature Engineering Complete ---")
    return df

if __name__ == '__main__':
    # This block is for testing the functions directly
    print("Running feature_engineering.py as a standalone script for testing.")

    # --- Test E-commerce Features ---
    print("\nTesting E-commerce Feature Engineering...")
    ecommerce_data = {
        'event_timestamp': ['2023-01-01 10:00:00', '2023-01-01 10:30:00', '2023-01-02 15:00:00', '2023-01-02 15:10:00', '2023-01-01 11:00:00'],
        'event_type': ['view', 'purchase', 'view', 'purchase', 'view'],
        'product_id': [1, 2, 1, 3, 4],
        'category_id': [10, 10, 20, 20, 10],
        'brand': ['brandA', 'brandB', 'brandA', 'brandC', 'brandA'],
        'price': [10.0, 20.0, 15.0, 25.0, 12.0],
        'user_id': [101, 101, 102, 102, 101]
    }
    df_ecommerce_test = pd.DataFrame(ecommerce_data)
    df_ecommerce_test_engineered = create_ecommerce_features(df_ecommerce_test.copy())
    print(df_ecommerce_test_engineered.head(5))
    print("\nE-commerce Feature Dtypes:")
    print(df_ecommerce_test_engineered[['hour_of_day', 'day_of_week', 'month', 'day_of_month',
                                       'user_id', 'product_id', 'category_id', 'brand', 'event_type',
                                       'time_since_last_transaction', 'user_transaction_count', 'user_avg_price', 'is_first_transaction']].dtypes)
    print("\nE-commerce hour_of_day unique values:", df_ecommerce_test_engineered['hour_of_day'].nunique())
    print("E-commerce day_of_week unique values:", df_ecommerce_test_engineered['day_of_week'].nunique())


    # --- Test Bank Features ---
    print("\nTesting Bank Feature Engineering...")
    # Create a dummy bank dataset mimicking the structure of creditcard.csv
    bank_data = {
        'Time': [0.0, 1.0, 3600.0, 3601.0, 86399.0, 86400.0, 172800.0, 172801.0],
        'V1': np.random.rand(8), 'V2': np.random.rand(8), 'V3': np.random.rand(8),
        'Amount': [10.0, 20.0, 15.0, 25.0, 30.0, 50.0, 100.0, 120.0],
        'Class': [0, 0, 0, 0, 0, 1, 0, 0]
    }
    df_bank_test = pd.DataFrame(bank_data)
    df_bank_test_engineered = create_bank_features(df_bank_test.copy())
    print(df_bank_test_engineered.head(8)) # Print all 8 rows to see changes
    print("\nBank Feature Dtypes:")
    print(df_bank_test_engineered[['hour_of_day', 'day_of_week']].dtypes)
    print("\nBank hour_of_day unique values:", df_bank_test_engineered['hour_of_day'].nunique())
    print("Bank day_of_week unique values:", df_bank_test_engineered['day_of_week'].nunique())
    print("\nBank hour_of_day values:", df_bank_test_engineered['hour_of_day'].unique())
    print("Bank day_of_week values:", df_bank_test_engineered['day_of_week'].unique())
# src/feature_engineering.py

import pandas as pd
import numpy as np

def create_ecommerce_features(df):
    """
    Engineers new features for the e-commerce fraud dataset.
    - Time-based features (hour_of_day, day_of_week, day_of_year).
    - time_since_signup.
    - Transaction frequency/velocity features (simple counts for now).
    """
    print("Starting feature engineering for e-commerce data...")

    # Ensure time columns are datetime objects before feature extraction
    if not pd.api.types.is_datetime64_any_dtype(df['purchase_time']):
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    if not pd.api.types.is_datetime64_any_dtype(df['signup_time']):
        df['signup_time'] = pd.to_datetime(df['signup_time'])

    # Time-Based features
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek # Monday=0, Sunday=6
    df['day_of_year'] = df['purchase_time'].dt.dayofyear # For seasonality detection
    print("Created 'hour_of_day', 'day_of_week', 'day_of_year'.")

    # time_since_signup
    df['time_since_signup_hours'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
    # Handle negative values (if purchase_time < signup_time due to data errors)
    df['time_since_signup_hours'] = df['time_since_signup_hours'].apply(lambda x: max(0, x))
    print("Created 'time_since_signup_hours'.")

    # Transaction frequency and velocity (simpler aggregations)
    # These typically rely on previous transactions, so ensure data is sorted by time if needed.
    # For global counts, sort is not strictly necessary.

    # Number of transactions per user_id, device_id, ip_address
    df['user_transaction_count'] = df.groupby('user_id')['purchase_time'].transform('count')
    df['device_transaction_count'] = df.groupby('device_id')['purchase_time'].transform('count')
    df['ip_transaction_count'] = df.groupby('ip_address')['purchase_time'].transform('count')
    print("Created transaction count features for user, device, and IP.")

    # Number of unique devices/IPs used by a user
    df['user_unique_devices'] = df.groupby('user_id')['device_id'].transform('nunique')
    df['user_unique_ips'] = df.groupby('user_id')['ip_address'].transform('nunique')
    print("Created user's unique device/IP count features.")

    # Number of unique users per device/IP (indicates shared resources)
    df['device_unique_users'] = df.groupby('device_id')['user_id'].transform('nunique')
    df['ip_unique_users'] = df.groupby('ip_address')['user_id'].transform('nunique')
    print("Created device/IP's unique user count features.")

    # Convert newly created time features to category if they have few unique values,
    # or keep as numerical if they act as a continuous signal.
    # For hour_of_day and day_of_week, often treated as categorical or binned.
    df['hour_of_day'] = df['hour_of_day'].astype('category')
    df['day_of_week'] = df['day_of_week'].astype('category')

    print("E-commerce feature engineering complete.")
    return df

def create_bank_features(df):
    """
    Engineers new features for the bank transaction dataset.
    - Time-based features (hour_of_day, day_of_week) from 'Time' column.
    """
    print("Starting feature engineering for bank data...")

    # Ensure 'Time' is numerical before conversion
    if not pd.api.types.is_numeric_dtype(df['Time']):
        df['Time'] = pd.to_numeric(df['Time'], errors='coerce')

    # Convert 'Time' (seconds from first transaction) to cyclical features
    # Assuming a 24-hour cycle for 'hour_of_day' and a 7-day cycle for 'day_of_week'.
    # Max time in dataset is ~172792 seconds, which is ~48 hours.
    # hour_of_day will wrap around every 24 hours.
    df['hour_of_day'] = (df['Time'] % (24 * 3600)) / 3600
    df['day_of_week'] = (df['Time'] // (24 * 3600)) % 7 # Simple day counter, wraps every 7 days

    # Convert newly created time features to category if they have few unique values,
    # or keep as numerical. For these, they might be more useful as categorical.
    df['hour_of_day'] = df['hour_of_day'].astype('category')
    df['day_of_week'] = df['day_of_week'].astype('category')

    print("Created 'hour_of_day' and 'day_of_week' for bank data.")
    print("Bank feature engineering complete.")
    return df

if __name__ == '__main__':
    # Example usage (for testing this module directly)
    print("--- Testing feature_engineering.py ---")

    # Dummy E-commerce Data for testing
    ecommerce_data = {
        'user_id': [1, 1, 2, 3, 4],
        'signup_time': ['2023-01-01 10:00:00', '2023-01-01 10:00:00', '2023-01-02 11:00:00', '2023-01-03 12:00:00', '2023-01-04 13:00:00'],
        'purchase_time': ['2023-01-01 10:30:00', '2023-01-01 10:45:00', '2023-01-02 11:30:00', '2023-01-03 12:30:00', '2023-01-04 13:30:00'],
        'purchase_value': [50, 60, 100, 20, 75],
        'device_id': ['A', 'A', 'B', 'C', 'D'],
        'source': ['SEO', 'Ads', 'Ads', 'Direct', 'SEO'],
        'browser': ['Chrome', 'Safari', 'Safari', 'Firefox', 'Edge'],
        'sex': ['M', 'M', 'F', 'M', 'F'],
        'age': [30, 30, 25, 40, 35],
        'ip_address': ['192.168.1.1', '192.168.1.1', '192.168.1.2', '192.168.1.3', '192.168.1.4'],
        'class': [0, 0, 1, 0, 0]
    }
    df_eco_test = pd.DataFrame(ecommerce_data)
    df_eco_test['signup_time'] = pd.to_datetime(df_eco_test['signup_time'])
    df_eco_test['purchase_time'] = pd.to_datetime(df_eco_test['purchase_time'])
    print("\nOriginal E-commerce Test Data:")
    print(df_eco_test)

    df_eco_engineered = create_ecommerce_features(df_eco_test.copy())
    print("\nE-commerce Data after Feature Engineering:")
    print(df_eco_engineered.head())
    print(df_eco_engineered[['purchase_time', 'signup_time', 'hour_of_day', 'day_of_week', 'time_since_signup_hours',
                            'user_transaction_count', 'device_unique_users']].head())


    # Dummy Bank Data for testing
    bank_data = {
        'Time': [0, 3600, 86400, 90000, 172800], # 0s, 1hr, 24hr, 25hr, 48hr
        'V1': [1, 2, 3, 4, 5],
        'Amount': [10, 20, 30, 40, 50],
        'Class': [0, 0, 1, 0, 0]
    }
    df_bank_test = pd.DataFrame(bank_data)
    print("\nOriginal Bank Test Data:")
    print(df_bank_test)

    df_bank_engineered = create_bank_features(df_bank_test.copy())
    print("\nBank Data after Feature Engineering:")
    print(df_bank_engineered.head())
    print(df_bank_engineered[['Time', 'hour_of_day', 'day_of_week']].head())
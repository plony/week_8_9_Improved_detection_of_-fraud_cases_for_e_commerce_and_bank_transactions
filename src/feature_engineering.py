# src/feature_engineering.py

import pandas as pd
import numpy as np
from bisect import bisect_left # Import bisect_left

# Helper function to convert IP address string to integer
def ip_to_int(ip_str):
    if pd.isna(ip_str):
        return np.nan
    try:
        parts = list(map(int, ip_str.split('.')))
        return parts[0] * 2**24 + parts[1] * 2**16 + parts[2] * 2**8 + parts[3]
    except Exception:
        # Handle cases where IP string is malformed
        return np.nan

def create_ecommerce_features(df: pd.DataFrame, ip_to_country_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates and engineers features for the e-commerce transaction data.

    Args:
        df (pd.DataFrame): The cleaned e-commerce transaction DataFrame.
                           Expected columns: signup_time, purchase_time, purchase_value,
                           device_id, source, browser, sex, age, ip_address, class.
        ip_to_country_df (pd.DataFrame): DataFrame mapping IP address ranges to countries.
                                       Expected columns: lower_bound_ip_address,
                                       upper_bound_ip_address, country.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    print("--- Starting Feature Engineering for E-commerce Data ---")

    # 1. Time-based Features
    if 'purchase_time' in df.columns and 'signup_time' in df.columns:
        # Ensure they are datetime objects (already done in cleaning, but good to be safe)
        df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')
        df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')

        # Time difference between signup and purchase
        df['time_to_purchase'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
        # Handle cases where signup_time is after purchase_time (e.g., set to 0 or NaN)
        df['time_to_purchase'] = df['time_to_purchase'].apply(lambda x: max(x, 0) if not pd.isna(x) else np.nan)


        # Hour of the day for purchase
        df['purchase_hour'] = df['purchase_time'].dt.hour
        # Day of the week for purchase (0=Monday, 6=Sunday)
        df['purchase_day_of_week'] = df['purchase_time'].dt.dayofweek

        print("Engineered time-based features: time_to_purchase, purchase_hour, purchase_day_of_week.")
    else:
        print("Warning: 'purchase_time' or 'signup_time' not found for time-based features.")


    # 2. IP Address to Country Mapping (Geolocation Feature)
    print("Mapping IP addresses to countries...")
    if 'ip_address' in df.columns and 'lower_bound_ip_address' in ip_to_country_df.columns:
        # Convert IP address strings to integers for both dataframes
        df['ip_address_int'] = df['ip_address'].apply(ip_to_int)
        ip_to_country_df['lower_bound_ip_int'] = ip_to_country_df['lower_bound_ip_address'].apply(ip_to_int)
        ip_to_country_df['upper_bound_ip_int'] = ip_to_country_df['upper_bound_ip_address'].apply(ip_to_int)

        # Sort IP to country mapping by lower bound for efficient bisect_left lookup
        ip_to_country_df_sorted = ip_to_country_df.sort_values('lower_bound_ip_int').reset_index(drop=True)

        lower_bounds = ip_to_country_df_sorted['lower_bound_ip_int'].tolist()
        upper_bounds = ip_to_country_df_sorted['upper_bound_ip_int'].tolist()
        countries = ip_to_country_df_sorted['country'].tolist()

        def get_country_from_ip(ip_int):
            if pd.isna(ip_int):
                return 'Unknown' # Handle NaN IPs by assigning 'Unknown' country
            
            # Find the index where ip_int would be inserted to maintain order
            idx = bisect_left(lower_bounds, ip_int)

            # Check the range just before or at idx
            # If idx is 0, it means ip_int is smaller than all lower bounds.
            # Otherwise, check the previous range for a match.
            if idx == len(lower_bounds) or ip_int < lower_bounds[idx]:
                idx -= 1 # Look at the previous range

            if idx >= 0 and ip_int >= lower_bounds[idx] and ip_int <= upper_bounds[idx]:
                return countries[idx]
            return 'Unknown' # Default for IPs not found in any range

        df['country'] = df['ip_address_int'].apply(get_country_from_ip)
        df['country'] = df['country'].astype('category') # Convert to category
        print("Successfully mapped IP addresses to 'country' feature using bisect_left.")
    else:
        print("Warning: 'ip_address' not found in e-commerce data or IP to Country mapping data invalid. Assigning 'Unknown' country.")
        df['country'] = 'Unknown' # Add a default country column to avoid downstream errors
        df['country'] = df['country'].astype('category')


    # 3. Other potential features from Fraud_Data.csv
    # Frequency encoding for high cardinality categorical features like user_id, device_id
    if 'user_id' in df.columns:
        df['user_id_count'] = df.groupby('user_id')['user_id'].transform('count')
        print("Engineered 'user_id_count' feature.")
    if 'device_id' in df.columns:
        df['device_id_count'] = df.groupby('device_id')['device_id'].transform('count')
        print("Engineered 'device_id_count' feature.")

    # 4. Drop original columns that are now redundant or not features for the model
    columns_to_drop = [
        'signup_time',          # Replaced by time_to_purchase, purchase_hour, purchase_day_of_week
        'purchase_time',        # Replaced by time_to_purchase, purchase_hour, purchase_day_of_week
        'ip_address',           # Replaced by 'country'
        'ip_address_int',       # Helper column for IP lookup
        # 'user_id',            # Keep if needed for other analysis, or drop if user_id_count is sufficient
        # 'device_id',          # Keep if needed for other analysis, or drop if device_id_count is sufficient
    ]
    # Also drop the temporary IP range columns from ip_to_country_df that were merged
    # (these are not in df, but good to be aware they are helper columns)
    # 'lower_bound_ip_int', 'upper_bound_ip_int'

    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True, errors='ignore')
    print(f"Dropped redundant columns from e-commerce data: {[col for col in columns_to_drop if col in df.columns]}")


    print("--- E-commerce Data Feature Engineering Complete ---")
    return df


def create_bank_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates and engineers features for the bank transaction data.

    Args:
        df (pd.DataFrame): The cleaned bank transaction DataFrame.
                           Expected columns: Time, V1-V28, Amount, Class.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    print("--- Starting Feature Engineering for Bank Data ---")

    # For `creditcard.csv`, 'Time' is the elapsed time in seconds from the first transaction.
    # It's usually kept as a direct numerical feature.
    # V1-V28 are already PCA transformed features and are kept as is.
    # 'Amount' is also kept as a direct numerical feature.
    # No further feature engineering is typically done on these due to their anonymized nature
    # or the dataset's specific structure.

    # Example of a simple derived feature if useful, e.g., amount per time unit (though 'Time' is elapsed time)
    # df['amount_per_time'] = df['Amount'] / (df['Time'] + 1e-6) # Adding small epsilon to avoid division by zero
    # print("Engineered 'amount_per_time' feature.")

    print("No additional features engineered for bank data (V1-V28 are PCA, Time and Amount are kept).")

    print("--- Bank Data Feature Engineering Complete ---")
    return df
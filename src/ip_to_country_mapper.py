# src/ip_to_country_mapper.py

import pandas as pd
import numpy as np
from tqdm import tqdm

def ip_to_int(ip_address):
    """Converts a dotted-decimal IP address string to a 32-bit integer."""
    if pd.isna(ip_address):
        return np.nan
    try:
        parts = list(map(int, ip_address.split('.')))
        return parts[0] * 2**24 + parts[1] * 2**16 + parts[2] * 2**8 + parts[3]
    except (AttributeError, ValueError):
        # Handle cases where ip_address might not be a string or is malformed
        return np.nan

def map_ips_to_countries(df_fraud, df_ip_country):
    """
    Maps IP addresses in the fraud DataFrame to countries using the IP range data.
    Adds an 'ip_address_int' and 'country' column to df_fraud.
    """
    print("Starting IP to Country mapping...")

    # Convert IP addresses to integer format in both dataframes
    df_fraud['ip_address_int'] = df_fraud['ip_address'].apply(ip_to_int)
    df_ip_country['lower_bound_ip_address_int'] = df_ip_country['lower_bound_ip_address'].apply(ip_to_int)
    df_ip_country['upper_bound_ip_address_int'] = df_ip_country['upper_bound_ip_address'].apply(ip_to_int)
    print("IP addresses converted to integer format.")

    # Sort IP country data for efficient lookup
    df_ip_country.sort_values(by='lower_bound_ip_address_int', inplace=True)
    # Reset index to allow iloc-based access after sorting
    df_ip_country.reset_index(drop=True, inplace=True)

    # Function to get country for a given IP integer
    def get_country(ip_int, ip_country_df_sorted):
        if pd.isna(ip_int):
            return 'Unknown'

        # Use searchsorted to find the insertion point quickly
        # 'right' means it will return index 'i' such that all v < a[i]
        idx = ip_country_df_sorted['lower_bound_ip_address_int'].searchsorted(ip_int, side='right') - 1

        if idx < 0: # IP is smaller than the smallest lower bound
            return 'Unknown'

        # Check if the IP falls within the range of the found row
        row = ip_country_df_sorted.iloc[idx]
        if ip_int >= row['lower_bound_ip_address_int'] and ip_int <= row['upper_bound_ip_address_int']:
            return row['country']
        else:
            return 'Unknown' # IP not found in any range (e.g., in a gap or outside)

    # Apply the mapping function with a progress bar
    tqdm.pandas(desc="Mapping IPs to Countries") # Enable progress_apply
    df_fraud['country'] = df_fraud['ip_address_int'].progress_apply(lambda x: get_country(x, df_ip_country))
    print("IP to Country mapping complete.")
    
    # Convert new 'country' column to categorical for memory efficiency and encoding
    df_fraud['country'] = df_fraud['country'].astype('category')

    return df_fraud

if __name__ == '__main__':
    # Example usage (for testing this module directly)
    print("--- Testing ip_to_country_mapper.py ---")

    # Create dummy dataframes
    fraud_data = {
        'user_id': [1, 2, 3, 4, 5],
        'ip_address': ['1.0.0.1', '1.0.0.254', '1.0.1.1', '1.0.3.5', '192.168.1.1'],
        'class': [0, 1, 0, 1, 0]
    }
    df_fraud_test = pd.DataFrame(fraud_data)

    ip_country_data = {
        'lower_bound_ip_address': ['1.0.0.0', '1.0.1.0', '1.0.3.0', '1.0.4.0'],
        'upper_bound_ip_address': ['1.0.0.255', '1.0.2.255', '1.0.3.255', '1.0.5.255'],
        'country': ['USA', 'Canada', 'Mexico', 'Germany']
    }
    df_ip_country_test = pd.DataFrame(ip_country_data)

    print("\nOriginal Fraud Data:")
    print(df_fraud_test)
    print("\nOriginal IP Country Data:")
    print(df_ip_country_test)

    df_fraud_mapped = map_ips_to_countries(df_fraud_test.copy(), df_ip_country_test.copy())

    print("\nFraud Data after IP to Country Mapping:")
    print(df_fraud_mapped)
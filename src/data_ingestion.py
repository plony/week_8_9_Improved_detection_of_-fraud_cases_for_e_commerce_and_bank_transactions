import pandas as pd

def load_data(file_path):
    """
    Loads a CSV file into a Pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame, or None if an error occurs.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return None

if __name__ == '__main__':
    # Example usage for testing
    print("Testing data_ingestion.py")
    # Create dummy CSVs for testing if they don't exist
    pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}).to_csv('test_data_ecommerce.csv', index=False)
    pd.DataFrame({'Time': [0, 1], 'Amount': [10, 20], 'Class': [0, 1]}).to_csv('test_data_bank.csv', index=False)

    df_eco = load_data('test_data_ecommerce.csv')
    if df_eco is not None:
        print("\nE-commerce test data head:")
        print(df_eco.head())

    df_bank = load_data('test_data_bank.csv')
    if df_bank is not None:
        print("\nBank test data head:")
        print(df_bank.head())

    import os
    os.remove('test_data_ecommerce.csv')
    os.remove('test_data_bank.csv')
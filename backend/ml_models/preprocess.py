import pandas as pd
import holidays

def load_and_preprocess_data(filepath='backend/data/crimes_2025_5k.csv'):
    """
    Loads and preprocesses the crime data from the given filepath.

    Args:
        filepath (str): The path to the crime data CSV file.

    Returns:
        pandas.DataFrame: A preprocessed DataFrame with corrected data types and holiday info.
    """
    try:
        df = pd.read_csv(filepath)

        # Convert 'Date' column to datetime objects
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M:%S %p')

        # Drop rows with missing latitude or longitude, just in case
        df.dropna(subset=['Latitude', 'Longitude'], inplace=True)

        # --- Add Holiday Information ---
        # Get the range of years in the data
        min_year, max_year = df['Date'].dt.year.min(), df['Date'].dt.year.max()
        us_holidays = holidays.US(years=range(min_year, max_year + 1))

        # Create a boolean column 'is_holiday'
        df['is_holiday'] = df['Date'].dt.date.isin(us_holidays.keys())

        # Ensure coordinates are numeric
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        df['X Coordinate'] = pd.to_numeric(df['X Coordinate'], errors='coerce')
        df['Y Coordinate'] = pd.to_numeric(df['Y Coordinate'], errors='coerce')

        # Drop rows where coordinate conversion failed
        df.dropna(subset=['X Coordinate', 'Y Coordinate'], inplace=True)

        print(f"Successfully loaded and preprocessed data from '{filepath}'.")
        print(f"Shape of the dataframe: {df.shape}")

        return df

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        return None

if __name__ == '__main__':
    preprocessed_df = load_and_preprocess_data()
    if preprocessed_df is not None:
        print("\nDataframe Info:")
        preprocessed_df.info()
        print("\nFirst 5 rows of preprocessed data:")
        print(preprocessed_df.head())
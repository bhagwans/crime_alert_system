import pandas as pd
import numpy as np
import joblib
import itertools
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .preprocess import load_and_preprocess_data
from .hotspot_detection import CityHotspotDetector  # Add this import

def train_forecasting_models():
    """
    Trains a SARIMAX forecasting model for each crime hotspot, using holiday
    data as an exogenous variable.
    """
    # Load the preprocessed data (which now includes holiday info)
    df = load_and_preprocess_data()
    if df is None:
        return

    # Load the trained CHD model
    try:
        labels = np.load('backend/saved_models/chd_labels.npy', allow_pickle=True)
        df['cluster'] = labels
    except FileNotFoundError:
        print("Error: CHD labels 'backend/saved_models/chd_labels.npy' not found.")
        print("Please run the hotspot_detection.py script first.")
        return

    print("\nStarting SARIMAX model training for each hotspot...")
    warnings.filterwarnings("ignore")  # specify to ignore warning messages

    # Get the unique cluster labels (excluding noise points, which are -1)
    hotspot_ids = sorted([label for label in df['cluster'].unique() if label != -1])

    for hotspot_id in hotspot_ids:
        print(f"\n--- Training model for Hotspot #{hotspot_id} ---")

        # Filter data for the current hotspot
        hotspot_df = df[df['cluster'] == hotspot_id].copy()

        if hotspot_df.empty:
            print(f"Hotspot #{hotspot_id} has no data points. Skipping.")
            continue

        # Aggregate crime counts and holiday data into a weekly time series
        hotspot_df.set_index('Date', inplace=True)
        time_series = hotspot_df.resample('W').size().rename("crime_count")

        # Create the exogenous variable by summing the boolean 'is_holiday' flag
        exog_data = hotspot_df.resample('W').agg({'is_holiday': 'sum'})
        exog_data = exog_data.rename(columns={'is_holiday': 'holiday_count'})

        # Align the exogenous data with the time series, filling missing values
        exog_data = exog_data.reindex(time_series.index, fill_value=0)

        # Lower the data requirement for the sample dataset
        if len(time_series) < 20:
            print(f"Not enough weekly data points ({len(time_series)}) for Hotspot #{hotspot_id}. Skipping.")
            continue

        # Use a fixed, sensible default configuration instead of a slow grid search
        # This mirrors the Spark pipeline's approach for consistency.
        params = (1, 1, 1)
        # The smaller dataset exhibits monthly seasonality (patterns every 4 weeks).
        seasonal_params = (1, 1, 1, 4)

        try:
            # Train the SARIMAX model with the exogenous holiday data
            model = SARIMAX(
                time_series,
                exog=exog_data,
                order=params,
                seasonal_order=seasonal_params,
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)

            # Save the trained model
            model_path = f'backend/saved_models/sarima_model_hotspot_{hotspot_id}.joblib'
            joblib.dump(model, model_path)
            print(f"SARIMAX model for Hotspot #{hotspot_id} saved to '{model_path}'")

        except Exception as e:
            print(f"Could not train model for Hotspot #{hotspot_id}. Error: {e}")
            continue

if __name__ == "__main__":
    train_forecasting_models()
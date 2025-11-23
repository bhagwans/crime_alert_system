from fastapi import FastAPI, HTTPException, Query
from datetime import date
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import holidays

# Imports are now relative to the backend directory
from .ml_models.hotspot_detection import CityHotspotDetector
from .ml_models.preprocess import load_and_preprocess_data


app = FastAPI(title="AI-Powered Crime Alert System")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allows the React app to make requests
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Robust Path Setup ---
# Get the absolute path to the directory containing this file (main.py)
# This makes the script runnable from any directory.
APP_DIR = Path(__file__).parent.resolve()
SAVED_MODELS_PATH = APP_DIR / "saved_models"
DATA_PATH = APP_DIR / "data"

# --- Load models and data on startup ---
try:
    labels_path = SAVED_MODELS_PATH / 'chd_labels.npy'
    # The data path can be configured here if needed
    data_path = DATA_PATH / 'crimes_2025_5k.csv'

    print(f"Attempting to load labels from: {labels_path}")
    print(f"Attempting to load data from: {data_path}")

    labels = np.load(labels_path, allow_pickle=True)
    # The loaded data now contains the 'is_holiday' column
    df_full = load_and_preprocess_data(filepath=data_path)

    # Make sure dataframe and labels align
    if len(labels) == len(df_full):
        df_full['cluster'] = labels
    else:
        # If lengths mismatch, it's safer to slice df_full to match labels
        # This can happen if preprocessing changes row counts and isn't perfectly aligned
        print(f"Warning: Mismatch in lengths. Labels: {len(labels)}, DataFrame: {len(df_full)}. Slicing DF to match labels.")
        df_full = df_full.iloc[:len(labels)].copy()
        df_full['cluster'] = labels

except FileNotFoundError:
    labels = None
    df_full = None
    print("WARNING: Models or data not found. API will have limited functionality.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI-Powered Crime Alert System API"}

@app.get("/api/hotspots")
def get_hotspots():
    """
    Returns the geographic coordinates for each detected crime hotspot.
    """
    if df_full is None:
        raise HTTPException(status_code=500, detail="Hotspot model or data is not loaded.")

    hotspots = []
    # Get unique labels, excluding noise (-1)
    unique_labels = sorted([label for label in df_full['cluster'].unique() if label != -1])

    for label in unique_labels:
        cluster_points = df_full[df_full['cluster'] == label]

        # For simplicity, we return the centroid and the points themselves
        # A more advanced implementation might return a polygon (e.g., convex hull)
        centroid = {
            "lat": cluster_points['Latitude'].mean(),
            "lon": cluster_points['Longitude'].mean()
        }

        points = cluster_points[['Latitude', 'Longitude']].to_dict(orient='records')

        hotspots.append({
            "id": int(label),
            "centroid": centroid,
            "crime_count": len(points),
            "points": points
        })

    return hotspots

@app.get("/api/forecast/{hotspot_id}")
def get_forecast(
    hotspot_id: int,
    start_date: date | None = None,
    end_date: date | None = None,
):
    """
    Returns a crime forecast for a specific hotspot using a SARIMAX model.
    """
    try:
        model_path = SAVED_MODELS_PATH / f'sarima_model_hotspot_{hotspot_id}.joblib'
        model = joblib.load(model_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Forecasting model for hotspot {hotspot_id} not found.")

    us_holidays = holidays.US()

    # This logic is mirrored from the main_spark.py for consistency
    # It creates future holiday data for the forecast period.
    def create_exog_data(start, end, freq):
        date_range = pd.date_range(start, end, freq=freq)
        exog = pd.DataFrame(index=date_range)
        exog['holiday_count'] = [
            sum(1 for day in pd.date_range(d, d + pd.DateOffset(days=6)) if day in us_holidays)
            for d in exog.index
        ]
        return exog

    # Determine and normalize forecast range to pandas Timestamps
    if not start_date or not end_date:
        start_date_ts = pd.Timestamp(date.today())
        end_date_ts = start_date_ts + pd.DateOffset(weeks=4)
    else:
        start_date_ts = pd.Timestamp(start_date)
        end_date_ts = pd.Timestamp(end_date)

    if start_date_ts > end_date_ts:
        raise HTTPException(status_code=400, detail="Start date cannot be after end date.")

    try:
        # Use the robust `forecast` method
        end_of_training_date = model.data.dates[-1]
        freq_str = model.data.freq
        if not freq_str:
            freq_str = 'W-SUN'  # Default if not found

        end_of_training_period = pd.Period(end_of_training_date, freq=freq_str)
        end_forecast_period = pd.Period(end_date_ts, freq=freq_str)
        steps = (end_forecast_period - end_of_training_period).n

        if steps <= 0:
            raise ValueError("Forecast end date must be after the training data period.")

        start_forecast_date = end_of_training_date + pd.tseries.frequencies.to_offset(freq_str)

        exog_future = create_exog_data(start_forecast_date, end_date_ts, freq=freq_str)

        # Ensure exog_future has the correct number of rows
        if len(exog_future) != steps:
            exog_future = create_exog_data(start_forecast_date, end_forecast_period.end_time, freq=freq_str)
        if len(exog_future) != steps:
            raise ValueError(f"Exogenous data shape mismatch. Required {steps}, got {len(exog_future)}")

        full_forecast = model.forecast(steps=steps, exog=exog_future)
        forecast = full_forecast.loc[start_date_ts:end_date_ts]

        if forecast.empty:
            raise ValueError("The requested date range is outside the valid forecastable period.")

        total_crimes = forecast.sum()

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not generate forecast. Error: {e}")

    response = {
        "hotspot_id": hotspot_id,
        "forecast_period": {
            "start": forecast.index[0].strftime('%Y-%m-%d'),
            "end": forecast.index[-1].strftime('%Y-%m-%d')
        },
        "total_predicted_crimes": round(total_crimes, 2),
        "forecast_breakdown": [
            {
                "date": date.strftime('%Y-%m-%d'),
                "predicted_crimes": round(value, 2),
                "is_holiday_week": any(day in us_holidays for day in pd.date_range(date, date + pd.DateOffset(days=6)))
            }
            for date, value in forecast.items()
        ]
    }
    return response

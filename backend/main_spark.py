import os
from fastapi import FastAPI, HTTPException, Query
from datetime import date
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import holidays

# This version of the API is designed to work with the outputs
# of the Spark-based ML pipeline.

app = FastAPI(title="AI-Powered Crime Alert System (SPARK)")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Robust Path Setup ---
APP_DIR = Path(__file__).parent.resolve()
SAVED_MODELS_PATH = APP_DIR / "saved_models"
CELL_SIZE = 0.001  # Must match the CELL_SIZE in the Spark script

# --- Load models and data on startup ---
hotspots_df = None
try:
    # Load the hotspot definitions from the Spark job's output
    hotspot_defs_path = SAVED_MODELS_PATH / "hotspot_definitions"
    print(f"Loading hotspot definitions from: {hotspot_defs_path}")

    # Read the Parquet file written by Spark
    hotspots_df = pd.read_parquet(hotspot_defs_path)

    print(f"Successfully loaded {len(hotspots_df)} hotspot definitions.")

except FileNotFoundError:
    print(f"WARNING: Could not load hotspot data from {hotspot_defs_path}. API will be non-functional.")
except Exception as e:
    print(f"An error occurred during initialization: {e}")


@app.get("/")
def read_root():
    return {"message": "Welcome to the AI-Powered Crime Alert System API (SPARK)"}


@app.get("/api/hotspots")
def get_hotspots():
    """
    Returns the geographic coordinates for each detected crime hotspot from the Spark pipeline.
    """
    if hotspots_df is None:
        raise HTTPException(status_code=500, detail="Hotspot model or data is not loaded.")

    hotspots_json = []
    for _, row in hotspots_df.iterrows():
        # Calculate the centroid of the grid cell
        centroid = {
            "lat": (row['hotspot_id_y'] + 0.5) * CELL_SIZE,
            "lon": (row['hotspot_id_x'] + 0.5) * CELL_SIZE,
        }

        hotspots_json.append({
            "id": int(row['hotspot_id_combined']),
            "centroid": centroid,
            "crime_count": int(row['crime_count'])
        })

    return hotspots_json


@app.get("/api/forecast/{hotspot_id}")
def get_forecast(
    hotspot_id: int,
    start_date: date | None = None,
    end_date: date | None = None,
):
    """
    Returns a crime forecast for a specific hotspot.
    """
    try:
        model_path = SAVED_MODELS_PATH / f'sarima_model_hotspot_{hotspot_id}.joblib'
        model = joblib.load(model_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Forecasting model for hotspot {hotspot_id} not found.")

    # For SARIMAX, we must provide the exogenous variables for the forecast period.
    us_holidays = holidays.US()

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
        # Use a more robust method: forecast a number of steps, then slice.
        # This avoids off-by-one errors from date range interpretation issues.

        # 1. Get the end of the training data from the model.
        end_of_training_date = model.data.dates[-1]

        # 2. Calculate the number of steps (weeks) to forecast using Period arithmetic.
        end_of_training_period = pd.Period(end_of_training_date, freq='W')
        end_forecast_period = pd.Period(end_date_ts, freq='W')
        steps = (end_forecast_period - end_of_training_period).n

        if steps <= 0:
            raise ValueError("Forecast end date must be after the training data period.")

        # 3. Generate exogenous holiday data for the exact number of steps.
        # The forecast starts on the period immediately after the training data ends.
        # `model.data.freq` is a string after loading, so it must be converted for date math.
        freq_str = model.data.freq
        if not freq_str:
            raise ValueError("Model frequency not found after loading.")

        start_forecast_date = end_of_training_date + pd.tseries.frequencies.to_offset(freq_str)
        future_date_range = pd.date_range(
            start=start_forecast_date,
            periods=steps,
            freq=freq_str
        )
        exog_future = pd.DataFrame(index=future_date_range)
        exog_future['holiday_count'] = [
            sum(1 for day in pd.date_range(d, d + pd.DateOffset(days=6)) if day in us_holidays)
            for d in exog_future.index
        ]

        # 4. Use the robust `forecast` method.
        full_forecast = model.forecast(steps=steps, exog=exog_future)

        # 5. Slice the complete forecast to the user's requested window.
        forecast = full_forecast.loc[start_date_ts:end_date_ts]

        if forecast.empty:
            raise ValueError("The requested date range is outside the valid forecastable period.")

        total_crimes = forecast.sum()

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not generate forecast for the given date range. Error: {e}")

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

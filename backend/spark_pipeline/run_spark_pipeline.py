import sys
import os
import pandas as pd
import joblib
import itertools
import holidays
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, year, month, weekofyear, dayofweek, floor, abs, hash, concat, lit, date_format, min, max
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, IntegerType, DateType

def main():
    """
    Runs the full ML pipeline using Spark to handle large datasets.
    This includes preprocessing, hotspot detection, and forecast model training.

    Usage:
    spark-submit backend/spark_pipeline/run_spark_pipeline.py <path_to_input_parquet> <path_to_output_models_dir>

    Example:
    spark-submit backend/spark_pipeline/run_spark_pipeline.py backend/data/crimes.parquet backend/saved_models
    """
    if len(sys.argv) != 3:
        print("Usage: spark-submit run_spark_pipeline.py <input_parquet_path> <output_models_path>", file=sys.stderr)
        sys.exit(-1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Initialize a Spark Session
    spark = (
        SparkSession.builder
        .appName("CrimeHotspotPipeline")
        .master("local[*]")
        .config("spark.driver.memory", "8g")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")  # For date parsing
        .getOrCreate()
    )

    print(f"Spark Session started. Reading Parquet data from: {input_path}")

    # 1. Load Data
    df = spark.read.parquet(input_path)

    # 2. Preprocessing
    print("Starting preprocessing...")

    # Drop rows with nulls in critical columns
    df = df.dropna(subset=["Latitude", "Longitude", "Date"])

    # Convert date string to timestamp
    df = df.withColumn("Timestamp", to_timestamp(col("Date"), 'MM/dd/yyyy hh:mm:ss a'))

    # Filter out records with invalid coordinates
    df = df.filter((col("Latitude") != 0) & (col("Longitude") != 0))

    # Extract date/time features
    df = df.withColumn("Year", year(col("Timestamp")))
    df = df.withColumn("Month", month(col("Timestamp")))
    df = df.withColumn("Week", weekofyear(col("Timestamp")))
    df = df.withColumn("DayOfWeek", dayofweek(col("Timestamp")))

    # For now, let's select the relevant columns and cache the result
    processed_df = df.select(
        "Timestamp", "Latitude", "Longitude", "Primary Type", "Year", "Week"
    ).cache()

    print(f"Preprocessing complete. Processed {processed_df.count()} records.")

    # 2.1. Generate and Join Holiday Data
    print("Generating and joining holiday data...")
    min_year, max_year = processed_df.select(year(col("Timestamp")).alias("year")).agg(min("year"), max("year")).first()

    us_holidays = holidays.US(years=range(min_year, max_year + 1))
    holiday_dates = [(date,) for date in us_holidays.keys()]
    holidays_df = spark.createDataFrame(holiday_dates, StructType([StructField("holiday_date", DateType(), True)]))

    # Add a date column to the main df to join on
    processed_df = processed_df.withColumn("date_only", date_format(col("Timestamp"), "yyyy-MM-dd"))

    # Join the holiday data
    enriched_df = processed_df.join(holidays_df, processed_df.date_only == holidays_df.holiday_date, "left")
    enriched_df = enriched_df.withColumn("is_holiday", col("holiday_date").isNotNull())
    enriched_df.cache()

    print("Holiday data joined successfully.")

    # 3. Grid-Based Hotspot Detection
    print("Starting grid-based hotspot detection...")

    # Define cell size. 0.001 degrees is approx 111 meters.
    CELL_SIZE = 0.001

    # Assign each crime to a grid cell
    grid_df = enriched_df.withColumn("grid_x", floor(col("Longitude") / CELL_SIZE)) \
        .withColumn("grid_y", floor(col("Latitude") / CELL_SIZE))

    # Count crimes in each cell
    cell_counts = grid_df.groupBy("grid_x", "grid_y").count()

    # Identify hotspots by finding cells with crime counts in the top 95th percentile
    crime_threshold = cell_counts.approxQuantile("count", [0.95], 0.01)[0]
    print(f"Identified crime count threshold for hotspots: {crime_threshold}")

    hotspot_cells = cell_counts.filter(col("count") > crime_threshold)

    # For now, we will treat each cell as a distinct hotspot.
    hotspot_cells = hotspot_cells.withColumnRenamed("count", "crime_count") \
        .withColumnRenamed("grid_x", "hotspot_id_x") \
        .withColumnRenamed("grid_y", "hotspot_id_y")

    hotspot_cells = hotspot_cells.withColumn(
        "hotspot_id_combined",
        abs(hash(concat(col("hotspot_id_x"), lit("_"), col("hotspot_id_y")))) % 100000
    )

    hotspot_cells.cache()

    print(f"Found {hotspot_cells.count()} hotspot cells.")
    print("Top 10 hotspot cells:")
    hotspot_cells.sort(col("crime_count").desc()).show(10)

    # Save the hotspot cell definitions for the API to use
    hotspot_defs_path = os.path.join(output_path, "hotspot_definitions")
    print(f"Writing hotspot definitions to: {hotspot_defs_path}")
    hotspot_cells.write.mode("overwrite").parquet(hotspot_defs_path)

    # 4. Parallel Forecasting Model Training
    print("Starting parallel forecast model training...")

    # Join crime data with hotspot cells to get crimes that occurred in hotspots
    hotspot_crimes_df = grid_df.join(
        hotspot_cells,
        (grid_df.grid_x == hotspot_cells.hotspot_id_x) & (grid_df.grid_y == hotspot_cells.hotspot_id_y),
        "inner"
    )

    # Define the schema for the output of the training function
    training_results_schema = StructType([
        StructField("hotspot_id_x", IntegerType(), True),
        StructField("hotspot_id_y", IntegerType(), True),
        StructField("training_successful", BooleanType(), True),
        StructField("error_message", StringType(), True),
    ])

    def train_sarima_model(pdf: pd.DataFrame) -> pd.DataFrame:
        """
        A pandas UDF that trains a SARIMA model for a single hotspot.
        This function is applied in parallel to each hotspot group by Spark.
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        import warnings
        warnings.filterwarnings("ignore")

        hotspot_x = pdf['hotspot_id_x'].iloc[0]
        hotspot_y = pdf['hotspot_id_y'].iloc[0]
        hotspot_id_combined = pdf['hotspot_id_combined'].iloc[0]

        try:
            pdf['Timestamp'] = pd.to_datetime(pdf['Timestamp'])
            pdf = pdf.set_index('Timestamp')
            time_series = pdf.resample('W').size().rename("crime_count")

            exog_data = pdf.resample('W').agg({'is_holiday': 'sum'})
            exog_data = exog_data.rename(columns={'is_holiday': 'holiday_count'})
            exog_data = exog_data.reindex(time_series.index, fill_value=0)

            params = (1, 1, 1)
            seasonal_params = (1, 1, 1, 52)

            model = SARIMAX(
                time_series,
                exog=exog_data,
                order=params,
                seasonal_order=seasonal_params,
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)

            model_path = os.path.join(output_path, f'sarima_model_hotspot_{hotspot_id_combined}.joblib')
            joblib.dump(model, model_path)

            return pd.DataFrame([[hotspot_x, hotspot_y, True, None]], columns=["hotspot_id_x", "hotspot_id_y", "training_successful", "error_message"])

        except Exception as e:
            return pd.DataFrame([[hotspot_x, hotspot_y, False, str(e)]], columns=["hotspot_id_x", "hotspot_id_y", "training_successful", "error_message"])

    training_results = hotspot_crimes_df.groupBy("hotspot_id_x", "hotspot_id_y").applyInPandas(train_sarima_model, schema=training_results_schema)

    results_path = os.path.join(output_path, "training_results")
    print(f"Writing training results to: {results_path}")

    training_results.write.mode("overwrite").csv(results_path, header=True)

    print("Pipeline finished successfully. Models saved to:", output_path)
    spark.stop()

if __name__ == "__main__":
    main()

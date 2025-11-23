import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType, IntegerType

def main():
    """
    Converts a large CSV crime dataset into the more efficient Parquet format.

    Usage:
    spark-submit backend/spark_pipeline/convert_to_parquet.py <path_to_input_csv> <path_to_output_parquet>

    Example:
    spark-submit backend/spark_pipeline/convert_to_parquet.py backend/data/crimes_full.csv backend/data/crimes.parquet
    """
    if len(sys.argv) != 3:
        print("Usage: spark-submit convert_to_parquet.py <input_csv_path> <output_parquet_path>", file=sys.stderr)
        sys.exit(-1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Initialize a Spark Session
    # In "local[*]" mode, Spark will use as many worker threads as there are logical cores on your machine.
    spark = (
        SparkSession.builder
        .appName("CrimeDataToParquet")
        .master("local[*]")
        .config("spark.driver.memory", "8g")  # Example memory config, adjust based on your hardware
        .getOrCreate()
    )

    print(f"Spark Session started. Reading CSV from: {input_path}")

    # Define the schema to ensure data types are correct and improve read performance
    schema = StructType([
        StructField("ID", IntegerType(), True),
        StructField("Case Number", StringType(), True),
        StructField("Date", StringType(), True),  # Read as string, then convert
        StructField("Block", StringType(), True),
        StructField("IUCR", StringType(), True),
        StructField("Primary Type", StringType(), True),
        StructField("Description", StringType(), True),
        StructField("Location Description", StringType(), True),
        StructField("Arrest", StringType(), True),
        StructField("Domestic", StringType(), True),
        StructField("Beat", StringType(), True),
        StructField("District", StringType(), True),
        StructField("Ward", StringType(), True),
        StructField("Community Area", StringType(), True),
        StructField("FBI Code", StringType(), True),
        StructField("X Coordinate", DoubleType(), True),
        StructField("Y Coordinate", DoubleType(), True),
        StructField("Year", IntegerType(), True),
        StructField("Updated On", StringType(), True),
        StructField("Latitude", DoubleType(), True),
        StructField("Longitude", DoubleType(), True),
        StructField("Location", StringType(), True),
    ])

    # Read the CSV file
    df = spark.read.csv(input_path, header=True, schema=schema)

    # Coalesce into a smaller number of partitions before writing if needed.
    # This can be useful to avoid creating too many small files.
    # For a very large dataset, you might want more partitions.
    # Let's start with a reasonable number like 8.
    df = df.coalesce(8)

    print(f"CSV data loaded. Writing to Parquet format at: {output_path}")

    # Write the DataFrame to Parquet format
    # "overwrite" mode will replace any existing data at the output path.
    df.write.mode("overwrite").parquet(output_path)

    print("Parquet conversion complete.")

    # Stop the Spark Session
    spark.stop()

if __name__ == "__main__":
    main()

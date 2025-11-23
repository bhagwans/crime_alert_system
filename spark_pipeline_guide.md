# Guide: Scalable ML Pipeline with Spark and Parquet

This guide details the setup and execution of the scalable machine learning pipeline built with Apache Spark. This pipeline is designed to handle datasets far too large to fit in memory (e.g., 9+ million records).

## 1. Core Concepts

- **Apache Spark**: A distributed computing engine that processes data in parallel across all available CPU cores. It avoids loading the entire dataset into memory.
- **Parquet**: A highly efficient, compressed, columnar data format. It is the standard for big data processing and is much faster for Spark to read than CSV.
- **Decoupled Architecture**: The system is split into two main parts:
1. An **offline Spark pipeline** that does the heavy lifting (data processing and model training).
2. A **lightweight FastAPI server** that serves the pre-computed results to the frontend.

## 2. One-Time Environment Setup

You only need to perform these steps once.

### Step 2.1: Install Java (Required for Spark)

Spark runs on the Java Virtual Machine (JVM) and requires a specific version of the Java Development Kit (JDK).

```bash
# Install OpenJDK 17
sudo apt-get update && sudo apt-get install -y openjdk-17-jdk

# Verify the installation (optional)
java -version
```

### Step 2.2: Install Python Dependencies

The project requires `pyspark` and `pyarrow` for the scalable pipeline.

```bash
# Ensure you are in your project's virtual environment
pip install -r requirements.txt
```

## 3. The Three-Step Workflow

Once the setup is complete, this is the standard workflow for running the scalable application.

### Step 3.1: Convert Data to Parquet (One-Time per Dataset)

This step takes your large CSV file and converts it into the efficient Parquet format. You only need to do this once for each new dataset.

**Command (run from project root):**

```bash
# Usage: spark-submit <script_path> <input_csv> <output_parquet>
spark-submit backend/spark_pipeline/convert_to_parquet.py backend/data/crimes_full.csv backend/data/crimes.parquet
```

This will create a `backend/data/crimes.parquet` directory containing the converted data.

### Step 3.2: Run the Spark ML Pipeline

This is the main event. This script reads the Parquet data, finds hotspots, and trains a forecast model for each one in parallel.

**Command (run from project root):**

```bash
# Usage: spark-submit <script_path> <input_parquet> <output_models_dir>
spark-submit backend/spark_pipeline/run_spark_pipeline.py backend/data/crimes.parquet backend/saved_models
```

This process is CPU-intensive and can take a significant amount of time. It will populate the `backend/saved_models` directory with:
- `sarima_model_hotspot_*.joblib` files (the trained models).
- A `hotspot_definitions/` directory (containing hotspot locations and crime counts).
- A `training_results/` directory (containing logs of the training process).

### Step 3.3: Run the Spark API Server

After the pipeline has finished successfully, run the dedicated API server for the Spark results.

**Command (run from project root):**

```bash
uvicorn backend.main_spark:app --reload --host 0.0.0.0 --port 8000
```

Your frontend application (running on `http://localhost:3000`) will now be able to fetch hotspot and forecast data from this scalable backend.
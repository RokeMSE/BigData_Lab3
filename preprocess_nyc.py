import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, TimestampType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

def preprocess_data(spark, input_path, output_path_train, output_path_val):
    """
    This function loads the raw NYC Taxi dataset, performs cleaning, feature engineering,
    and preprocessing, then saves the training and validation sets.
    """
    
    print("Starting data preprocessing...")

    # 1. Load Data
    # =================
    print(f"Loading data from {input_path}")
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # 2. Data Cleaning & Type Casting
    # ================================
    print("Cleaning and casting data types...")
    # Cast datetime columns from string to timestamp
    df = df.withColumn("pickup_datetime", F.col("pickup_datetime").cast(TimestampType()))

    # Filter out trips with no passengers or an excessive amount
    df = df.filter((F.col("passenger_count") > 0) & (F.col("passenger_count") < 7))

    # Filter out trips with trip_duration outside a reasonable range (e.g., > 1 min and < 2 hours)
    df = df.filter((F.col("trip_duration") > 60) & (F.col("trip_duration") < 7200))

    # Filter out trips with invalid coordinates (e.g., outside NYC area)
    nyc_longitude_bounds = (-74.05, -73.75)
    nyc_latitude_bounds = (40.58, 40.90)
    df = df.filter(
        (F.col("pickup_longitude").between(*nyc_longitude_bounds)) &
        (F.col("pickup_latitude").between(*nyc_latitude_bounds)) &
        (F.col("dropoff_longitude").between(*nyc_longitude_bounds)) &
        (F.col("dropoff_latitude").between(*nyc_latitude_bounds))
    )

    # 3. Feature Engineering
    # ======================
    print("Performing feature engineering...")

    # Time-based features
    df = df.withColumn("pickup_hour", F.hour("pickup_datetime"))
    df = df.withColumn("pickup_dayofweek", F.dayofweek("pickup_datetime")) # 1=Sun, 2=Mon, ...

    # Geospatial feature: Haversine distance
    # The Haversine formula calculates the distance between two points on a sphere.
    @F.udf(returnType=DoubleType())
    def haversine_distance(lon1, lat1, lon2, lat2):
        import math
        R = 6371.0  # Earth radius in kilometers

        lon1_rad, lat1_rad = math.radians(lon1), math.radians(lat1)
        lon2_rad, lat2_rad = math.radians(lon2), math.radians(lat2)

        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad

        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return distance

    df = df.withColumn("distance_km", haversine_distance(
        F.col("pickup_longitude"), F.col("pickup_latitude"),
        F.col("dropoff_longitude"), F.col("dropoff_latitude")
    ))

    # Rename the target column to 'label' which is the default for Spark ML
    df = df.withColumn("label", F.col("trip_duration"))
    
    # Filter trips with distance > 0
    df = df.filter(F.col("distance_km") > 0)

    # 4. Preprocessing
    # =========================
    print("Building and applying the preprocessing pipeline...")
    
    # Identify categorical and numerical columns for the model
    categorical_cols = ["vendor_id", "pickup_hour", "pickup_dayofweek", "store_and_fwd_flag"]
    numerical_cols = ["passenger_count", "distance_km", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]

    # Create pipeline stages for categorical features
    # StringIndexer converts string categories to numerical indices.
    # OneHotEncoder converts these indices into sparse binary vectors.
    indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep") for col in categorical_cols]
    encoders = [OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_vec") for col in categorical_cols]

    # Assemble all processed features into a single vector
    assembler_inputs = [f"{col}_vec" for col in categorical_cols] + numerical_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

    # Define the full pipeline
    pipeline = Pipeline(stages=indexers + encoders + [assembler])

    # Fit the pipeline to the data and transform it
    pipeline_model = pipeline.fit(df)
    processed_df = pipeline_model.transform(df)

    # Select only the necessary columns for the model
    final_df = processed_df.select("features", "label")
    final_df.show(5, truncate=False)

    # 5. Train-Test Split
    # ===================
    print("Splitting data into training and validation sets (80/20)...")
    train_df, val_df = final_df.randomSplit([0.8, 0.2], seed=42)

    # 6. Save Processed Data
    # ======================
    print(f"Saving training data to {output_path_train}")
    train_df.write.mode("overwrite").parquet(output_path_train)

    print(f"Saving validation data to {output_path_val}")
    val_df.write.mode("overwrite").parquet(output_path_val)

    print("Preprocessing finished successfully! 🎉")


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("NYC Taxi Preprocessing") \
        .master("local[*]") \
        .getOrCreate()

    # Input and output paths
    INPUT_PATH = "./nyc-taxi-trip-duration/train/train.csv"
    OUTPUT_TRAIN_PATH = "processed_data/train"
    OUTPUT_VAL_PATH = "processed_data/validation"

    # Run the preprocessing function (need it before training)
    preprocess_data(spark, INPUT_PATH, OUTPUT_TRAIN_PATH, OUTPUT_VAL_PATH)

    # REMEBER TO STOP
    spark.stop()


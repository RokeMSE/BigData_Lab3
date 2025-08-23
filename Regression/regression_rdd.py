import os
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.mllib.linalg import Vectors
import datetime
from io import StringIO
import csv

def parse_csv_line(line):
    """
    Uses the csv module to properly handle quoted fields.
    """
    try:
        reader = csv.reader(StringIO(line))
        return next(reader)
    except StopIteration:
        return None

def parse_row(cols):
    """
    Parses a list of strings from a CSV row into a structured tuple.
    Returns (id, duration, features_vector) or None if parsing fails.
    """
    try:
        id_ = cols[0]
        vendor = float(cols[1])
        pickup_dt = datetime.datetime.strptime(cols[2], "%Y-%m-%d %H:%M:%S")
        pass_count = float(cols[4])
        pick_long = float(cols[5])
        pick_lat = float(cols[6])
        drop_long = float(cols[7])
        drop_lat = float(cols[8])
        store_fwd = 1.0 if cols[9].strip().upper() == "Y" else 0.0
        duration = float(cols[10])
        
        # Feature Engineering: Extract hour from pickup time
        pickup_hour = float(pickup_dt.hour)
        
        features = [vendor, pass_count, pick_long, pick_lat, drop_long, drop_lat, pickup_hour, store_fwd]
        return (id_, duration, Vectors.dense(features))
    except (ValueError, IndexError):
        # Skip malformed rows
        return None

def main():
    """
    Main function for the RDD-based regression task.
    """
    # Initialize Spark Session and Context
    spark = SparkSession.builder.appName("Regression_RDD_MLlib_Fixed").getOrCreate()
    sc = spark.sparkContext

    # --- Data Loading and Preparation ---
    try:
        lines = sc.textFile("./nyc-taxi-trip-duration/train/train.csv")
        header = lines.first()
        data = lines.filter(lambda line: line != header)
    except Exception as e:
        print(f"Error loading data: {e}")
        sc.stop()
        return

    # Parse data and filter out any rows that failed parsing
    parsed_data = data.map(parse_csv_line).filter(lambda x: x is not None)
    rdd_data = parsed_data.map(parse_row).filter(lambda x: x is not None)
    rdd_data.cache()

    # --- Data Splitting ---
    # Perform a single split to ensure train and test sets are perfectly aligned
    train_full_rdd, test_full_rdd = rdd_data.randomSplit([0.8, 0.2], seed=42)

    # Create LabeledPoint RDD for training
    train_rdd = train_full_rdd.map(lambda x: LabeledPoint(x[1], x[2]))

    # --- Model Training ---
    model = DecisionTree.trainRegressor(train_rdd, categoricalFeaturesInfo={}, 
                                        impurity="variance", maxDepth=5)

    # --- Model Prediction (Serialization-Safe Method) ---
    # 1. Extract just the features from the test set
    test_features_rdd = test_full_rdd.map(lambda x: x[2])
    
    # 2. Use model.predict on the RDD of features. This is the correct way.
    predictions_rdd = model.predict(test_features_rdd)

    # 3. Zip the original test data with the predictions
    # The result is an RDD of ((id, label, features), prediction)
    results_zipped_rdd = test_full_rdd.zip(predictions_rdd)

    # Map to a more convenient format for DataFrame creation and evaluation
    results_rdd = results_zipped_rdd.map(lambda p: {
        "id": p[0][0],
        "label": p[0][1],
        "prediction": p[1],
        "residual": p[0][1] - p[1]
    })
    
    # --- Evaluation ---
    # Create an RDD of (prediction, label) for the metrics evaluator
    preds_and_labels = results_zipped_rdd.map(lambda p: (p[1], p[0][1]))
    metrics = RegressionMetrics(preds_and_labels)

    print("--- Evaluation Metrics ---")
    print(f"RMSE: {metrics.rootMeanSquaredError}")
    print(f"R²: {metrics.r2}")
    print(f"MAE: {metrics.meanAbsoluteError}")
    print("--------------------------")

    # --- Save Results ---
    output_dir = "Results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert to DataFrame and save as a single CSV file
    try:
        df = spark.createDataFrame(results_rdd)
        df.coalesce(1).write.csv("Results/Regression_RDD_MLlib.csv", header=True, mode="overwrite")
        print("Successfully saved predictions to Results/Regression_RDD_MLlib.csv")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

    sc.stop()

if __name__ == "__main__":
    main()

import os
import math
from pyspark.sql import SparkSession
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree

def save_results(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        print(f"Saving results to {output_path}")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print("Results saved successfully.")

def run_regression_rdd_api(data_path):
    print("Starting Regression with RDD API")

    spark = SparkSession.builder \
        .appName("NYC Taxi Regression RDD") \
        .master("local[*]") \
        .getOrCreate()

    df = spark.read.csv(data_path, header=True, inferSchema=True)
    df = df.filter("passenger_count > 0 AND trip_duration > 60 AND trip_duration < 7200")
    df = df.filter("pickup_longitude is not null and pickup_latitude is not null and dropoff_longitude is not null and dropoff_latitude is not null")

    def row_to_labeled_point(row):
        label = float(row['trip_duration'])
        features = [
            float(row['passenger_count']),
            float(row['pickup_longitude']),
            float(row['pickup_latitude']),
            float(row['dropoff_longitude']),
            float(row['dropoff_latitude'])
        ]
        return LabeledPoint(label, features)

    data_rdd = df.rdd.map(row_to_labeled_point)
    train_rdd, test_rdd = data_rdd.randomSplit([0.8, 0.2], seed=42)
    train_rdd.cache()
    test_rdd.cache()

    model = DecisionTree.trainRegressor(
        train_rdd,
        categoricalFeaturesInfo={},
        impurity="variance",
        maxDepth=5,
        maxBins=32
    )

    predictions = model.predict(test_rdd.map(lambda x: x.features))
    labels_and_predictions = test_rdd.map(lambda lp: lp.label).zip(predictions)
    
    mse = labels_and_predictions.map(lambda lp: (lp[0] - lp[1])**2).mean()
    rmse = math.sqrt(mse)
    
    mean_label = test_rdd.map(lambda lp: lp.label).mean()
    ss_total = test_rdd.map(lambda lp: (lp.label - mean_label) ** 2).sum()
    ss_res = labels_and_predictions.map(lambda lp: (lp[0] - lp[1]) ** 2).sum()
    r2 = 1 - (ss_res / ss_total) if ss_total != 0 else float('nan')
    
    # Results
    print("\n--- Evaluation Metrics ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    results_dict = {
        "Model": "Decision Tree Regressor (RDD API)",
        "Dataset": "NYC Taxi Trip Duration",
        "Root Mean Squared Error (RMSE)": f"{rmse:.4f}",
        "R2 Score": f"{r2:.4f}"
    }
    save_results(results_dict, "results/regression_rdd_results.txt")

    # REMEMBER TO STOP
    spark.stop()


if __name__ == "__main__":
    TAXI_DATA_PATH = "./nyc-taxi-trip-duration/train/train.csv"
    run_regression_rdd_api(TAXI_DATA_PATH)

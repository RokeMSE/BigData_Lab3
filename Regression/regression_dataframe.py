import os
from pyspark.sql import SparkSession
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator

def save_results(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        print(f"Saving results to {output_path}")
        for key, value in results.items():
            # Handle special case for long tree string
            if key == "Decision Tree Model Structure":
                f.write(f"{key}:\n{value}\n")
            else:
                f.write(f"{key}: {value}\n")
    print("Results saved successfully.")

def run_regression_dataframe_api(train_path, val_path):
    print("Regression with Structured API")

    spark = SparkSession.builder \
        .appName("NYC Taxi Regression DF") \
        .master("local[*]") \
        .getOrCreate()

    train_df = spark.read.parquet(train_path)
    val_df = spark.read.parquet(val_path)

    dt = DecisionTreeRegressor(featuresCol="features", labelCol="label", maxDepth=5)
    model = dt.fit(train_df)
    predictions = model.transform(val_df)

    # Evaluation
    # RMSE
    rmse_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = rmse_evaluator.evaluate(predictions)
    # R2
    r2_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
    r2 = r2_evaluator.evaluate(predictions)

    # Results
    print("\n--- Evaluation Metrics ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")

    # Inspect Model
    print("\n--- Model Inspection ---")
    print(f"Feature Importances: {model.featureImportances}")
    print("Decision Tree Model Structure (Debug String):")
    print(model.toDebugString)

    results_dict = {
        "Model": "Decision Tree Regressor (DataFrame API)",
        "Dataset": "NYC Taxi Trip Duration",
        "Root Mean Squared Error (RMSE)": f"{rmse:.4f}",
        "R-squared (R²)": f"{r2:.4f}",
        "Feature Importances": model.featureImportances,
        "Decision Tree Model Structure": model.toDebugString
    }
    save_results(results_dict, "results/regression_dataframe_results.txt")

    # REMEMBER TO STOP
    spark.stop()


if __name__ == "__main__":
    PROCESSED_TRAIN_PATH = "processed_data/train"
    PROCESSED_VAL_PATH = "processed_data/validation"
    run_regression_dataframe_api(PROCESSED_TRAIN_PATH, PROCESSED_VAL_PATH)

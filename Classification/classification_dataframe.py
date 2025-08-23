import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

def save_results(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        print(f"Saving results to {output_path}")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print("Results saved successfully.")

def run_classification_dataframe_api(data_path):
    print("Classification with Structured API")

    spark = SparkSession.builder \
        .appName("Fraud Detection DF") \
        .master("local[*]") \
        .getOrCreate()

    df = spark.read.csv(data_path, header=True, inferSchema=True)
    df = df.withColumn("label", col("Class").cast("double")).drop("Class")

    feature_cols = [c for c in df.columns if c not in ['label', 'Time']]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
    
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    lr = LogisticRegression(featuresCol="features", labelCol="label")
    pipeline = Pipeline(stages=[assembler, scaler, lr])
    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)

    # Evaluation
    binary_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc = binary_evaluator.evaluate(predictions)

    multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    accuracy = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "accuracy"})
    precision = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedPrecision"})
    recall = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedRecall"})
    
    # Results
    print("\n--- Evaluation Metrics ---")
    print(f"Area Under ROC (AUC): {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Inspect Model
    lr_model = model.stages[-1]
    print("\n--- Model Inspection ---")
    print(f"Intercept: {lr_model.intercept}")
    print(f"Coefficients (first 5): {lr_model.coefficients.values[:5]}")

    results_dict = {
        "Model": "Logistic Regression (DataFrame API)",
        "Dataset": "Credit Card Fraud",
        "Area Under ROC (AUC)": f"{auc:.4f}",
        "Accuracy": f"{accuracy:.4f}",
        "Precision": f"{precision:.4f}",
        "Recall": f"{recall:.4f}",
        "Intercept": lr_model.intercept,
        "Coefficients (first 5)": lr_model.coefficients.values[:5].tolist()
    }
    save_results(results_dict, "results/classification_dataframe_results.txt")

    # REMEMBER TO STOP
    spark.stop()


if __name__ == "__main__":
    CREDIT_CARD_DATA_PATH = "./creditcard.csv/creditcard.csv"
    run_classification_dataframe_api(CREDIT_CARD_DATA_PATH)

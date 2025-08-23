import os
from pyspark.sql import SparkSession
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics

def save_results(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        print(f"Saving results to {output_path}")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print("Results saved successfully.")

def run_classification_rdd_api(data_path):
    print("Classification with RDD API")

    spark = SparkSession.builder \
        .appName("Fraud Detection RDD") \
        .master("local[*]") \
        .getOrCreate()

    df = spark.read.csv(data_path, header=True, inferSchema=True)
    feature_cols = [c for c in df.columns if c not in ['Time', 'Class']]
    data_rdd = df.select(['Class'] + feature_cols).rdd.map(
        lambda row: LabeledPoint(float(row['Class']), [float(c) for c in row[1:]])
    )
    
    train_rdd, test_rdd = data_rdd.randomSplit([0.8, 0.2], seed=42)
    train_rdd.cache()
    test_rdd.cache()

    model = LogisticRegressionWithLBFGS.train(train_rdd, iterations=100)
    
    predictions_and_labels = test_rdd.map(lambda lp: (float(model.predict(lp.features)), lp.label))

    # Evaluation
    binary_metrics = BinaryClassificationMetrics(predictions_and_labels)
    auc = binary_metrics.areaUnderROC
    
    multi_metrics = MulticlassMetrics(predictions_and_labels)
    accuracy = multi_metrics.accuracy
    precision = multi_metrics.weightedPrecision
    recall = multi_metrics.weightedRecall

    # Results
    print("\n--- Evaluation Metrics ---")
    print(f"Area Under ROC (AUC): {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    results_dict = {
        "Model": "Logistic Regression (RDD API)",
        "Dataset": "Credit Card Fraud",
        "Area Under ROC (AUC)": f"{auc:.4f}",
        "Accuracy": f"{accuracy:.4f}",
        "Precision": f"{precision:.4f}",
        "Recall": f"{recall:.4f}"
    }
    save_results(results_dict, "results/classification_rdd_results.txt")

    # REMEMBER TO STOP
    spark.stop()


if __name__ == "__main__":
    CREDIT_CARD_DATA_PATH = "./creditcard.csv/creditcard.csv"
    run_classification_rdd_api(CREDIT_CARD_DATA_PATH)

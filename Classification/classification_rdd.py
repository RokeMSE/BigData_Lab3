# Note: This is a complete PySpark script for the Classification task using RDD-MLlib (spark.mllib).
# Run this in a PySpark environment.
# Assumptions: creditcard.csv is in the current directory.
# Output: Saves predictions to Results/Classification_RDD_MLlib.csv
# Also prints evaluation metrics.

from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.mllib.linalg import Vectors
from pyspark.sql import SparkSession
import os
import csv
from io import StringIO

# Create Spark context
spark = SparkSession.builder.appName("Regression_RDD_MLlib_Fixed").getOrCreate()
sc = spark.sparkContext

# Load and parse data
lines = sc.textFile("creditcard.csv")
header = lines.first()
data = lines.filter(lambda line: line != header)

# Parse CSV handling quoted values and potential whitespace
def parse_csv_line(line):
    # Use csv module for proper CSV parsing
    reader = csv.reader(StringIO(line))
    row = next(reader)
    return [float(x) for x in row]

parsed = data.map(parse_csv_line)

# Create RDD of (Time, Amount, Class, features as dense vector: V1-V28 + Amount)
# Index: Time=0, V1=1 to V28=28, Amount=29, Class=30 (0-based)
rdd_data = parsed.map(lambda cols: (cols[0], cols[29], cols[30], Vectors.dense(cols[1:29] + [cols[29]])))

# Create LabeledPoint: label=Class, features
labeled_rdd = rdd_data.map(lambda x: LabeledPoint(x[2], x[3]))

# Split data properly
train_rdd, test_rdd = labeled_rdd.randomSplit([0.8, 0.2], seed=42)

# Also split the full data to maintain correspondence
full_train_rdd, full_test_rdd = rdd_data.randomSplit([0.8, 0.2], seed=42)

# Train model
model = LogisticRegressionWithLBFGS.train(train_rdd, iterations=100)

# Clear threshold to get probabilities
model.clearThreshold()

# Get predictions and probabilities
test_features = test_rdd.map(lambda lp: lp.features)
probs = test_features.map(lambda features: model.predict(features))
preds = probs.map(lambda p: 1.0 if p > 0.5 else 0.0)

# Create results by zipping with full test data
# Fixed lambda syntax for Python 3
results_rdd = full_test_rdd.zip(probs).zip(preds).map(
    lambda x: (x[0][0][0], x[0][0][1], x[0][0][2], x[1], x[0][1])
)

# Create Results directory if it doesn't exist
os.makedirs("Results", exist_ok=True)

# Convert to DataFrame and save
df = spark.createDataFrame(results_rdd, ["Time", "Amount", "Class", "prediction", "fraud_probability"])
df.coalesce(1).write.csv("Results/Classification_RDD_MLlib.csv", header=True, mode="overwrite")

# Evaluate model
# For binary classification metrics (needs (score, label) format)
test_labels = test_rdd.map(lambda lp: lp.label)
scores_and_labels = probs.zip(test_labels)
binary_metrics = BinaryClassificationMetrics(scores_and_labels)
print("AUC:", binary_metrics.areaUnderROC)

# For multiclass metrics (needs (prediction, label) format)  
preds_and_labels = preds.zip(test_labels)
multi_metrics = MulticlassMetrics(preds_and_labels)
print("Accuracy:", multi_metrics.accuracy)
print("Precision:", multi_metrics.weightedPrecision)
print("Recall:", multi_metrics.weightedRecall)

# Reset threshold and stop
model.setThreshold(0.5)
sc.stop()
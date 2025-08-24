from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import udf, col, when
from pyspark.sql.types import DoubleType
import os

spark = SparkSession.builder.appName("Classification Structured").getOrCreate()

# Load data
df = spark.read.csv("./creditcard.csv/creditcard.csv", header=True, inferSchema=True)

# Assemble features
input_cols = [c for c in df.columns if c not in ["Class", "Time"]]
assembler = VectorAssembler(inputCols=input_cols, outputCol="assembled_features")
data = assembler.transform(df)

# Standardize features
scaler = StandardScaler(inputCol="assembled_features", outputCol="features", withStd=True, withMean=False)
scaler_model = scaler.fit(data)
data = scaler_model.transform(data)

data = data.withColumnRenamed("Class", "label")

# Handle class imbalance with classWeightCol
balancing_ratio = data.filter(col("label") == 1).count() / data.count()
data = data.withColumn("classWeight", when(col("label") == 1, 1 - balancing_ratio).otherwise(balancing_ratio))

# Train-test split
train, test = data.randomSplit([0.8, 0.2], seed=42)

# Train model
lr = LogisticRegression(featuresCol="features", labelCol="label", weightCol="classWeight")
model = lr.fit(train)

# Print model coefficients and intercept
print("Coefficients: \n" + str(model.coefficients))
print("Intercept: " + str(model.intercept))

# Predictions
predictions = model.transform(test)

# Evaluate
auc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = auc_eval.evaluate(predictions)
acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
acc = acc_eval.evaluate(predictions)
prec_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
prec = prec_eval.evaluate(predictions)
rec_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
rec = rec_eval.evaluate(predictions)

print(f"AUC: {auc}")
print(f"Accuracy: {acc}")
print(f"Precision: {prec}")
print(f"Recall: {rec}")

# Export predictions
extract_prob = udf(lambda x: float(x[1]), DoubleType())
preds = predictions.select("Time", "Amount", col("label").alias("Class"), "prediction", extract_prob("probability").alias("fraud_probability"))

os.makedirs("Results", exist_ok=True)
preds.coalesce(1).write.csv("Results/Classification_Structured.csv", header=True, mode="overwrite")

# Export metrics to a text file
with open("Results/Classification_Structured_metrics.txt", "w") as f:
    f.write(f"Coefficients: {model.coefficients}\n")
    f.write(f"Intercept: {model.intercept}\n")
    f.write(f"AUC: {auc}\n")
    f.write(f"Accuracy: {acc}\n")
    f.write(f"Precision: {prec}\n")
    f.write(f"Recall: {rec}\n")

spark.stop()
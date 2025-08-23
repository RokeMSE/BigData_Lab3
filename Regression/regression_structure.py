# src/Regression/Structured_API/regression_structured.py
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import udf, col, hour
from pyspark.sql.types import DoubleType
from math import radians, cos, sin, asin, sqrt
import os

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

spark = SparkSession.builder.appName("Regression Structured").getOrCreate()

# Load data
df = spark.read.csv("./nyc-taxi-trip-duration/train/train.csv", header=True, inferSchema=True)

# Preprocess
haversine_udf = udf(haversine, DoubleType())
df = df.withColumn("distance", haversine_udf(col("pickup_longitude"), col("pickup_latitude"), col("dropoff_longitude"), col("dropoff_latitude")))
indexer = StringIndexer(inputCol="store_and_fwd_flag", outputCol="store_flag_index")
df = indexer.fit(df).transform(df)
df = df.withColumn("pickup_hour", hour(col("pickup_datetime")))

# Assemble features
input_cols = ["vendor_id", "passenger_count", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "store_flag_index", "pickup_hour", "distance"]
assembler = VectorAssembler(inputCols=input_cols, outputCol="features", handleInvalid="skip")
data = assembler.transform(df)
data = data.withColumnRenamed("trip_duration", "label")

# Train-test split
train, test = data.randomSplit([0.8, 0.2], seed=42)

# Train model
dt = DecisionTreeRegressor(featuresCol="features", labelCol="label")
model = dt.fit(train)

# Predictions
predictions = model.transform(test)

# Evaluate
rmse_eval = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = rmse_eval.evaluate(predictions)
r2_eval = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
r2 = r2_eval.evaluate(predictions)

print(f"RMSE: {rmse}")
print(f"R2: {r2}")

# Export predictions
preds = predictions.select("id", "label", "prediction", (col("label") - col("prediction")).alias("residual"))

os.makedirs("Results", exist_ok=True)
preds.coalesce(1).write.csv("Results/Regression_Structured.csv", header=True, mode="overwrite")

# Export metrics to a text file
with open("Results/Regression_Structured_metrics.txt", "w") as f:
    f.write(f"RMSE: {rmse}\n")
    f.write(f"R2: {r2}\n")

spark.stop()
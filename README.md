# Machine Learning for Big Data with Apache Spark
This project explores the implementation of fundamental machine learning algorithms on large-scale datasets using Apache Spark. It tackles two core ML problems—classification and regression—using three distinct Spark programming paradigms to compare their performance, complexity, and ease of use.

## Project Overview
The primary goal of this project is to build and evaluate machine learning models for two tasks:
1.  **Classification**: Detecting fraudulent credit card transactions using Logistic Regression.
2.  **Regression**: Predicting the duration of NYC taxi trips using Decision Trees.

For each task, the models are implemented using three different approaches to showcase the evolution and abstraction levels of Spark's APIs:
* **Structured API (`spark.ml`)**: A high-level, DataFrame-based API that provides a streamlined pipeline for building ML models.
* **MLlib RDD-Based API (`spark.mllib`)**: The original Spark ML library built on Resilient Distributed Datasets (RDDs).
* **Low-Level RDD Operations**: A manual implementation of the algorithms using fundamental RDD transformations (`map`, `filter`, `reduce`, etc.) to understand the underlying distributed computation.

## Datasets
* **Classification**: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle.
* **Regression**: [New York City Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration/data) from Kaggle.
Before running the scripts, please download the datasets and place them in the appropriate directories as referenced in the code (`./creditcard.csv/creditcard.csv` and `./nyc-taxi-trip-duration/train/train.csv`).

## Setup and Installation
### Prerequisites
* Java 8 or later
* Apache Spark 3.0.0 or later
* Python 3.7 or later
* `pyspark` library

### Installation
1.  Ensure you have Java and Spark installed and configured on your system.
2.  Install the required Python library: `pip install pyspark`

## How to Run the Scripts
You can execute the Python scripts using the `spark-submit` command. Make sure you are in the root directory of the project.

### 1. Classification (Logistic Regression)
**Structured API:**
`spark-submit Classification/classification_structure.py`

**MLlib RDD-Based:**
`spark-submit Classification/classification_rdd.py`

2. Regression (Decision Tree)
**Structured API:**
`spark-submit Regression/regression_structure.py`

**MLlib RDD-Based:**
`spark-submit Regression/regression_rdd.py`

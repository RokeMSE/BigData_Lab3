import pandas as pd

data_paths = ['./results/Classification_RDD_Results.csv', 
              './results/Classification_Structured_Results.csv', 
              './results/Regression_RDD_Results.csv', 
              './results/Regression_Structured_Results.csv']

for path in data_paths:
    df = pd.read_csv(path)
    print(f"Data from {path}:")
    print(df.head(5))
    print("\n")

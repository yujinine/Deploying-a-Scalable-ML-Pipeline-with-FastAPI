import pandas as pd

# Load the data
df = pd.read_csv('census.csv')

# Inspect the first few rows
print(df.head())

# Get information about the data types and missing values
print(df.info())

# Get descriptive statistics
print(df.describe())

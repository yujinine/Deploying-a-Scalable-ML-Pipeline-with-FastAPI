import pandas as pd
from ml.data import process_data

# Load the data
df = pd.read_csv('data/census.csv')

# Define the categorical features.
categorical_features = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country'
]

# Preprocess the data
X, y, encoder, lb = process_data(
    X=df,
    categorical_features=categorical_features,
    label='salary',
    training=True
)

# Print the shape of the processed data
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

from ml.data import process_data
import pandas as pd

#load the dataset
df = pd.read_csv('data/census.csv')

# call the function
X, y, encoder, lb = process_data(X = df, categorical_features=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'], label = 'salary', training=True)

#print the output shapes
print((X.shape))
print((y[:5]))

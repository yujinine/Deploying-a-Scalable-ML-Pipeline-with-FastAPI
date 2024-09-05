import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

def process_data(
    X, categorical_features=None, label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline. """

    if categorical_features is None:
        categorical_features = []

    # Debug print statements
    print("DataFrame X before any operations:")
    print(X.head())  # This prints the first few rows of the DataFrame
    print("Columns in DataFrame before any operations:", X.columns.tolist())

    # Standardize column names by replacing underscores with hyphens
    X.columns = X.columns.str.replace('_', '-')  # This is the new line added

    # Separate the feature columns from the label
    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    print("Columns in DataFrame:", X.columns.tolist())

    # Check if all categorical features exist in the DataFrame
    for feature in categorical_features:
        if feature not in X.columns:
            raise KeyError(f"{feature} not found in columns")

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(columns=categorical_features, axis=1).values

    # Encode the categorical features using the encoder or OneHotEncoder if encoder is None
    encoder = encoder if encoder is not None else OneHotEncoder(handle_unknown="ignore")
    X_categorical = encoder.fit_transform(X_categorical) if training else encoder.transform(X_categorical)

    # If training, create and fit the label binarizer
    if lb is None and training:
        lb = LabelBinarizer()
        y = lb.fit_transform(y.values).ravel()
    elif lb is not None and not training:
        y = lb.transform(y.values).ravel()

    # Convert sparse matrix to dense if needed
    if isinstance(X_categorical, np.ndarray):
        X_categorical = X_categorical
    else:
        X_categorical = X_categorical.toarray()

    X = np.concatenate([X_continuous, X_categorical], axis=1)

    print("Processed X type:", type(X))
    print("Processed X shape:", X.shape)

    return X, y, encoder, lb

def apply_label(inference):
    """ Convert binary label to string """
    return ">50K" if inference[0] == 1 else "<=50K"

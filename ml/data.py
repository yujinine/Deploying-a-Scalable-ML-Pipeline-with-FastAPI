from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import numpy as np
import pandas as pd

def process_data(
    X, categorical_features=None, label=None, training=True, encoder=None, lb=None
):
    """ Process data for ML pipeline """
    if categorical_features is None:
        categorical_features = []

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1).values

    if training:
        encoder = OneHotEncoder(handle_unknown="ignore")
        X_categorical = encoder.fit_transform(X_categorical)
        lb = LabelBinarizer()
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        except AttributeError:
            pass

    # Convert sparse matrix to dense if needed
    if isinstance(X_categorical, np.ndarray):
        X_categorical = X_categorical
    else:
        X_categorical = X_categorical.toarray()

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    
    print(f"Processed X type: {type(X)}")  # Debugging step to check type of X
    print(f"Processed X shape: {X.shape}")  # Debugging step to check shape of X

    return X, y, encoder, lb

def apply_label(inference):
    """Convert binary label to string"""
    return ">50K" if inference[0] == 1 else "<=50K"

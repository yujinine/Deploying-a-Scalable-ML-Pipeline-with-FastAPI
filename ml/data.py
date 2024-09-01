import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
    X, categorical_features=None, label=None, training=True,
    encoder=None, lb=None
):
    """
    Process data for ML pipeline

    Encodes categorical features and labels for training or inference.

    Inputs:
    - X : pd.DataFrame, features and label
    - categorical_features: list[str], names of categorical features
    - label : str, name of the label column
    - training : bool, True for training mode
    - encoder : OneHotEncoder, used in inference mode
    - lb : LabelBinarizer, used in inference mode

    Returns:
    - X : np.array, processed data
    - y : np.array, processed labels (empty if no label)
    - encoder : OneHotEncoder, fitted encoder
    - lb : LabelBinarizer, fitted label binarizer
    """

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
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb


def apply_label(inference):
    """Convert binary label to string"""
    return ">50K" if inference[0] == 1 else "<=50K"

import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data
# TODO: add necessary import
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

        # Scale the training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    # Create a logistic regression model
    model = LogisticRegression(max_iter=1000)

    # Train the model using the training data
    model.fit(X_train, y_train)

    # Return the trained model
    return model



def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """
    Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.linear_model.LogisticRegression
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # Use the model to make predictions
    preds = model.predict(X)

    # Return the predictions
    return preds


def save_model(model, path):
    """
    Serializes model to a file.

    Inputs
    ------
    model : sklearn.linear_model.LogisticRegression or OneHotEncoder
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save the pickle file.
    """
    # Open the specified file in write-binary mode and serialize the model
    with open(path, 'wb') as file:
        pickle.dump(model, file)


def load_model(path):
    """
    Loads pickle file from `path` and returns it.

    Inputs
    ------
    path : str
        Path to load the pickle file from.

    Returns
    -------
    model : sklearn.linear_model.LogisticRegression or OneHotEncoder
        The deserialized model or OneHotEncoder.
    """
    # Open the specified file in read-binary mode and deserialize the model
    with open(path, 'rb') as file:
        model = pickle.load(file)
    
    return model



def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """
    Computes the model metrics on a slice of the data specified by a column name and slice value.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or inference/validation.

    Inputs
    ------
    data : pd.DataFrame
        DataFrame containing the features and label. Columns in `categorical_features` should be
        the categorical features.
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature.
    categorical_features : list
        List containing the names of the categorical features (default=[]).
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned for y (default=None)
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    model : sklearn model
        Model used for the task.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """

    # Filter the data for the specific slice
    data_slice = data[data[column_name] == slice_value]
    # The line above selects only the rows from `data` where the value in `column_name`
    # matches `slice_value`. This isolates the specific "slice" of data we want to analyze.

    # Process the data slice
    X_slice, y_slice, _, _ = process_data(
        data_slice, categorical_features=categorical_features, label=label,
        training=False, encoder=encoder, lb=lb
    )
    # This processes the filtered `data_slice` using the `process_data` function. 
    # Since `training=False`, it uses the provided `encoder` and `lb` to one-hot encode 
    # categorical features and binarize the labels without fitting them again.

    # Get predictions on the data slice
    preds = inference(model, X_slice)
    # This line uses the `inference` function to predict the labels for `X_slice` using the `model`.

    # Compute the metrics on the predictions
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    # Finally, this computes the precision, recall, and fbeta score using the true labels (`y_slice`) 
    # and the predicted labels (`preds`).

    return precision, recall, fbeta
    # The function returns the computed metrics: precision, recall, and fbeta.


import os
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from ml.data import apply_label, process_data
from ml.model import inference, load_model
from sklearn.preprocessing import OneHotEncoder

# Initialize encoder globally
encoder = None

# DO NOT MODIFY!
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=179398)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., example="Married-civ-spouse", alias="marital_status")
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native_country")

# Path for the saved encoder
path_encoder = "model/encoder.pkl"
try:
    encoder = load_model(path_encoder)
except FileNotFoundError:
    raise ValueError("Failed to load the encoder. Please check the encoder file path and content.")

# Path for the saved model
path_model = "model/model.pkl"
try:
    model = load_model(path_model)
except FileNotFoundError:
    raise ValueError("Failed to load the model. Please check the model file path and content.")

# Create a RESTful API using FastAPI
app = FastAPI()

@app.get("/")
async def get_root():
    return {"message": "Hello, welcome to the model inference API!"}

@app.post("/infer/")
async def post_inference(data: Data):
    global encoder
    # Convert the Pydantic model into a dict
    data_dict = data.dict()
    data_df = pd.DataFrame([data_dict])

    # Define the categorical features
    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country"
    ]

    # Process the input data
    data_processed, _, encoder, lb = process_data(
        data_df, categorical_features=cat_features, training=False, encoder=encoder, lb=None
    )

    # Make a prediction
    prediction = model.predict(data_processed)

    # Convert the prediction to a human-readable label
    result = apply_label(prediction)

    return {"result": result}

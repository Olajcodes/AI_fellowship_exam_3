# Importing of necessary libraries
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from fastapi import FastAPI
import joblib

# Initialize FastAPI app
app = FastAPI()

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Define the Pydantic input data model
class WineChemicalProperties(BaseModel):
    fixed_acidity: float = Field(..., examples=["7.0"])
    volatile_acidity: float = Field(..., examples=["0.27"])
    citric_acid: float = Field(..., examples=["0.36"])
    residual_sugar: float = Field(..., examples=["20.7"])
    chlorides: float = Field(..., examples=["0.045"])
    free_sulfur_dioxide: float = Field(..., examples=["45.0"])
    total_sulfur_dioxide: float = Field(..., examples=["170.0"])
    density: float = Field(..., examples=["1.0"])
    pH: float = Field(..., examples=["3.0"])
    sulphates: float = Field(..., examples=["0.45"])
    alcohol: float = Field(..., examples=["8.8"])


# let's define a home endpoint
@app.get("/")
def home_page():
    return {"message": "Welcome to the White Wine Quality Prediction API"}

# Define a prediction endpoint
@app.post("/predict")
def predict_wine_quality(features: WineChemicalProperties):
    # # Convert input data to 2d Array
    features = np.array([[features.fixed_acidity,
                features.volatile_acidity,
                features.citric_acid,
                features.residual_sugar,
                features.chlorides,
                features.free_sulfur_dioxide,
                features.total_sulfur_dioxide,
                features.density,
                features.pH,
                features.sulphates,
                features.alcohol]])
    
    # Scale the features
     #  lets scale our input features using the loaded scaler (to normalize the input features)
    scaled_features = scaler.transform(features)

    # lets make prediction with the loaded model
    prediction = model.predict(scaled_features)

    # Return the prediction and the prediction converted to string for serialization
    return {"predicted_quality": str(prediction[0])}

# To run the app with Uvicorn, I will use the command: uvicorn main:app --reload
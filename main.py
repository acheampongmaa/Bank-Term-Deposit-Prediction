import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union

# Load the saved model and pipelines
with open("term_deposit.pkl", "rb") as f:
    model_dict = pickle.load(f)

model = model_dict["model"]
pipeline = model_dict["pipeline"]

# Initialize FastAPI
app = FastAPI()

# Define the input data model
class ClientData(BaseModel):
    age: int
    balance: float
    day: int
    duration: float
    campaign: int
    pdays: float
    previous: int
    recent_contact: Union[int, None]
    financial_stability: Union[int, None]
    duration_per_contact: Union[float, None]
    contact_known: Union[int, None]
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    poutcome: str
    age_group: str

# Prediction endpoint
@app.post("/predict")
def predict(client_data: ClientData):
    # Convert input data to dictionary and prepare it for prediction
    input_data = client_data.dict()
    
    # Separate numerical and categorical data
    num_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 
                    'previous', 'recent_contact', 'financial_stability', 
                    'duration_per_contact', 'contact_known']
    cat_features = ['job', 'marital', 'education', 'default', 'housing', 
                    'loan', 'contact', 'month', 'poutcome', 'age_group']
    
    # Extract numerical and categorical values
    num_data = [[input_data[feature] for feature in num_features]]
    cat_data = [[input_data[feature] for feature in cat_features]]
    
    # Preprocess data using the full pipeline
    processed_data = pipeline.transform(np.hstack((num_data, cat_data)))
    
    # Get predictions and probabilities
    prediction = model.predict(processed_data)
    probability = model.predict_proba(processed_data)[:, 1]  # Probability of "yes"
    
    # Return prediction and probability
    return {
        "prediction": "yes" if prediction[0] == 1 else "no",
        "probability": float(probability[0])
    }

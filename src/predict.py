import joblib
import numpy as np
model=joblib.load("../model/depression_model.pkl")

def predict_depression(data):
    prediction=model.predict([data])
    if prediction[0]==1:
        return "High risk of depression"
    else:
        return "Low Risk"
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load model and scaler
with open("pcos_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Streamlit UI
st.title("PCOS/PCOD Prediction App")

# Create input fields
features = []
for i in range(10):  # Adjust for dataset
    features.append(st.number_input(f"Feature {i+1}", value=0.0))

# Prediction function
def predict_pcos(features):
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)
    prob = model.predict_proba(features)[0][1]  # Get probability

    if prob < 0.3:
        return "No PCOS detected - Just maintain a good lifestyle."
    elif prob < 0.7:
        return "May have PCOS - Can visit a doctor."
    else:
        return "High possibility of PCOS - Must visit a doctor."

# Button for prediction
if st.button("Predict"):
    result = predict_pcos(features)
    st.write(f"### Result: {result}")


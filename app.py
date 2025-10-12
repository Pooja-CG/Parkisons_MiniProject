import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open('parkinsons_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🧠 Parkinson’s Disease Prediction System")
st.write("Enter the patient’s medical parameters below:")

# Only 5 inputs
fo = st.number_input("MDVP:Fo(Hz)", value=120.0)
jitter = st.number_input("MDVP:Jitter(%)", value=0.005)
shimmer = st.number_input("MDVP:Shimmer", value=0.02)
nhr = st.number_input("NHR", value=0.03)
hnr = st.number_input("HNR", value=20.0)

if st.button("Predict"):
    try:
        # Create feature array
        features = np.array([fo, jitter, shimmer, nhr, hnr]).reshape(1, -1)
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)
        
        if prediction[0] == 1:
            st.error("Patient may have Parkinson’s disease")
        else:
            st.success("Patient is healthy")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
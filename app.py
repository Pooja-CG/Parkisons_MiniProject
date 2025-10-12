import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open('parkinsons_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🧠 Parkinson’s Disease Prediction System")
st.markdown("### Enter the Patient’s Medical Parameters Below:")

# Input fields
MDVP_Fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0)
MDVP_Fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0)
MDVP_Flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0)
Jitter_percent = st.number_input("Jitter(%)", min_value=0.0)
Shimmer = st.number_input("Shimmer", min_value=0.0)

# Prediction button
if st.button("🔍 Predict Parkinson’s"):
    features = np.array([[MDVP_Fo, MDVP_Fhi, MDVP_Flo, Jitter_percent, Shimmer]])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)

    if prediction[0] == 1:
        st.error("⚠️ Parkinson’s Disease Detected")
    else:
        st.success("✅ No Parkinson’s Detected")

import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open('parkinsons_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🧠 Parkinson’s Disease Prediction System")
st.write("Enter the patient’s medical parameters below:")

# User Inputs (5 selected features)
fo = st.number_input("MDVP:Fo(Hz)", value=120.000, format="%.3f")
jitter = st.number_input("MDVP:Jitter(%)", value=0.005, format="%.3f")
shimmer = st.number_input("MDVP:Shimmer", value=0.020, format="%.3f")
nhr = st.number_input("NHR", value=0.030, format="%.3f")
hnr = st.number_input("HNR", value=20.000, format="%.3f")

# Confidence threshold
threshold = st.slider("Select confidence threshold", 0.0, 1.0, 0.60)

# Default values for the remaining 17 features
# ⚠ EXACT 17 FEATURES → Total 22 = OK for model
default_values = {
    'MDVP:Fhi(Hz)': 197.084,
    'MDVP:Flo(Hz)': 104.315,
    'MDVP:Jitter(Abs)': 0.00004,
    'MDVP:RAP': 0.003,
    'MDVP:PPQ': 0.0035,
    'Jitter:DDP': 0.009,
    'MDVP:Shimmer(dB)': 0.300,
    'Shimmer:APQ3': 0.015,
    'Shimmer:APQ5': 0.020,
    'MDVP:APQ': 0.025,
    'Shimmer:DDA': 0.045,
    'RPDE': 0.45,
    'DFA': 0.72,
    'spread1': -5.33,
    'spread2': 0.25,
    'D2': 2.30,
    'PPE': 0.21
}

if st.button("Predict"):
    try:
        # User inputs (5 features)
        user_features = [fo, jitter, shimmer, nhr, hnr]

        # Default mean values (17 features)
        remaining_features = list(default_values.values())

        # All 22 features for prediction
        final_input = np.array(user_features + remaining_features).reshape(1, -1)

        # Scale the data
        scaled_input = scaler.transform(final_input)

        # Predict probability
        probability = model.predict_proba(scaled_input)[0][1]

        # Output
        st.subheader("Model Confidence & Prediction")
        st.write(f"🎯 Model confidence: **{probability * 100:.2f}%**")
        st.write(f"⚙ Threshold applied: **{threshold:.2f}**")

        if probability > threshold:
            st.error("🚨 Patient **may have Parkinson’s disease.**")
        else:
            st.success("✅ Patient is **likely healthy.**")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

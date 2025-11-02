import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open('parkinsons_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🧠 Parkinson’s Disease Prediction System")
st.write("Enter the patient’s medical parameters below:")

# Input fields
fo = st.number_input("MDVP:Fo(Hz)", value=120.0, format="%.3f")
jitter = st.number_input("MDVP:Jitter(%)", value=0.005, format="%.3f")
shimmer = st.number_input("MDVP:Shimmer", value=0.020, format="%.3f")
nhr = st.number_input("NHR", value=0.030, format="%.3f")
hnr = st.number_input("HNR", value=20.000, format="%.3f")

# 🔹 Add model confidence + threshold control
threshold = st.slider("Select confidence threshold", 0.0, 1.0, 0.9)

if st.button("Predict"):
    try:
        # Prepare input data
        input_data = np.array([[fo, jitter, shimmer, nhr, hnr]])
        scaled_data = scaler.transform(input_data)

        # Get probability of Parkinson’s
        probability = model.predict_proba(scaled_data)[0][1]

        # Show confidence
        st.subheader("Model Confidence & Prediction")
        st.write(f"🎯 Model confidence: **{probability * 100:.2f}%**")
        st.write(f"⚙️ Applied threshold: **{threshold:.2f}**")

        # Apply threshold dynamically
        if probability > threshold:
            st.error("🚨 Patient **may have Parkinson’s disease.**")
        else:
            st.success("✅ Patient is **likely healthy.**")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

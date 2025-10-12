#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('writefile', 'app.py', 'import streamlit as st\nimport pickle\nimport numpy as np\n')


# In[6]:


import os
print(os.getcwd())


# In[7]:


os.listdir()


# In[8]:


app_path = os.path.abspath("app.py")
print(app_path)


# In[1]:


import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open('parkinsons_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Title and subtitle
st.title("🧠 Parkinson’s Disease Prediction System")
st.markdown("### Enter the Patient’s Medical Parameters Below:")

# Input fields for user data
MDVP_Fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0)
MDVP_Fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0)
MDVP_Flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0)
Jitter_percent = st.number_input("Jitter(%)", min_value=0.0)
Shimmer = st.number_input("Shimmer", min_value=0.0)

# Predict button
if st.button("🔍 Predict Parkinson’s"):
    # Combine all inputs into a NumPy array (extend with all features if needed)
    features = np.array([[MDVP_Fo, MDVP_Fhi, MDVP_Flo, Jitter_percent, Shimmer]])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)

    if prediction[0] == 1:
        st.error("⚠️ Parkinson’s Disease Detected")
    else:
        st.success("✅ No Parkinson’s Detected")


# In[ ]:





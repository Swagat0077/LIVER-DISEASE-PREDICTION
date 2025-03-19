import streamlit as st
import pickle
import numpy as np

# ✅ Load the trained model and scaler
with open("liver_disease_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# ✅ Streamlit Web App UI
st.title("Liver Disease Prediction App")
st.write("Enter the details below to check if a patient is at risk of liver disease.")

# ✅ Input fields (Only 10 required features)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.radio("Gender", ["Male", "Female"])
bilirubin = st.number_input("Total Bilirubin", value=1.2)
direct_bilirubin = st.number_input("Direct Bilirubin", value=0.5)
alkaline_phosphotase = st.number_input("Alkaline Phosphotase", value=200)
alt = st.number_input("Alanine Aminotransferase", value=30)
ast = st.number_input("Aspartate Aminotransferase", value=40)
proteins = st.number_input("Total Proteins", value=6.5)
albumin = st.number_input("Albumin", value=3.0)
ag_ratio = st.number_input("Albumin and Globulin Ratio", value=1.2)

# Convert Gender to numeric
gender = 1 if gender == "Male" else 0

# ✅ Prepare Input Data (Ensure it matches training feature count)
input_data = np.array([[age, gender, bilirubin, direct_bilirubin, 
                         alkaline_phosphotase, alt, ast, 
                         proteins, albumin, ag_ratio]])

# ✅ Standardize input data
input_data_scaled = scaler.transform(input_data)

# ✅ Predict on User Input
if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    st.subheader("Liver Disease Risk: " + ("Yes" if prediction[0] == 1 else "No"))

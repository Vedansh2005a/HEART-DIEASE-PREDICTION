import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load your tools
model = joblib.load('KNN_MODEL.pkl')
scaler = joblib.load('Standadar_sclaer.pkl')
expected_columns = joblib.load('columns.pkl')

st.title("❤️ Heart Disease Prediction App")

# 2. Input Collection
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 25)
    sex = st.selectbox("Sex", ['M', 'F'])
    cp = st.selectbox("Chest Pain Type", ['ATA', 'NAP', 'TA', 'ASY'])
    rp = st.number_input("Resting Blood Pressure", 80, 200, 110)
    chol = st.number_input("Cholesterol", 100, 600, 180)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
    maxhr = st.number_input("Max Heart Rate", 60, 220, 185)
    exang = st.selectbox("Exercise Angina", ['N', 'Y'])
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 0.0)
    stslope = st.selectbox("ST Slope", ['Up', 'Flat', 'Down'])

if st.button("Predict"):
    # 3. Data Preparation
    raw_input = {
        'Age': age, 'Sex': sex, 'ChestPainType': cp, 'RestingBP': rp,
        'Cholesterol': chol, 'FastingBS': fbs, 'RestingECG': restecg,
        'MaxHR': maxhr, 'ExerciseAngina': exang, 'Oldpeak': oldpeak,
        'ST_Slope': stslope
    }
    input_df = pd.DataFrame([raw_input])

    # 4. Encoding
    # Match Training: F=0, M=1 | N=0, Y=1
    input_df['Sex'] = 1 if sex == 'M' else 0
    input_df['ExerciseAngina'] = 1 if exang == 'Y' else 0
    
    # One-Hot Encoding
    input_df = pd.get_dummies(input_df, columns=['ChestPainType', 'RestingECG', 'ST_Slope'])

    # 5. Alignment
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder to match the model's expectation
    input_df = input_df[expected_columns]
    
    # Convert dummy columns from True/False to 0/1
    input_df = input_df.astype(float)

    # 6. Scaling (THE FIX: All 5 columns)
    num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    input_df[num_cols] = scaler.transform(input_df[num_cols].values)
    
    # 7. Prediction
    prediction = model.predict(input_df)
    
    st.divider()
    if prediction[0] == 1:
        st.error("Prediction: **High Risk** of Heart Disease")
    else:
        st.success("Prediction: **Low Risk** of Heart Disease")
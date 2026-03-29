import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np

# =========================
# LOAD FILES
# =========================
model = tf.keras.models.load_model('model.h5', compile=False)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

# =========================
# STREAMLIT UI
# =========================
st.title("Customer Churn Prediction")

geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# =========================
# PREDICTION BUTTON
# =========================
if st.button("Predict"):

    # Encode gender
    gender_encoded = label_encoder_gender.transform([gender])[0]

    # Create dataframe
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [gender_encoded],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary],
        'Geography_France': [1 if geography == 'France' else 0],
        'Geography_Germany': [1 if geography == 'Germany' else 0],
        'Geography_Spain': [1 if geography == 'Spain' else 0],
    })

    # Ensure column order matches training
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)
    prediction_proba = float(prediction[0][0])

    # Output
    st.write(f"Churn Probability: {prediction_proba:.2f}")

    if prediction_proba > 0.5:
        st.error("The customer is likely to churn")
    else:
        st.success("The customer is not likely to churn")
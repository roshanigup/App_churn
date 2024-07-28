
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
model = joblib.load('trained_model.joblib')
scaler = joblib.load('scaler.joblib')
# Function to predict churn status
def predict_churn(average_screen_time, average_spent_on_app, left_review, ratings, new_password_request, last_visited_minutes,Installation_Status ):
    # Scale the input values
    
    input_values = scaler.transform([[average_screen_time, average_spent_on_app, left_review, ratings, new_password_request, last_visited_minutes,Installation_Status]])
    # Predict churn status
    prediction = model.predict(input_values)
    return prediction[0]

# Streamlit app
st.title('Customer Churn Prediction')

# Input values for independent variables
average_screen_time = st.number_input('Average Screen Time', value=0.0)
average_spent_on_app = st.number_input('Average Spent on App (INR)', value=0.0)
left_review = st.selectbox('Left Review', [0, 1])
ratings = st.slider('Ratings', min_value=0, max_value=5, value=0)
new_password_request = st.selectbox('New Password Request', [0, 1])
last_visited_minutes = st.number_input('Last Visited Minutes', value=0.0)
Installation_Status = st.selectbox('Installation Status', [0, 1])

# Predict churn status
if st.button('Predict'):
    prediction = predict_churn(average_screen_time, average_spent_on_app, left_review, ratings, new_password_request, last_visited_minutes,Installation_Status)
    st.write('Churn Status:', prediction)

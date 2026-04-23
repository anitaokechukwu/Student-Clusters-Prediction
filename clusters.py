import streamlit as st
import numpy as np
import joblib as jb

st. title ("Student Clusters Prediction")

hours = st.number_input('Enter Hours')
sleep_hours = st.number_input('Enter Sleep Hours')
attendance = st. number_input('Enter Attendance')
previous_score = st.number_input('Enter Previous Score')
exam_score = st.number_input('Enter Exam Score')

input = np.array([hours, sleep_hours, attendance, previous_score, exam_score]).reshape(1,-1)

scaler = jb.load('my_scaler.pkl')
model = jb.load('k_means.pkl')

if st.button('predict'):
    scaled = scaler.transform(input)
    result = model.predict(scaled)
    st.write(f'The predicted Cluster is {result}')


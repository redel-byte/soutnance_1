import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Load the trained pipeline
@st.cache_resource
def load_model(path, modified_at):
    return joblib.load(path)


st.title("Student Mental Health Risk Prediction")

model_path = Path(os.environ.get("STUDENT_MODEL_PATH", "models/best_pipeline.joblib"))
if not model_path.is_file():
    st.info("Run the student_risk_pipeline DAG to train the model, then refresh this page.")
    st.stop()
model = load_model(model_path, model_path.stat().st_mtime_ns)

with st.form("prediction_form"):
    age = st.number_input("Age", 17, 29, 21)
    year = st.selectbox("Year of Study", [1, 2, 3, 4])
    gender = st.selectbox("Gender", ["Male", "Female", "Non-binary"])
    study = st.number_input("Study Hours per Week", value=20.0)
    gpa = st.number_input("GPA", 0.0, 4.0, 2.8)
    attendance = st.number_input("Attendance Rate (%)", 0.0, 100.0, 85.0)
    job = st.selectbox("Part-Time Job", ["Yes", "No"])
    sleep = st.number_input("Sleep Hours", value=7.0)
    screen = st.number_input("Screen Time Hours", value=6.0)
    social = st.number_input("Social Media Hours", value=3.0)
    physical = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
    diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
    stress = st.slider("Financial Stress", 0, 10, 5)
    support = st.slider("Family Support", 0, 10, 6)
    history = st.selectbox("Previous Mental Health History", ["Yes", "No"])

    submitted = st.form_submit_button("Predict")

if submitted:
    data = pd.DataFrame([{
        "Age": age,
        "Year_of_Study": year,
        "Gender": gender,
        "Study_Hours_Per_Week": study,
        "GPA": gpa,
        "Attendance_Rate": attendance,
        "Part_Time_Job": job,
        "Sleep_Hours": sleep,
        "Screen_Time_Hours": screen,
        "Social_Media_Hours": social,
        "Physical_Activity": physical,
        "Diet_Quality": diet,
        "Financial_Stress": stress,
        "Family_Support": support,
        "Previous_Mental_Health_History": history,
    }])

    prediction = model.predict(data)[0]

    st.success(f"Predicted Risk: {prediction}")

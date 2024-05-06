import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
from pipeline import preprocess_user_input

DATASET_PATH = "data/heart_2022_no_nans.parquet"
LOG_MODEL_PATH = "logistic_model.pkl"

heart = pd.read_parquet(DATASET_PATH)

def main():
    
    st.set_page_config(
        page_title="Heart Disease Prediction App",
    )
    
    st.title("Heart Disease Prediction")
    st.subheader("Are you wondering about the condition of your heart? "
                 "This app will help you to diagnose it!")

    st.markdown("""
    Discover how machine learning models can aid in predicting heart disease with impressive accuracy! 
    With this application, estimating your likelihood of heart disease (yes/no) takes mere seconds!

    Our model leverages a Logistic Regression algorithm constructed using undersammpling techniques, 
    trained on survey data from over 200,000 US residents collected in 2022.

    To ascertain your heart disease status, simply follow these steps:

    1. Fill out the sidebar with parameters that best describe you.
    2. Click the "Predict" button and await the result.

    However, it's essential to note that this result does not equate to a medical diagnosis! Health care facilities would not adopt this model due to its less-than-perfect accuracy. If you have any concerns, it's crucial to consult a qualified medical professional.

    **Author: Vinicius Victorelli ([GitHub](https://github.com/vinisvictorelli))**

    For a detailed overview of how the model was developed, evaluated, and the data was cleaned, visit my GitHub repository [here](https://github.com/kamilpytlak/data-analyses/tree/main/heart-disease-prediction). 
    """)

    st.sidebar.title("Feature Selection")
    sex = st.sidebar.selectbox("What is you sex?", options=(sex for sex in heart['Sex'].unique()))
    age_cat = st.sidebar.selectbox("Select you age category:",
                                    options=(age_cat for age_cat in heart['AgeCategory'].unique()))
    height = st.sidebar.text_input("What is your height in meters?")
    weight = st.sidebar.text_input('What is your weight in kg?')
    sleep_time = st.sidebar.number_input("How many hours on average do you sleep?", 0, 24, 7)

    phys_health = st.sidebar.number_input("For how many days during the past 30 days was"
                                            " your physical health not good?", 0, 30, 0)
    phys_act = st.sidebar.selectbox("Have you played any sports (running, biking, etc.)"
                                    " in the past month?", options=("No", "Yes"))
    smoking = st.sidebar.selectbox("What is your level of smoking?",
                                    options=(smoke_stat for smoke_stat in heart['SmokerStatus'].unique()))
    alcohol_drink = st.sidebar.selectbox("Do you have more than 14 drinks of alcohol (men)"
                                            " or more than 7 (women) in a week?", options=("No", "Yes"))
    stroke = st.sidebar.selectbox("Did you have a stroke?", options=("No", "Yes"))
    diabetic = st.sidebar.selectbox("Have you ever had diabetes?",
                                    options=(diabetic for diabetic in heart['HadDiabetes'].unique()))
    asthma = st.sidebar.selectbox("Do you have asthma?", options=("No", "Yes"))
    kid_dis = st.sidebar.selectbox("Do you have kidney disease?", options=("No", "Yes"))
    skin_canc = st.sidebar.selectbox("Do you have skin cancer?", options=("No", "Yes"))
    copd = st.sidebar.selectbox("Do you have or had C.O.P.D. (chronic obstructive pulmonary disease), emphysema or chronic bronchitis?", options=("No", "Yes"))
    depressive_disorder = st.sidebar.selectbox("Do you have or had depressive disorder?", options=("No", "Yes"))
    arthritis = st.sidebar.selectbox('Do you have or had some form of arthritis, rheumatoid arthritis, gout, lupus, or fibromyalgia?', options=("No", "Yes"))
    e_cigarette = st.sidebar.selectbox('Do you have or had used any kind of eletronic cigarettes?', 
                                        options=(eciga for eciga in heart['ECigaretteUsage'].unique()))
    hiv_test = st.sidebar.selectbox("Have you ever been tested positive for HIV?", options=("No", "Yes"))
    flu_vaccine = st.sidebar.selectbox("During the past 12 months, have you had flu vaccine?", options=("No", "Yes"))
    pneumo = st.sidebar.selectbox("Have you ever had a pneumonia shot also known as a pneumococcal vaccine?", options=("No", "Yes"))
    covid = st.sidebar.selectbox('Has a doctor, nurse, or other health professional ever told you that you tested positive for COVID 19?', 
                                        options=(cov for cov in heart['CovidPos'].unique()))
    submit = st.sidebar.button("Predict")
        
    if submit:
        features = pd.DataFrame({
            'Sex':[sex],
            'AgeCategory':[age_cat],
            'PhysicalHealthDays': [phys_health], 
            'PhysicalActivities': [phys_act], 
            'SleepHours': [sleep_time], 
            'HadStroke': [stroke], 
            'HadAsthma': [asthma], 
            'HadSkinCancer': [skin_canc], 
            'HadCOPD': [copd], 
            'HadDepressiveDisorder': [depressive_disorder], 
            'HadKidneyDisease': [kid_dis], 
            'HadArthritis': [arthritis], 
            'HadDiabetes': [diabetic], 
            'SmokerStatus': [smoking], 
            'ECigaretteUsage': [e_cigarette], 
            'HeightInMeters': [float(height)],
            'WeightInKilograms' : [float(weight)],
            'AlcoholDrinkers': [alcohol_drink], 
            'HIVTesting': [hiv_test], 
            'FluVaxLast12': [flu_vaccine], 
            'PneumoVaxEver': [pneumo],
            'CovidPos' : [covid]
        })
        print(features)
        log_model = pickle.load(open(LOG_MODEL_PATH, "rb"))
        df = preprocess_user_input(features)
        print(df)
        prediction = log_model.predict(df)
        prediction_prob = log_model.predict_proba(df)
        st.markdown(f"**The probability that you will have"
                    f" heart disease is {round(prediction_prob[0][1] * 100, 2)}%.")

if __name__ == "__main__":
    main()
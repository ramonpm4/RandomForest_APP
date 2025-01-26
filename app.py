import numpy as np
import streamlit as st
import joblib
import pandas as pd

model = joblib.load("random_forest_model.pkl")

# Title of the application:
st.title("Cardiovascular Deseases Prediction App")
st.markdown("""
 This is an app where you can find the **Propability of having Cardiovascular Disease** based on a series of features. The predictions are powered by a **Supervised Machine Learning - Random Forest Classification Model**, which was trained on a dataset of 918 examples. 
            
### **Context** :

Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs and this dataset contains 11 features that can be used to predict a possible heart disease.

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management.

**Attribute Information**

- Age: age of the patient [years]
- Sex: sex of the patient [M: Male, F: Female]
- ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
- RestingBP: resting blood pressure [mm Hg]
- Cholesterol: serum cholesterol [mm/dl]
- FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
- RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
- MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
- ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
- Oldpeak: oldpeak = ST [Numeric value measured in depression]
- ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
- HeartDisease: output class [1: heart disease, 0: Normal]
            
### **Model Prediction** :""")

# Create entries for patient characteristics:


feature_names = [
"Age: age of the patient [years]",
"Sex: sex of the patient [M: Male, F: Female]",
"ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]",
"RestingBP: resting blood pressure [mm Hg]",
"Cholesterol: serum cholesterol [mm/dl]",
"FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]",
"RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]",
"MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]",
"ExerciseAngina: exercise-induced angina [Y: Yes, N: No]",
"Oldpeak: oldpeak = ST [Numeric value measured in depression]",
"ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]",
]

categorical_features = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

# The challenge is that some of the variables must be one-hot encoded, so first we will make the user to enter the numeric features:

### NUMERIC VALUES: ###
age = st.number_input("Age: age of the patient [years]", value=30)
restingbp = st.number_input("RestingBP: resting blood pressure [mm Hg]", value=140)
cholesterol = st.number_input("Cholesterol: serum cholesterol [mm/dl]", value = 289)
fastingbs = st.number_input("FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]", value = 0)
maxHR = st.number_input("MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]", value = 172)
oldpeak = st.number_input("Oldpeak: oldpeak = ST [Numeric value measured in depression]", value = 0)


### STRING VALUES ###
sex = st.selectbox("Choose Sex (M for masculine, F for femenine):", options=["M", "F"], index = 0) # Index sets an initial value for options
chestPainType = st.selectbox("chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]:", options=["TA", "ATA", "NAP", "ASY"], index = 0)
restingECG = st.selectbox("RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]", options=["Normal", "ST", "LVH"], index = 0)
exerciseAngina = st.selectbox("ExerciseAngina: exercise-induced angina [Y: Yes, N: No]", options=["Y", "N"], index = 0)
st_Slope = st.selectbox("ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]", options=["Up", "Flat", "Down"], index = 0)

# Now we will create a library to asign the data:
#(I put it in a library because I want to have control by name on each feature)

user_data = {
    "Age": age,
    "RestingBP": restingbp, 
    "Cholesterol": cholesterol,
    "FastingBS": fastingbs,
    "MaxHR": maxHR,
    "Oldpeak": oldpeak,
    "Sex": sex,
    "ChestPainType": chestPainType,
    "RestingECG": restingECG,
    "ExerciseAngina": exerciseAngina,
    "ST_Slope": st_Slope
}
df = pd.DataFrame([user_data])


df_encoded = pd.get_dummies(df, columns = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"])


expected_columns = ['Age',
 'RestingBP',
 'Cholesterol',
 'FastingBS',
 'MaxHR',
 'Oldpeak',
 'Sex_F',
 'Sex_M',
 'ChestPainType_ASY',
 'ChestPainType_ATA',
 'ChestPainType_NAP',
 'ChestPainType_TA',
 'RestingECG_LVH',
 'RestingECG_Normal',
 'RestingECG_ST',
 'ExerciseAngina_N',
 'ExerciseAngina_Y',
 'ST_Slope_Down',
 'ST_Slope_Flat',
 'ST_Slope_Up']

# To avoid errors if user misses to enter some feature:
for col in expected_columns:
    if col not in df_encoded.columns:
        df_encoded[col] = 0

# Reorder the columns of the data so that is equal to the order of the train dataset:
df_encoded = df_encoded[expected_columns]



# Now, we'll create a Preedict Button:
if st.button("Predict"):
    # First convert the inputs into a Numpy Array:
    data = df_encoded.values

    # Do the prediction:
    prediction = model.predict(data)[0]
    probabilities = model.predict_proba(data)[0]


    st.subheader("Results of the prediction")
    st.write(f"**Prediction:** {'High probability of Cardiovascular Disease' if prediction == 1 else 'Low probability of CVD'}")
    st.write("**Probabilities:**")
    st.write(f"Probability of not having Cardiovascular Disease: {probabilities[0]:.4f}")
    st.write(f"Probability of having Cardiovascular Disease: {probabilities[1]:.4f}")




st.markdown("""This model was developed by *Ramón Peralta Martínez*. 
            Deployed with Streamlit - January 2025""")



    # Remember: Streamlit must be run with streamlit run app.py into the terminal, as it deploys an independent webb appp.






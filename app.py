import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import tensorflow
from tensorflow.keras.models import load_model
import pickle
import streamlit as st

model = load_model("model.h5")

with open('artifacts\label_encoder_gender.pkl',"rb") as file:
    label_encoder_gender = pickle.load(file)

with open('artifacts\one_hot_encoder_geo.pkl',"rb") as file:
    one_hot_encoder_geo = pickle.load(file)

with open("artifacts\standard_scaler.pkl","rb") as file:
    scaler = pickle.load(file)

# Setting up the Streamlit

st.title("Customer Churn Prediction")

CreditScore = st.number_input("Enter the Credit Score",min_value=0,max_value=1000)
Geography= st.selectbox("Coumtry",options=['France', 'Spain', 'Germany'])
Gender = st.selectbox("Gender",options=['Male','Female'])
Age = st.slider("Enter Age",0,100)
Tenure = st.slider("Enter tenure",0,10)
Balance = st.number_input("Enter the Balance")
NumOfProducts =  st.slider("Number of Product",0,4)
HasCrCard = st.selectbox("Credit Card",options=[0,1])
IsActiveMember = st.selectbox("Active Member",options=[0,1])
EstimatedSalary = st.number_input("Enter Salary")

input_data = {
    'CreditScore' : [CreditScore],
    'Geography': [Geography],
    'Gender' : [Gender],
    'Age' : [Age],
    'Tenure' : [Tenure],
    'Balance' : [Balance],
    'NumOfProducts' :  [NumOfProducts],
    'HasCrCard' : [HasCrCard],
    'IsActiveMember': [IsActiveMember],
    "EstimatedSalary": [EstimatedSalary]
}

df = pd.DataFrame(input_data)


df['Gender'] = label_encoder_gender.transform(df['Gender'])


transform_geo = one_hot_encoder_geo.transform(df[['Geography']])
geo_df = pd.DataFrame(transform_geo,columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

final_df = pd.concat([df.drop('Geography',axis=1),geo_df],axis=1)
print(final_df)
scaled_df = scaler.transform(final_df)
print(scaled_df)

prediction = model.predict(scaled_df)

st.write("Churn Probalility: ",prediction[0][0])

if prediction[0][0] > 0.5:
    st.write("Customer will leave")
    print("Customer will leave")
else:
    st.write("Customer will not leave")
    print("Customer will not leave")
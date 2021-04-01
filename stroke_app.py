import pandas as pd
import streamlit as st
import pickle
import numpy as np
import py7zr
### Layout do app###

st.write('''
    # Stroke Prediction
According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths.
This web app is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status.

''')

st.image('brain_1.jpg')

st.sidebar.header('Patient data')

age = st.sidebar.selectbox('Age', list((range(0,100))))

gender = st.sidebar.radio('Gender', ['Male', 'Female'])

bmi = st.sidebar.slider('Basic mass Index (bmi)', 17.0, 100.0)

ever_married = st.sidebar.selectbox('Ever got married?', ['Yes', 'No'])

smoking_status = st.sidebar.radio('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

residence_type = st.sidebar.selectbox('Residence Type', ['Rural', 'Urban'])

avg_glucose_level = st.sidebar.slider('Average Glucose Level: A normal blood glucose level for adults, without diabetes, who haven’t eaten for at least eight hours (fasting) is less than 100 mg/dL. A normal blood glucose level for adults, without diabetes, two hours after eating is 90 to 110 mg/dL', 55.0, 272.0)

hypertension = st.sidebar.selectbox('Hypertension', ['Yes', 'No'])
if hypertension == 'Yes':
    hypertension = 1
else:
    hypertension = 0

heart_disease = st.sidebar.selectbox('Heart Disease', ['Yes', 'No'])
if heart_disease == 'Yes':
    heart_disease = 1
else:
    heart_disease = 0

work_type = st.sidebar.selectbox("Work Type",  ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])

bmi_group=[]
if bmi < 17.0:
        bmi_group.append(0)    
elif (bmi >= 17.0) & (bmi <= 18.49):
    bmi_group.append(1)

elif (bmi >= 18.50) & (bmi <= 24.99):
    bmi_group.append(2)

elif (bmi >= 25.0) & (bmi <= 29.99):
    bmi_group.append(3)

elif (bmi >= 30.0) & (bmi <= 34.99):
    bmi_group.append(4)

elif (bmi >= 35.0) & (bmi <= 39.99):
    bmi_group.append(5)

elif (bmi > 39.99):
    bmi_group.append(6)



d ={'age': [age],
    'gender': [gender],
    'ever_married': [ever_married],
    'smoking_status': [smoking_status],
    'Residence_type': [residence_type],
    'avg_glucose_level': [avg_glucose_level],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'work_type': [work_type]
    }

df = pd.DataFrame(data=d)
df['bmi_group'] = bmi_group 

#### unzip files ######

with py7zr.SevenZipFile('Stroke-Data.7z', mode='r') as z:
    z.extractall()

##### Scaling #########
scaler =  pickle.load(open('scaler.pkl', 'rb'))
df_scale = pd.DataFrame(scaler.fit_transform(df[['age', 'avg_glucose_level']]))
df_scale[['age', 'avg_glucose_level']] = df_scale
df_scale = df_scale[['age', 'avg_glucose_level']]

df.drop(columns = ['age', 'avg_glucose_level'], inplace=True)
df = pd.concat([df, df_scale], axis=1)
df['avg_glucose_level'] = avg_glucose_level
df['age'] = age
    
 
##############################################

df = df[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married','work_type', 'Residence_type', 'avg_glucose_level','smoking_status', 'bmi_group']]

## ohe

ohe = pickle.load(open('ohe.pkl', 'rb'))
df_t = pd.DataFrame(ohe.transform(df.select_dtypes('object')))
df_t.columns = ohe.get_feature_names()

df_n = df.drop(df.select_dtypes('object'), axis=1)
df = pd.concat([df_t, df_n], axis=1)

df = df[['x0_Male', 'x1_Yes', 'x2_Never_worked', 'x2_Private',
       'x2_Self-employed', 'x2_children', 'x3_Urban', 'x4_formerly smoked',
       'x4_never smoked', 'x4_smokes', 'hypertension', 'heart_disease',
       'bmi_group', 'age', 'avg_glucose_level']]


#### Carregando modelo #######################

arquivo_modelo = 'modelo_voting_classifier.pkl'
modelo = pickle.load(open(arquivo_modelo, 'rb'))

preds = modelo.predict(df)
resultado = preds#.item(0)
if resultado == 0:
    resultado = 'No Stroke'
else:
    resultado = 'Stroke'

#st.write(''' # RESULT: ''')
#st.button(label = 'Predict')

st.write('''# Click to get the result: ''')
if st.button('Predict'):
    st.write('''## Result: ''', resultado)






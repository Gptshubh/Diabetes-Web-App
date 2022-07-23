# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:13:34 2022

@author: subha
"""

import numpy as np
import pickle 
import streamlit as st

loaded_model=pickle.load(open('Diabetes-Web-App/trained_model.sav','rb'))

def diabetes_prediction(input_data):
    
    input_data_as_numpy=np.asarray(input_data)

    input_reshaped=input_data_as_numpy.reshape(1,-1)

    pred=loaded_model.predict(input_reshaped)

    if(pred[0]==0):
        return "Not Diabetic"
    else:
        return 'Diabetic'
    

def main():
    
    st.title('Diabetes Prediction Web App')
    
    Pregnancies=st.text_input('Number Of Pregnancies')
    
    Glucose=st.text_input('Glucose level')
    
    BloodPressure=st.text_input('Blood Pressure value')
    
    SkinThickness=st.text_input('Skin Thickness value')
    
    Insulin=st.text_input('Insulin level')
    
    BMI=st.text_input('BMI')
    
    DiabetesPedigreeFunction=st.text_input('Diabetes Pedigree Function')
    
    Age=st.text_input('Age')
    
    diagnosis=' '
    
    
    if st.button('Diabetes Test Result'):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,
        BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis) 
       
       
       
if __name__=='__main__':
        main()

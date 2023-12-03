# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 12:10:22 2023

@author: gauta
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('old/trained_model.sav', 'rb'))

def diabates_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
       return 'The person is not diabetic'
    else:
       return 'The person is diabetic' 
   

def main():
    
    # Giving a title 
    st.title("Diabetes Prediction web App")
    
  	
    #About getting the in
    Pregnancies = st.text_input('Number of Prefnancies')
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input('Blood Pressure level')
    SkinThickness = st.text_input('Skin thinkness value')
    Insulin = st.text_input('Insluin value')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('DiabetesPedigree : ')
    Age = st.text_input('Age of the person')
	
    
    #code for prediction
    diagnosis =  ''
    
    #creating the button for prediction
    if st.button('Diabatic Test Result '):
        diagnosis = diabates_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI, DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    
if __name__ == '__main__' :
    main()
    

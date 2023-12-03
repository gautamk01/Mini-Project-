# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 12:10:22 2023

@author: gauta
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('random_forest_model.sav', 'rb'))

def abnormality_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
       return 'The Cow has some Abnormality  '
    else:
       return 'The Cow is Healthy Cow 🐄' 
   

def main():
    
    # Giving a title 
    st.title("Cow Abnormality Prediction web App")
   


  	
    #About getting the in
    body_temperature = st.text_input('Body Temperature of Cow');
    milk_production = st.text_input('Milk Production')
    respiratory_rate = st.text_input('Respiratory Rate')
    walking_capacity  = st.text_input('Walking Capacity')
    sleeping_duration = st.text_input('Sleep Duration')
    body_condition_score = st.text_input('Body Condition Score')
    heart_rate = st.text_input('Heart Rate Score')
    eating_duration = st.text_input('eating_duration ')
    lying_down_duration = st.text_input('Lying Down Duration ')
    ruminating = st.text_input('Ruminating Rate ')
    rumen_fill = st.text_input('Rumen Fill')
    
	
    
    #code for prediction
    diagnosis =  ''
    
    #creating the button for prediction
    if st.button('Cow Abnormality Test Result '):
        diagnosis = abnormality_prediction([body_temperature,milk_production,respiratory_rate,walking_capacity,sleeping_duration, body_condition_score,heart_rate,eating_duration,
                                          lying_down_duration,ruminating,rumen_fill ])
        
    st.success(diagnosis)
    
if __name__ == '__main__' :
    main()
    

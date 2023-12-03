# -*- coding: utf-8 -*-
"""
Created on Fri Dec 1 12:10:22 2023

@author: gauta
"""

import numpy as np
import pickle
import streamlit as st
from streamlit_lottie import st_lottie
import json

loaded_model = pickle.load(open('random_forest_model.sav', 'rb'))


gif = 'animation30.gif'

# Display the GIF


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


def abnormality_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:

        return 'The Cow has some Abnormality  '
    else:
        return 'The Cow is Healthy Cow 🐄'


lottie_cowhappy = load_lottiefile("animation1.json")
lottie_cowsad = load_lottiefile("animation2.json")
lottie_cowtitle = load_lottiefile("animation3.json")


def main():

    # Giving a title
    # Create two columns
    # adjust the numbers to change the relative width of the columns
    col1, col2 = st.columns([2, 1])

# Put the title in the first column
    with col1:
        st.title("Cow:blue[Care]")
        st.text("An Abnormality Detection System :Using Machine learning Model")

# Put the animation in the second column
    with col2:
        st.image(gif, use_column_width=True)

    # Creating two columns
    col1, col2 = st.columns(2)

    # About getting the input
    with col1:

        body_temperature = st.text_input(
            'Body Temperature of Cow', key='body_temperature')
        milk_production = st.text_input(
            'Milk Production', key='milk_production')
        respiratory_rate = st.text_input(
            'Respiratory Rate', key='respiratory_rate')
        walking_capacity = st.text_input(
            'Walking Capacity', key='walking_capacity')
        sleeping_duration = st.text_input(
            'Sleep Duration', key='sleeping_duration')

    with col2:
        body_condition_score = st.text_input(
            'Body Condition Score', key='body_condition_score')
        heart_rate = st.text_input('Heart Rate Score', key='heart_rate')
        eating_duration = st.text_input(
            'Eating Duration', key='eating_duration')
        lying_down_duration = st.text_input(
            'Lying Down Duration', key='lying_down_duration')
        ruminating = st.text_input('Ruminating Rate', key='ruminating')
        rumen_fill = st.text_input('Rumen Fill', key='rumen_fill')

    # Code for prediction
    diagnosis = ''

    # Creating the button for prediction
    if st.button('Cow Abnormality Test Result'):
        if all([body_temperature, milk_production, respiratory_rate, walking_capacity, sleeping_duration, body_condition_score, heart_rate, eating_duration, lying_down_duration, ruminating, rumen_fill]):
            diagnosis = abnormality_prediction([body_temperature, milk_production, respiratory_rate, walking_capacity,
                                                sleeping_duration, body_condition_score, heart_rate, eating_duration,
                                                lying_down_duration, ruminating, rumen_fill])
            if diagnosis == 'The Cow is Healthy Cow 🐄':
                st.success(diagnosis)
                st_lottie(
                    lottie_cowhappy, speed=1, reverse=False, loop=True, quality="low", height="400px",
                    width=None, key=None
                )
            else:
                st.error(diagnosis)
                st_lottie(
                    lottie_cowsad, speed=1, reverse=False, loop=True, quality="low", height="400px",
                    width=None, key=None
                )
        else:
            st.error('Please fill in all the inputs.')


if __name__ == '__main__':
    main()

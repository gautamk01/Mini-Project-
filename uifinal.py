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

options = ['anorexia', 'abdominal_pain', 'anaemia', 'abortions', 'acetone', 'aggression', 'arthrogyposis',
           'ankylosis', 'anxiety', 'bellowing', 'blood_loss', 'blood_poisoning', 'blisters', 'colic', 'Condemnation_of_livers',
           'coughing', 'depression', 'discomfort', 'dyspnea', 'dysentery', 'diarrhoea', 'dehydration', 'drooling',
           'dull', 'decreased_fertility', 'diffculty_breath', 'emaciation', 'encephalitis', 'fever', 'facial_paralysis', 'frothing_of_mouth',
           'frothing', 'gaseous_stomach', 'highly_diarrhoea', 'high_pulse_rate', 'high_temp', 'high_proportion', 'hyperaemia', 'hydrocephalus',
           'isolation_from_herd', 'infertility', 'intermittent_fever', 'jaundice', 'ketosis', 'loss_of_appetite', 'lameness',
           'lack_of-coordination', 'lethargy', 'lacrimation', 'milk_flakes', 'milk_watery', 'milk_clots',
           'mild_diarrhoea', 'moaning', 'mucosal_lesions', 'milk_fever', 'nausea', 'nasel_discharges', 'oedema',
           'pain', 'painful_tongue', 'pneumonia', 'photo_sensitization', 'quivering_lips', 'reduction_milk_vields', 'rapid_breathing',
           'rumenstasis', 'reduced_rumination', 'reduced_fertility', 'reduced_fat', 'reduces_feed_intake', 'raised_breathing', 'stomach_pain',
           'salivation', 'stillbirths', 'shallow_breathing', 'swollen_pharyngeal', 'swelling', 'saliva', 'swollen_tongue',
           'tachycardia', 'torticollis', 'udder_swelling', 'udder_heat', 'udder_hardeness', 'udder_redness', 'udder_pain', 'unwillingness_to_move',
           'ulcers', 'vomiting', 'weight_loss', 'weakness']
disease = ['mastitis', 'blackleg', 'bloat', 'coccidiosis', 'cryptosporidiosis',
           'displaced_abomasum', 'gut_worms', 'listeriosis', 'liver_fluke', 'necrotic_enteritis', 'peri_weaning_diarrhoea',
           ' rift_valley_fever', 'rumen_acidosis',
           'traumatic_reticulitis', 'calf_diphtheria', 'foot_rot', 'foot_and_mouth', 'ragwort_poisoning', 'wooden_tongue', 'infectious_bovine_rhinotracheitis',
           'acetonaemia', 'fatty_liver_syndrome', 'calf_pneumonia', 'schmallen_berg_virus', 'trypanosomosis', 'fog_fever']

l2 = []
for i in range(0, len(options)):
    l2.append(0)


loaded_model = pickle.load(open('random_forest_model.sav', 'rb'))
loaded_knn = pickle.load(open('knn_model.sav', 'rb'))


gif = 'animation30.gif'

# Display the GIF


def create_dropdowns():

    selected_options = []

    for i in range(5):
        selected_option = st.selectbox(
            f'Symptom {i+1}', options, index=None, key=i, placeholder="Choose an option")
        selected_options.append(selected_option)

    if st.button('Predict the Disease'):
        result = type_of_disease(selected_options)
        for a in range(0, len(disease)):
            if (result == a):
                final = disease[a]
        st.success(f'{final.replace("_", " ")} {" Disease"}')


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
        return 'The Cow is Healthy 🐄'


def type_of_disease(input_data):
    for k in range(0, len(options)):
        for z in input_data:
            if (z == options[k]):
                l2[k] = 1
    inputtest = [l2]
    inputtest = np.array(inputtest)
    knn_predict = loaded_knn.predict(inputtest)   # knn_model
    loaded_knn_predicted = knn_predict[0]

    return loaded_knn_predicted


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
            'Body Temperature of Cow ', placeholder="in °C", key='body_temperature')
        milk_production = st.text_input(
            'Milk Production', placeholder="in liters", key='milk_production')
        respiratory_rate = st.text_input(
            'Respiratory Rate', placeholder="Range(1-100)", key='respiratory_rate')
        walking_capacity = st.text_input(
            'Walking Capacity', placeholder="in Steps", key='walking_capacity')
        sleeping_duration = st.text_input(
            'Sleep Duration', placeholder="in Hurs", key='sleeping_duration')

    with col2:
        body_condition_score = st.text_input(
            'Body Condition Score', placeholder="Range (1-5) ", key='body_condition_score')
        heart_rate = st.text_input(
            'Heart Rate Score', placeholder="Range(1-100)", key='heart_rate')
        eating_duration = st.text_input(
            'Eating Duration', placeholder="in Hours", key='eating_duration')
        lying_down_duration = st.text_input(
            'Lying Down Duration', placeholder="in Hours", key='lying_down_duration')
        ruminating = st.text_input(
            'Ruminating Rate', placeholder="Range(1-10)", key='ruminating')
        rumen_fill = st.text_input(
            'Rumen Fill', placeholder="Range(1-5)", key='rumen_fill')

    # Code for prediction
    # Initialize the session state for diagnosis if it doesn't exist
    if 'diagnosis' not in st.session_state:
        st.session_state['diagnosis'] = ''

# Creating the button for prediction
    if st.button('Cow Abnormality Test Result'):
        if all([body_temperature, milk_production, respiratory_rate, walking_capacity,
                sleeping_duration, body_condition_score, heart_rate, eating_duration,
                lying_down_duration, ruminating, rumen_fill]):
            st.session_state['diagnosis'] = abnormality_prediction([body_temperature, milk_production, respiratory_rate, walking_capacity,
                                                                    sleeping_duration, body_condition_score, heart_rate, eating_duration,
                                                                    lying_down_duration, ruminating, rumen_fill])
            if st.session_state['diagnosis'] == 'The Cow is Healthy 🐄':
                st.success(st.session_state['diagnosis'])
                st_lottie(
                    lottie_cowhappy, speed=1, reverse=False, loop=True, quality="low", height="400px",
                    width=None, key=None
                )
            else:
                st.error(st.session_state['diagnosis'])
                st_lottie(
                    lottie_cowsad, speed=1, reverse=False, loop=True, quality="low", height="400px",
                    width=None, key=None
                )

        else:
            st.error('Please fill in all the inputs.')
    if (st.session_state['diagnosis'] == 'The Cow has some Abnormality  '):
        create_dropdowns()


if __name__ == '__main__':
    main()

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
        st.success(
            f'{"Your cow is likely to have a "}{final.replace("_", " ")}')


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

    expander = st.expander("Help Desk")
    expander.write(""" 
    # Help Desk
                   
    ## Body temperature of cow

        A thermometer is an extremely useful and inexpensive piece of equipment for cattle keepers. 
        
        how to measure it 📏 
        The cow should be adequately restrained using a crush, the thermometer inserted into the rectum and pressed up against the mucosa (lining) of the rectum. The normal temperature of an adult cow is around 38.5°C.

    ## Body Condition Score

    Body condition refers to the relative amount of subcutaneous body fat or energy reserve in the cow.A 5-point (1-5) scoring system to measure the relative amount of this subcutaneous body fat. 
    ### how to measure it
    #### Resource 
    [document](https://www.vet.cornell.edu/sites/default/files/1e_Elanco%20Cow%20Body_condition_scoring_V3.pdf)

    [video](https://www.youtube.com/watch?v=BfW97OH02E0)

    ## Milk production

        Milk production refers to the process by which mammals, particularly cows in this context, produce milk.
        
        how to measure it :-
        1. Milking Records:
        - Daily Records: Keeping daily records of the amount of milk produced by each cow is a fundamental method. This can be done by recording the volume of milk collected during each milking session.
        - Milking Frequency: The number of times a cow is milked each day can impact overall milk production. Most dairy cows are milked two or three times a day.

        2. Milk Meters:
            - Installing milk meters is a more automated approach. These devices are attached to milking machines and can accurately measure the amount of milk produced by each cow during each milking session.

        3. Weighing:
            - An alternative method is to weigh the milk. This can be done by weighing the milk container before and after milking. The weight difference provides an estimate of the milk produced.

    ## Heart rate score

        The adult cow has a heart rate of between 48 and 84 beats per minute.
        
        how to measure it :-
        This can be assessed by using a stethoscope and listening over the left hand side of the cow's chest behind the cow's elbow.

    ## Respiratory Rate

        Respiratory rate is typically measured in breaths per minute (bpm). A normal respiratory rate for a healthy adult cow at rest is around 10 to 30 breaths per minute, but this can vary based on factors such as age, size, and environmental conditions.
        how to measure it :-
        This can be assessed by quietly watching the cow's ribs, count the number of times they move out on inspiration in 15 seconds and then multiply by 4.  A cow's respiratory rate may vary with the ambient temperature and if the cow is stressed but the adult cow's respiratory rate should be between 26 and 50 breaths per minute.

    ## Eating Duration


        The eating duration of a cow, or the time it spends eating, is an important aspect of its behavior and can provide insights into its health, nutrition, and overall well-being. Monitoring eating duration can be particularly relevant in dairy farming and beef production to ensure that cows are receiving adequate nutrition and are not facing any health issues.
        how to measure it :-
        This can be assessed by quietly watching the cow. count the time it take to eat and measure it using a stop watch.

    ## Walking Capacity

        The Walking Capacity is the cow's ability to walk or move around, and it is an important aspect of its overall well-being. A healthy walking capacity is indicative of good mobility, sound limb structure, and proper joint function. Regularly observe the cow's gait and movement patterns. A cow with a normal walking capacity will move freely and without obvious signs of lameness or discomfort.
        how to measure it :-
        1. Observe the cow's gait and movement patterns. Record it manually.

        2. Use technology, such as motion sensors or accelerometers, to monitor the activity and movement patterns of cows. These tools can provide data on walking distances and times.

    ## Lying down duration

        The lying down duration of a cow refers to the amount of time a cow spends in a lying position. Monitoring lying down behavior is an essential aspect of assessing the well-being and comfort of cows. Cows typically lie down for various reasons, including rest, rumination, and sleep. Changes in lying down patterns can be indicative of health 
        
        how to measure it:-

        1. Observe the cows to assess their lying down behavior. Record it manually.

        2. Use sensors or monitoring systems to track the activity of cows, including lying down times. These sensors may be attached to collars or ear tags and can provide data on behavioral patterns.

    ## Sleep duration

        Cows, like many mammals, require periods of sleep for their well-being. Sleep duration for cows refers to the amount of time they spend in various sleep stages, which include both non-REM (Rapid Eye Movement) sleep and REM sleep. During sleep, cows experience relaxation, rest, and physiological processes important for their overall health. 
        
        how to measure it:-

        1.  Observe the cows to determine their sleep patterns. Sleep in cows typically involves lying down, and they may exhibit different lying positions during sleep, including sternal (on their chest) or lateral (on their side) positions.. Record it manually.

        2. Use sensors or monitoring systems to track the activity of cows, including sleeping times. These sensors may be attached to collars or ear tags and can provide data on behavioral patterns.

    ## Ruminating Rate

        After eating, cows engage in rumination, a process where they regurgitate and re-chew their food to aid in digestion. Monitoring the time spent on rumination, in addition to actual eating time, can provide a more comprehensive view of their digestive health. They will eat for 3-4 hours.  Normal cows will spend on average 6 hours chewing the cud each day (depending on their diet), most of this will be when they are lying.
        how to measure it :-
        1.  Observe the cows to determine their ruminating rate. ruminating in cows typically involves lying down. Record it manually.

        2. Use sensors or monitoring systems to track the activity of cows, including ruminating rate. These sensors may be attached to collars or ear tags and can provide data on behavioral patterns.

    ## Rumen Fill

        The rumen in the cow is located predominantly on the left hand side of the abdomen and the area of it that can be detected is the top of the flank between the last rib and the pelvis; if a cow is off her feed the rumen will appear as a sharp triangle as it is empty, the more the rumen fills, the more this space fills.  If the space protrudes outwards, this may be a sign of bloat, seek veterinary advice.  There is a scoring system for rumen fill that vets may use, the rumen is assigned a score from 1 to 5.
        how to measure it :-
        1. Observation of a cow's left side (where the rumen is located) can provide an initial indication of rumen fill. A well-filled rumen appears rounded and firm. An empty or excessively distended rumen may be cause for concern.

        To assess rumen fill, use the left-hand side of the cow behind the last rib, under the transverse processes of the spine and in front of the hook bone (Figure 1). It may be possible to feel a slightly firmer area which indicates the fibre mat of the rumen – sitting atop the liquid portion - sometimes with a small cap of gas on top (indicated by a softer area). The rumen fill score reflects intakes in the past 2–6 hours and targets are different depending on the physiological status of the cow (Figure 2).
    """)
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

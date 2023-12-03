import pickle
import pandas as pd

# Load the pickled model
with open('D:/ML/trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the test dataset
dataset = pd.read_csv('D:/ML/cattle_dataset.csv')

# Preprocess the dataset
dataset = dataset.drop('faecal_consistency', axis=1)
dataset = dataset.drop('breed_type', axis=1)
dataset.dropna(inplace=True)
dataset = dataset.replace({'healthy': 1, 'unhealthy': 0})

# Split the dataset into features and target variables
X_test = dataset.drop('health_status', axis=1)
y_test = dataset['health_status']

# Evaluate the model's performance
performance = model.score(X_test, y_test)

# Print the performance metric
print("Model performance:", performance)

# Predict with the given values
new_data = pd.DataFrame({'body_temperature': [38.5],
                         'milk_production': [20.2],
                         'respiratory_rate': [16.0],
                         'walking_capacity': [5004.0],
                         'sleeping_duration': [4.6],
                         'body_condition_score': [5.0],
                         'heart_rate': [55.0],
                         'eating_duration': [2.5],
                         'lying_down_duration': [12.5],
                         'ruminating': [4.4],
                         'rumen_fill': [2.0]})


prediction = model.predict(new_data)
# note: predicted value should be healthy
# Map the predicted value to 'healthy' or 'unhealthy'
result = 'healthy' if prediction[0] == 1 else 'unhealthy'
print("Predicted value:", result)
def abnormality_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
       return 'The Cow has some Abnormality'
    else:
       return 'The Cow is Healty' 
   

def main():
    
    # Giving a title 
    st.title("Cow Abnormality web App")
    
                        
                       
                    
  	
    #About getting the in
    Body_temperature = st.text_input('body_temperature')
    milk_production = st.text_input("Milk Production")
    respiratory_rate = st.text_input('Respiratory Rate')
    walking_capacity = st.text_input('Walking Capacity')
    sleeping_duration = st.text_input('Sleeping Duration')
    body_condition_score = st.text_input('Body Condition Score')
    heart_rate = st.text_input('Heart Rate ')
    eating_duration = st.text_input('Eating Duration')
    lying_down_duration = st.text_input('Lying Down Duration')
    rumen_fill = st.text_input('Rumen Fill')
	
    
    #code for prediction
    diagnosis =  ''
    
    #creating the button for prediction
    if st.button('Abnormality Test Result '):
        diagnosis = abnormality_prediction([Body_temperature,milk_production,respiratory_rate,walking_capacity,sleeping_duration,
                                         body_condition_score,heart_rate,eating_duration,lying_down_duration,rumen_fill]);
        
    st.success(diagnosis)
    
if __name__ == '__main__' :
    main()
    
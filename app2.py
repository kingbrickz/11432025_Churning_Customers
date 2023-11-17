# import streamlit as st
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# import pickle
# from tensorflow import keras

# model_path = "final_model.h5"
# model = keras.models.load_model(model_path)

# scaler_path = "scaler.pkl"
# with open(scaler_path,"rb")as scaler_file:
#     scaler = pickle.load(scaler_file)

# # Create the Streamlit app
# st.title('Customer Churn Prediction')

# # Create a form to collect user input
# st.form('Customer Info:')
# monthly_charges = st.number_input('Monthly Charges')
# paperless_billing = st.selectbox('Paperless Billing', options=[0, 1])
# senior_citizen = st.selectbox('Senior Citizen', options=[0, 1])
# payment_method = st.selectbox('Payment Method', options=['Electronic Check', 'Bank Transfer (automatic)', 'Mailed Check', 'Credit card (automatic)'])
# multiple_lines = st.selectbox('Multiple Lines', options=[0, 1])
# phone_service = st.selectbox('Phone Service', options=['No', 'Yes'])
# gender = st.selectbox('Gender', options=['Male', 'Female'])
# streaming_tv = st.selectbox('Streaming TV', options=[0, 1])
# streaming_movies = st.selectbox('Streaming Movies', options=[0, 1])
# internet_service = st.selectbox('Internet Service', options=['Fiber optic', 'DSL'])
# partner = st.selectbox('Partner', options=['No', 'Yes'])
# submit_button = st.button('Predict Churn')

# # Make the prediction if the form is submitted
# if submit_button:
#     # Create a new dataset with user input
#     new_data = pd.DataFrame({
#         'MonthlyCharges': [monthly_charges],
#         'PaperlessBilling': [paperless_billing],
#         'SeniorCitizen': [senior_citizen],
#         'PaymentMethod': [payment_method],
#         'MultipleLines': [multiple_lines],
#         'PhoneService': [phone_service],
#         'gender': [gender],
#         'StreamingTV': [streaming_tv],
#         'StreamingMovies': [streaming_movies],
#         'InternetService': [internet_service],
#         'Partner': [partner]
#     })

#     # Preprocess the new data
#     new_data_scaled = scaler.transform(new_data)

#     # Make the prediction
#     prediction = model.predict(new_data_scaled)[0]

#     # Show the prediction result
#     if prediction == 0:
#         st.success('The customer is unlikely to churn.')
#     else:
#         st.error('The customer is likely to churn.')





import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from tensorflow import keras
import numpy as np

# Load the trained model
model_path = "final_model.h5"
model = keras.models.load_model(model_path)

# Load the scaler
scaler_path = "scaler.pkl"
with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Create the Streamlit app
st.title('Customer Churn Prediction')

# Create a form to collect user input
st.form('Customer Info:')
monthly_charges = st.number_input('Monthly Charges')
paperless_billing = st.selectbox('Paperless Billing', options=[0, 1])
senior_citizen = st.selectbox('Senior Citizen', options=[0, 1])
payment_method = st.selectbox('Payment Method', options=['2', '0', '3', '1'])
multiple_lines = st.selectbox('Multiple Lines', options=[0, 1])
phone_service = st.selectbox('Phone Service', options=[0, 1])
gender = st.selectbox('Gender', options=[0, 1])
streaming_tv = st.selectbox('Streaming TV', options=[0, 1])
streaming_movies = st.selectbox('Streaming Movies', options=[0, 1])
internet_service = st.selectbox('Internet Service', options=[0, 1])
partner = st.selectbox('Partner', options=[0, 1])
submit_button = st.button('Predict Churn')

# Make the prediction if the form is submitted
if submit_button:
    # Create a new dataset with user input
    new_data = pd.DataFrame({
        'MonthlyCharges': [monthly_charges],
        'PaperlessBilling': [paperless_billing],
        'SeniorCitizen': [senior_citizen],
        'PaymentMethod': [payment_method],
        'MultipleLines': [multiple_lines],
        'PhoneService': [phone_service],
        'gender': [gender],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'InternetService': [internet_service],
        'Partner': [partner]
    })

    # Preprocess the new data
    new_data_scaled = scaler.transform(new_data)

    # Make the prediction
    prediction = model.predict(new_data_scaled)[0]

        # Compute confidence interval

    n_simulations = 1000  
    simulations = np.random.binomial(1, prediction, n_simulations)
    confidence_interval = np.percentile(simulations, q=[2.5, 97.5])

    # Show the prediction result
    if prediction == 0:
        st.success('The customer is likely to churn.')
    else:
        st.error('The customer is likely to churn.')
    st.write(f'Confidence Interval:  {confidence_interval}')


st.markdown("## Additional Information")
st.markdown(
    """
    This is a paragraph of additional information that you need  to use  in your Streamlit app to predict churn .
   
    paperless biling  1 is a yes and 0 is a no ,
    senior citizen = 0 is not a senior citzen 1 is a senior citzen ,
    payment method= electronic is 2, bank is 0 , mailedcheck=3,creditcard is 1,
    multipleline= 0 is no and 1 is yes,
    phone serive= 1 is yes and 0 is no,
    gender 0 is male 1 is female,
    streamingtv 1 is yes and 0 is no,
    streaming movies 1 is yes 0 is no,
    internetservice 0 is dsl 1 is fiberoptic,
    partner 1 is yes and 0 is no ,

    
    """
)
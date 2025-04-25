import streamlit as st
import pandas as pd
import joblib

# Load the model and mappings
xgb_model = joblib.load('xgboost_model.pkl')
le = joblib.load('label_encoder_no_show.pkl')
neigh_mapping = joblib.load('neighbourhood_mapping.pkl')

# Column structure from training
X_columns = ['Gender', 'Age', 'Neighbourhood', 'Scholarship', 'Hipertension',
             'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received',
             'WaitingDays', 'AppointmentWeekday']


def predict_no_show(user_input):
    # Gender: 'F' -> 0, 'M' -> 1
    gender_map = {'F': 0, 'M': 1}
    user_input['Gender'] = gender_map.get(user_input['Gender'].upper(), 0)

    # Encode Neighbourhood using saved mapping
    neighbourhood = user_input['Neighbourhood']
    user_input['Neighbourhood'] = neigh_mapping.get(neighbourhood, 0)  # default to 0 if not found

    # SMS_received: 'Yes' -> 1, 'No' -> 0
    user_input['SMS_received'] = 1 if str(user_input['SMS_received']).strip().lower() == 'yes' else 0

    # Create input DataFrame
    input_df = pd.DataFrame([user_input], columns=X_columns)

    # Predict
    prediction = xgb_model.predict(input_df)
    prediction_decoded = le.inverse_transform(prediction)

    # Result
    input_text = f"Patient Info:\n{user_input}\n"
    input_text += f"\nPrediction: The patient will {'not show up' if prediction_decoded[0] == 1 else 'show up'} for the appointment."

    return input_text


# Example input
user_input = {
    'Gender': 'F',
    'Age': 45,
    'Neighbourhood': 'JARDIM CAMBURI',
    'Scholarship': 0,
    'Hipertension': 0,
    'Diabetes': 0,
    'Alcoholism': 0,
    'Handcap': 0,
    'SMS_received': 'Yes',
    'WaitingDays': 1,
    'AppointmentWeekday': 1
}

# Run prediction
prediction_text = predict_no_show(user_input)
print(prediction_text)

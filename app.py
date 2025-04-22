import streamlit as st
import pandas as pd
import joblib

# Load the model and encoders
xgb_model = joblib.load('xgboost_model.pkl')
le = joblib.load('label_encoder_no_show.pkl')
neigh_mapping = joblib.load('neighbourhood_mapping.pkl')

# Column structure from training
X_columns = ['Gender', 'Age', 'Neighbourhood', 'Scholarship', 'Hipertension',
             'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received',
             'WaitingDays', 'AppointmentWeekday']


def predict_no_show(user_input):
    gender_map = {'F': 0, 'M': 1}
    user_input['Gender'] = gender_map.get(user_input['Gender'].upper(), 0)
    user_input['Neighbourhood'] = neigh_mapping.get(user_input['Neighbourhood'], 0)
    user_input['SMS_received'] = 1 if str(user_input['SMS_received']).strip().lower() == 'yes' else 0

    input_df = pd.DataFrame([user_input], columns=X_columns)
    prediction = xgb_model.predict(input_df)
    prediction_decoded = le.inverse_transform(prediction)
    return prediction_decoded[0]


# Streamlit App
st.set_page_config(page_title="No-Show Prediction App", layout="centered")

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict"])

# Home Page
if page == "Home":
    st.title("📅 Medical Appointment No-Show Predictor")
    st.markdown("""
        Welcome to the **No-Show Prediction App**.  
        This tool predicts whether a patient will show up for their medical appointment based on input data.  
        Navigate to the **Predict** tab to get started.
    """)

# Predict Page
elif page == "Predict":
    st.title("📊 Predict No-Show")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["F", "M"])
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            scholarship = st.selectbox("Scholarship (Govt. Assistance)", ["No", "Yes"])
            hipertension = st.selectbox("Hypertension", ["No", "Yes"])
            diabetes = st.selectbox("Diabetes", ["No", "Yes"])

        with col2:
            neighbourhood = st.selectbox("Neighbourhood", sorted(neigh_mapping.keys()))
            alcoholism = st.selectbox("Alcoholism", ["No", "Yes"])
            handcap = st.selectbox("Handicap", [0, 1, 2, 3, 4])
            sms_received = st.selectbox("SMS Received", ["No", "Yes"])
            waiting_days = st.number_input("Waiting Days", min_value=0, max_value=100, value=1)
            weekday = st.selectbox("Appointment Weekday (0=Mon ... 6=Sun)", list(range(7)))

        submit = st.form_submit_button("Predict")

    if submit:
        user_input = {
            'Gender': gender,
            'Age': age,
            'Neighbourhood': neighbourhood,
            'Scholarship': 1 if scholarship == 'Yes' else 0,
            'Hipertension': 1 if hipertension == 'Yes' else 0,
            'Diabetes': 1 if diabetes == 'Yes' else 0,
            'Alcoholism': 1 if alcoholism == 'Yes' else 0,
            'Handcap': handcap,
            'SMS_received': sms_received,
            'WaitingDays': waiting_days,
            'AppointmentWeekday': weekday
        }

        result = predict_no_show(user_input)
        st.success(f"🩺 Prediction: The patient will {'NOT show up' if result == 1 else 'show up'} for the appointment.")

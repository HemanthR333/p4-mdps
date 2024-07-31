import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Set page title, layout, and icon
st.set_page_config(
    page_title="Health Assistant",
    layout="wide",
    page_icon="🧑‍⚕️"
)

# Function to load models
@st.cache_resource()
def load_models():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
    heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
    parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))
    return diabetes_model, heart_disease_model, parkinsons_model

# Load models
diabetes_model, heart_disease_model, parkinsons_model = load_models()

# Sidebar menu for disease selection
with st.sidebar:
    st.title('Multiple Disease Prediction System')
    selected = option_menu(
        'Choose a Disease',
        ['Diabetes Prediction',
         #'Heart Disease Prediction', 'Parkinsons Prediction'
         ],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

# Main content based on selected disease
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using Machine Learning')
    st.write('Please enter the following information:')

    # Input fields for diabetes prediction
    with st.form(key='diabetes_form'):
        col1, col2, col3 = st.columns(3)
        with col1:
            pregnancies = st.number_input('Number of Pregnancies', min_value=0, step=1)
            glucose = st.number_input('Glucose Level', min_value=0.0)
            blood_pressure = st.number_input('Blood Pressure', min_value=0.0)
        with col2:
            skin_thickness = st.number_input('Skin Thickness', min_value=0.0)
            insulin = st.number_input('Insulin Level', min_value=0.0)
            bmi = st.number_input('BMI', min_value=0.0)
        with col3:
            diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, format="%.3f")
            age = st.number_input('Age', min_value=0)

        submitted = st.form_submit_button('Predict Diabetes')

    if submitted:
        # Perform prediction
        user_input = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
        prediction = diabetes_model.predict([user_input])[0]
        if prediction == 1:
            st.error('The person is predicted to have diabetes.')
        else:
            st.success('The person is predicted to be diabetes-free.')

elif selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using Machine Learning')
    st.write('Please enter the following information:')

    # Input fields for heart disease prediction
    with st.form(key='heart_disease_form'):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input('Age', min_value=0)
            sex = st.selectbox('Sex', ['Male', 'Female'])
            cp = st.selectbox('Chest Pain Types', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
        with col2:
            trestbps = st.number_input('Resting Blood Pressure', min_value=0.0)
            chol = st.number_input('Serum Cholestoral (mg/dl)', min_value=0.0)
            fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['True', 'False'])
        with col3:
            restecg = st.selectbox('Resting Electrocardiographic Results', ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'])
            thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0.0)
            exang = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
        with col1:
            oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0)
            slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
            ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', [0, 1, 2, 3])
        with col2:
            thal = st.selectbox('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect', [0, 1, 2, 3])

        submitted = st.form_submit_button('Predict Heart Disease')

    if submitted:
        # Perform prediction
        sex_value = 1 if sex == 'Male' else 0
        fbs_value = 1 if fbs == 'True' else 0
        exang_value = 1 if exang == 'Yes' else 0
        slope_value = 0 if slope == 'Upsloping' else (1 if slope == 'Flat' else 2)
        user_input = [age, sex_value, cp, trestbps, chol, fbs_value, restecg, thalach, exang_value, oldpeak, slope_value, ca, thal]
        prediction = heart_disease_model.predict([user_input])[0]
        if prediction == 1:
            st.error('The person is predicted to have heart disease.')
        else:
            st.success('The person is predicted to be heart disease-free.')

elif selected == 'Parkinsons Prediction':
    st.title("Parkinson's Disease Prediction using Machine Learning")
    st.write('Please enter the following information:')

    # Input fields for Parkinson's disease prediction
    with st.form(key='parkinsons_form'):
        col1, col2, col3 = st.columns(3)
        with col1:
            fo = st.number_input('MDVP:Fo(Hz)', min_value=0.0)
            fhi = st.number_input('MDVP:Fhi(Hz)', min_value=0.0)
            flo = st.number_input('MDVP:Flo(Hz)', min_value=0.0)
            jitter_percent = st.number_input('MDVP:Jitter(%)', min_value=0.0)
            jitter_abs = st.number_input('MDVP:Jitter(Abs)', min_value=0.0)
        with col2:
            rap = st.number_input('MDVP:RAP', min_value=0.0)
            ppq = st.number_input('MDVP:PPQ', min_value=0.0)
            ddp = st.number_input('Jitter:DDP', min_value=0.0)
            shimmer = st.number_input('MDVP:Shimmer', min_value=0.0)
            shimmer_db = st.number_input('MDVP:Shimmer(dB)', min_value=0.0)
        with col3:
            apq3 = st.number_input('Shimmer:APQ3', min_value=0.0)
            apq5 = st.number_input('Shimmer:APQ5', min_value=0.0)
            apq = st.number_input('MDVP:APQ', min_value=0.0)
            dda = st.number_input('Shimmer:DDA', min_value=0.0)
            nhr = st.number_input('NHR', min_value=0.0)
        with col1:
            hnr = st.number_input('HNR', min_value=0.0)
            rpde = st.number_input('RPDE', min_value=0.0)
            dfa = st.number_input('DFA', min_value=0.0)
            spread1 = st.number_input('spread1', min_value=0.0)
            spread2 = st.number_input('spread2', min_value=0.0)
        with col2:
            d2 = st.number_input('D2', min_value=0.0)
            ppe = st.number_input('PPE', min_value=0.0)

        submitted = st.form_submit_button("Predict Parkinson's Disease")

    if submitted:
        # Perform prediction
        user_input = [fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db,
                      apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]
        prediction = parkinsons_model.predict([user_input])[0]
        if prediction == 1:
            st.error("The person is predicted to have Parkinson's disease.")
        else:
            st.success("The person is predicted to be Parkinson's disease-free.")

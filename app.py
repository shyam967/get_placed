import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# Set the page layout to wide
st.set_page_config(layout="wide")

# Load the saved scaler and model
with open("graduate_adm_scalar.pkl", "rb") as f:
    saved_scalar = pickle.load(f)
with open("graduate_adm_model.pkl", 'rb') as f:
    saved_model = pickle.load(f)

# Load the colleges data
colleges_df = pd.read_csv('mtech_colleges.csv')

# Add custom CSS for full screen and alignment
st.markdown("""
    <style>
    /* General page styling */
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #f4f4f9;
        color: #333;
    }
    .title {
        font-size: 3em;
        color: #4CAF50;
        text-align: center;
        margin-top: 0;
        padding-top: 1rem;
    }
    .description {
        text-align: center;
        font-size: 1.2em;
        margin-bottom: 2rem;
        color: #666;
    }

    /* Adjust the padding and margin for a cleaner layout */
    .block-container {
        padding: 0 3rem;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 1.2em;
        border-radius: 8px;
        padding: 0.5em 1em;
        transition: background-color 0.3s ease;
    }
    .stButton button:hover {
        background-color: #45a049;
    }

    /* Style input fields */
    input[type=number], select {
        width: 100%;
        padding: 12px;
        border: 1px solid #ccc;
        border-radius: 4px;
        box-sizing: border-box;
        margin-top: 6px;
        margin-bottom: 16px;
        resize: vertical;
    }
    input[type=number]:focus, select:focus {
        border-color: #4CAF50;
        box-shadow: 0 0 8px 0 rgba(76, 175, 80, 0.5);
    }

    /* Hover effects for the recommendation list */
    .recommended-colleges {
        list-style-type: none;
        padding-left: 0;
    }
    .recommended-colleges li {
        background-color: #fff;
        padding: 10px;
        margin: 8px 0;
        border: 1px solid #ddd;
        border-radius: 5px;
        transition: background-color 0.3s ease;
    }
    .recommended-colleges li:hover {
        background-color: #f0f8ff;
    }

    /* Footer styling */
    footer {
        font-size: 0.9em;
        text-align: center;
        padding: 2em;
        background-color: #f4f4f9;
        color: #999;
    }

    /* Hide the unnecessary UI components */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


# Top Navigation Menu
st.markdown("<h1 class='title'>Get Placed</h1>", unsafe_allow_html=True)
st.markdown("<p class='description'>This application predicts the chance of a student's admission to graduate school based on various parameters. Fill in the details below and click 'Predict' to see the results.</p>", unsafe_allow_html=True)

# Layout with two columns: left for input, right for results
col1, col2 = st.columns(2)

with col1:
    st.header('User Input Parameters')
    st.write("Please input the following parameters:")
    
    # Create input fields without predefined values
    gre = st.number_input('GRE Score (0 to 340)', min_value=0, max_value=340, step=1, value=0)
    toefl = st.number_input('TOEFL Score (0 to 120)', min_value=0, max_value=120, step=1, value=0)
    university_rating = st.selectbox('University Rating (1 to 5)', [1, 2, 3, 4, 5], index=0)
    sop = st.number_input('Statement of Purpose (1.0 to 5.0)', min_value=1.0, max_value=5.0, step=0.1, value=1.0)
    lor = st.number_input('Letter of Recommendation Strength (1.0 to 5.0)', min_value=1.0, max_value=5.0, step=0.1, value=1.0)
    cgpa = st.number_input('CGPA (1.00 to 10.00)', min_value=1.00, max_value=10.00, step=0.01, value=1.00)
    research = st.selectbox('Research Experience', ['No', 'Yes'])

    # Convert 'Yes'/'No' to 1/0
    research = 1 if research == 'Yes' else 0

    # Create a button for prediction
    if st.button('Predict'):
        # Placeholder for the loading message
        loading_message = st.empty()
        loading_message.info('Analyzing data and predicting...')

        # Simulate a delay for processing
        time.sleep(3)

        # Remove the loading message
        loading_message.empty()

        # Prepare the input data
        data = {
            'GRE Score': gre,
            'TOEFL Score': toefl,
            'University Rating': university_rating,
            'SOP': sop,
            'LOR ': lor,
            'CGPA': cgpa,
            'Research': research
        }

        user_input = pd.DataFrame(data, index=[0])
        scaled_input = saved_scalar.transform(user_input)
        prediction = saved_model.predict(scaled_input)
        predicted_percentage = np.round(prediction[0] * 100, 2)

        # Display the prediction
        with col2:
            st.subheader('Prediction')
            st.markdown(f"Your predicted admission chance is: {predicted_percentage}%")
            
            st.subheader('Recommended Universities')
            # Filter universities based on the predicted admission chance
            recommended_colleges = colleges_df[colleges_df['Cutoff Percentage'] <= predicted_percentage]

            if not recommended_colleges.empty:
                st.write("Based on your predicted admission chances, you can consider applying to the following universities:")
                for _, row in recommended_colleges.iterrows():
                    st.write("- " + row['College Name'])
            else:
                st.write("Unfortunately, there are no universities that match your predicted admission chances.")



import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# Load your model
pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)

# Define pages
def main_page():
    st.title("🏦 Bank Note Authenticator")
    st.markdown("""
        <style>
        .main-title {text-align: center; font-size: 30px; color: #0066cc;}
        </style>
        <h1 class="main-title">Welcome to the Bank Note Authenticator App!</h1>
    """, unsafe_allow_html=True)
    st.image("bank_notes.jpg", use_column_width=True)
    st.markdown("Use the navigation menu to explore various functionalities.")

def prediction_page():
    st.title("🔮 Predict Bank Note Authentication")
    st.markdown("""
        Enter the details below to authenticate a bank note:
    """)
    
    variance = st.text_input("Variance", "")
    skewness = st.text_input("Skewness", "")
    curtosis = st.text_input("Curtosis", "")
    entropy = st.text_input("Entropy", "")
    
    if st.button("Predict"):
        try:
            input_data = pd.DataFrame({
                'variance': [float(variance)],
                'skewness': [float(skewness)],
                'curtosis': [float(curtosis)],
                'entropy': [float(entropy)]
            })
            result = classifier.predict(input_data)
            st.success(f"✅ The predicted class is: {'Authentic' if result[0] == 1 else 'Fake'}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

def file_upload_page():
    st.title("📁 Bulk Prediction via CSV Upload")
    uploaded_file = st.file_uploader("Upload a CSV File", type=["csv"])
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.dataframe(data.head())
            predictions = classifier.predict(data)
            data['Prediction'] = predictions
            st.markdown("### Predictions:")
            st.dataframe(data)
            st.download_button("Download Predictions", data.to_csv(index=False), "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error in processing file: {e}")

def about_page():
    st.title("📖 About the Project")
    st.markdown("""
        ### Bank Note Authenticator
        This application uses a machine learning model to authenticate banknotes. The features of the model include:
        - **Variance** of the image wavelet transformed.
        - **Skewness** of the wavelet transformed image.
        - **Curtosis** of the wavelet transformed image.
        - **Entropy** of the image.

        **Technologies Used**:
        - **Machine Learning**: Scikit-learn
        - **Frontend**: Streamlit
        - **Backend**: Flask (optional)
        
        Built with full dedication by **Pahuldeep Singh Dhingra**.
    """)

# Navigation
def run_app():
    st.sidebar.title("Navigation")
    options = ["Home", "Predict", "Bulk Prediction", "About"]
    choice = st.sidebar.radio("Go to:", options)

    if choice == "Home":
        main_page()
    elif choice == "Predict":
        prediction_page()
    elif choice == "Bulk Prediction":
        file_upload_page()
    elif choice == "About":
        about_page()

if __name__ == "__main__":
    run_app()

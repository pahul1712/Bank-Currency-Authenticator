'''

from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
# import flasgger  
# from flasgger import Swagger
import streamlit as st
from PIL import Image

#app=Flask(__name__)
# Swagger(app)

pickle_in=open("classifier.pkl","rb")
classifier=pickle.load(pickle_in) 

# @app.route('/')
def welcome():
    return "Welcome All"

# @app.route('/predict', methods = ["Get"])
def predict_note_authetication(variance,skewness,curtosis,entropy):
    
    """Let's Authenticate the Banks Note
    This is using docstrings for specifications.
    ---
    parameters:
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query 
        type: number
        required: true
    responses:
        200:
            description: The output values
    
    """
    
    
    input_data = pd.DataFrame({
        'variance': [float(variance)],
        'skewness': [float(skewness)],
        'curtosis': [float(curtosis)],
        'entropy': [float(entropy)]
    })
    
    # Make prediction
    prediction = classifier.predict(input_data)
    return prediction

# @app.route('/predict_file', methods=["POST"])
def predict_note_file():
    """Let's Authenticate the Banks Note
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    responses:
        200:
            description: The output values
    
    """
    df_test = pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction = classifier.predict(df_test)
    return "The Predicted Value for the CSV File is" + str(list(prediction))


def main():
  st.title("Bank Authenticator")
  html_temp = """
  <div style='background-color:blue;padding:10px;'>
  <h2 style='color:white;text-align:center;'>Streamlit Bank Authenticator ML App</h2>
  </div>
  """
  st.markdown(html_temp, unsafe_allow_html=True)

  variance = st.text_input("Variance", "Type Here")
  skewness = st.text_input("Skewness", "Type Here")
  curtosis = st.text_input("Curtosis", "Type Here")
  entropy = st.text_input("Entropy", "Type Here")
  
  result=""
  
  if st.button("Predict"):
    result = predict_note_authetication(variance,skewness,curtosis,entropy)
  st.success("The output is {}".format(result))
  if st.button("About"):
    st.text("This App uses a ML Model to Authenticate Bank Notes")
    st.text("Built with Streamlit and Scikit-Learn")
  
  
if __name__=="__main__":
  main()
  
'''

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

from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
# import flasgger  
# from flasgger import Swagger
import streamlit as st

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
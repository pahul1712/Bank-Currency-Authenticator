 # **Bank-Currency-Authenticator**

Bank Currency Authenticator is a machine learning-based application that predicts whether a given currency note is genuine or fake. The model is built using the Random Forest Classifier and is deployed with both Flask and Streamlit frameworks for ease of access.

## Features 

- Predicts the authenticity of a banknote using features such as variance, skewness, curtosis, and entropy.
- Accepts single note predictions through user inputs.
- Allows batch predictions from uploaded CSV files.
- Two deployment options:
   -Flask API: Provides endpoints for predictions with Swagger documentation.
   -Streamlit Web App: Interactive web interface for user-friendly predictions.

## Technologies Used

- Machine Learning: Random Forest Classifier
- Backend Frameworks: Flask, Streamlit
- Libraries: pandas, NumPy, scikit-learn, flasgger
- Deployment: Docker (for containerization)

## File Structure

- BankNote_Authentication.csv: Dataset used to train the model.
- Bank Note Authentication.ipynb: Jupyter Notebook for training the model.
- classifier.pkl: Serialized Random Forest Classifier model.
- flask_api.py: Flask application for API-based predictions.
- app.py: Streamlit application for an interactive web interface.
- Dockerfile: Docker configuration for deployment.
- requirements.txt: Python dependencies required for the project.

## Feel free to tweak it to fit your preferences!

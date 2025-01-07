import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Load your model
pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)

# Dark Mode Toggle
st.set_page_config(page_title="Bank Note Authenticator", layout="centered")

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
    st.write("Want to learn more about the project? Visit the **About** section!")

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
            
            # Save the prediction result in session state for use in the confusion matrix
            if "prediction_result" not in st.session_state:
                st.session_state["prediction_result"] = []

            st.session_state["prediction_result"].append(result[0])
            
        except Exception as e:
            st.error(f"Error in prediction: {e}")
    
    # Render the Confusion Matrix if checkbox is selected
    if st.checkbox("Show Confusion Matrix"):
        if "prediction_result" in st.session_state:
            # Example confusion matrix data (replace this with actual results if available)
            # Sample static confusion matrix for demo purposes
            y_true = [1, 1, 0, 0]  # Replace with actual labels
            y_pred = [1, 0, 0, 0]  # Replace with actual predictions
            
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred)
            labels = ['Fake', 'Authentic']

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.title("Confusion Matrix")
            st.pyplot(fig)
        else:
            st.warning("No predictions available yet. Please make a prediction first.")

def file_upload_page():
    st.title("📁 Bulk Prediction via CSV Upload")
    uploaded_file = st.file_uploader("Upload a CSV File", type=["csv"])
    if uploaded_file:
        try:
            st.write("Processing uploaded file...")
            with st.spinner("Processing..."):
                data = pd.read_csv(uploaded_file)
                st.dataframe(data.head())
                predictions = classifier.predict(data)
                data['Prediction'] = predictions
                st.markdown("### Predictions:")
                st.dataframe(data)
                st.download_button("Download Predictions", data.to_csv(index=False), "predictions.csv", "text/csv")
                st.success("✅ Bulk predictions completed!")
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
    
    st.markdown("## Download Model")
    st.download_button(
        label="Download Model (Pickle File)",
        data=pickle_in.read(),
        file_name="classifier.pkl",
        mime="application/octet-stream"
    )

    st.markdown("### Data Visualization")
    if st.checkbox("Show Random Feature Distribution"):
        # Placeholder feature importance visualization
        sample_data = np.random.rand(100, 4)
        columns = ['Variance', 'Skewness', 'Curtosis', 'Entropy']
        df = pd.DataFrame(sample_data, columns=columns)
        fig, ax = plt.subplots()
        sns.boxplot(data=df)
        plt.title("Feature Distribution")
        st.pyplot(fig)

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

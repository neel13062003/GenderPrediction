import streamlit as st
import os
import pandas as pd
from tensorflow.keras.models import load_model
from joblib import load

# Set Streamlit page configuration
st.set_page_config(page_title="Gender Prediction", page_icon="🧑‍🎓", layout="centered")

# Load the pre-trained model
@st.cache_resource
def load_prediction_model():
    return load_model('gender_prediction_model.h5')

# Load the TF-IDF vectorizer
@st.cache_resource
def load_vectorizer():
    tfidf_vectorizer_file = 'tfidf_vectorizer.joblib'
    if not os.path.exists(tfidf_vectorizer_file):
        st.error(f"❌ {tfidf_vectorizer_file} not found. Please ensure the file exists in the current directory.")
        st.stop()
    return load(tfidf_vectorizer_file)

# Prediction function
def predict_gender(name, model, tfidf):
    vectorized_name = tfidf.transform([name]).toarray()  # Transform name into feature vector
    gender = model.predict(vectorized_name) > 0.5  # Get prediction
    return 'Male' if gender[0][0] == 1 else 'Female'

# Load model and vectorizer
model = load_prediction_model()
tfidf = load_vectorizer()

# Streamlit UI
st.title("Gender Prediction from Name")
st.write("Enter a name to predict the gender using the pre-trained model.")

# Input form
name = st.text_input("Enter a name:")
if st.button("Predict"):
    if name:
        predicted_gender = predict_gender(name, model, tfidf)
        st.success(f"The predicted gender for '{name}' is: **{predicted_gender}**")
    else:
        st.warning("Please enter a valid name.")

import pandas as pd
import os
from tensorflow.keras.models import load_model
from joblib import load

# Function to predict gender based on a name
def predict_gender(name, model, tfidf):
    vectorized_name = tfidf.transform([name]).toarray()  # Transform name into feature vector
    gender = model.predict(vectorized_name) > 0.5  # Get prediction
    return 'Male' if gender[0][0] == 1 else 'Female'

# Load the pre-trained model
model = load_model('gender_prediction_model.h5')

# Check if the TF-IDF vectorizer file exists
tfidf_vectorizer_file = 'tfidf_vectorizer.joblib'
if not os.path.exists(tfidf_vectorizer_file):
    raise FileNotFoundError(f"{tfidf_vectorizer_file} not found. Please ensure the file exists in the current directory.")

# Load the TF-IDF vectorizer
tfidf = load(tfidf_vectorizer_file)

# Main loop to take user input for predictions
while True:
    name = input("Enter a name to predict gender (or type 'exit' to quit): ")
    if name.lower() == 'exit':
        break
    predicted_gender = predict_gender(name, model, tfidf)
    print(f"The predicted gender for '{name}' is: {predicted_gender}")

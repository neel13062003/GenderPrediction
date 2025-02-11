import pandas as pd
import os
from tensorflow.keras.models import load_model
from joblib import load
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Function to predict gender based on a name
def predict_gender(name, model, tfidf):
    vectorized_name = tfidf.transform([name]).toarray()  # Transform name into feature vector
    prediction = model.predict(vectorized_name)  # Predict the probability
    return 1 if prediction[0][0] > 0.5 else 0  # 1 = Male, 0 = Female

# Load the pre-trained model
model = load_model('gender_prediction_model_onLarge.h5')

# Load the TF-IDF vectorizer
tfidf_vectorizer_file = 'tfidf_vectorizer_onLarge.joblib'
if not os.path.exists(tfidf_vectorizer_file):
    raise FileNotFoundError(f"{tfidf_vectorizer_file} not found.")
tfidf = load(tfidf_vectorizer_file)

# Load your validation dataset (replace 'validation_data.csv' with your file)
validation_data = pd.read_csv('FinalGender.csv')  # Ensure two columns: 'name', 'Gender'

# Predict genders for the validation dataset
validation_data['predicted_gender'] = validation_data['Name'].apply(
    lambda x: predict_gender(x, model, tfidf)
)

# Calculate accuracy
accuracy = accuracy_score(validation_data['Gender'], validation_data['predicted_gender'])
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Display a classification report
print("\nClassification Report:")
print(classification_report(validation_data['Gender'], validation_data['predicted_gender'], target_names=['Female', 'Male']))

# Display a confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(validation_data['Gender'], validation_data['predicted_gender']))

# Optional: Main loop to take user input for predictions
while True:
    name = input("Enter a name to predict gender (or type 'exit' to quit): ")
    if name.lower() == 'exit':
        break
    predicted_gender = 'Male' if predict_gender(name, model, tfidf) == 1 else 'Female'
    print(f"The predicted gender for '{name}' is: {predicted_gender}")

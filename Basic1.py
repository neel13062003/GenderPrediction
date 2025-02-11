import pandas as pd

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer    
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Read Data
# Ensure that the filename is correct, and the file is in the same directory as the script.
data = pd.read_excel('gender.xlsx')

# 2. Preprocess Data
# Map 'M' to 1 and 'F' to 0
data['Gender'] = data['Gender'].map({'M': 1, 'F': 0})

# 3. Convert text data into numerical data using TF-IDF
# Using character-level ngrams (1 to 3 characters)
tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
X = tfidf.fit_transform(data['Name'])  # Convert names into numerical features
y = data['Gender']  # Labels: 1 for Male, 0 for Female

# 4. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the model
# Using Logistic Regression as a classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 7. Predict the gender based on a new name
def predict_gender(name):
    vectorized_name = tfidf.transform([name])
    gender = model.predict(vectorized_name)[0]
    return 'Male' if gender == 1 else 'Female'

# Test Predictions
print(predict_gender("Ayushi"))   # Expected output: 'Male'
print(predict_gender("Mansi"))  # Expected output: 'Female'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer    
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load the dataset
data = pd.read_excel('gender.xlsx')

# Step 2: Preprocess Data
data['Gender'] = data['Gender'].map({'M': 1, 'F': 0})

# Visualize the Gender distribution in the dataset
plt.figure(figsize=(6, 4))
sns.countplot(data['Gender'].map({1: 'Male', 0: 'Female'}))
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Step 3: Convert the names to numerical format using TF-IDF
tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
X = tfidf.fit_transform(data['Name'])
y = data['Gender']

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Visualize confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Female', 'Male'], yticklabels=['Female', 'Male'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Visualizing the Train/Test Split
plt.figure(figsize=(6, 6))
plt.pie([len(X_train), len(X_test)], labels=['Train Set', 'Test Set'], autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#99ff99'])
plt.title('Train-Test Split')
plt.show()

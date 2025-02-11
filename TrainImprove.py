import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from joblib import dump

# 1. Read Data
data = pd.read_excel('gender.xlsx')

# 2. Preprocess Data
data['Gender'] = data['Gender'].map({'M': 1, 'F': 0})

# 3. Convert text data into numerical data using TF-IDF
tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
X = tfidf.fit_transform(data['Name']).toarray()  # Convert names into numerical features
y = data['Gender'].values  # Labels: 1 for Male, 0 for Female

# 4. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 5. Build the Neural Network Model
model = Sequential()
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01), input_shape=(X_train.shape[1],)))  # L2 regularization
model.add(BatchNormalization())  # Batch normalization
model.add(Dropout(0.5))  # Dropout to prevent overfitting
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))  # L2 regularization
model.add(BatchNormalization())  # Batch normalization
model.add(Dropout(0.5))  # Dropout to prevent overfitting
model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid for binary classification

# 6. Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 7. Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # Early stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)  # Learning rate reduction

# 8. Train the model with epochs and callbacks
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, 
          callbacks=[early_stopping, reduce_lr])

# 9. Save the model after training
model.save('gender_prediction_model_Improve.h5')

# 10. Save the TF-IDF vectorizer
dump(tfidf, 'tfidf_vectorizer_Improve.joblib')

# 11. Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype("int32")  # Convert probabilities to binary output
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

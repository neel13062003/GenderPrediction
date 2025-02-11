<html>

# Gender Prediction from Names using Neural Network

### 🚀 **Project Overview**

This project uses a **Neural Network** model trained on **TF-IDF vectors** of names to predict the gender (Male/Female). The model is deployed using **Streamlit**, making it easy to interact and predict the gender from a user-inputted name.

---

## 📸 **Application Screenshot**

## <a href="https://ibb.co/JjhspDnr"><img src="https://ibb.co/JjhspDnr" alt="Screenshot-2025-02-11-222451" border="0" /></a>

## 🛠 **How It Works (End-to-End)**

### 1. **Data Preparation**

- The dataset `gender.xlsx` contains names and their corresponding genders (Male/Female).
- The `Gender` column is mapped to numerical values:
  - **Male (M)** is mapped to `1`
  - **Female (F)** is mapped to `0`

### 2. **Feature Extraction (TF-IDF Vectorization)**

- The names are converted to **TF-IDF vectors** using character n-grams (1 to 3 characters).
- This helps the model learn important patterns in names.

### 3. **Model Training**

- A **Neural Network** is built using **Keras Sequential API**:
  - Dense layers with **ReLU activation**
  - **Batch Normalization** and **Dropout layers** to prevent overfitting
  - Output layer with **Sigmoid activation** for binary classification
- The model is trained with **callbacks** like early stopping and learning rate reduction.

### 4. **Saving the Model and Vectorizer**

- The trained model is saved as `gender_prediction_model_Improve.h5`
- The TF-IDF vectorizer is saved as `tfidf_vectorizer_Improve.joblib`

### 5. **Streamlit Application**

- Loads the pre-trained model and vectorizer.
- Accepts user input (name) and predicts gender.
- Displays the predicted gender in a clean UI.

---

## 📝 **Project File Structure**

```
.
├── TrainImprove.py          # Training script for the model
├── ml-st1.py                 # Streamlit app for gender prediction
├── gender.xlsx              # Dataset with names and gender
├── gender_prediction_model_Improve.h5  # Saved Keras model
├── tfidf_vectorizer_Improve.joblib     # Saved TF-IDF vectorizer
└── screenshot.png           # Screenshot of the app UI
```

---

## 🚀 **How to Run the Project**

### 1. **Clone the Repository**

```bash
$ git clone <repository-url>
$ cd <repository-folder>
```

### 2. **Install Dependencies**

```bash
$ pip install -r requirements.txt
```

### 3. **Train the Model (Optional)**

If you want to retrain the model, run the training script:

```bash
$ python TrainImprove.py
```

### 4. **Run the Streamlit Application**

```bash
$ streamlit run final.py
```

### 5. **Access the App**

Open your browser and go to: [http://localhost:8501](http://localhost:8501)

---

## 💡 **How the Code Works**

### **Training (TrainImprove.py)**

1. **Data Loading:** Reads the dataset from `gender.xlsx`.
2. **Preprocessing:** Converts names to TF-IDF vectors.
3. **Model Building:** Defines a neural network with regularization.
4. **Model Training:** Trains the model with early stopping.
5. **Saving Artifacts:** Stores the trained model (`.h5`) and vectorizer (`.joblib`).

### **Application (final.py)**

1. **Load Model and Vectorizer:** Loads the pre-trained model and TF-IDF vectorizer.
2. **User Input:** Accepts a name input from the user.
3. **Prediction:** Transforms the name using TF-IDF and makes a prediction.
4. **Output:** Displays the predicted gender (Male/Female) in the app.

---

## 📦 **Dependencies**

- Python 3.x
- TensorFlow
- Scikit-learn
- Pandas
- Streamlit

Install them using:

```bash
$ pip install tensorflow scikit-learn pandas streamlit joblib
```

---

## 🎨 **Future Enhancements**

- Improve the UI design.
- Include more diverse datasets for better generalization.
- Add confidence scores for predictions.
- Deploy the app online for public access.

---

## 🤝 **Contributing**

Feel free to fork the project and submit a pull request for improvements.

---

## 📜 **License**

This project is licensed under the MIT License.

</html>

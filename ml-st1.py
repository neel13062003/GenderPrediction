# import streamlit as st
# import os
# import pandas as pd
# from tensorflow.keras.models import load_model
# from joblib import load

# # Set Streamlit page configuration
# st.set_page_config(page_title="Gender Prediction", page_icon="🧑‍🎓", layout="centered")

# # Load the pre-trained model
# @st.cache_resource
# def load_prediction_model():
#     return load_model('gender_prediction_model.h5')

# # Load the TF-IDF vectorizer
# @st.cache_resource
# def load_vectorizer():
#     tfidf_vectorizer_file = 'tfidf_vectorizer.joblib'
#     if not os.path.exists(tfidf_vectorizer_file):
#         st.error(f"❌ {tfidf_vectorizer_file} not found. Please ensure the file exists in the current directory.")
#         st.stop()
#     return load(tfidf_vectorizer_file)

# # Prediction function
# def predict_gender(name, model, tfidf):
#     vectorized_name = tfidf.transform([name]).toarray()  # Transform name into feature vector
#     gender = model.predict(vectorized_name) > 0.5  # Get prediction
#     return 'Male' if gender[0][0] == 1 else 'Female'

# # Load model and vectorizer
# model = load_prediction_model()
# tfidf = load_vectorizer()

# # Streamlit UI
# st.title("Gender Prediction from Name")
# st.write("Enter a name to predict the gender using the pre-trained model.")

# # Input form
# name = st.text_input("Enter a name:")
# if st.button("Predict"):
#     if name:
#         predicted_gender = predict_gender(name, model, tfidf)
#         st.success(f"The predicted gender for '{name}' is: **{predicted_gender}**")
#     else:
#         st.warning("Please enter a valid name.")


import streamlit as st
import os
import pandas as pd
from tensorflow.keras.models import load_model
from joblib import load

# Set Streamlit page configuration
st.set_page_config(
    page_title="Neural Name Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Futuristic Custom CSS styling
st.markdown(
    """
    <style>
        /* Cyberpunk-inspired background */
        .main {
            background-color: #0a0a0a;
            background-image: 
                linear-gradient(45deg, #000 25%, transparent 25%),
                linear-gradient(-45deg, #000 25%, transparent 25%),
                linear-gradient(45deg, transparent 75%, #000 75%),
                linear-gradient(-45deg, transparent 75%, #000 75%);
            background-size: 20px 20px;
            background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
        }

        /* Neon title effect */
        h1 {
            color: #0ff;
            text-align: center;
            font-family: 'Courier New', monospace;
            font-size: 3.5rem !important;
            text-transform: uppercase;
            letter-spacing: 0.2em;
            position: relative;
            text-shadow: 0 0 10px #0ff,
                         0 0 20px #0ff,
                         0 0 40px #0ff;
            animation: flicker 3s infinite;
        }

        @keyframes flicker {
            0%, 19.999%, 22%, 62.999%, 64%, 64.999%, 70%, 100% {
                opacity: 1;
            }
            20%, 21.999%, 63%, 63.999%, 65%, 69.999% {
                opacity: 0.4;
            }
        }

        /* Neural interface container */
        .neural-interface {
            background: #111;
            border: 2px solid #0ff;
            border-radius: 5px;
            padding: 2rem;
            position: relative;
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.2);
        }

        .neural-interface::before {
            content: '';
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            background: linear-gradient(45deg, #0ff, #00f, #f0f);
            z-index: -1;
            filter: blur(10px);
            animation: borderGlow 3s linear infinite;
        }

        @keyframes borderGlow {
            0% { filter: blur(10px) hue-rotate(0deg); }
            100% { filter: blur(10px) hue-rotate(360deg); }
        }

        /* Futuristic input field */
        .stTextInput>div>div>input {
            background: #000;
            border: 1px solid #0ff;
            border-radius: 3px;
            color: #0ff;
            padding: 1rem;
            font-family: 'Courier New', monospace;
            font-size: 18px;
            transition: all 0.3s ease;
        }

        .stTextInput>div>div>input:focus {
            border-color: #f0f;
            box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);
        }

        /* Holographic button */
        .stButton>button {
            background: transparent;
            border: 2px solid #0ff;
            color: #0ff;
            padding: 1rem 2rem;
            font-family: 'Courier New', monospace;
            font-size: 18px;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 2px;
        }

        .stButton>button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                120deg,
                transparent,
                rgba(0, 255, 255, 0.2),
                transparent
            );
            animation: shine 2s infinite;
        }

        @keyframes shine {
            100% { left: 100%; }
        }

        /* Digital display for results */
        .digital-display {
            background: #000;
            border: 2px solid #0ff;
            padding: 2rem;
            margin-top: 2rem;
            position: relative;
            font-family: 'Courier New', monospace;
            animation: powerOn 0.5s ease-out;
        }

        @keyframes powerOn {
            0% { opacity: 0; transform: scale(0.9); }
            50% { opacity: 0.5; transform: scale(1.02); }
            100% { opacity: 1; transform: scale(1); }
        }

        /* Loading animation */
        .stSpinner > div {
            border-color: #0ff !important;
            border-right-color: transparent !important;
        }

        /* Success/warning messages */
        .element-container .stSuccess, .element-container .stWarning {
            background: #000;
            border: 1px solid #0ff;
            color: #0ff;
            font-family: 'Courier New', monospace;
        }

        /* Scan lines effect */
        .scan-lines {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                to bottom,
                transparent 50%,
                rgba(0, 255, 255, 0.02) 50%
            );
            background-size: 100% 4px;
            pointer-events: none;
            z-index: 9999;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Add scan lines effect
st.markdown('<div class="scan-lines"></div>', unsafe_allow_html=True)

# Load the pre-trained model
@st.cache_resource
def load_prediction_model():
    return load_model('gender_prediction_model.h5')

# Load the TF-IDF vectorizer
@st.cache_resource
def load_vectorizer():
    tfidf_vectorizer_file = 'tfidf_vectorizer.joblib'
    if not os.path.exists(tfidf_vectorizer_file):
        st.error("SYSTEM ERROR: Neural network components not found")
        st.stop()
    return load(tfidf_vectorizer_file)

# Prediction function
def predict_gender(name, model, tfidf):
    vectorized_name = tfidf.transform([name]).toarray()
    gender = model.predict(vectorized_name) > 0.5
    return 'Male' if gender[0][0] == 1 else 'Female'

# Load model and vectorizer
model = load_prediction_model()
tfidf = load_vectorizer()

# Main content
st.title("NEURAL NAME ANALYZER")

st.markdown(
    """
    <div style='text-align: center; color: #0ff; font-family: "Courier New", monospace; margin: 2rem 0;'>
        [ INITIATING NEURAL PATTERN RECOGNITION SEQUENCE ]
    </div>
    """,
    unsafe_allow_html=True
)

# Create three columns for layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    
    
    # Input field
    name = st.text_input(
        "",
        placeholder="ENTER NAME FOR ANALYSIS >>",
        help="Input name sequence for gender pattern analysis"
    )
    
    # Terminal-style divider
    st.markdown(
        """
        <div style='text-align: center; margin: 2rem 0; color: #0ff; font-family: "Courier New", monospace;'>
            >>> READY FOR ANALYSIS <<<
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Analysis button
    if st.button("INITIATE ANALYSIS"):
        if name.strip():
            with st.spinner("PROCESSING NEURAL PATTERNS..."):
                predicted_gender = predict_gender(name, model, tfidf)
                
                # Display result with digital effect
                st.markdown(
                    f"""
                    <div class='digital-display'>
                        <div style='color: #0ff; text-align: center;'>
                            <div style='font-size: 1.2rem; margin-bottom: 1rem;'>
                                >> ANALYSIS COMPLETE <<
                            </div>
                            <div style='font-size: 2rem; margin: 1rem 0; color: #f0f;'>
                                {name.upper()}
                            </div>
                            <div style='font-size: 1.5rem; margin: 1rem 0;'>
                                GENDER PATTERN: {predicted_gender.upper()}
                            </div>
                            <div style='font-size: 0.8rem; opacity: 0.7; margin-top: 1rem;'>
                                CONFIDENCE LEVEL: HIGH
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.warning("ERROR: Name input required for analysis")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <div style='text-align: center; color: #0ff; font-family: "Courier New", monospace; 
         font-size: 0.8rem; padding: 2rem; opacity: 0.7;'>
        SYSTEM NOTE: Analysis based on deep learning neural patterns.
        Results represent statistical probability across known data patterns.
    </div>
    """,
    unsafe_allow_html=True
)
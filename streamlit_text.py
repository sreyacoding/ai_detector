

import streamlit as st
import pickle
import librosa
import numpy as np
import xgboost as xgb

# Load the SVM model and TF-IDF vectorizer
@st.cache_resource
def load_text_model_and_vectorizer():
    """Load the SVM model and TF-IDF vectorizer for text classification."""
    with open('svm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# Load the pre-trained XGBoost model for audio classification
@st.cache_resource
def load_audio_model(model_path):
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model

# Feature extraction functions for audio
def extract_mfcc(y):
    return np.mean(librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13).T, axis=0)

def extract_delta(y):
    return np.mean(librosa.feature.delta(librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13)).T, axis=0)

def extract_mel_spectrogram(y):
    return np.mean(librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=16000), ref=np.max).T, axis=0)

def extract_zero_crossing_rate(y):
    return np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)

def extract_chroma(y):
    return np.mean(librosa.feature.chroma_stft(y=y, sr=16000).T, axis=0)

def extract_cqt(y):
    return np.mean(np.abs(librosa.cqt(y, sr=16000)).T, axis=0)

# Preprocess audio and extract features
def preprocess_and_extract_features(file):
    y, sr = librosa.load(file, sr=16000)
    y, _ = librosa.effects.trim(y)

    # Pad or truncate to 5 seconds
    max_len = sr * 5
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)), mode='constant')
    else:
        y = y[:max_len]

    # Extract features
    mfcc = extract_mfcc(y)
    delta_mfcc = extract_delta(y)
    mel_spectrogram = extract_mel_spectrogram(y)
    zcr = np.array([extract_zero_crossing_rate(y)])
    chroma = extract_chroma(y)
    cqt = extract_cqt(y)

    return np.concatenate([mfcc.flatten(), delta_mfcc.flatten(), mel_spectrogram.flatten(), zcr.flatten(), chroma.flatten(), cqt.flatten()])

# Predict audio class using the XGBoost model
def predict_audio_class(file, model):
    features = preprocess_and_extract_features(file)
    features = np.array([features])

    prediction = model.predict(features)
    probability = model.predict_proba(features)

    return prediction, probability

# Streamlit interface
def main():
    st.title("AI vs Human Text and Audio Classifier")

    # Tabs for Text and Audio Classification
    tab1, tab2 = st.tabs(["Text Classification", "Audio Classification"])

    # Text Classification Tab
    with tab1:
        st.subheader("Classify Text as AI-Generated or Human-Written")
        model, vectorizer = load_text_model_and_vectorizer()
        user_input = st.text_area("Text Input", height=150)

        if st.button("Classify Text"):
            if not user_input:
                st.warning("Please enter some text to classify.")
            else:
                with st.spinner('Classifying...'):
                    vectorized_input = vectorizer.transform([user_input])
                    prediction = model.predict(vectorized_input)

                    if prediction[0] == 1:
                        st.success("The text is AI-Generated!")
                    else:
                        st.success("The text is Human-Written!")

    # Audio Classification Tab
    with tab2:
        st.subheader("Classify Audio as AI-Generated or Human")
        audio_file = st.file_uploader("Upload an audio file (MP3 format)", type=["mp3"])
        model_path = r"C:/Users/SANJU/Downloads/xgb_audio_model (1).json"
        audio_model = load_audio_model(model_path)

        if audio_file is not None:
            st.audio(audio_file, format='audio/mp3')

            if st.button("Classify Audio"):
                with st.spinner('Classifying...'):
                    with open("temp_audio_file.mp3", "wb") as f:
                        f.write(audio_file.getbuffer())

                    predicted_class, predicted_proba = predict_audio_class("temp_audio_file.mp3", audio_model)

                    st.write("Classes: 0 for Real Audio, 1 for AI Generated Audio")
                    st.success(f"Predicted class: {predicted_class[0]}")

                    formatted_proba = ',\t'.join([f"{round(prob, 5):.5f}" for prob in predicted_proba[0]])
                    st.write(f"Predicted probabilities: [  {formatted_proba}  ]")

                    if predicted_class[0] == 0:
                        st.markdown("<h3><b>The audio is REAL!</b></h3>", unsafe_allow_html=True)
                    else:
                        st.markdown("<h3><b>The audio is AI-GENERATED!</b></h3>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

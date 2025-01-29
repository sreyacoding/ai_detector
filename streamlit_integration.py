import os
import streamlit as st
import pickle
import librosa
import numpy as np
import xgboost as xgb
import torch
import io
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

#######################################################################################################################################
##################################
###############TEXT###############
##################################
#######################################################################################################################################

# Suppress warnings and download NLTK data
warnings.filterwarnings("ignore")
nltk.download('stopwords')
nltk.download('wordnet')

# Load the SVM model and TF-IDF vectorizer for text classification
@st.cache_resource
def load_text_model_and_vectorizer(svm_model_path, tfidf_vectorizer_path):
    with open(svm_model_path, 'rb') as model_file:
        svm_model = pickle.load(model_file)
    with open(tfidf_vectorizer_path, 'rb') as vectorizer_file:
        tfidf_vectorizer = pickle.load(vectorizer_file)
    return svm_model, tfidf_vectorizer

# Text cleaning function
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stopwords.words('english')])
    return text

# Prediction function for text
def predict_text(text, model, vectorizer):
    text_cleaned = clean_text(text)
    text_tfidf = vectorizer.transform([text_cleaned])
    probabilities = model.predict_proba(text_tfidf)[0]  # Predicted probabilities
    prediction = model.predict(text_tfidf)[0]  # Predicted class
    return {
        "Predicted Class": 'LLM-Generated' if prediction == 1 else 'Human-Written',
        "Probability of LLM-Generated": probabilities[1],
        "Probability of Human-Written": probabilities[0],
    }   



#######################################################################################################################################
###################################
###############AUDIO###############
###################################
#######################################################################################################################################

# Load the XGBoost model for audio classification
@st.cache_resource
def load_audio_model(model_path):
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model

# Feature extraction functions
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
    try:
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

    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return np.zeros(251)  # Ensure the feature vector size matches the expected size

# Predict audio class using the XGBoost model
def predict_audio_class(file, model):
    features = preprocess_and_extract_features(file)
    features = np.array([features])  # Reshape for prediction

    prediction = model.predict(features)
    probability = model.predict_proba(features)

    return prediction, probability



#######################################################################################################################################
###################################
###############IMAGE###############
###################################
#######################################################################################################################################

# Define a custom unpickler class for CPU mapping
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=torch.device('cpu'))
        else:
            return super().find_class(module, name)

# Custom model loader using the CPU_Unpickler
@st.cache_resource
def load_image_model(model_path):
    with open(model_path, 'rb') as f:
        model = CPU_Unpickler(f).load()
    model.eval()  # Set model to evaluation mode
    return model

# Example preprocessing and prediction pipeline
def preprocess_image(img_path):
    from torchvision import transforms
    from PIL import Image

    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    img = transform(img)
    return img.unsqueeze(0)  # Add batch dimension



#######################################################################################################################################
################################
###############UI###############
################################
#######################################################################################################################################

# Streamlit App Layout
def main():
    st.title("AI vs Human Classifier")

    # Tabs for Text, Audio, and Image Classification
    tab1, tab2, tab3 = st.tabs(["Image Classification", "Audio Classification", "Text Classification"])


    # TEXT Classification Tab
    with tab3:
        st.subheader("Classify Text as AI-Generated or Human-Written")
        svm_model, tfidf_vectorizer = load_text_model_and_vectorizer("svm_model.pkl","tfidf_vectorizer.pkl")
        user_input = st.text_area("Enter text:(Minimum word count is 500)", "")

        if st.button('Classify Text'):
            if user_input:
                prediction = predict_text(user_input, svm_model, tfidf_vectorizer)
                st.success(f"**Predicted Class:** {prediction['Predicted Class']}")
                st.write(f"**Probability of LLM-Generated:** {prediction['Probability of LLM-Generated']:.2f}")
                st.write(f"**Probability of Human-Written:** {prediction['Probability of Human-Written']:.2f}")
            else:
                st.write("Please enter text to classify.")


    # AUDIO Classification Tab
    with tab2:
        st.subheader("Classify Audio as AI-Generated or Human")
        audio_file = st.file_uploader("Upload an audio file between 5 seconds to 1 minute", type=["mp3"])
        audio_model = load_audio_model("xgb_audio_model.json")

        if audio_file is not None:
            st.audio(audio_file, format='audio/mp3')

            if st.button("Classify Audio"):
                with st.spinner('Classifying...'):
                    with open("temp_audio_file.mp3", "wb") as f:
                        f.write(audio_file.getbuffer())

                    predicted_class, predicted_proba = predict_audio_class("temp_audio_file.mp3", audio_model)

                    if predicted_class[0]=='0':
                        st.success(f"Predicted Class: Human Voice")
                    else:
                        st.success(f"Predicted Class: AI-Generated")                    
                    formatted_proba = ',\t'.join([f"{round(prob, 5):.5f}" for prob in predicted_proba[0]])
                    st.write(f"Predicted probabilities: [  {formatted_proba}  ]")                    
                    os.remove("temp_audio_file.mp3")
    # IMAGE Classification Tab
    with tab1:
        st.subheader("Classify Image as AI-Generated or Real")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            if st.button("Classify"):
                with st.spinner("Loading model..."):
                    model = load_image_model("resnet18_cifake.pkl")

                with st.spinner("Processing image..."):
                    # Save the uploaded image temporarily
                    temp_path = "temp_image.jpg"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Preprocess and predict
                    img_tensor = preprocess_image(temp_path)
                    output = model(img_tensor)  # Model output tensor
                    probabilities = torch.softmax(output, dim=1)  # Get probabilities

                    # Get the predicted class
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0, predicted_class].item()

                    # Display the result
                    if predicted_class == 1:  # Assuming class 1 = Fake
                        st.success(f"Predicted Class: AI-Generated")
                        st.write(f"Predicted probabilities: [  {1-confidence:.2f},{confidence:.2f}  ]")
                    else:  # Class 0 = Real
                        st.success(f"Predicted Class: Real")
                        st.write(f"Predicted probabilities: [  {confidence:.2f},{1-confidence:.2f}  ]")
                    os.remove("temp_image.jpg")
if __name__ == "__main__":
    main()
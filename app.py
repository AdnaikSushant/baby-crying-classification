import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model_path = 'saved_models/audio_classification.keras'
model = tf.keras.models.load_model(model_path)

# Load the label encoder
def load_label_encoder(file_path):
    with open(file_path, 'rb') as file:
        classes = pickle.load(file)
    labelencoder = LabelEncoder()
    labelencoder.classes_ = classes
    return labelencoder

labelencoder = load_label_encoder('label_classes.pkl')

def preprocess_audio(file):
    y, sr = librosa.load(file, sr=None)
    mfccs_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features.reshape(1, -1)

def predict_audio_class(file, model, labelencoder):
    features = preprocess_audio(file)
    predicted_probabilities = model.predict(features)
    predicted_label = np.argmax(predicted_probabilities, axis=1)
    prediction_class = labelencoder.inverse_transform(predicted_label)
    return prediction_class[0]

# Streamlit UI
st.title('Audio File Classification')
st.write('Upload an audio file to get its predicted class.')

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    with open('temp_audio_file.wav', 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Predict and display the class
    predicted_class = predict_audio_class('temp_audio_file.wav', model, labelencoder)
    st.write(f'Predicted Class: {predicted_class}')

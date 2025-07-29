# --- IMPORTS ---
import os
import numpy as np
import cv2
import tensorflow as tf
import pickle
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# --- INITIALIZE FLASK APP ---
app = Flask(__name__)

# --- LOAD MODELS AND PREPROCESSORS ---
# Updated to load the correct model files

# Load OCR Model and its Label Encoder
try:
    ocr_model = tf.keras.models.load_model(r'c:\Users\ethan\Downloads\ocr_character_recognition_model.h5')
    with open(r'c:\Users\ethan\Downloads\ocr_label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    # Create the character map from the loaded label encoder
    char_map = {i: char for i, char in enumerate(label_encoder.classes_)}
except Exception as e:
    print(f"Error loading OCR model or label encoder: {e}")
    ocr_model = None

# Load Sentiment Model (trained on Stanford dataset) and its Tokenizer
try:
    sentiment_model = tf.keras.models.load_model(r'C:\Users\ethan\Downloads\sentiment_model_stanford.h5')
    with open(r'C:\Users\ethan\Downloads\tokenizer_stanford.pickle', 'rb') as f:
        tokenizer = pickle.load(f)
except Exception as e:
    print(f"Error loading Sentiment model or tokenizer: {e}")
    sentiment_model = None

# --- TEXT CLEANING FUNCTION (from your training script) ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"@[a-z0-9_]+", "", text)
    text = re.sub(r"https?://[a-z0-9./]+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- PREDICTION FUNCTION ---
def predict_text_and_sentiment_from_image(image_data):
    if ocr_model is None or sentiment_model is None:
        return "A required model is not loaded.", "Error"

    # OCR Part
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "Could not decode image.", "N/A"

    char_width, char_height = 28, 28
    h, w = img.shape
    characters_for_model = []
    space_placeholders = []

    for y in range(0, h, char_height):
        for x in range(0, w, char_width):
            char_slice = img[y:y + char_height, x:x + char_width]
            if char_slice.shape == (char_height, char_width):
                if np.mean(char_slice) < 5:
                    space_placeholders.append(' ')
                else:
                    char_slice = char_slice / 255.0
                    char_slice = char_slice.reshape((1, char_height, char_width, 1))
                    characters_for_model.append(char_slice)
                    space_placeholders.append('X')

    if not characters_for_model:
        return "No characters detected.", "N/A"

    # Predict all characters at once for efficiency
    all_chars_array = np.vstack(characters_for_model)
    predictions = ocr_model.predict(all_chars_array, verbose=0)
    predicted_indices = np.argmax(predictions, axis=1)
    predicted_chars_list = label_encoder.inverse_transform(predicted_indices)
    
    predicted_sentence = ""
    char_index = 0
    for placeholder in space_placeholders:
        if placeholder == ' ':
            predicted_sentence += ' '
        else:
            if char_index < len(predicted_chars_list):
                predicted_sentence += predicted_chars_list[char_index]
                char_index += 1
    predicted_sentence = predicted_sentence.strip()

    # Sentiment Analysis Part
    cleaned_sentence = clean_text(predicted_sentence)
    test_sequence = tokenizer.texts_to_sequences([cleaned_sentence])
    padded_test_sequence = pad_sequences(test_sequence, maxlen=200)
    
    # The model prediction will be a value between 0 and 1
    prediction_score = sentiment_model.predict(padded_test_sequence, verbose=0)[0][0]
    
    # If the score is > 0.5, it's positive, otherwise negative
    predicted_sentiment = "Positive" if prediction_score > 0.5 else "Negative"

    return predicted_sentence, predicted_sentiment


# --- DEFINE API ENDPOINTS ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        image_data = file.read()
        sentence, sentiment = predict_text_and_sentiment_from_image(image_data)
        return jsonify({'sentence': sentence, 'sentiment': sentiment})

# --- RUN THE APP ---
if __name__ == '__main__':
    app.run(debug=True)

# Sentiment Analysis of Handwritten Text Using Custom-built OCR and Sentiment Classification Models

This project analyzes handwritten text from an image by first recognizing the characters using a custom-built Optical Character Recognition (OCR) model and then determining the emotional sentiment of the recognized text using a sentiment classification model. The entire pipeline is deployed as a web application using Flask.

## Features

* **Handwriting Recognition (OCR):** A Convolutional Neural Network (CNN) trained to identify individual handwritten characters from an image.

* **Sentiment Analysis:** A Recurrent Neural Network (RNN) with Bidirectional LSTM layers that classifies the recognized text into 'Angry', 'Happy', or 'Neutral' categories.

* **Web Interface:** A clean, user-friendly front-end built with HTML and Tailwind CSS that allows users to upload an image and view the results.

* **Python Back-End:** A Flask server that hosts the trained models and handles the image processing and prediction logic.

## Methodology

The project follows a two-stage pipeline: OCR followed by Sentiment Analysis.

1. **OCR Data Loading and Preprocessing:**

   * The custom dataset is loaded, consisting of character images (`alphabet_images`) and their corresponding labels (`alphabet_labels.csv`).

   * Images are converted to grayscale, normalized to pixel values between 0 and 1, and labels are numerically encoded.

2. **OCR Model Training:**

   * A Convolutional Neural Network (CNN) is built and compiled using TensorFlow/Keras.

   * The model is trained on the preprocessed image data to learn to classify characters.

3. **Character Segmentation and Prediction:**

   * When a new image is uploaded, it is preprocessed and sliced into 28x28 pixel segments.

   * The trained OCR model predicts the character contained in each segment.

   * The individual character predictions are stitched back together, including spaces, to reconstruct the full sentence.

4. **Sentiment Data and Model Training:**

   * A separate dataset of sentences and their sentiment labels ('Angry', 'Happy', 'Neutral') is loaded.

   * The text is tokenized and converted into padded sequences.

   * A Bidirectional LSTM model is built and trained on this data to understand the emotional context of sentences.

5. **Final Sentiment Prediction:**

   * The sentence reconstructed by the OCR model is fed into the trained sentiment analysis model.

   * The final sentiment is predicted and displayed to the user.

## Technology Stack

* **Back-End:** Python, Flask

* **Machine Learning:** TensorFlow, Keras, Scikit-learn

* **Image Processing:** OpenCV, NumPy

* **Front-End:** HTML, Tailwind CSS, JavaScript

## Project Structure

To run this application, your project folder should be organized as follows:

```
ocr-webapp/
├── templates/
│   └── index.html          # Front-end HTML file
├── app.py                  # Main Flask application
├── my_ocr_model.h5         # Trained OCR model
├── label_encoder.pickle    # Saved OCR label encoder
├── sentiment_model.h5      # Trained sentiment model
└── tokenizer.pickle        # Saved sentiment tokenizer

```



## References

* [Sentiment Analysis with an Recurrent Neural Networks (RNN) - GeeksforGeeks](https://www.geeksforgeeks.org/sentiment-analysis-with-an-recurrent-neural-networks-rnn/)

* [Sentiment Analysis in Python using TensorFlow 2 | Text Classification | NLP - YouTube](https://www.youtube.com/watch?v=JgnbwKnHMZQ)

* [Sentiment Analysis | Natural Language Processing With Python & Tensorflow - YouTube](https://www.youtube.com/watch?v=eMPQw7Xbjd0)

* [Handwritten Text Recognition with TensorFlow - YouTube](https://www.youtube.com/watch?v=Zi4i7Q0zrBs)

* [Recognize Handwritten Digits with a CNN (Keras & TensorFlow) - YouTube](https://www.youtube.com/watch?v=Ixm_yQYK09E)

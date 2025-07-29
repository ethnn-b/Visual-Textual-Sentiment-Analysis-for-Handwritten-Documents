# Sentiment Analysis of Handwritten Text Using Custom-built OCR and Sentiment Classification Models

This project analyzes handwritten text from an image by first recognizing the characters using a custom-built Optical Character Recognition (OCR) model and then determining the emotional sentiment of the recognized text using a sentiment classification model. The entire pipeline is deployed as a web application using Flask.

<img width="597" height="750" alt="image" src="https://github.com/user-attachments/assets/9430bbf2-b592-4d9b-9390-b049c0e053f0" />

# iHandwriting OCR and Sentiment Analysis

A web application that performs Optical Character Recognition (OCR) on an uploaded image of handwritten text and then analyzes its sentiment. The project uses a custom CNN for OCR and an LSTM for sentiment analysis, all deployed with a Flask back-end.

## Features

* **Handwriting Recognition (OCR):** A Convolutional Neural Network (CNN) trained on a custom dataset to identify individual handwritten characters.

* **Sentiment Analysis:** An LSTM-based Recurrent Neural Network (RNN) trained on the Stanford Sentiment140 dataset to classify text as **Positive** or **Negative**.

* **Full-Stack Application:** A user-friendly front-end (HTML/JS/Tailwind) powered by a Python/Flask back-end that handles model inference.

## Methodology

The application employs a two-stage pipeline. When a user uploads an image, a CNN-based OCR model first recognizes the handwritten characters. The resulting text is then passed to an LSTM-based sentiment model to classify its sentiment as 'Positive' or 'Negative'. The final result is then displayed to the user.

## Technology Stack

* **Back-End:** Python, Flask

* **Machine Learning:** TensorFlow, Keras, Scikit-learn

* **Image Processing:** OpenCV, NumPy

* **Front-End:** HTML, Tailwind CSS, JavaScript

## Project Structure

Your project folder must be organized as follows to run the application:

```
ocr-webapp/
├── templates/
│   └── index.html                      # Front-end HTML file
├── app.py                              # Main Flask application
├── ocr_character_recognition_model.h5  # Trained OCR model
├── ocr_label_encoder.pkl               # Saved OCR label encoder
├── sentiment_model_stanford.h5         # Trained sentiment model
└── tokenizer_stanford.pickle           # Saved sentiment tokenizer
```








## References

* [Sentiment Analysis with an Recurrent Neural Networks (RNN) - GeeksforGeeks](https://www.geeksforgeeks.org/sentiment-analysis-with-an-recurrent-neural-networks-rnn/)

* [Sentiment Analysis in Python using TensorFlow 2 | Text Classification | NLP - YouTube](https://www.youtube.com/watch?v=JgnbwKnHMZQ)

* [Sentiment Analysis | Natural Language Processing With Python & Tensorflow - YouTube](https://www.youtube.com/watch?v=eMPQw7Xbjd0)

* [Handwritten Text Recognition with TensorFlow - YouTube](https://www.youtube.com/watch?v=Zi4i7Q0zrBs)

* [Recognize Handwritten Digits with a CNN (Keras & TensorFlow) - YouTube](https://www.youtube.com/watch?v=Ixm_yQYK09E)

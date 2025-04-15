# streamlit_sentiment_app.py

import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load saved model and vectorizer
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Initialize preprocessing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Streamlit app layout
st.title("📊 Amazon Review Sentiment Analyzer")
st.write("Enter a product review below and choose a model to classify the sentiment.")

# Input box for user review
user_input = st.text_area("✍️ Enter Review Text:", "This product is amazing! I absolutely love it.")

# Predict button
if st.button("🔍 Analyze Sentiment"):
    clean_text = preprocess(user_input)
    vectorized_text = vectorizer.transform([clean_text])
    prediction = model.predict(vectorized_text)[0]
    prediction_proba = model.predict_proba(vectorized_text)[0]

    label = "Positive 😊" if prediction == 1 else "Negative 😠"
    st.subheader(f"Predicted Sentiment: {label}")
    st.write(f"Confidence: {round(max(prediction_proba)*100, 2)}%")

    st.markdown("---")
    st.text("Raw Prediction Probabilities:")
    st.write({"Negative": round(prediction_proba[0]*100, 2), "Positive": round(prediction_proba[1]*100, 2)})

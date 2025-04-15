# Amazon Fine Food Reviews - Sentiment Analysis

# 📦 Step 1: Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

# 📥 Step 2: Load Full Dataset
file_path = r"G:\My Drive\MSc. DSA\Module V\DSA 8501 Text and Unstructured Data Analytics\Reviews.csv"
df = pd.read_csv(file_path)

# 🏷️ Step 3: Create Sentiment Labels
def label_sentiment(score):
    if score <= 2:
        return 0  # Negative
    elif score >= 4:
        return 1  # Positive
    else:
        return None

df['Sentiment'] = df['Score'].apply(label_sentiment)
df.dropna(inplace=True)

# ✂️ Step 4: Text Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['Clean_Text'] = df['Text'].apply(preprocess)

# 📊 Step 5: EDA
plt.figure(figsize=(6,4))
df['Clean_Text'].str.split().apply(len).hist(bins=50)
plt.title("Review Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()

# 🔠 Step 6: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(df['Clean_Text'])
y = df['Sentiment']

# 🔍 Step 7: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 Step 8: Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# 📈 Step 9: Evaluation
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 💾 Step 10: Save Model and Vectorizer
import joblib
joblib.dump(model, "logistic_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

import pandas as pd
import numpy as np
import re
import string
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
# Load Model & TF-IDF Vectorizer
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

stop_words = set(stopwords.words('english'))
# 📌 Function to Clean a Single Tweet
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # Remove mentions
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)  # Remove hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters & numbers
    words = word_tokenize(text)  # Tokenize
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)

# Function to Predict Sentiment
def predict_sentiment(tweet):
    # Clean the tweet
    cleaned_tweet = clean_text(tweet)
    
    # Convert to TF-IDF features
    tweet_vectorized = tfidf.transform([cleaned_tweet])
    
    # Predict sentiment
    prediction = model.predict(tweet_vectorized)[0]
    
    return prediction

# Example Predictions
tweet1 = "I love this product! It's amazing. 😊"
tweet2 = "This is the worst experience ever. So disappointed 😡"

print(f"Tweet: {tweet1} -> Sentiment: {predict_sentiment(tweet1)}")
print(f"Tweet: {tweet2} -> Sentiment: {predict_sentiment(tweet2)}")

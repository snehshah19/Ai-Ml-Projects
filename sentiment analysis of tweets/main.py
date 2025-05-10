

# 📌 Importing Required Libraries
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
# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load stopwords
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

# 📌 File Paths
input_file = "training.1600000.processed.noemoticon.csv"
output_file = "cleaned_sentiment140.csv"

# Define column names
columns = ['target', 'id', 'date', 'flag', 'user', 'text']

# Define chunk size (How many rows to process at a time)
chunksize = 100000  

# 📌 Process Data in Chunks
for chunk in pd.read_csv(input_file, encoding='latin-1', names=columns, chunksize=chunksize):
    # Keep only necessary columns
    chunk = chunk[['target', 'text']]
    
    # Convert sentiment labels (0 → Negative, 4 → Positive)
    chunk['target'] = chunk['target'].replace({0: 'negative', 4: 'positive'})

    # **Optimized Cleaning (Vectorized for Speed)**
    chunk['cleaned_text'] = chunk['text'].str.lower()
    chunk['cleaned_text'] = chunk['cleaned_text'].str.replace(r'http\S+', '', regex=True)
    chunk['cleaned_text'] = chunk['cleaned_text'].str.replace(r'@[A-Za-z0-9_]+', '', regex=True)
    chunk['cleaned_text'] = chunk['cleaned_text'].str.replace(r'#[A-Za-z0-9_]+', '', regex=True)
    chunk['cleaned_text'] = chunk['cleaned_text'].str.replace(r'[^a-zA-Z\s]', '', regex=True)

    # Remove stopwords (List Comprehension for Speed)
    chunk['cleaned_text'] = chunk['cleaned_text'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))

    # 📌 Append to Output CSV (Avoids memory overflow)
    chunk.to_csv(output_file, mode='a', header=not bool(pd.io.common.file_exists(output_file)), index=False)
    
    print(f"✅ Processed {len(chunk)} rows and saved to file...")

print("🎉 Full dataset cleaned and saved to 'cleaned_sentiment140.csv'!")

# 📌 Load the cleaned dataset into a DataFrame
df = pd.read_csv(output_file)

# Convert Text to Numerical Features (TF-IDF)
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_text'])  # ✅ Now df is defined
y = df['target']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save Model
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
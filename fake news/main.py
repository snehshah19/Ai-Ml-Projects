import pandas as pd
import nltk
import string
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load datasets
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

# Assign labels
fake_df["label"] = 0  # Fake News → 0
true_df["label"] = 1  # Real News → 1

# Combine both datasets
df = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)

# Drop missing values
df = df.dropna(subset=["text"])

# Convert text to lowercase
df["text"] = df["text"].str.lower()

# Remove punctuation
df["text"] = df["text"].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

# Tokenization
df["word_token"] = df["text"].apply(word_tokenize)

# Stopword Removal
stop_words = set(stopwords.words('english'))
df["filtered_words"] = df["word_token"].apply(lambda words: [word for word in words if word not in stop_words and len(word) > 2])

# Lemmatization
lemmatizer = WordNetLemmatizer()
df["lemmatized_words"] = df["filtered_words"].apply(lambda words: [lemmatizer.lemmatize(word) for word in words])

# Convert words back to text
df["cleaned_text"] = df["lemmatized_words"].apply(lambda words: " ".join(words))

# Remove empty rows
df = df[df["cleaned_text"].str.strip() != ""]

# TF-IDF Feature Engineering
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(df["cleaned_text"])
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the Naïve Bayes Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the trained model & TF-IDF vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")

print("\n Model Training Completed & Saved!")

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n Model Accuracy:", accuracy)
print("\n Classification Report:\n", classification_report(y_test, y_pred))
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#  Prediction Function (Loads the model & vectorizer)
def predict_fake_news(news_text):
    # Load model & vectorizer (ensures no retraining)
    model = joblib.load("fake_news_model.pkl")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

    # Preprocess input text
    news_text = news_text.lower()
    news_text = news_text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(news_text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    cleaned_text = " ".join(tokens)

    # Convert text to TF-IDF features
    text_features = tfidf_vectorizer.transform([cleaned_text])

    # Predict
    prediction = model.predict(text_features)[0]

    return "🟢 Real News" if prediction == 1 else "🔴 Fake News"

#  Example Usage
while True:
    sample_news = input("\nEnter news text (or type 'exit' to quit): ")
    if sample_news.lower() == "exit":
        break
    print("\n Prediction:", predict_fake_news(sample_news))

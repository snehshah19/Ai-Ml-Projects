import joblib
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load saved model and vectorizer
model = joblib.load("fake_news_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load stopwords and lemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def predict_fake_news(news_text):
    # Preprocess input text
    news_text = news_text.lower()
    news_text = news_text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(news_text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    cleaned_text = " ".join(tokens)

    # Convert to TF-IDF features
    text_features = tfidf_vectorizer.transform([cleaned_text])

    # Predict
    prediction = model.predict(text_features)[0]
    
    return "Real News" if prediction == 1 else "Fake News"

# Get user input
sample_news = input("\nEnter the news text to verify: ")
print("\n🔹 Prediction:", predict_fake_news(sample_news))

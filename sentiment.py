import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import random

# Generate synthetic workplace feedback
positive_templates = [
    "I enjoy working with my team, we have great synergy.",
    "Management is very supportive and responsive to our needs.",
    "The work-life balance here is fantastic!",
    "I feel valued and appreciated at my workplace.",
    "The projects I work on are interesting and fulfilling."
]

negative_templates = [
    "There's a lack of communication from management.",
    "The workload is overwhelming and poorly distributed.",
    "The company culture could use some improvement.",
    "The office environment is not as collaborative as it should be.",
    "I don't feel like my contributions are recognized."
]


# Function to generate feedback
def generate_feedback(sentiment, num_samples=100):
    data = []
    
    if sentiment == 'positive':
        templates = positive_templates
    elif sentiment == 'negative':
        templates = negative_templates

    for _ in range(num_samples):
        feedback = random.choice(templates)
        data.append(feedback)
    
    return data

# Generate synthetic data
positive_feedback = generate_feedback('positive', 100)
negative_feedback = generate_feedback('negative', 100)

# Combine into a single dataset
synthetic_data = positive_feedback + negative_feedback

# Assign labels to synthetic data
synthetic_labels = [1] * 100 + [0] * 100 

# Create a DataFrame for the synthetic data
df_synthetic = pd.DataFrame({
    'Text': synthetic_data,
    'Sentiment': synthetic_labels
})

# Download NLTK stopwords if needed
nltk.download('punkt_tab')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Function to preprocess text data
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stopwords and punctuation
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return " ".join(tokens)

# Apply the same preprocessing to the synthetic data
df_synthetic['cleaned_text'] = df_synthetic['Text'].apply(preprocess_text)


# Load the dataset
df_sentiment140 = pd.read_csv('sentiment140.csv', encoding='ISO-8859-1', header=None)
df_sentiment140.columns = ['Sentiment', 'ID', 'Date', 'Query', 'User', 'Text']

# Apply preprocessing to the text
df_sentiment140['cleaned_text'] = df_sentiment140['Text'].apply(preprocess_text)

# Convert sentiment to binary: 0 for negative, 1 for positive
df_sentiment140['Sentiment'] = df_sentiment140['Sentiment'].apply(lambda x: 1 if x == 4 else 0)

df_combined = pd.concat([df_sentiment140[['cleaned_text', 'Sentiment']], df_synthetic[['cleaned_text', 'Sentiment']]])


# Split data into training and test sets
X = df_combined['cleaned_text']
y = df_combined['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = LogisticRegression(max_iter=1000)   
model.fit(X_train_tfidf, y_train)

# Predict on the test data
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

joblib.dump(model, 'sentiment_model_fineTuned.pkl')
joblib.dump(vectorizer, 'vectorizer_fineTuned.pkl')

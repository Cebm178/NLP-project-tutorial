import pandas as pd
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import regex as re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pickle import dump

from urllib.parse import urlparse
import re

# Load the data
total_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv")
print(total_data.head())

# Handle missing values
total_data.dropna(inplace=True)

# Convert 'is_spam' to binary (1 for spam, 0 for not spam)
total_data["is_spam"] = total_data["is_spam"].apply(lambda x: 1 if x else 0).astype(int)

# Drop duplicates and reset the index
total_data.drop_duplicates(inplace=True)
total_data.reset_index(drop=True, inplace=True)

print(f"Spam: {len(total_data.loc[total_data.is_spam == 1])}")
print(f"No spam: {len(total_data.loc[total_data.is_spam == 0])}")

# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z ]', " ", text)  # Keep only letters and spaces
    text = re.sub(r'\s+', " ", text.lower())  # Remove multiple spaces and convert to lowercase
    return text

# Apply text preprocessing
total_data["url"] = total_data["url"].apply(preprocess_text)

# Download NLTK resources
download("wordnet")
download("stopwords")

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Lemmatization and stopwords removal
def lemmatize_text(text):
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 3]
    return " ".join(words)

# Apply the lemmatization
total_data["url"] = total_data["url"].apply(lemmatize_text)

# Check the first few entries in the processed_url column
print(total_data["processed_url"].head())

# Check for any empty or null values
print(total_data["processed_url"].isnull().sum())
print(total_data["processed_url"].apply(lambda x: len(x.strip()) == 0).sum())

# Function to extract meaningful parts of a URL with modified regex
def extract_text_from_url(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc  # Extract domain
    path = parsed_url.path      # Extract path
    # Allow letters, numbers, and hyphens
    words = re.sub(r"[^a-zA-Z0-9\-]", " ", domain + " " + path)
    return words

# Apply the modified function
total_data["processed_url"] = total_data["url"].apply(extract_text_from_url)

# Verify the processed URLs
print(total_data["processed_url"].head())

# Display stopwords to ensure they are not too restrictive
print(stopwords)

# Print first 10 entries of processed_url
print(total_data["processed_url"].head(10))

def extract_text_from_url(url):
    if not url:  # Check if URL is empty or None
        return ""
    
    parsed_url = urlparse(url)
    domain = parsed_url.netloc  # Extract domain
    path = parsed_url.path      # Extract path
    query = parsed_url.query    # Extract query parameters (optional)

    # Use a less restrictive regex to retain more meaningful content
    # Keep letters, numbers, hyphens, dots, and slashes
    words = re.sub(r"[^a-zA-Z0-9\-./]", " ", domain + " " + path + " " + query)
    return words.strip()

# Apply the modified function
total_data["processed_url"] = total_data["url"].apply(extract_text_from_url)

# Check for non-empty processed URLs
print(total_data["processed_url"].apply(lambda x: len(x.strip()) == 0).sum())  # Should return a lower number than 2369

# Generate word cloud with improved stopwords
wordcloud = WordCloud(width=800, height=800, background_color="black", stopwords=stopwords, max_words=1000, min_font_size=20, random_state=42).generate(" ".join(total_data["processed_url"]))

plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# Vectorization (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000, max_df=0.8, min_df=5)
X = vectorizer.fit_transform(total_data["url"]).toarray()
y = total_data["is_spam"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train SVC model
model = SVC(kernel="linear", random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Grid Search for hyperparameter tuning
hyperparams = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "degree": [1, 2, 3, 4, 5],
    "gamma": ["scale", "auto"]
}
grid = GridSearchCV(SVC(random_state=42), hyperparams, scoring="accuracy", cv=5)
grid.fit(X_train, y_train)

# Best parameters
print(f"Best hyperparameters: {grid.best_params_}")

# Optimal model with best parameters
opt_model = SVC(**grid.best_params_, random_state=42)
opt_model.fit(X_train, y_train)
y_pred_opt = opt_model.predict(X_test)

# Final evaluation
print(f"Optimized Accuracy: {accuracy_score(y_test, y_pred_opt)}")
print("Optimized Classification Report:")
print(classification_report(y_test, y_pred_opt))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_opt))

# Save the model and vectorizer
dump(opt_model, open("/models/optimized_svm_model.sav", "wb"))
dump(vectorizer, open("/models/tfidf_vectorizer.sav", "wb"))
"""
Phishing / Spam Message Detection using NLP
Course: Natural Language Processing (ET5M004)
Dataset: SMS Spam Collection Dataset
"""

import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources (only needed once)
nltk.download('stopwords')

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("../dataset/spam.csv", encoding="latin-1")

# Dataset has extra unnamed columns, keep only useful ones
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print("Dataset shape:", df.shape)
print(df.head())

# -----------------------------
# 2. Text Preprocessing
# -----------------------------
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)          # remove URLs
    text = re.sub(r'\d+', '', text)                      # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['clean_message'] = df['message'].apply(clean_text)

print("\nSample cleaned messages:")
print(df[['message', 'clean_message']].head())

# -----------------------------
# 3. Feature Extraction (TF-IDF)
# -----------------------------
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_message']).toarray()
y = df['label'].map({'ham': 0, 'spam': 1})   # 0 = safe, 1 = phishing/spam

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 5. Model Training (Naive Bayes)
# -----------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# -----------------------------
# 6. Evaluation
# -----------------------------
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n----- Model Evaluation -----")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1 Score  : {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# -----------------------------
# 7. Confusion Matrix (save to output/)
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Phishing Message Detection')
plt.tight_layout()
plt.savefig("../output/confusion_matrix.png")
plt.show()

# -----------------------------
# 8. Test on Custom/Sample Input
# -----------------------------
sample_messages = [
    "Congratulations! You have won a $1000 Walmart gift card. Click here to claim now!",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT: Your bank account has been suspended. Verify your details immediately."
]

sample_clean = [clean_text(msg) for msg in sample_messages]
sample_vec = vectorizer.transform(sample_clean).toarray()
sample_pred = model.predict(sample_vec)

print("\n----- Sample Predictions -----")
for msg, pred in zip(sample_messages, sample_pred):
    label = "SPAM/PHISHING" if pred == 1 else "SAFE (HAM)"
    print(f"Message: {msg}\nPrediction: {label}\n")
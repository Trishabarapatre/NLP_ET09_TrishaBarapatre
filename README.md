# NLP_ET09_TrishaBarapatre
Phishing Message Detection using NLP

**Student Name:** Trisha Ramesh Barapatre
**Roll Number:** ET09
**Semester/Branch:** V Semester ETC
**Course:** Natural Language Processing (ET5M004)

---

## Problem Statement / Objective
To build an NLP-based system that can automatically classify SMS messages as **Spam/Phishing** or **Ham (Safe)**, helping identify potentially fraudulent or malicious messages using text classification techniques.

## Introduction
Phishing and spam messages are a major security concern, often containing malicious links, fake offers, or urgent requests designed to deceive users. This project applies Natural Language Processing techniques to automatically detect such messages based on their textual content, using a machine learning classification approach.

## NLP Technique / Method Used
- Text Preprocessing (lowercasing, URL/number removal, punctuation removal, stopword removal, stemming)
- Feature Extraction using **TF-IDF (Term Frequency–Inverse Document Frequency)**
- Classification using **Multinomial Naive Bayes**

## Dataset / Source of Data
**SMS Spam Collection Dataset** (Kaggle / UCI Machine Learning Repository)
- Contains 5,572 SMS messages labeled as `ham` (safe) or `spam` (phishing/spam)

## Software / Tools / Libraries Used
- Python 3.13
- pandas, numpy
- scikit-learn
- nltk
- matplotlib, seaborn
- VS Code

## Methodology / Workflow
1. Load dataset (`spam.csv`)
2. Clean and preprocess text (remove URLs, numbers, punctuation, stopwords; apply stemming)
3. Convert text into numerical features using TF-IDF Vectorizer
4. Split data into training (80%) and testing (20%) sets
5. Train a Multinomial Naive Bayes classifier
6. Evaluate the model using accuracy, precision, recall, and F1-score
7. Visualize results using a confusion matrix
8. Test the model on custom sample messages

## Steps to Execute the Program
```bash
# Install dependencies
pip install pandas numpy scikit-learn nltk matplotlib seaborn

# Run the script
cd source_code
python phishing_detection.py
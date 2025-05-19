import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
from wordcloud import WordCloud
import os

# Define base directory for file paths
# BASE_DIR is the directory containing this script (Finora.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define file paths (all files are in the same directory as Finora.py)
MODEL_PATH = os.path.join(BASE_DIR, 'logistic_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')
TRAIN_PATH = os.path.join(BASE_DIR, 'train.csv')
VALID_PATH = os.path.join(BASE_DIR, 'valid.csv')

# Load model and vectorizer with error handling
try:
    model = pickle.load(open(MODEL_PATH, 'rb'))
except FileNotFoundError:
    st.error(f"Error: '{MODEL_PATH}' not found. Please ensure the file is included in the app directory.")
    st.stop()

try:
    vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
except FileNotFoundError:
    st.error(f"Error: '{VECTORIZER_PATH}' not found. Please ensure the file is included in the app directory.")
    st.stop()

# Load datasets with error handling
try:
    train_df = pd.read_csv(TRAIN_PATH)
except FileNotFoundError:
    st.error(f"Error: '{TRAIN_PATH}' not found. Please ensure the file is included in the app directory.")
    st.stop()

try:
    valid_df = pd.read_csv(VALID_PATH)
except FileNotFoundError:
    st.error(f"Error: '{VALID_PATH}' not found. Please ensure the file is included in the app directory.")
    st.stop()

# Clean text function
def clean_text(text):
    # Ensure input is a string to avoid errors with non-string data
    text = str(text)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text.lower()

# Apply text cleaning to training data
train_df['clean_text'] = train_df['text'].apply(clean_text)

# Sidebar Navigation
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", ["Dashboard", "Live Prediction"])

# App Title and Subtitle
st.title("🌟 Finora")
st.markdown("### Twitter Financial News Sentiment Analysis")

# Dashboard Section
if choice == "Dashboard":
    st.subheader("📊 Tweet Category Distribution")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.countplot(x='label', data=train_df, ax=ax1)
    ax1.set_xlabel("Category")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

    st.subheader("☁️ Word Cloud of Financial Tweets")
    text = ' '.join(train_df['clean_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.imshow(wordcloud, interpolation='bilinear')
    ax2.axis('off')
    st.pyplot(fig2)

    st.subheader("🔍 Explore Tweet Samples")
    category_filter = st.selectbox("Filter by Label", sorted(train_df['label'].unique()))
    st.write(train_df[train_df['label'] == category_filter][['text']].sample(5, random_state=42))

# Live Prediction Section
elif choice == "Live Prediction":
    st.subheader("🧠 Predict Tweet Category")
    user_input = st.text_area("Enter a financial tweet:", height=100)
    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter a tweet to predict.")
        else:
            try:
                cleaned = clean_text(user_input)
                vect = vectorizer.transform([cleaned]).toarray()
                prediction = model.predict(vect)
                st.success(f"Predicted Category: {prediction[0]}")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

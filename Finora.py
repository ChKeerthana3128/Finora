import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud
import os

# Define base directory for file paths
# BASE_DIR is the directory containing this script (Finora.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define file paths (all files are in the same directory as Finora.py)
TRAIN_PATH = os.path.join(BASE_DIR, 'train.csv')
VALID_PATH = os.path.join(BASE_DIR, 'valid.csv')

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
choice = st.sidebar.radio("Go to", ["Dashboard"])

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

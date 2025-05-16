import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
from wordcloud import WordCloud

# Load model and vectorizer
model = pickle.load(open('logistic_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Load datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Clean text function
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text.lower()

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
    fig1, ax1 = plt.subplots()
    sns.countplot(x='label', data=train_df, ax=ax1)
    st.pyplot(fig1)

    st.subheader("☁️ Word Cloud of Financial Tweets")
    text = ' '.join(train_df['clean_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig2, ax2 = plt.subplots()
    ax2.imshow(wordcloud, interpolation='bilinear')
    ax2.axis('off')
    st.pyplot(fig2)

    st.subheader("🔍 Explore Tweet Samples")
    category_filter = st.selectbox("Filter by Label", train_df['label'].unique())
    st.write(train_df[train_df['label'] == category_filter][['text']].sample(5))

# Live Prediction Section
elif choice == "Live Prediction":
    st.subheader("🧠 Predict Tweet Category")

    user_input = st.text_area("Enter a financial tweet:")
    if st.button("Predict"):
        cleaned = clean_text(user_input)
        vect = vectorizer.transform([cleaned]).toarray()
        prediction = model.predict(vect)
        st.success(f"Predicted Category: {prediction[0]}")

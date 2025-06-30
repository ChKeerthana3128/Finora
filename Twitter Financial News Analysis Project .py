import streamlit as st
import pandas as pd
import re
import tweepy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Twitter API credentials (stored in a .env file for security)
API_KEY = os.getenv("TWITTER_API_KEY")
API_SECRET = os.getenv("TWITTER_API_SECRET")
ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# Initialize Tweepy client
auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

# Load & preprocess
@st.cache_data
def load_and_prepare():
    train_df = pd.read_csv("train_data.csv")
    valid_df = pd.read_csv("valid_data.csv")

    def clean_text(text):
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = text.lower().strip()
        return text

    train_df['clean_text'] = train_df['text'].apply(clean_text)
    valid_df['clean_text'] = valid_df['text'].apply(clean_text)

    return train_df, valid_df

# Train model
@st.cache_resource
def train_model(train_df):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(train_df['clean_text'])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(train_df['label'])

    model = LogisticRegression(max_iter=300)
    model.fit(X, y)

    return model, vectorizer, label_encoder

# Label mapping
label_map = {
    0: "Analyst Update", 1: "Fed | Central Banks", 2: "Company | Product News",
    3: "Treasuries | Corporate Debt", 4: "Dividend", 5: "Earnings", 6: "Energy | Oil",
    7: "Financials", 8: "Currencies", 9: "General News | Opinion", 10: "Gold | Metals | Materials",
    11: "IPO", 12: "Legal | Regulation", 13: "M&A | Investments", 14: "Macro", 15: "Markets",
    16: "Politics", 17: "Personnel Change", 18: "Stock Commentary", 19: "Stock Movement"
}

# Function to fetch recent tweets from a Twitter handle
def fetch_tweets_from_handle(handle, count=10):
    try:
        tweets = api.user_timeline(screen_name=handle, count=count, tweet_mode="extended")
        return [tweet.full_text for tweet in tweets]
    except tweepy.TweepError as e:
        st.error(f"Error fetching tweets: {str(e)}")
        return []

# Streamlit app
def main():
    st.set_page_config(page_title="Twitter Financial News Classifier", layout="centered")
    st.title("📈 Twitter Financial News - Topic Classifier")

    st.markdown("Enter a Twitter handle or a tweet to predict its financial news category.")

    train_df, valid_df = load_and_prepare()
    model, vectorizer, label_encoder = train_model(train_df)

    # Input for Twitter handle or manual tweet
    input_type = st.radio("Choose input type:", ("Twitter Handle", "Manual Tweet"))

    if input_type == "Twitter Handle":
        twitter_handle = st.text_input("🐦 Enter Twitter handle (without @):", value="Reuters")
        tweet_count = st.slider("Number of tweets to fetch:", 1, 100, 10)
        if st.button("🔍 Fetch and Classify Tweets"):
            if not twitter_handle.strip():
                st.warning("Please enter a Twitter handle.")
            else:
                tweets = fetch_tweets_from_handle(twitter_handle, tweet_count)
                if tweets:
                    st.subheader(f"Classified Tweets from @{twitter_handle}")
                    for tweet in tweets:
                        clean_input = re.sub(r"http\S+|www\S+", "", tweet)
                        clean_input = re.sub(r"[^a-zA-Z\s]", "", clean_input).lower().strip()
                        X_input = vectorizer.transform([clean_input])
                        pred = model.predict(X_input)[0]
                        category = label_map[pred]
                        st.write(f"**Tweet**: {tweet}")
                        st.success(f"**Predicted Topic**: {category}")
                else:
                    st.warning("No tweets fetched. Check the handle or API access.")
    else:
        user_input = st.text_area("📝 Enter tweet here:", max_chars=1000)
        if st.button("🔍 Predict Category"):
            if not user_input.strip():
                st.warning("Please enter a tweet.")
            else:
                clean_input = re.sub(r"http\S+|www\S+", "", user_input)
                clean_input = re.sub(r"[^a-zA-Z\s]", "", clean_input).lower().strip()
                X_input = vectorizer.transform([clean_input])
                pred = model.predict(X_input)[0]
                category = label_map[pred]
                st.success(f"🏷 Predicted Topic: **{category}**")

    if st.checkbox("📊 Show Validation Data"):
        st.dataframe(valid_df[['text', 'label']].head())

if __name__ == "__main__":
    main()

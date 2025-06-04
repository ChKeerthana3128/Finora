import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load and preprocess data
@st.cache_data
def load_data():
    train_df = pd.read_csv('train_data.csv')
    valid_df = pd.read_csv('valid_data.csv')
    return train_df, valid_df

# Train model
@st.cache_resource
def train_model(train_df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train = tfidf.fit_transform(train_df['text'])
    y_train = train_df['label']
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    return model, tfidf

# Map labels
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Streamlit UI
def main():
    st.title("📊 Twitter Financial Sentiment Analyzer")
    st.write("Analyze the sentiment of financial news tweets")

    train_df, valid_df = load_data()
    model, tfidf = train_model(train_df)

    st.subheader("📌 Enter a tweet for analysis:")
    user_input = st.text_area("Tweet:", max_chars=280)

    if st.button("Predict Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter a tweet.")
        else:
            X_input = tfidf.transform([user_input])
            prediction = model.predict(X_input)[0]
            st.success(f"Predicted Sentiment: **{label_map[prediction]}**")

    st.subheader("📈 Validation Data Preview")
    st.dataframe(valid_df.head())

if __name__ == '__main__':
    main()

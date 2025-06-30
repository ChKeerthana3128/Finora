import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

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

# Streamlit app
def main():
    st.set_page_config(page_title="Twitter Financial News Classifier", layout="centered")
    st.title("📈 Twitter Financial News - Topic Classifier")

    st.markdown("Enter a financial tweet to predict its category.")

    train_df, valid_df = load_and_prepare()
    model, vectorizer, label_encoder = train_model(train_df)

    # Increased max_chars to 1000
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

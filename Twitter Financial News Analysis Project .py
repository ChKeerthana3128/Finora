import streamlit as st
import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Define text cleaning function (same as in Colab notebook)
def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Load models and vectorizer
@st.cache_resource
def load_models():
    try:
        lr_model = joblib.load('lr_model.pkl')
        tfidf = joblib.load('tfidf_vectorizer.pkl')
        le = joblib.load('label_encoder.pkl')
        return lr_model, tfidf, le
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        return None, None, None

lr_model, tfidf, le = load_models()

# Load dataset for statistics
@st.cache_data
def load_data():
    try:
        df_train = pd.read_csv('train_data.csv')
        return df_train
    except FileNotFoundError as e:
        st.error(f"Dataset not found: {e}")
        return None

df_train = load_data()

# Streamlit app layout
st.title("Twitter Financial News Analysis Dashboard")
st.markdown("Explore financial tweet analysis results and classify new tweets.")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page", ["Dataset Exploration", "Model Evaluation", "Tweet Classification"])

# Dataset Exploration Page
if page == "Dataset Exploration":
    st.header("Dataset Exploration")
    if df_train is not None:
        st.subheader("Dataset Statistics")
        st.write(f"Number of tweets: {df_train.shape[0]}")
        st.write(f"Number of unique labels: {df_train['label'].nunique()}")
        st.write("Label distribution:")
        st.dataframe(df_train['label'].value_counts().reset_index(name='Count'))

        # Tweet length distribution
        df_train['text_length'] = df_train['text'].apply(lambda x: len(x.split()))
        st.subheader("Tweet Length Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df_train['text_length'], bins=30, kde=True, ax=ax)
        ax.set_title('Tweet Length Distribution')
        ax.set_xlabel('Number of Words')
        st.pyplot(fig)

        # Word cloud
        st.subheader("Word Cloud")
        if os.path.exists('wordcloud_train.png'):
            image = Image.open('wordcloud_train.png')
            st.image(image, caption="Word Cloud of Training Data")
        else:
            st.warning("Word cloud image not found. Run Colab notebook to generate.")

        # Label distribution plot
        st.subheader("Label Distribution")
        if os.path.exists('label_distribution.png'):
            image = Image.open('label_distribution.png')
            st.image(image, caption="Label Distribution in Training Data")
        else:
            st.warning("Label distribution plot not found. Run Colab notebook to generate.")

# Model Evaluation Page
elif page == "Model Evaluation":
    st.header("Model Evaluation")
    st.subheader("Logistic Regression Performance")
    if os.path.exists('lr_confusion_matrix.png'):
        image = Image.open('lr_confusion_matrix.png')
        st.image(image, caption="Confusion Matrix for Logistic Regression")
    else:
        st.warning("Confusion matrix image not found. Run Colab notebook to generate.")

    # Display classification report (simulated, as report requires y_valid)
    st.write("Note: Detailed classification report requires validation data and predictions. Run Colab notebook for full metrics.")

# Tweet Classification Page
elif page == "Tweet Classification":
    st.header("Tweet Classification")
    if lr_model is None or tfidf is None or le is None:
        st.error("Models not loaded. Ensure model files are available.")
    else:
        st.subheader("Classify a New Tweet")
        user_input = st.text_area("Enter a financial tweet:", "Apple stock rises after strong earnings report")
        if st.button("Classify"):
            # Clean and transform input
            clean_input = clean_text(user_input)
            input_tfidf = tfidf.transform([clean_input])
            # Predict
            prediction = lr_model.predict(input_tfidf)[0]
            label = le.inverse_transform([prediction])[0]
            st.success(f"Predicted Label: **{label}**")
            st.write(f"Input Tweet: {user_input}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit for Twitter Financial News Analysis Project")

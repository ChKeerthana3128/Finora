import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Define text cleaning function
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

# Load dataset
try:
    df_train = pd.read_csv('train_data.csv')
except FileNotFoundError:
    print("Error: 'train_data.csv' not found. Please provide the dataset.")
    exit(1)

# Clean text data
df_train['cleaned_text'] = df_train['text'].apply(clean_text)

# Encode labels
le = LabelEncoder()
df_train['label_encoded'] = le.fit_transform(df_train['label'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df_train['cleaned_text'], df_train['label_encoded'], test_size=0.2, random_state=42
)

# Vectorize text
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train logistic regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

# Save models
joblib.dump(lr_model, 'lr_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(le, 'label_encoder.pkl')

# Generate word cloud
all_text = ' '.join(df_train['cleaned_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('wordcloud_train.png')
plt.close()

# Generate label distribution plot
plt.figure(figsize=(8, 4))
sns.countplot(x='label', data=df_train)
plt.title('Label Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.savefig('label_distribution.png')
plt.close()

# Generate confusion matrix
y_pred = lr_model.predict(X_test_tfidf)
cm = confusion_matrix(y_test, le.inverse_transform(y_pred))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix for Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('lr_confusion_matrix.png')
plt.close()

print("Models and visualizations generated successfully!")

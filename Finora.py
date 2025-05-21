import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk import word_tokenize
import nltk
from wordcloud import WordCloud
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Bidirectional, GRU, GlobalMaxPooling1D, Dropout, SimpleRNN, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy, categorical_crossentropy
from sklearn.feature_extraction.text import CountVectorizer

# nltk.download("stopwords")
# nltk.download("punkt")
# nltk.download("wordnet")

# get stopwords
stops = set(stopwords.words("english"))

df_train = pd.read_csv("/kaggle/input/twitter-financial-news/train_data.csv")
df_test = pd.read_csv("/kaggle/input/twitter-financial-news/valid_data.csv")
df_train.head()

# get correct mapping of ordinal encoded target variables
label_mapping = {
    "LABEL_0": "Analyst Update",
    "LABEL_1": "Fed | Central Banks",
    "LABEL_2": "Company | Product News",
    "LABEL_3": "Treasuries | Corporate Debt",
    "LABEL_4": "Dividend",
    "LABEL_5": "Earnings",
    "LABEL_6": "Energy | Oil",
    "LABEL_7": "Financials",
    "LABEL_8": "Currencies",
    "LABEL_9": "General News | Opinion",
    "LABEL_10": "Gold | Metals | Materials",
    "LABEL_11": "IPO",
    "LABEL_12": "Legal | Regulation",
    "LABEL_13": "M&A | Investments",
    "LABEL_14": "Macro",
    "LABEL_15": "Markets",
    "LABEL_16": "Politics",
    "LABEL_17": "Personnel Change",
    "LABEL_18": "Stock Commentary",
    "LABEL_19": "Stock Movement"
}
label_mapping = {k: v for k, v in zip(range(20), label_mapping.values())}
label_mapping

# get understanding of the text data
# number of the random articles
S = 5
inds = np.random.choice(train_corpus.index, 5)
for i in inds:
    print(train_corpus[i])

# there is a need to remove all hyperlinks, since they do not contain any contextual text data
def remove_hyperlinks_and_punctuation(text):
    pattern = r'\bhttps?:\/\/\S+|[^\w\s]'
    new_text = re.sub(pattern, "", text)
    return new_text

train_corpus_cleaned = train_corpus.apply(lambda x: remove_hyperlinks_and_punctuation(x))
test_corpus_cleaned = test_corpus.apply(lambda x: remove_hyperlinks_and_punctuation(x))

for i in inds:
    print(train_corpus_cleaned[i])

# use count vectorizer to feed in ANN without paying attention to the sequence
# Could make sense, because classification of the source of the news may not be dependent on the sequence of the words, but rather on some certain important words which are unique for the source
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]

MAX_VOCAB_SIZE_ANN = 18000
vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), stop_words=list(stops), max_features=MAX_VOCAB_SIZE_ANN)
vectorizer.fit(train_corpus_cleaned)
train_data_ann = vectorizer.transform(train_corpus_cleaned)
test_data_ann = vectorizer.transform(test_corpus_cleaned)

# create batch generator to feed sparse matrix into keras fit method
def batch_generator(X, y, batch_size):
    number_of_batches = X.shape[0] / batch_size
    counter = 0
    shuffle_index = np.arange(np.shape(y)[0])
    np.random.shuffle(shuffle_index)
    X = X[shuffle_index, :]
    y = y[shuffle_index]
    while 1:
        # each batch contains all the shuffled indices
        index_batch = shuffle_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[index_batch, :].todense()
        y_batch = y[index_batch]
        counter += 1
        yield (np.array(X_batch), y_batch)
        if counter < number_of_batches:
            np.random.shuffle(shuffle_index)
            counter = 0

# Apply lemmatization to corpus - remove maybe later in case of poor performance
# used for sequence models
wnl = WordNetLemmatizer()

def lemmatize_sentence(sentence):
    tokens = [wnl.lemmatize(t) for t in word_tokenize(sentence)]
    return (" ").join(tokens)

train_corpus_cleaned = train_corpus_cleaned.apply(lambda x: lemmatize_sentence(x))
test_corpus_cleaned = test_corpus_cleaned.apply(lambda x: lemmatize_sentence(x))

for i in inds:
    print(train_corpus_cleaned[i])

# transform the corpus to get training and test data
MAX_VOCAB_SIZE = 30000
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="OOV", lower=True)
tokenizer.fit_on_texts(train_corpus_cleaned)
train_data = tokenizer.texts_to_sequences(train_corpus_cleaned)
test_data = tokenizer.texts_to_sequences(test_corpus_cleaned)

# get max len of sentences for padding
maxlen1 = max(len(sent) for sent in train_data)
maxlen2 = max(len(sent) for sent in test_data)
T = max(maxlen1, maxlen2)
print(T)

word2idx = tokenizer.word_index
V = len(word2idx)
print("The corpus contains %s words!" % V)

# pad sequences
train_data_padded = pad_sequences(train_data, maxlen=T)
test_data_padded = pad_sequences(test_data, maxlen=T)

print("Train tensor shape:", train_data_padded.shape)
print("Test tensor shape:", test_data_padded.shape)

# get class number
K = len(set(train_labels))
print(f"We have %s unique labels!" % K)

# show word overview with wordcloud
# not using all words, but rather subsamples
random_subsample_word_wordcloud_size = round(len(train_corpus_cleaned) * 0.33)
wordcloud_ids = np.random.choice(len(train_corpus_cleaned), random_subsample_word_wordcloud_size)
concat_text = ""
for i in wordcloud_ids:
    concat_text += " " + train_corpus_cleaned[i]

wordcloud = WordCloud(background_color="white", width=800, height=400, stopwords=stops).generate(concat_text)
fig, ax = plt.subplots(figsize=(14, 10))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
plt.title("Wordcloud of most frequent words")
plt.show()

# distribution of train targets
fig, ax = plt.subplots(figsize=(12, 8))
sns.countplot(x=train_labels.index, data=train_labels, ax=ax, palette="hls")
plt.title("Distribution of train labels", fontsize=25)
plt.show()
print("Highly imbalanced data!!")

# distribution of test targets
fig, ax = plt.subplots(figsize=(12, 8))
sns.countplot(x=test_labels.index, data=test_labels, ax=ax, palette="hls")
plt.title("Distribution of test labels", fontsize=25)
plt.show()
print("Highly imbalanced data!!")

# highly imbalanced data, so inverse class weights are computed for the loss function
class_weights = train_labels.value_counts(normalize=True).sort_index()
inverse_class_weights = class_weights.apply(lambda x: 1 / x)
inverse_class_weights

# Embedding dimension
D = 64
hidden_states1 = 12

def make_model_rnn():
    with tf.device("/GPU:0"):
        i = Input(shape=(T,))
        x = Embedding(V + 1, D)(i)
        x = SimpleRNN(hidden_states1, return_sequences=True)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(K, activation="softmax")(x)
        model = Model(i, x)
        model.compile(loss=SparseCategoricalCrossentropy(), optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
        return model

def make_model_cnn():
    with tf.device("/GPU:0"):
        i = Input(shape=(T,))
        x = Embedding(V + 1, D)(i)
        x = Conv1D(32, 3)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.3)(x)
        x = Dense(K, activation="softmax")(x)
        model = Model(i, x)
        model.compile(loss=SparseCategoricalCrossentropy(), optimizer=SGD(learning_rate=0.0001, momentum=0.9), metrics=["accuracy"])
        return model

def make_model_ann():
    with tf.device("/GPU:0"):
        i = Input(shape=(MAX_VOCAB_SIZE_ANN,))
        x = Dense(1028, activation="relu")(i)
        x = Dropout(0.2)(x)
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(256, activation="relu")(x)
        x = Dense(K, activation="softmax")(x)
        model = Model(i, x)
        model.compile(loss=SparseCategoricalCrossentropy(), optimizer=SGD(learning_rate=0.0001, momentum=0.9), metrics=["accuracy"])
        # maybe focal loss
        return model

model = make_model_cnn()
print(model.summary())
r1 = model.fit(train_data_padded, train_labels, epochs=100, batch_size=32, validation_data=(test_data_padded, test_labels), class_weight=dict(inverse_class_weights))

# train also a simple ANN
model2 = make_model_ann()
print(model2.summary())
r2 = model2.fit(train_data_ann.toarray(), train_labels, epochs=100, batch_size=32, validation_data=(test_data_ann.toarray(), test_labels), class_weight=dict(inverse_class_weights), shuffle=True)

plt.plot(r2.history["accuracy"], label="Train accuracy")
plt.plot(r2.history["val_accuracy"], label="Validation accuracy")
plt.legend()
plt.show()

plt.plot(r2.history["loss"], label="Train loss")
plt.plot(r2.history["val_loss"], label="Validation loss")
plt.legend()
plt.show()

from sklearn.metrics import f1_score, classification_report
preds = np.argmax(model2.predict(test_data_ann), axis=1)
print("Classification report for test data")
print(classification_report(preds, test_labels))

plt.plot(r1.history["accuracy"], label="Train accuracy")
plt.plot(r1.history["val_accuracy"], label="Validation accuracy")
plt.legend()
plt.show()

plt.plot(r1.history["loss"], label="Train loss")
plt.plot(r1.history["val_loss"], label="Validation loss")
plt.legend()
plt.show()

preds = np.argmax(model.predict(test_data_padded), axis=1)
print("Classification report for test data")
print(classification_report(preds, test_labels))

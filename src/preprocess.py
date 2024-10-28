# Import libraries
# import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
import numpy as np
import pandas as pd
import re
import os

# Define paths for data loading
base_path = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_path, '../data')

# Load datasets
real_news = pd.read_csv(os.path.join(data_dir, 'BuzzFeed_real_news_content.csv'))
fake_news = pd.read_csv(os.path.join(data_dir, 'BuzzFeed_fake_news_content.csv'))
news_df = pd.concat([real_news, fake_news], ignore_index=True)

# Preprocess DataFrame
news_df['type'] = news_df['id'].str.split('_').str[0]
selected_columns = ["id", "title", "text", "source", "type", "images", "movies"]
news_df = news_df[selected_columns]
news_df['movies'] = news_df['movies'].notna().astype(int)
news_df['images'] = news_df['images'].notna().astype(int)

# Define text cleaning functions
def clean_text(text):
    return re.sub(r"…|⋆|–|‹|”|“|‘|’", " ", text)

# Universal vectorizer function
def vectorize_input(corpus, max_df=1.0, tokenizer=None):
    """
    Vectorizes input text corpus with optional max_df and tokenizer parameters.
    """
    vectorizer = CountVectorizer(max_df=max_df, tokenizer=tokenizer)
    dtm = vectorizer.fit_transform(corpus)
    return dtm.toarray(), vectorizer

# Preprocess title corpus
def preprocess_title_corpus(titles):
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer("english")
    processed_titles = []

    for title in titles:
        title = title.lower()  # Lowercase
        title = re.sub(r'\d+', '', title)  # Remove numbers
        title = re.sub(r'[^\w\s]', '', title)  # Remove punctuation
        tokens = word_tokenize(title)  # Tokenization
        tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
        tokens = [stemmer.stem(word) for word in tokens]  # Stem words
        processed_titles.append(' '.join(tokens))  # Join back into string

    return processed_titles

# Preprocess body corpus
def preprocess_body_corpus(corpus):
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer("english")
    processed_corpus = []

    for doc in corpus:
        doc = doc.lower()  # Lowercase
        doc = re.sub(r'\d+', '', doc)  # Remove numbers
        doc = re.sub(r'[^\w\s]', '', doc)  # Remove punctuation
        tokens = word_tokenize(doc)  # Tokenization
        tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
        tokens = [stemmer.stem(word) for word in tokens]  # Stem words
        processed_corpus.append(' '.join(tokens))  # Join back into string

    return processed_corpus

# Custom bigram tokenizer
def bigram_tokenizer(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    bigrams = list(ngrams(filtered_tokens, 2))
    return [' '.join(bigram) for bigram in bigrams]


# Vectorize title
title_corpus = preprocess_title_corpus(news_df['title'])
title_dtm, title_vectorizer = vectorize_input(title_corpus, max_df=0.997)

# Vectorize body
body_corpus = preprocess_body_corpus(news_df['text'])
body_dtm, body_vectorizer = vectorize_input(body_corpus, max_df=0.97)

# Vectorize bigrams
bigram_dtm, bigram_vectorizer = vectorize_input(news_df['text'], tokenizer=bigram_tokenizer)

# Combine title and body document term matrix (DTM)
def combine_title_body_dtm(body_dtm, title_dtm):
    # Assuming body_dtm and title_dtm are numpy arrays
    if body_dtm.shape[1] < title_dtm.shape[1]:  # Pad body_dtm with zeros if title has more features
        body_dtm = np.pad(body_dtm, ((0, 0), (0, title_dtm.shape[1] - body_dtm.shape[1])), 'constant')
    elif title_dtm.shape[1] < body_dtm.shape[1]:  # Pad title_dtm with zeros if body has more features
        title_dtm = np.pad(title_dtm, ((0, 0), (0, body_dtm.shape[1] - title_dtm.shape[1])), 'constant')
    
    combined_dtm = body_dtm + title_dtm  # Element-wise addition
    return combined_dtm

combined_dtm = combine_title_body_dtm(body_dtm, title_dtm)

# Train-test split
y_true = news_df['type'].values
x_train, x_test, y_train, y_test = train_test_split(combined_dtm, y_true, test_size=0.25, random_state=123, stratify=y_true)


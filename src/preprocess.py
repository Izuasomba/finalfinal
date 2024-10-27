# For text processing
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix 
from sklearn.preprocessing import StandardScaler




import numpy as np

from scipy.stats import chi2_contingency
from scipy import stats
from scipy.sparse import hstack, csr_matrix
# For data manipulation
import pandas as pd

# For visualization
import re



file_path = ('../data/BuzzFeed_real_news_content.csv')
real_news = pd.read_csv(file_path)
real_news.head()

file_path2= ('../data/BuzzFeed_fake_news_content.csv')
fake_news= pd.read_csv(file_path2)
fake_news.head()

news_df = pd.concat([real_news, fake_news], ignore_index=True)


news_df['type'] = news_df['id'].str.split('_').str[0]

#merged both frames into one, added a new column to cateorize as real or fake


selected_columns = ["id", "title", "text", "source", "type", "images", "movies"]
news_df = news_df[selected_columns]

# selected necessary columns to analyse

#I am going to change the "movies" and "images" column to categorical variables, "0" if not available and "1" if available
news_df['movies'] = news_df['movies'].notna().astype(int)
news_df['images'] = news_df['images'].notna().astype(int)

#I am ssuming that "addictinginfo.org" is one news source with different url so,I am combining all sources of "addictinginfo.org" into one

fake_news['source'] = fake_news['source'].str.replace(r'www.addic|author.addic', 'addic', regex=True)
real_news['source'] = real_news['source'].str.replace(r'www.addic|author.addic', 'addic', regex=True)

common_source = set(real_news['source']).intersection(set(fake_news['source']))

filtered_news_df = news_df[news_df['source'].isin(common_source)]

def clean_text(text):
    return re.sub(r"…|⋆|–|‹|”|“|‘|’", " ", text)


def preprocess_corpus(corpus):
    stemmer = SnowballStemmer('english')
    stop_words = set(stopwords.words('english'))
    additional_stopwords = {"eagle", "rising", "freedom", "daily"}

    processed_corpus = []

    for document in corpus:
        # Convert to lower case
        document = document.lower()
        # Remove numbers
        document = re.sub(r'\d+', '', document)
        # Remove punctuations
        document = re.sub(r'[^\w\s]', '', document)
        # Remove special characters
        document = clean_text(document)
        # Tokenize document
        tokens = word_tokenize(document)
        # Remove stopwords
        tokens = [word for word in tokens if word not in stop_words and word not in additional_stopwords]
        # Stem the words
        tokens = [stemmer.stem(word) for word in tokens]
        # Join tokens back to string
        processed_corpus.append(' '.join(tokens))

    return processed_corpus

''' clean_text Function:

Uses re.sub() to replace special characters with spaces.
preprocess_text Function:

Converts text to lowercase.
Removes numbers and punctuation.
Applies clean_text() to remove additional special characters.
Tokenizes text and removes stopwords and specific words.
Stems words using SnowballStemmer.
Joins tokens back into a single string.
Using the Functions'''

'''Now, we have a clean text corpus, we are interested in those words which are associated with one news category. For this analysis, we perform chi square test.'''


def find_category_representative_words_using_chi_sq(dtm_matrix, categories, top_n=20):
    # Convert dtm_matrix to DataFrame
    dtm_df = pd.DataFrame(dtm_matrix)
    
    # Compute chi-squared values for each feature
    chi2vals = {}
    for col in dtm_df.columns:
        contingency_table = pd.crosstab(np.array(dtm_df[col]), np.array(categories))
        chi2_stat, _, _, _ = chi2_contingency(contingency_table)
        chi2vals[col] = chi2_stat
    
    # Select top N features based on chi-squared values
    top_features = sorted(chi2vals, key=chi2vals.get, reverse=True)[:top_n]
    
    # Add the categories as a column to the DataFrame
    dtm_df['NewsType'] = categories
    
    # Group by 'NewsType' and sum term frequencies for top terms
    cat_freq_df = dtm_df.groupby('NewsType').sum().reset_index()
    top_words_freq = cat_freq_df[top_features + ['NewsType']]
    
    return top_words_freq

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

def create_dtm_matrix(titles):
    vectorizer = CountVectorizer()
    dtm_matrix = vectorizer.fit_transform(titles)
    return dtm_matrix.toarray(), vectorizer.get_feature_names_out()

def find_top_words(dtm_matrix, categories):
    return find_category_representative_words_using_chi_sq(dtm_matrix, categories, 20)



    
    
title_corpus = preprocess_title_corpus(news_df['title'])
title_dtm_matrix, feature_names = create_dtm_matrix(title_corpus)


# Now for Analysis of the body

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

# Convert body text to a document-term matrix
def create_body_dtm_matrix(texts):
    vectorizer = CountVectorizer()
    dtm_matrix = vectorizer.fit_transform(texts)
    return dtm_matrix.toarray(), vectorizer.get_feature_names_out()


# Function to find top words in the body text using Chi-Square
def find_body_top_words(dtm_matrix, categories, top_n=30):
    return find_category_representative_words_using_chi_sq(dtm_matrix, categories, top_n)

body_corpus = preprocess_body_corpus(news_df['text'])
body_dtm_matrix, feature_names = create_body_dtm_matrix(body_corpus)

title_length = title_dtm_matrix.sum(axis=1)

# Create a DataFrame with title length and categories
tl_df = pd.DataFrame({
    'title_length': title_length,
    'type': news_df['type']
})

# t-test (equal_var=False)
real_title_lengths = tl_df[tl_df['type'] == 'Real']['title_length']
fake_title_lengths = tl_df[tl_df['type'] == 'Fake']['title_length']

t_stat, p_value = stats.ttest_ind(real_title_lengths, fake_title_lengths, equal_var=False)


def bigram_tokenizer(text):
    tokens = word_tokenize(text.lower())  
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    bigrams = list(ngrams(filtered_tokens, 2))
    return [' '.join(bigram) for bigram in bigrams]

# Create the bigram matrix
vectorizer = CountVectorizer(tokenizer=bigram_tokenizer)
X = vectorizer.fit_transform(news_df['text'])

# Convert the matrix to a DataFrame
bigram_matrix = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Calculate the frequency of bigrams
bigram_freq = bigram_matrix.sum(axis=0).sort_values(ascending=False)

# Output the top bigrams and their frequencies

def find_top_bigram(bigrams, top_n):
    stop_words = set(stopwords.words('english'))
    top_bigram_list = []
    
    for bigram in bigrams:
        unigrams = bigram.split(" ")
        if not (unigrams[0] in stop_words or unigrams[1] in stop_words):
            top_bigram_list.append(bigram)
        if len(top_bigram_list) == top_n:
            break
    
    return top_bigram_list

# Get the top bigrams
top_bigrams = find_top_bigram(bigram_freq.index, 20)

# Filter bigram matrix to include only the top bigrams
filtered_bigram_matrix = bigram_matrix[top_bigrams].copy()

# Add news type to the DataFrame using .loc
filtered_bigram_matrix.loc[:, 'NewsType'] = news_df['type'].values

# Summarize term frequency by news type
cat_freq_df = filtered_bigram_matrix.groupby('NewsType').sum()


cat_freq_df_reset = cat_freq_df.reset_index()

# Melt the DataFrame
df_melted = cat_freq_df_reset.melt(id_vars='NewsType', var_name='bigram', value_name='bigram_frequency')


# Model Training

np.random.seed(123)

n_obs = len(news_df)
prop_split = 0.75


training_indices = np.random.choice(n_obs, size=int(n_obs * prop_split), replace=False)

# Create training and test sets
training_set = news_df.iloc[training_indices]
test_set = news_df.drop(training_set.index)

#Detecting fake news from title

vectorizer = CountVectorizer()

title_dtm = vectorizer.fit_transform(news_df['title'])

title_dtm_df = pd.DataFrame(title_dtm.toarray(), columns=vectorizer.get_feature_names_out())

subset_df = title_dtm_df.iloc[100:105, 100:105]



vectorizer = CountVectorizer(max_df=0.997)

# Fit and transform the titles to create the DTM
sparse_title_dtm = vectorizer.fit_transform(news_df['title'])

# Convert the DTM to a DataFrame
sparse_title_dtm_df = pd.DataFrame(sparse_title_dtm.toarray(), columns=vectorizer.get_feature_names_out())

# Calculate the number of documents and terms
num_docs, num_terms = sparse_title_dtm.shape

# Calculate non-zero entries
num_non_zero_entries = sparse_title_dtm.nnz

# Calculate sparsity
sparsity = 1 - (num_non_zero_entries / (num_docs * num_terms))

y_true = news_df['type']
x_train, x_test, y_train, y_test = train_test_split(
    title_dtm_df, y_true, test_size=0.25, random_state=123, stratify=y_true)



#detecting from body

def create_body_dtm(body_corpus):
    vectorizer = CountVectorizer(max_df=0.97)
    sparse_body_dtm = vectorizer.fit_transform(body_corpus)
    sparse_body_dtm_df = pd.DataFrame(sparse_body_dtm.toarray(), columns=vectorizer.get_feature_names_out())
    # print(f"Shape of the new sparse DTM: {sparse_body_dtm_df.shape}")
    return sparse_body_dtm.toarray()


# Detecting from the body
body_dtm = create_body_dtm(body_corpus)

y_true = news_df['type'].values  # Convert the type column to a numpy array
x_train_body = body_dtm[training_indices, :]  # Rows for training
x_test_body = body_dtm[np.setdiff1d(np.arange(body_dtm.shape[0]), training_indices), :]  # Rows for testing

y_train_body = y_true[training_indices]
y_test_body = y_true[np.setdiff1d(np.arange(body_dtm.shape[0]), training_indices)]

def combine_title_body_dtm(body_dtm, title_dtm):
    title_body_dtm = body_dtm.copy()
    
    # Get common features (column indices)
    common_features = list(set(range(body_dtm.shape[1])) & set(range(title_dtm.shape[1])))
    
    # Sum the common features
    if isinstance(title_body_dtm, csr_matrix):  # If title_body_dtm is a sparse matrix
        title_body_dtm[:, common_features] += title_dtm[:, common_features].toarray()
    else:
        title_body_dtm[:, common_features] += title_dtm[:, common_features]

    # Get title only features
    title_only_features = list(set(range(title_dtm.shape[1])) - set(range(body_dtm.shape[1])))
    
    # Combine title_only_features from title_dtm to title_body_dtm
    if title_only_features:
        # Convert title_dtm to dense
        title_only_data = title_dtm[:, title_only_features].toarray() if hasattr(title_dtm, 'toarray') else title_dtm[:, title_only_features]
        
        # Ensure that title_only_data is 2D
        if title_only_data.ndim == 1:
            title_only_data = title_only_data[:, np.newaxis]  # Reshape to 2D if necessary
        
        # Stack the arrays
        title_body_dtm = hstack([title_body_dtm, title_only_data])

    # Check if title_body_dtm is sparse or dense and convert to DataFrame
    title_body_dtm_df = pd.DataFrame(title_body_dtm.toarray()) if hasattr(title_body_dtm, 'toarray') else pd.DataFrame(title_body_dtm)

    # print(f"Combined title and body DTM shape: {title_body_dtm.shape}")
    return title_body_dtm




title_body_dtm = combine_title_body_dtm(body_dtm, title_dtm)


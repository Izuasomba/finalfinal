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
import matplotlib.pyplot as plt
import seaborn as sns
import re
# For machine learning
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB


import os
file_path = os.path.join(os.path.dirname(__file__), 'data', 'BuzzFeed_real_news_content.csv')
real_news = pd.read_csv(file_path)

import os
file_path2 = os.path.join(os.path.dirname(__file__), 'data', 'BuzzFeed_fake_news_content.csv')
fake_news = pd.read_csv(file_path2)


news_df = pd.concat([real_news, fake_news], ignore_index=True)


news_df['type'] = news_df['id'].str.split('_').str[0]

#merged both frames into one, added a new column to cateorize as real or fake

news_df.head()

news_df.shape
news_df.info()

selected_columns = ["id", "title", "text", "source", "type", "images", "movies"]
news_df = news_df[selected_columns]
news_df.head()

# selected necessary columns to analyse

#I am going to change the "movies" and "images" column to categorical variables, "0" if not available and "1" if available
news_df['movies'] = news_df['movies'].notna().astype(int)
news_df['images'] = news_df['images'].notna().astype(int)

#I am ssuming that "addictinginfo.org" is one news source with different url so,I am combining all sources of "addictinginfo.org" into one

fake_news['source'] = fake_news['source'].str.replace(r'www.addic|author.addic', 'addic', regex=True)
real_news['source'] = real_news['source'].str.replace(r'www.addic|author.addic', 'addic', regex=True)
news_df.head()

source_counts = real_news['source'].value_counts().reset_index()
source_counts.columns = ['source', 'count']

plt.figure(figsize=(10, 8))
sns.barplot(data=source_counts, y='source', x='count', palette='Greens_d')
plt.title('Source Count of Real News in Buzzfeed')
plt.xlabel('Count')
plt.ylabel('Source')
plt.show()

source_counts_fake = fake_news['source'].value_counts().reset_index()
source_counts_fake.columns = ['source', 'count']


plt.figure(figsize=(10, 8))
sns.barplot(data=source_counts_fake, y='source', x='count', palette='Reds_d')
plt.title('Source Count of Fake News in Buzzfeed')
plt.xlabel('Count')
plt.ylabel('Source')
plt.show()

''' plot shows that,the rightwingsnews reports the most fakenews. Also, the number of fake news sources are more than the number of real news sources.

There are some news which are reported and categorised as fake but their source is unknown. We do not remove such news because it shows that as some fake news came from unknown sources, all real news came from well known sources.

Since there are some sources which report real as well as fake news. Now, we focus on whether a particular news source reports more fake news than real news. For this analysis, we find all sources which reports both fake and real news, and then plot news counts in both categories.'''

common_source = set(real_news['source']).intersection(set(fake_news['source']))

filtered_news_df = news_df[news_df['source'].isin(common_source)]

plt.figure(figsize=(12, 8))
sns.countplot(data=filtered_news_df, y='source', hue='type', palette='coolwarm', dodge=True)
plt.title('Common Source of Real and Fake News in Buzzfeed')
plt.xlabel('Count')
plt.ylabel('Source')
plt.legend(title='Type')
plt.show()

news_df['movies'] = news_df['movies'].astype('category')

plt.figure(figsize=(10, 6))
sns.countplot(data=news_df, x='movies', hue='type', palette='Set1', dodge=True)
plt.xlabel('Movies Linked to News')
plt.ylabel('Counts')
plt.title('News based on Movies')
plt.legend(title='Type')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=news_df, x='images', hue='type', palette='Set2', dodge=True)
plt.xlabel('Images in News')
plt.ylabel('Counts')
plt.title('News Category by Images')
plt.legend(title='Type')
plt.show()

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

def plot_top_words(top_words_freq, feature_names, title):
    # Melting the data frame
    top_words_freq_melted = pd.melt(top_words_freq, id_vars='NewsType', 
                                    var_name='FeatureIndex', value_name='Frequency')

    # Map feature indices to feature names
    top_words_freq_melted['Word'] = top_words_freq_melted['FeatureIndex'].apply(lambda x: feature_names[int(x)])

    plt.figure(figsize=(12, 8))
    
    # create plot
    ax = sns.barplot(data=top_words_freq_melted, x='Word', y='Frequency', hue='NewsType', 
                     dodge=True, palette='Set1')
    
    # Add annotations to the bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 5), 
                    textcoords='offset points')

    # Flip the axes
    plt.xticks(rotation=90)
    plt.title(title)
    plt.xlabel('Top 20 Words')
    plt.ylabel('Term Frequency of Words')
    plt.tight_layout()

    
    plt.show()
    
    
title_corpus = preprocess_title_corpus(news_df['title'])
title_dtm_matrix, feature_names = create_dtm_matrix(title_corpus)

# Finding top 20 words in the news title for both fake and real news
title_top_words_freq = find_top_words(title_dtm_matrix, news_df['type'])


plot_top_words(title_top_words_freq, feature_names, "Top Words in the Title of News")


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


def plot_body_top_words(top_words_freq, feature_names, title):

    top_words_freq_melted = pd.melt(top_words_freq, id_vars='NewsType', 
                                    var_name='FeatureIndex', value_name='Frequency')

    # Map feature indices to original names
    top_words_freq_melted['Word'] = top_words_freq_melted['FeatureIndex'].apply(lambda x: feature_names[int(x)])

    plt.figure(figsize=(14, 10))
    
    # Create the bar plot
    ax = sns.barplot(data=top_words_freq_melted, x='Word', y='Frequency', hue='NewsType', 
                     dodge=True, palette='Set1')
    
    # Add annotations to the bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 5), 
                    textcoords='offset points')

    # Flip the axes
    plt.xticks(rotation=90)
    plt.title(title)
    plt.xlabel('Top 30 Words')
    plt.ylabel('Term Frequency of Words')
    plt.tight_layout()

    # Display the plot
    plt.show()


body_corpus = preprocess_body_corpus(news_df['text'])
body_dtm_matrix, feature_names = create_body_dtm_matrix(body_corpus)

# Finding top 30 words in the news body for both categories
body_top_words_freq = find_body_top_words(body_dtm_matrix, news_df['type'], top_n=30)

# Plotting the result
plot_body_top_words(body_top_words_freq, feature_names, "Top Words in the Body of News Article")


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


print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# mean of each group
mean_real = real_title_lengths.mean()
mean_fake = fake_title_lengths.mean()
print(f"Mean of Real: {mean_real}")
print(f"Mean of Fake: {mean_fake}")


plt.figure(figsize=(10, 6))


handles = []
labels = []


for category in tl_df['type'].unique():
    subset = tl_df[tl_df['type'] == category]
    kde_plot = sns.kdeplot(data=subset, x='title_length', fill=True, alpha=0.5, label=category)
    
    handles, labels = kde_plot.get_legend_handles_labels()

# Add labels and title
plt.xlabel('Title Length')
plt.ylabel('Density')
plt.title('Density Distribution of Title Length for Real and Fake News')


plt.legend(handles, labels, title='News Type')


plt.tight_layout()
plt.show()

'''We observed a statistically significant difference (p-value = 0.00225) between the lengths of news titles for real and fake news. On average, the titles of fake news articles are slightly longer than those of real news articles. 
Specifically, the mean title length for fake news is 8.33, while for real news it is 7.24. This indicates that the distribution of title lengths for fake news is centered slightly higher compared to real news. 
The t-test results provide evidence that the title length of real news is significantly shorter than that of fake news.'''

'''analysis has focused on unigrams from the articles. processed the text by removing common words using stopwords and applied stemming.

Now, we aim to analyze the phrases used in the text body of the news articles. To achieve this, I will use a function to tokenize bigrams.
Instead of using a basic text processing approach, we will use a method that preserves the original order and sequence of words by avoiding stemming and not removing common English words.
This approach allows that we retain important phrases, which could be significant for analysis.'''

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
print(bigram_freq.head(20))

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


print(cat_freq_df)

cat_freq_df_reset = cat_freq_df.reset_index()

# Melt the DataFrame
df_melted = cat_freq_df_reset.melt(id_vars='NewsType', var_name='bigram', value_name='bigram_frequency')


plt.figure(figsize=(12, 8))
sns.barplot(data=df_melted, x='bigram', y='bigram_frequency', hue='NewsType', dodge=True)


plt.xticks(rotation=90)  # Rotate x labels for better readability
plt.xlabel('Bigrams')
plt.ylabel('Bigram Frequency')
plt.title('High Frequency Bigrams in the Body of News Articles')
plt.legend(title='News Type')


plt.tight_layout()
plt.show()

# Model Training

np.random.seed(123)

n_obs = len(news_df)
prop_split = 0.75


training_indices = np.random.choice(n_obs, size=int(n_obs * prop_split), replace=False)

# Create training and test sets
training_set = news_df.iloc[training_indices]
test_set = news_df.drop(training_set.index)


print(f"Training size: {len(training_set)}")
print(f"Test size: {len(test_set)}")

#Detecting fake news from title

vectorizer = CountVectorizer()

title_dtm = vectorizer.fit_transform(news_df['title'])

title_dtm_df = pd.DataFrame(title_dtm.toarray(), columns=vectorizer.get_feature_names_out())

subset_df = title_dtm_df.iloc[100:105, 100:105]

print(subset_df)


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

print(f"Document-Term Matrix (documents: {num_docs}, terms: {num_terms})")
print(f"Non-/sparse entries: {num_non_zero_entries}/{num_docs * num_terms}")
print(f"Sparsity           : {sparsity:.2%}")
print(f"Maximal term length: {max(len(term) for term in vectorizer.get_feature_names_out())}")
print(f"Weighting          : term frequency (tf)")

y_true = news_df['type']
x_train, x_test, y_train, y_test = train_test_split(
    title_dtm_df, y_true, test_size=0.25, random_state=123, stratify=y_true)


print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test labels shape: {y_test.shape}")

def evaluate_logistic_regression_title(x_train, y_train, x_test, y_test):
    log_fit_title = LogisticRegression(penalty='l2', solver='liblinear', random_state=123)
    log_fit_title.fit(x_train, y_train)

    predicted_log_title = log_fit_title.predict(x_test)
    accuracy_log_title = accuracy_score(y_test, predicted_log_title)
    print(f"Accuracy of Logistic Regression on the title dataset: {accuracy_log_title:.4f}")

def evaluate_random_forest_title(x_train, y_train, x_test, y_test):
    rf_title = RandomForestClassifier(n_estimators=50, random_state=123, oob_score=True)
    rf_title.fit(x_train, y_train)  

    y_pred = rf_title.predict(x_test)
    accuracy_rf_title = accuracy_score(y_test, y_pred)

    print(f"Accuracy of Random Forest on the title dataset: {accuracy_rf_title:.4f}")
    
    # Additional outputs
    print(rf_title)
    print("Feature importances:")
    print(rf_title.feature_importances_)
    print(f"Number of trees in the forest: {len(rf_title.estimators_)}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=['Fake', 'Real'])
    print("Confusion Matrix:")
    print(conf_matrix)

    # Classification Report
    class_report = classification_report(y_test, y_pred, target_names=['Fake', 'Real'])
    print("Classification Report:")
    print(class_report)

    # Out-of-Bag Error
    if hasattr(rf_title, 'oob_score_'):
        print(f"Out-of-Bag Error Estimate: {1 - rf_title.oob_score_:.2%}")
    else:
        print("OOB score is not available. Make sure oob_score=True was set during initialization.")


def create_body_dtm(body_corpus):
    vectorizer = CountVectorizer(max_df=0.97)
    sparse_body_dtm = vectorizer.fit_transform(body_corpus)
    sparse_body_dtm_df = pd.DataFrame(sparse_body_dtm.toarray(), columns=vectorizer.get_feature_names_out())
    print(f"Shape of the new sparse DTM: {sparse_body_dtm_df.shape}")
    return sparse_body_dtm.toarray()

def evaluate_naive_bayes_body(x_train_body, y_train_body, x_test_body, y_test_body):
    nb_body = GaussianNB()
    nb_body.fit(x_train_body, y_train_body)  # Train the model
    predicted_naive_body = nb_body.predict(x_test_body)  # Make predictions
    accuracy_naive_body = accuracy_score(y_test_body, predicted_naive_body)
    print(f"Accuracy of Naive Bayes model on the body: {accuracy_naive_body:.4f}")

def evaluate_logistic_regression_body(x_train_body, y_train_body, x_test_body, y_test_body):
    log_reg_body = LogisticRegression(solver='liblinear')
    log_reg_body.fit(x_train_body, y_train_body)  # Train the model
    predicted_glm_body = log_reg_body.predict(x_test_body)  # Make predictions
    accuracy_log_body = accuracy_score(y_test_body, predicted_glm_body)
    print(f"Accuracy of Logistic Regression model on the body: {accuracy_log_body:.4f}")

def evaluate_random_forest_body(x_train_body, y_train_body, x_test_body, y_test_body):
    rf_body = RandomForestClassifier(n_estimators=500, random_state=123)
    rf_body.fit(x_train_body, y_train_body)  # Train the model
    predicted_rf_body = rf_body.predict(x_test_body)  # Make predictions
    accuracy_rf_body = accuracy_score(y_test_body, predicted_rf_body)
    print(f"Accuracy of Random Forest on text body: {accuracy_rf_body:.4f}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test_body, predicted_rf_body)
    print("Confusion Matrix:")
    print(conf_matrix)

# Detecting from the body
body_dtm = create_body_dtm(body_corpus)

y_true = news_df['type'].values  # Convert the type column to a numpy array
x_train_body = body_dtm[training_indices, :]  # Rows for training
x_test_body = body_dtm[np.setdiff1d(np.arange(body_dtm.shape[0]), training_indices), :]  # Rows for testing

y_train_body = y_true[training_indices]
y_test_body = y_true[np.setdiff1d(np.arange(body_dtm.shape[0]), training_indices)]

print(f"x_train_body shape: {x_train_body.shape}")
print(f"x_test_body shape: {x_test_body.shape}")
print(f"y_true shape: {y_true.shape}")


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

    print(f"Combined title and body DTM shape: {title_body_dtm.shape}")
    return title_body_dtm

def evaluate_naive_bayes_bt(title_body_dtm, training_indices, y_true):
    nb_body_tb = MultinomialNB()
    nb_body_tb.fit(title_body_dtm[training_indices], y_true[training_indices])

    predicted_nb_tb = nb_body_tb.predict(title_body_dtm[np.setdiff1d(np.arange(title_body_dtm.shape[0]), training_indices)])
    accuracy_nb_tb = accuracy_score(y_true[np.setdiff1d(np.arange(y_true.shape[0]), training_indices)], predicted_nb_tb)

    print(f'Accuracy of Naive Bayes on title or body terms: {accuracy_nb_tb:.4f}')

def evaluate_logistic_regression_bt(title_body_dtm, training_indices, y_true):
    scaler = StandardScaler(with_mean=False)  # Set with_mean=False for sparse matrices
    title_body_dtm_scaled = scaler.fit_transform(title_body_dtm)

    log_reg_fit_title_body = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000)
    log_reg_fit_title_body.fit(title_body_dtm_scaled[training_indices], y_true[training_indices])

    predicted_log_tb = log_reg_fit_title_body.predict(title_body_dtm_scaled[np.setdiff1d(np.arange(title_body_dtm.shape[0]), training_indices)])
    accuracy_log_tb = accuracy_score(y_true[np.setdiff1d(np.arange(y_true.shape[0]), training_indices)], predicted_log_tb)

    print(f'Accuracy of Logistic Regression on terms in either the title or body: {accuracy_log_tb:.4f}')

def evaluate_random_forest_bt(title_body_dtm, training_indices, y_true):
    rf_tb = RandomForestClassifier(n_estimators=500, random_state=123)
    rf_tb.fit(title_body_dtm[training_indices], y_true[training_indices])

    predicted_rf_tb = rf_tb.predict(title_body_dtm[np.setdiff1d(np.arange(title_body_dtm.shape[0]), training_indices)])
    accuracy_rf_tb = accuracy_score(y_true[np.setdiff1d(np.arange(y_true.shape[0]), training_indices)], predicted_rf_tb)

    print(f'Accuracy of Random Forest on title or body terms: {accuracy_rf_tb:.4f}')


title_body_dtm = combine_title_body_dtm(body_dtm, title_dtm)



evaluate_naive_bayes_bt(title_body_dtm, training_indices, y_true)
evaluate_logistic_regression_bt(title_body_dtm, training_indices, y_true)
evaluate_random_forest_bt(title_body_dtm, training_indices, y_true)

evaluate_naive_bayes_body(x_train_body, y_train_body, x_test_body, y_test_body)
evaluate_logistic_regression_body(x_train_body, y_train_body, x_test_body, y_test_body)
evaluate_random_forest_body(x_train_body, y_train_body, x_test_body, y_test_body)

evaluate_logistic_regression_title(x_train, y_train, x_test, y_test)
evaluate_random_forest_title(x_train, y_train, x_test, y_test)
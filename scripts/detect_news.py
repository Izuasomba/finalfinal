import joblib
import numpy as np
# from sklearn.ensemble import RandomForestClassifier
import sys
import os
from sklearn.feature_extraction.text import CountVectorizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from preprocess import preprocess_title_corpus, preprocess_body_corpus

# Load the saved model
model_filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../model.joblib'))
log_reg_model = joblib.load(model_filename)

# Universal vectorizer function from your preprocess script
def vectorize_input(corpus, max_df=1.0, tokenizer=None):
    """
    Vectorizes input text corpus with optional max_df and tokenizer parameters.
    Adjusts min_df based on the number of documents if necessary.
    """
    if not corpus:
        return np.array([]), None  # Return empty if corpus is empty

    # Determine appropriate min_df based on corpus size
    min_df = max(1, int(0.1 * len(corpus)))  # Minimum of 1 or 10% of the corpus size

    vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, tokenizer=tokenizer)
    dtm = vectorizer.fit_transform(corpus)
    
    return dtm.toarray(), vectorizer


# Main detection function
def detect_news(title, body):
    # Preprocess title and body
    processed_title = preprocess_title_corpus([title])
    processed_body = preprocess_body_corpus([body])

    # Vectorize title and body (remove min_df argument)
    title_vectorized, _ = vectorize_input(processed_title, max_df=0.997)
    body_vectorized, _ = vectorize_input(processed_body, max_df=0.97)

    # Combine title and body vectors
    title_body_vectorized = np.hstack((title_vectorized, body_vectorized))

    # Make prediction
    prediction = log_reg_model.predict(title_body_vectorized)

    return prediction[0]



# User input loop
if __name__ == "__main__":
    print("Welcome to the Fake News Detection System!")
    while True:
        title = input("Enter the title of the news article (or 'exit' to quit): ")
        if title.lower() == 'exit':
            break
        body = input("Enter the body of the news article: ")
        
        prediction = detect_news(title, body)
        if prediction == 'fake':
            print("The news article is classified as FAKE.")
        else:
            print("The news article is classified as REAL.")

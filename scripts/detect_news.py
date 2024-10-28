import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from preprocess import vectorize_input

# Load the saved model
model_filename = 'model.joblib'
rf_body_model = joblib.load(model_filename)

# Function to preprocess the input and make a prediction
def predict_news_article(title, body):
    # Combine title and body
    article = [title + ' ' + body]  # Combine title and body into one string
    
    # Vectorize the article using the vectorizer function from preprocess.py
    article_vectorized = vectorize_input(article)
    
    # Make prediction
    prediction = rf_body_model.predict(article_vectorized)
    
    return prediction[0]  # Return the predicted class

# Main interaction loop
if __name__ == "__main__":
    print("Welcome to the Fake News Detection System!")
    while True:
        title = input("Enter the title of the news article (or 'exit' to quit): ")
        if title.lower() == 'exit':
            break
        body = input("Enter the body of the news article: ")
        
        # Predict and display the result
        prediction = predict_news_article(title, body)
        if prediction == 'Real':
            print("The article is classified as: Real News")
        else:
            print("The article is classified as: Fake News")

    print("Thank you for using the Fake News Detection System!")

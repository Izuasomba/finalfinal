# Fake News Detection Project

## Overview
This project implements a fake news detection system using various machine learning algorithms. The goal is to classify news articles as either fake or real based on their content. The project includes data preprocessing, model training, and evaluation, with a variety of scripts to handle different tasks.

## Directory Structure

├── data # Contains the datasets for training and testing 
│   ├── BuzzFeed_fake_news_content.csv
│   └── BuzzFeed_real_news_content.csv
├── final year report.pdf # Report of the project 
├── model.joblib  # Saved trained model
├── notebook  # Jupyter notebook for exploration and analysis 
│   └── Fake_news_classifier.ipynb
├── __pycache__
│   └── preprocess.pyc
├── readme.md
├── requirements.txt # Python dependencies
├── run.py  # Entry point to run specific scripts 
├── scripts # Directory containing individual scripts for model evaluation and training
│   ├── detect_news.py
│   ├── logreg_bt.py # Logistic Regression on body and title
│   ├── lr_body.py # Logistic Regression on body
│   ├── lr_title.py # Logistic Regression on title
│   ├── main.py # Main execution script 
│   ├── naiveb_bt.py  # Naive Bayes on body and title
│   ├── nb_body.py  # Naive Bayes on body
│   ├── nb_title.py  # Naive Bayes on title
│   ├── rf_body.py # Random Forest on body text
│   ├── rf_bt.py # Random Forest on body and title text
│   └── rf_title.py # Random Forest on title text
├── setup_nltk.py #  Script to set up NLTK
└── src # Source code for evaluation and preprocessing 
    ├── eval_func.py
    ├── preprocess.py
    └── __pycache__
        ├── eval_func.cpython-310.pyc
        ├── Fake_news_classifier.cpython-310.pyc
        └── preprocess.cpython-310.pyc


## Requirements
To set up the project, you need to install the necessary Python dependencies. You can do this using pip:

```bash
pip install -r requirements.txt
```

Running the Scripts
To run a specific script, you can use the run.py script located in the project root directory. This script allows you to execute any of the individual scripts in the scripts folder.

Example Usage
To run the logistic regression script that evaluates both the body and title of the news articles, you would execute the following command:

``` bash
 python run.py logreg_bt.py
```
Making a new prediction
The already trained model has been saved locally (model.joblib) to make an inference on a new article, execute the following command:

``` bash
 python run.py detect_news.py
```
follow the prompt i.e provide the title and body of the article.

Available Scripts
detect_news.py: Script for classifying news articles.
logreg_bt.py: Logistic Regression model evaluating both body and title.
lr_body.py: Logistic Regression model evaluating body text only.
lr_title.py: Logistic Regression model evaluating title text only.
naiveb_bt.py: Naive Bayes model evaluating both body and title.
nb_body.py: Naive Bayes model evaluating body text only.
nb_title.py: Naive Bayes model evaluating title text only.
rf_body.py: Random Forest model evaluating body text only.
rf_bt.py: Random Forest model evaluating both body and title.
rf_title.py: Random Forest model evaluating title text only.

Data
The datasets used in this project are located in the data directory. The project uses two CSV files:

BuzzFeed_fake_news_content.csv: Contains fake news articles.
BuzzFeed_real_news_content.csv: Contains real news articles.
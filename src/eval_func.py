import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler




from preprocess import x_train, y_train, x_test, y_test

def evaluate_naive_bayes_title(x_train, y_train, x_test, y_test):
    # Start timing
    start_time = time.time()
    
    # Initialize and train Naive Bayes classifier
    nb_title = MultinomialNB()
    nb_title.fit(x_train, y_train)

    # Make predictions
    predicted_nb_title = nb_title.predict(x_test)
    
    # Calculate accuracy
    accuracy_nb_title = accuracy_score(y_test, predicted_nb_title)
    print(f"Accuracy of Naive Bayes on the title dataset: {accuracy_nb_title:.4f}")
    
    # End timing
    end_time = time.time()
    print(f"Naive Bayes (Title) Evaluation took {end_time - start_time:.4f} seconds.\n")



def evaluate_logistic_regression_title(x_train, y_train, x_test, y_test):
    start_time = time.time()
    
    # Initialize and train Logistic Regression
    log_fit_title = LogisticRegression(penalty='l2', solver='liblinear', random_state=123)
    log_fit_title.fit(x_train, y_train)

    # Make predictions
    predicted_log_title = log_fit_title.predict(x_test)
    
    # Calculate accuracy
    accuracy_log_title = accuracy_score(y_test, predicted_log_title)
    print(f"Accuracy of Logistic Regression on the title dataset: {accuracy_log_title:.4f}")
    
    end_time = time.time()
    print(f"Logistic Regression (Title) Evaluation took {end_time - start_time:.4f} seconds.\n")


def evaluate_random_forest_title(x_train, y_train, x_test, y_test):
    start_time = time.time()
    
    # Initialize and train Random Forest
    rf_title = RandomForestClassifier(n_estimators=50, random_state=123, oob_score=True)
    rf_title.fit(x_train, y_train)  

    # Make predictions
    y_pred = rf_title.predict(x_test)
    
    # Calculate accuracy
    accuracy_rf_title = accuracy_score(y_test, y_pred)
    print(f"Accuracy of Random Forest on the title dataset: {accuracy_rf_title:.4f}")
    
    end_time = time.time()
    print(f"Random Forest (Title) Evaluation took {end_time - start_time:.4f} seconds.\n")
    



def evaluate_naive_bayes_body(x_train_body, y_train_body, x_test_body, y_test_body):
    start_time = time.time()
    
    nb_body = GaussianNB()
    nb_body.fit(x_train_body, y_train_body)  # Train the model
    predicted_naive_body = nb_body.predict(x_test_body)  # Make predictions
    accuracy_naive_body = accuracy_score(y_test_body, predicted_naive_body)
    print(f"Accuracy of Naive Bayes model on the body: {accuracy_naive_body:.4f}")
    
    end_time = time.time()
    print(f"Naive Bayes (Body) Evaluation took {end_time - start_time:.4f} seconds.\n")

def evaluate_logistic_regression_body(x_train_body, y_train_body, x_test_body, y_test_body):
    start_time = time.time()
    
    log_reg_body = LogisticRegression(solver='liblinear')
    log_reg_body.fit(x_train_body, y_train_body)  # Train the model
    predicted_glm_body = log_reg_body.predict(x_test_body)  # Make predictions
    accuracy_log_body = accuracy_score(y_test_body, predicted_glm_body)
    print(f"Accuracy of Logistic Regression model on the body: {accuracy_log_body:.4f}")
    
    end_time = time.time()
    print(f"Logistic Regression (Body) Evaluation took {end_time - start_time:.4f} seconds.\n")

def evaluate_random_forest_body(x_train_body, y_train_body, x_test_body, y_test_body):
    start_time = time.time()
    
    rf_body = RandomForestClassifier(n_estimators=500, random_state=123)
    rf_body.fit(x_train_body, y_train_body)  # Train the model
    predicted_rf_body = rf_body.predict(x_test_body)  # Make predictions
    accuracy_rf_body = accuracy_score(y_test_body, predicted_rf_body)
    print(f"Accuracy of Random Forest on text body: {accuracy_rf_body:.4f}")
    
    # Additional Outputs
    print("\nModel Summary:")
    print(rf_body)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test_body, predicted_rf_body)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Classification Report
    class_report = classification_report(y_test_body, predicted_rf_body, target_names=['Fake', 'Real'])
    print("\nClassification Report:")
    print(class_report)

    end_time = time.time()
    print(f"Random Forest (Body) Evaluation took {end_time - start_time:.4f} seconds.\n")



def evaluate_naive_bayes_bt(title_body_dtm, training_indices, y_true):
    start_time = time.time()
    
    nb_body_tb = MultinomialNB()
    nb_body_tb.fit(title_body_dtm[training_indices], y_true[training_indices])

    # Predict using the rest of the data
    predicted_nb_tb = nb_body_tb.predict(title_body_dtm[np.setdiff1d(np.arange(title_body_dtm.shape[0]), training_indices)])
    accuracy_nb_tb = accuracy_score(y_true[np.setdiff1d(np.arange(y_true.shape[0]), training_indices)], predicted_nb_tb)

    print(f'Accuracy of Naive Bayes on title or body terms: {accuracy_nb_tb:.4f}')
    
    end_time = time.time()
    print(f"Naive Bayes (Title or Body) Evaluation took {end_time - start_time:.4f} seconds.\n")


def evaluate_logistic_regression_bt(title_body_dtm, training_indices, y_true):
    start_time = time.time()
    
    # Scale the features for Logistic Regression
    scaler = StandardScaler(with_mean=False)  # Set with_mean=False for sparse matrices
    title_body_dtm_scaled = scaler.fit_transform(title_body_dtm)

    log_reg_fit_title_body = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000)
    log_reg_fit_title_body.fit(title_body_dtm_scaled[training_indices], y_true[training_indices])

    # Predict using the rest of the data
    predicted_log_tb = log_reg_fit_title_body.predict(title_body_dtm_scaled[np.setdiff1d(np.arange(title_body_dtm.shape[0]), training_indices)])
    accuracy_log_tb = accuracy_score(y_true[np.setdiff1d(np.arange(y_true.shape[0]), training_indices)], predicted_log_tb)

    print(f'Accuracy of Logistic Regression on terms in either the title or body: {accuracy_log_tb:.4f}')
    
    end_time = time.time()
    print(f"Logistic Regression (Title or Body) Evaluation took {end_time - start_time:.4f} seconds.\n")


def evaluate_random_forest_bt(title_body_dtm, training_indices, y_true):
    start_time = time.time()
    
    rf_tb = RandomForestClassifier(n_estimators=500, random_state=123)
    rf_tb.fit(title_body_dtm[training_indices], y_true[training_indices])

    # Predict using the rest of the data
    predicted_rf_tb = rf_tb.predict(title_body_dtm[np.setdiff1d(np.arange(title_body_dtm.shape[0]), training_indices)])
    accuracy_rf_tb = accuracy_score(y_true[np.setdiff1d(np.arange(y_true.shape[0]), training_indices)], predicted_rf_tb)

    print(f'Accuracy of Random Forest on title or body terms: {accuracy_rf_tb:.4f}')
    
    # Additional Outputs
    print("\nModel Summary:")
    print(rf_tb)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true[np.setdiff1d(np.arange(y_true.shape[0]), training_indices)], predicted_rf_tb)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Classification Report
    class_report = classification_report(y_true[np.setdiff1d(np.arange(y_true.shape[0]), training_indices)], predicted_rf_tb, target_names=['Fake', 'Real'])
    print("\nClassification Report:")
    print(class_report)

    end_time = time.time()
    print(f"Random Forest (Title or Body) Evaluation took {end_time - start_time:.4f} seconds.\n")

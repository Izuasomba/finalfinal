import sys
import os

# Add the project root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))


from preprocess import x_train, y_train, x_test, y_test
from eval_func import evaluate_logistic_regression_title



evaluate_logistic_regression_title(x_train, y_train, x_test, y_test)

import sys
import os

# Absolute path to src, in order to access preprcess and eval_func
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from preprocess import x_train, y_train, x_test, y_test
from eval_func import evaluate_naive_bayes_title

evaluate_naive_bayes_title(x_train, y_train, x_test, y_test)




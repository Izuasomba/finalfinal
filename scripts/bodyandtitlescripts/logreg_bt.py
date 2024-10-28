import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from preprocess import title_body_dtm, training_indices, y_true
from eval_func import evaluate_logistic_regression_bt



evaluate_logistic_regression_bt(title_body_dtm, training_indices, y_true)
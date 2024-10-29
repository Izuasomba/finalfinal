import sys
import os

# Absolute path to src, in order to access preprcess and eval_func
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from preprocess import title_body_dtm, training_indices, y_true
from eval_func import evaluate_random_forest_bt


evaluate_random_forest_bt(title_body_dtm, training_indices, y_true)

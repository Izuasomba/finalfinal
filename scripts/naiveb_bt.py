import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from preprocess import title_body_dtm, y_true,training_indices
from eval_func import evaluate_naive_bayes_bt

evaluate_naive_bayes_bt(title_body_dtm, training_indices, y_true)





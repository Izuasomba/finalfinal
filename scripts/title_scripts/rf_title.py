import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from preprocess import x_train, y_train, x_test, y_test
from eval_func import evaluate_random_forest_title



evaluate_random_forest_title(x_train, y_train, x_test, y_test)
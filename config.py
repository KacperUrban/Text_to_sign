import os
import sys

#Define custom paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model", "model")
TOKENIZER_DIR = os.path.join(BASE_DIR, "model", "tokenizer","")
CUSTOM_UTILS_DIR = os.path.join(BASE_DIR, "custom_utils")

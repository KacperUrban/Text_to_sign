import os

#Define custom paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models", "model")
FINAL_MODEL_DIR = os.path.join(BASE_DIR, "models", "final_model")
TOKENIZER_DIR = os.path.join(BASE_DIR, "models", "tokenizer","")
TOKENIZER_DE_DIR = os.path.join(BASE_DIR, "models", "tokenizer_de", "")
CUSTOM_UTILS_DIR = os.path.join(BASE_DIR, "custom_utils")

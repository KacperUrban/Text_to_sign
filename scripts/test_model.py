import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from config import BASE_DIR, MODEL_DIR, TOKENIZER_DIR, DATA_DIR
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from dotenv import load_dotenv
import pandas as pd
from custom_utils import cross_validation_pt

if __name__ == '__main__':
    load_dotenv(dotenv_path=BASE_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    data = pd.read_csv(DATA_DIR + "/final_data/all_data.csv")

    print(f"Average BLEU score: {cross_validation_pt(model, tokenizer, data, device, num_epochs=2, n_splits=3, batch_size=16)}")
